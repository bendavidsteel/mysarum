"""
Graph Cellular Automata (GRA) — JAX simulation core.

Lenia-style Gaussian growth on graphs with Chebyshev spectral convolution,
cross-channel coupling, and dynamic topology (node division / collapse).
All operations use fixed-size buffers for JIT compatibility and vmap batching.
"""

import json
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── Configurable constants ───────────────────────────────────────────────────
# These are module-level so JIT can see them as static. Call configure() to
# change before any JIT compilation occurs.

MAX_NODES = 512
MAX_DEGREE = 8
MAX_CHEB_ORDER = 8
NUM_CHANNELS = 3
MAX_TOPO_OPS = 8
TOPO_EVERY = 10
EMPTY = -1


def configure(cfg):
    """Set module constants from a config object (OmegaConf or dict).

    Must be called before any JIT compilation.
    """
    global MAX_NODES, MAX_DEGREE, MAX_CHEB_ORDER, NUM_CHANNELS
    global MAX_TOPO_OPS, TOPO_EVERY

    sim = cfg if not hasattr(cfg, "sim") else cfg.sim
    MAX_NODES = int(sim.max_nodes)
    MAX_DEGREE = int(sim.max_degree)
    MAX_CHEB_ORDER = int(sim.max_cheb_order)
    NUM_CHANNELS = int(sim.num_channels)
    MAX_TOPO_OPS = int(sim.max_topo_ops)
    TOPO_EVERY = int(sim.topo_every)


# ── Data structures ──────────────────────────────────────────────────────────

class SimState(NamedTuple):
    """Full mutable simulation state (all arrays fixed-size)."""
    node_states: jnp.ndarray   # (MAX_NODES, 3)  in [0,1]
    node_active: jnp.ndarray   # (MAX_NODES,)     bool
    neighbors: jnp.ndarray     # (MAX_NODES, MAX_DEGREE)  int32, EMPTY=-1
    num_neighbors: jnp.ndarray # (MAX_NODES,)     int32, actual degree
    node_u: jnp.ndarray        # (MAX_NODES, 3)  coupling output (cached)
    num_active: jnp.ndarray    # ()  scalar int32
    rng: jnp.ndarray           # PRNGKey


class Params(NamedTuple):
    """Simulation parameters (static within a run)."""
    kernel_mu: jnp.ndarray       # (3,)  Chebyshev kernel center per channel
    kernel_sigma: jnp.ndarray    # (3,)  Chebyshev kernel width per channel
    growth_mu: jnp.ndarray       # (3,)  growth function center per channel
    growth_sigma: jnp.ndarray    # (3,)  growth function width per channel
    coupling: jnp.ndarray        # (3,3) cross-channel coupling matrix
    state_dt: jnp.ndarray        # ()    integration timestep
    div_mu: jnp.ndarray          # (3,)
    div_sigma: jnp.ndarray       # (3,)
    div_threshold: jnp.ndarray   # ()
    div_prob: jnp.ndarray        # ()
    death_mu: jnp.ndarray        # (3,)
    death_sigma: jnp.ndarray     # (3,)
    death_threshold: jnp.ndarray # ()
    death_prob: jnp.ndarray      # ()


# ── Parameter sampling ───────────────────────────────────────────────────────

def sample_params(rng: jnp.ndarray) -> Params:
    """Sample a random parameter set (matches Rust randomize_params ranges)."""
    keys = jax.random.split(rng, 20)
    k = iter(range(20))

    kernel_mu = 4.0 + jax.random.uniform(keys[next(k)], (3,)) * 5.0
    kernel_sigma = 0.5 + jax.random.uniform(keys[next(k)], (3,)) * 2.0
    growth_mu = jax.random.uniform(keys[next(k)], (3,))
    growth_sigma = 0.1 + jax.random.uniform(keys[next(k)], (3,)) * 0.5

    # Coupling: diagonal dominant, weak off-diagonal
    diag = 0.5 + jax.random.uniform(keys[next(k)], (3,)) * 0.5
    off = (jax.random.uniform(keys[next(k)], (3, 3)) - 0.5) * 0.5
    coupling = off.at[jnp.arange(3), jnp.arange(3)].set(diag)

    state_dt = jax.random.uniform(keys[next(k)], ()) * 0.1

    div_mu = jax.random.uniform(keys[next(k)], (3,))
    div_sigma = 0.1 + jax.random.uniform(keys[next(k)], (3,)) * 0.5
    div_threshold = 0.7 + jax.random.uniform(keys[next(k)], ()) * 0.25
    div_prob = 0.001 + jax.random.uniform(keys[next(k)], ()) * 0.02

    death_mu = jax.random.uniform(keys[next(k)], (3,))
    death_sigma = 0.1 + jax.random.uniform(keys[next(k)], (3,)) * 0.5
    death_threshold = 0.7 + jax.random.uniform(keys[next(k)], ()) * 0.25
    death_prob = 0.001 + jax.random.uniform(keys[next(k)], ()) * 0.02

    return Params(
        kernel_mu=kernel_mu, kernel_sigma=kernel_sigma,
        growth_mu=growth_mu, growth_sigma=growth_sigma,
        coupling=coupling, state_dt=state_dt,
        div_mu=div_mu, div_sigma=div_sigma,
        div_threshold=div_threshold, div_prob=div_prob,
        death_mu=death_mu, death_sigma=death_sigma,
        death_threshold=death_threshold, death_prob=death_prob,
    )


def sample_params_batch(rng: jnp.ndarray, batch_size: int) -> Params:
    """Sample a batch of parameter sets for vmap."""
    keys = jax.random.split(rng, batch_size)
    return jax.vmap(sample_params)(keys)


# ── Initial state from graph ─────────────────────────────────────────────────

def load_initial_graphs(path: str | None = None) -> list[dict]:
    """Load seed graphs from graphs.json."""
    if path is None:
        path = str(Path(__file__).parent / "graphs.json")
    with open(path) as f:
        return json.load(f)


def make_init_state(rng: jnp.ndarray, graph_idx: int = 0) -> SimState:
    """Create initial SimState from a seed graph (padded to MAX_NODES)."""
    graphs = load_initial_graphs()
    g = graphs[graph_idx % len(graphs)]
    n_nodes = len(g["state"])
    edges = g["edgelist"]

    import numpy as np
    nb_np = np.full((MAX_NODES, MAX_DEGREE), EMPTY, dtype=np.int32)
    deg_np = np.zeros(MAX_NODES, dtype=np.int32)

    for a, b in edges:
        if a < MAX_NODES and b < MAX_NODES:
            if deg_np[a] < MAX_DEGREE:
                nb_np[a, deg_np[a]] = b
                deg_np[a] += 1
            if deg_np[b] < MAX_DEGREE:
                nb_np[b, deg_np[b]] = a
                deg_np[b] += 1

    neighbors = jnp.array(nb_np)
    num_neighbors = jnp.array(deg_np)

    rng, k1 = jax.random.split(rng)
    node_states = jax.random.uniform(k1, (MAX_NODES, NUM_CHANNELS))
    node_active = jnp.arange(MAX_NODES) < n_nodes
    node_states = jnp.where(node_active[:, None], node_states, 0.0)

    return SimState(
        node_states=node_states,
        node_active=node_active,
        neighbors=neighbors,
        num_neighbors=num_neighbors,
        node_u=jnp.zeros((MAX_NODES, NUM_CHANNELS)),
        num_active=jnp.array(n_nodes, dtype=jnp.int32),
        rng=rng,
    )


# ── Chebyshev spectral convolution ──────────────────────────────────────────

def compute_cheb_coeffs(params: Params) -> jnp.ndarray:
    """Chebyshev coefficients with adaptive order selection (matches Rust).

    For each channel, compute raw Gaussian c_k = exp(-0.5*((k-mu)/sigma)^2),
    find the order that captures 95% of total mass, zero out higher orders,
    then normalize.
    """
    k = jnp.arange(MAX_CHEB_ORDER)[:, None]  # (K, 1)
    mu = params.kernel_mu[None, :]             # (1, 3)
    sigma = params.kernel_sigma[None, :]       # (1, 3)
    raw = jnp.exp(-0.5 * ((k - mu) / sigma) ** 2)  # (K, 3)

    # Adaptive order: find where cumulative sum reaches 95% of total
    total = raw.sum(axis=0, keepdims=True)        # (1, 3)
    cumsum = jnp.cumsum(raw, axis=0)               # (K, 3)
    threshold = 0.95 * total                       # (1, 3)
    # Order per channel = first k where cumsum >= 95%, clamped to [2, MAX_CHEB_ORDER]
    above = cumsum >= threshold                     # (K, 3) bool
    # argmax on bool gives first True index
    order_per_ch = jnp.argmax(above, axis=0) + 1   # (3,) add 1 for inclusive
    order_per_ch = jnp.clip(order_per_ch, 2, MAX_CHEB_ORDER)

    # Mask: zero out coefficients beyond the per-channel order
    mask = k < order_per_ch[None, :]  # (K, 3)
    raw = raw * mask

    # Normalize per channel
    total_masked = raw.sum(axis=0, keepdims=True)
    return raw / jnp.maximum(total_masked, 1e-8)


def neighbor_mean(states, neighbors, num_neighbors, node_active):
    """W(x): mean of neighbour states for each node."""
    safe_idx = jnp.clip(neighbors, 0, MAX_NODES - 1)
    nb_states = states[safe_idx]
    valid = (jnp.arange(MAX_DEGREE)[None, :] < num_neighbors[:, None])
    nb_active = node_active[safe_idx]
    mask = valid & nb_active
    masked = nb_states * mask[:, :, None]
    total = masked.sum(axis=1)
    degree = mask.sum(axis=1, keepdims=True).astype(jnp.float32)
    return total / jnp.maximum(degree, 1.0)


def chebyshev_convolution(state: SimState, params: Params) -> jnp.ndarray:
    """Chebyshev spectral filter on node states."""
    coeffs = compute_cheb_coeffs(params)
    x = state.node_states
    t_prev_prev = x
    t_prev = neighbor_mean(x, state.neighbors, state.num_neighbors, state.node_active)
    result = coeffs[0][None, :] * t_prev_prev + coeffs[1][None, :] * t_prev

    def cheb_step(carry, ck):
        t_pp, t_p, res = carry
        w_t_p = neighbor_mean(t_p, state.neighbors, state.num_neighbors, state.node_active)
        t_new = 2.0 * w_t_p - t_pp
        res = res + ck[None, :] * t_new
        return (t_p, t_new, res), None

    (_, _, result), _ = jax.lax.scan(
        cheb_step, (t_prev_prev, t_prev, result), coeffs[2:])
    return result * state.node_active[:, None]


# ── Growth function ──────────────────────────────────────────────────────────

def apply_growth(state: SimState, params: Params) -> SimState:
    """Convolve → couple → Gaussian growth → integrate."""
    conv = chebyshev_convolution(state, params)
    u = jnp.einsum("ij,nj->ni", params.coupling, conv)
    diff = (u - params.growth_mu[None, :]) / params.growth_sigma[None, :]
    g = 2.0 * jnp.exp(-0.5 * diff ** 2) - 1.0
    new_states = jnp.clip(state.node_states + params.state_dt * g, 0.0, 1.0)
    new_states = jnp.where(state.node_active[:, None], new_states, 0.0)
    return state._replace(node_states=new_states, node_u=u)


# ── Topology: config signal ──────────────────────────────────────────────────

def config_signal(u, mu, sigma):
    """Gaussian signal over u-space, averaged across channels."""
    s = jnp.maximum(sigma[None, :], 0.01)
    per_ch = jnp.exp(-0.5 * ((u - mu[None, :]) / s) ** 2)
    return per_ch.mean(axis=1)


# ── Topology: division ───────────────────────────────────────────────────────

def apply_divisions(state: SimState, params: Params) -> SimState:
    """Divide nodes whose u-signal exceeds threshold."""
    rng, k1, k2 = jax.random.split(state.rng, 3)

    div_sig = config_signal(state.node_u, params.div_mu, params.div_sigma)
    rand = jax.random.uniform(k1, (MAX_NODES,))
    wants_div = (
        state.node_active
        & (div_sig > params.div_threshold)
        & (rand < params.div_prob)
    )

    sort_key = jnp.where(wants_div, -div_sig, 1e6)
    div_order = jnp.argsort(sort_key)[:MAX_TOPO_OPS]

    free_mask = ~state.node_active
    total_free = jnp.sum(free_mask)
    free_order = jnp.argsort(~free_mask)
    free_slots = free_order[:MAX_TOPO_OPS * 2]

    num_wanting = jnp.sum(wants_div)
    max_by_free = total_free // 2
    num_divs = jnp.minimum(num_wanting, jnp.minimum(max_by_free, MAX_TOPO_OPS))
    op_valid = jnp.arange(MAX_TOPO_OPS) < num_divs

    parent_idx = div_order
    child_a_idx = free_slots[0::2][:MAX_TOPO_OPS]
    child_b_idx = free_slots[1::2][:MAX_TOPO_OPS]

    new_active = state.node_active
    new_active = new_active.at[child_a_idx].set(
        jnp.where(op_valid, True, new_active[child_a_idx]))
    new_active = new_active.at[child_b_idx].set(
        jnp.where(op_valid, True, new_active[child_b_idx]))

    parent_states = state.node_states[parent_idx]
    new_states = state.node_states
    new_states = new_states.at[child_a_idx].set(
        jnp.where(op_valid[:, None], parent_states, new_states[child_a_idx]))
    new_states = new_states.at[child_b_idx].set(
        jnp.where(op_valid[:, None], parent_states, new_states[child_b_idx]))

    new_neighbors = state.neighbors
    new_num_nb = state.num_neighbors

    def add_neighbor(neighbors, num_nb, node_a, node_b, valid):
        deg = num_nb[node_a]
        has_space = deg < MAX_DEGREE
        do_add = valid & has_space
        neighbors = neighbors.at[node_a, deg].set(
            jnp.where(do_add, node_b, neighbors[node_a, deg]))
        num_nb = num_nb.at[node_a].set(jnp.where(do_add, deg + 1, deg))
        return neighbors, num_nb

    def connect_division(carry, op):
        nb, n_nb = carry
        parent, ca, cb, valid = op

        # IMPORTANT: Redistribute parent's existing neighbors FIRST,
        # before adding triangle edges. This matches the Rust implementation
        # and prevents children from being redistributed to themselves.

        # Redistribute parent's first existing neighbor to child_a
        first_nb = nb[parent, 0]
        has_first = (n_nb[parent] > 0) & valid & (first_nb >= 0)

        # Remove first_nb from parent (swap with last)
        last_slot = jnp.maximum(n_nb[parent] - 1, 0)
        nb = nb.at[parent, 0].set(jnp.where(has_first, nb[parent, last_slot], nb[parent, 0]))
        nb = nb.at[parent, last_slot].set(jnp.where(has_first, EMPTY, nb[parent, last_slot]))
        n_nb = n_nb.at[parent].set(jnp.where(has_first, n_nb[parent] - 1, n_nb[parent]))

        # Connect child_a <-> first_nb (replace parent ref in first_nb's list)
        nb, n_nb = add_neighbor(nb, n_nb, ca, first_nb, has_first)
        safe_first = jnp.clip(first_nb, 0, MAX_NODES - 1)
        row = nb[safe_first]
        deg = n_nb[safe_first]
        matches = (row == parent) & (jnp.arange(MAX_DEGREE) < deg)
        first_match = jnp.argmax(matches)
        has_match = matches.any() & has_first
        row = row.at[first_match].set(jnp.where(has_match, ca, row[first_match]))
        nb = nb.at[safe_first].set(jnp.where(has_first, row, nb[safe_first]))

        # Redistribute parent's (new) first existing neighbor to child_b
        second_nb = nb[parent, 0]
        has_second = (n_nb[parent] > 0) & valid & (second_nb >= 0)

        last_slot2 = jnp.maximum(n_nb[parent] - 1, 0)
        nb = nb.at[parent, 0].set(jnp.where(has_second, nb[parent, last_slot2], nb[parent, 0]))
        nb = nb.at[parent, last_slot2].set(jnp.where(has_second, EMPTY, nb[parent, last_slot2]))
        n_nb = n_nb.at[parent].set(jnp.where(has_second, n_nb[parent] - 1, n_nb[parent]))

        nb, n_nb = add_neighbor(nb, n_nb, cb, second_nb, has_second)
        safe_second = jnp.clip(second_nb, 0, MAX_NODES - 1)
        row2 = nb[safe_second]
        deg2 = n_nb[safe_second]
        matches2 = (row2 == parent) & (jnp.arange(MAX_DEGREE) < deg2)
        first_match2 = jnp.argmax(matches2)
        has_match2 = matches2.any() & has_second
        row2 = row2.at[first_match2].set(jnp.where(has_match2, cb, row2[first_match2]))
        nb = nb.at[safe_second].set(jnp.where(has_second, row2, nb[safe_second]))

        # NOW add triangle edges: parent↔ca, parent↔cb, ca↔cb
        nb, n_nb = add_neighbor(nb, n_nb, parent, ca, valid)
        nb, n_nb = add_neighbor(nb, n_nb, ca, parent, valid)
        nb, n_nb = add_neighbor(nb, n_nb, parent, cb, valid)
        nb, n_nb = add_neighbor(nb, n_nb, cb, parent, valid)
        nb, n_nb = add_neighbor(nb, n_nb, ca, cb, valid)
        nb, n_nb = add_neighbor(nb, n_nb, cb, ca, valid)

        return (nb, n_nb), None

    ops = jnp.stack([parent_idx, child_a_idx, child_b_idx,
                     op_valid.astype(jnp.int32)], axis=1)
    (new_neighbors, new_num_nb), _ = jax.lax.scan(
        connect_division, (new_neighbors, new_num_nb), ops)

    new_num_active = jnp.sum(new_active.astype(jnp.int32))
    return state._replace(
        node_states=new_states, node_active=new_active,
        neighbors=new_neighbors, num_neighbors=new_num_nb,
        num_active=new_num_active, rng=rng)


# ── Topology: collapse ───────────────────────────────────────────────────────

def apply_collapses(state: SimState, params: Params) -> SimState:
    """Triangle merge: find a triangle around the dying node, merge 3→1.

    Matches the Rust implementation: for a dying node i, find two neighbors
    a, b where a-b are also connected (forming a triangle). Merge i, a, b
    into survivor a. Average states from all 3. Rewire all connections from
    i and b to point to a.
    """
    rng, k1 = jax.random.split(state.rng)

    death_sig = config_signal(state.node_u, params.death_mu, params.death_sigma)
    rand = jax.random.uniform(k1, (MAX_NODES,))
    wants_die = (
        state.node_active
        & (death_sig > params.death_threshold)
        & (rand < params.death_prob)
        & (state.num_neighbors > 1)  # need at least 2 neighbors to form triangle
    )

    num_dying = jnp.sum(wants_die)
    # Each collapse removes 2 nodes, need at least 3 to survive
    max_deaths = jnp.maximum((state.num_active - 3) // 2, 0)
    allowed_deaths = jnp.minimum(num_dying, jnp.minimum(max_deaths, MAX_TOPO_OPS))

    sort_key = jnp.where(wants_die, -death_sig, 1e6)
    die_order = jnp.argsort(sort_key)[:MAX_TOPO_OPS]
    op_valid = jnp.arange(MAX_TOPO_OPS) < allowed_deaths

    def add_neighbor(neighbors, num_nb, node_a, node_b, valid):
        """Add node_b to node_a's neighbor list if valid and space available."""
        deg = num_nb[node_a]
        already = jnp.any((neighbors[node_a] == node_b) & (jnp.arange(MAX_DEGREE) < deg))
        do_add = valid & (deg < MAX_DEGREE) & ~already & (node_a != node_b)
        neighbors = neighbors.at[node_a, deg].set(
            jnp.where(do_add, node_b, neighbors[node_a, deg]))
        num_nb = num_nb.at[node_a].set(jnp.where(do_add, deg + 1, deg))
        return neighbors, num_nb

    def remove_from_list(nb, n_nb, node, target, valid):
        """Remove target from node's neighbor list."""
        row = nb[node]
        deg = n_nb[node]
        matches = (row == target) & (jnp.arange(MAX_DEGREE) < deg)
        match_pos = jnp.argmax(matches)
        has_match = matches.any() & valid
        last_pos = jnp.maximum(deg - 1, 0)
        new_row = row.at[match_pos].set(
            jnp.where(has_match, row[last_pos], row[match_pos]))
        new_row = new_row.at[last_pos].set(
            jnp.where(has_match, EMPTY, new_row[last_pos]))
        nb = nb.at[node].set(jnp.where(has_match, new_row, nb[node]))
        n_nb = n_nb.at[node].set(jnp.where(has_match, deg - 1, deg))
        return nb, n_nb

    def collapse_triangle(carry, op):
        nb, n_nb, active, states = carry
        dead_idx, valid = op[0], op[1].astype(bool)
        valid = valid & active[dead_idx] & (n_nb[dead_idx] > 1)

        # Find a triangle: two neighbors of dead_idx that are also connected
        # Scan pairs of neighbors to find a connected pair
        dead_nbs = nb[dead_idx]  # (MAX_DEGREE,)
        dead_deg = n_nb[dead_idx]

        # For each pair (ni, nj) of dead_idx's neighbors, check if ni-nj connected
        # We'll find the first valid triangle
        def find_triangle_pair(carry, ni_idx):
            found, tri_a, tri_b = carry
            ni = jnp.clip(dead_nbs[ni_idx], 0, MAX_NODES - 1)
            ni_valid = (ni_idx < dead_deg) & ~found & valid & (dead_nbs[ni_idx] >= 0)

            # Check if any later neighbor of dead_idx is in ni's neighbor list
            def check_nj(inner_carry, nj_idx):
                found_inner, best_b = inner_carry
                nj = jnp.clip(dead_nbs[nj_idx], 0, MAX_NODES - 1)
                nj_valid = (nj_idx < dead_deg) & (nj_idx > ni_idx) & ~found_inner
                nj_valid = nj_valid & (dead_nbs[nj_idx] >= 0) & ni_valid

                # Check if nj is in ni's neighbor list
                ni_row = nb[ni]
                ni_deg = n_nb[ni]
                is_connected = jnp.any(
                    (ni_row == nj) & (jnp.arange(MAX_DEGREE) < ni_deg))
                match = nj_valid & is_connected

                best_b = jnp.where(match, nj, best_b)
                found_inner = found_inner | match
                return (found_inner, best_b), None

            (found_pair, best_b), _ = jax.lax.scan(
                check_nj, (False, jnp.int32(0)), jnp.arange(MAX_DEGREE))

            tri_a = jnp.where(ni_valid & found_pair & ~found, ni, tri_a)
            tri_b = jnp.where(ni_valid & found_pair & ~found, best_b, tri_b)
            found = found | (ni_valid & found_pair)
            return (found, tri_a, tri_b), None

        (found_tri, survivor, doomed), _ = jax.lax.scan(
            find_triangle_pair,
            (False, jnp.int32(0), jnp.int32(0)),
            jnp.arange(MAX_DEGREE))

        do_merge = valid & found_tri
        survivor = jnp.clip(survivor, 0, MAX_NODES - 1)
        doomed = jnp.clip(doomed, 0, MAX_NODES - 1)

        # Average states from all 3 nodes
        avg = (states[dead_idx] + states[survivor] + states[doomed]) / 3.0
        states = states.at[survivor].set(
            jnp.where(do_merge, avg, states[survivor]))

        # Deactivate dead_idx and doomed
        active = active.at[dead_idx].set(jnp.where(do_merge, False, active[dead_idx]))
        active = active.at[doomed].set(jnp.where(do_merge, False, active[doomed]))
        states = states.at[dead_idx].set(
            jnp.where(do_merge, jnp.zeros(NUM_CHANNELS), states[dead_idx]))
        states = states.at[doomed].set(
            jnp.where(do_merge, jnp.zeros(NUM_CHANNELS), states[doomed]))

        # Rewire: survivor inherits external neighbors of dead_idx and doomed.
        # Step 1: Remove internal triangle edges from survivor's list
        nb, n_nb = remove_from_list(nb, n_nb, survivor, dead_idx, do_merge)
        nb, n_nb = remove_from_list(nb, n_nb, survivor, doomed, do_merge)

        # Step 2: For each neighbor of dead_idx, rewire and add to survivor
        def process_dead_neighbor(carry, slot_idx):
            nb, n_nb = carry
            nb_node_raw = nb[dead_idx, slot_idx]
            nb_node = jnp.clip(nb_node_raw, 0, MAX_NODES - 1)
            slot_valid = do_merge & (slot_idx < n_nb[dead_idx]) & (nb_node_raw >= 0)
            is_external = (nb_node != survivor) & (nb_node != doomed) & (nb_node != dead_idx)
            ext_valid = slot_valid & is_external

            # In nb_node's list, replace dead_idx with survivor
            def fix_slot(inner_carry, s):
                nb_i, n_nb_i = inner_carry
                val = nb_i[nb_node, s]
                sv = ext_valid & (s < n_nb_i[nb_node]) & (val == dead_idx)
                nb_i = nb_i.at[nb_node, s].set(jnp.where(sv, survivor, val))
                return (nb_i, n_nb_i), None

            (nb, n_nb), _ = jax.lax.scan(fix_slot, (nb, n_nb), jnp.arange(MAX_DEGREE))

            # Add nb_node to survivor's neighbor list
            nb, n_nb = add_neighbor(nb, n_nb, survivor, nb_node, ext_valid)
            return (nb, n_nb), None

        (nb, n_nb), _ = jax.lax.scan(
            process_dead_neighbor, (nb, n_nb), jnp.arange(MAX_DEGREE))

        # Step 3: For each neighbor of doomed, rewire and add to survivor
        def process_doomed_neighbor(carry, slot_idx):
            nb, n_nb = carry
            nb_node_raw = nb[doomed, slot_idx]
            nb_node = jnp.clip(nb_node_raw, 0, MAX_NODES - 1)
            slot_valid = do_merge & (slot_idx < n_nb[doomed]) & (nb_node_raw >= 0)
            is_external = (nb_node != survivor) & (nb_node != doomed) & (nb_node != dead_idx)
            ext_valid = slot_valid & is_external

            def fix_slot(inner_carry, s):
                nb_i, n_nb_i = inner_carry
                val = nb_i[nb_node, s]
                sv = ext_valid & (s < n_nb_i[nb_node]) & (val == doomed)
                nb_i = nb_i.at[nb_node, s].set(jnp.where(sv, survivor, val))
                return (nb_i, n_nb_i), None

            (nb, n_nb), _ = jax.lax.scan(fix_slot, (nb, n_nb), jnp.arange(MAX_DEGREE))

            # Add nb_node to survivor's neighbor list
            nb, n_nb = add_neighbor(nb, n_nb, survivor, nb_node, ext_valid)
            return (nb, n_nb), None

        (nb, n_nb), _ = jax.lax.scan(
            process_doomed_neighbor, (nb, n_nb), jnp.arange(MAX_DEGREE))

        # Clear dead and doomed neighbor lists (cleanup done globally after scan)
        nb = nb.at[dead_idx].set(
            jnp.where(do_merge, jnp.full(MAX_DEGREE, EMPTY, dtype=jnp.int32), nb[dead_idx]))
        n_nb = n_nb.at[dead_idx].set(jnp.where(do_merge, 0, n_nb[dead_idx]))
        nb = nb.at[doomed].set(
            jnp.where(do_merge, jnp.full(MAX_DEGREE, EMPTY, dtype=jnp.int32), nb[doomed]))
        n_nb = n_nb.at[doomed].set(jnp.where(do_merge, 0, n_nb[doomed]))

        return (nb, n_nb, active, states), None

    ops = jnp.stack([die_order, op_valid.astype(jnp.int32)], axis=1)
    (new_nb, new_n_nb, new_active, new_states), _ = jax.lax.scan(
        collapse_triangle,
        (state.neighbors, state.num_neighbors, state.node_active, state.node_states),
        ops)

    # ── Global cleanup pass ──────────────────────────────────────────────
    # After all collapses, clean up ALL neighbor lists:
    # 1. Remove references to inactive nodes
    # 2. Remove self-loops
    # 3. Remove duplicates
    def cleanup_node(carry, node_idx):
        nb, n_nb = carry
        is_active = new_active[node_idx]

        def clean_slot(inner, s):
            nb_i, n_nb_i = inner
            val = nb_i[node_idx, s]
            deg = n_nb_i[node_idx]
            slot_valid = is_active & (s < deg) & (val >= 0)

            # Check: inactive ref, self-loop, or duplicate
            is_bad = slot_valid & (
                ~new_active[jnp.clip(val, 0, MAX_NODES - 1)]  # inactive
                | (val == node_idx)                             # self-loop
                | jnp.any((nb_i[node_idx, :] == val)           # duplicate
                          & (jnp.arange(MAX_DEGREE) < s))
            )

            # Remove by swap-with-last
            last = jnp.maximum(n_nb_i[node_idx] - 1, 0)
            nb_i = nb_i.at[node_idx, s].set(
                jnp.where(is_bad, nb_i[node_idx, last], nb_i[node_idx, s]))
            nb_i = nb_i.at[node_idx, last].set(
                jnp.where(is_bad, EMPTY, nb_i[node_idx, last]))
            n_nb_i = n_nb_i.at[node_idx].set(
                jnp.where(is_bad, n_nb_i[node_idx] - 1, n_nb_i[node_idx]))
            return (nb_i, n_nb_i), None

        # Run cleanup multiple passes (swap-with-last can reveal new issues)
        (nb, n_nb), _ = jax.lax.scan(clean_slot, (nb, n_nb), jnp.arange(MAX_DEGREE))
        (nb, n_nb), _ = jax.lax.scan(clean_slot, (nb, n_nb), jnp.arange(MAX_DEGREE))
        return (nb, n_nb), None

    (new_nb, new_n_nb), _ = jax.lax.scan(
        cleanup_node, (new_nb, new_n_nb), jnp.arange(MAX_NODES))

    new_num_active = jnp.sum(new_active.astype(jnp.int32))
    return state._replace(
        node_states=new_states, node_active=new_active,
        neighbors=new_nb, num_neighbors=new_n_nb,
        num_active=new_num_active, rng=rng)


# ── Metrics ──────────────────────────────────────────────────────────────────

class StepMetrics(NamedTuple):
    """Per-step metrics for evaluating interestingness."""
    num_active: jnp.ndarray
    mean_state: jnp.ndarray
    state_variance: jnp.ndarray
    state_change: jnp.ndarray
    mean_degree: jnp.ndarray


def compute_step_metrics(state: SimState, prev_states: jnp.ndarray) -> StepMetrics:
    active_f = state.node_active.astype(jnp.float32)
    n = jnp.maximum(state.num_active.astype(jnp.float32), 1.0)

    masked = state.node_states * active_f[:, None]
    mean_state = masked.sum(axis=0) / n
    diff_from_mean = (state.node_states - mean_state[None, :]) * active_f[:, None]
    state_var = (diff_from_mean ** 2).sum(axis=0) / n

    delta = jnp.abs(state.node_states - prev_states) * active_f[:, None]
    state_change = delta.sum() / (n * NUM_CHANNELS)

    mean_deg = (state.num_neighbors * active_f).sum() / n

    return StepMetrics(
        num_active=state.num_active,
        mean_state=mean_state,
        state_variance=state_var,
        state_change=state_change,
        mean_degree=mean_deg,
    )


# ── Graph structure features (divergence from random null model) ─────────────

class GraphFeatures(NamedTuple):
    """Structural features for comparison with random-graph analytical baseline.

    The null model is a graph with the same N and E but grown under
    spatially-uniform division/collapse probabilities — which converges
    to Erdos-Renyi-like statistics: Poisson degree distribution,
    clustering ~ k/(n-1), and zero degree assortativity.
    """
    degree_kl: jnp.ndarray         # () KL(actual degree dist || truncated Poisson)
    clustering: jnp.ndarray        # () mean clustering coefficient
    clustering_expected: jnp.ndarray  # () expected C for random graph with same N, E
    assortativity: jnp.ndarray     # () degree-degree Pearson correlation


def _truncated_poisson_pmf(mean_k: jnp.ndarray) -> jnp.ndarray:
    """Poisson PMF truncated at MAX_DEGREE, renormalized."""
    ks = jnp.arange(MAX_DEGREE + 1, dtype=jnp.float32)
    log_pmf = ks * jnp.log(jnp.maximum(mean_k, 1e-8)) \
        - mean_k - jax.lax.lgamma(ks + 1)
    pmf = jnp.exp(log_pmf)
    return pmf / jnp.maximum(pmf.sum(), 1e-8)


def _degree_histogram(num_neighbors, node_active, num_active):
    """Normalized degree histogram over active nodes."""
    degrees = num_neighbors * node_active.astype(jnp.int32)
    bins = jnp.arange(MAX_DEGREE + 1)
    one_hot = (degrees[:, None] == bins[None, :]).astype(jnp.float32)
    one_hot = one_hot * node_active[:, None].astype(jnp.float32)
    hist = one_hot.sum(axis=0)
    return hist / jnp.maximum(num_active.astype(jnp.float32), 1.0)


def _mean_clustering(neighbors, num_neighbors, node_active, num_active):
    """Mean clustering coefficient over active nodes.

    For each node, counts triangles among its neighbors using the
    neighbor-list representation. With MAX_DEGREE=8 the inner tensors
    are (8,8,8) — tiny per node, fine under nested vmap.
    """
    def node_clustering(i):
        nb = neighbors[i]                     # (D,)
        d = num_neighbors[i]
        safe_nb = jnp.clip(nb, 0, MAX_NODES - 1)

        # Neighbor-of-neighbor lists
        nb_of_nb = neighbors[safe_nb]          # (D, D)
        nb_of_nb_count = num_neighbors[safe_nb]  # (D,)

        # matches[s1, s2, slot] = (neighbors[nb[s1]][slot] == nb[s2])
        matches = (nb_of_nb[:, None, :] == nb[None, :, None])  # (D, D, D)

        slot_idx = jnp.arange(MAX_DEGREE)
        slot_valid = slot_idx[None, None, :] < nb_of_nb_count[:, None, None]
        connected = (matches & slot_valid).any(axis=2)  # (D, D)

        upper = slot_idx[None, :] > slot_idx[:, None]
        pair_valid = (slot_idx[:, None] < d) & (slot_idx[None, :] < d)
        nb_nonneg = (nb[:, None] >= 0) & (nb[None, :] >= 0)
        mask = upper & pair_valid & nb_nonneg

        triangles = (connected & mask).sum().astype(jnp.float32)
        possible = d.astype(jnp.float32) * (d.astype(jnp.float32) - 1.0) / 2.0
        cc = jnp.where(possible > 0, triangles / possible, 0.0)
        return cc * node_active[i].astype(jnp.float32)

    all_cc = jax.vmap(node_clustering)(jnp.arange(MAX_NODES))
    return all_cc.sum() / jnp.maximum(num_active.astype(jnp.float32), 1.0)


def _degree_assortativity(neighbors, num_neighbors, node_active):
    """Pearson correlation of degrees at edge endpoints."""
    degrees = num_neighbors.astype(jnp.float32)
    slot_idx = jnp.arange(MAX_DEGREE)[None, :]
    edge_valid = (slot_idx < num_neighbors[:, None]) \
        & node_active[:, None]                       # (N, D)
    safe_nb = jnp.clip(neighbors, 0, MAX_NODES - 1)

    deg_i = jnp.broadcast_to(degrees[:, None], (MAX_NODES, MAX_DEGREE))
    deg_j = degrees[safe_nb]
    w = edge_valid.astype(jnp.float32)
    total = jnp.maximum(w.sum(), 1.0)

    mean_i = (deg_i * w).sum() / total
    mean_j = (deg_j * w).sum() / total
    cov = ((deg_i - mean_i) * (deg_j - mean_j) * w).sum() / total
    std_i = jnp.sqrt(jnp.maximum(((deg_i - mean_i) ** 2 * w).sum() / total, 1e-8))
    std_j = jnp.sqrt(jnp.maximum(((deg_j - mean_j) ** 2 * w).sum() / total, 1e-8))
    return cov / jnp.maximum(std_i * std_j, 1e-8)


def compute_graph_features(state: SimState) -> GraphFeatures:
    """Compute structural features of the current graph."""
    n = jnp.maximum(state.num_active.astype(jnp.float32), 1.0)
    active = state.node_active.astype(jnp.float32)
    mean_k = (state.num_neighbors.astype(jnp.float32) * active).sum() / n

    # Degree distribution KL divergence vs truncated Poisson
    actual = _degree_histogram(state.num_neighbors, state.node_active,
                               state.num_active)
    expected = _truncated_poisson_pmf(mean_k)
    eps = 1e-8
    degree_kl = jnp.sum(jnp.where(
        actual > eps,
        actual * jnp.log(actual / (expected + eps)),
        0.0))

    # Clustering coefficient vs random expectation C_rand = k/(n-1)
    clustering = _mean_clustering(state.neighbors, state.num_neighbors,
                                  state.node_active, state.num_active)
    clustering_expected = mean_k / jnp.maximum(n - 1.0, 1.0)

    # Degree assortativity (random expectation ≈ 0)
    assortativity = _degree_assortativity(
        state.neighbors, state.num_neighbors, state.node_active)

    return GraphFeatures(
        degree_kl=degree_kl,
        clustering=clustering,
        clustering_expected=clustering_expected,
        assortativity=assortativity,
    )


def random_divergence(state: SimState) -> jnp.ndarray:
    """Scalar: how far is this graph from a random graph with the same N, E?

    Combines three signals — degree-distribution KL, excess clustering
    beyond the random baseline, and nonzero degree assortativity.
    Higher values ⇒ the CA dynamics sculpted more non-random structure.
    """
    f = compute_graph_features(state)
    clustering_excess = jnp.where(
        f.clustering_expected > 1e-6,
        f.clustering / f.clustering_expected - 1.0,
        f.clustering * 100.0)
    return (
        f.degree_kl * 2.0
        + jnp.abs(clustering_excess) * 1.5
        + jnp.abs(f.assortativity) * 1.0
    )


# ── Simulation runner ────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(2,))
def run_simulation(params: Params, init_state: SimState,
                   num_steps: int = 500) -> tuple[SimState, StepMetrics]:
    """Run full simulation, collecting per-step metrics."""
    def step_fn(carry, step_idx):
        state, prev_states = carry
        state = apply_growth(state, params)

        do_topo = (step_idx % TOPO_EVERY == 0) & (step_idx > 0)
        state = jax.lax.cond(
            do_topo,
            lambda s: apply_collapses(apply_divisions(s, params), params),
            lambda s: s,
            state,
        )

        metrics = compute_step_metrics(state, prev_states)
        return (state, state.node_states), metrics

    init_carry = (init_state, init_state.node_states)
    (final_state, _), all_metrics = jax.lax.scan(
        step_fn, init_carry, jnp.arange(num_steps))
    return final_state, all_metrics
