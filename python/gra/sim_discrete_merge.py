"""
Discrete Graph-Rewriting Automata with Split + Merge — JAX simulation core.

Extends the Cousin & Maignan (2022) model with a third state for merging:
- 3-regular graphs with ternary node states: 0=merge, 1=neutral, 2=split
- 21 possible configurations: c(v) = 7*s(v) + sum(s(neighbors))
  (own state in {0,1,2}, neighbor sum in {0..6})
- Rule R maps config -> new state in {0,1,2}
- State 2 triggers division (same 3-regular rewiring as the 2-state model)
- State 0 triggers triangle merge (inverse of division, maintains 3-regularity)
- 3^21 ~ 10.5 billion possible rules
"""

import json
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

# ── Constants (module-level for JIT visibility) ──────────────────────────────

MAX_NODES = 2048
DEGREE = 3
MAX_TOPO_OPS = 128
EMPTY = -1

# State semantics
MERGE = 0
NEUTRAL = 1
SPLIT = 2
NUM_STATES = 3
NUM_CONFIGS = NUM_STATES * (DEGREE * (NUM_STATES - 1) + 1)  # 3 * 7 = 21


def configure(cfg):
    """Set module constants from config. Must call before JIT."""
    global MAX_NODES, MAX_TOPO_OPS
    sim = cfg if not hasattr(cfg, "sim") else cfg.sim
    MAX_NODES = int(sim.max_nodes)
    MAX_TOPO_OPS = int(sim.max_topo_ops)


# ── Data structures ──────────────────────────────────────────────────────────

class SimState(NamedTuple):
    """Fixed-size simulation state for a 3-regular graph with ternary states."""
    node_states: jnp.ndarray   # (MAX_NODES,) int32 {0, 1, 2}
    node_active: jnp.ndarray   # (MAX_NODES,) bool
    neighbors: jnp.ndarray     # (MAX_NODES, 3) int32, EMPTY=-1 for inactive
    num_active: jnp.ndarray    # () scalar int32
    rng: jnp.ndarray           # PRNGKey


class Rule(NamedTuple):
    """A single lookup table: 21 configs -> {0, 1, 2}."""
    table: jnp.ndarray  # (21,) int32 in {0, 1, 2}


class StepMetrics(NamedTuple):
    """Per-step observables for fitness evaluation."""
    num_active: jnp.ndarray         # () total graph nodes
    num_per_state: jnp.ndarray      # (3,) count of nodes in each state
    state_fractions: jnp.ndarray    # (3,) fraction in each state
    num_state_changes: jnp.ndarray  # () nodes that changed state
    num_divisions: jnp.ndarray      # () divisions performed
    num_merges: jnp.ndarray         # () merges performed


# ── Rule encoding / decoding ─────────────────────────────────────────────────

def rule_from_trits(trits: jnp.ndarray) -> Rule:
    """Create a Rule from a (21,) array of trits in {0,1,2}."""
    return Rule(table=trits.astype(jnp.int32))


def rule_to_number(rule: Rule) -> jnp.ndarray:
    """Encode a Rule to a large integer (for display, uses int64)."""
    powers = 3 ** jnp.arange(NUM_CONFIGS, dtype=jnp.int64)
    return (rule.table.astype(jnp.int64) * powers).sum()


def sample_rule(rng: jnp.ndarray) -> Rule:
    """Sample a uniformly random rule."""
    table = jax.random.randint(rng, (NUM_CONFIGS,), 0, NUM_STATES)
    return Rule(table=table)


def mutate_rule(rule: Rule, rng: jnp.ndarray, flip_prob: float = 0.1) -> Rule:
    """Mutate a rule by independently re-randomizing each trit with probability p."""
    k1, k2 = jax.random.split(rng)
    do_flip = jax.random.bernoulli(k1, p=flip_prob, shape=(NUM_CONFIGS,))
    new_vals = jax.random.randint(k2, (NUM_CONFIGS,), 0, NUM_STATES)
    table = jnp.where(do_flip, new_vals, rule.table)
    return Rule(table=table)


def rules_from_tables(tables: jnp.ndarray) -> Rule:
    """Batch-create Rules from (batch, 21) array."""
    return Rule(table=tables.astype(jnp.int32))


# ── Initial graphs ───────────────────────────────────────────────────────────

def make_k4_state(rng: jnp.ndarray) -> SimState:
    """K4: complete graph on 4 nodes (simplest 3-regular graph)."""
    n = 4
    nb = np.full((MAX_NODES, DEGREE), EMPTY, dtype=np.int32)
    nb[0] = [1, 2, 3]
    nb[1] = [0, 2, 3]
    nb[2] = [0, 1, 3]
    nb[3] = [0, 1, 2]

    states = np.full(MAX_NODES, NEUTRAL, dtype=np.int32)
    states[0] = SPLIT  # seed one split node

    active = np.zeros(MAX_NODES, dtype=bool)
    active[:n] = True
    # Inactive nodes get neutral state (doesn't matter, masked out)
    states[n:] = 0

    return SimState(
        node_states=jnp.array(states),
        node_active=jnp.array(active),
        neighbors=jnp.array(nb),
        num_active=jnp.array(n, dtype=jnp.int32),
        rng=rng,
    )


def make_petersen_state(rng: jnp.ndarray) -> SimState:
    """Petersen graph: 10 nodes, 3-regular, good config coverage."""
    n = 10
    nb = np.full((MAX_NODES, DEGREE), EMPTY, dtype=np.int32)
    nb[0] = [1, 4, 5]
    nb[1] = [0, 2, 6]
    nb[2] = [1, 3, 7]
    nb[3] = [2, 4, 8]
    nb[4] = [3, 0, 9]
    nb[5] = [0, 7, 8]
    nb[6] = [1, 8, 9]
    nb[7] = [2, 5, 9]
    nb[8] = [3, 5, 6]
    nb[9] = [4, 6, 7]

    # Mix of states to seed activity
    states = np.full(MAX_NODES, NEUTRAL, dtype=np.int32)
    states[:10] = [SPLIT, NEUTRAL, SPLIT, NEUTRAL, MERGE,
                   SPLIT, NEUTRAL, MERGE, SPLIT, NEUTRAL]
    states[10:] = 0

    active = np.zeros(MAX_NODES, dtype=bool)
    active[:n] = True

    return SimState(
        node_states=jnp.array(states),
        node_active=jnp.array(active),
        neighbors=jnp.array(nb),
        num_active=jnp.array(n, dtype=jnp.int32),
        rng=rng,
    )


def make_prism_state(rng: jnp.ndarray) -> SimState:
    """Triangular prism: 6 nodes, 3-regular, compact seed graph."""
    n = 6
    nb = np.full((MAX_NODES, DEGREE), EMPTY, dtype=np.int32)
    nb[0] = [1, 2, 3]
    nb[1] = [0, 2, 4]
    nb[2] = [0, 1, 5]
    nb[3] = [0, 4, 5]
    nb[4] = [1, 3, 5]
    nb[5] = [2, 3, 4]

    states = np.full(MAX_NODES, NEUTRAL, dtype=np.int32)
    states[:6] = [SPLIT, NEUTRAL, SPLIT, NEUTRAL, MERGE, NEUTRAL]
    states[6:] = 0

    active = np.zeros(MAX_NODES, dtype=bool)
    active[:n] = True

    return SimState(
        node_states=jnp.array(states),
        node_active=jnp.array(active),
        neighbors=jnp.array(nb),
        num_active=jnp.array(n, dtype=jnp.int32),
        rng=rng,
    )


GRAPH_BUILDERS = {
    "k4": make_k4_state,
    "petersen": make_petersen_state,
    "prism": make_prism_state,
}


def _make_state_from_json(rng: jnp.ndarray, graph_idx: int = 0) -> SimState:
    """Load a 3-regular seed graph from graphs.json."""
    path = Path(__file__).parent / "graphs.json"
    with open(path) as f:
        graphs = json.load(f)
    g = graphs[graph_idx % len(graphs)]
    n = len(g["state"])

    nb = np.full((MAX_NODES, DEGREE), EMPTY, dtype=np.int32)
    deg = np.zeros(MAX_NODES, dtype=np.int32)
    for a, b in g["edgelist"]:
        if a < MAX_NODES and b < MAX_NODES:
            if deg[a] < DEGREE:
                nb[a, deg[a]] = b
                deg[a] += 1
            if deg[b] < DEGREE:
                nb[b, deg[b]] = a
                deg[b] += 1

    # Map binary states from JSON to ternary: 1->SPLIT, 0->NEUTRAL
    raw = np.array(g["state"], dtype=np.int32)
    states = np.full(MAX_NODES, NEUTRAL, dtype=np.int32)
    states[:n] = np.where(raw == 1, SPLIT, NEUTRAL)
    states[n:] = 0

    active = np.zeros(MAX_NODES, dtype=bool)
    active[:n] = True

    return SimState(
        node_states=jnp.array(states),
        node_active=jnp.array(active),
        neighbors=jnp.array(nb),
        num_active=jnp.array(n, dtype=jnp.int32),
        rng=rng,
    )


def make_init_state(rng: jnp.ndarray, graph: str = "petersen") -> SimState:
    """Create an initial state from a built-in or graphs.json seed graph."""
    if graph.startswith("json"):
        idx = int(graph.split(":")[1]) if ":" in graph else 0
        return _make_state_from_json(rng, idx)
    builder = GRAPH_BUILDERS.get(graph, make_petersen_state)
    return builder(rng)


# ── Configuration computation ────────────────────────────────────────────────

def compute_configs(state: SimState) -> jnp.ndarray:
    """Configuration vector: c(v) = 7*s(v) + sum(s(neighbors)).

    With 3 states and 3 neighbors, sum ranges 0..6, so c in [0..20].
    Returns (MAX_NODES,) int32, zeroed for inactive nodes.
    """
    safe_nb = jnp.clip(state.neighbors, 0, MAX_NODES - 1)
    nb_states = state.node_states[safe_nb]              # (N, 3)
    nb_valid = (state.neighbors >= 0)                    # (N, 3)
    nb_sum = (nb_states * nb_valid.astype(jnp.int32)).sum(axis=1)
    configs = 7 * state.node_states + nb_sum
    return configs * state.node_active.astype(jnp.int32)


# ── Division (maintains 3-regularity) ────────────────────────────────────────

def apply_divisions(state: SimState, new_states: jnp.ndarray) \
        -> tuple[SimState, jnp.ndarray]:
    """Divide all nodes with state==SPLIT after rule application.

    Same rewiring as the 2-state model:
    Each division of parent P with neighbors [n0, n1, n2]:
      - Creates children C1, C2
      - P.neighbors = [n0, C1, C2]
      - C1.neighbors = [n1, P, C2]
      - C2.neighbors = [n2, P, C1]
      - n1 replaces P->C1, n2 replaces P->C2
    Children start as NEUTRAL. Graph stays 3-regular.

    Returns (new_state, num_divisions_performed).
    """
    wants_div = state.node_active & (new_states == SPLIT)

    sort_key = jnp.where(wants_div, jnp.arange(MAX_NODES), MAX_NODES)
    div_order = jnp.argsort(sort_key)[:MAX_TOPO_OPS]
    num_wanting = jnp.sum(wants_div)

    free_mask = ~state.node_active
    total_free = jnp.sum(free_mask)
    free_order = jnp.argsort(~free_mask)

    max_divs = jnp.minimum(num_wanting,
                           jnp.minimum(total_free // 2,
                                       jnp.int32(MAX_TOPO_OPS)))

    def do_division(carry, op_idx):
        states, active, nb, num_active, free_ptr = carry
        valid = op_idx < max_divs

        parent = div_order[op_idx]
        valid = valid & active[parent]

        n0 = nb[parent, 0]
        n1 = nb[parent, 1]
        n2 = nb[parent, 2]

        safe_ptr = jnp.clip(free_ptr, 0, MAX_NODES - 2)
        c1 = free_order[safe_ptr]
        c2 = free_order[safe_ptr + 1]

        # Activate children as NEUTRAL
        active = active.at[c1].set(jnp.where(valid, True, active[c1]))
        active = active.at[c2].set(jnp.where(valid, True, active[c2]))
        states = states.at[c1].set(jnp.where(valid, NEUTRAL, states[c1]))
        states = states.at[c2].set(jnp.where(valid, NEUTRAL, states[c2]))

        # Parent becomes NEUTRAL after splitting
        states = states.at[parent].set(jnp.where(valid, NEUTRAL, states[parent]))

        # Rewire
        nb = nb.at[parent].set(jnp.where(
            valid, jnp.stack([n0, c1, c2]), nb[parent]))
        nb = nb.at[c1].set(jnp.where(
            valid, jnp.stack([n1, parent, c2]), nb[c1]))
        nb = nb.at[c2].set(jnp.where(
            valid, jnp.stack([n2, parent, c1]), nb[c2]))

        # In n1's list: replace parent with c1
        safe_n1 = jnp.clip(n1, 0, MAX_NODES - 1)
        row1 = nb[safe_n1]
        row1 = jnp.where(valid & (row1 == parent), c1, row1)
        nb = nb.at[safe_n1].set(jnp.where(valid, row1, nb[safe_n1]))

        # In n2's list: replace parent with c2
        safe_n2 = jnp.clip(n2, 0, MAX_NODES - 1)
        row2 = nb[safe_n2]
        row2 = jnp.where(valid & (row2 == parent), c2, row2)
        nb = nb.at[safe_n2].set(jnp.where(valid, row2, nb[safe_n2]))

        num_active = num_active + jnp.where(valid, 2, 0).astype(jnp.int32)
        free_ptr = free_ptr + jnp.where(valid, 2, 0).astype(jnp.int32)

        return (states, active, nb, num_active, free_ptr), None

    init_carry = (new_states, state.node_active, state.neighbors,
                  state.num_active, jnp.int32(0))
    (out_states, out_active, out_nb, out_num_active, _), _ = jax.lax.scan(
        do_division, init_carry, jnp.arange(MAX_TOPO_OPS))

    actual_divs = jnp.minimum(num_wanting, max_divs)
    new_state = state._replace(
        node_states=out_states, node_active=out_active,
        neighbors=out_nb, num_active=out_num_active)
    return new_state, actual_divs


# ── Collapse / merge (inverse of division, maintains 3-regularity) ───────────

def apply_merges(state: SimState, new_states: jnp.ndarray) \
        -> tuple[SimState, jnp.ndarray]:
    """Merge triangles around nodes with state==MERGE after rule application.

    For a merge node M with neighbors [a, b, c], find a pair (a, b) that
    are connected to each other (forming triangle M-a-b). Then:
      - Let M's third neighbor be ext_m (the one not in the triangle)
      - Let a's external neighbor (not M or b) be ext_a
      - Let b's external neighbor (not M or a) be ext_b
      - Survivor = M, with neighbors [ext_m, ext_a, ext_b]
      - a is deactivated; ext_a replaces a->M in its neighbor list
      - b is deactivated; ext_b replaces b->M in its neighbor list
      - ext_m keeps its reference to M unchanged
    This is the inverse of division and preserves 3-regularity.
    Survivor state becomes NEUTRAL.

    Returns (new_state, num_merges_performed).
    """
    wants_merge = state.node_active & (new_states == MERGE)

    sort_key = jnp.where(wants_merge, jnp.arange(MAX_NODES), MAX_NODES)
    merge_order = jnp.argsort(sort_key)[:MAX_TOPO_OPS]
    num_wanting = jnp.sum(wants_merge)

    # Need at least 4 active nodes to do a merge (removes 2)
    max_merges = jnp.minimum(num_wanting,
                             jnp.minimum((state.num_active - 4) // 2,
                                         jnp.int32(MAX_TOPO_OPS)))
    max_merges = jnp.maximum(max_merges, 0)

    def do_merge(carry, op_idx):
        states, active, nb, num_active = carry
        valid = op_idx < max_merges

        m_idx = merge_order[op_idx]
        valid = valid & active[m_idx]

        # M's 3 neighbors
        na = nb[m_idx, 0]
        n_b = nb[m_idx, 1]
        nc = nb[m_idx, 2]

        safe_na = jnp.clip(na, 0, MAX_NODES - 1)
        safe_nb_ = jnp.clip(n_b, 0, MAX_NODES - 1)
        safe_nc = jnp.clip(nc, 0, MAX_NODES - 1)

        # All three neighbors must still be active (previous merges
        # may have deactivated them)
        valid = valid & (na >= 0) & (n_b >= 0) & (nc >= 0)
        valid = valid & active[safe_na] & active[safe_nb_] & active[safe_nc]

        # Check which pair forms a triangle with M
        # a-b connected?
        ab_connected = jnp.any(nb[safe_na] == n_b)
        # a-c connected?
        ac_connected = jnp.any(nb[safe_na] == nc)
        # b-c connected?
        bc_connected = jnp.any(nb[safe_nb_] == nc)

        # Pick first valid triangle (priority: a-b, a-c, b-c)
        # Triangle nodes: (doomed1, doomed2, ext_of_m)
        use_ab = ab_connected
        use_ac = ac_connected & ~use_ab
        use_bc = bc_connected & ~use_ab & ~use_ac
        found_tri = use_ab | use_ac | use_bc

        doomed1 = jnp.where(use_ab, na,
                  jnp.where(use_ac, na, n_b))
        doomed2 = jnp.where(use_ab, n_b,
                  jnp.where(use_ac, nc, nc))
        ext_m = jnp.where(use_ab, nc,
                jnp.where(use_ac, n_b, na))

        valid = valid & found_tri

        safe_d1 = jnp.clip(doomed1, 0, MAX_NODES - 1)
        safe_d2 = jnp.clip(doomed2, 0, MAX_NODES - 1)

        # Find external neighbor of doomed1 (not M, not doomed2)
        d1_nbs = nb[safe_d1]
        d1_ext_mask = (d1_nbs != m_idx) & (d1_nbs != doomed2) & (d1_nbs >= 0)
        # Take first external neighbor
        ext_d1 = jnp.where(d1_ext_mask[0], d1_nbs[0],
                 jnp.where(d1_ext_mask[1], d1_nbs[1],
                 jnp.where(d1_ext_mask[2], d1_nbs[2], EMPTY)))

        # Find external neighbor of doomed2 (not M, not doomed1)
        d2_nbs = nb[safe_d2]
        d2_ext_mask = (d2_nbs != m_idx) & (d2_nbs != doomed1) & (d2_nbs >= 0)
        ext_d2 = jnp.where(d2_ext_mask[0], d2_nbs[0],
                 jnp.where(d2_ext_mask[1], d2_nbs[1],
                 jnp.where(d2_ext_mask[2], d2_nbs[2], EMPTY)))

        # Valid only if we found external neighbors for both
        valid = valid & (ext_d1 >= 0) & (ext_d2 >= 0)
        # External neighbors must be active too
        safe_ext_d1 = jnp.clip(ext_d1, 0, MAX_NODES - 1)
        safe_ext_d2 = jnp.clip(ext_d2, 0, MAX_NODES - 1)
        safe_ext_m = jnp.clip(ext_m, 0, MAX_NODES - 1)
        valid = valid & active[safe_ext_d1] & active[safe_ext_d2]
        valid = valid & active[safe_ext_m]
        # No multi-edges or self-loops after rewiring
        valid = valid & (ext_d1 != ext_d2)
        valid = valid & (ext_d1 != ext_m) & (ext_d2 != ext_m)
        valid = valid & (ext_d1 != m_idx) & (ext_d2 != m_idx)

        # Rewire M: neighbors become [ext_m, ext_d1, ext_d2]
        nb = nb.at[m_idx].set(jnp.where(
            valid, jnp.stack([ext_m, ext_d1, ext_d2]), nb[m_idx]))

        # In ext_d1's list: replace doomed1 with M
        row_ed1 = nb[safe_ext_d1]
        row_ed1 = jnp.where(valid & (row_ed1 == doomed1), m_idx, row_ed1)
        nb = nb.at[safe_ext_d1].set(jnp.where(valid, row_ed1, nb[safe_ext_d1]))

        # In ext_d2's list: replace doomed2 with M
        row_ed2 = nb[safe_ext_d2]
        row_ed2 = jnp.where(valid & (row_ed2 == doomed2), m_idx, row_ed2)
        nb = nb.at[safe_ext_d2].set(jnp.where(valid, row_ed2, nb[safe_ext_d2]))

        # Deactivate doomed nodes
        active = active.at[safe_d1].set(jnp.where(valid, False, active[safe_d1]))
        active = active.at[safe_d2].set(jnp.where(valid, False, active[safe_d2]))
        states = states.at[safe_d1].set(jnp.where(valid, 0, states[safe_d1]))
        states = states.at[safe_d2].set(jnp.where(valid, 0, states[safe_d2]))

        # Clear doomed neighbor lists
        empty_row = jnp.full(DEGREE, EMPTY, dtype=jnp.int32)
        nb = nb.at[safe_d1].set(jnp.where(valid, empty_row, nb[safe_d1]))
        nb = nb.at[safe_d2].set(jnp.where(valid, empty_row, nb[safe_d2]))

        # Survivor becomes NEUTRAL
        states = states.at[m_idx].set(jnp.where(valid, NEUTRAL, states[m_idx]))

        num_active = num_active - jnp.where(valid, 2, 0).astype(jnp.int32)

        return (states, active, nb, num_active), None

    init_carry = (new_states, state.node_active, state.neighbors,
                  state.num_active)
    (out_states, out_active, out_nb, out_num_active), _ = jax.lax.scan(
        do_merge, init_carry, jnp.arange(MAX_TOPO_OPS))

    actual_merges = jnp.minimum(num_wanting, max_merges)
    new_state = state._replace(
        node_states=out_states, node_active=out_active,
        neighbors=out_nb, num_active=out_num_active)
    return new_state, actual_merges


# ── One GRA step ─────────────────────────────────────────────────────────────

def gra_step(state: SimState, rule: Rule) -> tuple[SimState, StepMetrics]:
    """One complete GRA step: compute configs, apply rule, then topology.

    The rule determines new states from the current configuration.
    Topology operations (split/merge) are triggered by the new states.
    Divisions are applied first, then merges on the remaining graph.
    """
    prev_states = state.node_states
    configs = compute_configs(state)

    # Apply rule simultaneously
    new_states = rule.table[configs] * state.node_active.astype(jnp.int32)

    # Count state changes
    changed = (new_states != prev_states) & state.node_active
    num_changes = jnp.sum(changed)

    # Apply divisions first (split nodes), then merges
    state, num_divs = apply_divisions(state, new_states)

    # Re-read states after division (division sets some to NEUTRAL)
    # Merge operates on the post-division graph
    state, num_merges = apply_merges(state, state.node_states)

    # Count states
    active_mask = state.node_active.astype(jnp.int32)
    num_per_state = jnp.array([
        jnp.sum((state.node_states == s) * active_mask)
        for s in range(NUM_STATES)
    ])
    n_active = jnp.maximum(state.num_active.astype(jnp.float32), 1.0)
    state_fracs = num_per_state.astype(jnp.float32) / n_active

    metrics = StepMetrics(
        num_active=state.num_active,
        num_per_state=num_per_state,
        state_fractions=state_fracs,
        num_state_changes=num_changes,
        num_divisions=num_divs,
        num_merges=num_merges,
    )
    return state, metrics


# ── Simulation runner ────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(2,))
def run_simulation(rule: Rule, init_state: SimState,
                   num_steps: int = 200) -> tuple[SimState, StepMetrics]:
    """Run a 3-state discrete GRA for num_steps, collecting per-step metrics."""
    def step_fn(state, _):
        return gra_step(state, rule)

    final_state, all_metrics = jax.lax.scan(
        step_fn, init_state, jnp.arange(num_steps))
    return final_state, all_metrics


# ── Graph structure features ─────────────────────────────────────────────────

class GraphFeatures(NamedTuple):
    """Structural features of the graph."""
    clustering: jnp.ndarray          # () mean clustering coefficient
    clustering_expected: jnp.ndarray # () random baseline: k/(n-1) = 3/(n-1)
    state_entropy: jnp.ndarray       # () Shannon entropy of state distribution
    boundary_fraction: jnp.ndarray   # () fraction of edges joining different states


def compute_graph_features(state: SimState) -> GraphFeatures:
    """Compute structural features of the current graph."""
    n = jnp.maximum(state.num_active.astype(jnp.float32), 1.0)
    safe_nb = jnp.clip(state.neighbors, 0, MAX_NODES - 1)

    # ── Clustering (triangles among each node's 3 neighbors) ─────────
    nb_of_n0 = safe_nb[safe_nb[:, 0]]
    nb_of_n1 = safe_nb[safe_nb[:, 1]]
    e01 = jnp.any(nb_of_n0 == safe_nb[:, 1:2], axis=1)
    e02 = jnp.any(nb_of_n0 == safe_nb[:, 2:3], axis=1)
    e12 = jnp.any(nb_of_n1 == safe_nb[:, 2:3], axis=1)

    triangles = (e01.astype(jnp.float32)
                 + e02.astype(jnp.float32)
                 + e12.astype(jnp.float32))
    cc_per_node = (triangles / 3.0) * state.node_active.astype(jnp.float32)
    clustering = cc_per_node.sum() / n
    clustering_expected = 3.0 / jnp.maximum(n - 1.0, 1.0)

    # ── State entropy ─────────────────────────────────────────────────
    active_mask = state.node_active.astype(jnp.float32)
    counts = jnp.array([
        jnp.sum((state.node_states == s).astype(jnp.float32) * active_mask)
        for s in range(NUM_STATES)
    ])
    probs = counts / jnp.maximum(n, 1.0)
    # Shannon entropy (log base 2), with epsilon for stability
    entropy = -jnp.sum(jnp.where(probs > 1e-8,
                                  probs * jnp.log2(jnp.maximum(probs, 1e-8)),
                                  0.0))

    # ── Boundary fraction (edges connecting different states) ─────────
    s = state.node_states.astype(jnp.float32)
    nb_s = s[safe_nb]
    diff = (nb_s != s[:, None]).astype(jnp.float32) \
        * state.node_active[:, None].astype(jnp.float32)
    total_edges = jnp.maximum(
        state.node_active.astype(jnp.float32).sum() * DEGREE, 1.0)
    boundary = diff.sum() / total_edges

    return GraphFeatures(
        clustering=clustering,
        clustering_expected=clustering_expected,
        state_entropy=entropy,
        boundary_fraction=boundary,
    )


def structure_novelty(state: SimState) -> jnp.ndarray:
    """Scalar: how far is this graph's structure from a random 3-regular?"""
    f = compute_graph_features(state)
    c_excess = jnp.where(
        f.clustering_expected > 1e-6,
        f.clustering / f.clustering_expected - 1.0,
        f.clustering * 100.0)
    return (
        jnp.abs(c_excess) * 2.0
        + f.state_entropy * 1.0
        + f.boundary_fraction * 1.5
    )
