"""
Discrete Graph-Rewriting Automata (GRA) — JAX simulation core.

Implements the original discrete CA model from Cousin & Maignan (2022):
- 3-regular graphs with binary node states (alive=1, dead=0)
- 8 possible configurations: c(v) = 4*s(v) + sum(s(neighbors))
- Rule R maps config → new state, Rule R' maps config → divide
- 65,536 possible rules encoded as 16-bit integers
- Division maintains 3-regularity by redistributing neighbors
"""

from functools import partial
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

# ── Constants (module-level for JIT visibility) ──────────────────────────────

MAX_NODES = 2048
DEGREE = 3
MAX_TOPO_OPS = 128
EMPTY = -1


def configure(cfg):
    """Set module constants from config. Must call before JIT."""
    global MAX_NODES, MAX_TOPO_OPS
    sim = cfg if not hasattr(cfg, "sim") else cfg.sim
    MAX_NODES = int(sim.max_nodes)
    MAX_TOPO_OPS = int(sim.max_topo_ops)


# ── Data structures ──────────────────────────────────────────────────────────

class SimState(NamedTuple):
    """Fixed-size simulation state for a 3-regular graph."""
    node_states: jnp.ndarray   # (MAX_NODES,) int32 {0, 1}
    node_active: jnp.ndarray   # (MAX_NODES,) bool
    neighbors: jnp.ndarray     # (MAX_NODES, 3) int32, EMPTY=-1 for inactive
    num_active: jnp.ndarray    # () scalar int32
    rng: jnp.ndarray           # PRNGKey


class Rule(NamedTuple):
    """A GRA rule: two lookup tables mapping configuration → action."""
    state_rule: jnp.ndarray    # (8,) int32 — R[c] → new state
    div_rule: jnp.ndarray      # (8,) int32 — R'[c] → divide?


class StepMetrics(NamedTuple):
    """Per-step observables for fitness evaluation."""
    num_active: jnp.ndarray         # () total graph nodes
    num_alive: jnp.ndarray          # () nodes with state=1
    alive_fraction: jnp.ndarray     # ()
    num_state_changes: jnp.ndarray  # () nodes that flipped state
    num_divisions: jnp.ndarray      # () divisions performed


# ── Rule encoding / decoding ─────────────────────────────────────────────────

def rule_from_number(n: jnp.ndarray) -> Rule:
    """Decode a 16-bit rule number into lookup tables.

    Bits 0-7  → R  (state update for configs 0..7)
    Bits 8-15 → R' (division signal for configs 0..7)
    """
    bits = (n >> jnp.arange(16, dtype=jnp.int32)) & 1
    return Rule(state_rule=bits[:8], div_rule=bits[8:])


def rule_to_number(rule: Rule) -> jnp.ndarray:
    """Encode a Rule back to its 16-bit number."""
    powers = jnp.arange(16, dtype=jnp.int32)
    bits = jnp.concatenate([rule.state_rule, rule.div_rule])
    return (bits * (1 << powers)).sum()


def sample_rule(rng: jnp.ndarray) -> Rule:
    """Sample a uniformly random rule."""
    k1, k2 = jax.random.split(rng)
    return Rule(
        state_rule=jax.random.randint(k1, (8,), 0, 2),
        div_rule=jax.random.randint(k2, (8,), 0, 2),
    )


def mutate_rule(rule: Rule, rng: jnp.ndarray, flip_prob: float = 0.1) -> Rule:
    """Mutate a rule by independently flipping each bit with probability p."""
    k1, k2 = jax.random.split(rng)
    flip_s = jax.random.bernoulli(k1, p=flip_prob, shape=(8,))
    flip_d = jax.random.bernoulli(k2, p=flip_prob, shape=(8,))
    return Rule(
        state_rule=jnp.where(flip_s, 1 - rule.state_rule, rule.state_rule),
        div_rule=jnp.where(flip_d, 1 - rule.div_rule, rule.div_rule),
    )


def rules_from_numbers(numbers: jnp.ndarray) -> Rule:
    """Batch-decode an array of rule numbers to a batched Rule pytree."""
    bits = (numbers[:, None] >> jnp.arange(16, dtype=jnp.int32)[None, :]) & 1
    return Rule(state_rule=bits[:, :8], div_rule=bits[:, 8:])


# ── Initial graphs ───────────────────────────────────────────────────────────

def make_k4_state(rng: jnp.ndarray) -> SimState:
    """K4: complete graph on 4 nodes (simplest 3-regular graph)."""
    n = 4
    nb = np.full((MAX_NODES, DEGREE), EMPTY, dtype=np.int32)
    nb[0] = [1, 2, 3]
    nb[1] = [0, 2, 3]
    nb[2] = [0, 1, 3]
    nb[3] = [0, 1, 2]

    states = np.zeros(MAX_NODES, dtype=np.int32)
    states[0] = 1

    active = np.zeros(MAX_NODES, dtype=bool)
    active[:n] = True

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
    # Outer cycle: 0-1-2-3-4-0
    # Inner star:  5-7-9-6-8-5
    # Spokes:      i ↔ i+5
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

    # Initial state covering configs {1,2,3,4,5,6}
    states = np.zeros(MAX_NODES, dtype=np.int32)
    states[:10] = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

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
    # Two triangles (0-1-2) and (3-4-5) with connecting edges
    nb[0] = [1, 2, 3]
    nb[1] = [0, 2, 4]
    nb[2] = [0, 1, 5]
    nb[3] = [0, 4, 5]
    nb[4] = [1, 3, 5]
    nb[5] = [2, 3, 4]

    states = np.zeros(MAX_NODES, dtype=np.int32)
    states[:6] = [1, 0, 1, 0, 1, 0]

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


def make_init_state(rng: jnp.ndarray, graph: str = "petersen") -> SimState:
    """Create an initial state from one of the built-in seed graphs."""
    builder = GRAPH_BUILDERS.get(graph, make_petersen_state)
    return builder(rng)


# ── Configuration computation ────────────────────────────────────────────────

def compute_configs(state: SimState) -> jnp.ndarray:
    """Configuration vector: c(v) = 4*s(v) + sum(s(neighbors)).

    Returns (MAX_NODES,) int32 in [0..7], zeroed for inactive nodes.
    """
    safe_nb = jnp.clip(state.neighbors, 0, MAX_NODES - 1)
    nb_states = state.node_states[safe_nb]            # (N, 3)
    nb_valid = (state.neighbors >= 0)                  # (N, 3)
    nb_sum = (nb_states * nb_valid.astype(jnp.int32)).sum(axis=1)
    configs = 4 * state.node_states + nb_sum
    return configs * state.node_active.astype(jnp.int32)


# ── Division (maintains 3-regularity) ────────────────────────────────────────

def apply_divisions(state: SimState, div_signals: jnp.ndarray) \
        -> tuple[SimState, jnp.ndarray]:
    """Divide all signalled nodes, processing lowest index first.

    Each division of parent P with neighbors [n0, n1, n2]:
      - Creates children C1, C2
      - P keeps n0, gains C1, C2  → P.neighbors = [n0, C1, C2]
      - C1 gets n1, P, C2        → C1.neighbors = [n1, P, C2]
      - C2 gets n2, P, C1        → C2.neighbors = [n2, P, C1]
      - n1 replaces P→C1, n2 replaces P→C2 in their lists
    All new nodes inherit the parent's state. Graph stays 3-regular.

    Returns (new_state, num_divisions_performed).
    """
    wants_div = state.node_active & (div_signals == 1)

    # Sort dividing nodes by index (lowest first, matching paper)
    sort_key = jnp.where(wants_div, jnp.arange(MAX_NODES), MAX_NODES)
    div_order = jnp.argsort(sort_key)[:MAX_TOPO_OPS]
    num_wanting = jnp.sum(wants_div)

    # Identify free (inactive) slots
    free_mask = ~state.node_active
    total_free = jnp.sum(free_mask)
    free_order = jnp.argsort(~free_mask)  # inactive indices first

    max_divs = jnp.minimum(num_wanting,
                           jnp.minimum(total_free // 2,
                                       jnp.int32(MAX_TOPO_OPS)))

    def do_division(carry, op_idx):
        states, active, nb, num_active, free_ptr = carry
        valid = op_idx < max_divs

        parent = div_order[op_idx]
        valid = valid & active[parent]

        # Parent's 3 current neighbors
        n0 = nb[parent, 0]
        n1 = nb[parent, 1]
        n2 = nb[parent, 2]

        # Allocate two free slots
        safe_ptr = jnp.clip(free_ptr, 0, MAX_NODES - 2)
        c1 = free_order[safe_ptr]
        c2 = free_order[safe_ptr + 1]

        # Activate children with parent's state
        active = active.at[c1].set(jnp.where(valid, True, active[c1]))
        active = active.at[c2].set(jnp.where(valid, True, active[c2]))
        states = states.at[c1].set(jnp.where(valid, states[parent], states[c1]))
        states = states.at[c2].set(jnp.where(valid, states[parent], states[c2]))

        # Rewire parent → [n0, c1, c2]
        nb = nb.at[parent].set(jnp.where(
            valid, jnp.stack([n0, c1, c2]), nb[parent]))

        # C1 → [n1, parent, c2]
        nb = nb.at[c1].set(jnp.where(
            valid, jnp.stack([n1, parent, c2]), nb[c1]))

        # C2 → [n2, parent, c1]
        nb = nb.at[c2].set(jnp.where(
            valid, jnp.stack([n2, parent, c1]), nb[c2]))

        # In n1's neighbor list: replace parent with c1
        safe_n1 = jnp.clip(n1, 0, MAX_NODES - 1)
        row1 = nb[safe_n1]
        row1 = jnp.where(valid & (row1 == parent), c1, row1)
        nb = nb.at[safe_n1].set(jnp.where(valid, row1, nb[safe_n1]))

        # In n2's neighbor list: replace parent with c2
        safe_n2 = jnp.clip(n2, 0, MAX_NODES - 1)
        row2 = nb[safe_n2]
        row2 = jnp.where(valid & (row2 == parent), c2, row2)
        nb = nb.at[safe_n2].set(jnp.where(valid, row2, nb[safe_n2]))

        num_active = num_active + jnp.where(valid, 2, 0).astype(jnp.int32)
        free_ptr = free_ptr + jnp.where(valid, 2, 0).astype(jnp.int32)

        return (states, active, nb, num_active, free_ptr), None

    init_carry = (state.node_states, state.node_active, state.neighbors,
                  state.num_active, jnp.int32(0))
    (new_states, new_active, new_nb, new_num_active, _), _ = jax.lax.scan(
        do_division, init_carry, jnp.arange(MAX_TOPO_OPS))

    actual_divs = jnp.minimum(num_wanting, max_divs)
    new_state = state._replace(
        node_states=new_states, node_active=new_active,
        neighbors=new_nb, num_active=new_num_active)
    return new_state, actual_divs


# ── One GRA step ─────────────────────────────────────────────────────────────

def gra_step(state: SimState, rule: Rule) -> tuple[SimState, StepMetrics]:
    """One complete GRA step: update states, then divide.

    Both the new-state and division decisions are computed from the
    SAME configuration vector (pre-update state), matching the paper.
    """
    prev_states = state.node_states
    configs = compute_configs(state)

    # Apply rule simultaneously
    new_states = rule.state_rule[configs] * state.node_active.astype(jnp.int32)
    div_signals = rule.div_rule[configs] * state.node_active.astype(jnp.int32)

    # Count state changes before overwriting
    changed = (new_states != prev_states) & state.node_active
    num_changes = jnp.sum(changed)

    state = state._replace(node_states=new_states)

    # Divisions
    state, num_divs = apply_divisions(state, div_signals)

    # Metrics
    num_alive = jnp.sum(
        state.node_states * state.node_active.astype(jnp.int32))
    n_active = jnp.maximum(state.num_active.astype(jnp.float32), 1.0)
    alive_frac = num_alive.astype(jnp.float32) / n_active

    metrics = StepMetrics(
        num_active=state.num_active,
        num_alive=num_alive,
        alive_fraction=alive_frac,
        num_state_changes=num_changes,
        num_divisions=num_divs,
    )
    return state, metrics


# ── Simulation runner ────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(2,))
def run_simulation(rule: Rule, init_state: SimState,
                   num_steps: int = 200) -> tuple[SimState, StepMetrics]:
    """Run a discrete GRA for num_steps, collecting per-step metrics."""
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
    state_assortativity: jnp.ndarray # () state correlation at edge endpoints
    boundary_fraction: jnp.ndarray   # () fraction of edges joining 0↔1


def compute_graph_features(state: SimState) -> GraphFeatures:
    """Compute structural features of the current graph."""
    n = jnp.maximum(state.num_active.astype(jnp.float32), 1.0)
    safe_nb = jnp.clip(state.neighbors, 0, MAX_NODES - 1)

    # ── Clustering (triangles among each node's 3 neighbors) ─────────
    nb_of_n0 = safe_nb[safe_nb[:, 0]]  # (N, 3) neighbors of each node's 1st nb
    nb_of_n1 = safe_nb[safe_nb[:, 1]]  # neighbors of 2nd nb
    # Edge between n0-n1?
    e01 = jnp.any(nb_of_n0 == safe_nb[:, 1:2], axis=1)
    # Edge between n0-n2?
    e02 = jnp.any(nb_of_n0 == safe_nb[:, 2:3], axis=1)
    # Edge between n1-n2?
    e12 = jnp.any(nb_of_n1 == safe_nb[:, 2:3], axis=1)

    triangles = (e01.astype(jnp.float32)
                 + e02.astype(jnp.float32)
                 + e12.astype(jnp.float32))
    cc_per_node = (triangles / 3.0) * state.node_active.astype(jnp.float32)
    clustering = cc_per_node.sum() / n
    clustering_expected = 3.0 / jnp.maximum(n - 1.0, 1.0)

    # ── State assortativity (Pearson corr of states at edge endpoints) ─
    s = state.node_states.astype(jnp.float32)
    nb_s = s[safe_nb]                                  # (N, 3)
    active = state.node_active.astype(jnp.float32)
    valid = active[:, None] * jnp.ones(DEGREE)[None, :]  # (N, 3)

    s_i = jnp.broadcast_to(s[:, None], (MAX_NODES, DEGREE))
    s_j = nb_s
    total = jnp.maximum(valid.sum(), 1.0)
    mean_i = (s_i * valid).sum() / total
    mean_j = (s_j * valid).sum() / total
    cov = ((s_i - mean_i) * (s_j - mean_j) * valid).sum() / total
    std_i = jnp.sqrt(jnp.maximum(((s_i - mean_i) ** 2 * valid).sum() / total, 1e-8))
    std_j = jnp.sqrt(jnp.maximum(((s_j - mean_j) ** 2 * valid).sum() / total, 1e-8))
    state_assort = cov / jnp.maximum(std_i * std_j, 1e-8)

    # ── Boundary fraction (edges connecting alive ↔ dead) ─────────────
    diff = jnp.abs(s_i - s_j) * valid
    total_half = jnp.maximum(valid.sum(), 1.0)
    boundary = diff.sum() / total_half

    return GraphFeatures(
        clustering=clustering,
        clustering_expected=clustering_expected,
        state_assortativity=state_assort,
        boundary_fraction=boundary,
    )


def structure_novelty(state: SimState) -> jnp.ndarray:
    """Scalar: how far is this graph's structure from a random 3-regular?

    Combines clustering excess (above random baseline) and state mixing.
    """
    f = compute_graph_features(state)
    c_excess = jnp.where(
        f.clustering_expected > 1e-6,
        f.clustering / f.clustering_expected - 1.0,
        f.clustering * 100.0)
    return (
        jnp.abs(c_excess) * 2.0
        + jnp.abs(f.state_assortativity) * 1.5
        + f.boundary_fraction * 1.0
    )
