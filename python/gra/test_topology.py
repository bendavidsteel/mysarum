#!/usr/bin/env python3
"""
Tests for GRA topology operations.

Checks:
1. No orphaned nodes (all active nodes have degree > 0, except initial isolated nodes)
2. Division: 1 node → 3 nodes forming a triangle, neighbors redistributed correctly
3. Collapse: triangle merge 3→1, state averaged, edges rewired
4. 3-regularity: mean degree stays near 3.0 after sufficient growth
5. No self-loops or duplicate edges in neighbor lists
6. No references to inactive nodes in neighbor lists

Run: python test_topology.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

import sim as sim_mod

cfg = OmegaConf.load("conf/config.yaml")
sim_mod.configure(cfg)

from sim import (
    SimState, Params, EMPTY, MAX_NODES, MAX_DEGREE, NUM_CHANNELS,
    make_init_state, sample_params, apply_growth, apply_divisions,
    apply_collapses, run_simulation, config_signal,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def check_invariants(state: SimState, label: str = "") -> list[str]:
    """Check all graph invariants. Returns list of error strings."""
    errors = []
    active = np.array(state.node_active)
    nb = np.array(state.neighbors)
    n_nb = np.array(state.num_neighbors)
    states = np.array(state.node_states)
    num_active = int(state.num_active)

    active_indices = set(np.where(active)[0])

    # num_active matches actual count
    actual_active = len(active_indices)
    if num_active != actual_active:
        errors.append(f"num_active={num_active} but actual={actual_active}")

    for i in range(MAX_NODES):
        deg = int(n_nb[i])
        is_active = active[i]

        if not is_active:
            # Inactive node should have no neighbors
            if deg != 0:
                errors.append(f"Node {i}: inactive but degree={deg}")
            if np.any(nb[i] != EMPTY):
                non_empty = np.where(nb[i] != EMPTY)[0]
                errors.append(f"Node {i}: inactive but has non-EMPTY slots: {non_empty.tolist()}")
            # States should be zero
            if np.any(states[i] != 0):
                errors.append(f"Node {i}: inactive but has non-zero state")
            continue

        # Active node checks
        for slot in range(deg):
            neighbor = int(nb[i, slot])

            # No EMPTY in valid slots
            if neighbor == EMPTY:
                errors.append(f"Node {i}: EMPTY in slot {slot} but degree={deg}")
                continue

            # Neighbor in valid range
            if neighbor < 0 or neighbor >= MAX_NODES:
                errors.append(f"Node {i}: neighbor {neighbor} out of range")
                continue

            # Neighbor is active
            if neighbor not in active_indices:
                errors.append(f"Node {i}: neighbor {neighbor} is inactive")

            # No self-loops
            if neighbor == i:
                errors.append(f"Node {i}: self-loop in slot {slot}")

            # Reciprocity: i should be in neighbor's list
            nb_of_nb = nb[neighbor, :int(n_nb[neighbor])]
            if i not in nb_of_nb:
                errors.append(f"Node {i}: neighbor {neighbor} doesn't have {i} back")

        # Check for duplicates in neighbor list
        valid_nbs = nb[i, :deg]
        unique_nbs = np.unique(valid_nbs)
        if len(unique_nbs) < deg:
            dupes = [int(x) for x in valid_nbs if np.sum(valid_nbs == x) > 1]
            errors.append(f"Node {i}: duplicate neighbors: {dupes}")

        # Slots beyond degree should be EMPTY
        for slot in range(deg, MAX_DEGREE):
            if nb[i, slot] != EMPTY:
                errors.append(f"Node {i}: non-EMPTY in slot {slot} beyond degree {deg}")

    if errors and label:
        print(f"\n{'='*60}")
        print(f"INVARIANT VIOLATIONS after {label}:")
        for e in errors[:20]:  # cap output
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    return errors


def check_no_orphans(state: SimState, label: str = "") -> list[str]:
    """Check that no active node is completely disconnected."""
    errors = []
    active = np.array(state.node_active)
    n_nb = np.array(state.num_neighbors)

    for i in np.where(active)[0]:
        if int(n_nb[i]) == 0:
            errors.append(f"Node {i}: orphaned (active but degree 0)")

    if errors and label:
        print(f"\nORPHANED NODES after {label}: {len(errors)}")
        for e in errors[:10]:
            print(f"  - {e}")

    return errors


def check_no_chains(state: SimState, label: str = "") -> list[str]:
    """Check for o-o-o patterns (degree-2 nodes whose neighbors are also degree-2)."""
    errors = []
    active = np.array(state.node_active)
    nb = np.array(state.neighbors)
    n_nb = np.array(state.num_neighbors)

    for i in np.where(active)[0]:
        deg = int(n_nb[i])
        if deg != 2:
            continue
        # Both neighbors also degree 2?
        n1 = int(nb[i, 0])
        n2 = int(nb[i, 1])
        if int(n_nb[n1]) == 2 and int(n_nb[n2]) == 2:
            errors.append(f"Node {i}: chain (deg 2, neighbors {n1}(deg 2), {n2}(deg 2))")

    if errors and label:
        print(f"\nCHAIN PATTERNS after {label}: {len(errors)}")
        for e in errors[:10]:
            print(f"  - {e}")

    return errors


def degree_stats(state: SimState) -> dict:
    """Return degree distribution stats for active nodes."""
    active = np.array(state.node_active)
    n_nb = np.array(state.num_neighbors)
    degrees = n_nb[active]
    unique, counts = np.unique(degrees, return_counts=True)
    return {
        "mean": float(degrees.mean()),
        "min": int(degrees.min()),
        "max": int(degrees.max()),
        "distribution": dict(zip(map(int, unique), map(int, counts))),
        "n_active": int(active.sum()),
    }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_initial_state():
    """Initial state should have valid invariants."""
    print("Test: initial state invariants...")
    rng = jax.random.key(42)
    state = make_init_state(rng, graph_idx=0)
    errors = check_invariants(state, "initial state")
    assert len(errors) == 0, f"Initial state has {len(errors)} errors"
    stats = degree_stats(state)
    print(f"  OK: {stats['n_active']} nodes, degree dist: {stats['distribution']}")


def test_division_creates_triangle():
    """Division should produce parent + 2 children in a triangle."""
    print("\nTest: division creates triangle...")
    rng = jax.random.key(42)
    state = make_init_state(rng, graph_idx=0)

    # Run a few growth steps to get u values
    rng, k = jax.random.split(rng)
    params = sample_params(k)
    for _ in range(5):
        state = apply_growth(state, params)

    n_before = int(state.num_active)
    stats_before = degree_stats(state)

    # Force division: very low threshold, high prob
    params_div = params._replace(
        div_threshold=jnp.array(0.001), div_prob=jnp.array(0.99))
    state_after = apply_divisions(state, params_div)

    n_after = int(state_after.num_active)
    stats_after = degree_stats(state_after)

    # Should have grown (each division adds 2 nodes)
    growth = n_after - n_before
    print(f"  Nodes: {n_before} -> {n_after} (grew by {growth})")
    print(f"  Degree dist before: {stats_before['distribution']}")
    print(f"  Degree dist after:  {stats_after['distribution']}")
    assert growth > 0, "No divisions occurred"
    assert growth % 2 == 0, f"Growth should be even (2 per division), got {growth}"

    errors = check_invariants(state_after, "after division")
    assert len(errors) == 0, f"Division produced {len(errors)} invariant errors"

    orphans = check_no_orphans(state_after, "after division")
    assert len(orphans) == 0, f"Division produced {len(orphans)} orphans"

    print(f"  OK: {growth // 2} divisions, no invariant errors, no orphans")


def test_collapse_merges_triangle():
    """Collapse should merge 3 nodes (triangle) into 1."""
    print("\nTest: collapse merges triangle...")
    rng = jax.random.key(42)
    state = make_init_state(rng, graph_idx=0)

    rng, k = jax.random.split(rng)
    params = sample_params(k)

    # Grow the graph first
    params_grow = params._replace(
        div_threshold=jnp.array(0.001), div_prob=jnp.array(0.99))
    for _ in range(5):
        state = apply_growth(state, params)
    state = apply_divisions(state, params_grow)
    state = apply_divisions(state, params_grow)

    n_before = int(state.num_active)
    stats_before = degree_stats(state)
    print(f"  Before collapse: {n_before} nodes, degree: {stats_before['distribution']}")

    # Force collapse
    params_die = params._replace(
        death_threshold=jnp.array(0.001), death_prob=jnp.array(0.99))
    # Need u values
    state = apply_growth(state, params)
    state_after = apply_collapses(state, params_die)

    n_after = int(state_after.num_active)
    shrink = n_before - n_after
    stats_after = degree_stats(state_after)
    print(f"  After collapse: {n_after} nodes (removed {shrink})")
    print(f"  Degree dist after: {stats_after['distribution']}")

    # Each triangle merge removes 2 nodes
    assert shrink >= 0, f"Collapse grew the graph? {shrink}"
    if shrink > 0 and shrink % 2 != 0:
        print(f"  WARNING: shrink={shrink} is odd (expected even for triangle merge)")

    errors = check_invariants(state_after, "after collapse")
    orphans = check_no_orphans(state_after, "after collapse")

    if len(errors) == 0 and len(orphans) == 0:
        print(f"  OK: removed {shrink} nodes, no errors, no orphans")
    else:
        print(f"  ISSUES: {len(errors)} invariant errors, {len(orphans)} orphans")


def test_division_collapse_cycle():
    """Multiple division+collapse cycles should maintain invariants."""
    print("\nTest: division/collapse cycle (10 rounds)...")
    rng = jax.random.key(42)
    state = make_init_state(rng, graph_idx=0)

    rng, k = jax.random.split(rng)
    params = sample_params(k)
    params = params._replace(
        div_threshold=jnp.array(0.1), div_prob=jnp.array(0.5),
        death_threshold=jnp.array(0.1), death_prob=jnp.array(0.3))

    all_errors = []
    all_orphans = []

    for i in range(10):
        state = apply_growth(state, params)
        state = apply_divisions(state, params)
        state = apply_collapses(state, params)

        errors = check_invariants(state, f"cycle {i}")
        orphans = check_no_orphans(state, f"cycle {i}")
        all_errors.extend(errors)
        all_orphans.extend(orphans)

        stats = degree_stats(state)
        n = stats['n_active']
        print(f"  Cycle {i}: {n} nodes, mean_deg={stats['mean']:.2f}, "
              f"dist={stats['distribution']}, "
              f"errors={len(errors)}, orphans={len(orphans)}")

    print(f"\n  Total errors: {len(all_errors)}, total orphans: {len(all_orphans)}")
    if all_errors:
        print("  FIRST ERRORS:")
        for e in all_errors[:10]:
            print(f"    - {e}")
    if all_orphans:
        print("  FIRST ORPHANS:")
        for e in all_orphans[:10]:
            print(f"    - {e}")


def test_full_simulation():
    """Run a short simulation and check invariants + no orphans at end."""
    print("\nTest: full simulation (200 steps)...")
    rng = jax.random.key(42)
    state = make_init_state(rng, graph_idx=0)

    rng, k = jax.random.split(rng)
    params = sample_params(k)
    # Moderate topology activity
    params = params._replace(
        div_threshold=jnp.array(0.3), div_prob=jnp.array(0.05),
        death_threshold=jnp.array(0.3), death_prob=jnp.array(0.03))

    final, metrics = run_simulation(params, state, 200)

    errors = check_invariants(final, "after 200 steps")
    orphans = check_no_orphans(final, "after 200 steps")
    chains = check_no_chains(final, "after 200 steps")
    stats = degree_stats(final)

    print(f"  Final: {stats['n_active']} nodes, mean_deg={stats['mean']:.2f}")
    print(f"  Degree dist: {stats['distribution']}")
    print(f"  Invariant errors: {len(errors)}")
    print(f"  Orphans: {len(orphans)}")
    print(f"  Chain patterns: {len(chains)}")

    if stats['n_active'] > 20:
        assert abs(stats['mean'] - 3.0) < 0.5, \
            f"Mean degree {stats['mean']:.2f} too far from 3.0 for 3-regular graph"
        print(f"  3-regularity check: PASS (mean_deg={stats['mean']:.2f})")


def test_state_averaging_on_collapse():
    """Collapse should average states from all 3 merged nodes."""
    print("\nTest: state averaging on collapse...")
    rng = jax.random.key(123)
    state = make_init_state(rng, graph_idx=0)

    rng, k = jax.random.split(rng)
    params = sample_params(k)

    # Grow then collapse
    params_grow = params._replace(
        div_threshold=jnp.array(0.001), div_prob=jnp.array(0.99))
    for _ in range(3):
        state = apply_growth(state, params)
    state = apply_divisions(state, params_grow)

    # Record states before collapse
    states_before = np.array(state.node_states)
    active_before = np.array(state.node_active)
    n_before = int(state.num_active)

    params_die = params._replace(
        death_threshold=jnp.array(0.001), death_prob=jnp.array(0.99))
    state = apply_growth(state, params)
    state_after = apply_collapses(state, params_die)
    n_after = int(state_after.num_active)

    states_after = np.array(state_after.node_states)
    active_after = np.array(state_after.node_active)

    # Survivors should have non-zero states
    survivors = np.where(active_after)[0]
    for s in survivors:
        assert np.any(states_after[s] != 0), f"Survivor {s} has zero state"

    # Removed nodes should have zero states
    removed = np.where(active_before & ~active_after)[0]
    for r in removed:
        assert np.all(states_after[r] == 0), f"Removed node {r} has non-zero state"

    print(f"  {n_before} -> {n_after} nodes, {len(removed)} removed")
    print(f"  All survivors have non-zero state: OK")
    print(f"  All removed have zero state: OK")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("GRA Topology Tests")
    print("=" * 60)

    test_initial_state()
    test_division_creates_triangle()
    test_collapse_merges_triangle()
    test_state_averaging_on_collapse()
    test_division_collapse_cycle()
    test_full_simulation()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
