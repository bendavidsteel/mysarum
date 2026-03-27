#!/usr/bin/env python3
"""
Visualize graph growth step-by-step with invariant annotations.

Produces a grid of snapshots showing the graph after each topology event,
with node labels and edge reciprocity visible. Useful for visually verifying
that division and collapse preserve graph invariants.

Run: python viz_growth.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

import sim as sim_mod

cfg = OmegaConf.load("conf/config.yaml")
sim_mod.configure(cfg)

from sim import (
    EMPTY, MAX_NODES, MAX_DEGREE,
    make_init_state, sample_params, apply_growth,
    apply_divisions, apply_collapses,
)

if not hasattr(np, "alltrue"):
    np.alltrue = np.all
import networkx as nx


def state_to_graph(state):
    """Convert SimState to networkx graph, returning (G, colors, label_map).

    label_map: {networkx_node_id: original_buffer_index} for labeling.
    """
    active = np.array(state.node_active)
    neighbors = np.array(state.neighbors)
    num_nb = np.array(state.num_neighbors)
    states = np.array(state.node_states)

    active_idx = np.where(active)[0]
    idx_map = {int(g): i for i, g in enumerate(active_idx)}

    G = nx.Graph()
    for i in range(len(active_idx)):
        G.add_node(i)

    for g_idx in active_idx:
        for slot in range(int(num_nb[g_idx])):
            nb = int(neighbors[g_idx, slot])
            if nb >= 0 and nb in idx_map:
                a, b = idx_map[int(g_idx)], idx_map[nb]
                if a < b:
                    G.add_edge(a, b)

    colors = np.clip(states[active_idx, :3], 0, 1)
    label_map = {i: int(g) for i, g in enumerate(active_idx)}
    return G, colors, label_map


def check_reciprocity(state):
    """Return count of broken reciprocal edges."""
    active = np.array(state.node_active)
    nb = np.array(state.neighbors)
    n_nb = np.array(state.num_neighbors)
    broken = 0
    for i in np.where(active)[0]:
        for s in range(int(n_nb[i])):
            j = int(nb[i, s])
            if j < 0:
                continue
            nb_of_j = nb[j, :int(n_nb[j])]
            if i not in nb_of_j:
                broken += 1
    orphans = sum(1 for i in np.where(active)[0] if int(n_nb[i]) == 0)
    return broken, orphans


def draw_snapshot(ax, G, colors, label_map, title, broken, orphans):
    """Draw one graph snapshot with labels and status."""
    ax.set_facecolor("#0a0a14")
    n = G.number_of_nodes()

    if n == 0:
        ax.text(0.5, 0.5, "EMPTY", ha="center", va="center",
                fontsize=16, color="red", transform=ax.transAxes)
        ax.axis("off")
        return

    pos = nx.spring_layout(G, iterations=120, seed=42,
                           k=2.0 / max(np.sqrt(n), 1))

    edge_alpha = np.clip(2.0 / np.sqrt(max(n, 1)), 0.2, 0.7)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha,
                           edge_color="#1a8aaa", width=1.2)

    node_size = np.clip(3000 / max(n, 1), 30, 400)
    bright = colors * 0.5 + 0.5
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color=bright, edgecolors="white",
                           linewidths=0.4)

    if n <= 80:
        labels = {k: str(v) for k, v in label_map.items()}
        font_size = max(4, min(8, 120 // n))
        nx.draw_networkx_labels(G, pos, labels, ax=ax,
                                font_size=font_size, font_color="white")

    status_color = "#44ff44" if (broken == 0 and orphans == 0) else "#ff4444"
    status = f"n={n} e={G.number_of_edges()}"
    if broken > 0 or orphans > 0:
        status += f"  BAD: {broken} broken, {orphans} orphans"
    else:
        status += "  OK"

    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.text(0.5, -0.02, status, ha="center", va="top",
            fontsize=7, color=status_color, transform=ax.transAxes)
    ax.axis("off")


def main():
    rng = jax.random.key(42)
    state = make_init_state(rng, graph_idx=0)
    rng, k = jax.random.split(rng)
    params = sample_params(k)

    # Moderate division/collapse rates for visible changes each step
    params = params._replace(
        div_threshold=jnp.array(0.05),
        div_prob=jnp.array(0.8),
        death_threshold=jnp.array(0.05),
        death_prob=jnp.array(0.5),
    )

    snapshots = []

    def snap(label):
        b, o = check_reciprocity(state)
        G, colors, lmap = state_to_graph(state)
        snapshots.append((G, colors, lmap, label, b, o))

    snap("initial")

    n_cycles = 11
    for cycle in range(n_cycles):
        # Growth steps to drive u values
        for _ in range(3):
            state = apply_growth(state, params)

        state = apply_divisions(state, params)
        snap(f"cycle {cycle} div")

        for _ in range(3):
            state = apply_growth(state, params)

        state = apply_collapses(state, params)
        snap(f"cycle {cycle} col")

    # Layout grid
    n_snaps = len(snapshots)
    cols = min(6, n_snaps)
    rows = (n_snaps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.5 * rows),
                             facecolor="#0a0a14")
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    flat_axes = [ax for row in axes for ax in row]

    for i, (G, colors, lmap, title, broken, orphans) in enumerate(snapshots):
        draw_snapshot(flat_axes[i], G, colors, lmap, title, broken, orphans)

    for i in range(n_snaps, len(flat_axes)):
        flat_axes[i].set_facecolor("#0a0a14")
        flat_axes[i].axis("off")

    plt.suptitle("GRA Graph Growth — Division & Collapse Steps",
                 fontsize=14, color="white", y=1.0)
    plt.tight_layout()

    path = "growth_viz.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
