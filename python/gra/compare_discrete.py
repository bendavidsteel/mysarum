#!/usr/bin/env python3
"""
Compare network diversity between discrete GRA (split-only) and
3-state discrete GRA (split+merge).

Runs MAP-Elites on both models with matched parameters, then compares:
- Archive coverage (fraction of BD grid filled)
- Distribution of graph sizes
- Structural diversity: clustering, boundary fraction, state patterns
- Topological activity: divisions vs merges

Usage:
    python compare_discrete.py [--generations N] [--batch_size N] [--sim_steps N]
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf


def make_cfg(sim_type, max_nodes=1024, max_topo_ops=64,
             num_generations=50, batch_size=64, sim_steps=200,
             seed=42):
    """Build a minimal OmegaConf config matching what the modules expect."""
    return OmegaConf.create({
        "sim": {
            "type": sim_type,
            "max_nodes": max_nodes,
            "max_topo_ops": max_topo_ops,
            "graph": "petersen",
        },
        "search": {
            "num_generations": num_generations,
            "batch_size": batch_size,
            "sim_steps": sim_steps,
            "seed": seed,
            "mode": "map_elites",
        },
        "map_elites": {
            "bd1_min": -2.0,
            "bd1_max": 8.0,
            "bd1_bins": 20,
            "bd2_min": 0.0,
            "bd2_max": 1.0,
            "bd2_bins": 20,
            "exploration_fraction": 0.25,
        },
        "wandb": {"enabled": False},
        "viz": {
            "top_n": 5,
            "replay_steps": 200,
            "save_dir": "results",
            "animate": False,
        },
    })


def run_discrete_search(cfg):
    """Run split-only discrete GRA search."""
    import sim_discrete as sim_mod
    from search_discrete import (
        run_map_elites, configure_search, Archive,
        evaluate_batch, make_eval_fn,
    )
    from sim_discrete import (
        sample_rule, make_init_state, compute_graph_features,
        structure_novelty,
    )

    sim_mod.configure(cfg)
    configure_search(cfg)
    print("\n" + "=" * 70)
    print("DISCRETE GRA (split-only, 2-state)")
    print("=" * 70)
    archive = run_map_elites(cfg)
    return archive


def run_discrete_merge_search(cfg):
    """Run split+merge 3-state discrete GRA search."""
    import sim_discrete_merge as sim_mod
    from search_discrete_merge import (
        run_map_elites, configure_search, Archive,
    )

    sim_mod.configure(cfg)
    # BD2 for this model is entropy (max ~1.585), adjust range
    cfg_dm = OmegaConf.create(OmegaConf.to_container(cfg))
    cfg_dm.map_elites.bd2_max = 1.6
    configure_search(cfg_dm)
    print("\n" + "=" * 70)
    print("3-STATE GRA (split+merge)")
    print("=" * 70)
    archive = run_map_elites(cfg_dm)
    return archive


def collect_final_features(archive, sim_type):
    """Replay top rules and collect structural features for comparison."""
    features = {
        "final_nodes": [],
        "clustering": [],
        "boundary_fraction": [],
    }

    if archive.summary is None:
        return features

    # Collect from all filled cells
    fitness = np.array(archive.fitness)
    filled = fitness > -1e6

    for key in features:
        if key in archive.summary:
            vals = np.array(archive.summary[key])
            features[key] = vals[filled].tolist()

    if sim_type == "discrete_merge":
        features["total_divisions"] = np.array(
            archive.summary.get("total_divisions", np.zeros_like(fitness))
        )[filled].tolist()
        features["total_merges"] = np.array(
            archive.summary.get("total_merges", np.zeros_like(fitness))
        )[filled].tolist()

    return features


def plot_comparison(archive_disc, archive_merge, features_disc, features_merge,
                    save_path="comparison.png"):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── Row 1: Archive heatmaps ──────────────────────────────────────
    fitness_d = np.array(archive_disc.fitness)
    fitness_m = np.array(archive_merge.fitness)
    vmin = min(fitness_d[fitness_d > -1e6].min() if np.any(fitness_d > -1e6) else 0,
               fitness_m[fitness_m > -1e6].min() if np.any(fitness_m > -1e6) else 0)
    vmax = max(fitness_d[fitness_d > -1e6].max() if np.any(fitness_d > -1e6) else 1,
               fitness_m[fitness_m > -1e6].max() if np.any(fitness_m > -1e6) else 1)

    ax = axes[0, 0]
    fd_masked = np.ma.masked_where(fitness_d < -1e6, fitness_d)
    im = ax.imshow(fd_masked.T, origin="lower", aspect="auto",
                   extent=[-2, 8, 0, 1], cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("log2(growth ratio)")
    ax.set_ylabel("BD2")
    ax.set_title(f"Discrete (split-only)\ncoverage={archive_disc.coverage():.0%}")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    fm_masked = np.ma.masked_where(fitness_m < -1e6, fitness_m)
    im = ax.imshow(fm_masked.T, origin="lower", aspect="auto",
                   extent=[-2, 8, 0, 1.6], cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("log2(growth ratio)")
    ax.set_ylabel("BD2 (entropy)")
    ax.set_title(f"3-State (split+merge)\ncoverage={archive_merge.coverage():.0%}")
    plt.colorbar(im, ax=ax)

    # Coverage bar chart
    ax = axes[0, 2]
    covs = [archive_disc.coverage(), archive_merge.coverage()]
    bars = ax.bar(["Split-only", "Split+Merge"], covs,
                  color=["#4477AA", "#CC6677"])
    ax.set_ylabel("Coverage")
    ax.set_title("Archive Coverage")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, covs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.0%}", ha="center", fontsize=11)

    # ── Row 2: Feature distributions ─────────────────────────────────

    # Final node count distribution
    ax = axes[1, 0]
    if features_disc["final_nodes"] and features_merge["final_nodes"]:
        all_vals = features_disc["final_nodes"] + features_merge["final_nodes"]
        bins = np.linspace(0, np.percentile(all_vals, 95), 30)
        ax.hist(features_disc["final_nodes"], bins=bins, alpha=0.6,
                label="Split-only", color="#4477AA", density=True)
        ax.hist(features_merge["final_nodes"], bins=bins, alpha=0.6,
                label="Split+Merge", color="#CC6677", density=True)
        ax.set_xlabel("Final node count")
        ax.set_ylabel("Density")
        ax.set_title("Graph Size Distribution")
        ax.legend()

    # Clustering distribution
    ax = axes[1, 1]
    if features_disc["clustering"] and features_merge["clustering"]:
        bins = np.linspace(0, 1, 30)
        ax.hist(features_disc["clustering"], bins=bins, alpha=0.6,
                label="Split-only", color="#4477AA", density=True)
        ax.hist(features_merge["clustering"], bins=bins, alpha=0.6,
                label="Split+Merge", color="#CC6677", density=True)
        ax.set_xlabel("Clustering coefficient")
        ax.set_ylabel("Density")
        ax.set_title("Clustering Distribution")
        ax.legend()

    # Boundary fraction distribution
    ax = axes[1, 2]
    if features_disc["boundary_fraction"] and features_merge["boundary_fraction"]:
        bins = np.linspace(0, 1, 30)
        ax.hist(features_disc["boundary_fraction"], bins=bins, alpha=0.6,
                label="Split-only", color="#4477AA", density=True)
        ax.hist(features_merge["boundary_fraction"], bins=bins, alpha=0.6,
                label="Split+Merge", color="#CC6677", density=True)
        ax.set_xlabel("Boundary fraction (inter-state edges)")
        ax.set_ylabel("Density")
        ax.set_title("State Mixing Distribution")
        ax.legend()

    plt.suptitle("Discrete GRA: Split-Only vs Split+Merge", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {save_path}")
    plt.close(fig)


def print_summary(archive, features, label):
    """Print summary statistics."""
    print(f"\n{'─' * 50}")
    print(f"{label}")
    print(f"{'─' * 50}")
    print(f"  Coverage: {archive.coverage():.1%}")

    fitness = np.array(archive.fitness)
    filled = fitness[fitness > -1e6]
    if len(filled) > 0:
        print(f"  Filled cells: {len(filled)}")
        print(f"  Fitness: mean={filled.mean():.2f}, "
              f"max={filled.max():.2f}, std={filled.std():.2f}")

    for key in ["final_nodes", "clustering", "boundary_fraction"]:
        vals = features.get(key, [])
        if vals:
            arr = np.array(vals)
            print(f"  {key}: mean={arr.mean():.3f}, "
                  f"std={arr.std():.3f}, "
                  f"range=[{arr.min():.3f}, {arr.max():.3f}]")

    if "total_divisions" in features and features["total_divisions"]:
        divs = np.array(features["total_divisions"])
        merges = np.array(features["total_merges"])
        print(f"  Divisions: mean={divs.mean():.0f}")
        print(f"  Merges: mean={merges.mean():.0f}")
        ratio = divs.sum() / max(merges.sum(), 1)
        print(f"  Div/Merge ratio: {ratio:.2f}")


def replay_snapshots(sim_mod, rule, graph, sim_steps, n_snapshots=6):
    """Run a rule and capture snapshots at evenly-spaced intervals."""
    rng = jax.random.key(0)
    state = sim_mod.make_init_state(rng, graph)
    snap_every = max(1, sim_steps // (n_snapshots - 1))
    snapshots = [(0, state)]
    for chunk_start in range(0, sim_steps, snap_every):
        chunk_size = min(snap_every, sim_steps - chunk_start)
        state, _ = sim_mod.run_simulation(rule, state, chunk_size)
        snapshots.append((chunk_start + chunk_size, state))
    return snapshots[:n_snapshots]


def state_to_nx(state, num_states=2):
    """Convert a SimState to a networkx graph + node colors."""
    import networkx as nx

    active = np.array(state.node_active)
    neighbors = np.array(state.neighbors)
    states = np.array(state.node_states)
    active_idx = np.where(active)[0]
    idx_map = {int(g): i for i, g in enumerate(active_idx)}

    G = nx.Graph()
    for i in range(len(active_idx)):
        G.add_node(i)
    for g_idx in active_idx:
        for slot in range(3):
            nb = int(neighbors[g_idx, slot])
            if nb >= 0 and nb in idx_map:
                a, b = idx_map[int(g_idx)], idx_map[nb]
                if a < b:
                    G.add_edge(a, b)

    if num_states == 2:
        color_map = np.array([
            [0.2, 0.4, 0.8],    # state 0: blue
            [0.85, 0.3, 0.15],  # state 1: red
        ])
    else:
        color_map = np.array([
            [0.85, 0.15, 0.15],  # merge: red
            [0.6, 0.6, 0.6],     # neutral: gray
            [0.15, 0.35, 0.85],  # split: blue
        ])
    colors = color_map[np.clip(states[active_idx], 0, num_states - 1)]
    return G, colors


def layout_igraph(G):
    """Fast force-directed layout via igraph's C implementation.

    Uses Fruchterman-Reingold with grid-based optimization (O(n log n)
    approximation), much faster than networkx's naive O(n^2) spring_layout.
    """
    import igraph as ig

    nn = G.number_of_nodes()
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    ig_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    ig_g = ig.Graph(n=nn, edges=ig_edges, directed=False)

    # grid=True enables grid-based optimization (Barnes-Hut-like)
    layout = ig_g.layout_fruchterman_reingold(
        niter=max(100, 500 - nn), grid=True, seed=None)
    coords = np.array(layout.coords)

    # Normalize to [-1, 1]
    if nn > 1:
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        span = mx - mn
        span[span < 1e-8] = 1.0
        coords = 2.0 * (coords - mn) / span - 1.0

    return {node_list[i]: coords[i] for i in range(nn)}


def draw_snapshot(ax, G, colors, title=""):
    """Draw a single graph snapshot on an axes."""
    import networkx as nx

    nn = G.number_of_nodes()
    if nn == 0:
        ax.text(0.5, 0.5, "DEAD", ha="center", va="center",
                fontsize=14, color="red", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        return
    if nn > 2000:
        ax.text(0.5, 0.5, f"{nn} nodes\n(too large)", ha="center",
                va="center", fontsize=10, transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        return

    pos = layout_igraph(G)
    node_size = np.clip(2500 / max(nn, 1), 5, 250)
    edge_width = np.clip(3.0 / np.sqrt(max(nn, 1)), 0.15, 2.0)
    edge_alpha = np.clip(2.0 / np.sqrt(max(nn, 1)), 0.05, 0.4)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha,
                           edge_color="#666", width=edge_width)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color=colors, edgecolors="white",
                           linewidths=0.3)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_top_n(archive_disc, archive_merge, top_n=3, sim_steps=200,
               save_path="top_structures.png"):
    """Replay and visualize top-N rules from each system side by side."""
    import sim_discrete as sd
    import sim_discrete_merge as sdm
    import matplotlib.gridspec as gridspec

    N_SNAPS = 6

    fig = plt.figure(figsize=(N_SNAPS * 3.5, top_n * 2 * 3.8 + 1.5))
    outer = gridspec.GridSpec(2, 1, hspace=0.25, top=0.94, bottom=0.02)

    panels = [
        ("Split-Only (2-state)", archive_disc, sd, 2),
        ("Split+Merge (3-state)", archive_merge, sdm, 3),
    ]

    for panel_idx, (label, archive, sim_mod, num_states) in enumerate(panels):
        top = archive.best_cells(top_n)
        inner = gridspec.GridSpecFromSubplotSpec(
            top_n, N_SNAPS, subplot_spec=outer[panel_idx],
            hspace=0.35, wspace=0.08)

        for rank, (r, c, fitness) in enumerate(top):
            rule = jax.tree.map(lambda arr: arr[r, c], archive.params)
            summary = jax.tree.map(lambda arr: float(arr[r, c]), archive.summary)

            print(f"  Rendering {label} rank {rank + 1} "
                  f"(fit={fitness:.1f}, nodes={summary.get('final_nodes', 0):.0f})...")

            snapshots = replay_snapshots(sim_mod, rule, "petersen", sim_steps, N_SNAPS)

            for snap_i, (step, snap_state) in enumerate(snapshots):
                ax = fig.add_subplot(inner[rank, snap_i])
                G, colors = state_to_nx(snap_state, num_states)
                nn = G.number_of_nodes()

                col_title = f"t={step}  n={nn}"
                if snap_i == 0:
                    extra = f"#{rank+1} fit={fitness:.1f}"
                    if num_states == 3:
                        extra += (f"\ndivs={summary.get('total_divisions', 0):.0f} "
                                  f"merges={summary.get('total_merges', 0):.0f}")
                    col_title = f"{extra}\n{col_title}"

                draw_snapshot(ax, G, colors, title=col_title)

        fig.text(0.5, 0.97 if panel_idx == 0 else 0.48,
                 label, ha="center", fontsize=15, fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved top-{top_n} structures to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare discrete GRA (split-only) vs 3-state (split+merge)")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sim_steps", type=int, default=200)
    parser.add_argument("--max_nodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_n", type=int, default=0,
                        help="Visualize top N structures from each system (0=skip)")
    args = parser.parse_args()

    print(f"Comparing discrete GRAs: {args.generations} gens x "
          f"{args.batch_size} batch x {args.sim_steps} steps")
    print(f"Device: {jax.devices()[0]}")

    # ── Run split-only discrete search ───────────────────────────────
    cfg_disc = make_cfg(
        "discrete",
        max_nodes=args.max_nodes,
        max_topo_ops=64,
        num_generations=args.generations,
        batch_size=args.batch_size,
        sim_steps=args.sim_steps,
        seed=args.seed,
    )
    t0 = time.time()
    archive_disc = run_discrete_search(cfg_disc)
    time_disc = time.time() - t0

    # ── Run split+merge 3-state search ───────────────────────────────
    cfg_merge = make_cfg(
        "discrete_merge",
        max_nodes=args.max_nodes,
        max_topo_ops=64,
        num_generations=args.generations,
        batch_size=args.batch_size,
        sim_steps=args.sim_steps,
        seed=args.seed,
    )
    t0 = time.time()
    archive_merge = run_discrete_merge_search(cfg_merge)
    time_merge = time.time() - t0

    # ── Collect features and compare ─────────────────────────────────
    features_disc = collect_final_features(archive_disc, "discrete")
    features_merge = collect_final_features(archive_merge, "discrete_merge")

    print_summary(archive_disc, features_disc, "DISCRETE (split-only, 2-state)")
    print_summary(archive_merge, features_merge, "3-STATE (split+merge)")

    print(f"\nTime: discrete={time_disc:.1f}s, split+merge={time_merge:.1f}s")

    # ── Plot comparison ──────────────────────────────────────────────
    plot_comparison(archive_disc, archive_merge,
                    features_disc, features_merge,
                    save_path="comparison.png")

    # ── Top-N structure visualization ────────────────────────────────
    if args.top_n > 0:
        plot_top_n(archive_disc, archive_merge,
                   top_n=args.top_n, sim_steps=args.sim_steps,
                   save_path="top_structures.png")


if __name__ == "__main__":
    main()
