"""
Visualization for GRA search results.

- MAP-Elites archive heatmap
- Replay interesting simulations with force-directed layout
- Metric time series
"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np

# Fix numpy 2.0 compat with older networkx
if not hasattr(np, "alltrue"):
    np.alltrue = np.all
import networkx as nx

import sim as sim_mod
from sim import (
    Params, SimState, StepMetrics,
    EMPTY,
    make_init_state, run_simulation,
)
from search import (
    Archive, BD1_MIN, BD1_MAX, BD2_MIN, BD2_MAX, BD1_BINS, BD2_BINS,
)


# ── Force-directed layout ───────────────────────────────────────────────────

def force_layout(G, iterations=500, k_repel=1.5, k_spring=8.0, seed=42):
    """Spring-electrical force-directed layout for organic graph structures.

    Uses Coulomb repulsion (1/r²) between all nodes and Hooke spring attraction
    on edges, with simulated annealing. Produces layouts matching the paper's
    Mathematica-style organic visualizations.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}
    if n == 1:
        return {list(G.nodes())[0]: np.array([0.0, 0.0])}

    rng_state = np.random.RandomState(seed)
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    pos = rng_state.randn(n, 2).astype(np.float64) * np.sqrt(n)
    vel = np.zeros_like(pos)

    edges = np.array([(node_idx[u], node_idx[v]) for u, v in G.edges()],
                     dtype=np.int32) if G.number_of_edges() > 0 \
            else np.empty((0, 2), dtype=np.int32)

    rest_len = 1.0

    for it in range(iterations):
        forces = np.zeros_like(pos)
        t = max(0.005, 1.0 - it / iterations)

        # Repulsion: Coulomb 1/r²
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        dist2 = dx**2 + dy**2 + 0.01
        dist = np.sqrt(dist2)

        rep_f = k_repel / dist2
        np.fill_diagonal(rep_f, 0)
        forces[:, 0] += (dx / dist * rep_f).sum(axis=1)
        forces[:, 1] += (dy / dist * rep_f).sum(axis=1)

        # Spring attraction on edges
        if len(edges) > 0:
            ei, ej = edges[:, 0], edges[:, 1]
            edx = pos[ej, 0] - pos[ei, 0]
            edy = pos[ej, 1] - pos[ei, 1]
            ed = np.sqrt(edx**2 + edy**2 + 1e-6)
            f = k_spring * (ed - rest_len) / ed
            np.add.at(forces[:, 0], ei, edx * f)
            np.add.at(forces[:, 1], ei, edy * f)
            np.add.at(forces[:, 0], ej, -edx * f)
            np.add.at(forces[:, 1], ej, -edy * f)

        # Damped velocity update
        vel = vel * 0.8 + forces * 0.015 * t
        disp = np.sqrt((vel**2).sum(axis=1, keepdims=True) + 1e-6)
        max_d = t * 2.0
        vel *= np.minimum(max_d / disp, 1.0)
        pos += vel
        pos -= pos.mean(axis=0)

    return {nodes[i]: pos[i] for i in range(n)}


def compute_layout(G, seed=42):
    """Compute force-directed layout, falling back to spring for huge graphs."""
    n = G.number_of_nodes()
    if n == 0:
        return {}
    if n <= 800:
        return force_layout(G, iterations=500, seed=seed)
    # For large graphs, use spring_layout with tuned params
    return nx.spring_layout(G, iterations=200, seed=seed,
                            k=3.0 / max(np.sqrt(n), 1))


# ── Archive heatmap ──────────────────────────────────────────────────────────

def plot_archive(archive: Archive, save_path: str | None = None):
    """Plot MAP-Elites archive as a heatmap with fitness values."""
    has_novelty = archive.summary is not None and "novelty" in archive.summary
    ncols = 4 if has_novelty else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    fitness = np.array(archive.fitness)
    fitness_masked = np.ma.masked_where(fitness < -1e6, fitness)

    ax = axes[0]
    im = ax.imshow(fitness_masked.T, origin="lower", aspect="auto",
                   extent=[BD1_MIN, BD1_MAX, BD2_MIN, BD2_MAX], cmap="viridis")
    ax.set_xlabel("log2(growth ratio)")
    ax.set_ylabel("Mean state variance")
    ax.set_title("Fitness")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    count = np.array(archive.count)
    im2 = ax.imshow(count.T, origin="lower", aspect="auto",
                    extent=[BD1_MIN, BD1_MAX, BD2_MIN, BD2_MAX],
                    cmap="hot", norm=mcolors.LogNorm(vmin=1, vmax=max(count.max(), 2)))
    ax.set_xlabel("log2(growth ratio)")
    ax.set_ylabel("Mean state variance")
    ax.set_title("Evaluation count")
    plt.colorbar(im2, ax=ax)

    ax = axes[2]
    if archive.summary is not None:
        dynamics = np.array(archive.summary["dynamics"])
        dynamics_masked = np.ma.masked_where(fitness < -1e6, dynamics)
        im3 = ax.imshow(dynamics_masked.T, origin="lower", aspect="auto",
                        extent=[BD1_MIN, BD1_MAX, BD2_MIN, BD2_MAX], cmap="plasma")
        ax.set_xlabel("log2(growth ratio)")
        ax.set_ylabel("Mean state variance")
        ax.set_title("Dynamics score")
        plt.colorbar(im3, ax=ax)

    if has_novelty:
        ax = axes[3]
        novelty = np.array(archive.summary["novelty"])
        novelty_masked = np.ma.masked_where(fitness < -1e6, novelty)
        im4 = ax.imshow(novelty_masked.T, origin="lower", aspect="auto",
                        extent=[BD1_MIN, BD1_MAX, BD2_MIN, BD2_MAX], cmap="inferno")
        ax.set_xlabel("log2(growth ratio)")
        ax.set_ylabel("Mean state variance")
        ax.set_title("Novelty (divergence from random)")
        plt.colorbar(im4, ax=ax)

    plt.tight_layout()
    path = save_path or "archive.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved archive plot to {path}")
    plt.close(fig)


# ── Extract graph from SimState ──────────────────────────────────────────────

def state_to_networkx(state: SimState) -> tuple[nx.Graph, np.ndarray]:
    """Convert a SimState to a networkx graph + node colors."""
    active = np.array(state.node_active)
    neighbors = np.array(state.neighbors)
    num_nb = np.array(state.num_neighbors)
    states = np.array(state.node_states)

    active_indices = np.where(active)[0]
    idx_map = {int(g): i for i, g in enumerate(active_indices)}

    G = nx.Graph()
    for i in range(len(active_indices)):
        G.add_node(i)

    for g_idx in active_indices:
        for slot in range(int(num_nb[g_idx])):
            nb = int(neighbors[g_idx, slot])
            if nb >= 0 and nb in idx_map:
                local_a = idx_map[int(g_idx)]
                local_b = idx_map[nb]
                if local_a < local_b:
                    G.add_edge(local_a, local_b)

    colors = np.clip(states[active_indices, :3], 0, 1)
    return G, colors


# ── Drawing helper ───────────────────────────────────────────────────────────

def _draw_graph(ax, G, colors, pos):
    """Draw a graph in paper-matching style (white bg, thin edges, colored nodes)."""
    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0:
        ax.text(0.5, 0.5, "DEAD", ha="center", va="center",
                fontsize=20, color="red", transform=ax.transAxes)
        return

    if not pos:
        ax.text(0.5, 0.5, f"{n} nodes\n{e} edges", ha="center", va="center",
                fontsize=12, color="black", transform=ax.transAxes)
        return

    # Thin edges colored by endpoint average
    edge_width = max(0.4, min(1.5, 60.0 / n))
    edge_alpha = max(0.2, min(0.6, 50.0 / n))

    edge_colors = []
    nodes_list = list(G.nodes())
    for u, v in G.edges():
        c = (colors[u] + colors[v]) / 2
        c = np.clip(c * 0.7, 0, 1)  # darken for edge visibility
        edge_colors.append(c)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha,
                           edge_color=edge_colors if edge_colors else "#888",
                           width=edge_width)

    # Small colored nodes, no border
    node_size = max(8, min(80, 6000.0 / n))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color=colors, edgecolors="none",
                           linewidths=0)


# ── Replay a simulation ─────────────────────────────────────────────────────

def replay_simulation(params: Params, graph_idx: int = 0,
                      num_steps: int = 500,
                      snapshot_every: int = 50,
                      seed: int = 0,
                      save_path: str | None = None):
    """Run a simulation and show snapshots of the graph at intervals."""
    MAX_NODES = sim_mod.MAX_NODES
    NUM_CHANNELS = sim_mod.NUM_CHANNELS

    rng = jax.random.key(seed)
    init_state = make_init_state(rng, graph_idx)

    snapshots = []
    metrics_list = []
    state = init_state

    rng, k = jax.random.split(rng)
    state = state._replace(
        rng=k,
        node_states=jax.random.uniform(k, (MAX_NODES, NUM_CHANNELS))
        * state.node_active[:, None].astype(jnp.float32),
    )

    snapshots.append((0, state))

    for chunk_start in range(0, num_steps, snapshot_every):
        chunk_size = min(snapshot_every, num_steps - chunk_start)
        state, chunk_metrics = run_simulation(params, state, chunk_size)
        metrics_list.append(chunk_metrics)
        snapshots.append((chunk_start + chunk_size, state))

    n_snaps = len(snapshots)
    cols = min(5, n_snaps)
    rows = (n_snaps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows),
                              facecolor="white")
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, (step, snap_state) in enumerate(snapshots):
        if i >= len(axes):
            break
        ax = axes[i]
        G, colors = state_to_networkx(snap_state)
        n_nodes = G.number_of_nodes()

        if n_nodes > 0 and n_nodes <= 1500:
            pos = compute_layout(G, seed=42)
        else:
            pos = {}

        _draw_graph(ax, G, colors, pos)
        ax.set_title(f"t={step}  n={n_nodes}  e={G.number_of_edges()}",
                     fontsize=10)
        ax.axis("off")

    for i in range(len(snapshots), len(axes)):
        axes[i].axis("off")

    plt.suptitle("GRA Simulation Replay", fontsize=14)
    plt.tight_layout()

    path = save_path or "replay.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved replay to {path}")
    plt.close(fig)

    metrics_path = path.rsplit(".", 1)[0] + "_metrics.png"
    _plot_metrics(metrics_list, snapshot_every, num_steps, save_path=metrics_path)


def _plot_metrics(metrics_list: list[StepMetrics], snapshot_every: int,
                  total_steps: int, save_path: str | None = None):
    """Plot time series of simulation metrics."""
    num_active = np.concatenate([np.array(m.num_active) for m in metrics_list])
    state_var = np.concatenate([np.array(m.state_variance) for m in metrics_list])
    state_change = np.concatenate([np.array(m.state_change) for m in metrics_list])
    mean_degree = np.concatenate([np.array(m.mean_degree) for m in metrics_list])
    steps = np.arange(len(num_active))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, num_active, color="steelblue")
    axes[0, 0].set_ylabel("Active nodes")
    axes[0, 0].set_title("Node count over time")

    for ch, (color, label) in enumerate(zip(["red", "green", "blue"], "RGB")):
        axes[0, 1].plot(steps, state_var[:, ch], color=color, alpha=0.7, label=label)
    axes[0, 1].set_ylabel("State variance")
    axes[0, 1].set_title("Spatial heterogeneity")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, state_change, color="purple")
    axes[1, 0].set_ylabel("Mean |delta state|")
    axes[1, 0].set_title("Temporal dynamics")
    axes[1, 0].set_xlabel("Step")

    axes[1, 1].plot(steps, mean_degree, color="orange")
    axes[1, 1].set_ylabel("Mean degree")
    axes[1, 1].set_title("Graph connectivity")
    axes[1, 1].set_xlabel("Step")

    plt.suptitle("Simulation Metrics", fontsize=14)
    plt.tight_layout()
    path = save_path or "metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved metrics to {path}")
    plt.close(fig)


# ── Visualize top results from archive ───────────────────────────────────────

def show_top_results(archive: Archive, n: int = 5, graph_idx: int = 0,
                     sim_steps: int = 500, save_dir: str = "results"):
    """Replay the top-n results from the archive."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    top = archive.diverse_cells(n)
    print(f"Replaying top {len(top)} diverse results (saving to {save_dir}/)...")

    for rank, (r, c, fitness) in enumerate(top):
        bd1_val = BD1_MIN + (r + 0.5) / BD1_BINS * (BD1_MAX - BD1_MIN)
        bd2_val = BD2_MIN + (c + 0.5) / BD2_BINS * (BD2_MAX - BD2_MIN)
        print(f"\n{'='*60}")
        print(f"Rank {rank+1}: fitness={fitness:.2f}  "
              f"growth=2^{bd1_val:.1f}  hetero={bd2_val:.3f}")

        params = jax.tree.map(lambda arr: arr[r, c], archive.params)
        print(f"  kernel_mu={np.array(params.kernel_mu)}")
        print(f"  growth_mu={np.array(params.growth_mu)}")
        print(f"  state_dt={float(params.state_dt):.4f}")
        print(f"  div_threshold={float(params.div_threshold):.3f}  "
              f"div_prob={float(params.div_prob):.4f}")
        print(f"  death_threshold={float(params.death_threshold):.3f}  "
              f"death_prob={float(params.death_prob):.4f}")

        if archive.summary is not None:
            summary = jax.tree.map(lambda arr: float(arr[r, c]), archive.summary)
            print(f"  osc={summary['oscillation']:.3f}  "
                  f"dyn={summary['dynamics']:.3f}  "
                  f"het={summary['heterogeneity']:.3f}  "
                  f"final_n={summary['final_nodes']:.0f}")
            if "novelty" in summary:
                print(f"  novelty={summary['novelty']:.3f}  "
                      f"div={summary['random_divergence']:.3f}  "
                      f"C={summary['clustering']:.4f} "
                      f"(rand={summary['clustering_expected']:.4f})  "
                      f"r={summary['assortativity']:.3f}  "
                      f"KL={summary['degree_kl']:.3f}")

        replay_simulation(params, graph_idx=graph_idx,
                          num_steps=sim_steps, seed=rank,
                          save_path=f"{save_dir}/rank{rank+1}.png")


# ── Animated replay ──────────────────────────────────────────────────────────

def animate_simulation(params: Params, graph_idx: int = 0,
                       num_steps: int = 300,
                       step_per_frame: int = 5,
                       seed: int = 0,
                       save_path: str | None = None):
    """Create an animated visualization of a GRA simulation."""
    MAX_NODES = sim_mod.MAX_NODES
    NUM_CHANNELS = sim_mod.NUM_CHANNELS

    rng = jax.random.key(seed)
    init_state = make_init_state(rng, graph_idx)

    rng, k = jax.random.split(rng)
    state = init_state._replace(
        rng=k,
        node_states=jax.random.uniform(k, (MAX_NODES, NUM_CHANNELS))
        * init_state.node_active[:, None].astype(jnp.float32),
    )

    frames = []
    for i in range(0, num_steps, step_per_frame):
        state, _ = run_simulation(params, state, step_per_frame)
        G, colors = state_to_networkx(state)
        n = G.number_of_nodes()
        if n > 0 and n <= 800:
            pos = compute_layout(G, seed=42)
        else:
            pos = {}
        frames.append((G, colors, pos, i + step_per_frame, n))

    if not frames:
        print("No frames generated.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor="white")

    def draw_frame(frame_idx):
        ax.clear()
        G, colors, pos, step, n = frames[frame_idx]
        _draw_graph(ax, G, colors, pos)
        ax.set_title(f"t={step}  nodes={n}", fontsize=14)
        ax.axis("off")

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(frames),
                                    interval=200, repeat=True)

    path = save_path or "animation.gif"
    anim.save(path, writer="pillow", fps=5, dpi=100)
    print(f"Saved animation to {path}")
    plt.close(fig)
    return anim
