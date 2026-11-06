"""
MAP-Elites search for 3-state discrete Graph-Rewriting Automata (split+merge).

Searches the 3^21 ~ 10.5B rule space for interesting behaviors using
quality-diversity search with trit-flip mutation.

Behavioral descriptors (2D grid):
  - BD1: Growth ratio = log2(final_nodes / initial_nodes)
  - BD2: Mean state entropy across simulation (max = log2(3) ~ 1.585)
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import sim_discrete_merge as sim_mod
from sim_discrete_merge import (
    Rule, SimState, StepMetrics, DEGREE, NUM_CONFIGS,
    sample_rule, mutate_rule, rules_from_tables,
    make_init_state, run_simulation,
    compute_graph_features, structure_novelty,
)

# ── MAP-Elites grid ──────────────────────────────────────────────────────────

BD1_MIN, BD1_MAX = -2.0, 8.0
BD1_BINS = 20
BD2_MIN, BD2_MAX = 0.0, 1.6
BD2_BINS = 20

EXPLORATION_FRACTION = 0.25


def configure_search(cfg):
    """Set search constants from config."""
    global BD1_MIN, BD1_MAX, BD1_BINS, BD2_MIN, BD2_MAX, BD2_BINS
    global EXPLORATION_FRACTION
    me = cfg if not hasattr(cfg, "map_elites") else cfg.map_elites
    BD1_MIN = float(me.bd1_min)
    BD1_MAX = float(me.bd1_max)
    BD1_BINS = int(me.bd1_bins)
    BD2_MIN = float(me.bd2_min)
    BD2_MAX = float(me.bd2_max)
    BD2_BINS = int(me.bd2_bins)
    EXPLORATION_FRACTION = float(me.exploration_fraction)


# ── Archive ──────────────────────────────────────────────────────────────────

class Archive:
    """MAP-Elites archive: best fitness + rule + summary per grid cell."""

    def __init__(self):
        shape = (BD1_BINS, BD2_BINS)
        self.fitness = jnp.full(shape, -jnp.inf)
        self.params = None
        self.summary = None
        self.count = jnp.zeros(shape, dtype=jnp.int32)

    def update(self, bd1, bd2, fitness, params, summary):
        batch_size = fitness.shape[0]
        b1 = jnp.clip(
            (bd1 - BD1_MIN) / (BD1_MAX - BD1_MIN) * BD1_BINS,
            0, BD1_BINS - 1).astype(jnp.int32)
        b2 = jnp.clip(
            (bd2 - BD2_MIN) / (BD2_MAX - BD2_MIN) * BD2_BINS,
            0, BD2_BINS - 1).astype(jnp.int32)

        if self.params is None:
            shape = (BD1_BINS, BD2_BINS)
            self.params = jax.tree.map(
                lambda x: jnp.zeros(shape + x.shape[1:], dtype=x.dtype), params)
            self.summary = jax.tree.map(
                lambda x: jnp.zeros(shape + x.shape[1:], dtype=x.dtype), summary)

        fitness_np = jax.device_get(fitness)
        b1_np = jax.device_get(b1)
        b2_np = jax.device_get(b2)

        for i in range(batch_size):
            r, c = int(b1_np[i]), int(b2_np[i])
            if fitness_np[i] > float(self.fitness[r, c]):
                self.fitness = self.fitness.at[r, c].set(fitness_np[i])
                self.count = self.count.at[r, c].set(self.count[r, c] + 1)
                self.params = jax.tree.map(
                    lambda arr, val: arr.at[r, c].set(val[i]),
                    self.params, params)
                self.summary = jax.tree.map(
                    lambda arr, val: arr.at[r, c].set(val[i]),
                    self.summary, summary)

    def coverage(self):
        return float(jnp.sum(self.fitness > -jnp.inf)) / (BD1_BINS * BD2_BINS)

    def best_cells(self, n=10):
        flat = self.fitness.flatten()
        top_idx = jnp.argsort(-flat)[:n]
        results = []
        for idx in top_idx:
            r, c = divmod(int(idx), BD2_BINS)
            f = float(flat[idx])
            if f > -jnp.inf:
                results.append((r, c, f))
        return results

    def diverse_cells(self, n=5, min_dist=3):
        """Select top-fitness cells with minimum BD grid distance between them."""
        flat = np.array(self.fitness.flatten())
        order = np.argsort(-flat)
        selected = []
        for idx in order:
            f = float(flat[idx])
            if f <= -1e6:
                break
            r, c = divmod(int(idx), BD2_BINS)
            too_close = False
            for sr, sc, _ in selected:
                if abs(r - sr) + abs(c - sc) < min_dist:
                    too_close = True
                    break
            if not too_close:
                selected.append((r, c, f))
            if len(selected) >= n:
                break
        return selected


# ── Fitness evaluation ───────────────────────────────────────────────────────

def evaluate_metrics(all_metrics, final_state, init_num_active):
    """Compute fitness + behavioral descriptors for one simulation."""
    active = all_metrics.num_active.astype(jnp.float32)
    init_n = jnp.maximum(jnp.float32(init_num_active), 1.0)
    final_n = active[-1]

    # BD1: growth ratio
    bd1 = jnp.log2(jnp.maximum(final_n, 1.0) / init_n)

    # BD2: mean state entropy
    # state_fractions is (T, 3), compute entropy per step then average
    fracs = all_metrics.state_fractions  # (T, 3)
    per_step_entropy = -jnp.sum(
        jnp.where(fracs > 1e-8,
                  fracs * jnp.log2(jnp.maximum(fracs, 1e-8)),
                  0.0),
        axis=1)
    bd2 = jnp.mean(per_step_entropy)

    # ── Fitness components ────────────────────────────────────────────

    # 1. State activity: oscillation in node count
    active_std = jnp.std(active) / jnp.maximum(jnp.mean(active), 1.0)
    activity_score = jnp.tanh(active_std * 5.0)

    # 2. Growth variation
    growth_std = jnp.std(active) / jnp.maximum(jnp.mean(active), 1.0)
    growth_score = jnp.tanh(growth_std * 3.0)

    # 3. Temporal dynamics: fraction of steps with state changes
    changes_frac = all_metrics.num_state_changes.astype(jnp.float32) \
        / jnp.maximum(active, 1.0)
    mean_change_frac = jnp.mean(changes_frac)
    dynamics_score = jnp.exp(
        -((jnp.log(jnp.maximum(mean_change_frac, 1e-6))
           - jnp.log(0.2)) ** 2) / 2.0)

    # 4. Graph structure novelty
    novelty = structure_novelty(final_state)
    structure_score = jnp.tanh(novelty * 0.3)

    # 5. State diversity: reward high entropy (all 3 states used)
    diversity_score = jnp.tanh(bd2 * 2.0)

    # 6. Topology balance: reward rules that both split and merge
    total_divs = all_metrics.num_divisions.sum().astype(jnp.float32)
    total_merges = all_metrics.num_merges.sum().astype(jnp.float32)
    total_topo = jnp.maximum(total_divs + total_merges, 1.0)
    topo_balance = 1.0 - jnp.abs(total_divs - total_merges) / total_topo
    topo_activity = jnp.tanh((total_divs + total_merges) * 0.01)
    topo_score = topo_balance * topo_activity

    # 7. Penalty: halted (no growth AND no state changes in last quarter)
    last_quarter = active.shape[0] // 4
    late_changes = all_metrics.num_state_changes[-last_quarter:]
    late_divs = all_metrics.num_divisions[-last_quarter:]
    late_merges = all_metrics.num_merges[-last_quarter:]
    halted = (jnp.sum(late_changes) == 0) \
        & (jnp.sum(late_divs) == 0) \
        & (jnp.sum(late_merges) == 0)
    halted_penalty = jnp.where(halted, -1.0, 0.0)

    fitness = (
        activity_score * 1.0
        + growth_score * 1.0
        + dynamics_score * 1.0
        + structure_score * 2.0
        + diversity_score * 1.5
        + topo_score * 1.5
        + halted_penalty
    )

    # ── Summary for archive ───────────────────────────────────────────
    gf = compute_graph_features(final_state)
    summary = {
        "activity": activity_score,
        "growth": growth_score,
        "dynamics": dynamics_score,
        "structure": structure_score,
        "diversity": diversity_score,
        "topo_balance": topo_score,
        "clustering": gf.clustering,
        "state_entropy": gf.state_entropy,
        "boundary_fraction": gf.boundary_fraction,
        "final_nodes": final_n,
        "max_nodes": jnp.max(active),
        "mean_change_frac": mean_change_frac,
        "total_divisions": total_divs,
        "total_merges": total_merges,
    }
    return fitness, bd1, bd2, summary


@jax.jit
def evaluate_batch(all_metrics, final_states, init_num_active):
    return jax.vmap(evaluate_metrics, in_axes=(0, 0, None))(
        all_metrics, final_states, init_num_active)


# ── Batch evaluation helper ──────────────────────────────────────────────────

def make_eval_fn(init_state, sim_steps, batch_size):
    """Create a JIT-compiled batch evaluation function."""
    @partial(jax.jit, static_argnums=(1,))
    def eval_generation(rules_batch, sim_steps):
        def run_one(rule):
            return run_simulation(rule, init_state, sim_steps)
        return jax.vmap(run_one)(rules_batch)
    return eval_generation


# ── Mutation from archive ────────────────────────────────────────────────────

def _mutate_from_archive(archive, elite_indices, rng, n_mutated):
    """Select elite rules from archive and apply trit-flip mutation."""
    def gather_field(arr):
        indices = jnp.array(elite_indices)
        return arr[indices[:, 0], indices[:, 1]]

    elite_rules = jax.tree.map(gather_field, archive.params)
    keys = jax.random.split(rng, n_mutated)

    def mutate_one(table, key):
        r = Rule(table=table)
        m = mutate_rule(r, key, flip_prob=0.15)
        return m.table

    mut_tables = jax.vmap(mutate_one)(elite_rules.table, keys)
    return Rule(table=mut_tables)


# ── Search loop ──────────────────────────────────────────────────────────────

def run_search(cfg):
    """MAP-Elites search over 3-state discrete GRA rules."""
    return run_map_elites(cfg)


def run_map_elites(cfg):
    """Standard MAP-Elites with random sampling + trit-flip mutation."""
    import time
    from tqdm import tqdm

    sim_mod.configure(cfg)
    configure_search(cfg)

    num_generations = cfg.search.num_generations
    batch_size = cfg.search.batch_size
    sim_steps = cfg.search.sim_steps
    seed = cfg.search.seed
    graph = cfg.sim.get("graph", "petersen")

    # Wandb
    use_wandb = cfg.wandb.enabled
    wb_run = None
    if use_wandb:
        import wandb
        wb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            config=dict(cfg),
            tags=list(cfg.wandb.get("tags", [])),
            notes=cfg.wandb.get("notes", ""),
        )

    rng = jax.random.key(seed)
    init_state = make_init_state(rng, graph)
    init_num_active = int(init_state.num_active)

    print(f"MAP-Elites (3-state split+merge): {num_generations} gens x "
          f"{batch_size} batch x {sim_steps} steps")
    print(f"Initial graph: '{graph}' ({init_num_active} nodes), "
          f"MAX_NODES={sim_mod.MAX_NODES}")
    print(f"MAP-Elites grid: {BD1_BINS}x{BD2_BINS} = {BD1_BINS * BD2_BINS} cells")
    print(f"Rule space: 3^{NUM_CONFIGS} ~ {3**NUM_CONFIGS:.1e} rules")
    print(f"Device: {jax.devices()[0]}")

    archive = Archive()

    # JIT compile
    print("JIT compiling...")
    t0 = time.time()
    eval_fn = make_eval_fn(init_state, sim_steps, batch_size)

    warmup_rules = jax.vmap(sample_rule)(jax.random.split(rng, batch_size))
    _ = eval_fn(warmup_rules, sim_steps)
    print(f"Compiled in {time.time() - t0:.1f}s")

    total_evals = 0
    exploration_gens = int(num_generations * EXPLORATION_FRACTION)
    pbar = tqdm(range(num_generations), desc="Search", unit="gen")

    for gen in pbar:
        rng, k_rules, k_mut = jax.random.split(rng, 3)

        if gen < exploration_gens:
            keys = jax.random.split(k_rules, batch_size)
            rules_batch = jax.vmap(sample_rule)(keys)
        else:
            n_random = batch_size // 2
            n_mutated = batch_size - n_random

            keys_r = jax.random.split(k_rules, n_random)
            random_rules = jax.vmap(sample_rule)(keys_r)

            best = archive.best_cells(
                n=max(1, int(archive.coverage() * BD1_BINS * BD2_BINS)))
            if len(best) > 0:
                elite_indices = [
                    (best[i % len(best)][0], best[i % len(best)][1])
                    for i in range(n_mutated)]
                mutated_rules = _mutate_from_archive(
                    archive, elite_indices, k_mut, n_mutated)
            else:
                keys_m = jax.random.split(k_mut, n_mutated)
                mutated_rules = jax.vmap(sample_rule)(keys_m)

            rules_batch = jax.tree.map(
                lambda r, m: jnp.concatenate([r, m], axis=0),
                random_rules, mutated_rules)

        final_states, all_metrics = eval_fn(rules_batch, sim_steps)
        fitness, bd1, bd2, summary = evaluate_batch(
            all_metrics, final_states, init_num_active)

        archive.update(bd1, bd2, fitness, rules_batch, summary)

        total_evals += batch_size
        best_fitness = float(jnp.max(
            jnp.where(archive.fitness > -jnp.inf, archive.fitness, -999)))
        mean_fitness = float(jnp.mean(fitness))
        cov = archive.coverage()

        pbar.set_postfix(
            evals=total_evals, cov=f"{cov:.0%}",
            best=f"{best_fitness:.1f}", batch=f"{mean_fitness:.1f}")

        if wb_run and gen % cfg.wandb.log_every == 0:
            wb_run.log({
                "gen": gen,
                "total_evals": total_evals,
                "coverage": cov,
                "best_fitness": best_fitness,
                "batch_mean_fitness": mean_fitness,
                "batch_mean_nodes": float(
                    jnp.mean(final_states.num_active.astype(jnp.float32))),
                "batch_mean_divs": float(jnp.mean(summary["total_divisions"])),
                "batch_mean_merges": float(jnp.mean(summary["total_merges"])),
            }, step=gen)

    _print_results(archive)

    if wb_run:
        wb_run.finish()

    return archive


def _print_results(archive):
    """Print summary of search results."""
    print(f"\nDone. Coverage {archive.coverage():.1%}")
    top = archive.best_cells(10)
    print(f"\nTop {len(top)} rules:")
    for r, c, f in top:
        bd1_val = BD1_MIN + (r + 0.5) / BD1_BINS * (BD1_MAX - BD1_MIN)
        bd2_val = BD2_MIN + (c + 0.5) / BD2_BINS * (BD2_MAX - BD2_MIN)
        final_n = int(archive.summary["final_nodes"][r, c])
        total_d = int(archive.summary["total_divisions"][r, c])
        total_m = int(archive.summary["total_merges"][r, c])
        print(f"  [{r:2d},{c:2d}] fit={f:5.2f}  growth=2^{bd1_val:4.1f}  "
              f"entropy={bd2_val:.2f}  nodes={final_n}  "
              f"divs={total_d}  merges={total_m}")


# ── Visualization helpers ────────────────────────────────────────────────────

def plot_archive(archive, save_path=None):
    """Plot MAP-Elites archive heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fitness = np.array(archive.fitness)
    fitness_masked = np.ma.masked_where(fitness < -1e6, fitness)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    im = ax.imshow(fitness_masked.T, origin="lower", aspect="auto",
                   extent=[BD1_MIN, BD1_MAX, BD2_MIN, BD2_MAX], cmap="viridis")
    ax.set_xlabel("log2(growth ratio)")
    ax.set_ylabel("Mean state entropy")
    ax.set_title("Fitness")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    count = np.array(archive.count)
    im2 = ax.imshow(count.T, origin="lower", aspect="auto",
                    extent=[BD1_MIN, BD1_MAX, BD2_MIN, BD2_MAX],
                    cmap="hot",
                    norm=mcolors.LogNorm(vmin=1, vmax=max(count.max(), 2)))
    ax.set_xlabel("log2(growth ratio)")
    ax.set_ylabel("Mean state entropy")
    ax.set_title("Evaluation count")
    plt.colorbar(im2, ax=ax)

    ax = axes[2]
    if archive.summary is not None:
        structure = np.array(archive.summary["structure"])
        structure_masked = np.ma.masked_where(fitness < -1e6, structure)
        im3 = ax.imshow(structure_masked.T, origin="lower", aspect="auto",
                        extent=[BD1_MIN, BD1_MAX, BD2_MIN, BD2_MAX],
                        cmap="plasma")
        ax.set_xlabel("log2(growth ratio)")
        ax.set_ylabel("Mean state entropy")
        ax.set_title("Structure score")
        plt.colorbar(im3, ax=ax)

    plt.suptitle("3-State GRA (split+merge) — MAP-Elites Archive", fontsize=14)
    plt.tight_layout()
    path = save_path or "archive_discrete_merge.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved archive plot to {path}")
    plt.close(fig)


def state_to_networkx(state):
    """Convert a SimState to a networkx graph + state-based colors."""
    import networkx as nx

    active = np.array(state.node_active)
    neighbors = np.array(state.neighbors)
    states = np.array(state.node_states)

    active_indices = np.where(active)[0]
    idx_map = {int(g): i for i, g in enumerate(active_indices)}

    G = nx.Graph()
    for i in range(len(active_indices)):
        G.add_node(i)

    for g_idx in active_indices:
        for slot in range(DEGREE):
            nb = int(neighbors[g_idx, slot])
            if nb >= 0 and nb in idx_map:
                a, b = idx_map[int(g_idx)], idx_map[nb]
                if a < b:
                    G.add_edge(a, b)

    # Colors: merge=red, neutral=gray, split=blue
    merge_color = np.array([0.85, 0.15, 0.15])
    neutral_color = np.array([0.6, 0.6, 0.6])
    split_color = np.array([0.15, 0.35, 0.85])
    color_map = np.array([merge_color, neutral_color, split_color])
    colors = color_map[states[active_indices]]
    return G, colors


def show_top_results(archive, n=5, graph="petersen",
                     sim_steps=200, save_dir="results_discrete_merge"):
    """Replay top-n rules and save graph snapshots."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    os.makedirs(save_dir, exist_ok=True)
    top = archive.diverse_cells(n)
    print(f"\nReplaying top {len(top)} diverse rules (saving to {save_dir}/)...")

    for rank, (r, c, fitness) in enumerate(top):
        rule = jax.tree.map(lambda arr: arr[r, c], archive.params)

        bd1_val = BD1_MIN + (r + 0.5) / BD1_BINS * (BD1_MAX - BD1_MIN)
        bd2_val = BD2_MIN + (c + 0.5) / BD2_BINS * (BD2_MAX - BD2_MIN)
        print(f"\n{'=' * 60}")
        print(f"Rank {rank + 1}: fitness={fitness:.2f}  "
              f"growth=2^{bd1_val:.1f}  entropy={bd2_val:.2f}")
        print(f"  Rule = {np.array(rule.table)}")

        summary = jax.tree.map(lambda arr: float(arr[r, c]), archive.summary)
        print(f"  final_nodes={summary['final_nodes']:.0f}  "
              f"divs={summary['total_divisions']:.0f}  "
              f"merges={summary['total_merges']:.0f}  "
              f"clustering={summary['clustering']:.4f}")

        rng = jax.random.key(rank)
        init_state = make_init_state(rng, graph)

        snapshot_every = max(1, sim_steps // 8)
        snapshots = [(0, init_state)]
        state = init_state
        for chunk_start in range(0, sim_steps, snapshot_every):
            chunk_size = min(snapshot_every, sim_steps - chunk_start)
            state, _ = run_simulation(rule, state, chunk_size)
            snapshots.append((chunk_start + chunk_size, state))

        n_snaps = len(snapshots)
        cols = min(5, n_snaps)
        rows = (n_snaps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows),
                                 facecolor="white")
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for i, (step, snap_state) in enumerate(snapshots):
            if i >= len(axes):
                break
            ax = axes[i]
            G, colors = state_to_networkx(snap_state)
            nn = G.number_of_nodes()

            if 0 < nn <= 1500:
                from viz import compute_layout
                pos = compute_layout(G, seed=42)
            else:
                pos = {}

            if nn > 0 and pos:
                edge_width = max(0.4, min(1.5, 60.0 / nn))
                edge_alpha = max(0.2, min(0.6, 50.0 / nn))

                edge_colors = []
                for u, v in G.edges():
                    c = (colors[u] + colors[v]) / 2
                    c = np.clip(c * 0.7, 0, 1)
                    edge_colors.append(c)

                nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha,
                                       edge_color=edge_colors if edge_colors else "#888",
                                       width=edge_width)
                node_size = max(8, min(80, 6000.0 / nn))
                nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                                       node_color=colors, edgecolors="none",
                                       linewidths=0)
            elif nn == 0:
                ax.text(0.5, 0.5, "DEAD", ha="center", va="center",
                        fontsize=16, color="red", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f"{nn} nodes",
                        ha="center", va="center", fontsize=10,
                        transform=ax.transAxes)

            ax.set_title(f"t={step}  n={nn}", fontsize=10)
            ax.axis("off")

        for i in range(len(snapshots), len(axes)):
            axes[i].axis("off")

        plt.suptitle(
            f"3-State Rule (rank {rank + 1}, fitness={fitness:.2f})",
            fontsize=13)
        plt.tight_layout()
        path = f"{save_dir}/rank{rank + 1}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved to {path}")
        plt.close(fig)
