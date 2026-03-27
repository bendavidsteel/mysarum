"""
MAP-Elites parameter search for Graph Cellular Automata.

Searches the GRA parameter space for interesting behaviors using a
quality-diversity algorithm. Evaluates simulations in parallel via vmap.

Behavioral descriptors (2D grid):
  - BD1: Growth ratio = log2(final_nodes / initial_nodes)
  - BD2: Mean spatial heterogeneity = mean state variance across nodes

Fitness combines oscillation, temporal dynamics, non-uniformity, survival,
with penalties for immediate death.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import sim as sim_mod
from sim import (
    Params, SimState, StepMetrics,
    sample_params, make_init_state, run_simulation,
    random_divergence, compute_graph_features,
)


# ── MAP-Elites grid (defaults, overridden by config) ─────────────────────────

BD1_MIN, BD1_MAX = -3.0, 7.0
BD1_BINS = 20
BD2_MIN, BD2_MAX = 0.0, 0.25
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


class Archive:
    """MAP-Elites archive: stores best fitness + params in each grid cell."""

    def __init__(self):
        shape = (BD1_BINS, BD2_BINS)
        self.fitness = jnp.full(shape, -jnp.inf)
        self.params = None
        self.summary = None
        self.count = jnp.zeros(shape, dtype=jnp.int32)

    def update(self, bd1, bd2, fitness, params, summary):
        batch_size = fitness.shape[0]
        b1 = jnp.clip((bd1 - BD1_MIN) / (BD1_MAX - BD1_MIN) * BD1_BINS,
                       0, BD1_BINS - 1).astype(jnp.int32)
        b2 = jnp.clip((bd2 - BD2_MIN) / (BD2_MAX - BD2_MIN) * BD2_BINS,
                       0, BD2_BINS - 1).astype(jnp.int32)

        if self.params is None:
            shape = (BD1_BINS, BD2_BINS)
            self.params = jax.tree.map(lambda x: jnp.zeros(shape + x.shape[1:]), params)
            self.summary = jax.tree.map(lambda x: jnp.zeros(shape + x.shape[1:]), summary)

        fitness_np = jax.device_get(fitness)
        b1_np = jax.device_get(b1)
        b2_np = jax.device_get(b2)

        for i in range(batch_size):
            r, c = int(b1_np[i]), int(b2_np[i])
            if fitness_np[i] > float(self.fitness[r, c]):
                self.fitness = self.fitness.at[r, c].set(fitness_np[i])
                self.count = self.count.at[r, c].set(self.count[r, c] + 1)
                self.params = jax.tree.map(
                    lambda arr, val: arr.at[r, c].set(val[i]), self.params, params)
                self.summary = jax.tree.map(
                    lambda arr, val: arr.at[r, c].set(val[i]), self.summary, summary)

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


# ── Fitness evaluation ───────────────────────────────────────────────────────

def evaluate_metrics(all_metrics, final_state, init_num_active):
    num_steps = all_metrics.num_active.shape[0]
    active = all_metrics.num_active.astype(jnp.float32)
    init_n = jnp.maximum(jnp.float32(init_num_active), 1.0)

    final_n = active[-1]
    bd1 = jnp.log2(jnp.maximum(final_n, 1.0) / init_n)
    bd2 = all_metrics.state_variance.mean()

    count_std = jnp.std(active) / jnp.maximum(jnp.mean(active), 1.0)
    oscillation_score = jnp.tanh(count_std * 5.0)

    mean_change = jnp.mean(all_metrics.state_change)
    dynamics_score = jnp.exp(
        -((jnp.log(jnp.maximum(mean_change, 1e-6)) - jnp.log(0.02)) ** 2) / 2.0)

    hetero_score = jnp.tanh(bd2 * 20.0)
    survival_frac = jnp.mean((active > init_n * 0.5).astype(jnp.float32))
    degree_score = jnp.tanh(jnp.std(all_metrics.mean_degree) * 2.0)

    max_active = jnp.max(active)
    died_immediately = (max_active < init_n * 1.1) & (final_n < init_n * 0.5)
    death_penalty = jnp.where(died_immediately, -2.0, 0.0)

    # Divergence from random null model
    div_score = random_divergence(final_state)
    novelty_score = jnp.tanh(div_score * 0.5)

    fitness = (
        oscillation_score * 1.0
        + dynamics_score * 1.5
        + hetero_score * 1.0
        + survival_frac * 1.0
        + degree_score * 0.5
        + novelty_score * 2.0
        + death_penalty
    )

    gf = compute_graph_features(final_state)
    summary = {
        "oscillation": oscillation_score,
        "dynamics": dynamics_score,
        "heterogeneity": hetero_score,
        "survival": survival_frac,
        "degree_var": degree_score,
        "novelty": novelty_score,
        "random_divergence": div_score,
        "degree_kl": gf.degree_kl,
        "clustering": gf.clustering,
        "clustering_expected": gf.clustering_expected,
        "assortativity": gf.assortativity,
        "final_nodes": final_n,
        "max_nodes": max_active,
        "mean_change": mean_change,
    }
    return fitness, bd1, bd2, summary


@jax.jit
def evaluate_batch(all_metrics, final_states, init_num_active):
    return jax.vmap(evaluate_metrics, in_axes=(0, 0, None))(
        all_metrics, final_states, init_num_active)


# ── Mutation ─────────────────────────────────────────────────────────────────

def _mutate_from_archive(archive, elite_indices, keys):
    def gather_field(arr):
        indices = jnp.array(elite_indices)
        return arr[indices[:, 0], indices[:, 1]]

    elite_params = jax.tree.map(gather_field, archive.params)

    def mutate_field(vals, key):
        noise = jax.random.normal(key, vals.shape) * 0.1
        scale = jnp.maximum(jnp.abs(vals).mean(), 0.01)
        return vals + noise * scale

    mutated = jax.tree.map(
        lambda v, k: mutate_field(v, k),
        elite_params,
        jax.tree.unflatten(
            jax.tree.structure(elite_params),
            jax.random.split(keys[0], len(jax.tree.leaves(elite_params))),
        ),
    )

    mutated = mutated._replace(
        kernel_mu=jnp.clip(mutated.kernel_mu, 0.0, 18.0),
        kernel_sigma=jnp.clip(mutated.kernel_sigma, 0.1, 4.0),
        growth_mu=jnp.clip(mutated.growth_mu, 0.0, 1.0),
        growth_sigma=jnp.clip(mutated.growth_sigma, 0.05, 1.0),
        state_dt=jnp.clip(mutated.state_dt, 0.001, 0.2),
        div_mu=jnp.clip(mutated.div_mu, 0.0, 1.0),
        div_sigma=jnp.clip(mutated.div_sigma, 0.05, 1.0),
        div_threshold=jnp.clip(mutated.div_threshold, 0.3, 0.99),
        div_prob=jnp.clip(mutated.div_prob, 0.0001, 0.05),
        death_mu=jnp.clip(mutated.death_mu, 0.0, 1.0),
        death_sigma=jnp.clip(mutated.death_sigma, 0.05, 1.0),
        death_threshold=jnp.clip(mutated.death_threshold, 0.3, 0.99),
        death_prob=jnp.clip(mutated.death_prob, 0.0001, 0.05),
    )
    return mutated


# ── Search loop ──────────────────────────────────────────────────────────────

def run_search(cfg):
    """Run MAP-Elites search. cfg is a hydra/OmegaConf config."""
    import time
    from tqdm import tqdm

    # Configure modules from config
    sim_mod.configure(cfg)
    configure_search(cfg)

    num_generations = cfg.search.num_generations
    batch_size = cfg.search.batch_size
    sim_steps = cfg.search.sim_steps
    seed = cfg.search.seed
    graph_idx = cfg.search.graph_idx

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
    rng, k_init = jax.random.split(rng)
    init_state = make_init_state(k_init, graph_idx)
    init_num_active = int(init_state.num_active)

    MAX_NODES = sim_mod.MAX_NODES
    NUM_CHANNELS = sim_mod.NUM_CHANNELS

    print(f"Search: {num_generations} gens x {batch_size} batch x {sim_steps} steps")
    print(f"Initial graph: {init_num_active} nodes, MAX_NODES={MAX_NODES}")
    print(f"MAP-Elites grid: {BD1_BINS}x{BD2_BINS} = {BD1_BINS*BD2_BINS} cells")
    print(f"Device: {jax.devices()[0]}")

    archive = Archive()

    # JIT compile
    print("JIT compiling...")
    t0 = time.time()

    @partial(jax.jit, static_argnums=(2,))
    def eval_generation(params_batch, init_state, sim_steps):
        batch_size = params_batch.kernel_mu.shape[0]
        rngs = jax.random.split(init_state.rng, batch_size)

        def run_one(params, rng):
            state = init_state._replace(
                rng=rng,
                node_states=jax.random.uniform(rng, (MAX_NODES, NUM_CHANNELS))
                * init_state.node_active[:, None].astype(jnp.float32),
            )
            return run_simulation(params, state, sim_steps)

        return jax.vmap(run_one)(params_batch, rngs)

    # Warmup with actual batch size to avoid recompilation
    rng, k_w = jax.random.split(rng)
    warmup_params = jax.vmap(sample_params)(jax.random.split(k_w, batch_size))
    _ = eval_generation(warmup_params, init_state, sim_steps)
    compile_time = time.time() - t0
    print(f"Compiled in {compile_time:.1f}s")

    if wb_run:
        wb_run.log({"compile_time_s": compile_time})

    # Main loop
    total_evals = 0
    exploration_gens = int(num_generations * EXPLORATION_FRACTION)
    pbar = tqdm(range(num_generations), desc="Search", unit="gen")

    for gen in pbar:
        rng, k_params, k_state = jax.random.split(rng, 3)

        if gen < exploration_gens:
            keys = jax.random.split(k_params, batch_size)
            params_batch = jax.vmap(sample_params)(keys)
        else:
            n_random = batch_size // 2
            n_mutated = batch_size - n_random

            keys_r = jax.random.split(k_params, n_random + 1)
            k_params = keys_r[0]
            random_params = jax.vmap(sample_params)(keys_r[1:])

            best = archive.best_cells(
                n=max(1, int(archive.coverage() * BD1_BINS * BD2_BINS)))
            if len(best) > 0:
                keys_m = jax.random.split(k_params, n_mutated + 1)
                elite_indices = [(best[i % len(best)][0], best[i % len(best)][1])
                                 for i in range(n_mutated)]
                mutated_params = _mutate_from_archive(archive, elite_indices, keys_m[1:])
            else:
                keys_m = jax.random.split(k_params, n_mutated)
                mutated_params = jax.vmap(sample_params)(keys_m)

            params_batch = jax.tree.map(
                lambda r, m: jnp.concatenate([r, m], axis=0),
                random_params, mutated_params)

        init_state_gen = init_state._replace(rng=k_state)
        final_states, all_metrics = eval_generation(params_batch, init_state_gen, sim_steps)
        fitness, bd1, bd2, summary = evaluate_batch(all_metrics, final_states, init_num_active)
        archive.update(bd1, bd2, fitness, params_batch, jax.tree.map(lambda x: x, summary))

        total_evals += batch_size
        best_fitness = float(jnp.max(
            jnp.where(archive.fitness > -jnp.inf, archive.fitness, -999)))
        mean_fitness = float(jnp.mean(fitness))
        cov = archive.coverage()

        pbar.set_postfix(
            evals=total_evals, cov=f"{cov:.0%}",
            best=f"{best_fitness:.1f}", batch=f"{mean_fitness:.1f}")

        # Wandb logging
        if wb_run and gen % cfg.wandb.log_every == 0:
            wb_run.log({
                "gen": gen,
                "total_evals": total_evals,
                "coverage": cov,
                "best_fitness": best_fitness,
                "batch_mean_fitness": mean_fitness,
                "batch_max_fitness": float(jnp.max(fitness)),
                "batch_mean_nodes": float(jnp.mean(final_states.num_active.astype(jnp.float32))),
                "batch_mean_novelty": float(jnp.mean(summary["novelty"])),
                "batch_mean_divergence": float(jnp.mean(summary["random_divergence"])),
                "batch_mean_clustering": float(jnp.mean(summary["clustering"])),
                "batch_mean_assortativity": float(jnp.mean(summary["assortativity"])),
            }, step=gen)

        if wb_run and cfg.wandb.log_archive_every > 0 \
                and gen % cfg.wandb.log_archive_every == 0 and gen > 0:
            from viz import plot_archive
            plot_archive(archive, save_path=f"archive_gen{gen}.png")
            wb_run.log({"archive": wandb.Image(f"archive_gen{gen}.png")}, step=gen)

    print(f"\nDone. {total_evals} evals, coverage {archive.coverage():.1%}")

    top = archive.best_cells(5)
    print("\nTop 5:")
    for r, c, f in top:
        bd1_val = BD1_MIN + (r + 0.5) / BD1_BINS * (BD1_MAX - BD1_MIN)
        bd2_val = BD2_MIN + (c + 0.5) / BD2_BINS * (BD2_MAX - BD2_MIN)
        print(f"  [{r},{c}] fitness={f:.2f}  growth=2^{bd1_val:.1f}  hetero={bd2_val:.3f}")

    if wb_run:
        from viz import plot_archive
        plot_archive(archive, save_path="archive_final.png")
        wb_run.log({"archive_final": wandb.Image("archive_final.png")})
        wb_run.finish()

    return archive
