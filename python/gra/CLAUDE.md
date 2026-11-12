# GRA — Graph Cellular Automata Parameter Search

JAX implementation of a Lenia-style graph cellular automaton with Chebyshev spectral convolution, cross-channel coupling, and dynamic topology (division/collapse). Uses MAP-Elites quality-diversity search to find interesting parameter regimes.

## Architecture

- `sim.py` — Core JAX simulation: Chebyshev convolution, growth function, topology ops. All fixed-size buffers for JIT/vmap.
- `search.py` — MAP-Elites search with vmap parallel evaluation, fitness metrics, mutation from archive elites.
- `viz.py` — Post-search visualization: archive heatmaps, graph snapshots with networkx, metric time series.
- `run.py` — Hydra entry point. Config in `conf/config.yaml`.
- `graphs.json` — Seed graphs (10-12 node small graphs from the Rust implementation).
- `swim.py` — Grown swimmers: GRAs grow until their topology stops changing, then a continuous CA on the nodes (channels mix only across edges) drives edge rest lengths in a drag fluid. Subcommands `grow`, `swim`, `render`; outputs in `outputs/swim/`. Runs on GPU (jit+vmap) or CPU (batch sharded across every core with pmap). Swim sims are grouped by body size and run on a node buffer trimmed to 32/64/128/256 (~3x faster on a mixed batch); bucket batches are padded to a coarse size grid so changing counts don't recompile. Node repulsion is the exact all-pairs sum written as two matmuls (`repulsion`), ~3.7x faster than the direct form; at n ≤ 256 that leaves it ~40% of a swim step, so a Barnes-Hut tree wouldn't pay off. Repulsion has a compact cutoff (`R_CUT`), muscle stroke is proportional to each edge's rest length, and each node's mean edge stretch feeds back into the CA (`Genome.S`), so the same controller rules scale with body size and can entrain to the body's own mechanics; `--legacy` restores the older long-range repulsion, fixed stroke and no feedback (needed to replay archives from before). `--drag CT CN CQ` sets the fluid.
- `swim_fast.py` — The swim simulation as a numba kernel for CPU: each sim on its body's exact node and edge count, structure-of-arrays loops padded to multiples of 8 with uint32 gather tables so LLVM vectorises them, one sim per thread task. 11-14x faster per step than the JAX path on 16 threads (0.3 / 0.56 / 1.3 / 3.1 µs per body-step in the 32/64/128/256 buckets). All-pairs repulsion is at the divide-throughput bound and is ~60% of a step at n=180, so `rep_every` refreshes it every few steps and holds it in between. `check` compares against `swim.swim` (the reference), `bench` times it. Also `describe` (net progress and gait descriptors), `settle`, `recording`.
- `swim_render.py` — Video rendering without JAX: matplotlib artists built once per panel and updated per frame; every video is split into frame chunks rendered across all cores and joined with ffmpeg concat (~10x faster than redrawing serially).
- `swim_opt.py` — CMA-ES speed optimisation of the controllers of the fastest `swim_qd` elites, all problems in one batch (drag is per sim). Arms: `base` (original controller), `pos` (+ node position along/across the body axis, `Genome.P`), `pos8` (+ 8 channels). On 8 seeds x 150 gens: mean speed 3.7 -> 5.0 (base), 5.6 (pos), 5.7 (pos8); position input helps most on 34-36 node bodies (2.5-3.4x).
- `swim_grad.py` — Trial of gradient refinement through the simulation (truncated BPTT via `swim(..., truncate_at=)`). Negative result from CMA-ES optima: gradients agree across CA initial states but not with finite differences (the speed landscape is rough), and Adam over 100-step windows always lowered true speed; 400-step windows gave at most +3%.
- `swim_coevo.py` — Bodies and controllers evolved together: MAP-Elites over body size × beat frequency × signed wave number (muscle drive binned along the body axis; < 0 retrograde travelling wave, ~0 pulsing, > 0 direct), fitness = net progress in body lengths per 20 time units (displacement after the transient, so circling scores nothing). Children keep the parent body, regrow a mutated one (GRA params, sometimes a new seed) or borrow another elite's. Rhythm stays emergent from the CA (no clock input); controllers have the position input and 8 channels. Cells also split by rest-shape elongation, and bigger size buckets swim longer per evaluation (x1.6/2.4/3.2). `emit` replaces random mutation with CMA-ME: improvement emitters (CMA-ES per body, ranked by new cells then cell gains), half of them started on newly grown bodies that are ranked by their own speed for a few protected generations so their controllers can catch up with tuned incumbents; warm-started from an earlier archive. Plain MAP-Elites plateaued with only 3-7 distinct bodies per size class. Sims run on `swim_fast` at dt 0.04 with repulsion refreshed every 2 steps (2-3x fewer kernel-seconds per time unit); `verify` re-scores an archive at dt 0.02 with repulsion every step. `emit --regrow` re-lays-out the bodies of an archive from before the repulsion cutoff. Run `net_cma` (900 gens x 2048 swims, ~1.8 h on 16 threads): 588 cells on 194 bodies; best speeds by size class, verified, 2.0 / 1.08 / 0.57 / 0.38 / 0.31 BL per 20 time units (n = 16 / 32 / 56 / 82 / 132). At search settings the typical elite is 1-13% optimistic (worst in the smallest class) and a tenth of them 12-47%; a few small ones only swim from the initial state they were found with. Zeroing the stretch feedback of the final elites cuts the top-10 speed of a size class by 1.3-3x, the position input by 2.4-3.8x.
- `swim_chemo.py` — Gradient following: every node senses the nutrient concentration where it is, as one more additive CA input (`Genome.C`, in `swim.swim(source=)` and the `swim_fast` kernel); the field is a cone, 1 at a point source and 0 at 5 body lengths. CMA-ES per body from the fastest `swim_coevo` elites with `C = 0`; a controller meets 6 sources evenly round the body (2.5 body lengths away, x0.6-1.65 per generation) from one CA initial state, so a sensor-blind swimmer repeats one path, and scores the share of the gap closed, averaged over the trial (`homing`). `verify` re-scores at dt 0.02 on fresh directions and initial states, and with the nodes told only the body's mean concentration, only their difference from it, a uniform field, or nothing (`told`; `sense` weights of `swim_fast.simulate`). Run `homing` (21 bodies, 300 gens x 4032 swims, 1.6 h): verified homing 0.41 (0.07 in a uniform field), 76% of trials end within a body length of the source (3% in a uniform field), 7 of 21 bodies every time and 6 under half the time; the best also follow a source that jumps, which they were not trained on. Steering comes from concentration differences across the body (62% arrive on those alone, 5% on the body mean alone); several swimmers barely move without a gradient. From 4 body lengths only 3% arrive in the trial time (about half given 2.5x as long).
- `swim_pool.py` — Video of the `swim_opt` swimmers growing on a ring in one pool, each setting off when its topology settles. One joint sim (each body keeps its own controller and fluid); nodes are pushed off other bodies' edges (thick-rod contact), and bodies still growing are fixed obstacles drawn as shown.
- `swim_qd.py` — Swimming-style search on the grown bodies: MAP-Elites over body size × path straightness × spin rate, fitness = windowed speed, genomes hop between bodies. Renders a gallery, per-swimmer mp4s and a map per run.

## Running

The environment is managed by uv (`uv sync`), so prefix commands with `uv run`.

```bash
# Grown swimmers
uv run python swim.py grow --gens 40 --batch 768
uv run python swim.py swim --gens 100 --batch 1024
uv run python swim.py render --top 6 --growth
uv run python swim_qd.py search --tag stokes --drag 0.5 1.0 0.0
uv run python swim_qd.py render --tag stokes
uv run python swim_opt.py search --fluids water_sized stokes --per-fluid 2
uv run python swim_opt.py render
uv run python swim_pool.py --tag speed
uv run python swim_coevo.py search --tag water --seed-from qd_water_sized.pkl
uv run python swim_coevo.py emit --tag water_cma --resume coevo_water.pkl
uv run python swim_coevo.py verify --tag water_cma
uv run python swim_coevo.py render --tag water_cma
uv run python swim_chemo.py search --tag homing --from coevo_net_cma.pkl
uv run python swim_chemo.py verify --tag homing
uv run python swim_chemo.py tracks --tag homing
uv run python swim_chemo.py render --tag homing
uv run python swim_fast.py check

# Default search
python run.py

# Quick test
python run.py search.num_generations=10 search.batch_size=32 search.sim_steps=100

# Override params
python run.py search.batch_size=64 search.num_generations=200 wandb.enabled=true

# Replay saved archive
python run.py mode=replay replay_path=archive.pkl

# Disable wandb
python run.py wandb.enabled=false
```

## Key constants (in conf/config.yaml)

- `sim.max_nodes`: Fixed buffer size. 512 is good for fast vmap search; increase for detailed viz runs.
- `sim.max_topo_ops`: Max divisions or collapses per topology step. Higher = faster growth but slower JIT.
- `search.batch_size`: Number of parallel sims per generation. 64 compiles in ~6s on GPU.
- `search.sim_steps`: Timesteps per evaluation. 500-1000 for search, more for visualization.

## Relation to Rust implementation

The Rust version (`../../rust/gra/`) is the interactive real-time version with GPU rendering via wgpu. This Python version strips out physics (spring forces, repulsion, Barnes-Hut) and focuses on parameter space exploration. The CA model (Chebyshev convolution, Gaussian growth, coupling matrix, division/collapse signals) matches the Rust version.
