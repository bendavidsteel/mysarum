# GRA — Graph Cellular Automata Parameter Search

JAX implementation of a Lenia-style graph cellular automaton with Chebyshev spectral convolution, cross-channel coupling, and dynamic topology (division/collapse). Uses MAP-Elites quality-diversity search to find interesting parameter regimes.

## Architecture

- `sim.py` — Core JAX simulation: Chebyshev convolution, growth function, topology ops. All fixed-size buffers for JIT/vmap.
- `search.py` — MAP-Elites search with vmap parallel evaluation, fitness metrics, mutation from archive elites.
- `viz.py` — Post-search visualization: archive heatmaps, graph snapshots with networkx, metric time series.
- `run.py` — Hydra entry point. Config in `conf/config.yaml`.
- `graphs.json` — Seed graphs (10-12 node small graphs from the Rust implementation).
- `swim.py` — Grown swimmers: GRAs grow until their topology stops changing, then a continuous CA on the nodes (channels mix only across edges) drives edge rest lengths in a drag fluid. Subcommands `grow`, `swim`, `render`; outputs in `outputs/swim/`. Runs on GPU (jit+vmap) or CPU (batch sharded across every core with pmap). Swim sims are grouped by body size and run on a node buffer trimmed to 32/64/128/256, since O(n²) repulsion dominates (~3x faster on a mixed batch). `--drag CT CN CQ` sets the fluid.
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
