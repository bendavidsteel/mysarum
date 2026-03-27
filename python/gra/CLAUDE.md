# GRA — Graph Cellular Automata Parameter Search

JAX implementation of a Lenia-style graph cellular automaton with Chebyshev spectral convolution, cross-channel coupling, and dynamic topology (division/collapse). Uses MAP-Elites quality-diversity search to find interesting parameter regimes.

## Architecture

- `sim.py` — Core JAX simulation: Chebyshev convolution, growth function, topology ops. All fixed-size buffers for JIT/vmap.
- `search.py` — MAP-Elites search with vmap parallel evaluation, fitness metrics, mutation from archive elites.
- `viz.py` — Post-search visualization: archive heatmaps, graph snapshots with networkx, metric time series.
- `run.py` — Hydra entry point. Config in `conf/config.yaml`.
- `graphs.json` — Seed graphs (10-12 node small graphs from the Rust implementation).

## Running

```bash
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
