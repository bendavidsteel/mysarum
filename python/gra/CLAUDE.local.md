# Local environment

## Python
- Use system conda python: `/home/ben/miniconda3/bin/python`
- Do NOT use the `.venv` directory (it lacks JAX)
- Run scripts directly with `python`, not `uv run`

## Key dependencies (conda base)
- JAX 0.4.30 with CUDA 12 (cuda:0)
- numpy 2.2, matplotlib 3.10, networkx 2.8
- hydra-core 1.3, omegaconf 2.3
- wandb 0.23
- tqdm 4.65

## Notes
- JIT compilation of vmap'd simulation takes ~6s on first run
- Hydra changes cwd to `outputs/<date>/<time>/` — output files land in the original dir via relative paths
- `np.alltrue` shim needed for networkx compat with numpy 2.0 (in viz.py)
