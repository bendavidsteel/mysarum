# Growth — GPU differential growth of sheets and shells

Thin elastic surfaces that grow, buckle and fold, simulated on the GPU with
PyTorch + NVIDIA Warp (package `dgrow/`). On top of the simulator:
quality-diversity search (`qd.py`) and target imitation (`imitate.py`), both
judged from renders by image models.

## Engine (`dgrow/`)

- `mesh.py` — plain `pos (N,3)` + `faces (F,3)`; `Topology` rebuilds edges,
  hinges, boundary and CSR adjacency from faces with sorts. Rest lengths live on
  undirected edges keyed `min<<32 | max`. Edge splits and intrinsic Delaunay
  flips run in parallel on an independent edge set (no face touches two picked
  edges). Builders: jittered hex `make_disc`, `make_icosphere`, `make_hemisphere`.
- `sim.py` — `Sim` holds a batch of individuals as one disjoint mesh; every
  per-individual parameter (`Params`) is read through `ind[v]`. Explicit damped
  dynamics in Warp kernels: strain springs, Bridson dihedral bending (rest angle
  = `curl · edge`), hash-grid repulsion between non-adjacent vertices, radial
  foundation, normal pressure, gravity, ground plane.
- Growth: per-vertex rate `g = base + edge·exp(-d/λ) + curv·H + morph·V +
  height·tanh(z/10)`, smoothed on the graph, then applied as a per-face growth
  tensor to each rest triangle (`aniso` grows along/across ∇d). Mapping whole
  triangles keeps the rest metric realisable; per-edge rates tore the mesh.
  `d` is fast-marching geodesic distance from the boundary or source vertices.
- Gray-Scott morphogens (`U`, `V`) run on the normalised graph Laplacian when
  `SimConfig.morph_steps > 0`.
- Individuals that go non-finite are quarantined (frozen), not fatal.
- `render.py` — Warp BVH ray tracer: AO, soft key-light shadow, two-tone
  front/back faces. `views()` = iso/side/top strip, `turntable()` = mp4/gif.
- `presets.py` — seed meshes + hand-built presets after real organisms
  (lettuce, kale, saddle, hyperbolic, trumpet, brain, turing, coral).
- `genome.py` — 18 genes in [0,1] → (seed shape, `Params`); gated genes switch
  mechanisms fully off in the lower part of their range.
- `evaluate.py` — grow a batch of genomes until each hits its vertex budget,
  relax, render iso + top.
- `judge.py` — DINOv2-small embeddings (descriptors) and CLIP ViT-L/14
  organism-vs-decoy probabilities (fitness, imitation targets).

Units: lengths are in initial edge lengths (split at 1.4, floor 0.4).

## Scripts

```bash
# Presets side by side (overrides after ':'), montage in --out
uv run python explore.py brain coral lettuce:k_bend=6 --out outputs/explore

# CVT-MAP-Elites, one grower process per GPU, judge in the main process
uv run python qd.py out=outputs/qd max_hours=10.5

# Evolve toward a text target or a folder of photos, seeded from a QD run
uv run python imitate.py "a nudibranch sea slug" --qd outputs/qd
uv run python imitate.py --refs ~/photos/morels --qd outputs/qd

# The 12 h chain used for the first big run (QD then four imitation targets)
./run_overnight.sh
```

QD writes `evals.parquet` (every evaluation), `embeddings.npz`,
`archive.json` (cell → id, fitness), `imgs/<id>.jpg`, `meshes/<id>.npz`, and
`top.png` / `diverse.png` montages. The descriptor PCA is refit on the whole
history every `refit_every` evals and the archive rebuilt.

`printability.py` (overhang / layers / support volume, pure NumPy) is not
wired in yet; it is meant as an extra QD fitness term for 3D-printable forms.

## Environment

Runs on a remote 2-GPU box (RTX 3090s); code is rsynced there from this
repo. uv only. Its driver is CUDA 12.5, so `pyproject.toml`
pins torch 2.6.0 from the cu124 index and warp-lang 1.9.1 (newer warp wheels
need a CUDA 13 driver). Models download from Hugging Face on first use; set
`HF_HUB_OFFLINE=1` for long runs.

## Tuning notes

- Fold scale is set by bending vs stretching (`k_bend / k_stretch`) and by
  growth speed vs relaxation: slow growth gives smooth large folds, fast growth
  crinkles at mesh scale.
- Positive curvature feedback (`g_curv`) needs `curv_smooth` / `g_smooth`
  diffusion or it spikes at mesh scale.
- Remeshing dominates step cost; `SimConfig.remesh_every` trades it off.
- CLIP fitness favours clay-coloured wrinkled forms (brain, walnut); colour
  matters to it, which is why green targets like lettuce score badly on clay.

Keep a minimal set of scripts; parameterize via hydra/argparse, use git for
versions.
