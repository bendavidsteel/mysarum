# Floraform growth — Python/JAX

Differential growth on an intrinsic half-edge mesh with XPBD. Three growth
rules share the same physics layer, picked by `growth_mode`:

| `growth_mode`   | drives growth from                                                       |
| --------------- | ------------------------------------------------------------------------ |
| `lenia`         | Chebyshev graph CA on one channel                                        |
| `nca`           | untrained MLP rule, sigmoid(ch0) → one-sided growth                      |
| `nca_bipolar`   | same MLP, tanh(ch0) → signed growth (allows folding/buckling)            |
| `phototropic`   | deterministic: ch0=tissue, ch1=light from normals, ch2=nutrient + decay  |

The phototropic mode is a hand-tuned, nature-inspired alternative to NCA: no
MLP, fewer parameters, diversity comes from light direction, gravity, and
resource diffusion/decay. See `growth_halfedge_jax.py` "# ── Phototropic"
section. Channels per the convention:
- `ch0` tissue (drives `grow_intrinsic_lengths_phototropic`, in [0, 1])
- `ch1` light (recomputed each substep from vertex normals · `light_dir` + 1/r falloff to `light_pos`)
- `ch2..N` resource channels (graph diffusion + decay, refreshed each substep
  on vertices within `0.5 · spring_len` of the ground plane)

A hemisphere mesh (`make_hemisphere`) sits on the ground plane (`z =
ground_z`). Boundary half-edges around the equator are detected by topology
(`get_on_boundary`) and pinned to `z = ground_z` each XPBD substep via
`project_ground`. Gravity is a constant `-z` force per vertex in
`compute_external_forces`, magnitude `params.gravity_strength`.

## Run

Always use the base conda env: `conda run -n base python <script>`.

```bash
# Phototropic sweep (default 25 configs, hemisphere only):
conda run -n base python nca_sweep.py sweep=phototropic

# Other sweeps:
conda run -n base python nca_sweep.py sweep=bipolar
conda run -n base python nca_sweep.py sweep=stratified
conda run -n base python nca_sweep.py sweep=original

# Compare against a previous sweep's features.npz:
conda run -n base python nca_sweep.py sweep=phototropic \
    compare_to=outputs/20260524-132644_bipolar/features.npz

# Override frames / resolution / max_splits etc:
conda run -n base python nca_sweep.py sweep=phototropic \
    frames=200 resolution=512 max_splits=30
```

Each run writes `outputs/<timestamp>_<sweep>/`:
- `imgs/*.png` rendered tiles (gitignored under outputs/*/imgs)
- `features.npz` shape-descriptor matrix + tags
- `index.json` per-config metadata + growth curves
- `gallery.png` farthest-point-picked diversity subset
- `nca_sweep.log` per-config Vendi numbers
- `resolved_config.yaml` what hydra actually ran with

`nca_shape_descriptors.py` produces the per-mesh feature vector (length 32:
15 cotangent-Laplace-Beltrami spectrum + Gaussian/mean curvature moments +
PCA-ratio anisotropy + surface-area/volume^(2/3) compactness + log-scale
features). `vendi_score` is exp-entropy of the normalized RBF-kernel
eigenvalues; that's how we judge whether a sweep is *actually* diverse vs
visually-different-but-statistically-equivalent (this distinction bit us:
the earlier `nca` hand-picks had Vendi ≈ random sampling within the same
basin; `nca_bipolar` was the basin-break).

## Tunables

- `MAX_VERTICES` (top of `growth_halfedge_jax.py`): the JIT-static array
  size. On the 4 GB local box this stays at 12000. On a bigger GPU bump it
  (and `MAX_HALF_EDGES = MAX_VERTICES * 6`, `MAX_FACES = MAX_VERTICES * 2`)
  to run longer sweeps with higher final vert counts.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set inside `nca_sweep.py` and
  `render_floraform.py` for the 4 GB local box. Leave as-is or unset to
  preallocate on hosts with plenty of VRAM.
- `MeshParams.ground_pin_strength` (default 0.15 = within `0.15·spring_len`)
  is a fallback z-threshold pin used only by callers that don't pass the
  topology-based mask; the runtime path in `batched_physics_step` uses the
  topology `get_on_boundary` mask instead and ignores this field.

## Files

- `growth_halfedge_jax.py` — half-edge state, mesh builders, XPBD, all four
  growth modes, nvdiffrast renderer.
- `nca_sweep.py` — hydra entry point that runs a list of configs, renders
  each, computes shape descriptors, writes a farthest-point gallery + Vendi
  scores.
- `nca_shape_descriptors.py` — descriptor + Vendi + diversity-pick helpers.
- `conf/config.yaml`, `conf/sweep/*.yaml` — hydra config groups. Add a new
  sweep by dropping a YAML in `conf/sweep/` and running `sweep=<name>`.
- `render_floraform.py` — single-config offline render to PNG/MP4.

Do not create v1/v2 scripts — keep a minimal set, parameterize via hydra,
use git for versioning.
