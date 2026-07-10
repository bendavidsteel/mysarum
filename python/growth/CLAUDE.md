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

**Anisotropic growth** is orthogonal to `growth_mode` — `_anisotropy_factor`
scales every half-edge's length gain by `mix(1, |cos θ|, anisotropy_strength)`,
where θ is the angle between the edge's 3D direction and `params.anisotropy_dir`.
`anisotropy_strength=0` is exactly isotropic (factor 1.0 everywhere, no behaviour
change); 1.0 grows only along the axis. Unlike the Rust port's three cardinal-axis
choices, `anisotropy_dir` is a free vec3 (normalized internally), so oblique /
diagonal preferred directions are possible. Applied inside all three
`grow_intrinsic_lengths*` functions. See `conf/sweep/anisotropic.yaml`.

## Coral-inspired additions (ported from joel-simon/coral-growth)

Four mechanisms make the phototropic mode branch / diversify / smooth like
Simon's evolved corals. All default to **off** (exact old behaviour) and are
demoed in `conf/sweep/coral.yaml`. They only fire in `growth_mode=phototropic`.

1. **Self-shadowing occlusion** (`compute_occlusion`, `occlusion_strength`>0):
   a vertex loses light for each nearby vertex sitting toward the light source
   (within a cone `occlusion_cone`, radius `occlusion_radius` or repulsion_dist).
   Reuses the collision spatial-hash candidates (so a vertex isn't shaded by its
   own 1-ring). This is the branching driver: a bump shades its neighbours, they
   stop growing, the bump elongates. Multiplies `ch1` light each substep.
2. **Growth-field shaping** (in `grow_intrinsic_lengths_phototropic`):
   `growth_budget_gate`∈[0,1] gates growth by the (occluded) light channel so lit
   tips outcompete shaded valleys; `growth_field_smooth`∈[0,1] does a 1-ring
   average of the per-vertex growth signal before it's injected (smoother folds,
   `_smooth_vertex_field`).
3. **Gray-Scott morphogens** (`gray_scott_step`/`run_morphogens`,
   `morphogen_steps`>0, needs `state_dims>=5`): reaction-diffusion on the mesh
   graph in `ch3`(U)/`ch4`(V); `morphogen_coupling` modulates the tissue growth
   signal by V (`tissue *= 1 + coupling·(2V−1)`) → Turing-patterned growth zones.
   With morphogens on, `ch3/ch4` are NOT treated as diffusing resources; `ch2`
   (+`ch5+`) are. Seed with `seed_morphogens` (U=1, random V spots).
4. **Normal-displacement inflation blend** (`growth_mode_mix`∈[0,1],
   `inflation_strength`): adds a normal-directed outward *force*
   `mix·strength·tissue·n` in the predict step. NB it is a force, not hard
   displacement — the Rust port found plastic capture fails against stiff
   springs (see [[project_floraform_growth_alternatives]]); XPBD balances it into
   rounder lobes.

## MAP-Elites diversity + printability search

`map_elites.py` (hydra `config_name=map_elites`) illuminates a 2D archive whose
axes are **shape** (compactness = SA/V^(2/3) × print height) and whose per-cell
fitness is **printability** — each cell keeps the most printable genome that
lands in it. Genotype = ~18 continuous environmental genes + the
normal-displacement growth mix; fixed to phototropic hemisphere, `state_dims=5`
(morphogens on). Reuses `nca_sweep.run_one(..., do_descriptors=False)`.

Printability metrics live in `printability.py` (`print_metrics`,
`printability_penalty`), pure NumPy on the extracted mesh. Build direction is
**+z** (hemisphere flat side down): `overhang_fraction` = area-weighted faces
with `n·(+z) < −cos(45°)`; plus `layers` (height/layer_height), `support_volume`
(Σ overhang projected-area·height), `surface_area`. Fitness = weighted,
reference-normalised sum (minimised); weights in `conf/map_elites.yaml`.

```bash
# First run (4 GB box): archive.npz + archive.png montage + elites.json
GROWTH_MAX_VERTICES=4000 conda run -n base python map_elites.py
# Bigger budget / resume from a checkpoint archive:
conda run -n base python map_elites.py n_evals=400 sigma=0.1
conda run -n base python map_elites.py resume_from=outputs/<run>/archive.npz
```

## Run

Always use the base conda env: `conda run -n base python <script>`.

```bash
# Phototropic sweep (default 25 configs, hemisphere only):
conda run -n base python nca_sweep.py sweep=phototropic

# Other sweeps:
conda run -n base python nca_sweep.py sweep=bipolar
conda run -n base python nca_sweep.py sweep=stratified
conda run -n base python nca_sweep.py sweep=original
conda run -n base python nca_sweep.py sweep=anisotropic

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

## Fold smoothness — growth speed, NOT bending compliance

Mesh smoothness is governed by **how fast the surface grows relative to how
fast XPBD relaxes it**, not by the bending constraint. Differential growth
injects excess area; XPBD absorbs it by buckling. Inject slowly → the mesh
stays near mechanical equilibrium → smooth folds. Inject fast → it buckles
into sharp creases before the solver catches up. The lever is
`growth_rate × state_dt` (growth per substep):

- Quasi-static (smooth, ~22° mean dihedral): `growth_rate≈1.5, state_dt≈0.04`.
  Used by `anisotropic*.yaml`. Mirrors the Rust port's `0.3 / 0.02`.
- Fast (sharp, ~58°): `growth_rate=6.0, state_dt=0.08`. The old phototropic
  sweep tuned these up to reach big meshes in fewer frames — at the cost of
  sharp buckling. Smooth growth needs ~2.5× more frames for the same size,
  and the 20-iteration solver roughly doubles per-frame cost, so budget a
  smooth 4500-vert config at ~15 min (a full 112-config sweep ≈ a day, not 6h).

Things that do **not** smooth the folds (measured): bending compliance
(invariant ~57°, and below ~1e-4 it overshoots *sharper*), Laplacian smoothing
(`stiffness`), tessellation (`max_edge_len`), more XPBD iterations alone.

Two solver params (ported from Rust) make the XPBD relaxation effective:
- `MeshParams.relaxation` (default 0.7): SOR under-relaxation on every spring +
  bending correction. <1 damps the full-Jacobi overshoot; 1.0 = old un-damped
  behaviour (why stiff bending used to oscillate into *worse* sharpness).
- `XPBD_ITERATIONS` (now 20, was 10): more relaxation passes per substep.

Bending α̃ is `bending_compliance / dt²` here vs the Rust port's direct α̃; the
Python default `0.001` (dt 0.02) and Rust's `2.5` are the same effective value.

The `JAXNvdiffrastRenderer` auto-fits an orthographic camera to the live mesh
bbox each render (fills ~85% of frame), so grown meshes stay framed regardless
of size — the old fixed `[0,width]×[0,height]` window clipped them to close-ups.

## Files

- `growth_halfedge_jax.py` — half-edge state, mesh builders, XPBD, all four
  growth modes, the four coral-inspired mechanisms above, nvdiffrast renderer.
- `map_elites.py` — MAP-Elites diversity+printability search (hydra
  `config_name=map_elites`, config `conf/map_elites.yaml`).
- `printability.py` — pure-NumPy print metrics (overhang / layers / support /
  area) + weighted penalty, for the MAP-Elites fitness.
- `nca_sweep.py` — hydra entry point that runs a list of configs, renders
  each, computes shape descriptors, writes a farthest-point gallery + Vendi
  scores.
- `nca_shape_descriptors.py` — descriptor + Vendi + diversity-pick helpers.
- `conf/config.yaml`, `conf/sweep/*.yaml` — hydra config groups. Add a new
  sweep by dropping a YAML in `conf/sweep/` and running `sweep=<name>`.
- `render_floraform.py` — single-config offline render to PNG/MP4.

Do not create v1/v2 scripts — keep a minimal set, parameterize via hydra,
use git for versioning.
