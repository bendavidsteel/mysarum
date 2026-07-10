# callsong

Generative animal-call synthesis: a configurable **nonlinear syrinx** rendered
in JAX, whose instrument space is **illuminated by MAP-Elites** using the first
two PCA axes of **BirdNET embeddings** as behaviour descriptors.

The premise (after Mindlin, *The physics of birdsong production*): the syrinx is
a low-dimensional nonlinear oscillator driven by a slow motor gesture
(air-sac pressure `α`, syringeal tension `β`). Its bifurcations — onset,
harmonic stacks, subharmonics, period-doubling, chaos — are exactly what make a
call sound alive. Rather than backpropagate a neural ODE at sample rate, we
keep the model **forward-only** and search it evolutionarily.

## Pipeline

```
genome (instrument) ──┐
                      ├─► JAX render ─► BirdNET embed ─► PCA(2) ─► archive cell
gesture bank (fixed) ─┘        │                                      ▲
                               └─► acoustic fitness ──────────────────┘
                                   (nonlinear richness, anti-degeneracy)
```

- **`callsong/synth.py`** — generalised limit-cycle source + trachea reflection
  comb + 2-formant state-variable band-pass bank + aspiration noise + tanh
  saturation. One `lax.scan`, oversampled, `vmap`-ed over the population. The
  default genome reproduces the Rust reference (`rust/birdsong`); genome gains
  `k_cub, k_sq, k_sqy, k_xy, k_vdp` deform the vector field, moving its
  bifurcation structure.
- **`callsong/gestures.py`** — a fixed, seeded bank of `(α, β)` paths probing
  each instrument's phase transitions: one pressure ramp, `gesture_sine`
  sinusoidal trill/warble probes (repetitive calls — α oscillates through the
  onset so the call breaks into syllables), and random cubic splines for the
  rest. The gesture is a *probe*, not part of the genome.
- **`callsong/genome.py`** — 16-parameter instrument genome in normalised
  `[0,1]` space (sample / mutate / crossover / decode).
- **`callsong/features.py`** — fitness = audibility · tonality ·
  (spectral-contrast + spectral-flux). Rewards structured nonlinearity while
  punishing the degenerate corners (silence, white noise).
- **`callsong/birdnet.py`** — BirdNET 1024-D embeddings (via `tensorflow-cpu`,
  off the GPU) + the 2-D PCA descriptor projector.
- **`callsong/archive.py`** — MAP-Elites grid over the PCA descriptor space.

## Run

```bash
# hear the physics: an α ramp through the phonation onset at 3 tensions
uv run python scripts/demo_bifurcation.py            # -> _demo/*.wav, *.png

# illuminate the instrument space
uv run python run_mapelites.py                       # -> outputs/<ts>/
uv run python run_mapelites.py generations=300 offspring=32 archive_resolution=32

# (optional) ground the descriptor axes on REAL recordings, then use them
uv run python scripts/fit_projector.py --audio_dir ~/sounds --out projector.npz
XC_API_KEY=... uv run python scripts/fit_projector.py --xc \
    --groups birds,frogs,grasshoppers --per_group 80 --out projector.npz
uv run python run_mapelites.py projector_path=projector.npz
```

Each run writes to `outputs/<timestamp>/`: `archive.npz`, `projector.npz`,
`archive_scatter.png`, `top_calls.png`, and `calls/*.wav` (the highest-fitness
elite per top cell, re-rendered).

## Notes / knobs

- **Throughput.** The `scan` is sequential in time (~4 s wall for a 3-s call at
  48 kHz, OS=6) but parallel across the batch, so a whole generation renders in
  one `vmap`. BirdNET (~77 ms/call, CPU) is the real bottleneck.
- **Descriptor grounding.** PCA is fit on a bootstrap random population by
  default. `scripts/fit_projector.py` instead fits it on real recordings —
  a local folder and/or Xeno-canto across taxa (birds + frogs + insects), so
  the axes span a wide cross-taxa timbral gamut. Xeno-canto API v3 needs a
  personal `XC_API_KEY` (required for downloads since 2025-10-10). BirdNET is
  bird-trained but still embeds any audio, so frog/insect clips project fine.
- **GTX 1650 (4 GB):** `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `MEM_FRACTION≈.6` (set by the scripts); BirdNET stays on `tensorflow-cpu`.
- Fitness weights, archive resolution, gesture count, and generation budget are
  all in `conf/config.yaml` (Hydra overrides on the CLI).
