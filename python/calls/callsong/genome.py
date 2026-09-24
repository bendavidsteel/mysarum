"""Genome: the parameters MAP-Elites varies — instrument *and* gesture.

A genome is a flat float vector in *normalised* [0, 1] space (so mutation is
uniform and bounds are trivial to respect). ``decode`` maps it to physical
units: the first :data:`N_INSTRUMENT` entries are the syrinx parameters
consumed by :mod:`callsong.synth`, the rest are the CPG parameters that
:mod:`callsong.gestures` turns into the ``(alpha, beta)`` motor path.

The gesture used to be a fixed probing bank held outside the genome, so that
one apparatus swept every instrument. Folding it in instead means a genome is a
whole *performance* — an instrument together with the motor score that plays
it — which is what the descriptor was measuring all along. It also removes the
bank's failure mode, where a couple of gestures won most of the archive because
every instrument had to be judged through the same handful of paths, and it
renders one call per genome instead of ``n_gestures``, so a generation covers
that many more distinct candidates for the same compute.
"""

from __future__ import annotations

import numpy as np

from .gestures import GESTURE_SPEC

# name, low, high, mindlin-default (physical units)
#
# The source block is a generalised limit-cycle vector field that reduces to
# the Sitt/Arneodo/Mindlin normal form when the k_* gains are (1,1,1,1,0). The
# filter block is a trachea comb + two state-variable formants + saturation +
# aspiration noise.
INSTRUMENT_SPEC: list[tuple[str, float, float, float]] = [
    # ── source (labial oscillator) ──────────────────────────────────────────
    ("gamma",      8_000.0, 60_000.0, 24_000.0),  # time constant / spectral range
    ("k_cub",          0.2,      3.0,      1.0),   # cubic restoring   (-g^2 x^3)
    ("k_sq",          -1.0,      2.0,      1.0),   # quadratic asymmetry (+g^2 x^2)
    ("k_sqy",          0.0,      3.0,      1.0),   # x^2 y nonlinear damping
    ("k_xy",           0.0,      3.0,      1.0),   # x y nonlinear damping
    ("k_vdp",          0.0,      1.5,      0.0),   # van der Pol negative damping
    # ── trachea reflection comb ─────────────────────────────────────────────
    ("trachea_ms",    0.05,      4.0,     0.25),   # round-trip delay T (ms)
    ("r",              0.0,     0.95,     0.75),   # beak reflection (comb depth)
    # ── oro-esophageal formant bank (2 state-variable band-passes) ──────────
    ("f1_hz",        300.0,  6_000.0,  2_200.0),   # formant 1 centre
    ("q1",             1.0,     20.0,      2.0),   # formant 1 Q
    ("g1",             0.0,      1.0,      1.0),   # formant 1 gain
    ("f2_hz",        300.0,  8_000.0,  4_000.0),   # formant 2 centre
    ("q2",             1.0,     20.0,      3.0),   # formant 2 Q
    ("g2",             0.0,      1.0,      0.3),   # formant 2 gain
    # ── output shaping ──────────────────────────────────────────────────────
    ("drive",          0.5,      6.0,      2.0),   # tanh saturation drive
    ("noise_gain",     0.0,      0.3,     0.02),   # aspiration noise into source
]

N_INSTRUMENT = len(INSTRUMENT_SPEC)

# The gesture half. GESTURE_SPEC carries no reference value, so the default
# genome takes the midpoint of each range — the Mindlin reference is a
# statement about the instrument, not about any particular motor score.
PARAM_SPEC: list[tuple[str, float, float, float]] = INSTRUMENT_SPEC + [
    (name, lo, hi, 0.5 * (lo + hi)) for name, lo, hi in GESTURE_SPEC
]

NAMES = [s[0] for s in PARAM_SPEC]
LOW = np.array([s[1] for s in PARAM_SPEC], dtype=np.float64)
HIGH = np.array([s[2] for s in PARAM_SPEC], dtype=np.float64)
DEFAULT_PHYS = np.array([s[3] for s in PARAM_SPEC], dtype=np.float64)
N_PARAMS = len(PARAM_SPEC)
N_GESTURE = N_PARAMS - N_INSTRUMENT

INSTRUMENT = slice(0, N_INSTRUMENT)   # phys[..., INSTRUMENT] -> synth
GESTURE = slice(N_INSTRUMENT, None)   # phys[..., GESTURE]    -> gestures


def decode(genome: np.ndarray) -> np.ndarray:
    """Normalised [0,1] vector(s) -> physical parameters (same trailing shape)."""
    g = np.clip(np.asarray(genome), 0.0, 1.0)
    return LOW + g * (HIGH - LOW)


def encode(phys: np.ndarray) -> np.ndarray:
    """Physical parameters -> normalised [0,1] vector(s)."""
    return np.clip((np.asarray(phys) - LOW) / (HIGH - LOW), 0.0, 1.0)


def mindlin_genome() -> np.ndarray:
    """The normalised genome reproducing the Rust reference syrinx."""
    return encode(DEFAULT_PHYS)


def random_genomes(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` uniform random genomes in normalised space, shape (n, N_PARAMS)."""
    return rng.random((n, N_PARAMS))


def mutate(rng: np.random.Generator, genomes: np.ndarray, sigma: float = 0.12) -> np.ndarray:
    """Isotropic Gaussian mutation in normalised space with reflecting bounds."""
    g = genomes + rng.normal(0.0, sigma, size=genomes.shape)
    # reflect into [0,1] so density stays well-behaved at the edges
    g = np.abs(g)
    g = 1.0 - np.abs(1.0 - g)
    return np.clip(g, 0.0, 1.0)


def crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Uniform crossover between two parent genome batches."""
    mask = rng.random(a.shape) < 0.5
    return np.where(mask, a, b)
