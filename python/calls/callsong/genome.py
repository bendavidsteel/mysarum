"""Instrument genome: the parameters MAP-Elites varies.

A genome is a flat float vector in *normalised* [0, 1] space (so mutation is
uniform and bounds are trivial to respect). ``decode`` maps it to the physical
instrument parameters consumed by :mod:`callsong.synth`.

The gesture (the time-varying air-sac pressure / syringeal tension path) is NOT
part of the genome — it is a fixed probing apparatus (see
:mod:`callsong.gestures`). The genome is the *instrument*; the gesture *plays*
it. This keeps the two concerns separate, as intended: MAP-Elites illuminates
the space of instruments, each of which is then swept by the same gestures.
"""

from __future__ import annotations

import numpy as np

# name, low, high, mindlin-default (physical units)
#
# The source block is a generalised limit-cycle vector field that reduces to
# the Sitt/Arneodo/Mindlin normal form when the k_* gains are (1,1,1,1,0). The
# filter block is a trachea comb + two state-variable formants + saturation +
# aspiration noise.
PARAM_SPEC: list[tuple[str, float, float, float]] = [
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

NAMES = [s[0] for s in PARAM_SPEC]
LOW = np.array([s[1] for s in PARAM_SPEC], dtype=np.float64)
HIGH = np.array([s[2] for s in PARAM_SPEC], dtype=np.float64)
DEFAULT_PHYS = np.array([s[3] for s in PARAM_SPEC], dtype=np.float64)
N_PARAMS = len(PARAM_SPEC)


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
