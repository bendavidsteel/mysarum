"""Gesture bank — the bird's motor score.

A gesture is a smooth path (alpha(t), beta(t)) through the phonating region of
parameter space: alpha ~ air-sac pressure, beta ~ syringeal tension. We use a
small, FIXED, seeded set of random cubic splines shared across every instrument
so that the same probing motions sweep each candidate. The splines deliberately
range across the phonation onset (alpha=0) and the full tension axis so that a
single instrument is driven through its bifurcations / phase transitions.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

# Phonating region (from the Rust reference model). Below alpha~0 the labia do
# not oscillate, so gestures that dip left of it produce silences / onsets.
ALPHA_MIN, ALPHA_MAX = -0.05, 0.25
BETA_MIN, BETA_MAX = -0.10, 0.40


def _one_spline(rng: np.random.Generator, n_ctrl: int, n_samples: int,
                lo: float, hi: float) -> np.ndarray:
    """A single cubic-spline curve through n_ctrl random control points."""
    t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
    y_ctrl = rng.uniform(lo, hi, size=n_ctrl)
    cs = CubicSpline(t_ctrl, y_ctrl, bc_type="clamped")
    t = np.linspace(0.0, 1.0, n_samples)
    return np.clip(cs(t), lo, hi)


def _one_sine(rng: np.random.Generator, n_samples: int, sr: float,
              duration: float) -> tuple[np.ndarray, np.ndarray]:
    """A trill / repetitive-call gesture: sinusoidally modulated (alpha, beta).

    The pressure (alpha) oscillates around a voiced mean with an amplitude that
    periodically dips toward / below the phonation onset, so the call breaks
    into repeated syllables; the tension (beta) warbles for pitch modulation.
    A slow envelope (raised-cosine attack/decay) shapes the whole phrase.
    """
    t = np.linspace(0.0, duration, n_samples)
    rate = rng.uniform(2.0, 12.0)          # syllable / warble rate (Hz)
    phase = rng.uniform(0.0, 2 * np.pi)

    a_mean = rng.uniform(0.05, 0.18)
    a_amp = rng.uniform(0.06, 0.16)        # large enough to graze the onset
    alpha = a_mean + a_amp * np.sin(2 * np.pi * rate * t + phase)

    b_mean = rng.uniform(0.0, 0.30)
    b_amp = rng.uniform(0.02, 0.15)
    b_rate = rate * rng.choice([0.5, 1.0, 2.0])   # harmonically related warble
    beta = b_mean + b_amp * np.sin(2 * np.pi * b_rate * t + phase)

    return (np.clip(alpha, ALPHA_MIN, ALPHA_MAX).astype(np.float32),
            np.clip(beta, BETA_MIN, BETA_MAX).astype(np.float32))


def make_gesture_bank(seed: int, n_gestures: int, n_samples: int,
                      n_ctrl: int = 5, n_sine: int = 0,
                      sr: float = 48_000.0) -> np.ndarray:
    """Fixed bank of gestures.

    Returns array (n_gestures, 2, n_samples): channel 0 = alpha(t),
    channel 1 = beta(t), sampled at the audio rate.

    Composition: gesture 0 is a slow diagonal pressure ramp (onset -> sustain
    -> offset sweep); the next ``n_sine`` are sinusoidal trill/warble gestures
    that probe repetitive calls; the remainder are random cubic splines. All
    are generated from ``seed`` so the bank is identical for every instrument
    and across runs.
    """
    rng = np.random.default_rng(seed)
    gestures = np.empty((n_gestures, 2, n_samples), dtype=np.float32)
    duration = n_samples / sr

    # Gesture 0: pressure ramp crossing the onset + rising tension.
    t = np.linspace(0.0, 1.0, n_samples)
    gestures[0, 0] = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN)
                      * np.clip(1.5 * t, 0.0, 1.0)).astype(np.float32)
    gestures[0, 1] = (BETA_MIN + (BETA_MAX - BETA_MIN) * t).astype(np.float32)

    n_sine = min(n_sine, max(0, n_gestures - 1))
    for i in range(1, 1 + n_sine):
        gestures[i, 0], gestures[i, 1] = _one_sine(rng, n_samples, sr, duration)

    for i in range(1 + n_sine, n_gestures):
        gestures[i, 0] = _one_spline(rng, n_ctrl, n_samples, ALPHA_MIN, ALPHA_MAX)
        gestures[i, 1] = _one_spline(rng, n_ctrl, n_samples, BETA_MIN, BETA_MAX)

    return gestures
