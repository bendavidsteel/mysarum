"""Acoustic fitness — reward nonlinear richness, punish degeneracy.

Fitness in MAP-Elites only decides which call wins a *given* cell; the spread
across cells is the descriptor's job. So the reward terms are optional, and
turning them off is a real design choice rather than a degenerate one: a
reward that prefers one kind of call will fill every cell with the nearest
thing to that call, working against the descriptor. Set ``base`` to 1 and the
reward weights to 0 for gates-only fitness — every audible, tonal call is then
equally acceptable and the archive keeps whatever landed in each cell first.

"Reward nonlinearity" naively maximised (e.g. raw spectral entropy) collapses
to white noise, which is nonlinear-looking but sounds terrible; pure silence or
a pure sine are the other degenerate corners. We therefore combine:

  * audibility gate   — kills silence (low RMS),
  * tonality          — (1 - spectral flatness), kills broadband hiss,
  * spectral contrast — rewards structured partials (harmonics / subharmonics),
  * spectral flux     — frame-to-frame change of the *normalised* spectrum.
                        Intended as "dynamism", but measured against test
                        signals it scores white noise at 0.53 and a tonal FM
                        sweep at 0.08 — a noise spectrum jitters every frame,
                        a glide does not. It is closer to a noise detector
                        than a liveliness one, so weight it with care: it
                        pulls the archive towards hiss.

The behaviour *descriptor* (where a call lands in the archive) is the BirdNET
embedding; fitness only decides which call wins a given cell.
"""

from __future__ import annotations

import numpy as np
import librosa

_N_FFT = 1024
_HOP = 256
_FRAME_FLOOR = 0.05   # frames below this fraction of the loudest are silence

# Tonality anchors. Raw (1 - flatness) sits between ~0.34 and 1.0 for calls this
# model produces, so used directly as a multiplier it varies only ~3x and barely
# reorders anything — noisy calls lose almost nothing. Mapping that observed
# range onto [0, 1] makes the same measurement discriminate: a hissy call drops
# to near zero instead of to two thirds. _TONAL_FLOOR then rejects outright
# anything below it, so a cell that only ever gets noise stays empty rather than
# being filled with the best noise that happened to reach it.
_TONAL_LO = 0.30
_TONAL_FLOOR = 0.55

# Unforced-phonation gate. The gesture drives the syrinx only during the phrase
# window; outside it the pressure is held below the phonation onset, so a
# physically honest instrument is silent there. An instrument sitting on a limit
# cycle that does not depend on the gesture — a large k_vdp, say — sings right
# through the gap, and this is what scores that: the ratio of un-driven to
# driven energy. At or above _UNFORCED_TOL the call scores zero however good it
# sounds, which lets the search keep the van der Pol term for the timbres it
# unlocks while rejecting the settings that decouple sound from gesture.
_UNFORCED_TOL = 0.15

# Empirical anchors for the two reward terms (scripts/calibrate_fitness.py).
# Both are mapped affinely onto [0, 1] rather than divided by a round number:
# a nominal scale either compresses the term into a narrow band or — as the
# original flux x20 did — saturates it at 1.0 for every call, silently reducing
# fitness to the other term alone.
#
# The low anchors come from a random population, the high anchors from the
# elites of a completed run *plus headroom*. Fitting the top to random calls is
# not enough: MAP-Elites hunts the extremes, so elites reach 24.6 dB contrast
# and 0.38 flux where random draws stop at 19.7 dB and 0.34, and anchors fitted
# to the random range left a third of the archive tied at the ceiling with no
# selection pressure left between those cells. Re-run the script (against a run
# directory) if the source model, gesture bank or STFT settings change.
_CONTRAST_LO, _CONTRAST_HI = 9.0, 25.0    # dB
_FLUX_LO, _FLUX_HI = 0.03, 0.40


def _unforced_gate(w: np.ndarray) -> tuple[float, float]:
    """(gate, ratio) for phonation during the un-driven part of the window.

    ``ratio`` is un-driven rms over driven rms; ``gate`` falls linearly from 1
    to 0 as that ratio approaches :data:`_UNFORCED_TOL`.
    """
    from . import gestures

    n = len(w)
    lo, hi = int(gestures.PHRASE_LO * n), int(gestures.PHRASE_HI * n)
    if lo <= 0 or hi >= n or hi <= lo:
        return 1.0, 0.0                      # no un-driven window to judge
    driven = float(np.sqrt(np.mean(w[lo:hi] ** 2)))
    quiet = np.concatenate([w[:lo], w[hi:]])
    undriven = float(np.sqrt(np.mean(quiet ** 2)))
    if driven <= 1e-9:
        return 0.0, 1.0                      # nothing during the phrase at all
    ratio = undriven / driven
    return float(np.clip(1.0 - ratio / _UNFORCED_TOL, 0.0, 1.0)), ratio


def compute(wave: np.ndarray, sr: int, weights: dict | None = None) -> tuple[float, dict]:
    """Return (fitness, feature_dict) for a mono waveform."""
    w = np.asarray(wave, dtype=np.float32)
    weights = weights or {}
    # Defaults match conf/config.yaml: gates only. Keep the two in step so a
    # bare compute() reports the same fitness the search is actually using.
    w_base = weights.get("base", 1.0)
    w_tonal = weights.get("tonal", 0.0)
    w_contrast = weights.get("contrast", 0.0)
    w_flux = weights.get("flux", 0.0)
    rms_floor = weights.get("rms_floor", 1e-3)
    rms_ref = weights.get("rms_ref", 0.1)

    rms = float(np.sqrt(np.mean(w**2)) + 1e-12)
    audible = float(np.clip((rms - rms_floor) / (rms_ref - rms_floor), 0.0, 1.0))
    unforced, driven = _unforced_gate(w)

    S = np.abs(librosa.stft(w, n_fft=_N_FFT, hop_length=_HOP)) + 1e-9

    # Score the call, not the silence around it. Gestures are gated to a phrase
    # in the middle of the window, so most frames are empty; near-zero frames
    # read as broadband and would drag tonality down in proportion to how much
    # silence a call has, penalising short calls for being short.
    frame_e = S.sum(axis=0)
    loud = frame_e > _FRAME_FLOOR * frame_e.max()
    if loud.sum() >= 2:
        S = S[:, loud]

    # Spectral flatness in [0,1]: ~1 white noise, ~0 tonal.
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    tonal_raw = 1.0 - flat
    tonal = float(np.clip((tonal_raw - _TONAL_LO) / (1.0 - _TONAL_LO), 0.0, 1.0))
    if tonal_raw < _TONAL_FLOOR:
        tonal = 0.0

    # Spectral contrast (dB) averaged over sub-bands and time -> structure.
    contrast = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
    contrast_n = float(np.clip((contrast - _CONTRAST_LO)
                               / (_CONTRAST_HI - _CONTRAST_LO), 0.0, 1.0))

    # Spectral flux: mean positive frame-to-frame change of the normalised
    # magnitude spectrum -> dynamism (onsets, jumps, period doubling). Each
    # column of Sn sums to 1, so the per-frame value is already in [0, 1].
    Sn = S / (np.sum(S, axis=0, keepdims=True) + 1e-9)
    flux = np.maximum(np.diff(Sn, axis=1), 0.0).sum(axis=0)
    flux_n = float(np.clip((np.mean(flux) - _FLUX_LO)
                           / (_FLUX_HI - _FLUX_LO), 0.0, 1.0))

    fitness = audible * tonal * unforced * (w_base + w_tonal * tonal
                                            + w_contrast * contrast_n
                                            + w_flux * flux_n)

    feats = dict(rms=rms, audible=audible, flatness=flat, tonal=tonal,
                 tonal_raw=tonal_raw,
                 contrast=contrast_n, flux=flux_n, unforced=unforced,
                 unforced_ratio=driven, fitness=fitness)
    return float(fitness), feats
