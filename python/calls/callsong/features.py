"""Acoustic fitness — reward nonlinear richness, punish degeneracy.

"Reward nonlinearity" naively maximised (e.g. raw spectral entropy) collapses
to white noise, which is nonlinear-looking but sounds terrible; pure silence or
a pure sine are the other degenerate corners. We therefore combine:

  * audibility gate   — kills silence (low RMS),
  * tonality          — (1 - spectral flatness), kills broadband hiss,
  * spectral contrast — rewards structured partials (harmonics / subharmonics),
  * spectral flux     — rewards temporal change: the gesture crossing
                        bifurcations / phase transitions is exactly what makes
                        a call sound alive.

The behaviour *descriptor* (where a call lands in the archive) is the BirdNET
embedding; fitness only decides which call wins a given cell.
"""

from __future__ import annotations

import numpy as np
import librosa

_N_FFT = 1024
_HOP = 256


def compute(wave: np.ndarray, sr: int, weights: dict | None = None) -> tuple[float, dict]:
    """Return (fitness, feature_dict) for a mono waveform."""
    w = np.asarray(wave, dtype=np.float32)
    weights = weights or {}
    w_contrast = weights.get("contrast", 1.0)
    w_flux = weights.get("flux", 1.0)
    rms_floor = weights.get("rms_floor", 1e-3)
    rms_ref = weights.get("rms_ref", 0.1)

    rms = float(np.sqrt(np.mean(w**2)) + 1e-12)
    audible = float(np.clip((rms - rms_floor) / (rms_ref - rms_floor), 0.0, 1.0))

    S = np.abs(librosa.stft(w, n_fft=_N_FFT, hop_length=_HOP)) + 1e-9

    # Spectral flatness in [0,1]: ~1 white noise, ~0 tonal.
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    tonal = 1.0 - flat

    # Spectral contrast (dB) averaged over sub-bands and time -> structure.
    contrast = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
    contrast_n = float(np.clip(contrast / 30.0, 0.0, 1.0))

    # Spectral flux: mean positive frame-to-frame change of the normalised
    # magnitude spectrum -> dynamism (onsets, jumps, period doubling).
    Sn = S / (np.sum(S, axis=0, keepdims=True) + 1e-9)
    flux = np.maximum(np.diff(Sn, axis=1), 0.0).sum(axis=0)
    flux_n = float(np.clip(np.mean(flux) * 20.0, 0.0, 1.0))

    fitness = audible * tonal * (w_contrast * contrast_n + w_flux * flux_n)

    feats = dict(rms=rms, audible=audible, flatness=flat, tonal=tonal,
                 contrast=contrast_n, flux=flux_n, fitness=fitness)
    return float(fitness), feats
