"""Calibrate the fitness normalisers in ``callsong.features``.

Fitness combines an audibility gate, a tonality gate and two reward terms
(spectral contrast, spectral flux). The reward terms only discriminate between
calls if their normalisers map the *observed* range onto [0, 1] — a normaliser
that saturates turns its term into a constant and silently drops it out of the
sum. This script renders a population through the CPG gesture bank, reports the
raw distribution of each feature, and prints the affine anchors to paste back
into ``features.py``.

Fit the *upper* anchors to an evolved archive, not to a random population.
MAP-Elites hunts the extremes by construction, so elites routinely exceed
anything a random draw produces; anchors fitted to random calls alone leave a
large fraction of the archive pinned at the ceiling with no selection pressure
left between those cells. Random draws still give the honest lower anchor.

    uv run python scripts/calibrate_fitness.py                    # random pop
    uv run python scripts/calibrate_fitness.py --archive outputs/2026-.../
"""

import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".6")

import numpy as np
import jax.numpy as jnp
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from callsong import genome as gm
from callsong import synth, gestures as G, features

SR, OS, DUR = 48_000, 6, 3.0
PCT = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None,
                    help="run directory to calibrate against its elites")
    ap.add_argument("--instruments", type=int, default=40)
    ap.add_argument("--gestures", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n_samples = int(SR * DUR)
    if args.archive:
        z = np.load(os.path.join(args.archive, "archive.npz"))
        m = np.isfinite(z["fitness"])
        phys = gm.decode(z["genomes"][m])
        source = f"{int(m.sum())} elites of {args.archive}"
    else:
        phys = gm.decode(gm.random_genomes(rng, args.instruments))
        source = f"{args.instruments} random genomes"

    # Each genome carries its own gesture, so the motor path comes from the
    # genome rather than from a shared bank.
    P = phys[:, gm.INSTRUMENT].astype(np.float32)
    paths = G.paths_from_params(phys[:, gm.GESTURE], n_samples, SR)
    A, B = paths[:, 0], paths[:, 1]

    render = synth.make_renderer(SR, OS, n_samples)
    NZ = rng.uniform(-1, 1, size=(len(P), n_samples)).astype(np.float32)

    waves = []
    for i in range(0, len(P), args.chunk):
        sl = slice(i, i + args.chunk)
        waves.append(np.asarray(render(jnp.asarray(P[sl]), jnp.asarray(A[sl]),
                                       jnp.asarray(B[sl]), jnp.asarray(NZ[sl]))))
    W = np.concatenate(waves, axis=0)
    print(f"rendered {W.shape[0]} calls ({source})\n")

    raw = {"contrast_db": [], "flux": [], "tonal": [], "rms": []}
    for w in W:
        S = np.abs(librosa.stft(w, n_fft=features._N_FFT,
                                hop_length=features._HOP)) + 1e-9
        Sn = S / (np.sum(S, axis=0, keepdims=True) + 1e-9)
        raw["flux"].append(float(np.mean(
            np.maximum(np.diff(Sn, axis=1), 0.0).sum(axis=0))))
        raw["contrast_db"].append(float(np.mean(
            librosa.feature.spectral_contrast(S=S, sr=SR))))
        raw["tonal"].append(1.0 - float(np.mean(
            librosa.feature.spectral_flatness(S=S))))
        raw["rms"].append(float(np.sqrt(np.mean(w ** 2))))

    for name, v in raw.items():
        q = np.percentile(v, PCT)
        print(f"{name:12s} " + "  ".join(f"p{p}={x:.4f}" for p, x in zip(PCT, q)))

    print("\nsuggested anchors (p0/p100, rounded outward):")
    for name, const in (("contrast_db", "_CONTRAST"), ("flux", "_FLUX")):
        lo, hi = np.min(raw[name]), np.max(raw[name])
        print(f"  {const}_LO, {const}_HI = "
              f"{np.floor(lo * 100) / 100:g}, {np.ceil(hi * 100) / 100:g}")

    print("\nwith the anchors currently in features.py:")
    fits = []
    for w in W:
        f, d = features.compute(w, SR)
        fits.append((f, d["contrast"], d["flux"], d["audible"], d["tonal"]))
    fits = np.array(fits)
    for k, name in enumerate(["fitness", "contrast_n", "flux_n",
                              "audible", "tonal"]):
        col = fits[:, k]
        sat = np.mean(col >= 0.999) if name != "fitness" else np.nan
        print(f"  {name:11s} mean {col.mean():.3f}  std {col.std():.3f}  "
              f"range {col.min():.3f}-{col.max():.3f}"
              + ("" if np.isnan(sat) else f"  saturated {sat:.1%}"))
    print("\nA term saturated for most calls contributes no ranking signal.")


if __name__ == "__main__":
    main()
