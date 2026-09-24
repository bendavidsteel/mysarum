"""Map the phonating region of the syrinx over the (alpha, beta) plane.

This is the provenance for the ALPHA_*/BETA_* bounds in ``callsong.gestures``.
Published normal forms differ in how they absorb the signs of alpha and beta,
so rather than transcribe a range from a paper we measure it against the model
we actually integrate.

The formants, trachea comb and aspiration noise are bypassed so we characterise
the *source* alone. Waves are peak-normalised inside the renderer, so sustained
oscillation shows up as a healthy rms over the second half of the window while
a decaying startup transient collapses towards zero.

    uv run python scripts/probe_phonation.py
    uv run python scripts/probe_phonation.py --amin -0.1 --amax 1.0 --bmax 4.0
"""

import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".6")

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from callsong import genome as gm
from callsong import synth

SR, OS, DUR = 48_000, 6, 0.3
PHONATING_RMS = 0.15  # empirical split between a limit cycle and a dying transient


def source_only_genome() -> np.ndarray:
    """Mindlin default with the filter chain effectively bypassed."""
    phys = gm.decode(gm.mindlin_genome()[None])[:, gm.INSTRUMENT].astype(np.float32).copy()
    phys[0, synth.I_NOISE] = 0.0     # no aspiration noise
    phys[0, synth.I_R] = 0.0         # no trachea comb
    phys[0, synth.I_G2] = 0.0        # second formant off
    phys[0, synth.I_F1] = 200.0      # first formant well below the source...
    phys[0, synth.I_Q1] = 0.7        # ...and broad, so it just passes the source
    return phys


def analyse(wave: np.ndarray, sr: int) -> tuple[float, float]:
    """(rms, f0) over the second half of the window; f0 = 0 if not oscillating."""
    w = wave[len(wave) // 2:]
    w = w - w.mean()
    rms = float(np.sqrt(np.mean(w ** 2)))
    if rms < PHONATING_RMS:
        return rms, 0.0
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
    freqs = np.fft.rfftfreq(len(w), 1.0 / sr)
    m = freqs > 150.0
    f, s = freqs[m], spec[m]
    # lowest spectral peak within 20 dB of the strongest -> the fundamental
    thr = s.max() / 100.0
    pk = np.where((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] > thr))[0] + 1
    return rms, float(f[pk[0]]) if len(pk) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amin", type=float, default=-0.10)
    ap.add_argument("--amax", type=float, default=0.50)
    ap.add_argument("--bmin", type=float, default=0.0)
    ap.add_argument("--bmax", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=13, help="grid points per axis")
    args = ap.parse_args()

    n_samples = int(SR * DUR)
    alphas = np.linspace(args.amin, args.amax, args.n)
    betas = np.linspace(args.bmin, args.bmax, args.n)
    rows = [(a, b) for a in alphas for b in betas]

    render = synth.make_renderer(SR, OS, n_samples)
    phys = np.repeat(source_only_genome(), len(rows), axis=0)
    A = np.array([[a] * n_samples for a, _ in rows], dtype=np.float32)
    B = np.array([[b] * n_samples for _, b in rows], dtype=np.float32)
    NZ = np.zeros((len(rows), n_samples), dtype=np.float32)

    waves = []
    for i in range(0, len(rows), 32):
        sl = slice(i, i + 32)
        waves.append(np.asarray(render(jnp.asarray(phys[sl]), jnp.asarray(A[sl]),
                                       jnp.asarray(B[sl]), jnp.asarray(NZ[sl]))))
    waves = np.concatenate(waves, axis=0)

    rms = np.zeros((args.n, args.n))
    f0 = np.zeros((args.n, args.n))
    for k in range(len(rows)):
        i, j = divmod(k, args.n)
        rms[i, j], f0[i, j] = analyse(waves[k], SR)

    hdr = "        " + " ".join(f"{b:6.2f}" for b in betas)
    print("rms over the second half (beta across, alpha down):")
    print(hdr)
    for i, a in enumerate(alphas):
        print(f"a={a:+.3f}" + " ".join(f"{v:6.2f}" for v in rms[i]))
    print("\nf0 (Hz; 0 = silent):")
    print(hdr)
    for i, a in enumerate(alphas):
        print(f"a={a:+.3f}" + " ".join(f"{v:6.0f}" for v in f0[i]))

    on = rms > PHONATING_RMS
    if not on.any():
        print("\nnothing phonated anywhere — widen the grid")
        return
    ai, bi = np.where(on)
    live = f0[on]
    print(f"\nphonating cells: {on.mean():.0%} of the grid")
    print(f"  alpha {alphas[ai].min():+.3f} .. {alphas[ai].max():+.3f}"
          f"   (onset near {alphas[ai].min():+.3f})")
    print(f"  beta  {betas[bi].min():+.3f} .. {betas[bi].max():+.3f}")
    print(f"  f0    {live.min():.0f} .. {live.max():.0f} Hz")


if __name__ == "__main__":
    main()
