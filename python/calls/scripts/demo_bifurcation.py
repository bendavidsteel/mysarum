"""Bifurcation demo — hear the phase transitions.

Sweeps the Mindlin-default instrument along a slow gesture and renders it, so
the spectrogram shows the oscillator moving through its bifurcations (onset,
harmonic stacks, subharmonic / period-doubling regions). This validates that
the JAX port reproduces the nonlinear behaviour of the Rust reference before
any MAP-Elites search runs.

    uv run python scripts/demo_bifurcation.py
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".7")

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from callsong import genome as gm
from callsong import synth, gestures, audio_io

SR = 48_000
OS = 6
DUR = 4.0
N = int(SR * DUR)


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_demo")
    os.makedirs(outdir, exist_ok=True)
    render = synth.make_renderer(SR, OS, N)
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, N)

    # A slow alpha (pressure) ramp across the phonation onset at a few fixed
    # tensions; each sweep traces a different slice of the bifurcation diagram.
    for beta_val in [0.05, 0.6, 1.8]:
        alpha = (gestures.ALPHA_MIN
                 + (gestures.ALPHA_MAX - gestures.ALPHA_MIN) * t).astype(np.float32)
        beta = np.full(N, beta_val, dtype=np.float32)
        phys = gm.decode(gm.mindlin_genome()[None])[:, gm.INSTRUMENT].astype(np.float32)
        noise = rng.uniform(-1, 1, size=(1, N)).astype(np.float32)
        wave = np.asarray(render(jnp.asarray(phys), jnp.asarray(alpha[None]),
                                 jnp.asarray(beta[None]), jnp.asarray(noise))[0])
        nm = f"sweep_beta{beta_val:+.2f}"
        audio_io.save_wav(os.path.join(outdir, nm + ".wav"), wave, SR)
        audio_io.spectrogram_png(os.path.join(outdir, nm + ".png"), wave, SR,
                                 f"alpha ramp, beta={beta_val:+.2f}")
        print(f"{nm}: peak={np.max(np.abs(wave)):.3f} "
              f"nan={bool(np.isnan(wave).any())}")
    print("wrote sweeps to", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
