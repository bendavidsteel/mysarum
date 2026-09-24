"""Quick synth smoke test: render the Mindlin default + one variant, report
timing and basic stats, and save wavs + spectrograms."""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".7")

import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from callsong import genome as gm
from callsong import synth, gestures, features, audio_io

SR = 48_000
OS = 6
DUR = 2.0
N = int(SR * DUR)

print("jax devices:", jax.devices())

bank = gestures.make_gesture_bank(seed=0, n_gestures=3, n_samples=N)
print("gesture bank:", bank.shape)

# Two instruments: Mindlin default, and a van-der-Pol-heavy variant.
mindlin = gm.mindlin_genome()
variant = mindlin.copy()
variant[gm.NAMES.index("k_sqy")] = 0.9
variant[gm.NAMES.index("k_sq")] = 0.2
phys = gm.decode(np.stack([mindlin, mindlin, variant]))[:, gm.INSTRUMENT]  # 3 calls

# instrument i, gesture ids: mindlin+g0 (ramp), mindlin+g1 (CPG), variant+g0
alpha = np.stack([bank[0, 0], bank[1, 0], bank[0, 0]]).astype(np.float32)
beta = np.stack([bank[0, 1], bank[1, 1], bank[0, 1]]).astype(np.float32)
rng = np.random.default_rng(0)
noise = rng.uniform(-1, 1, size=(3, N)).astype(np.float32)

render = synth.make_renderer(SR, OS, N)

t0 = time.time()
wave = render(jax.numpy.asarray(phys), jax.numpy.asarray(alpha),
              jax.numpy.asarray(beta), jax.numpy.asarray(noise))
wave.block_until_ready()
t1 = time.time()
print(f"compile+first render (3 calls, {DUR}s@{SR}, OS={OS}): {t1 - t0:.2f}s")

t0 = time.time()
wave = render(jax.numpy.asarray(phys), jax.numpy.asarray(alpha),
              jax.numpy.asarray(beta), jax.numpy.asarray(noise))
wave.block_until_ready()
t1 = time.time()
print(f"warm render: {t1 - t0:.3f}s  ({(t1 - t0) / 3 * 1000:.0f} ms/call)")

wave = np.asarray(wave)
print("NaN?", bool(np.isnan(wave).any()), " shape", wave.shape)

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_smoke")
os.makedirs(outdir, exist_ok=True)
names = ["mindlin_ramp", "mindlin_cpg", "sqy_ramp"]
for k, nm in enumerate(names):
    fit, feats = features.compute(wave[k], SR)
    print(f"{nm:16s} rms={feats['rms']:.3f} tonal={feats['tonal']:.2f} "
          f"contrast={feats['contrast']:.2f} flux={feats['flux']:.2f} "
          f"fitness={fit:.3f}")
    audio_io.save_wav(os.path.join(outdir, nm + ".wav"), wave[k], SR)
    audio_io.spectrogram_png(os.path.join(outdir, nm + ".png"), wave[k], SR, nm)
print("wrote wavs + spectrograms to", os.path.abspath(outdir))
