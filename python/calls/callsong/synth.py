"""JAX syrinx synthesiser — the forward model, no gradients required.

A generalised nonlinear labial oscillator (source) chained through a trachea
reflection comb, a two-formant state-variable band-pass bank, aspiration noise
and a saturating output stage (filters). The whole chain runs on an
oversampled grid inside a single ``lax.scan`` and is ``vmap``-ed over a
population of (instrument, gesture) pairs.

The default genome reproduces the Sitt/Arneodo/Mindlin normal form:

    dx/dt = y
    dy/dt = g^2(-alpha - beta*x - x^3 + x^2) + g(-x^2 y - x y)

The signs on alpha and beta are the ones that make them read as the *physical*
gestures of Amador et al. (Nature 2013) / Boari et al. (J. Neurophysiol. 2015):
alpha rising past the phonation onset starts the labia oscillating, and beta
falling raises the fundamental. Papers vary in how they absorb these signs into
the normal form, so the phonating region here was mapped empirically rather
than transcribed — see ``callsong.gestures``.

Genome gains k_cub, k_sq, k_sqy, k_xy scale those four nonlinear terms and
k_vdp adds a van der Pol term, so a genome is a deformation of that vector
field — moving its bifurcation structure.

Note that k_vdp's negative damping does not depend on alpha, so a large k_vdp
makes an instrument that sings with no pressure at all and ignores the motor
gesture. That is not forbidden here; it is *scored*, by the unforced-phonation
gate in :mod:`callsong.features`, which measures output during the window where
the gesture applies no drive.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

# Genome index layout — must match callsong.genome.PARAM_SPEC.
(I_GAMMA, I_KCUB, I_KSQ, I_KSQY, I_KXY, I_KVDP, I_TRMS, I_R,
 I_F1, I_Q1, I_G1, I_F2, I_Q2, I_G2, I_DRIVE, I_NOISE) = range(16)

# Aspiration noise is turbulence in the airflow, so it needs airflow: it fades
# out with the pressure rather than hissing through the silences. Full
# amplitude by this alpha, zero at zero pressure.
NOISE_FLOW_REF = 0.05

# The oscillator starts away from its rest fixed point, and settling onto it
# rings the trachea comb and the formants — a broadband click at t=0 on every
# call, which BirdNET embeds as readily as anything else. Hold the first
# gesture value for this long before the window starts, and throw the result
# away, so each call begins already at rest.
WARMUP_S = 0.1

TRACHEA_MS_MAX = 4.0  # buffer headroom; matches genome.PARAM_SPEC upper bound

# State clamps (safety net at extreme parameters; RK4 rarely trips them).
X_CLAMP = 3.0
Y_CLAMP_FACTOR = 3.0  # |y| <= factor * gamma


def _trachea_len(sr: float, oversample: int) -> int:
    sr_os = sr * oversample
    return int(math.ceil(TRACHEA_MS_MAX * 1e-3 * sr_os)) + 1


def make_renderer(sr: float, oversample: int, n_samples: int):
    """Build a jitted, vmapped renderer for a fixed (sr, oversample, length).

    Returns ``render(phys, alpha, beta, noise) -> waveform`` where the inputs
    carry a leading batch axis:
        phys   (B, N_PARAMS)   physical instrument parameters
        alpha  (B, n_samples)  air-sac pressure gesture
        beta   (B, n_samples)  syringeal tension gesture
        noise  (B, n_samples)  white aspiration noise in [-1, 1]
    and waveform is (B, n_samples), peak-normalised per call.
    """
    sr_os = sr * oversample
    dt = 1.0 / sr_os
    L = _trachea_len(sr, oversample)
    n_warmup = int(round(WARMUP_S * sr))

    def render_one(phys, alpha, beta, noise):
        gamma = phys[I_GAMMA]
        g2 = gamma * gamma
        k_cub, k_sq = phys[I_KCUB], phys[I_KSQ]
        k_sqy, k_xy, k_vdp = phys[I_KSQY], phys[I_KXY], phys[I_KVDP]
        r = phys[I_R]
        noise_gain = phys[I_NOISE]
        drive = phys[I_DRIVE]

        # Trachea delay in oversampled slots (invariant to oversample).
        delay = jnp.clip(jnp.round(phys[I_TRMS] * 1e-3 * sr_os).astype(jnp.int32),
                         1, L - 1)

        # State-variable filter coefficients (SVF runs on the oversampled grid).
        f1 = 2.0 * jnp.sin(jnp.pi * phys[I_F1] / sr_os)
        f2 = 2.0 * jnp.sin(jnp.pi * phys[I_F2] / sr_os)
        d1 = 1.0 / phys[I_Q1]
        d2 = 1.0 / phys[I_Q2]
        g1, g2f = phys[I_G1], phys[I_G2]

        y_clamp = Y_CLAMP_FACTOR * gamma

        def deriv(x, y, a, b):
            dx = y
            dy = (g2 * (-a - b * x - k_cub * x**3 + k_sq * x**2)
                  + gamma * (-k_sqy * x**2 * y - k_xy * x * y
                             + k_vdp * (1.0 - x**2) * y))
            return dx, dy

        def frame_step(carry, inp):
            x, y, buf, pos, lo1, ba1, lo2, ba2 = carry
            a, b, nz = inp
            acc = 0.0
            for _ in range(oversample):
                # 1. Labial oscillator — classic RK4 (stable on the imaginary axis).
                k1x, k1y = deriv(x, y, a, b)
                k2x, k2y = deriv(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, a, b)
                k3x, k3y = deriv(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, a, b)
                k4x, k4y = deriv(x + dt * k3x, y + dt * k3y, a, b)
                x = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
                y = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
                x = jnp.clip(x, -X_CLAMP, X_CLAMP)
                y = jnp.clip(y, -y_clamp, y_clamp)

                # source + aspiration noise, the noise scaled by airflow so it
                # cannot hiss through the silences between phrases
                flow = jnp.clip(a / NOISE_FLOW_REF, 0.0, 1.0)
                src = x + noise_gain * flow * nz

                # 2. Trachea reflection comb: Pi = src - r*Pi(t-T).
                read_pos = (pos - delay) % L
                delayed = buf[read_pos]
                pi = src - r * delayed
                buf = buf.at[pos].set(pi)
                pos = (pos + 1) % L

                # 3. Two parallel state-variable band-passes (OEC formants).
                lo1 = lo1 + f1 * ba1
                hi1 = pi - lo1 - d1 * ba1
                ba1 = ba1 + f1 * hi1
                lo2 = lo2 + f2 * ba2
                hi2 = pi - lo2 - d2 * ba2
                ba2 = ba2 + f2 * hi2

                acc = acc + g1 * ba1 + g2f * ba2

            # 4. Average the oversampled output and soft-clip.
            out = jnp.tanh(drive * acc / oversample)
            return (x, y, buf, pos, lo1, ba1, lo2, ba2), out

        init = (jnp.float32(0.01), jnp.float32(0.0),
                jnp.zeros(L, jnp.float32), jnp.int32(0),
                jnp.float32(0.0), jnp.float32(0.0),
                jnp.float32(0.0), jnp.float32(0.0))
        pad = jnp.full(n_warmup, alpha[0], alpha.dtype)
        a_in = jnp.concatenate([pad, alpha])
        b_in = jnp.concatenate([jnp.full(n_warmup, beta[0], beta.dtype), beta])
        n_in = jnp.concatenate([jnp.zeros(n_warmup, noise.dtype), noise])
        _, wave = jax.lax.scan(frame_step, init, (a_in, b_in, n_in))
        wave = wave[n_warmup:]

        # Per-call peak normalisation (leave true silence near zero so it stays
        # low-fitness rather than being amplified into noise).
        peak = jnp.max(jnp.abs(wave))
        wave = jnp.where(peak > 1e-4, wave / peak, wave)
        return wave

    render = jax.jit(jax.vmap(render_one))
    return render
