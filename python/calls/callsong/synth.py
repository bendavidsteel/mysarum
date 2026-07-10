"""JAX syrinx synthesiser — the forward model, no gradients required.

A generalised nonlinear labial oscillator (source) chained through a trachea
reflection comb, a two-formant state-variable band-pass bank, aspiration noise
and a saturating output stage (filters). The whole chain runs on an
oversampled grid inside a single ``lax.scan`` and is ``vmap``-ed over a
population of (instrument, gesture) pairs.

The default genome reproduces the Sitt/Arneodo/Mindlin normal form used in the
Rust reference model:

    dx/dt = y
    dy/dt = g^2(-alpha - beta*x - x^3 + x^2) + g(-x^2 y - x y)

Genome gains k_cub, k_sq, k_sqy, k_xy scale those four nonlinear terms and
k_vdp adds a van der Pol self-oscillation term, so a genome is a deformation of
that vector field — moving its bifurcation structure.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

# Genome index layout — must match callsong.genome.PARAM_SPEC.
(I_GAMMA, I_KCUB, I_KSQ, I_KSQY, I_KXY, I_KVDP, I_TRMS, I_R,
 I_F1, I_Q1, I_G1, I_F2, I_Q2, I_G2, I_DRIVE, I_NOISE,
 I_AMRATE, I_AMDEPTH, I_AMSHAPE, I_MIX2, I_GAMMA2, I_DALPHA2, I_DBETA2) = range(23)

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

    def render_one(phys, alpha, beta, noise):
        gamma = phys[I_GAMMA]
        g2 = gamma * gamma
        k_cub, k_sq = phys[I_KCUB], phys[I_KSQ]
        k_sqy, k_xy, k_vdp = phys[I_KSQY], phys[I_KXY], phys[I_KVDP]
        r = phys[I_R]
        noise_gain = phys[I_NOISE]
        drive = phys[I_DRIVE]

        # Second sound source (bilateral syrinx). It shares the nonlinear gains
        # but has its own time constant and a pressure/tension offset on the
        # gesture, so it sings an independent pitch/contour. mix2 -> 0 collapses
        # the instrument back to a single oscillator.
        gamma2 = phys[I_GAMMA2]
        g2_2 = gamma2 * gamma2
        mix2 = phys[I_MIX2]
        dalpha2, dbeta2 = phys[I_DALPHA2], phys[I_DBETA2]

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
        y_clamp2 = Y_CLAMP_FACTOR * gamma2

        def deriv(x, y, a, b, g, gg):
            dx = y
            dy = (gg * (-a - b * x - k_cub * x**3 + k_sq * x**2)
                  + g * (-k_sqy * x**2 * y - k_xy * x * y
                         + k_vdp * (1.0 - x**2) * y))
            return dx, dy

        def rk4(x, y, a, b, g, gg, yc):
            k1x, k1y = deriv(x, y, a, b, g, gg)
            k2x, k2y = deriv(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, a, b, g, gg)
            k3x, k3y = deriv(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, a, b, g, gg)
            k4x, k4y = deriv(x + dt * k3x, y + dt * k3y, a, b, g, gg)
            x = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            y = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
            return jnp.clip(x, -X_CLAMP, X_CLAMP), jnp.clip(y, -yc, yc)

        def frame_step(carry, inp):
            x, y, x2, y2, buf, pos, lo1, ba1, lo2, ba2 = carry
            a, b, nz = inp
            acc = 0.0
            for _ in range(oversample):
                # 1. Labial oscillators — classic RK4 (stable on the imaginary axis).
                x, y = rk4(x, y, a, b, gamma, g2, y_clamp)
                x2, y2 = rk4(x2, y2, a + dalpha2, b + dbeta2, gamma2, g2_2, y_clamp2)

                # combined source (two labia into one trachea) + aspiration noise
                src = x + mix2 * x2 + noise_gain * nz

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
            return (x, y, x2, y2, buf, pos, lo1, ba1, lo2, ba2), out

        z = jnp.float32(0.0)
        init = (jnp.float32(0.01), z, jnp.float32(0.01), z,
                jnp.zeros(L, jnp.float32), jnp.int32(0), z, z, z, z)
        _, wave = jax.lax.scan(frame_step, init, (alpha, beta, noise))

        # 5. Fast amplitude modulation (pulse trains) — a pointwise output gate.
        t = jnp.arange(n_samples, dtype=jnp.float32) / sr
        pulse = (0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * phys[I_AMRATE] * t)) ** phys[I_AMSHAPE]
        am_env = (1.0 - phys[I_AMDEPTH]) + phys[I_AMDEPTH] * pulse
        wave = wave * am_env

        # Per-call peak normalisation (leave true silence near zero so it stays
        # low-fitness rather than being amplified into noise).
        peak = jnp.max(jnp.abs(wave))
        wave = jnp.where(peak > 1e-4, wave / peak, wave)
        return wave

    render = jax.jit(jax.vmap(render_one))
    return render
