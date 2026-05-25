"""
Physarum via reintegration tracking — unified 2D / 3D JAX implementation.

State (2D): per-pixel particle (X, V, M) on an (H, W) grid.
State (3D): per-voxel particle (X, V, M) on a (D, H, W) grid.
  X: cell-centered global position (last axis = (x,y) or (x,y,z))
  V: velocity
  M: (mass, trail)

One step = reintegrate (mass-conservative advection over 5×5 or 5×5×5 stencil)
         + simulate    (SPH pressure + per-cell-trail-modulated sensing
                        + border + integrate, parameterized)
         + update_trail (decay + deposit).

Agent behaviour is parameterized in the style of Sage Jenson's "36 Points"
(https://sagejenson.com/36points): each per-cell quantity Q is computed as
    Q = Q_base + Q_scale * trail ^ Q_power
With Q_scale=0 the parameter is a constant Q_base; otherwise the agents
become more sensitive in dense regions ("primary special sauce").

3D-only post-processing:
  printability_report  — GPU threshold sweep over the trail field; for each
    threshold keeps the largest 26-connected component and records EDT
    statistics, morphological-opening survival, and surface/volume.
  export_stl           — marching cubes on the chosen mask + binary STL.
  map_elites_run       — quality-diversity search over PhysarumParams,
    archive indexed by (opening_ratio, edt_median).

Entry point: hydra. Run with
    python reint_jax.py                          # default config (3D, weblike)
    python reint_jax.py dim=2                    # 2D mode
    python reint_jax.py preset=tendrils
    python reint_jax.py mapelites.enabled=true   # quality-diversity sweep
See conf/config.yaml for all knobs.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import struct
import time
from pathlib import Path
from typing import NamedTuple

import cupy as cp
import cupyx.scipy.ndimage as cndi
# cupy 13.6 + locally installed CUDA 12.0 NVRTC ships C++14 by default, but the
# bundled CCCL bfloat16 constexpr declarations require C++17 to compile (we hit
# `constexpr function return is non-constant` from `cuda/std/limits` otherwise).
# Force `--std=c++17` on every NVRTC compile.
def _patch_cupy_nvrtc_std():
    import cupy.cuda.compiler as _comp
    _orig = _comp.compile_using_nvrtc
    def _patched(source, options=(), arch=None, *a, **kw):
        opts = tuple(o for o in options if not o.startswith(('-std=', '--std=')))
        return _orig(source, opts + ('--std=c++17',), arch, *a, **kw)
    _comp.compile_using_nvrtc = _patched
_patch_cupy_nvrtc_std()

import hydra
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from skimage.measure import marching_cubes


# === physics constants ======================================================
DT = 1.5
MASS = 2.0
FLUID_RHO = 0.2
DIF = 1.3
ACCELERATION = 0.01
BORDER_H = 5.0
GRAVITY = 0.001
# Inward-pointing repulsive force per (mass × penetration-depth) near walls.
# In addition to the hard-stop bounce — without it, mass slides along walls
# and accumulates in corners because tangential motion is unopposed there.
BORDER_REPULSION = 0.04


# === agent-behaviour parameters (Sage Jenson "36 Points" style) =============

class PhysarumParams(NamedTuple):
    """Per-cell agent behaviour, modulated by local trail concentration.

    For each Q in (sd, sa, ra, md), the effective value at a cell is
        Q = Q_base + Q_scale * trail ^ Q_power
    so Q_scale=0 (or Q_power=0) gives a constant Q_base. trail is clamped
    to a tiny positive minimum so the `pow` is well-defined.

    sd : sensor distance (pixels/voxels)  — how far ahead each tap reads
    sa : sensor cone half-angle (rad)     — 2D: ±sa from forward (2 taps);
                                            3D: ±sa cone (4 taps right/up)
    ra : rotation/turn strength           — gain on the perpendicular force
    md : move distance / max speed        — per-cell velocity cap

    sensor_offset_right : bulk lateral displacement of sample positions in
        the local right-perpendicular direction. (Active in 2D and 3D.)
    sensor_offset_up    : same, along the local up axis. (3D only;
        ignored in 2D.)

    deposit : trail added each step from local mass.
    decay   : multiplicative trail decay each step (1.0 = no decay).

    NOTE: the reintegration stencil is 5 voxels wide (±2), so md * DT must
    stay below ~2.5 to preserve mass conservation. With DT=1.5, keep
    md ≤ ~1.6 in steady state.
    """
    sd_base: float = 3.0
    sd_power: float = 0.0
    sd_scale: float = 0.0
    sa_base: float = 0.45
    sa_power: float = 0.0
    sa_scale: float = 0.0
    ra_base: float = 0.2
    ra_power: float = 0.0
    ra_scale: float = 0.0
    md_base: float = 1.0
    md_power: float = 0.0
    md_scale: float = 0.0
    sensor_offset_right: float = 0.0
    sensor_offset_up: float = 0.0
    deposit: float = 0.15
    decay: float = 0.94


def param_value(base, power, scale, trail):
    return base + scale * jnp.power(jnp.maximum(trail, 1e-9), power)


PRESETS = {
    "default": PhysarumParams(),
    "weblike": PhysarumParams(
        sd_base=2.0, sd_scale=3.0, sd_power=1.0,
        sa_base=0.35, sa_scale=0.4, sa_power=1.5,
        ra_base=0.15, ra_scale=0.6, ra_power=1.5,
        md_base=1.0,
        deposit=0.2, decay=0.93,
    ),
    "tendrils": PhysarumParams(
        sd_base=2.5, sd_scale=2.0, sd_power=2.0,
        sa_base=0.25,
        ra_base=0.4,
        md_base=0.9,
        deposit=0.25, decay=0.92,
    ),
    "spiral": PhysarumParams(
        sd_base=3.0, sd_scale=1.5, sd_power=1.0,
        sa_base=0.5,
        ra_base=0.3,
        md_base=1.0,
        sensor_offset_right=0.8, sensor_offset_up=0.0,
        deposit=0.18, decay=0.95,
    ),
}


# === shared helpers =========================================================

def Pf(M):
    return M[..., 0]


def position_grid_2d(H, W):
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    Y, X = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([X, Y], axis=-1)


def position_grid_3d(D, H, W):
    zs = jnp.arange(D, dtype=jnp.float32)
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    Z, Y, X = jnp.meshgrid(zs, ys, xs, indexing="ij")
    return jnp.stack([X, Y, Z], axis=-1)


def _wrap_pad_2d(arr, halo=2):
    return jnp.pad(arr, ((halo, halo), (halo, halo), (0, 0)), mode="wrap")


def _wrap_pad_3d(arr, halo=2):
    return jnp.pad(arr,
                   ((halo, halo), (halo, halo), (halo, halo), (0, 0)),
                   mode="wrap")


def bilinear_sample(field, pos):
    """Sample a (H,W,C) field at fractional (x,y) positions, periodic."""
    H, W = field.shape[:2]
    x = jnp.mod(pos[..., 0], W)
    y = jnp.mod(pos[..., 1], H)
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H
    fx = (x - jnp.floor(x))[..., None]
    fy = (y - jnp.floor(y))[..., None]
    f00 = field[y0, x0]; f01 = field[y0, x1]
    f10 = field[y1, x0]; f11 = field[y1, x1]
    return (1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f01 \
         + (1 - fx) * fy * f10 + fx * fy * f11


def trilinear_sample(field, pos, mode="wrap"):
    """Sample a (D,H,W,C) field at fractional (x,y,z) positions.
    mode='wrap' (periodic) or 'clamp' (zeros outside)."""
    D, H, W = field.shape[:3]
    x = pos[..., 0]; y = pos[..., 1]; z = pos[..., 2]

    if mode == "wrap":
        x = jnp.mod(x, W); y = jnp.mod(y, H); z = jnp.mod(z, D)
        outside_mask = None
    else:
        outside_mask = ((x >= 0) & (x <= W - 1)
                        & (y >= 0) & (y <= H - 1)
                        & (z >= 0) & (z <= D - 1)).astype(field.dtype)
        x = jnp.clip(x, 0.0, W - 1); y = jnp.clip(y, 0.0, H - 1); z = jnp.clip(z, 0.0, D - 1)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    z0 = jnp.floor(z).astype(jnp.int32)
    if mode == "wrap":
        x1 = (x0 + 1) % W; y1 = (y0 + 1) % H; z1 = (z0 + 1) % D
    else:
        x1 = jnp.minimum(x0 + 1, W - 1)
        y1 = jnp.minimum(y0 + 1, H - 1)
        z1 = jnp.minimum(z0 + 1, D - 1)

    fx = (x - jnp.floor(x))[..., None]
    fy = (y - jnp.floor(y))[..., None]
    fz = (z - jnp.floor(z))[..., None]
    f000 = field[z0, y0, x0]; f001 = field[z0, y0, x1]
    f010 = field[z0, y1, x0]; f011 = field[z0, y1, x1]
    f100 = field[z1, y0, x0]; f101 = field[z1, y0, x1]
    f110 = field[z1, y1, x0]; f111 = field[z1, y1, x1]
    c00 = (1 - fx) * f000 + fx * f001
    c01 = (1 - fx) * f010 + fx * f011
    c10 = (1 - fx) * f100 + fx * f101
    c11 = (1 - fx) * f110 + fx * f111
    c0 = (1 - fy) * c00 + fy * c01
    c1 = (1 - fy) * c10 + fy * c11
    out = (1 - fz) * c0 + fz * c1
    if outside_mask is not None:
        out = out * outside_mask[..., None]
    return out


def _border_value_2d(pos, R):
    d = jnp.abs(pos - R * 0.5) - R * 0.5
    sd = jnp.linalg.norm(jnp.maximum(d, 0), axis=-1) \
       + jnp.minimum(jnp.max(d, axis=-1), 0)
    return -sd


def border_normal_2d(pos, R):
    h = 1.0
    ex = jnp.array([h, 0.0]); ey = jnp.array([0.0, h])
    f_xp = _border_value_2d(pos + ex, R); f_xm = _border_value_2d(pos - ex, R)
    f_yp = _border_value_2d(pos + ey, R); f_ym = _border_value_2d(pos - ey, R)
    grad = jnp.stack([(f_xp - f_xm), (f_yp - f_ym)], axis=-1) / (2 * h)
    n = grad / (jnp.linalg.norm(grad, axis=-1, keepdims=True) + 1e-12)
    val = 0.25 * (f_xp + f_xm + f_yp + f_ym) + 1e-4
    return n, val


def _border_value_3d(pos, R):
    d = jnp.abs(pos - R * 0.5) - R * 0.5
    sd = jnp.linalg.norm(jnp.maximum(d, 0), axis=-1) \
       + jnp.minimum(jnp.max(d, axis=-1), 0)
    return -sd


def border_normal_3d(pos, R):
    h = 1.0
    ex = jnp.array([h, 0.0, 0.0]); ey = jnp.array([0.0, h, 0.0]); ez = jnp.array([0.0, 0.0, h])
    f_xp = _border_value_3d(pos + ex, R); f_xm = _border_value_3d(pos - ex, R)
    f_yp = _border_value_3d(pos + ey, R); f_ym = _border_value_3d(pos - ey, R)
    f_zp = _border_value_3d(pos + ez, R); f_zm = _border_value_3d(pos - ez, R)
    grad = jnp.stack([f_xp - f_xm, f_yp - f_ym, f_zp - f_zm], axis=-1) / (2 * h)
    n = grad / (jnp.linalg.norm(grad, axis=-1, keepdims=True) + 1e-12)
    val = (f_xp + f_xm + f_yp + f_ym + f_zp + f_zm) / 6.0 + 1e-4
    return n, val


def basis_from_velocity_3d(V, eps=1e-6):
    """(forward, right, up) for 3D 4-tap sensing. Robust to V≈0 and to
    forward parallel to world-up (swaps helper axis)."""
    speed = jnp.linalg.norm(V, axis=-1, keepdims=True)
    forward = jnp.where(speed > eps, V / jnp.maximum(speed, eps),
                        jnp.array([1.0, 0.0, 0.0]))
    parallel = jnp.abs(forward[..., 2:3]) > 0.95
    aux = jnp.where(parallel,
                    jnp.array([1.0, 0.0, 0.0]),
                    jnp.array([0.0, 0.0, 1.0]))
    right = jnp.cross(forward, aux)
    right = right / (jnp.linalg.norm(right, axis=-1, keepdims=True) + eps)
    up = jnp.cross(right, forward)
    return forward, right, up


# === 2D =====================================================================

class State2D(NamedTuple):
    X: jnp.ndarray  # (H, W, 2)
    V: jnp.ndarray  # (H, W, 2)
    M: jnp.ndarray  # (H, W, 2)


def reintegrate_2d(state: State2D) -> State2D:
    X, V, M = state
    H, W = X.shape[:2]
    pos = position_grid_2d(H, W)
    R = jnp.array([W, H], dtype=jnp.float32)

    Xp = _wrap_pad_2d(X)
    Vp = _wrap_pad_2d(V)
    Mp = _wrap_pad_2d(M)

    M_acc = jnp.zeros((H, W), dtype=jnp.float32)
    X_acc = jnp.zeros((H, W, 2), dtype=jnp.float32)
    V_acc = jnp.zeros((H, W, 2), dtype=jnp.float32)
    half_p = DIF * 0.5
    inv_K2 = 1.0 / (DIF * DIF)

    for dy in range(5):
        for dx in range(5):
            Xn = Xp[dy:dy+H, dx:dx+W]
            Vn = Vp[dy:dy+H, dx:dx+W]
            Mn = Mp[dy:dy+H, dx:dx+W]

            rel = (Xn + Vn * DT) - pos
            rel = rel - jnp.round(rel / R) * R

            inter_lo = jnp.maximum(-0.5, rel - half_p)
            inter_hi = jnp.minimum(0.5, rel + half_p)
            size = jnp.maximum(inter_hi - inter_lo, 0.0)
            center_rel = 0.5 * (inter_lo + inter_hi)
            frac = size[..., 0] * size[..., 1] * inv_K2

            m = Mn[..., 0] * frac
            M_acc = M_acc + m
            X_acc = X_acc + center_rel * m[..., None]
            V_acc = V_acc + Vn * m[..., None]

    safe = M_acc[..., None] > 1e-12
    inv_m = jnp.where(safe, 1.0 / jnp.maximum(M_acc[..., None], 1e-12), 0.0)
    X_rel = X_acc * inv_m
    V_new = V_acc * inv_m
    X_new = pos + jnp.clip(X_rel, -0.5, 0.5)
    # Trail is NOT advected with mass — it's a static chemical field that
    # decays in place and is replenished by mass deposit in update_trail_2d.
    M_new = jnp.stack([M_acc, M[..., 1]], axis=-1)
    return State2D(X_new, V_new, M_new)


def simulate_2d(state: State2D, params: PhysarumParams) -> State2D:
    X, V, M = state
    H, W = X.shape[:2]
    R = jnp.array([W, H], dtype=jnp.float32)

    sv = M[..., 1]
    sd = param_value(params.sd_base, params.sd_power, params.sd_scale, sv)
    sa = param_value(params.sa_base, params.sa_power, params.sa_scale, sv)
    ra = param_value(params.ra_base, params.ra_power, params.ra_scale, sv)
    md = param_value(params.md_base, params.md_power, params.md_scale, sv)

    Xp = _wrap_pad_2d(X)
    Mp = _wrap_pad_2d(M)

    F = jnp.zeros((H, W, 2), dtype=jnp.float32)
    for dy in range(5):
        for dx_i in range(5):
            Xn = Xp[dy:dy+H, dx_i:dx_i+W]
            Mn = Mp[dy:dy+H, dx_i:dx_i+W]
            dxv = Xn - X
            dxv = dxv - jnp.round(dxv / R) * R
            Gw = jnp.exp(-jnp.sum(dxv * dxv, axis=-1))
            avgP = 0.5 * Mn[..., 0] * (Pf(M) + Pf(Mn))
            F = F - 0.5 * (Gw * avgP)[..., None] * dxv

    # 2-tap sensing with per-cell sense distance + cone angle.
    ang = jnp.arctan2(V[..., 1], V[..., 0])
    cos_l = jnp.cos(ang + sa); sin_l = jnp.sin(ang + sa)
    cos_r = jnp.cos(ang - sa); sin_r = jnp.sin(ang - sa)
    dl = sd[..., None] * jnp.stack([cos_l, sin_l], axis=-1)
    dr = sd[..., None] * jnp.stack([cos_r, sin_r], axis=-1)
    perp_l = jnp.stack([jnp.cos(ang + jnp.pi / 2),
                        jnp.sin(ang + jnp.pi / 2)], axis=-1)
    perp_r = jnp.stack([jnp.cos(ang - jnp.pi / 2),
                        jnp.sin(ang - jnp.pi / 2)], axis=-1)
    sense_origin = X + perp_r * params.sensor_offset_right
    trail = M[..., 1:2]
    sd_l = bilinear_sample(trail, sense_origin + dl)[..., 0]
    sd_r = bilinear_sample(trail, sense_origin + dr)[..., 0]
    F = F + ra[..., None] * (perp_l * sd_l[..., None] + perp_r * sd_r[..., None])

    F = F - GRAVITY * M[..., 0:1] * jnp.array([0.0, 1.0])

    # Inward-pointing repulsion + reflective bounce share the SDF query.
    N_xy, N_z = border_normal_2d(X, R)
    penetration = jnp.maximum(BORDER_H - N_z, 0.0)[..., None]
    F = F + BORDER_REPULSION * M[..., 0:1] * penetration * N_xy

    inv_m = 1.0 / jnp.maximum(M[..., 0:1], 1e-12)
    V_new = V + F * DT * inv_m

    close = (N_z <= BORDER_H).astype(jnp.float32)[..., None]
    vdotN = close[..., 0] * jnp.sum(-N_xy * V_new, axis=-1)
    bounce = 0.5 * (N_xy * vdotN[..., None] + N_xy * jnp.abs(vdotN)[..., None])
    V_new = V_new + bounce
    V_new = jnp.where((N_z < 0)[..., None], 0.0, V_new)

    V_new = V_new * (1.0 + ACCELERATION)
    speed = jnp.linalg.norm(V_new, axis=-1, keepdims=True)
    md_e = jnp.maximum(md[..., None], 1e-6)
    V_new = V_new / jnp.maximum(speed / md_e, 1.0)

    live = (M[..., 0:1] > 1e-9).astype(jnp.float32)
    V_new = live * V_new + (1.0 - live) * V

    return State2D(X, V_new, M)


def update_trail_2d(state: State2D, params: PhysarumParams) -> State2D:
    X, V, M = state
    mass = M[..., 0]
    trail = M[..., 1] * params.decay + mass * params.deposit
    return State2D(X, V, jnp.stack([mass, trail], axis=-1))


def step_2d(state: State2D, params: PhysarumParams) -> State2D:
    return update_trail_2d(simulate_2d(reintegrate_2d(state), params), params)


def init_state_2d(key, H, W, radius_frac=0.1) -> State2D:
    pos = position_grid_2d(H, W)
    centre = jnp.array([W * 0.5, H * 0.5])
    r = jnp.linalg.norm(pos - centre, axis=-1) / max(H, W)
    in_disk = (r < radius_frac)[..., None]

    rand = jax.random.uniform(key, (H, W, 2)) - 0.5
    X = pos
    V = jnp.where(in_disk, 0.5 * rand, 0.0)
    M = jnp.where(in_disk,
                  jnp.array([MASS, MASS * 0.1]),
                  jnp.array([1e-6, 0.0]))
    return State2D(X, V, M)


def render_2d(state: State2D) -> jnp.ndarray:
    """Gaussian-weighted (mass, trail) → RGB; same colormap as 3D MIPs."""
    X, V, M = state
    H, W = X.shape[:2]
    pos = position_grid_2d(H, W)
    diff = pos - X
    g = jnp.exp(-jnp.sum((diff / 0.75) ** 2, axis=-1))

    rho_z = M[..., 0] * g
    rho_w = M[..., 1] * g

    a = jnp.power(jnp.clip(rho_z / (FLUID_RHO * 2.0 + 1e-12), 0.0, 1.0), 0.1)
    a = a * a * (3 - 2 * a)
    col = jnp.broadcast_to((0.2 * a)[..., None], (H, W, 3))
    palette = jnp.array([0.2, 0.8, 0.6])
    col = col + (0.5 - 0.5 * jnp.cos(8.0 * palette * rho_w[..., None]))
    col = jnp.tanh(4.0 * jnp.power(jnp.clip(col, 0.0, 10.0), 1.5))
    return jnp.clip(col, 0.0, 1.0)


# === 3D =====================================================================

class State3D(NamedTuple):
    X: jnp.ndarray  # (D, H, W, 3)
    V: jnp.ndarray  # (D, H, W, 3)
    M: jnp.ndarray  # (D, H, W, 2)


def reintegrate_3d(state: State3D) -> State3D:
    X, V, M = state
    D, H, W = X.shape[:3]
    pos = position_grid_3d(D, H, W)
    R = jnp.array([W, H, D], dtype=jnp.float32)

    Xp = _wrap_pad_3d(X)
    Vp = _wrap_pad_3d(V)
    Mp = _wrap_pad_3d(M)

    half_p = DIF * 0.5
    inv_K3 = 1.0 / (DIF * DIF * DIF)

    # Outer dz wrapped in fori_loop to bound peak memory.
    def dz_body(dz_idx, acc):
        M_acc, X_acc, V_acc = acc
        Xpd = jax.lax.dynamic_slice_in_dim(Xp, dz_idx, D, axis=0)
        Vpd = jax.lax.dynamic_slice_in_dim(Vp, dz_idx, D, axis=0)
        Mpd = jax.lax.dynamic_slice_in_dim(Mp, dz_idx, D, axis=0)
        for dy in range(5):
            for dx in range(5):
                Xn = Xpd[:, dy:dy + H, dx:dx + W]
                Vn = Vpd[:, dy:dy + H, dx:dx + W]
                Mn = Mpd[:, dy:dy + H, dx:dx + W]

                rel = (Xn + Vn * DT) - pos
                rel = rel - jnp.round(rel / R) * R

                inter_lo = jnp.maximum(-0.5, rel - half_p)
                inter_hi = jnp.minimum(0.5, rel + half_p)
                size = jnp.maximum(inter_hi - inter_lo, 0.0)
                center_rel = 0.5 * (inter_lo + inter_hi)
                frac = size[..., 0] * size[..., 1] * size[..., 2] * inv_K3

                m = Mn[..., 0] * frac
                M_acc = M_acc + m
                X_acc = X_acc + center_rel * m[..., None]
                V_acc = V_acc + Vn * m[..., None]
        return M_acc, X_acc, V_acc

    M_acc, X_acc, V_acc = jax.lax.fori_loop(
        0, 5, dz_body,
        (jnp.zeros((D, H, W), dtype=jnp.float32),
         jnp.zeros((D, H, W, 3), dtype=jnp.float32),
         jnp.zeros((D, H, W, 3), dtype=jnp.float32)),
    )

    inv_m = jnp.where(M_acc[..., None] > 1e-12,
                      1.0 / jnp.maximum(M_acc[..., None], 1e-12), 0.0)
    X_rel = X_acc * inv_m
    V_new = V_acc * inv_m
    X_new = pos + jnp.clip(X_rel, -0.5, 0.5)
    M_new = jnp.stack([M_acc, M[..., 1]], axis=-1)
    return State3D(X_new, V_new, M_new)


def simulate_3d(state: State3D, params: PhysarumParams) -> State3D:
    X, V, M = state
    D, H, W = X.shape[:3]
    R = jnp.array([W, H, D], dtype=jnp.float32)

    sv = M[..., 1]
    sd = param_value(params.sd_base, params.sd_power, params.sd_scale, sv)
    sa = param_value(params.sa_base, params.sa_power, params.sa_scale, sv)
    ra = param_value(params.ra_base, params.ra_power, params.ra_scale, sv)
    md = param_value(params.md_base, params.md_power, params.md_scale, sv)

    Xp = _wrap_pad_3d(X)
    Mp = _wrap_pad_3d(M)

    def dz_body(dz_idx, F):
        Xpd = jax.lax.dynamic_slice_in_dim(Xp, dz_idx, D, axis=0)
        Mpd = jax.lax.dynamic_slice_in_dim(Mp, dz_idx, D, axis=0)
        for dy in range(5):
            for dx in range(5):
                Xn = Xpd[:, dy:dy + H, dx:dx + W]
                Mn = Mpd[:, dy:dy + H, dx:dx + W]
                dxv = Xn - X
                dxv = dxv - jnp.round(dxv / R) * R
                Gw = jnp.exp(-jnp.sum(dxv * dxv, axis=-1))
                avgP = 0.5 * Mn[..., 0] * (Pf(M) + Pf(Mn))
                F = F - 0.5 * (Gw * avgP)[..., None] * dxv
        return F

    F = jax.lax.fori_loop(0, 5, dz_body,
                          jnp.zeros((D, H, W, 3), dtype=jnp.float32))

    forward, right_b, up_b = basis_from_velocity_3d(V)
    ca = jnp.cos(sa)[..., None]
    sa_s = jnp.sin(sa)[..., None]
    fwd_part = forward * ca
    sd_e = sd[..., None]
    sense_origin = X + (right_b * params.sensor_offset_right
                        + up_b * params.sensor_offset_up)
    dir_R = (fwd_part + right_b * sa_s) * sd_e
    dir_L = (fwd_part - right_b * sa_s) * sd_e
    dir_U = (fwd_part + up_b * sa_s) * sd_e
    dir_D = (fwd_part - up_b * sa_s) * sd_e
    trail = M[..., 1:2]
    sR = trilinear_sample(trail, sense_origin + dir_R)[..., 0]
    sL = trilinear_sample(trail, sense_origin + dir_L)[..., 0]
    sU = trilinear_sample(trail, sense_origin + dir_U)[..., 0]
    sDn = trilinear_sample(trail, sense_origin + dir_D)[..., 0]
    F = F + ra[..., None] * (right_b * (sR - sL)[..., None]
                             + up_b * (sU - sDn)[..., None])

    F = F - GRAVITY * M[..., 0:1] * jnp.array([0.0, 0.0, 1.0])

    # Inward-pointing repulsion + reflective bounce share the SDF query.
    N_xyz, N_d = border_normal_3d(X, R)
    penetration = jnp.maximum(BORDER_H - N_d, 0.0)[..., None]
    F = F + BORDER_REPULSION * M[..., 0:1] * penetration * N_xyz

    inv_m = 1.0 / jnp.maximum(M[..., 0:1], 1e-12)
    V_new = V + F * DT * inv_m

    close = (N_d <= BORDER_H).astype(jnp.float32)
    vdotN = close * jnp.sum(-N_xyz * V_new, axis=-1)
    bounce = 0.5 * (N_xyz * vdotN[..., None] + N_xyz * jnp.abs(vdotN)[..., None])
    V_new = V_new + bounce
    V_new = jnp.where((N_d < 0)[..., None], 0.0, V_new)

    V_new = V_new * (1.0 + ACCELERATION)
    speed = jnp.linalg.norm(V_new, axis=-1, keepdims=True)
    md_e = jnp.maximum(md[..., None], 1e-6)
    V_new = V_new / jnp.maximum(speed / md_e, 1.0)

    live = (M[..., 0:1] > 1e-9).astype(jnp.float32)
    V_new = live * V_new + (1.0 - live) * V

    return State3D(X, V_new, M)


def update_trail_3d(state: State3D, params: PhysarumParams) -> State3D:
    X, V, M = state
    mass = M[..., 0]
    trail = M[..., 1] * params.decay + mass * params.deposit
    return State3D(X, V, jnp.stack([mass, trail], axis=-1))


def step_3d(state: State3D, params: PhysarumParams) -> State3D:
    return update_trail_3d(simulate_3d(reintegrate_3d(state), params), params)


def init_state_3d(key, D, H, W, radius_frac=0.22) -> State3D:
    """Seed mass as a flat-bottom hemisphere centered on the bottom face.

    Centre at (W/2, H/2, 0); the cube only contains z ≥ 0 so the full ball
    is automatically clipped to its upper half — a dome sitting on the
    z=0 plane. Gravity (−z) keeps it there, giving the agents a printable
    base to grow from. radius_frac defaults higher than the old centered
    sphere so the hemisphere holds comparable total mass.
    """
    pos = position_grid_3d(D, H, W)
    radius = radius_frac * max(D, H, W)
    centre = jnp.array([W * 0.5, H * 0.5, 0.0])
    r = jnp.linalg.norm(pos - centre, axis=-1)
    in_hemi = (r < radius)[..., None]

    rand = jax.random.uniform(key, (D, H, W, 3)) - 0.5
    X = pos
    V = jnp.where(in_hemi, 0.5 * rand, 0.0)
    M = jnp.where(in_hemi,
                  jnp.array([MASS, MASS * 0.1]),
                  jnp.array([1e-6, 0.0]))
    return State3D(X, V, M)


def orbit_view(frame, n_frames, phi_amp=1.0):
    """Camera (theta, phi) for an orbiting MIP gif.

    theta sweeps one full revolution. phi oscillates as `phi_amp * sin(theta)`
    so the camera tilts up/down once per revolution — the object's top is
    visible near theta=π/2 and bottom near 3π/2. With phi_amp=1.0 rad (~57°)
    we stay well below the gimbal at ±π/2.
    """
    theta = 2 * jnp.pi * frame / n_frames
    phi = phi_amp * jnp.sin(theta)
    return theta, phi


def _colorize(mass_2d, trail_2d):
    a = jnp.power(jnp.clip(mass_2d / (FLUID_RHO * 2.0), 0.0, 1.0), 0.1)
    a = a * a * (3 - 2 * a)
    col = jnp.broadcast_to((0.2 * a)[..., None], a.shape + (3,))
    palette = jnp.array([0.2, 0.8, 0.6])
    col = col + (0.5 - 0.5 * jnp.cos(8.0 * palette * trail_2d[..., None]))
    col = jnp.tanh(4.0 * jnp.power(jnp.clip(col, 0.0, 10.0), 1.5))
    return jnp.clip(col, 0.0, 1.0)


def render_raymarch(state: State3D, theta: float = 0.6, phi: float = 0.4,
                    screen: int = 256, n_steps: int = 160,
                    density_scale: float = 12.0,
                    density_floor: float = 0.03,
                    mode: str = "mip") -> jnp.ndarray:
    """Orthographic ray-march with trilinear sampling.

    mode='mip'   : max-density-along-ray (robust; same 2D colormap).
    mode='alpha' : emission-absorption alpha compositing (depth shaded but
                   collapses to a translucent blob unless veins are dense).
    """
    D, H, W = state.M.shape[:3]
    size = float(max(D, H, W))
    centre = jnp.array([W * 0.5, H * 0.5, D * 0.5])

    cp_, sp = jnp.cos(phi), jnp.sin(phi)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    view = jnp.array([cp_ * ct, cp_ * st, sp])
    right = jnp.array([-st, ct, 0.0])
    up = jnp.cross(right, view)
    up = up / jnp.linalg.norm(up)

    uu = jnp.linspace(-1.0, 1.0, screen)
    vv = jnp.linspace(-1.0, 1.0, screen)
    U, Vm = jnp.meshgrid(uu, vv, indexing="xy")
    half = size * 0.7

    origin = (centre[None, None, :]
              - view[None, None, :] * size
              + (U[..., None] * right[None, None, :]
                 + Vm[..., None] * up[None, None, :]) * half)
    ts = jnp.linspace(0.0, 2.0 * size, n_steps, dtype=jnp.float32)
    coords = origin[..., None, :] + ts[None, None, :, None] * view[None, None, None, :]

    samples = trilinear_sample(state.M, coords, mode="clamp")
    mass = samples[..., 0]
    trail = samples[..., 1]

    if mode == "mip":
        return _colorize(mass.max(axis=-1), trail.max(axis=-1))

    ds = ts[1] - ts[0]
    density = jnp.maximum(mass - density_floor, 0.0)
    alpha = 1.0 - jnp.exp(-density * density_scale * ds)
    palette = jnp.array([0.2, 0.8, 0.6])
    hue = 0.5 - 0.5 * jnp.cos(8.0 * palette[None, None, None, :] * trail[..., None])
    a_term = jnp.clip(mass / FLUID_RHO, 0.0, 1.0)[..., None] * 0.25
    color = jnp.clip(a_term + hue, 0.0, 1.5)
    log_oma = jnp.log(jnp.clip(1.0 - alpha, 1e-6, 1.0))
    cum = jnp.cumsum(log_oma, axis=-1)
    cum = jnp.concatenate([jnp.zeros_like(cum[..., :1]), cum[..., :-1]], axis=-1)
    T = jnp.exp(cum)
    weight = (T * alpha)[..., None]
    out = (weight * color).sum(axis=-2)
    out = jnp.tanh(1.6 * jnp.power(out, 1.1))
    return jnp.clip(out, 0.0, 1.0)


def render_mip(state: State3D, theta: float = 0.6, phi: float = 0.0,
               screen: int = 256, n_steps: int = 160) -> jnp.ndarray:
    return render_raymarch(state, theta=theta, phi=phi,
                           screen=screen, n_steps=n_steps, mode="mip")


# === printability (3D) ======================================================

def _ball_struct(r):
    r_ceil = int(np.ceil(r))
    c = np.arange(-r_ceil, r_ceil + 1)
    Z, Y, X = np.meshgrid(c, c, c, indexing="ij")
    return (Z * Z + Y * Y + X * X) <= r * r


def _struct_6conn():
    s = np.zeros((3, 3, 3), dtype=bool)
    s[1, 1, :] = True; s[1, :, 1] = True; s[:, 1, 1] = True
    return s


def _largest_cc_gpu(mask, struct26):
    labels, n = cndi.label(mask, structure=struct26)
    if n <= 1:
        return mask
    counts = cp.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(cp.argmax(counts))


def printability_report(trail, r_min_voxels=1.5, thresholds=None,
                        n_thresholds=12,
                        kept_fraction_min=0.8,
                        opening_ratio_min=0.9):
    """GPU threshold sweep for 3D-printability of a trail volume.

    For each candidate threshold t:
      1) mask = trail > t, then keep only the largest 26-connected component.
      2) EDT on the kept mask — interior voxel value = distance (in voxels)
         to the nearest exterior voxel, i.e. half the local thickness.
      3) Morphological opening with a Euclidean ball of radius r_min_voxels.
      4) Surface area = 6-conn boundary voxel count; volume = mask voxels.

    A row is "printable" iff kept_fraction ≥ kept_fraction_min AND
    opening_ratio ≥ opening_ratio_min. Among printable rows the "best" is
    the one with the highest surface_area/volume (most veiny while still
    admitting a ball of radius r_min_voxels nearly everywhere).

    Returns (df, best_mask, best_row).
    """
    trail_gpu = cp.asarray(trail, dtype=cp.float32)
    nz = trail_gpu[trail_gpu > 0]
    if nz.size == 0:
        raise ValueError("trail is all zeros")
    if thresholds is None:
        lo = float(cp.percentile(nz, 5.0))
        hi = float(cp.percentile(nz, 90.0))
        thresholds = np.linspace(lo, hi, n_thresholds).tolist()

    struct26 = cp.ones((3, 3, 3), dtype=bool)
    struct6 = cp.asarray(_struct_6conn())
    ball = cp.asarray(_ball_struct(r_min_voxels))

    rows = []
    masks = {}
    for t in thresholds:
        mask0 = trail_gpu > t
        vol0 = int(mask0.sum())
        if vol0 == 0:
            rows.append(dict(threshold=float(t), kept_fraction=0.0, volume=0,
                             edt_min=0.0, edt_q01=0.0, edt_median=0.0,
                             opening_ratio=0.0, surface_area=0,
                             sa_to_vol=0.0, printable=False))
            continue

        mask = _largest_cc_gpu(mask0, struct26)
        vol = int(mask.sum())
        kept_fraction = vol / vol0

        edt = cndi.distance_transform_edt(mask)
        edt_in = edt[mask]
        edt_min = float(edt_in.min())
        edt_q01 = float(cp.percentile(edt_in, 1.0))
        edt_med = float(cp.percentile(edt_in, 50.0))

        opened = cndi.binary_opening(mask, structure=ball)
        opening_ratio = float(opened.sum()) / vol

        eroded = cndi.binary_erosion(mask, structure=struct6)
        surface_area = int((mask & ~eroded).sum())
        sa_to_vol = surface_area / vol

        printable = (kept_fraction >= kept_fraction_min
                     and opening_ratio >= opening_ratio_min)
        rows.append(dict(threshold=float(t),
                         kept_fraction=kept_fraction,
                         volume=vol,
                         edt_min=edt_min, edt_q01=edt_q01, edt_median=edt_med,
                         opening_ratio=opening_ratio,
                         surface_area=surface_area,
                         sa_to_vol=sa_to_vol,
                         printable=printable))
        masks[float(t)] = mask

    df = pl.DataFrame(rows).sort("sa_to_vol", descending=True)
    printable_df = df.filter(pl.col("printable"))
    best_row = printable_df.row(0, named=True) if printable_df.height > 0 else None
    best_mask = (cp.asnumpy(masks[best_row["threshold"]]).astype(np.uint8)
                 if best_row is not None else None)
    return df, best_mask, best_row


# === STL export =============================================================

def write_binary_stl(path, verts, faces, normals=None):
    """Write a binary STL file given mesh verts/faces (and optional face normals).

    verts: (N, 3) float32 in mm.
    faces: (M, 3) int32, indexing verts.
    normals: (M, 3) float32 per-face. If None — or if its length doesn't
    match faces (e.g. per-vertex normals from marching_cubes) — face
    normals are computed from vertex winding.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    tri = verts[faces]  # (M, 3, 3)
    if normals is None or len(normals) != len(faces):
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        normals = np.cross(a, b)
        n_len = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(n_len, 1e-12)
    normals = np.asarray(normals, dtype=np.float32)

    n_tri = faces.shape[0]
    with open(path, "wb") as f:
        f.write(b"\0" * 80)               # 80-byte header (ignored)
        f.write(struct.pack("<I", n_tri))
        # Per-triangle record: 12 floats + uint16 attribute byte count = 50 bytes.
        record = np.zeros(n_tri, dtype=[("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                                        ("v1", "<f4", 3),
                                        ("v2", "<f4", 3),
                                        ("v3", "<f4", 3),
                                        ("attr", "<u2")])
        record["nx"] = normals[:, 0]; record["ny"] = normals[:, 1]; record["nz"] = normals[:, 2]
        record["v1"] = tri[:, 0]; record["v2"] = tri[:, 1]; record["v3"] = tri[:, 2]
        f.write(record.tobytes())


def export_stl(mask, path, voxel_size_mm=0.4, level=0.5):
    """Marching cubes on a uint8 mask → binary STL at `path`.

    `voxel_size_mm` sets the physical scale of one voxel. With a 128³ grid
    and voxel_size_mm=0.4 the printed cube is ~5.1 cm on a side.
    Returns the number of triangles written.
    """
    spacing = (voxel_size_mm,) * 3
    # marching_cubes wants a smooth field; uint8 mask gives blocky output —
    # tolerable for prototyping, fine since the mask itself is voxelized.
    verts, faces, normals, _ = marching_cubes(mask.astype(np.float32),
                                              level=level, spacing=spacing)
    write_binary_stl(path, verts, faces, normals)
    return int(faces.shape[0])


# === MAP-Elites =============================================================

# Search bounds for each PhysarumParams field. (Conservative — sensor offsets
# limited to ±1 voxel since they multiply against the local basis.)
PARAM_BOUNDS = dict(
    sd_base=(1.0, 5.0),
    sd_power=(0.0, 3.0),
    sd_scale=(0.0, 3.0),
    sa_base=(0.1, 1.0),
    sa_power=(0.0, 3.0),
    sa_scale=(0.0, 1.0),
    ra_base=(0.0, 0.8),
    ra_power=(0.0, 3.0),
    ra_scale=(0.0, 1.0),
    md_base=(0.6, 1.5),
    md_power=(0.0, 3.0),
    md_scale=(0.0, 0.5),
    sensor_offset_right=(-1.0, 1.0),
    sensor_offset_up=(-1.0, 1.0),
    deposit=(0.05, 0.5),
    decay=(0.85, 0.99),
)
PARAM_FIELDS = list(PARAM_BOUNDS.keys())


def _sample_params_uniform(rng):
    vals = {f: float(rng.uniform(lo, hi)) for f, (lo, hi) in PARAM_BOUNDS.items()}
    return PhysarumParams(**vals)


def _mutate_params(rng, p: PhysarumParams, sigma: float):
    vals = {}
    for f in PARAM_FIELDS:
        lo, hi = PARAM_BOUNDS[f]
        span = hi - lo
        val = getattr(p, f) + rng.normal() * sigma * span
        vals[f] = float(np.clip(val, lo, hi))
    return PhysarumParams(**vals)


def _bd_cell(bd, archive_shape, bd_ranges):
    """Map continuous behavior descriptors to archive cell indices."""
    idx = []
    for v, (lo, hi), n in zip(bd, bd_ranges, archive_shape):
        frac = (v - lo) / max(hi - lo, 1e-12)
        idx.append(int(np.clip(frac * n, 0, n - 1)))
    return tuple(idx)


def _evaluate_params(params: PhysarumParams, grid, frames, steps_per_frame,
                     r_min_voxels, seed):
    """Run a short 3D sim with `params`. Returns (fitness, (bd1, bd2),
    best_mask, M_final).

    fitness = sa_to_vol at the best printable threshold, or -inf if no
    threshold was printable.
    bd1 = opening_ratio, bd2 = edt_median at that threshold.
    Falls back to the threshold with the highest opening_ratio when nothing
    is printable, so unprintable individuals still get a BD coordinate.
    M_final is the (D,H,W,2) (mass,trail) field as a host np.float32 array —
    used downstream to render a rotating-MIP gif for printable archive cells.
    Returned only when the individual is printable (None otherwise), to keep
    the per-eval memory footprint bounded.
    """
    D, H, W = grid
    key = jax.random.PRNGKey(int(seed))
    state = init_state_3d(key, D, H, W)
    for _ in range(frames):
        for _ in range(steps_per_frame):
            state = step_3d_jit(state, params)
    jax.block_until_ready(state.X)

    trail = np.asarray(state.M[..., 1])
    df, best_mask, best_row = printability_report(
        trail, r_min_voxels=r_min_voxels, n_thresholds=10)
    if best_row is not None:
        return (best_row["sa_to_vol"],
                (best_row["opening_ratio"], best_row["edt_median"]),
                best_mask,
                np.asarray(state.M))

    best_idx = int(df.select(pl.col("opening_ratio").arg_max()).item())
    fallback = df.row(best_idx, named=True)
    return (-float("inf"),
            (fallback["opening_ratio"], fallback["edt_median"]),
            None,
            None)


def map_elites_run(cfg: DictConfig, out_dir: Path):
    """Quality-diversity search over PhysarumParams.

    Archive is a 2D grid indexed by (opening_ratio, edt_median).  Each cell
    holds the highest-sa_to_vol individual whose BDs land there. We:
      1) seed the archive with `init_evals` uniform-random params
      2) loop `n_evals - init_evals` more times, each iteration mutating a
         random archive member with Gaussian noise
      3) save final archive as a CSV, drop the best mask per occupied cell
         under <out_dir>/archive/, and emit per-cell STLs for printable
         survivors.
    """
    me = cfg.mapelites
    archive_shape = tuple(me.archive_shape)
    bd_ranges = (tuple(me.bd_ranges.opening_ratio),
                 tuple(me.bd_ranges.edt_median))
    grid = tuple(me.grid)
    rng = np.random.default_rng(int(cfg.seed))

    archive = {}  # (i,j) -> dict(params, fitness, bd, mask, eval_idx)
    arch_dir = out_dir / "archive"
    arch_dir.mkdir(exist_ok=True)

    print(f"\n=== MAP-Elites: {me.n_evals} evals, archive {archive_shape}, "
          f"eval grid {grid}, frames {me.frames}×{me.steps_per_frame} ===")
    t0 = time.perf_counter()
    for i in range(int(me.n_evals)):
        if i < int(me.init_evals) or len(archive) == 0:
            params = _sample_params_uniform(rng)
            origin = "init"
        else:
            parent_cell = list(archive.keys())[rng.integers(len(archive))]
            parent = archive[parent_cell]["params"]
            params = _mutate_params(rng, parent, float(me.mutation_sigma))
            origin = f"mut[{parent_cell[0]},{parent_cell[1]}]"

        fitness, bd, mask, M_final = _evaluate_params(
            params, grid, int(me.frames), int(me.steps_per_frame),
            float(me.r_min_voxels), int(cfg.seed) + i)
        cell = _bd_cell(bd, archive_shape, bd_ranges)

        cur = archive.get(cell)
        replaced = cur is None or fitness > cur["fitness"]
        if replaced:
            archive[cell] = dict(params=params, fitness=fitness, bd=bd,
                                 mask=mask, M_final=M_final, eval_idx=i)
        elapsed = time.perf_counter() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        print(f"  eval {i+1:3d}/{me.n_evals} {origin:>14s}  "
              f"bd=({bd[0]:.3f},{bd[1]:.2f})  cell={cell}  "
              f"fit={fitness:7.3f}  archive={len(archive):3d}  "
              f"{rate:.2f} ev/s  {'*' if replaced else ''}")

    # Persist final archive: per-cell .npy + .stl + a rotating-MIP gif for
    # every printable survivor, plus a single archive.csv with all params.
    rows = []
    n_gif = 0
    gif_screen = max(96, int(cfg.render_size) // 2)
    gif_n_steps = max(64, int(cfg.n_steps) // 2)
    for (i, j), e in archive.items():
        d = dict(cell_i=i, cell_j=j,
                 opening_ratio=e["bd"][0], edt_median=e["bd"][1],
                 fitness=e["fitness"], eval_idx=e["eval_idx"])
        for f in PARAM_FIELDS:
            d[f] = getattr(e["params"], f)
        rows.append(d)
        if e["mask"] is not None:
            stem = f"cell_{i:02d}_{j:02d}_fit{e['fitness']:.3f}"
            np.save(arch_dir / f"{stem}.npy", e["mask"])
            export_stl(e["mask"], arch_dir / f"{stem}.stl",
                       voxel_size_mm=float(cfg.printability.voxel_size_mm))
            if e["M_final"] is not None:
                M_jax = jnp.asarray(e["M_final"])
                D_, H_, W_ = M_jax.shape[:3]
                # render_mip walks state.X/V shapes but only samples state.M;
                # pass dummy X/V of the right shape so the JIT cache reuses.
                dummy = jnp.zeros((D_, H_, W_, 3), dtype=jnp.float32)
                state = State3D(dummy, dummy, M_jax)
                gif_frames = []
                n_views = 24
                for v in range(n_views):
                    theta, phi = orbit_view(v, n_views, phi_amp=1.0)
                    img = np.asarray(render_mip_jit(state,
                                                    theta=float(theta),
                                                    phi=float(phi),
                                                    screen=gif_screen,
                                                    n_steps=gif_n_steps))
                    gif_frames.append((img * 255).astype(np.uint8))
                iio.imwrite(arch_dir / f"{stem}.gif", gif_frames,
                            duration=80, loop=0)
                n_gif += 1

    df = pl.DataFrame(rows).sort("fitness", descending=True)
    df.write_csv(out_dir / "mapelites_archive.csv")
    print(f"\nfilled {len(archive)} / {archive_shape[0]*archive_shape[1]} cells "
          f"in {time.perf_counter() - t0:.1f}s — "
          f"wrote {out_dir / 'mapelites_archive.csv'} and "
          f"{arch_dir} ({sum(e['mask'] is not None for e in archive.values())} "
          f"printable masks, {n_gif} gifs)")


# === Hydra entry ============================================================

step_2d_jit = jax.jit(step_2d)
step_3d_jit = jax.jit(step_3d)
render_2d_jit = jax.jit(render_2d)
render_mip_jit = jax.jit(render_mip, static_argnames=("screen", "n_steps"))
render_rm_jit = jax.jit(render_raymarch, static_argnames=("screen", "n_steps", "mode"))


def _params_from_cfg(cfg: DictConfig) -> PhysarumParams:
    return PhysarumParams(**OmegaConf.to_container(cfg.preset, resolve=True))


def run_2d(cfg: DictConfig, params: PhysarumParams, out_dir: Path):
    H, W = list(cfg.grid)
    key = jax.random.PRNGKey(int(cfg.seed))
    state = init_state_2d(key, H, W)

    print(f"grid {H}×{W}, jit compiling...")
    t0 = time.perf_counter()
    state = step_2d_jit(state, params); jax.block_until_ready(state.X)
    print(f"  first step: {time.perf_counter() - t0:.1f} s (compile+exec)")

    imgs = []
    for f in range(int(cfg.frames)):
        for _ in range(int(cfg.steps_per_frame)):
            state = step_2d_jit(state, params)
        img = np.asarray(render_2d_jit(state))
        imgs.append((img * 255).astype(np.uint8))
        if (f + 1) % 40 == 0:
            print(f"  frame {f+1}/{cfg.frames}  "
                  f"mass={float(state.M[..., 0].sum()):.1f}  "
                  f"trail={float(state.M[..., 1].sum()):.1f}")

    iio.imwrite(out_dir / cfg.out_2d_gif, imgs, duration=40, loop=0)
    print(f"wrote {out_dir / cfg.out_2d_gif}")


def run_3d(cfg: DictConfig, params: PhysarumParams, out_dir: Path):
    D, H, W = list(cfg.grid)
    key = jax.random.PRNGKey(int(cfg.seed))
    state = init_state_3d(key, D, H, W)

    print(f"grid {D}×{H}×{W}, jit compiling...")
    t0 = time.perf_counter()
    state = step_3d_jit(state, params); jax.block_until_ready(state.X)
    print(f"  first step: {time.perf_counter() - t0:.1f} s (compile+exec)")

    t0 = time.perf_counter()
    for _ in range(10):
        state = step_3d_jit(state, params)
    jax.block_until_ready(state.X)
    print(f"step: {(time.perf_counter() - t0) * 100:.1f} ms/step (10 iters)")

    mips, rms = [], []
    for f in range(int(cfg.frames)):
        for _ in range(int(cfg.steps_per_frame)):
            state = step_3d_jit(state, params)
        # MIP: full theta sweep + phi=sin(theta) so we see top & bottom too.
        theta_mip, phi_mip = orbit_view(f, int(cfg.frames), phi_amp=1.0)
        # RM: slower orbit, raised baseline phi, same sin variation.
        theta_rm = 0.6 + 0.02 * f
        phi_rm = 0.5 + 0.8 * jnp.sin(theta_rm)
        mip = np.asarray(render_mip_jit(state,
                                        theta=float(theta_mip),
                                        phi=float(phi_mip),
                                        screen=int(cfg.render_size),
                                        n_steps=int(cfg.n_steps)))
        rm = np.asarray(render_rm_jit(state,
                                      theta=float(theta_rm),
                                      phi=float(phi_rm),
                                      screen=int(cfg.render_size),
                                      n_steps=int(cfg.n_steps),
                                      mode="mip"))
        mips.append((mip * 255).astype(np.uint8))
        rms.append((rm * 255).astype(np.uint8))
        if (f + 1) % 20 == 0:
            print(f"  frame {f+1}/{cfg.frames}  "
                  f"mass={float(state.M[..., 0].sum()):.1f}  "
                  f"trail={float(state.M[..., 1].sum()):.1f}  "
                  f"max|v|={float(jnp.abs(state.V).max()):.3f}")

    iio.imwrite(out_dir / cfg.out_mip, mips, duration=60, loop=0)
    iio.imwrite(out_dir / cfg.out_rm, rms, duration=60, loop=0)
    print(f"wrote {out_dir / cfg.out_mip} and {out_dir / cfg.out_rm}")

    if cfg.printability.enabled:
        pcfg = cfg.printability
        print(f"\nprintability sweep at r_min={pcfg.r_min_voxels} voxels "
              f"({pcfg.n_thresholds} thresholds)...")
        trail = np.asarray(state.M[..., 1])
        ts = time.perf_counter()
        df, best_mask, best_row = printability_report(
            trail,
            r_min_voxels=float(pcfg.r_min_voxels),
            n_thresholds=int(pcfg.n_thresholds),
            kept_fraction_min=float(pcfg.kept_fraction_min),
            opening_ratio_min=float(pcfg.opening_ratio_min))
        print(f"  sweep took {time.perf_counter() - ts:.2f}s")
        with pl.Config(tbl_rows=int(pcfg.n_thresholds), tbl_cols=-1,
                       float_precision=3):
            print(df)
        df.write_csv(out_dir / cfg.out_report)
        print(f"wrote {out_dir / cfg.out_report}")
        if best_row is None:
            print("no printable threshold found — lower r_min_voxels, loosen "
                  "the printability thresholds, or tune params toward thicker "
                  "structure (higher deposit, lower decay, lower sd/ra).")
        else:
            print(f"best printable threshold: t={best_row['threshold']:.3f}  "
                  f"sa/vol={best_row['sa_to_vol']:.3f}  "
                  f"volume={best_row['volume']}  "
                  f"opening_ratio={best_row['opening_ratio']:.3f}  "
                  f"edt_q01={best_row['edt_q01']:.2f}")
            np.save(out_dir / cfg.out_mask, best_mask)
            print(f"wrote {out_dir / cfg.out_mask} "
                  f"(np.uint8, shape {best_mask.shape})")
            if pcfg.export_stl:
                n_tri = export_stl(best_mask, out_dir / cfg.out_stl,
                                   voxel_size_mm=float(pcfg.voxel_size_mm))
                print(f"wrote {out_dir / cfg.out_stl} "
                      f"({n_tri} triangles, voxel={pcfg.voxel_size_mm} mm)")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    print(f"output dir: {out_dir}")

    if cfg.mapelites.enabled:
        if int(cfg.dim) != 3:
            raise ValueError("MAP-Elites is 3D-only.")
        map_elites_run(cfg, out_dir)
        return

    params = _params_from_cfg(cfg)
    if int(cfg.dim) == 2:
        run_2d(cfg, params, out_dir)
    elif int(cfg.dim) == 3:
        run_3d(cfg, params, out_dir)
    else:
        raise ValueError(f"dim must be 2 or 3, got {cfg.dim}")


if __name__ == "__main__":
    main()
