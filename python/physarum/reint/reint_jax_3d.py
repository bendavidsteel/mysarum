"""
3D physarum via reintegration tracking — extension of reint_jax.py to a volume.

State: per-voxel particle (X, V, M) on a (D, H, W) grid.
  X: (D,H,W,3) global voxel position (near cell center; pos[...,0]=x, 1=y, 2=z)
  V: (D,H,W,3) velocity
  M: (D,H,W,2) (mass, trail)
  Arrays indexed [z, y, x].

One step = reintegrate (mass-conservative advection over 5x5x5 stencil)
         + simulate  (SPH pressure + sense + border + integrate, parameterized)
         + update_trail (decay + deposit).

Agent behaviour is parameterized in the style of Sage Jenson's "36 Points"
(https://sagejenson.com/36points): each per-cell quantity Q is computed as

    Q = Q_base + Q_scale * trail ^ Q_power

where `trail` is the local trail-chemical value. With Q_scale=0 the parameter
reduces to a constant Q_base. With Q_scale>0 the agents become more sensitive
in dense regions — Sage's "primary special sauce". Pass a PhysarumParams to
`step` / `main(preset=...)`.

Rendering: MIP along an axis (fast preview), or orthographic ray-march with
trilinear sampling, MIP or emission-absorption compositing along the ray.

Run from the Python loop with async dispatch — fori_loop is slower on this
hardware (see memory note `feedback_jax_fori_loop_slow`).
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
from typing import NamedTuple

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np


# --- physics constants (advection / SPH / border; not per-agent) ------------
DT = 1.5
MASS = 2.0
FLUID_RHO = 0.2
DIF = 1.3
ACCELERATION = 0.01
BORDER_H = 5.0
GRAVITY = 0.001


# --- agent-behaviour parameters (Sage Jenson "36 Points" style) -------------

class PhysarumParams(NamedTuple):
    """Per-cell agent behaviour, modulated by local trail concentration.

    For each Q in (sd, sa, ra, md), the effective value at a cell is
        Q = Q_base + Q_scale * trail ^ Q_power
    so Q_scale=0 (or Q_power=0) gives a constant Q_base. trail is clamped
    to a tiny positive minimum so the `pow` is well-defined.

    sd : sensor distance (voxels)        — how far ahead each cone tap reads
    sa : sensor cone half-angle (rad)    — wider = bigger 4-tap spread
    ra : rotation/turn strength          — gain on the (right,up) sense force
    md : move distance / max speed       — per-cell velocity cap (voxels/step)

    sensor_offset_(right|up) : voxels    — bulk displacement of the 4 taps in
        the local (right, up) frame (Sage's "secondary special sauce").

    deposit : amount of mass added to trail each step.
    decay   : multiplicative trail decay each step (1.0 = no decay).

    NOTE: the reintegration stencil is 5×5×5 (±2 voxels), so md * DT must
    stay below ~2.5 to preserve mass conservation. With DT=1.5, keep
    md ≤ ~1.6 in steady state; otherwise mass leaks out of the stencil.
    """
    # Sensor distance (voxels)
    sd_base: float = 3.0
    sd_power: float = 0.0
    sd_scale: float = 0.0
    # Sensor cone half-angle (radians)
    sa_base: float = 0.45
    sa_power: float = 0.0
    sa_scale: float = 0.0
    # Rotation/turn strength
    ra_base: float = 0.2
    ra_power: float = 0.0
    ra_scale: float = 0.0
    # Move distance / max speed (voxels/step)
    md_base: float = 1.0
    md_power: float = 0.0
    md_scale: float = 0.0
    # Sensor offsets in the local (right, up) frame, voxels
    sensor_offset_right: float = 0.0
    sensor_offset_up: float = 0.0
    # Trail deposit (added each step from mass)
    deposit: float = 0.15
    # Trail decay (multiplied each step; 1.0 = no decay)
    decay: float = 0.94


def param_value(base, power, scale, trail):
    """Sage-style trail-modulated parameter: base + scale * trail^power.
    `trail` is an array; output broadcasts to the same shape."""
    return base + scale * jnp.power(jnp.maximum(trail, 1e-9), power)


PRESETS = {
    # constant behaviour everywhere — close to the pre-parameterized defaults
    "default": PhysarumParams(),
    # agents reach further and turn harder where trail is dense → branching webs
    "weblike": PhysarumParams(
        sd_base=2.0, sd_scale=3.0, sd_power=1.0,
        sa_base=0.35, sa_scale=0.4, sa_power=1.5,
        ra_base=0.15, ra_scale=0.6, ra_power=1.5,
        md_base=1.0,
        deposit=0.2, decay=0.93,
    ),
    # tight angles + strong rotation → thin filamentary tendrils
    "tendrils": PhysarumParams(
        sd_base=2.5, sd_scale=2.0, sd_power=2.0,
        sa_base=0.25,
        ra_base=0.4,
        md_base=0.9,
        deposit=0.25, decay=0.92,
    ),
    # offset sensors break symmetry → spiral / chiral patterns
    "spiral": PhysarumParams(
        sd_base=3.0, sd_scale=1.5, sd_power=1.0,
        sa_base=0.5,
        ra_base=0.3,
        md_base=1.0,
        sensor_offset_right=0.8, sensor_offset_up=0.0,
        deposit=0.18, decay=0.95,
    ),
}


class State3D(NamedTuple):
    X: jnp.ndarray  # (D, H, W, 3)
    V: jnp.ndarray  # (D, H, W, 3)
    M: jnp.ndarray  # (D, H, W, 2)


# --- helpers ----------------------------------------------------------------

def position_grid_3d(D, H, W):
    zs = jnp.arange(D, dtype=jnp.float32)
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    Z, Y, X = jnp.meshgrid(zs, ys, xs, indexing="ij")
    return jnp.stack([X, Y, Z], axis=-1)


def _wrap_pad_3d(arr, halo=2):
    return jnp.pad(arr,
                   ((halo, halo), (halo, halo), (halo, halo), (0, 0)),
                   mode="wrap")


def Pf(M):
    return M[..., 0]


def trilinear_sample(field, pos, mode="wrap"):
    """Sample a (D,H,W,C) field at fractional pos with last axis (x, y, z).
    mode='wrap' uses periodic boundary; mode='clamp' clips and zeros outside."""
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


# --- border: sdBox of the cubic domain --------------------------------------

def _border_value_3d(pos, R):
    # -sdBox(p - R/2, R/2): positive inside, negative outside.
    d = jnp.abs(pos - R * 0.5) - R * 0.5
    sd = jnp.linalg.norm(jnp.maximum(d, 0), axis=-1) \
       + jnp.minimum(jnp.max(d, axis=-1), 0)
    return -sd


def border_normal_3d(pos, R):
    """6-tap finite-difference gradient of border.
    Returns (N (..., 3) unit, val (..., ) smoothed)."""
    h = 1.0
    ex = jnp.array([h, 0.0, 0.0]); ey = jnp.array([0.0, h, 0.0]); ez = jnp.array([0.0, 0.0, h])
    f_xp = _border_value_3d(pos + ex, R); f_xm = _border_value_3d(pos - ex, R)
    f_yp = _border_value_3d(pos + ey, R); f_ym = _border_value_3d(pos - ey, R)
    f_zp = _border_value_3d(pos + ez, R); f_zm = _border_value_3d(pos - ez, R)
    grad = jnp.stack([f_xp - f_xm, f_yp - f_ym, f_zp - f_zm], axis=-1) / (2 * h)
    n = grad / (jnp.linalg.norm(grad, axis=-1, keepdims=True) + 1e-12)
    val = (f_xp + f_xm + f_yp + f_ym + f_zp + f_zm) / 6.0 + 1e-4
    return n, val


# --- 3D local basis from velocity (for 4-tap sensing) -----------------------

def basis_from_velocity(V, eps=1e-6):
    """Returns (forward, right, up), each (..., 3). Robust to V ~ 0 and
    forward parallel to world-up by swapping helper axis."""
    speed = jnp.linalg.norm(V, axis=-1, keepdims=True)
    forward = jnp.where(speed > eps, V / jnp.maximum(speed, eps),
                        jnp.array([1.0, 0.0, 0.0]))
    # If forward ~ world_z, use world_x as helper; else use world_z.
    parallel = jnp.abs(forward[..., 2:3]) > 0.95
    aux = jnp.where(parallel,
                    jnp.array([1.0, 0.0, 0.0]),
                    jnp.array([0.0, 0.0, 1.0]))
    right = jnp.cross(forward, aux)
    right = right / (jnp.linalg.norm(right, axis=-1, keepdims=True) + eps)
    up = jnp.cross(right, forward)
    return forward, right, up


# --- reintegration: mass-conservative advection over 5x5x5 stencil ----------

def reintegrate(state: State3D) -> State3D:
    X, V, M = state
    D, H, W = X.shape[:3]
    pos = position_grid_3d(D, H, W)
    R = jnp.array([W, H, D], dtype=jnp.float32)

    Xp = _wrap_pad_3d(X)
    Vp = _wrap_pad_3d(V)
    Mp = _wrap_pad_3d(M)

    half_p = DIF * 0.5
    inv_K3 = 1.0 / (DIF * DIF * DIF)

    # Outer dz axis wrapped in fori_loop to bound peak memory:
    # only 25 stencil ops live at a time (vs all 125 fully unrolled).
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
    # Trail is NOT advected — it's a static chemical field that decays in
    # place and is replenished by mass deposit in update_trail().
    M_new = jnp.stack([M_acc, M[..., 1]], axis=-1)
    return State3D(X_new, V_new, M_new)


# --- simulation: SPH forces + 4-tap 3D sense + border + integrate -----------

def simulate(state: State3D, params: PhysarumParams) -> State3D:
    X, V, M = state
    D, H, W = X.shape[:3]
    R = jnp.array([W, H, D], dtype=jnp.float32)

    # Per-cell trail modulates per-cell sensor / rotation / move parameters
    # (Sage's "primary special sauce" applied at the agent's own location).
    sv = M[..., 1]
    sd = param_value(params.sd_base, params.sd_power, params.sd_scale, sv)
    sa = param_value(params.sa_base, params.sa_power, params.sa_scale, sv)
    ra = param_value(params.ra_base, params.ra_power, params.ra_scale, sv)
    md = param_value(params.md_base, params.md_power, params.md_scale, sv)

    Xp = _wrap_pad_3d(X)
    Mp = _wrap_pad_3d(M)

    # Same memory-conserving outer-axis scan as in reintegrate.
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

    # 3D sense: 4 cone samples (±right, ±up) around forward. Sensor angle
    # and reach are now per-cell arrays driven by params + local trail.
    forward, right_b, up_b = basis_from_velocity(V)
    ca = jnp.cos(sa)[..., None]
    sa_s = jnp.sin(sa)[..., None]
    fwd_part = forward * ca
    sd_e = sd[..., None]
    # Secondary "sauce": pre-displace all 4 sample positions in (right, up).
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

    inv_m = 1.0 / jnp.maximum(M[..., 0:1], 1e-12)
    V_new = V + F * DT * inv_m

    N_xyz, N_d = border_normal_3d(X, R)
    close = (N_d <= BORDER_H).astype(jnp.float32)
    vdotN = close * jnp.sum(-N_xyz * V_new, axis=-1)
    bounce = 0.5 * (N_xyz * vdotN[..., None] + N_xyz * jnp.abs(vdotN)[..., None])
    V_new = V_new + bounce
    V_new = jnp.where((N_d < 0)[..., None], 0.0, V_new)

    # Cap speed at per-cell move distance (replaces the old fixed 1.0 cap).
    V_new = V_new * (1.0 + ACCELERATION)
    speed = jnp.linalg.norm(V_new, axis=-1, keepdims=True)
    md_e = jnp.maximum(md[..., None], 1e-6)
    V_new = V_new / jnp.maximum(speed / md_e, 1.0)

    live = (M[..., 0:1] > 1e-9).astype(jnp.float32)
    V_new = live * V_new + (1.0 - live) * V

    return State3D(X, V_new, M)


def update_trail(state: State3D, params: PhysarumParams) -> State3D:
    """Decay then deposit. Replaces the old trail = mass coupling."""
    X, V, M = state
    mass = M[..., 0]
    trail = M[..., 1] * params.decay + mass * params.deposit
    return State3D(X, V, jnp.stack([mass, trail], axis=-1))


def step(state: State3D, params: PhysarumParams) -> State3D:
    return update_trail(simulate(reintegrate(state), params), params)


# --- init -------------------------------------------------------------------

def init_state(key, D, H, W, radius_frac=0.18) -> State3D:
    # Init: filled sphere of radius=radius_frac*max(D,H,W) centred in the volume.
    # ~2-3% volume fraction matches the 2D disk init's structural density.
    pos = position_grid_3d(D, H, W)
    centre = jnp.array([W * 0.5, H * 0.5, D * 0.5])
    r = jnp.linalg.norm(pos - centre, axis=-1) / max(D, H, W)
    in_sphere = (r < radius_frac)[..., None]

    rand = jax.random.uniform(key, (D, H, W, 3)) - 0.5
    X = pos
    V = jnp.where(in_sphere, 0.5 * rand, 0.0)
    # Seed trail inside the sphere too — gives sensing signal on step 1.
    M = jnp.where(in_sphere,
                  jnp.array([MASS, MASS * 0.1]),
                  jnp.array([1e-6, 0.0]))
    return State3D(X, V, M)


# --- rendering --------------------------------------------------------------

def _colorize(mass_2d, trail_2d):
    """Shared 2D scalar-field → RGB colormap (matches 2D render.glsl)."""
    a = jnp.power(jnp.clip(mass_2d / (FLUID_RHO * 2.0), 0.0, 1.0), 0.1)
    a = a * a * (3 - 2 * a)
    col = jnp.broadcast_to((0.2 * a)[..., None], a.shape + (3,))
    palette = jnp.array([0.2, 0.8, 0.6])
    col = col + (0.5 - 0.5 * jnp.cos(8.0 * palette * trail_2d[..., None]))
    col = jnp.tanh(4.0 * jnp.power(jnp.clip(col, 0.0, 10.0), 1.5))
    return jnp.clip(col, 0.0, 1.0)


def render_mip_axis(state: State3D, axis: int = 0) -> jnp.ndarray:
    """Max-intensity projection along a volume axis (0=z, 1=y, 2=x).
    No rotation; good for a quick fixed-view debug image."""
    mip_m = jnp.max(state.M[..., 0], axis=axis)
    mip_t = jnp.max(state.M[..., 1], axis=axis)
    return _colorize(mip_m, mip_t)


def render_mip(state: State3D, theta: float = 0.6, phi: float = 0.0,
               screen: int = 256, n_steps: int = 160) -> jnp.ndarray:
    """Rotating MIP — orthographic ray cast picking max density per ray.
    Thin wrapper over render_raymarch(mode='mip')."""
    return render_raymarch(state, theta=theta, phi=phi,
                           screen=screen, n_steps=n_steps, mode="mip")


def render_raymarch(state: State3D, theta: float = 0.6, phi: float = 0.4,
                    screen: int = 256, n_steps: int = 160,
                    density_scale: float = 12.0,
                    density_floor: float = 0.03,
                    mode: str = "mip") -> jnp.ndarray:
    """Orthographic ray-march with trilinear sampling, rotatable view.

    mode='mip'  : max-density-along-ray (robust to dispersed mass; uses the
                  same 2D colormap as render_mip — recommended default).
    mode='alpha': emission-absorption alpha compositing (depth-shaded, but
                  collapses to a translucent blob unless veins are dense).
    """
    D, H, W = state.M.shape[:3]
    size = float(max(D, H, W))
    centre = jnp.array([W * 0.5, H * 0.5, D * 0.5])

    cp, sp = jnp.cos(phi), jnp.sin(phi)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    view = jnp.array([cp * ct, cp * st, sp])
    right = jnp.array([-st, ct, 0.0])
    up = jnp.cross(right, view)
    up = up / jnp.linalg.norm(up)

    uu = jnp.linspace(-1.0, 1.0, screen)
    vv = jnp.linspace(-1.0, 1.0, screen)
    U, Vm = jnp.meshgrid(uu, vv, indexing="xy")
    half = size * 0.7  # zoom

    origin = (centre[None, None, :]
              - view[None, None, :] * size
              + (U[..., None] * right[None, None, :]
                 + Vm[..., None] * up[None, None, :]) * half)
    ts = jnp.linspace(0.0, 2.0 * size, n_steps, dtype=jnp.float32)
    coords = origin[..., None, :] + ts[None, None, :, None] * view[None, None, None, :]

    samples = trilinear_sample(state.M, coords, mode="clamp")  # (screen, screen, n_steps, 2)
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


# --- main -------------------------------------------------------------------

step_jit = jax.jit(step)
render_mip_jit = jax.jit(render_mip, static_argnames=("screen", "n_steps"))
render_rm_jit = jax.jit(render_raymarch, static_argnames=("screen", "n_steps", "mode"))


def benchmark(state, params, n=20):
    state = step_jit(state, params); jax.block_until_ready(state.X)
    t0 = time.perf_counter()
    for _ in range(n):
        state = step_jit(state, params)
    jax.block_until_ready(state.X)
    t1 = time.perf_counter()
    print(f"step: {(t1 - t0) * 1000 / n:.1f} ms/step  ({n} iters)")
    return state


def main(D=128, H=128, W=128, frames=80, steps_per_frame=4, seed=0,
         render_size=192, n_steps=128, preset="weblike",
         out_mip="reint_3d_mip.gif", out_rm="reint_3d_raymarch.gif"):
    params = PRESETS[preset] if isinstance(preset, str) else preset
    key = jax.random.PRNGKey(seed)
    state = init_state(key, D, H, W)

    print(f"grid {D}x{H}x{W}, preset='{preset}', jit compiling...")
    t0 = time.perf_counter()
    state = step_jit(state, params); jax.block_until_ready(state.X)
    print(f"  first step: {time.perf_counter() - t0:.1f} s (compile+exec)")
    state = benchmark(state, params, n=10)

    # Two complementary orbits so the two GIFs reveal different facets:
    #   mip  — equatorial spin (phi=0), full 2π over the GIF.
    #   rm   — tilted-from-above spin (phi=0.5), slower; depth from parallax.
    mips, rms = [], []
    for f in range(frames):
        for _ in range(steps_per_frame):
            state = step_jit(state, params)
        theta_mip = 2 * jnp.pi * f / frames
        theta_rm = 0.6 + 0.02 * f
        mip = np.asarray(render_mip_jit(state,
                                        theta=float(theta_mip), phi=0.0,
                                        screen=render_size, n_steps=n_steps))
        rm = np.asarray(render_rm_jit(state,
                                      theta=theta_rm, phi=0.5,
                                      screen=render_size, n_steps=n_steps,
                                      mode="mip"))
        mips.append((mip * 255).astype(np.uint8))
        rms.append((rm * 255).astype(np.uint8))
        if (f + 1) % 20 == 0:
            print(f"  frame {f+1}/{frames}  "
                  f"mass={float(state.M[..., 0].sum()):.1f}  "
                  f"trail={float(state.M[..., 1].sum()):.1f}  "
                  f"max|v|={float(jnp.abs(state.V).max()):.3f}")

    iio.imwrite(out_mip, mips, duration=60, loop=0)
    iio.imwrite(out_rm, rms, duration=60, loop=0)
    print(f"wrote {out_mip} and {out_rm}")


if __name__ == "__main__":
    main()
