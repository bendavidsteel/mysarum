"""Diffusion-limited nutrient field around each individual.

Laplace's equation on a voxel cube per individual: the surface absorbs
(c = 0), the far boundary supplies (c = 1), and grounded individuals get a
no-flux floor. The flux into the surface is the harmonic measure that drives
DLA and Mullins–Sekerka tip splitting: tips catch the nutrient and screen the
gaps between branches, which ambient occlusion only does along sight lines.
`up` closes the side walls (no-flux) so the supply comes from above, as for
a coral under a current or a plant reaching for light: upward tips win.

Solved coarse to fine (cascadic Jacobi): each level starts from the
upsampled coarser solution, so a few dozen sweeps per level suffice.
"""

import torch
import torch.nn.functional as F

LEVELS = (8, 4, 2, 1)       # coarsening factors, coarsest first
SWEEPS = (200, 60, 40, 30)  # Jacobi sweeps per level


def _jacobi(c, wall, floor, up, sweeps):
    """c, wall: (B, 1, Z, Y, X). Ghost layers hold c = 1, except a mirrored
    floor and side walls blended toward mirrored by `up` (B,)."""
    fl = floor[:, None, None, None]
    u = up[:, None, None, None]
    keep = ~wall
    for _ in range(sweeps):
        p = F.pad(c, (1, 1, 1, 1, 1, 1), value=1.0)
        p[:, :, 0] = torch.where(fl, p[:, :, 1], p[:, :, 0])
        p[:, :, :, 0] = u * p[:, :, :, 1] + 1 - u
        p[:, :, :, -1] = u * p[:, :, :, -2] + 1 - u
        p[:, :, :, :, 0] = u * p[:, :, :, :, 1] + 1 - u
        p[:, :, :, :, -1] = u * p[:, :, :, :, -2] + 1 - u
        s = p[:, :, :-2, 1:-1, 1:-1] + p[:, :, 2:, 1:-1, 1:-1] + p[:, :, 1:-1, :-2, 1:-1] \
            + p[:, :, 1:-1, 2:, 1:-1] + p[:, :, 1:-1, 1:-1, :-2] + p[:, :, 1:-1, 1:-1, 2:]
        c = s * keep / 6
    return c


def boxes(pos, ind, n_ind, res, pad, floor, h_min=1.0):
    """Per-individual voxel cube: origin (B, 3) and voxel size (B,).
    Voxels are no smaller than `h_min` so splatted vertices leave no gaps."""
    lo = torch.full((n_ind, 3), float("inf"), device=pos.device).scatter_reduce(
        0, ind[:, None].expand(-1, 3), pos, "amin")
    hi = torch.full((n_ind, 3), -float("inf"), device=pos.device).scatter_reduce(
        0, ind[:, None].expand(-1, 3), pos, "amax")
    side = (hi - lo).max(1).values + 2 * pad
    h = (side / res).clamp(min=h_min)
    origin = (lo + hi) / 2 - res * h[:, None] / 2
    # Grounded: the floor sits half a voxel under the lowest vertex.
    origin[:, 2] = torch.where(floor, lo[:, 2] - h / 2, origin[:, 2])
    return origin, h


def solve(pos, faces, ind, n_ind, floor, up, res=96, pad=10.0):
    """-> concentration (B, 1, res, res, res), origin, voxel size."""
    dev = pos.device
    origin, h = boxes(pos, ind, n_ind, res, pad, floor)
    # Splat vertices and face centroids as absorbing voxels.
    pts = torch.cat([pos, pos[faces].mean(1)])
    pid = torch.cat([ind, ind[faces[:, 0]]])
    v = ((pts - origin[pid]) / h[pid, None]).floor().long().clamp(0, res - 1)
    lin = ((pid * res + v[:, 2]) * res + v[:, 1]) * res + v[:, 0]
    wall = torch.zeros(n_ind * res**3, dtype=torch.bool, device=dev)
    wall[lin] = True
    wall = wall.view(n_ind, 1, res, res, res)
    c = None
    for f, sweeps in zip(LEVELS, SWEEPS):
        w = F.max_pool3d(wall.float(), f) > 0 if f > 1 else wall
        c = torch.ones_like(w, dtype=torch.float32) if c is None else \
            F.interpolate(c, size=w.shape[2:], mode="trilinear", align_corners=False)
        c = _jacobi(c.masked_fill(w, 0.0), w, floor, up, sweeps)
    return c, origin, h


def sample(c, origin, h, pts, pid):
    """Trilinear c at points (N, 3) of individuals pid; voxel centres at (i + ½)h."""
    res = c.shape[-1]
    u = (pts - origin[pid]) / h[pid, None] - 0.5
    i0 = u.floor().clamp(0, res - 2)
    f = (u - i0).clamp(0, 1)
    i0 = i0.long()
    flat = c.reshape(-1)
    out = torch.zeros(len(pts), device=c.device)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (f[:, 2] if dz else 1 - f[:, 2]) * (f[:, 1] if dy else 1 - f[:, 1]) \
                    * (f[:, 0] if dx else 1 - f[:, 0])
                lin = ((pid * res + i0[:, 2] + dz) * res + i0[:, 1] + dy) * res + i0[:, 0] + dx
                out = out + w * flat[lin]
    return out


def surface_flux(pos, faces, ind, n_ind, normals, floor, up, res=96, pad=10.0):
    """Flux into the surface ≈ c one and a half voxels out along the normal / that distance."""
    c, origin, h = solve(pos, faces, ind, n_ind, floor, up, res, pad)
    d = 1.5 * h[ind]
    return sample(c, origin, h, pos + normals * d[:, None], ind) / d
