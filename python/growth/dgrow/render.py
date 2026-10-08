"""Ray-traced clay renders of a triangle mesh (Warp BVH): AO + soft key-light shadow.

Front and back faces get different albedos so the two sides of a sheet read
apart; an optional per-vertex scalar tints the surface through a colormap.
"""

import math

import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw

wp.config.quiet = True
wp.init()


@wp.func
def _hemi(n: wp.vec3, rng: wp.uint32):
    # Cosine-weighted direction around n.
    r1 = wp.randf(rng)
    r2 = wp.randf(rng)
    phi = 6.2831853 * r1
    s = wp.sqrt(r2)
    t = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.9:
        t = wp.vec3(0.0, 1.0, 0.0)
    u = wp.normalize(wp.cross(n, t))
    v = wp.cross(n, u)
    return wp.normalize(u * (wp.cos(phi) * s) + v * (wp.sin(phi) * s) + n * wp.sqrt(1.0 - r2))


@wp.kernel
def k_render(mesh: wp.uint64, vn: wp.array(dtype=wp.vec3), vcol: wp.array(dtype=wp.vec3),
             use_vcol: int, eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
             half_w: float, ortho: int, W: int, H: int, spp: int, ao_n: int, ao_dist: float,
             light: wp.vec3, front: wp.vec3, back: wp.vec3, bg0: wp.vec3, bg1: wp.vec3,
             seed: int, img: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    px = tid % W
    py = tid // W
    rng = wp.rand_init(seed, tid)
    acc = wp.vec3()
    for s in range(spp):
        sx = (float(px) + wp.randf(rng)) / float(W) * 2.0 - 1.0
        sy = 1.0 - (float(py) + wp.randf(rng)) / float(H) * 2.0
        sx = sx * half_w
        sy = sy * half_w * float(H) / float(W)
        o = eye
        d = fwd
        if ortho != 0:
            o = eye + right * sx + up * sy
        else:
            d = wp.normalize(fwd + right * sx + up * sy)
        q = wp.mesh_query_ray(mesh, o, d, 1.0e6)
        if not q.result:
            acc += bg0 * (1.0 - 0.5 * (sy / half_w + 1.0)) + bg1 * 0.5 * (sy / half_w + 1.0)
            continue
        f = q.face
        m = wp.mesh_get(mesh)
        i0 = m.indices[f * 3 + 0]
        i1 = m.indices[f * 3 + 1]
        i2 = m.indices[f * 3 + 2]
        w2 = 1.0 - q.u - q.v
        n = wp.normalize(vn[i0] * q.u + vn[i1] * q.v + vn[i2] * w2)
        albedo = front
        if use_vcol != 0:
            albedo = vcol[i0] * q.u + vcol[i1] * q.v + vcol[i2] * w2
        if wp.dot(n, d) > 0.0:
            n = -n
            if use_vcol == 0:
                albedo = back
            else:
                albedo = albedo * 0.8
        p = o + d * q.t + n * 1.0e-3 * ao_dist
        occ = float(0.0)
        for k in range(ao_n):
            r = _hemi(n, rng)
            qa = wp.mesh_query_ray(mesh, p, r, ao_dist)
            if qa.result:
                occ += 1.0 - qa.t / ao_dist
        ao = 1.0 - occ / float(ao_n)
        ndl = wp.max(wp.dot(n, light), 0.0)
        sh = float(1.0)
        if ndl > 0.0:
            jl = wp.normalize(light + 0.08 * wp.vec3(wp.randf(rng) - 0.5, wp.randf(rng) - 0.5, wp.randf(rng) - 0.5))
            qs = wp.mesh_query_ray(mesh, p, jl, 1.0e6)
            if qs.result:
                sh = 0.0
        hv = wp.normalize(light - d)
        spec = wp.pow(wp.max(wp.dot(n, hv), 0.0), 40.0) * 0.12 * sh
        rim = wp.pow(1.0 - wp.abs(wp.dot(n, d)), 3.0) * 0.15
        lit = albedo * (0.35 * ao + 0.85 * ndl * sh * (0.4 + 0.6 * ao)) + wp.vec3(spec + rim * ao)
        acc += lit
    c = acc / float(spp)
    img[tid] = wp.vec3(wp.pow(wp.clamp(c[0], 0.0, 1.0), 1.0 / 2.2),
                       wp.pow(wp.clamp(c[1], 0.0, 1.0), 1.0 / 2.2),
                       wp.pow(wp.clamp(c[2], 0.0, 1.0), 1.0 / 2.2))


def _vertex_normals(pos, faces):
    n = torch.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]], dim=1)
    vn = torch.zeros_like(pos)
    for i in range(3):
        vn.index_add_(0, faces[:, i], n)
    return torch.nn.functional.normalize(vn, dim=1)


def _cmap(x, name="magma"):
    import matplotlib
    return torch.as_tensor(matplotlib.colormaps[name](x.cpu().numpy())[:, :3], dtype=torch.float32)


FRONT = (0.80, 0.62, 0.48)
BACK = (0.55, 0.30, 0.32)


def render(pos, faces, W=768, H=768, azim=35.0, elev=30.0, spp=16, ao_n=4,
           scalar=None, cmap="magma", ortho=False, fov=28.0, seed=0, center=None,
           radius=None, front=FRONT, back=BACK, light=(0.4, -0.3, 0.85)):
    """Render a mesh to an (H, W, 3) uint8 array. pos/faces are torch tensors on CUDA."""
    dev = pos.device
    wdev = wp.device_from_torch(dev)
    pos = pos.float().contiguous()
    faces = faces.int().contiguous()
    if center is None:
        lo, hi = pos.min(0).values, pos.max(0).values
        center = 0.5 * (lo + hi)
        radius = 0.5 * (hi - lo).norm().item()
    center = torch.as_tensor(center, dtype=torch.float32, device=dev)
    a, e = math.radians(azim), math.radians(elev)
    dirn = torch.tensor([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    fwd = -dirn
    right = torch.nn.functional.normalize(torch.cross(fwd, torch.tensor([0.0, 0.0, 1.0]), dim=0), dim=0)
    up = torch.cross(right, fwd, dim=0)
    if ortho:
        half_w = radius * 1.05
        eye = center.cpu() + dirn * radius * 4
    else:
        half_w = math.tan(math.radians(fov) / 2)
        eye = center.cpu() + dirn * (radius * 1.05 / math.sin(math.radians(fov) / 2))
    wmesh = wp.Mesh(points=wp.from_torch(pos, dtype=wp.vec3), indices=wp.from_torch(faces.reshape(-1), dtype=wp.int32))
    vn = _vertex_normals(pos, faces.long())
    use_vcol = scalar is not None
    if use_vcol:
        s = scalar.float()
        lo_, hi_ = torch.quantile(s, 0.02), torch.quantile(s, 0.98)
        vcol = _cmap(((s - lo_) / (hi_ - lo_ + 1e-9)).clamp(0, 1), cmap).to(dev) ** 2.2
    else:
        vcol = torch.zeros_like(pos)
    img = wp.zeros(W * H, dtype=wp.vec3, device=wdev)
    lin = lambda c: wp.vec3(*[x**2.2 for x in c])
    L = torch.nn.functional.normalize(torch.tensor(light), dim=0).tolist()
    wp.launch(k_render, dim=W * H, inputs=[
        wmesh.id, wp.from_torch(vn, dtype=wp.vec3), wp.from_torch(vcol.contiguous(), dtype=wp.vec3), int(use_vcol),
        wp.vec3(*eye.tolist()), wp.vec3(*fwd.tolist()), wp.vec3(*right.tolist()), wp.vec3(*up.tolist()),
        float(half_w), int(ortho), W, H, spp, ao_n, float(radius * 0.15),
        wp.vec3(*L), lin(front), lin(back), lin((0.93, 0.92, 0.90)), lin((0.75, 0.77, 0.80)), seed, img], device=wdev)
    out = wp.to_torch(img).reshape(H, W, 3)
    return (out * 255).clamp(0, 255).byte().cpu().numpy()


def views(pos, faces, size=512, label=None, **kw):
    """Three-view strip (iso, side, top) as one PIL image."""
    ims = [render(pos, faces, W=size, H=size, azim=az, elev=el, **kw)
           for az, el in [(35, 30), (0, 2), (0, 89)]]
    im = Image.fromarray(np.concatenate(ims, 1))
    if label:
        ImageDraw.Draw(im).text((8, 6), label, fill=(30, 30, 30))
    return im


def turntable(pos, faces, path, n=72, size=512, elev=25.0, fps=24, **kw):
    import imageio.v2 as imageio
    lo, hi = pos.min(0).values, pos.max(0).values
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo).norm().item()
    frames = [render(pos, faces, W=size, H=size, azim=360 * i / n, elev=elev, center=c,
                     radius=r, spp=8, **kw) for i in range(n)]
    imageio.mimsave(path, frames, fps=fps, **({"loop": 0} if path.endswith(".gif") else {}))
