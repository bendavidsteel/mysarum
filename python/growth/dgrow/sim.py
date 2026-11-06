"""Differential growth of thin elastic sheets/shells on the GPU (Warp + torch).

Physics is explicit damped dynamics on a triangle mesh:
  stretch  — strain springs on every edge toward its (growing) rest length
  bend     — Bridson dihedral hinges, rest angle = curl · edge length
  repel    — hash-grid vertex repulsion between non-adjacent vertices
  extras   — radial elastic foundation, normal pressure, gravity, ground plane
  accrete  — deposit along the normal ∝ exposure·(bias + β·H + γ·√K)₊ with
             surface tension; tips outgrow flanks and split (Kaandorp-style)
  plastic  — rest curvature relaxes toward the current one (shapes set)
Growth multiplies each edge's rest length by exp(dt · g_e); edges that grow
past `split_len` are bisected, and intrinsic Delaunay flips keep triangles
well-shaped. Several individuals can be simulated at once as one disjoint
mesh: every per-individual parameter is read through `ind[v]`.
"""

import math
from dataclasses import dataclass, field, fields

import numpy as np
import scipy.sparse as sp
import torch
import warp as wp
from scipy.sparse.csgraph import connected_components

from . import field as FD, mesh as M
from .render import _hemi

wp.config.quiet = True
wp.init()


@dataclass
class Params:
    """Per-individual parameters. Lengths are in units of the initial edge."""
    # mechanics
    k_stretch: float = 20.0
    k_bend: float = 0.5
    curl: float = 0.0            # spontaneous curvature (1/length) of the sheet
    k_repel: float = 20.0
    repel_r: float = 1.0
    damping: float = 0.15
    gravity: float = 0.0
    foundation_k: float = 0.0    # radial tether toward foundation_r about the origin
    foundation_r: float = 0.0
    pressure: float = 0.0
    ground: float = 0.0          # 1 = keep z >= 0
    # growth field: g_v = base + edge·exp(-d/λ) + curv·H + morph·V + height·tanh(z/10)
    g_base: float = 0.0
    g_edge: float = 0.0
    edge_lambda: float = 4.0
    g_curv: float = 0.0
    g_morph: float = 0.0
    g_height: float = 0.0        # growth ∝ z (light from above)
    aniso: float = 0.0           # +1 grow along ∇d (radial), -1 across it (circumferential)
    g_noise: float = 0.0         # static per-vertex growth heterogeneity (inherited on split)
    # neural CA channels (see nca.py): growth, anisotropy, spontaneous curvature
    g_nca: float = 0.0
    nca_aniso: float = 0.0
    nca_curl: float = 0.0
    nca_tropic: float = 0.0      # NCA-gated anisotropic growth toward the light
    g_light: float = 0.0         # growth ∝ light exposure (shaded parts grow less)
    g_ambient: float = 0.0       # growth ∝ ambient (hemisphere) exposure
    # accretion: move the surface along its normal ∝ ambient exposure above a
    # threshold (skeleton deposition; Kaandorp's branching rule), keeping the
    # deposited surface stress-free
    accrete: float = 0.0
    acc_thresh: float = 0.3
    acc_curv: float = 0.0        # tip enhancement: deposit ∝ (acc_bias + acc_curv·H)₊ (diffusion-limited flux)
    acc_bias: float = 1.0        # < 0: only tips sharper than -bias/curv deposit
    acc_light: float = 0.0       # resource mix: 0 = ambient (nutrient), 1 = overhead light
    acc_field: float = 0.0       # then mix toward diffusion-limited flux (dgrow.field): tips screen gaps
    acc_up: float = 0.0          # field supply: 0 = from all sides, 1 = from above only (side walls no-flux)
    acc_smooth: float = 2.0      # diffusion passes on the flux: sets the branch width
    acc_gauss: float = 0.0       # γ in the flux, on sign(K)√|K|: tips grow, cylinder flanks and saddles don't
    acc_tension: float = 0.1     # surface tension on the accreting front (keeps it well-posed)
    acc_solid: float = 0.0       # 0: tension everywhere (old shafts thin away), 1: only where depositing
    acc_thicken: float = 0.0     # uniform deposit everywhere (cambium): older parts end up thicker
    plastic: float = 0.0         # rate the rest curvature relaxes toward the current one
    g_smooth: float = 4.0        # diffusion passes on the growth field
    # NCA-controlled material: polarity axis, bending stiffness, anisotropic curl, accretion
    nca_polar: float = 0.0
    nca_stiff: float = 0.0       # k_bend · exp(nca_stiff · S) per vertex
    nca_acurl: float = 0.0       # spontaneous curvature along the polarity axis (tubes)
    nca_acc: float = 0.0
    nca_hole: float = 0.0        # cut faces where the CA's hole channel peaks (lace, fenestrae; Sim.perforate)
    # Gray-Scott morphogens on the mesh graph (off when morph_steps == 0)
    gs_feed: float = 0.0367
    gs_kill: float = 0.0649
    gs_du: float = 1.0           # on the normalised graph Laplacian (mean - u)
    gs_dv: float = 0.5

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]

    def to_row(self):
        return [float(getattr(self, n)) for n in self.names()]


PNAMES = Params.names()
# Rest curl that makes a hinge stress-free at measured mean curvature H
# (calibrated on an outward icosphere: bulging toward the normal is negative curl).
PCURL = -0.4
PI = {n: i for i, n in enumerate(PNAMES)}


# ── Warp kernels ─────────────────────────────────────────────────────────────

@wp.kernel
def k_springs(x: wp.array(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i),
              rest: wp.array(dtype=float), ind: wp.array(dtype=int),
              P: wp.array2d(dtype=float), ks_col: int, f: wp.array(dtype=wp.vec3)):
    e = wp.tid()
    a = edges[e][0]
    b = edges[e][1]
    d = x[b] - x[a]
    l = wp.length(d)
    r = rest[e]
    if l < 1e-9:
        return
    s = P[ind[a], ks_col] * (l - r) / r
    fd = s * d / l
    wp.atomic_add(f, a, fd)
    wp.atomic_sub(f, b, fd)


@wp.kernel
def k_bending(x: wp.array(dtype=wp.vec3), hinges: wp.array(dtype=wp.vec4i),
              ind: wp.array(dtype=int), P: wp.array2d(dtype=float),
              kb_col: int, curl_col: int, hcurl: wp.array(dtype=float),
              hk: wp.array(dtype=float), f: wp.array(dtype=wp.vec3)):
    h = wp.tid()
    a = hinges[h][0]
    b = hinges[h][1]
    c = hinges[h][2]
    d = hinges[h][3]
    x1 = x[c]
    x2 = x[d]
    x3 = x[a]
    x4 = x[b]
    E = x4 - x3
    el = wp.length(E)
    n1 = wp.cross(x1 - x3, x1 - x4)
    n2 = wp.cross(x2 - x4, x2 - x3)
    l1 = wp.length_sq(n1)
    l2 = wp.length_sq(n2)
    if el < 1e-9 or l1 < 1e-12 or l2 < 1e-12:
        return
    En = E / el
    u1 = el * n1 / l1
    u2 = el * n2 / l2
    u3 = wp.dot(x1 - x4, En) * n1 / l1 + wp.dot(x2 - x4, En) * n2 / l2
    u4 = -wp.dot(x1 - x3, En) * n1 / l1 - wp.dot(x2 - x3, En) * n2 / l2
    m1 = n1 / wp.sqrt(l1)
    m2 = n2 / wp.sqrt(l2)
    cos_t = wp.clamp(wp.dot(m1, m2), -1.0, 1.0)
    sgn = wp.sign(wp.dot(wp.cross(m1, m2), En))
    sin_half = sgn * wp.sqrt(wp.max((1.0 - cos_t) * 0.5, 0.0))
    p = ind[a]
    theta0 = (P[p, curl_col] + hcurl[h]) * el
    s = P[p, kb_col] * hk[h] * el * el / (wp.sqrt(l1) + wp.sqrt(l2)) * (sin_half - wp.sin(0.5 * theta0))
    wp.atomic_add(f, c, s * u1)
    wp.atomic_add(f, d, s * u2)
    wp.atomic_add(f, a, s * u3)
    wp.atomic_add(f, b, s * u4)


@wp.kernel
def k_repel(grid: wp.uint64, x: wp.array(dtype=wp.vec3), ind: wp.array(dtype=int),
            P: wp.array2d(dtype=float), kr_col: int, rr_col: int,
            adj: wp.array(dtype=int), adj_off: wp.array(dtype=int),
            max_r: float, f: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = x[i]
    p = ind[i]
    R = P[p, rr_col]
    k = P[p, kr_col]
    acc = wp.vec3()
    q = wp.hash_grid_query(grid, xi, max_r)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j == i or ind[j] != p:
            continue
        d = xi - x[j]
        l = wp.length(d)
        if l >= R or l < 1e-9:
            continue
        nb = int(0)
        for t in range(adj_off[i], adj_off[i + 1]):
            if adj[t] == j:
                nb = 1
        if nb == 0:
            acc += k * (R - l) / R * d / l
    f[i] = f[i] + acc


@wp.kernel
def k_face_normals(x: wp.array(dtype=wp.vec3), faces: wp.array(dtype=wp.vec3i),
                   vn: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    a = faces[t][0]
    b = faces[t][1]
    c = faces[t][2]
    n = wp.cross(x[b] - x[a], x[c] - x[a]) * 0.5
    wp.atomic_add(vn, a, n)
    wp.atomic_add(vn, b, n)
    wp.atomic_add(vn, c, n)


@wp.kernel
def k_integrate(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                f: wp.array(dtype=wp.vec3), vn: wp.array(dtype=wp.vec3),
                pinned: wp.array(dtype=int), ind: wp.array(dtype=int),
                center: wp.array(dtype=wp.vec3), P: wp.array2d(dtype=float),
                c_damp: int, c_grav: int, c_fk: int, c_fr: int, c_pres: int, c_ground: int,
                dt: float, max_step: float):
    i = wp.tid()
    if pinned[i] != 0:
        v[i] = wp.vec3()
        f[i] = wp.vec3()
        return
    p = ind[i]
    fi = f[i]
    fi += P[p, c_pres] * vn[i]
    fi -= wp.vec3(0.0, 0.0, P[p, c_grav])
    fk = P[p, c_fk]
    if fk > 0.0:
        r = x[i] - center[p]
        rl = wp.length(r)
        if rl > 1e-9:
            fi -= fk * (rl - P[p, c_fr]) * r / rl
    vi = v[i] * (1.0 - P[p, c_damp]) + fi * dt
    step = vi * dt
    sl = wp.length(step)
    if sl > max_step:
        step = step * (max_step / sl)
        vi = step / dt
    xi = x[i] + step
    if P[p, c_ground] > 0.0 and xi[2] < 0.0:
        xi = wp.vec3(xi[0], xi[1], 0.0)
        vi = wp.vec3(vi[0], vi[1], 0.0)
    x[i] = xi
    v[i] = vi
    f[i] = wp.vec3()


@wp.kernel
def k_geodesic(edges: wp.array(dtype=wp.vec2i), rest: wp.array(dtype=float),
               d: wp.array(dtype=float)):
    e = wp.tid()
    a = edges[e][0]
    b = edges[e][1]
    r = rest[e]
    wp.atomic_min(d, b, d[a] + r)
    wp.atomic_min(d, a, d[b] + r)


@wp.func
def _tri_update(da: float, db: float, lab: float, lac: float, lbc: float):
    # Planar unfolding (a at origin, b on +x, c above): distance at c from a
    # virtual point source consistent with da, db, or the edge paths.
    best = wp.min(da + lac, db + lbc)
    if da > 1.0e8 or db > 1.0e8 or lab < 1.0e-9:
        return best
    cx = (lab * lab + lac * lac - lbc * lbc) / (2.0 * lab)
    cy = wp.sqrt(wp.max(lac * lac - cx * cx, 0.0))
    sx = (lab * lab + da * da - db * db) / (2.0 * lab)
    s2 = da * da - sx * sx
    if s2 < 0.0:
        return best
    sy = -wp.sqrt(s2)
    # The straight path must cross segment ab.
    t = -sy / (cy - sy)
    xc = sx + t * (cx - sx)
    if xc < 0.0 or xc > lab:
        return best
    return wp.min(best, wp.sqrt((cx - sx) * (cx - sx) + (cy - sy) * (cy - sy)))


@wp.kernel
def k_geodesic_tri(faces: wp.array(dtype=wp.vec3i), frest: wp.array(dtype=wp.vec3),
                   d: wp.array(dtype=float)):
    # frest[t] = rest lengths of (v0v1, v1v2, v2v0).
    t = wp.tid()
    v0 = faces[t][0]
    v1 = faces[t][1]
    v2 = faces[t][2]
    l01 = frest[t][0]
    l12 = frest[t][1]
    l20 = frest[t][2]
    wp.atomic_min(d, v2, _tri_update(d[v0], d[v1], l01, l20, l12))
    wp.atomic_min(d, v0, _tri_update(d[v1], d[v2], l12, l01, l20))
    wp.atomic_min(d, v1, _tri_update(d[v2], d[v0], l20, l12, l01))


@wp.kernel
def k_light(mesh: wp.uint64, x: wp.array(dtype=wp.vec3), vn: wp.array(dtype=wp.vec3),
            light: wp.vec3, n_rays: int, spread: float, max_t: float, seed: int,
            out: wp.array(dtype=float)):
    # Fraction of jittered rays toward the light that escape, times the cosine
    # term; a sheet is lit from whichever side faces the light.
    i = wp.tid()
    n = vn[i]
    nl = wp.length(n)
    if nl < 1.0e-12:
        out[i] = 0.0
        return
    n = n / nl
    if wp.dot(n, light) < 0.0:
        n = -n
    rng = wp.rand_init(seed, i)
    p = x[i] + n * 0.05
    free = float(0.0)
    for k in range(n_rays):
        j = wp.vec3(wp.randf(rng) - 0.5, wp.randf(rng) - 0.5, wp.randf(rng) - 0.5)
        q = wp.mesh_query_ray(mesh, p, wp.normalize(light + spread * j), max_t)
        if not q.result:
            free += 1.0
    out[i] = free / float(n_rays) * (0.3 + 0.7 * wp.dot(n, light))


@wp.kernel
def k_ambient(mesh: wp.uint64, x: wp.array(dtype=wp.vec3), vn: wp.array(dtype=wp.vec3),
              n_rays: int, max_t: float, seed: int, out: wp.array(dtype=float)):
    # Fraction of cosine-weighted rays over the front hemisphere that travel
    # max_t without hitting the surface: access to nutrient / ambient light.
    i = wp.tid()
    n = vn[i]
    nl = wp.length(n)
    if nl < 1.0e-12:
        out[i] = 0.0
        return
    n = n / nl
    rng = wp.rand_init(seed, i)
    p = x[i] + n * 0.05
    free = float(0.0)
    for k in range(n_rays):
        q = wp.mesh_query_ray(mesh, p, _hemi(n, rng), max_t)
        if not q.result:
            free += 1.0
        else:
            free += q.t / max_t
    out[i] = free / float(n_rays)


def _wp(t, dtype):
    dtype = {float: wp.float32, int: wp.int32}.get(dtype, dtype)
    return wp.from_torch(t.contiguous(), dtype=dtype)


# ── Simulation ───────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    dt: float = 0.05
    substeps: int = 20            # physics substeps per growth tick
    split_len: float = 1.4
    flip_rounds: int = 2
    max_verts: int = 60000        # per individual
    max_step: float = 0.2         # per-substep displacement cap (stability guard)
    morph_steps: int = 0
    nca_steps: int = 2            # neural CA updates per growth tick (when attached)
    light_every: int = 4          # growth ticks between light-exposure ray casts
    amb_rays: int = 16
    amb_dist: float = 12.0        # ambient rays that travel this far count as open
    field_res: int = 96           # voxels per side of each individual's nutrient-field cube
    field_pad: float = 10.0       # far boundary distance beyond the bounding box
    remesh_every: int = 3         # growth ticks between split/flip passes
    hole_after: int = 100         # no perforation before this tick (the seed is too small)
    curv_smooth: int = 8          # diffusion passes on mean curvature
    min_rest: float = 0.4
    seed: int = 0


class Sim:
    """A batch of growing meshes. Build with `Sim.from_meshes`."""

    def __init__(self, pos, faces, ind, params, centers, cfg, device="cuda",
                 pinned=None, sources=None):
        self.dev = torch.device(device)
        if self.dev.type == "cuda" and self.dev.index is None:
            self.dev = torch.device("cuda", torch.cuda.current_device())
        self.wdev = wp.device_from_torch(self.dev)
        self.cfg = cfg
        self.gen = torch.Generator(device=self.dev).manual_seed(cfg.seed)
        self.pos = pos.to(self.dev)
        self.vel = torch.zeros_like(self.pos)
        self.faces = faces.to(self.dev)
        self.n_ind = len(params)
        self.P = torch.tensor([p.to_row() for p in params], dtype=torch.float32, device=self.dev)
        self.centers = centers.to(self.dev).float()
        N = self.pos.shape[0]
        z = torch.zeros(N, device=self.dev)
        # Per-vertex fields carried through splits (new vertex = endpoint mean).
        self.vf = {
            "ind": ind.to(self.dev).float(),
            "pinned": (pinned if pinned is not None else torch.zeros(N)).to(self.dev).float(),
            "source": (sources if sources is not None else torch.zeros(N)).to(self.dev).float(),
            "noise": torch.rand(N, device=self.dev, generator=self.gen) * 2 - 1,
            "U": torch.ones(N, device=self.dev),
            "V": z.clone(),
            "age": z.clone(),
        }
        self.t = 0.0
        self.ticks = 0
        self._bvh_tick = -1
        self.dead = torch.zeros(self.n_ind, dtype=torch.bool, device=self.dev)
        self.active = torch.ones(self.n_ind, device=self.dev)  # growth on/off per individual
        self.budget = torch.full((self.n_ind,), cfg.max_verts, device=self.dev)  # vertex cap per individual
        self.nca = None
        self._cast_tick = {}
        self._rebuild(initial=True)
        # Plastic rest curvature starts at the seed's own, so seeds aren't flattened.
        self.vf["pcurl"] = PCURL * self.mean_curvature()

    @classmethod
    def from_meshes(cls, meshes, params, cfg, spacing=None, device="cuda"):
        """meshes: list of dicts with pos (N,3), faces (F,3), optional pinned/source."""
        pos, faces, ind, pin, src, centers = [], [], [], [], [], []
        off = 0
        for i, m in enumerate(meshes):
            p = torch.as_tensor(m["pos"]).float()
            c = torch.zeros(3) if spacing is None else torch.tensor([i * spacing, 0.0, 0.0])
            pos.append(p + c)
            centers.append(c)
            faces.append(torch.as_tensor(m["faces"]).long() + off)
            ind.append(torch.full((p.shape[0],), i))
            pin.append(torch.as_tensor(m.get("pinned", np.zeros(len(p)))).float())
            src.append(torch.as_tensor(m.get("source", np.zeros(len(p)))).float())
            off += p.shape[0]
        return cls(torch.cat(pos), torch.cat(faces), torch.cat(ind), params,
                   torch.stack(centers), cfg, device, torch.cat(pin), torch.cat(src))

    # ── topology ──
    @property
    def ind(self):
        return self.vf["ind"].round().long()

    def _rebuild(self, initial=False, old=None):
        self.topo = M.Topology(self.faces, self.pos.shape[0])
        if initial:
            e = self.topo.edges
            self.rest = (self.pos[e[:, 1]] - self.pos[e[:, 0]]).norm(dim=1)
        else:
            keys, rest = old
            self.rest = M.lookup_rest(keys, rest, self.topo.keys, torch.ones_like(self.topo.keys, dtype=torch.float32))
        self.ind_i32 = self.ind.int()
        self.pinned_i32 = (self.vf["pinned"] > 0.5).int()
        self._geodesic()

    def _geodesic(self):
        """Intrinsic graph distance from source vertices (or the boundary)."""
        src = self.vf["source"] > 0.5
        # Individuals without explicit sources measure from their boundary.
        has_src = torch.zeros(self.n_ind, dtype=torch.bool, device=self.dev)
        has_src[self.ind[src]] = True
        src = src | (self.topo.boundary & ~has_src[self.ind])
        d = torch.where(src, 0.0, 1e9).float()
        wd = _wp(d, float)
        wf = _wp(self.faces.int(), wp.vec3i)
        he_rest = self.rest[self.topo.he_edge].reshape(-1, 3)  # (v0v1, v1v2, v2v0)
        wfr = _wp(he_rest, wp.vec3)
        for it in range(4096):
            wp.launch(k_geodesic_tri, dim=wf.shape[0], inputs=[wf, wfr, wd], device=self.wdev)
            if it % 16 == 15:
                cur = d.clone()
                wp.launch(k_geodesic_tri, dim=wf.shape[0], inputs=[wf, wfr, wd], device=self.wdev)
                if torch.equal(cur, d):
                    break
        self.dist = torch.where(d > 1e8, 0.0, d)

    def remesh(self):
        cfg = self.cfg
        counts = torch.bincount(self.ind, minlength=self.n_ind)
        allow = (counts[self.ind] < self.budget[self.ind]) & (self.vf["pinned"] < 0.5)
        keys, rest = self.topo.keys, self.rest
        sel = M.select_splits(self.topo, keys, rest, cfg.split_len, allow, self.gen)
        n_split = int(sel.sum())
        if n_split:
            vf = dict(self.vf, vel_x=self.vel)
            self.pos, self.faces, keys, rest, vf = M.split_edges(
                self.pos, self.faces, keys, rest, self.topo, sel, vf)
            self.vel = vf.pop("vel_x")
            self.vf = vf
            # New vertices sit on a pinned/unpinned border only if both ends were pinned.
            self.vf["pinned"] = (self.vf["pinned"] > 0.99).float()
            self.vf["source"] = (self.vf["source"] > 0.99).float()
            self.topo = M.Topology(self.faces, self.pos.shape[0])
        n_flip = 0
        for _ in range(cfg.flip_rounds):
            self.faces, keys, rest, n = M.flip_edges(self.faces, keys, rest, self.topo, self.gen)
            n_flip += n
            if n == 0:
                break
            self.topo = M.Topology(self.faces, self.pos.shape[0])
        if n_split or n_flip:
            self._rebuild(old=(keys, rest))
        return n_split, n_flip

    # ── fields ──
    def vertex_normals(self):
        vn = torch.zeros_like(self.pos)
        wp.launch(k_face_normals, dim=self.faces.shape[0],
                  inputs=[_wp(self.pos, wp.vec3), _wp(self.faces.int(), wp.vec3i), _wp(vn, wp.vec3)],
                  device=self.wdev)
        return vn

    def mean_curvature(self):
        """Signed umbrella-Laplacian curvature (positive on outward bumps)."""
        e = self.topo.edges
        N = self.pos.shape[0]
        nb_sum = torch.zeros_like(self.pos)
        nb_sum.index_add_(0, e[:, 0], self.pos[e[:, 1]])
        nb_sum.index_add_(0, e[:, 1], self.pos[e[:, 0]])
        deg = self.topo.valence.clamp(min=1).float()[:, None]
        lap = nb_sum / deg - self.pos
        n = torch.nn.functional.normalize(self.vertex_normals(), dim=1)
        h = -(lap * n).sum(1) * 4.0  # ≈ 2H·(h²/4)·4 with h≈1
        h[self.topo.boundary] = 0.0
        return self.smooth(h, self.cfg.curv_smooth)

    def gauss_curvature(self):
        """Signed √|K| from the angle defect (1/length, positive at tips, 0 on cylinders)."""
        f = self.faces
        p = self.pos[f]
        ang = []
        area = 0.5 * torch.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], dim=1).norm(dim=1)
        for k in range(3):
            u = torch.nn.functional.normalize(p[:, (k + 1) % 3] - p[:, k], dim=1)
            v = torch.nn.functional.normalize(p[:, (k + 2) % 3] - p[:, k], dim=1)
            ang.append(torch.acos((u * v).sum(1).clamp(-1, 1)))
        N = self.pos.shape[0]
        asum = torch.zeros(N, device=self.dev).index_add_(0, f.reshape(-1), torch.stack(ang, 1).reshape(-1))
        A = torch.zeros(N, device=self.dev).index_add_(0, f.reshape(-1), (area / 3).repeat_interleave(3))
        K = (2 * math.pi - asum) / A.clamp(min=1e-6)
        K[self.topo.boundary] = 0.0
        K = self.smooth(K, self.cfg.curv_smooth)
        return torch.sign(K) * K.abs().sqrt()

    def _graph_lap(self, u):
        e = self.topo.edges
        s = torch.zeros_like(u)
        s.index_add_(0, e[:, 0], u[e[:, 1]])
        s.index_add_(0, e[:, 1], u[e[:, 0]])
        deg = self.topo.valence.clamp(min=1).float()
        return s / (deg[:, None] if u.dim() > 1 else deg) - u

    def attach_nca(self, nca, preroll=32):
        """Drive growth, anisotropy, curl and colour from a neural CA (nca.NCA)."""
        self.nca = nca
        self.vf["nca"] = nca.init_state(self)
        nca.step(self, preroll)

    def morphogens(self, steps):
        P = self.P[self.ind]
        U, V = self.vf["U"], self.vf["V"]
        F, K = P[:, PI["gs_feed"]], P[:, PI["gs_kill"]]
        Du, Dv = P[:, PI["gs_du"]], P[:, PI["gs_dv"]]
        for _ in range(steps):
            uvv = U * V * V
            U = U + Du * self._graph_lap(U) - uvv + F * (1 - U)
            V = V + Dv * self._graph_lap(V) + uvv - (F + K) * V
            U, V = U.clamp(0, 1), V.clamp(0, 1)
        self.vf["U"], self.vf["V"] = U, V

    def seed_morphogens(self, frac=0.01, grow_hops=2, preroll=1500):
        """Random V patches, then let the pattern settle before growth starts."""
        spots = (torch.rand(self.pos.shape[0], device=self.dev, generator=self.gen) < frac).float()
        for _ in range(grow_hops):
            spots = (spots + self._graph_lap(spots) > 0).float()
        self.vf["U"] = torch.where(spots > 0, 0.5, 1.0)
        self.vf["V"] = torch.where(spots > 0, 0.25, 0.0)
        self.morphogens(preroll)

    def _bvh(self):
        """Warp BVH over the current surface, built at most once per tick."""
        if self._bvh_tick != self.ticks:
            self._bvh_pos = self.pos.contiguous()
            self._bvh_idx = self.faces.int().reshape(-1).contiguous()  # must outlive the mesh
            # The median BVH traces this sparse multi-individual layout far faster than LBVH.
            self._bvh_mesh = wp.Mesh(points=_wp(self._bvh_pos, wp.vec3), indices=_wp(self._bvh_idx, int),
                                     bvh_constructor="median")
            self._bvh_tick = self.ticks
        return self._bvh_mesh

    def _cached(self, name):
        """Ray-cast fields are recast every `cfg.light_every` ticks; in between they are
        vertex fields, so vertices from splits inherit their endpoints' values."""
        return name in self.vf and self.ticks - self._cast_tick.get(name, -10**9) < self.cfg.light_every

    def light(self, n_rays=4, spread=0.6):
        """Per-vertex exposure in [0,1] to light from +z, with self-shading."""
        if self._cached("light"):
            return self.vf["light"]
        mesh = self._bvh()
        out = torch.zeros(self.pos.shape[0], device=self.dev)
        wp.launch(k_light, dim=out.shape[0], inputs=[
            mesh.id, _wp(self._bvh_pos, wp.vec3), _wp(self.vertex_normals(), wp.vec3), wp.vec3(0.0, 0.0, 1.0),
            n_rays, spread, 200.0, self.ticks, _wp(out, float)], device=self.wdev)
        self.vf["light"] = out
        self._cast_tick["light"] = self.ticks
        return out

    def ambient(self):
        """Per-vertex openness in [0,1] of the front hemisphere (crevices → 0, tips → 1)."""
        if self._cached("ambient"):
            return self.vf["ambient"]
        mesh = self._bvh()
        out = torch.zeros(self.pos.shape[0], device=self.dev)
        wp.launch(k_ambient, dim=out.shape[0], inputs=[
            mesh.id, _wp(self._bvh_pos, wp.vec3), _wp(self.vertex_normals(), wp.vec3),
            self.cfg.amb_rays, self.cfg.amb_dist, self.ticks + 7919, _wp(out, float)], device=self.wdev)
        self.vf["ambient"] = self.smooth(out, 2)
        self._cast_tick["ambient"] = self.ticks
        return self.vf["ambient"]

    def nutrient(self):
        """Per-vertex diffusion-limited flux, scaled to [0,1] per individual."""
        if self._cached("nutrient"):
            return self.vf["nutrient"]
        # Only individuals still growing that use the field get a voxel cube.
        need = (self.P[:, PI["acc_field"]] > 0) & (self.P[:, PI["accrete"]] > 0) & (self.active > 0)
        q = torch.zeros(self.pos.shape[0], device=self.dev)
        if need.any():
            sub = torch.nonzero(need).squeeze(1)
            cid = torch.full((self.n_ind,), -1, dtype=torch.long, device=self.dev)
            cid[sub] = torch.arange(len(sub), device=self.dev)
            vsel = need[self.ind]
            vidx = torch.nonzero(vsel).squeeze(1)
            remap = torch.full((self.pos.shape[0],), -1, dtype=torch.long, device=self.dev)
            remap[vidx] = torch.arange(len(vidx), device=self.dev)
            faces = remap[self.faces[vsel[self.faces[:, 0]]]]
            n = torch.nn.functional.normalize(self.vertex_normals()[vidx], dim=1)
            q[vidx] = FD.surface_flux(self.pos[vidx], faces, cid[self.ind[vidx]], len(sub), n,
                                      self.P[sub, PI["ground"]] > 0, self.P[sub, PI["acc_up"]],
                                      self.cfg.field_res, self.cfg.field_pad)
        q = self.smooth(q, 2)
        top = torch.zeros(self.n_ind, device=self.dev).scatter_reduce(0, self.ind, q, "amax")
        self.vf["nutrient"] = q / top.clamp(min=1e-9)[self.ind]
        self._cast_tick["nutrient"] = self.ticks
        return self.vf["nutrient"]

    def growth_rate(self):
        P = self.P[self.ind]
        g = P[:, PI["g_base"]] + P[:, PI["g_edge"]] * torch.exp(-self.dist / P[:, PI["edge_lambda"]].clamp(min=1e-3))
        if (self.P[:, PI["g_curv"]] != 0).any():
            g = g + P[:, PI["g_curv"]] * self.mean_curvature().clamp(-2, 2)
        if (self.P[:, PI["g_morph"]] != 0).any():
            g = g + P[:, PI["g_morph"]] * self.vf["V"]
        if (self.P[:, PI["g_height"]] != 0).any():
            z = self.pos[:, 2] - self.centers[self.ind, 2]
            g = g + P[:, PI["g_height"]] * torch.tanh(z / 10.0)
        if (self.P[:, PI["g_light"]] != 0).any():
            g = g + P[:, PI["g_light"]] * self.light()
        if (self.P[:, PI["g_ambient"]] != 0).any():
            g = g + P[:, PI["g_ambient"]] * self.ambient()
        g = g * (1 + P[:, PI["g_noise"]] * self.vf["noise"])
        if self.nca is not None:
            g = g + P[:, PI["g_nca"]] * 0.5 * (self.vf["nca"][:, self.nca.CH_GROW] + 1)
        return g

    def smooth(self, u, iters, rate=0.5):
        """Graph diffusion of a per-vertex field (≈ Gaussian blur of radius √iters edges)."""
        for _ in range(iters):
            u = u + rate * self._graph_lap(u)
        return u

    def smooth_var(self, u, iters):
        """`smooth` with a per-vertex (fractional) number of passes."""
        for k in range(math.ceil(float(iters.max()))):
            u = u + 0.5 * (iters - k).clamp(0, 1) * self._graph_lap(u)
        return u

    def _face_grad(self, d, l01, x2, y2):
        """Intrinsic gradient of a vertex field per face, in the rest frame
        p0=(0,0), p1=(l01,0), p2=(x2,y2)."""
        d = d[self.faces]
        gx = (d[:, 1] - d[:, 0]) / l01
        gy = (d[:, 2] - d[:, 0] - gx * x2) / y2
        return gx, gy

    def _face_frame(self):
        """Embedded frame per face: e1 along v0→v1, e2 in-plane, unit normal."""
        p0, p1, p2 = self.pos[self.faces[:, 0]], self.pos[self.faces[:, 1]], self.pos[self.faces[:, 2]]
        e1 = torch.nn.functional.normalize(p1 - p0, dim=1)
        nrm = torch.nn.functional.normalize(torch.cross(p1 - p0, p2 - p0, dim=1), dim=1)
        return e1, torch.cross(nrm, e1, dim=1), nrm

    def accrete(self, dt_g):
        """Deposit along the normal ∝ ambient exposure; the new surface is stress-free."""
        P = self.P[self.ind]
        amb = self.ambient()
        if (self.P[:, PI["acc_light"]] != 0).any():
            m = P[:, PI["acc_light"]]
            amb = (1 - m) * amb + m * self.light()
        if (self.P[:, PI["acc_field"]] != 0).any():
            m = P[:, PI["acc_field"]]
            amb = (1 - m) * amb + m * self.nutrient()
        th = P[:, PI["acc_thresh"]]
        r = P[:, PI["accrete"]] * ((amb - th) / (1 - th).clamp(min=1e-3)).clamp(min=0)
        if self.nca is not None and (self.P[:, PI["nca_acc"]] != 0).any():
            r = r * (1 + P[:, PI["nca_acc"]] * self.vf["nca"][:, self.nca.CH_ACC]).clamp(min=0)
        # Diffusion-limited flux concentrates on tips: (bias + β·H + γ·√K)₊.
        flux = P[:, PI["acc_bias"]]
        if (self.P[:, PI["acc_curv"]] != 0).any():
            flux = flux + P[:, PI["acc_curv"]] * self.mean_curvature().clamp(-2, 2)
        if (self.P[:, PI["acc_gauss"]] != 0).any():
            flux = flux + P[:, PI["acc_gauss"]] * self.gauss_curvature().clamp(-2, 2)
        r = self.smooth_var(r * flux.clamp(min=0), P[:, PI["acc_smooth"]])
        # Off on and next to pinned vertices: edges touching a pin never split.
        pin = self.vf["pinned"]
        near_pin = (pin + self._graph_lap(pin)) > 0
        on = self.active[self.ind] * ~near_pin * (P[:, PI["accrete"]] > 0)
        r = r * on
        # Curvature flow shrinks every tube it acts on, and the rest lengths lock
        # that in, so on solid tissue it only acts on the depositing front.
        front = (r / P[:, PI["accrete"]].clamp(min=1e-6)).clamp(max=1.0)
        solid = P[:, PI["acc_solid"]]
        tension = P[:, PI["acc_tension"]] * on * (1 - solid + solid * front)
        r = r + P[:, PI["acc_thicken"]] * on
        n = torch.nn.functional.normalize(self.vertex_normals(), dim=1)
        # Umbrella Laplacian = curvature flow along n plus tangential relaxation.
        lap = self._graph_lap(self.pos)
        lap[self.topo.boundary] = 0.0
        # At most a tenth of an edge per tick, so the springs keep up.
        new = self.pos + (dt_g * r).clamp(max=0.1)[:, None] * n \
            + tension.clamp(max=0.5)[:, None] * lap
        e = self.topo.edges
        l0 = (self.pos[e[:, 1]] - self.pos[e[:, 0]]).norm(dim=1)
        l1 = (new[e[:, 1]] - new[e[:, 0]]).norm(dim=1)
        self.rest = (self.rest + (l1 - l0).clamp(-0.1, 0.1)).clamp(min=self.cfg.min_rest)
        self.pos = new

    def grow(self, dt_g):
        """Apply a per-face growth tensor to each rest triangle, average per edge.

        F = I + dt·g·(I + a·(2uuᵀ-I) + b·(2llᵀ-I) + c·(2ppᵀ-I)), u = intrinsic
        ∇dist, l = the light direction in the face plane (NCA tropism), p = the
        intrinsic gradient of the NCA's polarity channel. Mapping whole
        triangles keeps the rest metric realisable, unlike per-edge rates.
        """
        P = self.P[self.ind]
        g = self.smooth_var(self.growth_rate(), P[:, PI["g_smooth"]])
        g = torch.where(self.vf["pinned"] > 0.5, 0.0, g) * self.active[self.ind]
        faces = self.faces
        fi = self.ind[faces[:, 0]]
        PF = self.P[fi]
        nca = self.nca is not None
        S = self.vf["nca"] if nca else None
        he = self.topo.he_edge.reshape(-1, 3)
        l01, l12, l20 = self.rest[he[:, 0]], self.rest[he[:, 1]], self.rest[he[:, 2]]
        x2 = (l01**2 + l20**2 - l12**2) / (2 * l01)
        y2 = torch.sqrt((l20**2 - x2**2).clamp(min=1e-8))
        gx, gy = self._face_grad(self.dist, l01, x2, y2)
        gn = torch.sqrt(gx**2 + gy**2)
        ux, uy = gx / gn.clamp(min=1e-6), gy / gn.clamp(min=1e-6)
        a = PF[:, PI["aniso"]]
        if nca:
            a = (a + PF[:, PI["nca_aniso"]] * S[:, self.nca.CH_ANISO][faces].mean(1)).clamp(-0.9, 0.9)
        a = a * (gn > 0.05)
        # Light direction in the same triangle frame, read off the embedded triangle.
        z = torch.zeros_like(a)
        b, c, lx, ly, px, py, w = z, z, z, z, z, z, z
        if nca and (self.P[:, PI["nca_tropic"]] != 0).any():
            e1, e2, _ = self._face_frame()
            tx, ty = e1[:, 2], e2[:, 2]
            tl = torch.sqrt(tx**2 + ty**2)
            lx, ly = tx / tl.clamp(min=1e-6), ty / tl.clamp(min=1e-6)
            s = 0.5 * (S[:, self.nca.CH_TROPIC][faces].mean(1) + 1)
            b = PF[:, PI["nca_tropic"]] * s * tl.clamp(max=1)
        # Polarity: the CA's own gradient sets a growth axis (auxin-like).
        if nca and (self.P[:, PI["nca_polar"]] != 0).any():
            qx, qy = self._face_grad(S[:, self.nca.CH_POL], l01, x2, y2)
            qn = torch.sqrt(qx**2 + qy**2)
            px, py = qx / qn.clamp(min=1e-6), qy / qn.clamp(min=1e-6)
            w = torch.tanh(qn / 0.02) * (PF[:, PI["nca_polar"]] != 0)
            c = PF[:, PI["nca_polar"]] * S[:, self.nca.CH_PSTR][faces].mean(1) * w
        tot = a.abs() + b + c.abs()
        k = torch.where(tot > 0.9, 0.9 / tot.clamp(min=1e-6), 1.0)
        a, b, c = a * k, b * k, c * k
        # Director for anisotropic curl: polarity where it is defined, else ∇d (doubled angle).
        c2 = (1 - w) * (ux * ux - uy * uy) + w * (px * px - py * py)
        s2 = (1 - w) * 2 * ux * uy + w * 2 * px * py
        th = 0.5 * torch.atan2(s2, c2)
        self._axis = (torch.cos(th), torch.sin(th))
        gf = dt_g * g[faces].mean(1).clamp(-1, 2)
        gf = torch.where((self.vf["pinned"][faces] > 0.5).any(1), 0.0, gf)
        fxx = 1 + gf * (1 + a * (2 * ux * ux - 1) + b * (2 * lx * lx - 1) + c * (2 * px * px - 1))
        fyy = 1 + gf * (1 + a * (2 * uy * uy - 1) + b * (2 * ly * ly - 1) + c * (2 * py * py - 1))
        fxy = gf * 2 * (a * ux * uy + b * lx * ly + c * px * py)
        def mapped(ex, ey):
            return torch.sqrt((fxx * ex + fxy * ey) ** 2 + (fxy * ex + fyy * ey) ** 2)
        # Stretch ratios times the stored lengths: a rest triangle that breaks the
        # triangle inequality then can't feed back into its own edge lengths.
        ex, ey = torch.stack([l01, x2 - l01, -x2], 1), torch.stack([z, y2, -y2], 1)
        old = torch.sqrt(ex**2 + ey**2).clamp(min=1e-6)
        new = torch.stack([l01, l12, l20], 1) * torch.stack(
            [mapped(ex[:, k], ey[:, k]) for k in range(3)], 1) / old
        acc = torch.zeros_like(self.rest).index_add_(0, he.reshape(-1), new.reshape(-1))
        cnt = torch.zeros_like(self.rest).index_add_(0, he.reshape(-1), torch.ones_like(new.reshape(-1)))
        self.rest = (acc / cnt.clamp(min=1)).clamp(min=self.cfg.min_rest)

    def hinge_material(self):
        """Per-hinge extra rest curvature and bending-stiffness multiplier."""
        h = self.topo.hinges
        a, b = h[:, 0], h[:, 1]
        P = self.P[self.ind]
        vc = torch.zeros(self.pos.shape[0], device=self.dev)
        if (self.P[:, PI["plastic"]] > 0).any():
            vc = vc + self.vf["pcurl"] * (P[:, PI["plastic"]] > 0)
        vk = torch.ones_like(vc)
        hc_aniso = torch.zeros(h.shape[0], device=self.dev)
        if self.nca is not None:
            S = self.vf["nca"]
            vc = vc + P[:, PI["nca_curl"]] * S[:, self.nca.CH_CURL]
            vk = torch.exp(P[:, PI["nca_stiff"]] * S[:, self.nca.CH_STIFF])
            if (self.P[:, PI["nca_acurl"]] != 0).any() and hasattr(self, "_axis"):
                # Curvature κ·(t·p)² across each hinge: rolls the sheet about the axis ⊥ p.
                e1, e2, nrm = self._face_frame()
                p3 = self._axis[0][:, None] * e1 + self._axis[1][:, None] * e2
                E = torch.nn.functional.normalize(self.pos[b] - self.pos[a], dim=1)
                proj = 0
                for s in range(2):
                    f = self.topo.he_face[self.topo.edge_he[self.topo.hinge_edge, s]]
                    t = torch.nn.functional.normalize(torch.cross(nrm[f], E, dim=1), dim=1)
                    proj = proj + 0.5 * (t * p3[f]).sum(1) ** 2
                va = P[:, PI["nca_acurl"]] * S[:, self.nca.CH_ACURL]
                hc_aniso = 0.5 * (va[a] + va[b]) * proj
        return 0.5 * (vc[a] + vc[b]) + hc_aniso, 0.5 * (vk[a] + vk[b])

    # ── physics ──
    def physics(self, n):
        cfg = self.cfg
        P = _wp(self.P, float)
        x = _wp(self.pos, wp.vec3)
        v = _wp(self.vel, wp.vec3)
        f = wp.zeros(self.pos.shape[0], dtype=wp.vec3, device=self.wdev)
        vn_t = torch.zeros_like(self.pos)
        vn = _wp(vn_t, wp.vec3)
        edges = _wp(self.topo.edges.int(), wp.vec2i)
        rest = _wp(self.rest, float)
        hinges = _wp(self.topo.hinges.int(), wp.vec4i)
        faces = _wp(self.faces.int(), wp.vec3i)
        ind = _wp(self.ind_i32, int)
        pinned = _wp(self.pinned_i32, int)
        centers = _wp(self.centers, wp.vec3)
        adj = _wp(self.topo.adj, int)
        adj_off = _wp(self.topo.adj_off, int)
        max_r = float(self.P[:, PI["repel_r"]].max())
        grid = wp.HashGrid(128, 128, 128, device=self.wdev)
        need_normals = bool((self.P[:, PI["pressure"]] != 0).any())
        hc_t, hk_t = self.hinge_material()
        hcurl, hk = _wp(hc_t, float), _wp(hk_t, float)
        N = self.pos.shape[0]
        for _ in range(n):
            wp.launch(k_springs, dim=edges.shape[0], inputs=[x, edges, rest, ind, P, PI["k_stretch"], f], device=self.wdev)
            if hinges.shape[0]:
                wp.launch(k_bending, dim=hinges.shape[0], inputs=[x, hinges, ind, P, PI["k_bend"], PI["curl"], hcurl, hk, f], device=self.wdev)
            grid.build(x, max_r)
            wp.launch(k_repel, dim=N, inputs=[grid.id, x, ind, P, PI["k_repel"], PI["repel_r"], adj, adj_off, max_r, f], device=self.wdev)
            if need_normals:
                vn_t.zero_()
                wp.launch(k_face_normals, dim=faces.shape[0], inputs=[x, faces, vn], device=self.wdev)
            wp.launch(k_integrate, dim=N, inputs=[
                x, v, f, vn, pinned, ind, centers, P,
                PI["damping"], PI["gravity"], PI["foundation_k"], PI["foundation_r"],
                PI["pressure"], PI["ground"], cfg.dt, cfg.max_step], device=self.wdev)

    def step(self, dt_g):
        """One growth tick: grow rest lengths, accrete, relax, remesh."""
        cfg = self.cfg
        if cfg.morph_steps:
            self.morphogens(cfg.morph_steps)
        if self.nca is not None:
            self.nca.step(self, cfg.nca_steps)
        self.grow(dt_g)
        if (self.P[:, PI["accrete"]] > 0).any():
            self.accrete(dt_g)
        if (self.P[:, PI["plastic"]] > 0).any():
            k = (dt_g * self.P[self.ind, PI["plastic"]] * self.active[self.ind]).clamp(max=1)
            self.vf["pcurl"] = self.vf["pcurl"] + k * (PCURL * self.mean_curvature() - self.vf["pcurl"])
        self.physics(cfg.substeps)
        self.vf["age"] += dt_g
        self.t += dt_g
        self._quarantine()
        self.ticks += 1
        if self.ticks % cfg.remesh_every:
            return 0, 0
        out = self.remesh()
        if self.nca is not None and self.ticks >= cfg.hole_after and (self.P[:, PI["nca_hole"]] > 0).any():
            self.perforate()
        return out

    def perforate(self, max_frac=0.1):
        """Cut faces whose vertices all carry the NCA hole channel more than
        0.6 - nca_hole above its local mean (a blur over ~6 edges), then keep
        each individual's largest piece. Contrast, not level: a channel that
        saturates everywhere would cut the whole sheet. The new rims are
        boundary, so margin growth (boundary distance) can frill them. At most
        `max_frac` of an individual's faces go per pass."""
        P = self.P[self.ind]
        pin = self.vf["pinned"]
        near_pin = (pin + self._graph_lap(pin)) > 0
        h = self.vf["nca"][:, self.nca.CH_HOLE]
        hot = (h - self.smooth(h, 36) > 0.6 - P[:, PI["nca_hole"]]) \
            & (P[:, PI["nca_hole"]] > 0) & ~near_pin & (self.active[self.ind] > 0)
        cut = hot[self.faces].all(1)
        if not cut.any():
            return
        fi = self.ind[self.faces[:, 0]]
        nf = torch.bincount(fi, minlength=self.n_ind)
        ok = torch.bincount(fi[cut], minlength=self.n_ind) <= max_frac * nf
        cut &= ok[fi]
        if not cut.any():
            return
        faces = self.faces[~cut]
        # Largest connected piece per individual (scipy on the CPU; cuts are rare).
        fc = faces.cpu().numpy()
        N = self.pos.shape[0]
        e = np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]]])
        A = sp.coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(N, N))
        _, lab = connected_components(A, directed=False)
        flab, find = lab[fc[:, 0]], self.ind[faces[:, 0]].cpu().numpy()
        pair = find.astype(np.int64) * N + flab
        u, cnt = np.unique(pair, return_counts=True)
        best = {}
        for k, c in zip(u, cnt):
            i = int(k // N)
            if c > best.get(i, (0, -1))[0]:
                best[i] = (c, k)
        keep = np.isin(pair, [k for _, k in best.values()])
        self._compact(faces[torch.as_tensor(keep, device=self.dev)])

    def retire(self, ids):
        """Take finished individuals out of the batch: {i: individual(i)} for each,
        then drop their vertices so the rest step without carrying them."""
        out = {int(i): self.individual(int(i)) for i in ids}
        gone = torch.zeros(self.n_ind, dtype=torch.bool, device=self.dev)
        gone[torch.as_tensor(list(out), device=self.dev)] = True
        self.active = torch.where(gone, 0.0, self.active)
        keep = ~gone[self.ind[self.faces[:, 0]]]
        if keep.any():  # nothing left to step once the last one goes
            self._compact(self.faces[keep])
        return out

    def _compact(self, faces):
        """Keep only `faces`; drop vertices no face uses. The remap is monotone, so
        edge keys stay sorted."""
        N = self.pos.shape[0]
        used = torch.zeros(N, dtype=torch.bool, device=self.dev)
        used[faces.reshape(-1)] = True
        idx = torch.nonzero(used).squeeze(1)
        remap = torch.full((N,), -1, dtype=torch.long, device=self.dev)
        remap[idx] = torch.arange(len(idx), device=self.dev)
        e = self.topo.edges
        live = used[e[:, 0]] & used[e[:, 1]]
        keys = M.edge_key(remap[e[live, 0]], remap[e[live, 1]])
        rest = self.rest[live]
        self.pos, self.vel = self.pos[idx], self.vel[idx]
        self.vf = {k: v[idx] for k, v in self.vf.items()}
        self.faces = remap[faces]
        self._bvh_tick = -1
        self._rebuild(old=(keys, rest))

    def _quarantine(self):
        """Freeze individuals whose simulation went non-finite; keep the rest going."""
        bad_v = ~torch.isfinite(self.pos).all(1) | ~torch.isfinite(self.vel).all(1)
        bad_e = ~torch.isfinite(self.rest)
        if not (bad_v.any() or bad_e.any()):
            return
        bad = torch.zeros(self.n_ind, dtype=torch.bool, device=self.dev)
        bad[self.ind[bad_v]] = True
        bad[self.ind[self.topo.edges[bad_e, 0]]] = True
        self.dead |= bad
        dv = self.dead[self.ind]
        self.pos = torch.where(dv[:, None] & ~torch.isfinite(self.pos), self.centers[self.ind], self.pos)
        self.vel = torch.where(dv[:, None], 0.0, self.vel)
        self.rest = torch.nan_to_num(self.rest, nan=1.0, posinf=1.0)
        if "nca" in self.vf:
            self.vf["nca"] = torch.nan_to_num(self.vf["nca"])
        self.vf["pinned"] = torch.where(dv, 1.0, self.vf["pinned"])
        self.pinned_i32 = (self.vf["pinned"] > 0.5).int()

    # ── output ──
    def individual(self, i):
        """(pos, faces, vertex-field dict) for individual i, re-indexed."""
        sel = self.ind == i
        idx = torch.nonzero(sel).squeeze(1)
        remap = torch.full((self.pos.shape[0],), -1, dtype=torch.long, device=self.dev)
        remap[idx] = torch.arange(idx.shape[0], device=self.dev)
        fsel = sel[self.faces[:, 0]]
        faces = remap[self.faces[fsel]]
        pos = self.pos[idx] - self.centers[i]
        extra = {k: v[idx] for k, v in self.vf.items()}
        extra["dist"] = self.dist[idx]
        extra["g"] = self.growth_rate()[idx]
        if self.nca is not None:
            extra["color"] = self.nca.color(self.vf["nca"][idx])
            extra["material"] = self.nca.material(self.vf["nca"][idx])
        return pos, faces, extra

    def counts(self):
        return torch.bincount(self.ind, minlength=self.n_ind).tolist()
