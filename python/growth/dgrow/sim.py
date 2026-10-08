"""Differential growth of thin elastic sheets/shells on the GPU (Warp + torch).

Physics is explicit damped dynamics on a triangle mesh:
  stretch  — strain springs on every edge toward its (growing) rest length
  bend     — Bridson dihedral hinges, rest angle = curl · edge length
  repel    — hash-grid vertex repulsion between non-adjacent vertices
  extras   — radial elastic foundation, normal pressure, gravity, ground plane
Growth multiplies each edge's rest length by exp(dt · g_e); edges that grow
past `split_len` are bisected, and intrinsic Delaunay flips keep triangles
well-shaped. Several individuals can be simulated at once as one disjoint
mesh: every per-individual parameter is read through `ind[v]`.
"""

import math
from dataclasses import dataclass, field, fields

import numpy as np
import torch
import warp as wp

from . import mesh as M

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
              kb_col: int, curl_col: int, vcurl: wp.array(dtype=float),
              f: wp.array(dtype=wp.vec3)):
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
    theta0 = (P[p, curl_col] + 0.5 * (vcurl[a] + vcurl[b])) * el
    s = P[p, kb_col] * el * el / (wp.sqrt(l1) + wp.sqrt(l2)) * (sin_half - wp.sin(0.5 * theta0))
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
    remesh_every: int = 3         # growth ticks between split/flip passes
    g_smooth: int = 4             # diffusion passes on the growth field
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
        self.dead = torch.zeros(self.n_ind, dtype=torch.bool, device=self.dev)
        self.active = torch.ones(self.n_ind, device=self.dev)  # growth on/off per individual
        self.nca = None
        self._light_tick = 0
        self._rebuild(initial=True)

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
        allow = (counts[self.ind] < cfg.max_verts) & (self.vf["pinned"] < 0.5)
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

    def light(self, n_rays=4, spread=0.6):
        """Per-vertex exposure in [0,1] to light from +z, with self-shading.

        Recast every `cfg.light_every` ticks; in between it is a vertex field,
        so vertices from splits inherit their endpoints' exposure.
        """
        tick = getattr(self, "ticks", 0)
        if "light" in self.vf and tick - self._light_tick < self.cfg.light_every:
            return self.vf["light"]
        pos = self.pos.contiguous()
        idx = self.faces.int().reshape(-1).contiguous()  # must outlive the mesh
        # The median BVH traces this sparse multi-individual layout far faster than LBVH.
        mesh = wp.Mesh(points=_wp(pos, wp.vec3), indices=_wp(idx, int), bvh_constructor="median")
        out = torch.zeros(pos.shape[0], device=self.dev)
        wp.launch(k_light, dim=pos.shape[0], inputs=[
            mesh.id, _wp(pos, wp.vec3), _wp(self.vertex_normals(), wp.vec3), wp.vec3(0.0, 0.0, 1.0),
            n_rays, spread, 200.0, tick, _wp(out, float)], device=self.wdev)
        self.vf["light"] = out
        self._light_tick = tick
        return out

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
        g = g * (1 + P[:, PI["g_noise"]] * self.vf["noise"])
        if self.nca is not None:
            g = g + P[:, PI["g_nca"]] * 0.5 * (self.vf["nca"][:, self.nca.CH_GROW] + 1)
        return g

    def smooth(self, u, iters, rate=0.5):
        """Graph diffusion of a per-vertex field (≈ Gaussian blur of radius √iters edges)."""
        for _ in range(iters):
            u = u + rate * self._graph_lap(u)
        return u

    def grow(self, dt_g):
        """Apply a per-face growth tensor to each rest triangle, average per edge.

        F = I + dt·g·(I + a·(2uuᵀ-I) + b·(2llᵀ-I)), u = intrinsic ∇dist, l = the
        light direction in the face plane (NCA tropism). Mapping whole triangles
        keeps the rest metric realisable, unlike per-edge rates.
        """
        cfg = self.cfg
        g = self.smooth(self.growth_rate(), cfg.g_smooth)
        g = torch.where(self.vf["pinned"] > 0.5, 0.0, g) * self.active[self.ind]
        faces = self.faces
        he = self.topo.he_edge.reshape(-1, 3)
        l01, l12, l20 = self.rest[he[:, 0]], self.rest[he[:, 1]], self.rest[he[:, 2]]
        x2 = (l01**2 + l20**2 - l12**2) / (2 * l01)
        y2 = torch.sqrt((l20**2 - x2**2).clamp(min=1e-8))
        d = self.dist[faces]
        # ∇d in the frame p0=(0,0), p1=(l01,0), p2=(x2,y2).
        gx = (d[:, 1] - d[:, 0]) / l01
        gy = (d[:, 2] - d[:, 0] - gx * x2) / y2
        gn = torch.sqrt(gx**2 + gy**2)
        ux, uy = gx / gn.clamp(min=1e-6), gy / gn.clamp(min=1e-6)
        a = self.P[self.ind[faces[:, 0]], PI["aniso"]]
        if self.nca is not None:
            s = self.vf["nca"][:, self.nca.CH_ANISO][faces].mean(1)
            a = (a + self.P[self.ind[faces[:, 0]], PI["nca_aniso"]] * s).clamp(-0.9, 0.9)
        a = a * (gn > 0.05)
        # Light direction in the same triangle frame, read off the embedded triangle.
        b = torch.zeros_like(a)
        lx = ly = torch.zeros_like(a)
        if self.nca is not None and (self.P[:, PI["nca_tropic"]] != 0).any():
            p0, p1, p2 = self.pos[faces[:, 0]], self.pos[faces[:, 1]], self.pos[faces[:, 2]]
            e1 = torch.nn.functional.normalize(p1 - p0, dim=1)
            nrm = torch.nn.functional.normalize(torch.cross(p1 - p0, p2 - p0, dim=1), dim=1)
            e2 = torch.cross(nrm, e1, dim=1)
            tx, ty = e1[:, 2], e2[:, 2]
            tl = torch.sqrt(tx**2 + ty**2)
            lx, ly = tx / tl.clamp(min=1e-6), ty / tl.clamp(min=1e-6)
            s = 0.5 * (self.vf["nca"][:, self.nca.CH_TROPIC][faces].mean(1) + 1)
            b = self.P[self.ind[faces[:, 0]], PI["nca_tropic"]] * s * tl.clamp(max=1)
            tot = a.abs() + b
            k = torch.where(tot > 0.9, 0.9 / tot.clamp(min=1e-6), 1.0)
            a, b = a * k, b * k
        gf = dt_g * g[faces].mean(1).clamp(-1, 2)
        gf = torch.where((self.vf["pinned"][faces] > 0.5).any(1), 0.0, gf)
        # F = I + gf·(I + a·(2uuᵀ - I) + b·(2llᵀ - I))
        fxx = 1 + gf * (1 + a * (2 * ux * ux - 1) + b * (2 * lx * lx - 1))
        fyy = 1 + gf * (1 + a * (2 * uy * uy - 1) + b * (2 * ly * ly - 1))
        fxy = gf * 2 * (a * ux * uy + b * lx * ly)
        def mapped(ex, ey):
            return torch.sqrt((fxx * ex + fxy * ey) ** 2 + (fxy * ex + fyy * ey) ** 2)
        z = torch.zeros_like(l01)
        new = torch.stack([mapped(l01, z), mapped(x2 - l01, y2), mapped(-x2, -y2)], 1)
        acc = torch.zeros_like(self.rest).index_add_(0, he.reshape(-1), new.reshape(-1))
        cnt = torch.zeros_like(self.rest).index_add_(0, he.reshape(-1), torch.ones_like(new.reshape(-1)))
        self.rest = (acc / cnt.clamp(min=1)).clamp(min=cfg.min_rest)

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
        vcurl_t = torch.zeros(self.pos.shape[0], device=self.dev)
        if self.nca is not None:
            vcurl_t = self.P[self.ind, PI["nca_curl"]] * self.vf["nca"][:, self.nca.CH_CURL]
        vcurl = _wp(vcurl_t, float)
        N = self.pos.shape[0]
        for _ in range(n):
            wp.launch(k_springs, dim=edges.shape[0], inputs=[x, edges, rest, ind, P, PI["k_stretch"], f], device=self.wdev)
            if hinges.shape[0]:
                wp.launch(k_bending, dim=hinges.shape[0], inputs=[x, hinges, ind, P, PI["k_bend"], PI["curl"], vcurl, f], device=self.wdev)
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
        """One growth tick: grow rest lengths, relax, remesh."""
        cfg = self.cfg
        if cfg.morph_steps:
            self.morphogens(cfg.morph_steps)
        if self.nca is not None:
            self.nca.step(self, cfg.nca_steps)
        self.grow(dt_g)
        self.physics(cfg.substeps)
        self.vf["age"] += dt_g
        self.t += dt_g
        self._quarantine()
        self.ticks = getattr(self, "ticks", 0) + 1
        if self.ticks % cfg.remesh_every:
            return 0, 0
        return self.remesh()

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
        return pos, faces, extra

    def counts(self):
        return torch.bincount(self.ind, minlength=self.n_ind).tolist()
