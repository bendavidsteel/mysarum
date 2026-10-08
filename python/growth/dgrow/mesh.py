"""Triangle-mesh topology on the GPU: builders, edge tables, parallel split/flip.

The mesh is just `pos (N,3)` + `faces (F,3)`; everything else (edges, hinges,
adjacency) is rebuilt from `faces` with sorts after each remesh. Rest lengths
live on undirected edges, keyed by `min<<32 | max` so they survive rebuilds.
Several independent meshes ("individuals") can share one disjoint mesh; `ind`
maps each vertex to its individual.
"""

import math

import numpy as np
import torch
from scipy.spatial import Delaunay


def edge_key(a, b):
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    return (lo << 32) | hi


class Topology:
    """Derived connectivity for a face list. Rebuilt after every remesh."""

    def __init__(self, faces, n_verts):
        dev = faces.device
        F = faces.shape[0]
        src = faces.reshape(-1)
        dst = faces[:, [1, 2, 0]].reshape(-1)
        opp = faces[:, [2, 0, 1]].reshape(-1)
        fid = torch.arange(F, device=dev).repeat_interleave(3)
        keys = edge_key(src, dst)
        ukeys, he_edge = torch.unique(keys, sorted=True, return_inverse=True)
        M = ukeys.shape[0]

        # Each undirected edge has 1 (boundary) or 2 (interior) half-edges.
        order = torch.argsort(he_edge, stable=True)
        e_sorted = he_edge[order]
        first = torch.ones_like(e_sorted, dtype=torch.bool)
        first[1:] = e_sorted[1:] != e_sorted[:-1]
        slot = (~first).long()  # 0 for first half-edge of an edge, 1 for second
        he_slot = torch.empty_like(slot)
        he_slot[order] = slot

        edge_he = torch.full((M, 2), -1, dtype=torch.long, device=dev)
        edge_he[he_edge, he_slot] = torch.arange(3 * F, device=dev)

        self.keys = ukeys
        self.edges = torch.stack([ukeys >> 32, ukeys & 0xFFFFFFFF], 1)
        self.edge_he = edge_he
        self.he_src, self.he_dst, self.he_opp, self.he_face = src, dst, opp, fid
        self.he_edge = he_edge
        self.interior = edge_he[:, 1] >= 0
        self.n_verts = n_verts

        # Hinges (a, b, c, d): shared edge a->b in face (a,b,c), twin b->a in (b,a,d).
        ie = torch.nonzero(self.interior).squeeze(1)
        h1, h2 = edge_he[ie, 0], edge_he[ie, 1]
        self.hinge_edge = ie
        self.hinges = torch.stack([src[h1], dst[h1], opp[h1], opp[h2]], 1)

        # Boundary vertices: endpoints of edges with a single face.
        bnd = torch.zeros(n_verts, dtype=torch.bool, device=dev)
        be = self.edges[~self.interior]
        bnd[be.reshape(-1)] = True
        self.boundary = bnd

        # CSR vertex adjacency (both directions of every edge).
        a = torch.cat([self.edges[:, 0], self.edges[:, 1]])
        b = torch.cat([self.edges[:, 1], self.edges[:, 0]])
        o = torch.argsort(a, stable=True)
        self.adj = b[o].int()
        counts = torch.bincount(a, minlength=n_verts)
        self.adj_off = torch.zeros(n_verts + 1, dtype=torch.int32, device=dev)
        self.adj_off[1:] = torch.cumsum(counts, 0).int()
        self.valence = counts


def lookup_rest(old_keys, old_rest, new_keys, fallback):
    """Rest length for each of new_keys from a sorted (old_keys, old_rest) table."""
    idx = torch.searchsorted(old_keys, new_keys).clamp(max=old_keys.shape[0] - 1)
    hit = old_keys[idx] == new_keys
    return torch.where(hit, old_rest[idx], fallback)


def _independent_edges(topo, cand, gen):
    """Subset of candidate edges such that no face touches two of them."""
    M = cand.shape[0]
    prio = torch.rand(M, device=cand.device, generator=gen)
    prio = torch.where(cand, prio, torch.full_like(prio, -1.0))
    F = topo.he_face.max().item() + 1
    fmax = torch.full((F,), -1.0, device=cand.device)
    fmax.scatter_reduce_(0, topo.he_face, prio[topo.he_edge], reduce="amax")
    ok_he = prio[topo.he_edge] >= fmax[topo.he_face]
    ok = torch.ones(M, dtype=torch.int32, device=cand.device)
    ok.scatter_reduce_(0, topo.he_edge, ok_he.int(), reduce="amin")
    return cand & ok.bool()


def _median_len(r_ca, r_cb, r_ab):
    # Length from c to the midpoint of ab, from the triangle's side lengths.
    return torch.sqrt(torch.clamp(0.5 * r_ca**2 + 0.5 * r_cb**2 - 0.25 * r_ab**2, min=1e-8))


def split_edges(pos, faces, rest_keys, rest, topo, sel, vfields):
    """Split the selected edges at their midpoints (sel must be independent).

    Returns new (pos, faces, rest_keys, rest, vfields). `vfields` is a dict of
    per-vertex tensors; new vertices get the endpoint average.
    """
    dev = pos.device
    ei = torch.nonzero(sel).squeeze(1)
    K = ei.shape[0]
    if K == 0:
        return pos, faces, rest_keys, rest, vfields
    N = pos.shape[0]
    m = torch.arange(N, N + K, device=dev)
    a, b = topo.edges[ei, 0], topo.edges[ei, 1]
    r_ab = lookup_rest(rest_keys, rest, topo.keys[ei], torch.zeros(K, device=dev))

    pos = torch.cat([pos, 0.5 * (pos[a] + pos[b])])
    vfields = {k: torch.cat([v, 0.5 * (v[a] + v[b])]) for k, v in vfields.items()}

    new_faces = [faces]
    new_keys = [edge_key(a, m), edge_key(m, b)]
    new_rest = [0.5 * r_ab, 0.5 * r_ab]
    replace_face = torch.full((faces.shape[0],), -1, dtype=torch.long, device=dev)
    keep = torch.ones(faces.shape[0], dtype=torch.bool, device=dev)
    for s in range(2):
        he = topo.edge_he[ei, s]
        has = he >= 0
        he, mm, rab = he[has], m[has], r_ab[has]
        u, v, c = topo.he_src[he], topo.he_dst[he], topo.he_opp[he]
        f = topo.he_face[he]
        keep[f] = False
        # Face (u, v, c) -> (u, m, c) + (m, v, c), preserving orientation.
        new_faces.append(torch.stack([u, mm, c], 1))
        new_faces.append(torch.stack([mm, v, c], 1))
        r_uc = lookup_rest(rest_keys, rest, edge_key(u, c), torch.ones_like(rab))
        r_vc = lookup_rest(rest_keys, rest, edge_key(v, c), torch.ones_like(rab))
        new_keys.append(edge_key(mm, c))
        new_rest.append(_median_len(r_uc, r_vc, rab))
    new_faces[0] = faces[keep]
    faces = torch.cat(new_faces)

    # Drop the split edges' keys, add the new ones, keep the table sorted.
    gone = torch.zeros_like(rest_keys, dtype=torch.bool)
    idx = torch.searchsorted(rest_keys, topo.keys[ei]).clamp(max=rest_keys.shape[0] - 1)
    gone[idx[rest_keys[idx] == topo.keys[ei]]] = True
    keys = torch.cat([rest_keys[~gone]] + new_keys)
    vals = torch.cat([rest[~gone]] + new_rest)
    o = torch.argsort(keys)
    return pos, faces, keys[o], vals[o], vfields


def _flip_len(r_ab, r_bc, r_ca, r_ad, r_db):
    """Intrinsic length of c-d after flipping ab, unfolding both triangles flat."""
    # Place a=(0,0), b=(r_ab,0); c above, d below.
    cx = (r_ab**2 + r_ca**2 - r_bc**2) / (2 * r_ab)
    cy = torch.sqrt(torch.clamp(r_ca**2 - cx**2, min=1e-10))
    dx = (r_ab**2 + r_ad**2 - r_db**2) / (2 * r_ab)
    dy = -torch.sqrt(torch.clamp(r_ad**2 - dx**2, min=1e-10))
    return torch.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)


def _opp_angle(r_opp, r1, r2):
    cos = (r1**2 + r2**2 - r_opp**2) / (2 * r1 * r2 + 1e-12)
    return torch.acos(cos.clamp(-1, 1))


def flip_edges(faces, rest_keys, rest, topo, gen, min_valence=4):
    """One parallel round of intrinsic Delaunay flips. Returns (faces, keys, rest, n)."""
    dev = faces.device
    ie = topo.hinge_edge
    a, b, c, d = topo.hinges.unbind(1)
    L = lambda u, v: lookup_rest(rest_keys, rest, edge_key(u, v), torch.ones(u.shape[0], device=dev))
    r_ab, r_bc, r_ca, r_ad, r_db = L(a, b), L(b, c), L(c, a), L(a, d), L(d, b)
    ang = _opp_angle(r_ab, r_bc, r_ca) + _opp_angle(r_ab, r_ad, r_db)
    want = ang > math.pi + 1e-3
    want &= (topo.valence[a] > min_valence) & (topo.valence[b] > min_valence)
    # A c-d edge that already exists would make the mesh non-manifold.
    kcd = edge_key(c, d)
    idx = torch.searchsorted(topo.keys, kcd).clamp(max=topo.keys.shape[0] - 1)
    want &= topo.keys[idx] != kcd
    cand = torch.zeros(topo.keys.shape[0], dtype=torch.bool, device=dev)
    cand[ie[want]] = True
    sel = _independent_edges(topo, cand, gen)
    hsel = sel[ie]
    n = int(hsel.sum())
    if n == 0:
        return faces, rest_keys, rest, 0

    a, b, c, d = a[hsel], b[hsel], c[hsel], d[hsel]
    r_cd = _flip_len(r_ab[hsel], r_bc[hsel], r_ca[hsel], r_ad[hsel], r_db[hsel])
    h1, h2 = topo.edge_he[ie[hsel], 0], topo.edge_he[ie[hsel], 1]
    keep = torch.ones(faces.shape[0], dtype=torch.bool, device=dev)
    keep[topo.he_face[h1]] = False
    keep[topo.he_face[h2]] = False
    # (a,b,c) + (b,a,d) -> (c,a,d) + (d,b,c)
    faces = torch.cat([faces[keep], torch.stack([c, a, d], 1), torch.stack([d, b, c], 1)])

    gone = torch.zeros_like(rest_keys, dtype=torch.bool)
    kab = topo.keys[ie[hsel]]
    idx = torch.searchsorted(rest_keys, kab).clamp(max=rest_keys.shape[0] - 1)
    gone[idx[rest_keys[idx] == kab]] = True
    keys = torch.cat([rest_keys[~gone], edge_key(c, d)])
    vals = torch.cat([rest[~gone], r_cd])
    o = torch.argsort(keys)
    return faces, keys[o], vals[o], n


def select_splits(topo, rest_keys, rest, split_len, allow_v, gen):
    """Independent set of edges whose rest length exceeds split_len.

    allow_v: per-vertex bool; an edge may split only if both ends are allowed
    (used for per-individual vertex budgets and pinned regions).
    """
    r = lookup_rest(rest_keys, rest, topo.keys, torch.zeros(topo.keys.shape[0], device=rest.device))
    cand = (r > split_len) & allow_v[topo.edges[:, 0]] & allow_v[topo.edges[:, 1]]
    return _independent_edges(topo, cand, gen)


# ── Builders (NumPy, then moved to the device) ──────────────────────────────

def make_disc(radius, edge=1.0, jitter=0.15, seed=0):
    """Jittered hex-lattice disc in the xy plane, CCW faces (normal +z)."""
    pts = []
    rows = int(radius / (edge * math.sqrt(3) / 2)) + 1
    for j in range(-rows, rows + 1):
        y = j * edge * math.sqrt(3) / 2
        off = 0.5 * edge * (j % 2)
        for i in range(-rows - 1, rows + 2):
            x = i * edge + off
            if x * x + y * y <= radius * radius + 1e-6:
                pts.append((x, y))
    pts = np.array(pts)
    r = np.linalg.norm(pts, axis=1)
    inner = r < radius - 0.6 * edge
    pts[inner] += np.random.default_rng(seed).uniform(-jitter, jitter, (inner.sum(), 2)) * edge
    tri = Delaunay(pts).simplices
    # Drop slivers on the hull.
    p = pts[tri]
    e = np.stack([np.linalg.norm(p[:, i] - p[:, (i + 1) % 3], axis=1) for i in range(3)], 1)
    tri = tri[e.max(1) < 1.5 * edge]
    v = np.c_[pts, np.zeros(len(pts))]
    cr = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])[:, 2]
    tri[cr < 0] = tri[cr < 0][:, [0, 2, 1]]
    used = np.unique(tri)
    remap = -np.ones(len(v), int)
    remap[used] = np.arange(len(used))
    return v[used].astype(np.float32), remap[tri].astype(np.int64)


def make_icosphere(radius, edge=1.0):
    """Icosphere with outward faces, subdivided until edges are ~`edge`."""
    t = (1 + 5**0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
                  [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
                  [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
                  [8, 6, 7], [9, 8, 1]])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    while radius * np.linalg.norm(v[f[0, 0]] - v[f[0, 1]]) > 1.3 * edge:
        cache = {}
        vl = list(v)

        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                p = vl[i] + vl[j]
                vl.append(p / np.linalg.norm(p))
                cache[k] = len(vl) - 1
            return cache[k]

        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v, f = np.array(vl), np.array(nf)
    return (v * radius).astype(np.float32), f.astype(np.int64)


def make_hemisphere(radius, edge=1.0):
    """Open upper hemisphere (z >= 0) cut from an icosphere; rim on z=0."""
    v, f = make_icosphere(radius, edge)
    keep = (v[f][:, :, 2] >= -1e-4 * radius).all(1)
    f = f[keep]
    used = np.unique(f)
    remap = -np.ones(len(v), int)
    remap[used] = np.arange(len(used))
    v = v[used]
    v[:, 2] = np.maximum(v[:, 2], 0)
    return v.astype(np.float32), remap[f].astype(np.int64)
