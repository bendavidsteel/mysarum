"""Grow a batch of genomes as one disjoint mesh and render each result."""

import time

import numpy as np
import scipy.sparse as sp
import torch
import warp as wp
from scipy.sparse.csgraph import connected_components, dijkstra

from . import genome as G, nca as NC, presets, render, sim as S

GENOMES = {"param": G, "nca": NC}


def slice_count(f, faces, h, min_faces=6):
    """Separate curves cut from the mesh by the level set f = h."""
    ff = f[faces]
    cut = np.nonzero((ff.min(1) < h) & (ff.max(1) >= h))[0]
    if len(cut) < min_faces:
        return 0
    # Faces in one curve share the edges the level set crosses.
    fc = faces[cut]
    e = np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]])
    fi = np.tile(np.arange(len(cut)), 3)
    fe = f[e]
    cross = (fe.min(1) < h) & (fe.max(1) >= h)
    e, fi = np.sort(e[cross], 1), fi[cross]
    _, eid = np.unique(e[:, 0] * len(f) + e[:, 1], return_inverse=True)
    n = len(cut) + eid.max() + 1
    A = sp.coo_matrix((np.ones(len(fi)), (fi, len(cut) + eid)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    return int((np.bincount(lab[:len(cut)]) >= min_faces).sum())


def base_distance(pos, faces):
    """Geodesic (edge-path) distance from the lowest vertices, and the edge graph."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    w = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1) + 1e-6
    A = sp.csr_matrix((w, (e[:, 0], e[:, 1])), shape=(len(pos),) * 2)
    A = A.maximum(A.T).tocsr()
    base = np.nonzero(pos[:, 2] < pos[:, 2].min() + 0.5)[0]
    d = dijkstra(A, indices=base, min_only=True)
    return np.where(np.isfinite(d), d, 0.0), A


def fingers(pos, faces, n_levels=16, min_len=3.0):
    """Most protrusions cut off by one geodesic level set: components of
    {d > h} that reach `min_len` past h and are longer than they are wide
    (cap length > boundary length / pi). Fold crests also make caps, but they
    are long thin loops, so they fail the width test."""
    d, A = base_distance(pos, faces)
    el = A.data.mean()
    N = (A > 0).astype(np.float32)
    U = sp.triu(A).tocoo()
    best = 1
    for h in np.linspace(0.15, 0.9, n_levels) * d.max():
        up = d > h
        keep = up[U.row] & up[U.col]
        S = sp.coo_matrix((np.ones(keep.sum()), (U.row[keep], U.col[keep])), shape=A.shape)
        _, lab = connected_components(S, directed=False)
        lab = np.where(up, lab, -1)
        # Boundary of a cap: its vertices with a neighbour at or below h.
        rim = up & (N @ (~up).astype(np.float32) > 0)
        k = lab.max() + 1
        cap = np.zeros(k)
        np.maximum.at(cap, lab[up], d[up] - h)
        perim = np.bincount(lab[rim], minlength=k) * el
        best = max(best, int(((cap > min_len) & (cap > perim / np.pi)).sum()))
    return best


def structure(pos, faces, levels=(0.35, 0.5, 0.65, 0.8, 0.9), min_faces=6):
    """Structural descriptors: `branches`, the most separate curves cut by a
    horizontal slice; `fingers`, protrusions off geodesic level sets (see
    `fingers`), which ignores folds; log height/width `aspect`."""
    pos, faces = np.asarray(pos, np.float32), np.asarray(faces, np.int64)
    z = pos[:, 2]
    lo, hi = z.min(), z.max()
    ext = pos.max(0) - pos.min(0)
    aspect = float(np.log((hi - lo + 1e-3) / np.sqrt(ext[0] * ext[1] + 1e-3)))
    branches = max([1] + [slice_count(z, faces, lo + f * (hi - lo), min_faces) for f in levels])
    return dict(branches=branches, aspect=aspect, fingers=fingers(pos, faces))


@wp.kernel
def k_thickness(mesh: wp.uint64, x: wp.array(dtype=wp.vec3), vn: wp.array(dtype=wp.vec3),
                max_d: float, n_dir: int, out: wp.array(dtype=float), tube: wp.array(dtype=float)):
    i = wp.tid()
    d = -vn[i]
    q = wp.mesh_query_ray(mesh, x[i] + d * 1.0e-3, d, max_d)
    out[i] = -1.0
    tube[i] = 0.0
    # Only the inside of the opposite wall counts: then the ray crossed solid
    # tissue. Open sheets and gaps between folds face the other way or miss.
    if q.result and wp.dot(q.normal, d) > 0.0:
        t = q.t
        out[i] = t
        # From the chord's midpoint, rays across it: inside a tube most hit the
        # wall within 1.5 t; between the parallel walls of a fold almost none do.
        m = x[i] + d * (0.5 * t)
        a = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(d[0]) > 0.8:
            a = wp.vec3(0.0, 1.0, 0.0)
        u = wp.normalize(wp.cross(d, a))
        v = wp.cross(d, u)
        hits = float(0.0)
        for k in range(n_dir):
            th = 2.0 * wp.pi * float(k) / float(n_dir)
            r = wp.mesh_query_ray(mesh, m, wp.cos(th) * u + wp.sin(th) * v, 1.5 * t)
            if r.result:
                hits += 1.0
        tube[i] = hits / float(n_dir)


def thickness(pos, faces, max_d=50.0, n_dir=8):
    """Per-vertex local diameter in edge lengths (distance along the inward normal
    to the opposite wall; -1 where the ray leaves through open sheet or misses)
    and tubeness in [0, 1] (≈0.8 for a tube, ≈0 for a fold or fin)."""
    pos = pos.float().contiguous()
    f = faces.int().contiguous()
    wdev = wp.device_from_torch(pos.device)
    m = wp.Mesh(points=wp.from_torch(pos, dtype=wp.vec3), indices=wp.from_torch(f.reshape(-1), dtype=wp.int32))
    vn = render._vertex_normals(pos, f.long()).contiguous()
    out = torch.empty(len(pos), device=pos.device)
    tube = torch.empty(len(pos), device=pos.device)
    wp.launch(k_thickness, dim=len(pos), inputs=[m.id, wp.from_torch(pos, dtype=wp.vec3), wp.from_torch(vn, dtype=wp.vec3),
                                                  max_d, n_dir, wp.from_torch(out), wp.from_torch(tube)], device=wdev)
    return out, tube


def thin_fraction(pos, faces, max_d=1.5):
    """Share of vertices on tubes narrower than `max_d` edges: limbs the mesh can't
    resolve. Folds, fins and open sheets don't count (see `thickness`)."""
    t, tube = thickness(pos, faces)
    return float(((t >= 0) & (t < max_d) & (tube >= 0.5)).float().mean())


def thin_factor(thin, free=0.15, scale=0.15):
    """Fitness multiplier for wire limbs. Search otherwise maxes surface tension
    until every limb is one edge wide: a wire silhouette reads as twigs to the judge.
    Brain-coral ridge crests come to ~0.14, wire trees to 0.4-0.7."""
    return float(np.exp(-max(0.0, thin - free) / scale))


def branching(seed, p):
    """Forms that grow limbs: the stalk seed, or accretion fed by the supply field.
    Limbs are tubes of surface that take many vertices to resolve."""
    return seed == "stalk" or (p.accrete > 0 and p.acc_field > 0)


def grow_batch(genomes, max_verts=40000, max_ticks=1000, dt=0.01, relax=200,
               substeps=20, morph_steps=4, seed=0, size=384, kind="param", on_tick=None,
               retire=True, on_end=None, branch_verts=None):
    """kind: 'param' (genome.py, hand-built growth fields) or 'nca' (nca.py, neural CA).
    on_tick(sim, tick) is called after every growth tick (film.py records with it).
    retire=False keeps settled individuals in the batch until everyone has settled;
    then on_end(sim) runs before they are extracted (film.py animates them).
    branch_verts: vertex budget for `branching` forms (default max_verts)."""
    dec = [GENOMES[kind].decode(g) for g in genomes]
    seeds, params = [d[0] for d in dec], [d[1] for d in dec]
    meshes = [presets.seed_mesh(k, GENOMES[kind].SEED_RADIUS[k]) for k in seeds]
    budget = np.array([branch_verts if branch_verts and branching(k, p) else max_verts
                       for k, p in zip(seeds, params)])
    cfg = S.SimConfig(max_verts=int(budget.max()), substeps=substeps,
                      morph_steps=morph_steps if kind == "param" else 0, seed=seed)
    sim = S.Sim.from_meshes(meshes, list(params), cfg, spacing=1e4)
    sim.budget = torch.as_tensor(budget, device=sim.dev)
    if any(p.g_morph for p in params):
        sim.seed_morphogens(preroll=1000)
    if kind == "nca":
        ctrl = [d[2] for d in dec]
        sim.attach_nca(NC.NCA(np.stack([c["weights"] for c in ctrl]), [c["dt"] for c in ctrl],
                              [c["gain"] for c in ctrl], [c["fire"] for c in ctrl],
                              [c["sh_local"] for c in ctrl], device=sim.dev))
    n = len(genomes)
    stop_tick = np.full(n, max_ticks)
    # A stopped individual settles for `relax` substeps, then leaves the batch
    # (Sim.retire) so the stragglers don't keep paying for it.
    settle = max(1, relax // substeps)
    retire_at, final, settled = {}, {}, set()
    t0 = time.time()
    for it in range(max_ticks):
        sim.step(dt)
        if on_tick is not None:
            on_tick(sim, it)
        if it % 10 == 0:
            # Growing meshes fragment torch's cache until Warp can't allocate; hand it back.
            torch.cuda.empty_cache()
            full = torch.tensor(sim.counts(), device=sim.dev) >= (0.97 * sim.budget).long()
            stopped = (full | sim.dead).cpu().numpy()
            for i in np.nonzero(stopped & (stop_tick == max_ticks))[0]:
                stop_tick[i] = it
                retire_at[int(i)] = it + settle
            sim.active = torch.where(torch.as_tensor(stopped, device=sim.dev), 0.0, sim.active)
        due = [i for i, t in retire_at.items() if t <= it and i not in final and i not in settled]
        if due and retire:
            final.update(sim.retire(due))
        settled.update(due)
        if len(settled) == n:
            break
    if len(settled) < n:
        sim.active.zero_()
        for _ in range(settle):
            sim.step(dt)
    if on_end is not None:
        on_end(sim)
    rest = [i for i in range(n) if i not in final]
    if rest:
        final.update(sim.retire(rest))
    secs = time.time() - t0

    out = []
    dead = sim.dead.cpu().numpy()
    for i in range(n):
        pos, faces, ex = final[i]
        res = dict(seed=seeds[i], dead=bool(dead[i]), n_verts=int(pos.shape[0]), floor=float(meshes[i]["pos"][:, 2].min()),
                   stop_tick=int(stop_tick[i]), secs=secs)
        if not dead[i] and torch.isfinite(pos).all():
            ext = (pos.max(0).values - pos.min(0).values).cpu().numpy()
            res["extent"] = ext.tolist()
            res.update(structure(pos.cpu().numpy(), faces.cpu().numpy()))
            res["thin"] = thin_fraction(pos, faces)
            col, mat = ex.get("color"), ex.get("material")
            # Along x and y near the horizon, plus angled from above, all on a floor at
            # the seed's base that hides anything grown below it (an ungrounded seed
            # can't pass a mirrored double as a tree). Straight down reads as nothing
            # to the judge, so `top` is only for descriptors and contact sheets.
            fz = float(meshes[i]["pos"][:, 2].min())
            res["views"] = [render.render(pos, faces, W=size, H=size, azim=a, elev=e, spp=8, ao_n=3, color=col,
                                          floor=fz, material=mat) for a, e in ((0, 5), (90, 5), (35, 30))]
            res["iso"] = res["views"][2]
            res["top"] = render.render(pos, faces, W=size, H=size, azim=0, elev=89, spp=8, ao_n=3, color=col,
                                      material=mat)
            if col is not None:
                res["color"] = (col * 255).byte().cpu().numpy()
                res["material"] = (mat * 255).byte().cpu().numpy()
            res["pos"] = pos.half().cpu().numpy()
            res["faces"] = faces.int().cpu().numpy()
        else:
            res["dead"] = True
        out.append(res)
    return out


def grow_reps(genomes, reps=1, **kw):
    """Grow each genome `reps` times with different random draws -> one result list per genome."""
    res = grow_batch(np.repeat(np.asarray(genomes), reps, axis=0), **kw)
    return [res[i * reps:(i + 1) * reps] for i in range(len(genomes))]
