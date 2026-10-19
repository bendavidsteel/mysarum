"""Grow a batch of genomes as one disjoint mesh and render each result."""

import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from . import genome as G, nca as NC, presets, render, sim as S

GENOMES = {"param": G, "nca": NC}


def structure(pos, faces, levels=(0.35, 0.5, 0.65, 0.8, 0.9), min_faces=6):
    """Structural descriptors: branch count (most separate curves cut by a
    horizontal slice) and log height/width aspect."""
    pos, faces = np.asarray(pos, np.float32), np.asarray(faces, np.int64)
    z = pos[:, 2]
    lo, hi = z.min(), z.max()
    ext = pos.max(0) - pos.min(0)
    aspect = float(np.log((hi - lo + 1e-3) / np.sqrt(ext[0] * ext[1] + 1e-3)))
    fz = z[faces]
    best = 1
    for f in levels:
        h = lo + f * (hi - lo)
        cut = np.nonzero((fz.min(1) < h) & (fz.max(1) >= h))[0]
        if len(cut) < min_faces:
            continue
        # Faces in one slice curve share the edges the plane crosses.
        fc = faces[cut]
        e = np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]])
        fi = np.tile(np.arange(len(cut)), 3)
        ze = z[e]
        cross = (ze.min(1) < h) & (ze.max(1) >= h)
        e, fi = np.sort(e[cross], 1), fi[cross]
        _, eid = np.unique(e[:, 0] * len(pos) + e[:, 1], return_inverse=True)
        n = len(cut) + eid.max() + 1
        A = sp.coo_matrix((np.ones(len(fi)), (fi, len(cut) + eid)), shape=(n, n))
        _, lab = connected_components(A, directed=False)
        sizes = np.bincount(lab[:len(cut)])
        best = max(best, int((sizes >= min_faces).sum()))
    return dict(branches=best, aspect=aspect)


def grow_batch(genomes, max_verts=40000, max_ticks=1000, dt=0.01, relax=200,
               substeps=20, morph_steps=4, seed=0, size=384, kind="param"):
    """kind: 'param' (genome.py, hand-built growth fields) or 'nca' (nca.py, neural CA)."""
    dec = [GENOMES[kind].decode(g) for g in genomes]
    seeds, params = [d[0] for d in dec], [d[1] for d in dec]
    meshes = [presets.seed_mesh(k, GENOMES[kind].SEED_RADIUS[k]) for k in seeds]
    cfg = S.SimConfig(max_verts=max_verts, substeps=substeps,
                      morph_steps=morph_steps if kind == "param" else 0, seed=seed)
    sim = S.Sim.from_meshes(meshes, list(params), cfg, spacing=1e4)
    if any(p.g_morph for p in params):
        sim.seed_morphogens(preroll=1000)
    if kind == "nca":
        ctrl = [d[2] for d in dec]
        sim.attach_nca(NC.NCA(np.stack([c["weights"] for c in ctrl]), [c["dt"] for c in ctrl],
                              [c["gain"] for c in ctrl], device=sim.dev))
    n = len(genomes)
    stop_tick = np.full(n, max_ticks)
    t0 = time.time()
    for it in range(max_ticks):
        sim.step(dt)
        if it % 10 == 0:
            full = torch.tensor(sim.counts(), device=sim.dev) >= int(0.97 * max_verts)
            stopped = full | sim.dead
            newly = stopped.cpu().numpy() & (stop_tick == max_ticks)
            stop_tick[newly] = it
            sim.active = torch.where(stopped, 0.0, sim.active)
            if bool(stopped.all()):
                break
    # Let the last growth settle before measuring.
    sim.active.zero_()
    for _ in range(relax // substeps):
        sim.step(dt)
    secs = time.time() - t0

    out = []
    dead = sim.dead.cpu().numpy()
    for i in range(n):
        pos, faces, ex = sim.individual(i)
        res = dict(seed=seeds[i], dead=bool(dead[i]), n_verts=int(pos.shape[0]),
                   stop_tick=int(stop_tick[i]), secs=secs)
        if not dead[i] and torch.isfinite(pos).all():
            ext = (pos.max(0).values - pos.min(0).values).cpu().numpy()
            res["extent"] = ext.tolist()
            res.update(structure(pos.cpu().numpy(), faces.cpu().numpy()))
            col = ex.get("color")
            res["iso"] = render.render(pos, faces, W=size, H=size, azim=35, elev=30, spp=8, ao_n=3, color=col)
            res["top"] = render.render(pos, faces, W=size, H=size, azim=0, elev=89, spp=8, ao_n=3, color=col)
            if col is not None:
                res["color"] = (col * 255).byte().cpu().numpy()
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
