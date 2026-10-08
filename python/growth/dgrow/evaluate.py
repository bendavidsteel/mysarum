"""Grow a batch of genomes as one disjoint mesh and render each result."""

import time

import numpy as np
import torch

from . import genome as G, nca as NC, presets, render, sim as S

GENOMES = {"param": G, "nca": NC}


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
