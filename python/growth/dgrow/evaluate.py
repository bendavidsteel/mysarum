"""Grow a batch of genomes as one disjoint mesh and render each result."""

import time

import numpy as np
import torch

from . import genome as G, presets, render, sim as S


def grow_batch(genomes, max_verts=40000, max_ticks=1000, dt=0.01, relax=200,
               substeps=20, morph_steps=4, seed=0, size=384):
    seeds, params = zip(*[G.decode(g) for g in genomes])
    meshes = [presets.seed_mesh(k, G.SEED_RADIUS[k]) for k in seeds]
    cfg = S.SimConfig(max_verts=max_verts, substeps=substeps, morph_steps=morph_steps, seed=seed)
    sim = S.Sim.from_meshes(meshes, list(params), cfg, spacing=1e4)
    if any(p.g_morph for p in params):
        sim.seed_morphogens(preroll=1000)
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
            res["iso"] = render.render(pos, faces, W=size, H=size, azim=35, elev=30, spp=8, ao_n=3)
            res["top"] = render.render(pos, faces, W=size, H=size, azim=0, elev=89, spp=8, ao_n=3)
            res["pos"] = pos.half().cpu().numpy()
            res["faces"] = faces.int().cpu().numpy()
        else:
            res["dead"] = True
        out.append(res)
    return out
