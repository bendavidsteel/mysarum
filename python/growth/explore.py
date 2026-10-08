"""Grow a batch of presets side by side and render a labelled montage.

    uv run python explore.py lettuce kale brain --out outputs/explore
    uv run python explore.py lettuce:k_bend=6 lettuce:k_bend=12   # overrides
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from dgrow import presets, render, sim as S


def parse(spec):
    name, *ov = spec.split(":")
    kw = {}
    for o in ov:
        for kv in o.split(","):
            k, v = kv.split("=")
            kw[k] = float(v)
    return name, kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--out", default="outputs/explore")
    ap.add_argument("--max_verts", type=int, default=60000)
    ap.add_argument("--substeps", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=0, help="override preset tick count")
    ap.add_argument("--morph_steps", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--turntable", action="store_true")
    ap.add_argument("--ply", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meshes, params, ticks, dts = [], [], [], []
    for s in a.specs:
        name, kw = parse(s)
        m, p, t, dt = presets.preset(name, **kw)
        meshes.append(m), params.append(p), ticks.append(a.ticks or t), dts.append(dt)
    morph = a.morph_steps or (4 if any(p.g_morph for p in params) else 0)
    cfg = S.SimConfig(max_verts=a.max_verts, substeps=a.substeps, morph_steps=morph)
    sim = S.Sim.from_meshes(meshes, params, cfg, spacing=1e4)
    if morph:
        sim.seed_morphogens()
    t0 = time.time()
    # All presets share one clock; per-preset tick counts become growth freezes.
    for it in range(max(ticks)):
        done = torch.tensor([it >= t for t in ticks], device=sim.dev)
        sim.P[:, S.PI["g_base"]] = torch.where(done, 0.0, torch.tensor([p.g_base for p in params], device=sim.dev))
        for k in ["g_edge", "g_curv", "g_morph", "g_height"]:
            sim.P[:, S.PI[k]] = torch.where(done, 0.0, torch.tensor([getattr(p, k) for p in params], device=sim.dev))
        sim.step(dts[0])
        if it % 100 == 0:
            print(f"tick {it} verts {sim.counts()} {time.time() - t0:.0f}s", flush=True)
    print(f"done {time.time() - t0:.0f}s verts {sim.counts()}")

    tiles = []
    for i, s in enumerate(a.specs):
        pos, faces, ex = sim.individual(i)
        tag = s.replace(":", "_").replace("=", "").replace(",", "_")
        im = render.views(pos, faces, size=a.size, label=f"{s}  nv={pos.shape[0]}")
        im.save(f"{a.out}/{tag}.png")
        tiles.append(np.asarray(im))
        np.savez_compressed(f"{a.out}/{tag}.npz", pos=pos.cpu().numpy(), faces=faces.cpu().numpy())
        if a.turntable:
            render.turntable(pos, faces, f"{a.out}/{tag}.mp4")
    Image.fromarray(np.concatenate(tiles, 0)).save(f"{a.out}/montage.png")
    with open(f"{a.out}/specs.json", "w") as fh:
        json.dump(a.specs, fh)


if __name__ == "__main__":
    main()
