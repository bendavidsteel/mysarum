"""Growth time-lapse mp4s: regrow genomes, snapshot every few ticks, then spin the
finished form. The camera zooms out as the form grows and orbits slowly. Each
form is filmed until it has settled, so the film ends on the form that the
search scored. With --live, the closing turn keeps the NCA running on the still
form (as MeshNCA textures do), so its colour, materials and bumps keep moving.

    uv run python film.py --qd outputs/qd_v2 --ids 87 305 158 --out outputs/films
    uv run python film.py --results outputs/imitate_v2/*/result.json --out outputs/films
    uv run python film.py --qd outputs/qd_v4 --ids 652 --size 1080 --spp 64 --deg 0.5 --spin 300 --elev 18

A regrowth uses fresh random draws, so it can differ in detail from the archived run.
"""

import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
import polars as pl
import torch

from dgrow import evaluate, render


def snapshot(sim, i):
    sel = sim.ind == i
    idx = torch.nonzero(sel).squeeze(1)
    remap = torch.full((sim.pos.shape[0],), -1, dtype=torch.long, device=sim.dev)
    remap[idx] = torch.arange(len(idx), device=sim.dev)
    faces = remap[sim.faces[sel[sim.faces[:, 0]]]]
    pos = (sim.pos[idx] - sim.centers[i]).half().cpu()
    if sim.nca is None:
        return pos, faces.int().cpu(), None, None
    S = sim.vf["nca"][idx]
    return pos, faces.int().cpu(), sim.nca.color(S).half().cpu(), sim.nca.material(S).half().cpu()


def frame_box(pos, floor):
    p = pos[pos[:, 2] >= floor]
    p = p if len(p) else pos
    lo, hi = p.min(0).values, p.max(0).values
    return 0.5 * (lo + hi), 0.5 * (hi - lo).norm().item()


def film(frames, floor, path, size, spp, deg_per_frame=0.8, spin=120, elev=22.0, fps=30, live=None):
    """frames: snapshots (pos, faces, color, material); live: snapshots for the closing
    turn (else the last frame held). Camera radius only ever grows, eased."""
    out, c_s, r_s, az = [], None, 0.0, 0.0
    with imageio.get_writer(path, fps=fps, quality=8, macro_block_size=8) as w:
        for k, (pos, faces, col, mat) in enumerate(frames + (live or [frames[-1]] * spin)):
            pos = pos.float().cuda()
            c, r = frame_box(pos, floor)
            r = max(r, 6.0)
            c_s = c if c_s is None else c_s + 0.15 * (c - c_s)
            r_s = max(r_s, r_s + 0.15 * (r - r_s)) if r_s else r
            az += deg_per_frame if k < len(frames) else 360.0 / spin
            w.append_data(render.render(pos, faces.long().cuda(), W=size, H=size, azim=35 + az, elev=elev,
                                        center=c_s, radius=r_s, spp=spp, ao_n=4, floor=floor,
                                        color=None if col is None else col.float().cuda(),
                                        material=None if mat is None else mat.float().cuda()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", help="QD run dir (with --ids)")
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    ap.add_argument("--results", nargs="*", default=[], help="imitation result.json files")
    ap.add_argument("--genome", default="nca")
    ap.add_argument("--max_verts", type=int, default=40000)
    ap.add_argument("--max_ticks", type=int, default=1000)
    ap.add_argument("--every", type=int, default=4, help="ticks per frame")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--deg", type=float, default=0.8, help="orbit degrees per growth frame")
    ap.add_argument("--spin", type=int, default=120, help="frames for the closing full turn")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--live", type=int, default=2, help="NCA steps per closing-turn frame (0: hold still)")
    ap.add_argument("--out", default="outputs/films")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G = evaluate.GENOMES[a.genome]
    names, genomes = [], []
    if a.qd:
        df = pl.read_parquet(f"{a.qd}/evals.parquet")
        for i in a.ids:
            r = df.filter(pl.col("id") == i).row(0, named=True)
            names.append(f"qd{i}_{r['label'].split(' ', 1)[-1].replace(' ', '_')}")
            genomes.append(r["genome"])
    for p in a.results:
        r = json.load(open(p))
        names.append(r["target"].split(" ", 1)[-1].replace(" ", "_"))
        genomes.append(r["genome"])
    genomes = [G.upgrade(g) if len(g) != G.D else g for g in map(np.asarray, genomes)]
    shots = [[] for _ in genomes]
    live = [[] for _ in genomes]
    stopped = {}
    settle = 200 // 20  # grow_batch's relax // substeps

    def record(sim, t):
        if t % a.every:
            return
        for i in range(len(genomes)):
            # Until it has settled (grow_batch keeps it in the batch when retire=False).
            if not bool(sim.active[i] > 0):
                stopped.setdefault(i, t)
            if t <= stopped.get(i, t) + settle and bool((sim.ind == i).any()):
                shots[i].append(snapshot(sim, i))

    def animate(sim):
        for _ in range(a.spin):
            sim.nca.step(sim, a.live, live=True)
            for i in range(len(genomes)):
                live[i].append(snapshot(sim, i))

    res = evaluate.grow_batch(np.array(genomes), max_verts=a.max_verts, max_ticks=a.max_ticks,
                              seed=a.seed, kind=a.genome, on_tick=record, retire=a.live == 0,
                              on_end=animate if a.live and a.genome == "nca" else None)
    for name, fr, lv, r in zip(names, shots, live, res):
        if r["dead"]:
            print(name, "died", flush=True)
            continue
        film(fr, r["floor"], f"{a.out}/{name}.mp4", a.size, a.spp, a.deg, a.spin, a.elev, live=lv or None)
        print(name, len(fr), "frames", flush=True)


if __name__ == "__main__":
    main()
