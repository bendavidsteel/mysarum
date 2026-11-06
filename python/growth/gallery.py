"""High-quality renders of saved meshes (QD archive members or imitation bests).

    uv run python gallery.py outputs/qd_morph/meshes/12.npz outputs/imitate/x/best_0.npz --out outputs/gallery
    uv run python gallery.py --qd outputs/qd_morph --top 24 --diverse 24 --out outputs/gallery_qd
"""

import argparse
import json
import os

import numpy as np
import polars as pl
import torch
from PIL import Image, ImageDraw

from dgrow import render


def load(path):
    m = np.load(path)
    pos = torch.as_tensor(m["pos"].astype(np.float32), device="cuda")
    faces = torch.as_tensor(m["faces"].astype(np.int64), device="cuda")
    col = torch.as_tensor(m["color"].astype(np.float32) / 255, device="cuda") if "color" in m else None
    return pos, faces, col


def qd_picks(run, top, diverse):
    """Archive elites: the `top` fittest, then `diverse` farthest-apart in descriptor space."""
    from qd import farthest
    df = pl.read_parquet(f"{run}/evals.parquet").filter(~pl.col("dead"))
    elite = json.load(open(f"{run}/archive.json"))
    ids = [v[0] for v in elite.values()]
    el = df.filter(pl.col("id").is_in(ids)).sort("fitness", descending=True)
    picks = el["id"][:top].to_list()
    rest = el.filter(~pl.col("id").is_in(picks))
    if diverse and len(rest):
        emb = np.load(f"{run}/embeddings.npz")
        row = {int(i): k for k, i in enumerate(emb["ids"])}
        z = emb["emb" if "emb" in emb else "dino"][[row[i] for i in rest["id"]]]
        picks += [rest["id"][k] for k in farthest(z, diverse, np.random.default_rng(0))]
    labels = dict(zip(el["id"], [f"#{i} {l} {f:.2f}" for i, l, f in zip(el["id"], el["label"], el["fitness"])]))
    return [(f"{run}/meshes/{i}.npz", labels[i]) for i in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="*")
    ap.add_argument("--qd", help="QD run dir: render archive elites")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--diverse", type=int, default=0)
    ap.add_argument("--out", default="outputs/gallery")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--turntable", type=int, default=0, help="turntables for the first N")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = [(p, os.path.relpath(p)) for p in a.meshes]
    if a.qd:
        items += qd_picks(a.qd, a.top, a.diverse)
    tiles = []
    for k, (path, label) in enumerate(items):
        pos, faces, col = load(path)
        tag = f"{k:03d}"
        im = render.views(pos, faces, size=a.size, spp=a.spp, ao_n=8, color=col)
        im.save(f"{a.out}/{tag}_views.png")
        hero = render.render(pos, faces, W=a.size, H=a.size, azim=35, elev=25, spp=a.spp, ao_n=8, color=col)
        t = Image.fromarray(hero)
        ImageDraw.Draw(t).text((8, 6), label, fill=(30, 30, 30))
        tiles.append(np.asarray(t))
        if k < a.turntable:
            render.turntable(pos, faces, f"{a.out}/{tag}.mp4", size=a.size, color=col)
        print(tag, label, flush=True)
    blank = np.full_like(tiles[0], 235)
    tiles += [blank] * (-len(tiles) % a.cols)
    rows = [np.concatenate(tiles[i:i + a.cols], 1) for i in range(0, len(tiles), a.cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(f"{a.out}/gallery.jpg", quality=92)
    with open(f"{a.out}/items.json", "w") as fh:
        json.dump(items, fh, indent=1)


if __name__ == "__main__":
    main()
