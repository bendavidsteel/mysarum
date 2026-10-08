"""Evolve a growth genome to look like a target organism (CLIP-guided CMA-ES).

Target is a text prompt or a folder of reference photos. CMA-ES starts from the
QD-archive member that already best matches the target, so the search refines a
form rather than starting from noise. Each genome is grown `--reps` times and
scored by the mean, so a lucky random draw can't carry it. Each generation is
grown as one batch.

    uv run python imitate.py "an oyster mushroom" --qd outputs/qd_run1
    uv run python imitate.py --refs ~/photos/brain_coral --qd outputs/qd_run1
"""

import argparse
import glob
import json
import os
import time

import cma
import numpy as np
import polars as pl
from PIL import Image

from dgrow import evaluate
from dgrow.judge import Judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?")
    ap.add_argument("--refs", help="folder of reference photos (instead of a text target)")
    ap.add_argument("--qd", help="QD run dir to seed from")
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--max_verts", type=int, default=40000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--genome", default="param", choices=["param", "nca"])
    ap.add_argument("--w_std", type=float, default=0.4, help="nca: weight step relative to sigma")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    G = evaluate.GENOMES[a.genome]
    name = a.target or os.path.basename(os.path.normpath(a.refs))
    out = a.out or f"outputs/imitate/{name.replace(' ', '_')}"
    os.makedirs(out, exist_ok=True)

    J = Judge()
    refs = None
    if a.refs:
        refs = [Image.open(p).convert("RGB") for p in sorted(glob.glob(f"{a.refs}/*"))]
    score = J.target_scorer(a.target, refs)

    x0 = np.full(G.D, 0.5)
    if a.qd:
        df = pl.read_parquet(f"{a.qd}/evals.parquet").filter(~pl.col("dead"))
        ims = [np.asarray(Image.open(f"{a.qd}/imgs/{i}.jpg")) for i in df["id"]]
        h = ims[0].shape[0]
        s = np.concatenate([np.maximum(score([im[:, :h] for im in ims[k:k + 64]]),
                                       score([im[:, h:] for im in ims[k:k + 64]]))
                            for k in range(0, len(ims), 64)])
        best = int(np.argmax(s))
        x0 = np.array(df["genome"][best])
        print(f"seed from QD #{df['id'][best]} score {s[best]:.3f}", flush=True)

    opts = {"popsize": a.popsize, "seed": 1}
    if a.genome == "param":
        opts["bounds"] = [0, 1]
    else:
        # Physics genes are clipped in decode; small weight steps keep the seed's CA
        # pattern intact; diagonal CMA for ~1k dims.
        opts["CMA_stds"] = [1.0] * G.D_PHYS + [a.w_std] * (G.D - G.D_PHYS)
        opts["CMA_diagonal"] = True
    es = cma.CMAEvolutionStrategy(x0, a.sigma, opts)
    es.inject([x0])
    hist, best = [], (-1.0, None, None)
    t0 = time.time()
    for gen in range(a.gens):
        X = np.array(es.ask())
        res = evaluate.grow_reps(X, a.reps, max_verts=a.max_verts, seed=gen, kind=a.genome)
        fr = np.array([[max(score([r["iso"]])[0], score([r["top"]])[0]) if not r["dead"] else 0.0
                        for r in reps] for reps in res])
        f = fr.mean(1)
        es.tell(list(X), list(-f))
        k = int(np.argmax(f))
        if f[k] > max(best[0], 0.0):
            best = (float(f[k]), X[k].tolist(), fr[k].tolist())
            alive = [r for r in res[k] if not r["dead"]]
            # One row per replicate: iso | top.
            strip = np.concatenate([np.concatenate([r["iso"], r["top"]], 1) for r in alive], 0)
            Image.fromarray(strip).save(f"{out}/best.jpg", quality=92)
            Image.fromarray(strip).save(f"{out}/gen{gen:03d}.jpg", quality=85)
            for j, r in enumerate(alive):
                np.savez_compressed(f"{out}/best_{j}.npz", pos=r["pos"], faces=r["faces"],
                                    **({"color": r["color"]} if "color" in r else {}))
        hist.append(dict(gen=gen, best=float(f.max()), mean=float(f.mean())))
        print(f"[{time.time() - t0:6.0f}s] gen {gen} best {f.max():.3f} mean {f.mean():.3f} overall {best[0]:.3f}", flush=True)
    with open(f"{out}/result.json", "w") as fh:
        json.dump(dict(target=name, score=best[0], rep_scores=best[2], genome=best[1], history=hist), fh, indent=1)


if __name__ == "__main__":
    main()
