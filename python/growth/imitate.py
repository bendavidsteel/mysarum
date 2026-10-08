"""Evolve a growth genome to look like a target organism (CLIP-guided CMA-ES).

Target is a text prompt or a folder of reference photos. CMA-ES starts from the
QD-archive member that already best matches the target, so the search refines a
form rather than starting from noise. Each generation is grown as one batch.

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

from dgrow import evaluate, genome as G
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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
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

    es = cma.CMAEvolutionStrategy(x0, a.sigma, {"bounds": [0, 1], "popsize": a.popsize, "seed": 1})
    hist, best = [], (-1.0, None, None)
    t0 = time.time()
    for gen in range(a.gens):
        X = np.array(es.ask())
        res = evaluate.grow_batch(X, max_verts=a.max_verts, seed=gen)
        f = np.zeros(len(X))
        for i, r in enumerate(res):
            if not r["dead"]:
                f[i] = max(score([r["iso"]])[0], score([r["top"]])[0])
        es.tell(list(X), list(-f))
        k = int(np.argmax(f))
        if f[k] > best[0]:
            best = (float(f[k]), X[k].tolist(), res[k])
            Image.fromarray(np.concatenate([res[k]["iso"], res[k]["top"]], 1)).save(f"{out}/best.jpg", quality=92)
            np.savez_compressed(f"{out}/best.npz", pos=res[k]["pos"], faces=res[k]["faces"])
            Image.fromarray(np.concatenate([res[k]["iso"], res[k]["top"]], 1)).save(f"{out}/gen{gen:03d}.jpg", quality=85)
        hist.append(dict(gen=gen, best=float(f.max()), mean=float(f.mean())))
        print(f"[{time.time() - t0:6.0f}s] gen {gen} best {f.max():.3f} mean {f.mean():.3f} overall {best[0]:.3f}", flush=True)
    with open(f"{out}/result.json", "w") as fh:
        json.dump(dict(target=name, score=best[0], genome=best[1], history=hist), fh, indent=1)


if __name__ == "__main__":
    main()
