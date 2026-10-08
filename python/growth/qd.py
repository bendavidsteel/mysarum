"""CVT-MAP-Elites over growth genomes, with image-embedding descriptors.

Descriptor = PCA of DINOv2 embeddings of the iso + top renders (refit on the
whole history every `refit_every` evaluations, then the archive is rebuilt).
Fitness = how specifically CLIP thinks the render is a real organism, averaged
over `reps` growths of the genome (different random draws); the replicate
nearest that mean stands for the genome in the archive.
One grower process per GPU; the judge runs in the main process.

    uv run python qd.py                       # -> outputs/qd_<timestamp>/
    uv run python qd.py n_evals=8000 gpus=[0,1] batch=32
    uv run python qd.py genome=nca            # neural-CA genomes (nca.py)
"""

import io
import json
import os
import queue
import time

import hydra
import numpy as np
import polars as pl
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def worker(gpu, cfg, inq, outq):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from dgrow import evaluate
    while True:
        job = inq.get()
        if job is None:
            return
        bid, ids, genomes = job
        try:
            res = evaluate.grow_reps(genomes, cfg.reps, max_verts=cfg.max_verts, max_ticks=cfg.max_ticks,
                                     seed=bid, size=cfg.render_size, kind=cfg.genome)
        except Exception as e:  # a bad batch must not kill the run
            res = [[dict(dead=True, error=repr(e), n_verts=0, seed="?")] * cfg.reps for _ in ids]
        outq.put((bid, ids, genomes, res))


class Archive:
    def __init__(self, n_cells, dims, seed=0):
        self.n_cells, self.dims, self.seed = n_cells, dims, seed
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(40 * n_cells, dims))
        z = z[np.abs(z).max(1) < 2.5]
        self.centroids = KMeans(n_cells, n_init=1, random_state=seed).fit(z).cluster_centers_
        self.pca = None

    def fit(self, emb):
        self.pca = PCA(self.dims, whiten=True, random_state=self.seed).fit(emb)

    def cells(self, emb):
        z = self.pca.transform(emb)
        d = ((z[:, None, :] - self.centroids[None]) ** 2).sum(-1)
        return d.argmin(1), z

    def build(self, ids, emb, fit):
        """cell -> (id, fitness) keeping the fittest per cell."""
        cells, z = self.cells(emb)
        elite = {}
        for i, c, f in zip(ids, cells, fit):
            if c not in elite or f > elite[c][1]:
                elite[int(c)] = (int(i), float(f))
        return elite, dict(zip(map(int, ids), map(int, cells))), z


def fitness(r):
    return 0.5 * r["organic"] + 0.5 * r["label_p"]


def representative(reps):
    """(replicate nearest the mean fitness, mean fitness, per-replicate fitness); dead reps count 0."""
    fits = [fitness(r) if not r["dead"] else 0.0 for r in reps]
    alive = [i for i, r in enumerate(reps) if not r["dead"]]
    if not alive:
        return reps[0], -1.0, fits
    m = float(np.mean(fits))
    return reps[min(alive, key=lambda i: abs(fits[i] - m))], m, fits


def montage(rows, run, path, cols=6, tile=256):
    """rows: list of dicts with id/label/fitness; tiles are the iso render."""
    n = len(rows)
    H = (n + cols - 1) // cols
    im = Image.new("RGB", (cols * tile, H * tile), (235, 235, 235))
    d = ImageDraw.Draw(im)
    for k, r in enumerate(rows):
        t = Image.open(f"{run}/imgs/{r['id']}.jpg")
        t = t.crop((0, 0, t.height, t.height)).resize((tile, tile))
        x, y = (k % cols) * tile, (k // cols) * tile
        im.paste(t, (x, y))
        d.text((x + 5, y + 4), f"#{r['id']} {r['label']} {r['fitness']:.2f}", fill=(20, 20, 20))
    im.save(path)


def farthest(z, k, rng):
    sel = [int(rng.integers(len(z)))]
    d = ((z - z[sel[0]]) ** 2).sum(1)
    while len(sel) < min(k, len(z)):
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, ((z - z[i]) ** 2).sum(1))
    return sel


@hydra.main(version_base=None, config_path="conf", config_name="qd")
def main(cfg: DictConfig):
    from dgrow.evaluate import GENOMES
    from dgrow.judge import Judge

    G = GENOMES[cfg.genome]

    run = os.getcwd() if cfg.out is None else cfg.out
    os.makedirs(f"{run}/imgs", exist_ok=True)
    os.makedirs(f"{run}/meshes", exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    ctx = mp.get_context("spawn")
    inq, outq = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(g, cfg, inq, outq), daemon=True) for g in cfg.gpus]
    for p in procs:
        p.start()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.judge_gpu)
    judge = Judge(device="cuda")

    rows, embs = [], {}
    archive = Archive(cfg.n_cells, cfg.dims, cfg.seed)
    elite, next_id, next_bid, in_flight = {}, 0, 0, 0
    last_refit = 0
    t0 = time.time()
    log = open(f"{run}/qd.log", "a")

    def emit():
        nonlocal next_id, next_bid, in_flight
        if not elite or next_id < cfg.n_init:
            g = G.random(cfg.batch, rng)
        else:
            pool = [e[0] for e in elite.values()]
            genomes = {r["id"]: np.array(r["genome"]) for r in rows if r["id"] in set(pool)}
            pa = np.stack([genomes[pool[i]] for i in rng.integers(len(pool), size=cfg.batch)])
            pb = np.stack([genomes[pool[i]] for i in rng.integers(len(pool), size=cfg.batch)])
            cross = rng.random(cfg.batch) < cfg.p_cross
            g = np.where(cross[:, None], G.iso_dd(pa, pb, rng), G.mutate(pa, rng, cfg.sigma))
        ids = list(range(next_id, next_id + cfg.batch))
        inq.put((next_bid, ids, g))
        next_id += cfg.batch
        next_bid += 1
        in_flight += 1

    def rebuild():
        ok = [r for r in rows if not r["dead"]]
        ids = [r["id"] for r in ok]
        E = np.stack([embs[i] for i in ids])
        archive.fit(E)
        el, cell_of, z = archive.build(ids, E, [r["fitness"] for r in ok])
        for r, zz in zip(ok, z):
            r["cell"], r["z"] = cell_of[r["id"]], zz.tolist()
        return el

    def save():
        df = pl.DataFrame([{k: v for k, v in r.items() if k != "z"} for r in rows])
        df.write_parquet(f"{run}/evals.parquet", compression="zstd")
        ids = sorted(embs)
        np.savez_compressed(f"{run}/embeddings.npz", ids=np.array(ids), dino=np.stack([embs[i] for i in ids]))
        with open(f"{run}/archive.json", "w") as fh:
            json.dump({str(c): v for c, v in elite.items()}, fh)
        er = [next(r for r in rows if r["id"] == e[0]) for e in elite.values()]
        er.sort(key=lambda r: -r["fitness"])
        if er:
            montage(er[:36], run, f"{run}/top.png")
            z = np.array([r["z"] for r in er])
            montage([er[i] for i in farthest(z, 36, np.random.default_rng(0))], run, f"{run}/diverse.png")

    for _ in range(2 * len(procs)):
        emit()
    while in_flight:
        try:
            bid, ids, genomes, res = outq.get(timeout=1800)
        except queue.Empty:
            print("timeout waiting for workers", file=log, flush=True)
            break
        in_flight -= 1
        judge.score([r for reps in res for r in reps])
        for i, g, reps in zip(ids, genomes, res):
            r, fit, fits = representative(reps)
            row = dict(id=i, batch=bid, genome=[float(x) for x in g], seed=r.get("seed", "?"),
                       dead=bool(r["dead"]), n_verts=r["n_verts"], stop_tick=r.get("stop_tick", -1),
                       organic=r.get("organic", 0.0), label=r.get("label", ""),
                       label_p=r.get("label_p", 0.0), error=r.get("error", ""))
            row["fitness"], row["fitness_reps"] = fit, fits
            if not r["dead"]:
                embs[i] = r["dino"]
                Image.fromarray(np.concatenate([r["iso"], r["top"]], 1)).save(f"{run}/imgs/{i}.jpg", quality=88)
                np.savez_compressed(f"{run}/meshes/{i}.npz", pos=r["pos"], faces=r["faces"],
                                    **({"color": r["color"]} if "color" in r else {}))
            rows.append(row)
        n_done = len(rows)
        alive = [r for r in rows if not r["dead"]]
        if n_done >= cfg.n_init and (archive.pca is None or n_done - last_refit >= cfg.refit_every):
            elite = rebuild()
            last_refit = n_done
        elif archive.pca is not None:
            new = [r for r in rows[-len(ids):] if not r["dead"]]
            if new:
                cells, z = archive.cells(np.stack([embs[r["id"]] for r in new]))
                for r, c, zz in zip(new, cells, z):
                    r["cell"], r["z"] = int(c), zz.tolist()
                    if int(c) not in elite or r["fitness"] > elite[int(c)][1]:
                        elite[int(c)] = (r["id"], r["fitness"])
        qd = sum(f for _, f in elite.values())
        msg = (f"[{time.time() - t0:7.0f}s] evals {n_done} alive {len(alive)} cells {len(elite)}/{cfg.n_cells} "
               f"qd {qd:.1f} best {max((f for _, f in elite.values()), default=0):.3f} "
               f"batch {res[0][0].get('secs', 0):.0f}s")
        print(msg, flush=True)
        print(msg, file=log, flush=True)
        if bid % cfg.save_every == 0:
            save()
        if next_id < cfg.n_evals and time.time() - t0 < cfg.max_hours * 3600:
            emit()
    save()
    for _ in procs:
        inq.put(None)
    OmegaConf.save(cfg, f"{run}/config.yaml")


if __name__ == "__main__":
    main()
