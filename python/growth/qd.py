"""MAP-Elites over growth genomes on a 2-D UMAP of image embeddings.

Descriptor = UMAP of the SigLIP 2 image embeddings of the angled + top renders, cut
into a `grid` x `grid` lattice whose axes span the 1st-99th percentile of the
fitted points (outliers land in the edge cells). UMAP is refit on the whole
history every `refit_every` evaluations and the archive rebuilt; in between,
new forms are placed with UMAP's transform. Neighbouring cells hold forms that
look alike, so niches follow visual kinds rather than PCA's colour/size axes.
Fitness = how specifically the judge (SigLIP 2, dgrow.judge) thinks the
renders are a real organism (mean over the floored x, y and angled views that
imitation scores), discounted for limbs thinner than the mesh resolves
(evaluate.thin_factor). Its label term is relative: the judge's confidence in a
form's top label over that label's `label_q` quantile in the history (at least
`label_floor`), so easy labels like lettuce coral don't outscore every other
kind and a form that is the best of a rare kind scores well. Averaged over
`reps` growths of the genome (different random draws); the replicate nearest
that mean stands for the genome in the archive.
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
                                     branch_verts=cfg.branch_verts, seed=bid, size=cfg.render_size, kind=cfg.genome)
        except Exception as e:  # a bad batch must not kill the run
            res = [[dict(dead=True, error=repr(e), n_verts=0, seed="?")] * cfg.reps for _ in ids]
        outq.put((bid, ids, genomes, res))


class Archive:
    def __init__(self, grid, n_neighbors, min_dist, seed=0):
        self.grid, self.nn, self.md, self.seed = grid, n_neighbors, min_dist, seed
        self.umap = None

    @staticmethod
    def _unit(emb):
        # Euclidean on unit vectors ranks like cosine.
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def fit(self, emb):
        from cuml.manifold import UMAP
        self.umap = UMAP(n_components=2, n_neighbors=self.nn, min_dist=self.md, random_state=self.seed)
        z = np.asarray(self.umap.fit_transform(self._unit(emb).astype(np.float32)))
        self.lo, self.hi = np.percentile(z, 1, 0), np.percentile(z, 99, 0)
        return z

    def cells(self, emb=None, z=None):
        if z is None:
            z = np.asarray(self.umap.transform(self._unit(emb).astype(np.float32)))
        g = (((z - self.lo) / (self.hi - self.lo)).clip(0, 1 - 1e-9) * self.grid).astype(int)
        return g[:, 0] * self.grid + g[:, 1], z

    def build(self, ids, emb, fit):
        """Refit, then cell -> (id, fitness) keeping the fittest per cell."""
        cells, z = self.cells(z=self.fit(emb))
        elite = {}
        for i, c, f in zip(ids, cells, fit):
            if c not in elite or f > elite[c][1]:
                elite[int(c)] = (int(i), float(f))
        return elite, dict(zip(map(int, ids), map(int, cells))), z


def fitness(r, ref):
    """ref: label -> reference confidence for the relative label term (see module doc)."""
    # Imported here: dgrow initialises Warp, which must not happen before a worker
    # has picked its GPU (spawned workers re-import this module).
    from dgrow.evaluate import thin_factor
    rel = min(1.0, r["label_p"] / ref.get(r["label"], ref[None]))
    return (0.5 * r["organic"] + 0.5 * rel) * thin_factor(r.get("thin", 0.0))


SCORES = ("dead", "organic", "label", "label_p", "thin")


def representative(reps, ref):
    """(replicate nearest the mean fitness, mean fitness, per-replicate fitness); dead reps count 0."""
    fits = [fitness(r, ref) if not r["dead"] else 0.0 for r in reps]
    alive = [i for i, r in enumerate(reps) if not r["dead"]]
    if not alive:
        return reps[0], -1.0, fits
    m = float(np.mean(fits))
    return reps[min(alive, key=lambda i: abs(fits[i] - m))], m, fits


def label_refs(rows, q, floor):
    """label -> max(floor, q-quantile of label_p over live replicates with that label); None -> floor."""
    reps = [r for row in rows for r in row["reps"] if not r["dead"]]
    ref = {None: floor}
    if reps:
        df = pl.DataFrame(reps).group_by("label").agg(pl.col("label_p").quantile(q))
        ref.update({k: max(floor, v) for k, v in df.iter_rows()})
    return ref


def montage(rows, run, path, cols=6, tile=256):
    """rows: list of dicts with id/label/fitness; tiles are the angled view."""
    n = len(rows)
    H = (n + cols - 1) // cols
    im = Image.new("RGB", (cols * tile, H * tile), (235, 235, 235))
    d = ImageDraw.Draw(im)
    for k, r in enumerate(rows):
        t = Image.open(f"{run}/imgs/{r['id']}.jpg")
        t = t.crop((2 * t.height, 0, 3 * t.height, t.height)).resize((tile, tile))
        x, y = (k % cols) * tile, (k // cols) * tile
        im.paste(t, (x, y))
        d.text((x + 5, y + 4), f"#{r['id']} {r['label']} {r['fitness']:.2f}", fill=(20, 20, 20))
    im.save(path)


def map_montage(elite, grid, run, path, tile=96):
    """The archive laid out as its grid: each cell's elite, angled view."""
    im = Image.new("RGB", (grid * tile, grid * tile), (235, 236, 238))
    for c, (i, _) in elite.items():
        t = Image.open(f"{run}/imgs/{i}.jpg")
        t = t.crop((2 * t.height, 0, 3 * t.height, t.height)).resize((tile, tile))
        im.paste(t, ((c // grid) * tile, (grid - 1 - c % grid) * tile))
    im.save(path)


@hydra.main(version_base=None, config_path="conf", config_name="qd")
def main(cfg: DictConfig):
    # Before dgrow is imported: it initialises Warp, which opens a CUDA context on
    # every visible device. Workers set their own GPU in a fresh process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.judge_gpu)
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
    judge = Judge(device="cuda")

    rows, embs = [], {}
    archive = Archive(cfg.grid, cfg.umap_neighbors, cfg.umap_min_dist, cfg.seed)
    ref = {None: cfg.label_floor}
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
        """Refresh the label references, rescore the history, refit UMAP, rebuild."""
        nonlocal ref
        ref = label_refs(rows, cfg.label_q, cfg.label_floor)
        for r in rows:
            r["fitness"] = representative(r["reps"], ref)[1]
        ok = [r for r in rows if not r["dead"]]
        ids = [r["id"] for r in ok]
        el, cell_of, z = archive.build(ids, np.stack([embs[i] for i in ids]), [r["fitness"] for r in ok])
        for r, zz in zip(ok, z):
            r["cell"], r["z"] = cell_of[r["id"]], zz.tolist()
        return el

    def save():
        df = pl.DataFrame([{k: v for k, v in r.items() if k not in ("z", "reps")} for r in rows])
        df.write_parquet(f"{run}/evals.parquet", compression="zstd")
        ids = sorted(embs)
        if not ids:
            return
        np.savez_compressed(f"{run}/embeddings.npz", ids=np.array(ids), emb=np.stack([embs[i] for i in ids]))
        with open(f"{run}/archive.json", "w") as fh:
            json.dump({str(c): v for c, v in elite.items()}, fh)
        with open(f"{run}/label_refs.json", "w") as fh:
            json.dump({str(k): v for k, v in ref.items()}, fh)
        er = [next(r for r in rows if r["id"] == e[0]) for e in elite.values()]
        er.sort(key=lambda r: -r["fitness"])
        if er:
            montage(er[:36], run, f"{run}/top.png")
            map_montage(elite, cfg.grid, run, f"{run}/map.png")

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
            r, fit, fits = representative(reps, ref)
            row = dict(id=i, batch=bid, genome=[float(x) for x in g], seed=r.get("seed", "?"),
                       dead=bool(r["dead"]), n_verts=r["n_verts"], stop_tick=r.get("stop_tick", -1),
                       organic=r.get("organic", 0.0), label=r.get("label", ""),
                       label_p=r.get("label_p", 0.0), error=r.get("error", ""),
                       branches=r.get("branches", 1), fingers=r.get("fingers", 1),
                       aspect=r.get("aspect", 0.0), thin=r.get("thin", 0.0),
                       reps=[{k: x.get(k, 0.0) for k in SCORES} for x in reps])
            row["fitness"], row["fitness_reps"] = fit, fits
            if not r["dead"]:
                embs[i] = r["emb"]
                # x | y | angled | top, the first three as scored.
                Image.fromarray(np.concatenate(r["views"] + [r["top"]], 1)).save(f"{run}/imgs/{i}.jpg", quality=88)
                np.savez_compressed(f"{run}/meshes/{i}.npz", pos=r["pos"], faces=r["faces"],
                                    **{k: r[k] for k in ("color", "material") if k in r})
            rows.append(row)
        n_done = len(rows)
        alive = [r for r in rows if not r["dead"]]
        if n_done >= cfg.n_init and (archive.umap is None or n_done - last_refit >= cfg.refit_every):
            elite = rebuild()
            last_refit = n_done
        elif archive.umap is not None:
            new = [r for r in rows[-len(ids):] if not r["dead"]]
            if new:
                cells, z = archive.cells(np.stack([embs[r["id"]] for r in new]))
                for r, c, zz in zip(new, cells, z):
                    r["cell"], r["z"] = int(c), zz.tolist()
                    if int(c) not in elite or r["fitness"] > elite[int(c)][1]:
                        elite[int(c)] = (r["id"], r["fitness"])
        qd = sum(f for _, f in elite.values())
        labels = len({r["label"] for r in rows if r["id"] in {e[0] for e in elite.values()}})
        msg = (f"[{time.time() - t0:7.0f}s] evals {n_done} alive {len(alive)} cells {len(elite)}/{cfg.grid ** 2} "
               f"labels {labels} qd {qd:.1f} best {max((f for _, f in elite.values()), default=0):.3f} "
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
