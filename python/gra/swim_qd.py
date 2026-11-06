"""Swimming-style search: MAP-Elites over body size x path straightness x spin.

Uses the grown bodies from `swim.py grow`. Fitness is speed: centre-of-mass
displacement per 1000 steps, measured over 200-step windows so wobbling in
place doesn't count. Genomes may hop between bodies, since every node runs the
same CA rule whatever the body. Body-size classes keep their own niches;
without them the smallest body takes over every cell.

    uv run python swim_qd.py search --tag water --gens 70
    uv run python swim_qd.py search --tag stokes --drag 0.5 1.0 0.0
    uv run python swim_qd.py render --tag water
"""

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import swim as sw

STRAIGHT_BINS = np.linspace(0.0, 1.0, 11)
SPIN_BINS = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, np.inf])
SIZE_BINS = (0, 30, 50, 80, 128, sw.N)
WINDOW = 200


def describe(trace, final_pos, active, extent0):
    """trace: (T, 3) com + orientation. Returns speed, straightness, revs/1000 steps, ok."""
    T = trace.shape[0]
    t0 = T // 5
    c = trace[t0::WINDOW, :2]
    path = jnp.sqrt(((c[1:] - c[:-1]) ** 2).sum(-1)).sum()
    disp = jnp.sqrt(((c[-1] - c[0]) ** 2).sum())
    dth = jnp.diff(trace[t0:, 2])
    dth = (dth + jnp.pi) % (2 * jnp.pi) - jnp.pi
    span = (c.shape[0] - 1) * WINDOW / 1000.0
    revs = jnp.abs(dth.sum()) / (2 * jnp.pi) / ((T - t0) / 1000.0)
    rel = final_pos - trace[-1, :2]
    radius = jnp.sqrt((rel ** 2).sum(-1) * active).max()
    ok = jnp.all(jnp.isfinite(trace)) & (radius < 3.0 * extent0 + 4.0)
    return jnp.where(ok, path / span, -1.0), disp / (path + 1e-6), revs, ok


def cell_of(straight, revs, n):
    k = int(np.searchsorted(SIZE_BINS, n, "left") - 1)
    i = int(np.clip(np.searchsorted(STRAIGHT_BINS, straight, "right") - 1, 0, len(STRAIGHT_BINS) - 2))
    j = int(np.clip(np.searchsorted(SPIN_BINS, revs, "right") - 1, 0, len(SPIN_BINS) - 2))
    return k, i, j


def run_search(args):
    out = sw.OUT / f"qd_{args.tag}.pkl"
    growers = [g for g in sw.load_growers(sw.OUT / args.growers) if args.min_n <= g["n"] <= args.max_n]
    growers.sort(key=lambda e: e["n"])
    if args.max_bodies and len(growers) > args.max_bodies:
        sel = np.linspace(0, len(growers) - 1, args.max_bodies).round().astype(int)
        growers = [growers[i] for i in sel]
    growers = sw.settle(growers)
    nb, B = len(growers), args.batch
    bodies_all = sw.stack_bodies([g["body"] for g in growers])
    print(f"{nb} bodies, batch {B}, {args.steps} steps, drag {(sw.C_T, sw.C_N, sw.C_Q)}", flush=True)

    def evaluate(body, genome, key):
        trace, _, pos = sw.swim(body, genome, key, args.steps)
        rel0 = body.pos * body.active[:, None]
        extent0 = jnp.sqrt((rel0 ** 2).sum(-1)).max()
        return describe(trace, pos, body.active, extent0)

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    archive = {}
    t0 = time.time()
    for gen in range(args.gens):
        key, kg, km, kk = jax.random.split(key, 4)
        elites = list(archive.values())
        rand = jax.vmap(sw.sample_genome)(jax.random.split(kg, B))
        if gen < 2 or not elites:
            bi = rng.integers(0, nb, B)
            genomes = rand
        else:
            pick = rng.integers(0, len(elites), B)
            bi = np.array([elites[p]["body"] for p in pick])
            bi = np.where(rng.random(B) < args.hop, rng.integers(0, nb, B), bi)
            parents = jax.tree.map(lambda *x: jnp.stack(x), *[elites[p]["genome"] for p in pick])
            genomes = jax.vmap(sw.mutate_genome)(parents, jax.random.split(km, B))
            fresh = rng.random(B) < 0.05
            genomes = jax.tree.map(
                lambda a, b: jnp.where(fresh.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), genomes, rand)
        keys = jax.random.split(kk, B)
        bodies = sw.Body(*(x[bi] for x in bodies_all))
        speed, straight, revs, ok = sw.run_bucketed(evaluate, bodies, genomes, keys)
        gnp, knp = jax.device_get((genomes, keys))
        n_new = 0
        for b in np.nonzero(ok & (speed > 0))[0]:
            cell = cell_of(straight[b], revs[b], growers[bi[b]]["n"])
            if cell not in archive or archive[cell]["speed"] < speed[b]:
                archive[cell] = dict(speed=float(speed[b]), straight=float(straight[b]),
                                     revs=float(revs[b]), body=int(bi[b]), key=knp[b],
                                     genome=jax.tree.map(lambda x: x[b], gnp))
                n_new += 1
        sp = np.array([e["speed"] for e in archive.values()])
        print(f"gen {gen:3d}  cells {len(archive):3d}  new {n_new:4d}  max {sp.max():6.2f}  "
              f"mean {sp.mean():5.2f}  ok {ok.mean():.2f}  {time.time() - t0:6.0f}s", flush=True)
        with open(out, "wb") as f:
            pickle.dump(dict(growers=growers, archive=archive, steps=args.steps,
                             drag=(sw.C_T, sw.C_N, sw.C_Q)), f)


# ── Rendering ────────────────────────────────────────────────────────────────

def style(e):
    s = "straight" if e["straight"] > 0.8 else "curving" if e["straight"] > 0.4 else "looping"
    r = e["revs"]
    spin = "" if r < 0.02 else " tumbling" if r < 0.16 else " spinning" if r < 0.64 else " whirling"
    return s + spin


def pick_diverse(elites, k, min_speed):
    """Farthest-point sampling in descriptor space, starting from the fastest."""
    es = [e for e in elites if e["speed"] >= min_speed]
    if not es:
        return []
    d = np.array([[e["straight"], np.log1p(100 * e["revs"]) / np.log1p(64)] for e in es])
    chosen = [int(np.argmax([e["speed"] for e in es]))]
    while len(chosen) < min(k, len(es)):
        dist = np.min(((d[:, None] - d[chosen][None]) ** 2).sum(-1), 1)
        chosen.append(int(np.argmax(dist)))
    return [es[i] for i in chosen]


def heatmap(archive, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted({c[0] for c in archive})
    grid = np.full((len(ks), len(STRAIGHT_BINS) - 1, len(SPIN_BINS) - 1), np.nan)
    for (k, i, j), e in archive.items():
        grid[ks.index(k), i, j] = e["speed"]
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4.4), dpi=110, sharey=True, squeeze=False)
    vmax = np.nanmax(grid)
    for ax, k, g in zip(axes[0], ks, grid):
        im = ax.imshow(g.T, origin="lower", cmap="magma", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(0, len(STRAIGHT_BINS) - 1, 2), [f"{v:.1f}" for v in STRAIGHT_BINS[:-1:2]])
        ax.set_yticks(np.arange(len(SPIN_BINS) - 1), [f"{v:g}" for v in SPIN_BINS[:-1]])
        ax.set_xlabel("path straightness")
        ax.set_title(f"bodies of {SIZE_BINS[k] + 1}-{SIZE_BINS[k + 1]} nodes", fontsize=10)
    axes[0, 0].set_ylabel("body rotation (revs / 1000 steps)")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes[0].tolist(), label="speed (displacement / 1000 steps)")
    fig.savefig(path)
    plt.close(fig)


def run_render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(sw.OUT / f"qd_{args.tag}.pkl", "rb") as f:
        res = pickle.load(f)
    sw.set_drag(res["drag"])
    growers, archive = res["growers"], res["archive"]
    vid = sw.OUT / "videos" / f"qd_{args.tag}"
    vid.mkdir(parents=True, exist_ok=True)
    heatmap(archive, vid / "map.png", f"{args.tag}  drag CT,CN,CQ={res['drag']}")
    ks = sorted({c[0] for c in archive})
    picks = []
    for k in ks:
        elites = [e for c, e in archive.items() if c[0] == k]
        floor = np.quantile([e["speed"] for e in elites], args.min_quantile)
        picks += pick_diverse(elites, max(args.top // len(ks), 1), floor)
    print(f"{len(archive)} cells, rendering {len(picks)}", flush=True)

    sds = [(sw._simulate(growers[e["body"]], e["genome"], args.steps, e["key"]),
            f"{style(e)}  n={growers[e['body']]['n']}  v={e['speed']:.1f}") for e in picks]
    cols = 4 if len(sds) > 9 else 3
    rows = (len(sds) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), dpi=110)
    fig.patch.set_facecolor(sw.BG)
    fig.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(sds):]:
        ax.set_facecolor(sw.BG)
        ax.axis("off")

    def draw(t):
        for ax, (sd, lab) in zip(axes, sds):
            sw._draw_swimmer(ax, sd, t, 4, lab)
    print("gallery", flush=True)
    sw._write_video(fig, draw, range(0, args.steps, 4), vid / "gallery.mp4", 30)

    for rank, ((sd, lab), e) in enumerate(zip(sds, picks)):
        name = f"{rank:02d}_{style(e).replace(' ', '_')}_n{growers[e['body']]['n']}_v{e['speed']:.0f}"
        print(name, flush=True)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        fig.patch.set_facecolor(sw.BG)
        fig.subplots_adjust(0, 0, 1, 1)
        sw._write_video(fig, lambda t: sw._draw_swimmer(ax, sd, t, 4, lab),
                        range(0, args.steps, 4), vid / f"{name}.mp4", 30)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = sw.common_args()
    common.add_argument("--tag", required=True)
    s = sub.add_parser("search", parents=[common])
    s.add_argument("--gens", type=int, default=70)
    s.add_argument("--batch", type=int, default=1024)
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--max-bodies", type=int, default=64)
    s.add_argument("--min-n", type=int, default=0)
    s.add_argument("--max-n", type=int, default=sw.N)
    s.add_argument("--hop", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--growers", default="growers.pkl")
    r = sub.add_parser("render", parents=[common])
    r.add_argument("--top", type=int, default=12)
    r.add_argument("--steps", type=int, default=3000)
    r.add_argument("--min-quantile", type=float, default=0.5)
    args = ap.parse_args()
    sw.configure(args)
    {"search": run_search, "render": run_render}[args.cmd](args)


if __name__ == "__main__":
    main()
