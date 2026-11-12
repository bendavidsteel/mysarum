"""Chemotaxis: swimmers that find a nutrient source from the concentration at their nodes.

Every node senses the nutrient concentration where it is, as one more additive
input to the muscle CA (Genome.C), next to edge stretch and body position.
Nothing tells the swimmer where the source is or which way it is facing: any
steering has to come out of the CA and the body.

A trial puts a source START body lengths from the swimmer; the concentration
is 1 at the source and falls linearly to 0 at RANGE body lengths. A controller
is scored on sources placed evenly all round the body, every trial from the
same CA initial state, by the share of the gap it closes, averaged over the
trial (`homing`: 0 if it never gets closer, towards 1 if it gets there fast
and stays). A swimmer that ignores the sensor swims the same path in every
trial, so it only scores on the sources that happen to lie near that path.
Controls: the same swimmer in a uniform field at the starting concentration
(nutrient but no gradient), and with no nutrient input.

    search   CMA-ES per body, from the fastest swimmers of a swim_coevo archive
    verify   re-score at swim.DT on fresh directions, CA states and distances, with both controls
    render   videos: sources all round (gradient / uniform field), and chasing a source that moves
    tracks   figure of the paths taken to sources all round, with a gradient and without

Usage:
    python swim_chemo.py search --tag homing --from coevo_net_cma.pkl
    python swim_chemo.py verify --tag homing
    python swim_chemo.py tracks --tag homing
    python swim_chemo.py render --tag homing
"""

import argparse
import pickle
import time

import jax
import numpy as np

import swim as sw
import swim_fast as sf
from swim_coevo import DT, K, REP_EVERY, SIZE_BINS, STEP_SCALE
from swim_opt import CMAES, _flatten, _layout, _unflatten, scale_vector

START = 2.5      # source distance from the swimmer's centre, body lengths
RANGE = 5.0      # distance from the source at which the concentration reaches 0, body lengths
DIRECTIONS = 6   # sources per evaluation, evenly spaced round the body
TAU = 2 * np.pi


def clip(g):
    return g._replace(log_tau=np.clip(g.log_tau, -1.5, 1.5), amp=np.clip(g.amp, 0.02, 0.45))


def told(mode, dist=START):
    """What each node is told about the nutrient, as the `sense` weights of swim_fast.simulate
    (its own concentration, the body's mean, a constant):

    on       its own concentration
    shared   the body's mean: how the level changes over time, no differences across the body
    spatial  its difference from the body's mean, on top of the starting level: differences
             across the body, nothing about the level
    uniform  the starting level everywhere: nutrient but no gradient
    off      nothing
    """
    c0 = 1.0 - dist / RANGE
    return dict(on=(1.0, 0.0, 0.0), shared=(0.0, 1.0, 0.0), spatial=(1.0, -1.0, c0), uniform=(0.0, 0.0, c0),
                off=(0.0, 0.0, 0.0))[mode]


def score(sim, source, r0, extent, n):
    """Per sim, from the centre-of-mass trace, with distances as shares of the starting distance r0:

    homing   mean over the trial of the share of the gap closed (0 while further away than at the start)
    closed   share of the gap closed at the end (mean of the last tenth; negative if it ends further away)
    nearest  share of the gap closed at the closest approach
    """
    tr = sim["trace"][..., :2].astype(np.float64)
    T = sim["steps"] // sim["every"]
    own = np.arange(tr.shape[1])[None] < T[:, None]          # samples of a sim's own length
    with np.errstate(invalid="ignore"):
        r = np.sqrt(((tr - source[:, None, :2]) ** 2).sum(-1)) / r0[:, None]
        pos = sim["pos"].astype(np.float64)
        on = np.arange(pos.shape[1])[None] < n[:, None]
        last = tr[np.arange(len(T)), T - 1]
        radius = np.sqrt(np.where(on, ((pos - last[:, None]) ** 2).sum(-1), 0.0)).max(1)
        ok = np.isfinite(np.where(own, r, 0.0)).all(1) & np.isfinite(radius) & (radius < 3.0 * extent + 4.0)
    r = np.where(own & ok[:, None], r, np.inf)       # a blown-up sim scores as never getting closer
    tail = own & (np.arange(tr.shape[1])[None] >= (0.9 * T[:, None]))
    out = dict(homing=np.where(own, np.maximum(1.0 - r, 0.0), 0.0).sum(1) / T,
               closed=1.0 - np.where(tail & ok[:, None], r, 0.0).sum(1) / tail.sum(1),
               nearest=1.0 - r.min(1))
    return dict({k: np.where(ok, v, -1.0) for k, v in out.items()}, ok=ok)


class Trials:
    """Homing trials on the swim_fast kernel: every controller meets every source direction."""

    def __init__(self, recs, time, drag, dt, rep_every):
        self.bodies = [sf.prep_body(r["body"]) for r in recs]   # recs: body records of a swim_coevo archive
        self.length = np.array([b["length"] for b in self.bodies])
        self.extent = np.array([b["extent"] for b in self.bodies])
        self.n = np.array([b["n"] for b in self.bodies])
        self.centre = np.stack([b["pos"].mean(0) for b in self.bodies])
        scale = np.array([STEP_SCALE[next(s for s in sw.BUCKETS if b["n"] <= s)] for b in self.bodies])
        self.steps = (time / dt * scale).astype(np.int64)
        self.drag, self.dt, self.rep_every = np.asarray(drag, np.float32), dt, rep_every

    def sources(self, body, angles, dist):
        """(M, D, 3) sources (x, y, range) at dist body lengths from each body's centre."""
        angles = np.broadcast_to(angles, (len(body), np.shape(angles)[-1]))
        L = self.length[body][:, None]
        c = self.centre[body]
        return np.stack([c[:, :1] + dist * L * np.cos(angles), c[:, 1:] + dist * L * np.sin(angles),
                         np.broadcast_to(RANGE * L, angles.shape)], -1)

    def __call__(self, body, genomes, keys, angles, dist=START, sensor="on"):
        """body (M,), genomes with leading M, keys (M, 2), angles (D,) or (M, D) -> scores (M, D)."""
        body = np.asarray(body)
        src = self.sources(body, angles, dist)
        D = src.shape[1]
        rep = lambda v: np.repeat(np.asarray(v), D, 0)
        g = clip(jax.tree.map(rep, genomes))
        idx = rep(body)
        flat = src.reshape(-1, 3)
        sim = sf.simulate(self.bodies, idx, g, rep(keys), self.steps[idx], drag=self.drag, dt=self.dt,
                          rep_every=self.rep_every, source=flat, sense=told(sensor, dist))
        res = score(sim, flat, dist * self.length[idx], self.extent[idx], self.n[idx])
        return {k: v.reshape(len(body), D) for k, v in res.items()}


def pick(res, per_class):
    """The fastest verified elites of each size class of a swim_coevo archive, one per body."""
    out = []
    for k, count in enumerate(per_class):
        elites = sorted((e for c, e in res["archive"].items() if c[0] == k),
                        key=lambda e: -e.get("robust", e["speed"]))
        seen = set()
        for e in elites:
            if e["body"] in seen or len(seen) == count:
                continue
            seen.add(e["body"])
            out.append(dict(rec=res["bodies"][e["body"]], genome0=sw.upgrade_genome(e["genome"], K, chem=True),
                            n=e["n"], size=k, speed=e.get("robust", e["speed"])))
    return out


def stack(genomes):
    return jax.tree.map(lambda *x: np.stack(x), *genomes)


def run_search(args):
    with open(sw.OUT / getattr(args, "from"), "rb") as f:
        res = pickle.load(f)
    if args.resume:   # carry on from an earlier search's bodies and controllers
        probs = [{k: v for k, v in dict(p, genome0=p["genome"]).items() if k != "verified"}
                 for p in load(args.resume)["probs"]]
    else:
        probs = pick(res, args.per_class)
    trials = Trials([p["rec"] for p in probs], args.time, res["drag"], args.dt, args.rep_every)
    layout = _layout(probs[0]["genome0"])
    scale = scale_vector(layout)
    P, lam, dim = len(probs), args.lam, len(scale)
    x0 = np.stack([_flatten(p["genome0"]) for p in probs])
    cmas = [CMAES(dim, lam, args.sigma, args.seed + i) for i in range(P)]
    rng = np.random.default_rng(args.seed)
    body = np.repeat(np.arange(P), lam)
    # a fixed test: sources all round, two CA initial states the search never sees
    test_angles = (np.arange(12) + 0.5) * TAU / 12
    test_keys = [np.asarray(jax.random.PRNGKey(10 ** 6 + args.seed + i)) for i in range(2)]

    def test(x, sensor="on"):
        g = _unflatten(x, layout)
        return np.mean([trials(np.arange(P), g, np.tile(k, (P, 1)), test_angles, sensor=sensor)["homing"].mean(1)
                        for k in test_keys], 0)

    for p in probs:
        p.update(test=-np.inf, history=[])
    best_x = x0.copy()
    print(f"{P} bodies, n = {[p['n'] for p in probs]}; {dim} dims, {P * lam * DIRECTIONS} sims per generation "
          f"of {trials.steps.min()}-{trials.steps.max()} steps", flush=True)
    t0 = time.time()
    for gen in range(args.gens + 1):
        if gen % args.test_every == 0 or gen == args.gens:
            mean = x0 + np.stack([c.m for c in cmas]) * scale
            t = test(mean)
            for i, p in enumerate(probs):
                p["history"].append((gen, float(t[i])))
                if t[i] > p["test"]:
                    p["test"], best_x[i] = float(t[i]), mean[i]
            if gen == 0 and not args.resume:   # the score before the sensor is used at all
                for p, v in zip(probs, t):
                    p["test0"] = float(v)
            by_size = [np.mean([p["test"] for p in probs if p["size"] == k]) for k in sorted({p["size"] for p in probs})]
            print(f"gen {gen:4d}  test homing by size class " + " ".join(f"{v:.3f}" for v in by_size)
                  + f"   best {max(p['test'] for p in probs):.3f}   sigma {np.median([c.sigma for c in cmas]):.3f}"
                  + f"   {time.time() - t0:6.0f}s", flush=True)
            for i, p in enumerate(probs):
                p["genome"] = jax.tree.map(np.asarray, clip(_unflatten(best_x[i], layout)))
            with open(sw.OUT / f"chemo_{args.tag}.pkl", "wb") as f:
                pickle.dump(dict(probs=probs, time=args.time, dt=args.dt, rep_every=args.rep_every,
                                 drag=tuple(res["drag"]), start=START, range=RANGE, gens=gen), f)
        if gen == args.gens:
            break
        u = np.stack([c.ask() for c in cmas])
        xs = x0[:, None] + u * scale
        # one CA initial state, one set of directions and one distance per generation, shared by all samples
        key = np.asarray(jax.random.PRNGKey(args.seed * 100003 + gen))
        angles = rng.uniform(0, TAU) + np.arange(DIRECTIONS) * TAU / DIRECTIONS
        dist = START * np.exp(rng.uniform(-args.jitter, args.jitter))
        f = trials(body, _unflatten(xs.reshape(-1, dim), layout), np.tile(key, (P * lam, 1)), angles, dist)
        for c, uu, ff in zip(cmas, u, f["homing"].mean(1).reshape(P, lam)):
            c.tell(uu, ff)
    flat = test(best_x, sensor="uniform")
    print("\n   n  speed  homing before -> after   after, in a uniform field")
    for p, v in zip(probs, flat):
        print(f"{p['n']:4d}  {p['speed']:5.2f}  {p['test0']:13.3f} -> {p['test']:.3f}   {v:25.3f}")


def load(tag):
    with open(sw.OUT / f"chemo_{tag}.pkl", "rb") as f:
        return pickle.load(f)


def run_verify(args):
    """Re-score with the reference integration on directions, CA initial states and distances the
    search didn't use, and with each change to what the nodes are told (see `told`)."""
    res = load(args.tag)
    probs = res["probs"]
    P = len(probs)
    trials = Trials([p["rec"] for p in probs], res["time"], res["drag"], sw.DT, 1)
    g = stack([p["genome"] for p in probs])
    angles = (np.arange(args.directions) + 0.25) * TAU / args.directions
    keys = [np.asarray(jax.random.PRNGKey(args.seed + i)) for i in range(args.keys)]
    for p in probs:
        p["verified"] = {}
    modes = tuple(NAMES)
    print(f"reference integration (dt {sw.DT}, repulsion every step), {args.directions} directions x {args.keys} "
          f"CA initial states; " + " / ".join(NAMES.values()))
    for dist in args.dist:
        runs = {s: [trials(np.arange(P), g, np.tile(k, (P, 1)), angles, dist, sensor=s) for k in keys] for s in modes}
        m = {s: {k: np.mean([r[k] for r in runs[s]], (0, 2)) for k in ("homing", "closed", "nearest")} for s in modes}
        for s in modes:   # share of trials that end within a body length of the source
            m[s]["arrive"] = np.mean([r["closed"] > 1 - 1 / dist for r in runs[s]], (0, 2))
        row = lambda v: "     ".join(" / ".join(f"{v(s, k):5.2f}" for s in modes) for k in ("homing", "arrive"))
        print(f"\nsource at {dist} body lengths (concentration at the start {1 - dist / res['range']:.2f})")
        print(f"   n   {'homing':37s}     ends within 1 body length")
        for i, p in enumerate(probs):
            p["verified"][dist] = {s: {k: float(v[i]) for k, v in m[s].items()} for s in modes}
            print(f"{p['n']:4d}   " + row(lambda s, k: m[s][k][i]))
        print(" all   " + row(lambda s, k: m[s][k].mean()))
    with open(sw.OUT / f"chemo_{args.tag}.pkl", "wb") as f:
        pickle.dump(res, f)

# ── Videos and figures ───────────────────────────────────────────────────────

FRAME_T = 0.08   # time per frame of the standard swimmer videos
NAMES = dict(on="gradient", shared="body mean only", spatial="differences across the body only",
             uniform="uniform field", off="no nutrient input")


def trial_steps(p, time, dt=None):
    return int(time / (sw.DT if dt is None else dt) * STEP_SCALE[next(s for s in sw.BUCKETS if p["n"] <= s)])


def record(p, genome, keys, steps, source, source_steps, drag, stride, mode="on"):
    """Run one body on several source paths (S, Q, 3) at the reference integration; each run as a
    renderer recording of every stride-th step, with the source it saw."""
    body = sf.prep_body(p["rec"]["body"])
    S = len(source)
    g = jax.tree.map(lambda v: np.repeat(np.asarray(v)[None], S, 0), genome)
    sim = sf.simulate([body], np.zeros(S, int), g, np.asarray(keys), steps, drag=drag, record=stride,
                      source=source, source_steps=source_steps, sense=told(mode))
    n, edges = body["n"], body["edges"]
    seg = np.minimum(np.arange(0, steps, stride) // source_steps, source.shape[1] - 1)
    out = []
    for s in range(S):
        pos = sim["rec_pos"][s, :, :n]
        act = np.zeros((len(pos), n, sw.D), sf.F)
        act[:, edges[:, 0], edges[:, 1]] = sim["rec_act"][s]
        out.append(dict(com=pos.mean(1), pos=pos, x=sim["rec_x"][s, :, :n], act=act, active=np.ones(n, bool),
                        edges=edges, source=source[s][seg]))
    return out


def overlay(sds, view):
    """Independent runs drawn in one panel with a fixed camera."""
    off = np.cumsum([0] + [sd["pos"].shape[1] for sd in sds])
    cat = lambda k: np.concatenate([sd[k] for sd in sds], 1)
    return dict(com=sds[0]["com"], coms=[sd["com"] for sd in sds], pos=cat("pos"), x=cat("x"), act=cat("act"),
                active=np.ones(off[-1], bool), view=view, extent=view[2],
                **({"source": sds[0]["source"]} if "source" in sds[0] else {}),
                edges=np.concatenate([sd["edges"] + (o, 0, o) for sd, o in zip(sds, off)]))


def best(probs, top, dist=START):
    """The best verified homers, at most two per size class."""
    order = sorted(probs, key=lambda p: -p["verified"][dist]["on"]["homing"])
    out = []
    for p in order:
        if sum(q["size"] == p["size"] for q in out) < 2 and len(out) < top:
            out.append(p)
    return out


def run_render(args):
    res = load(args.tag)
    probs = best(res["probs"], args.top)
    vid = sw.OUT / "videos" / f"chemo_{args.tag}"
    vid.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    specs = []

    # 1. sources all round: each panel is args.runs independent runs of one swimmer, drawn with the
    # source at the centre, so the runs start on a ring around it. Bottom row: the same swimmers in
    # a uniform field at the starting concentration, so they have nothing to steer by.
    angles = (np.arange(args.runs) + 0.5) * TAU / args.runs
    key = np.asarray(jax.random.PRNGKey(args.seed))
    rows = {"on": [], "uniform": []}
    for p in probs[:args.cols]:
        body = sf.prep_body(p["rec"]["body"])
        L = body["length"]
        steps = trial_steps(p, res["time"])
        stride = max(steps // args.frames, 1)
        c = body["pos"].mean(0)
        src = np.stack([c[0] + START * L * np.cos(angles), c[1] + START * L * np.sin(angles),
                        np.full(args.runs, RANGE * L)], 1).astype(sf.F)[:, None]
        for sensor in rows:
            sds = record(p, p["genome"], np.tile(key, (args.runs, 1)), steps, src, steps, res["drag"], stride,
                         sensor)
            for sd, s in zip(sds, src[:, 0]):       # shift each run so its source is at the origin
                sd["pos"] = sd["pos"] - s[:2]
                sd["com"] = sd["com"] - s[:2]
                sd["source"] = sd["source"] - (s[0], s[1], 0)
            if sensor == "uniform":
                for sd in sds:
                    del sd["source"]
            label = f"n={p['n']}  {NAMES[sensor]}  reaches it {p['verified'][START][sensor]['arrive']:.0%}  " \
                    f"x{stride * sw.DT / FRAME_T:.1f} speed"
            rows[sensor].append((overlay(sds, (0.0, 0.0, (START + 0.9) * L)), label))
    frames = min(len(sd["pos"]) for sd, _ in rows["on"])
    specs.append(dict(path=vid / "all_round.mp4", panels=rows["on"] + rows["uniform"], cols=len(rows["on"]), cell=4.0,
                      dpi=110, steps=frames, every=1, fps=30))

    # 2. chasing: the source jumps to a new place START body lengths on, turning sharply each time
    panels = []
    for p in probs:
        body = sf.prep_body(p["rec"]["body"])
        L = body["length"]
        seg = trial_steps(p, res["time"])
        steps = seg * args.jumps
        stride = max(steps // (args.frames * 2), 1)
        heading = rng.uniform(0, TAU) + np.arange(args.jumps) * rng.choice([-1, 1]) * 0.8 * np.pi
        way = body["pos"].mean(0) + np.cumsum(START * L * np.stack([np.cos(heading), np.sin(heading)], 1), 0)
        src = np.concatenate([way, np.full((args.jumps, 1), RANGE * L)], 1).astype(sf.F)[None]
        sd = record(p, p["genome"], key[None], steps, src, seg, res["drag"], stride)[0]
        ends = np.minimum(np.arange(1, args.jumps + 1) * seg // stride, len(sd["com"])) - 1
        gap = np.sqrt(((sd["com"][ends] - way) ** 2).sum(1)) / L
        print(f"chase n={p['n']:4d}  body lengths from the source at the end of each leg: "
              + " ".join(f"{v:.2f}" for v in gap), flush=True)
        lo, hi = np.minimum(way.min(0), body["pos"].min(0)), np.maximum(way.max(0), body["pos"].max(0))
        sd.update(view=(*(lo + hi) / 2, (hi - lo).max() / 2 + 0.8 * L), extent=(hi - lo).max() / 2 + 0.8 * L)
        panels.append((sd, f"n={p['n']}  x{stride * sw.DT / FRAME_T:.1f} speed"))
    frames = min(len(sd["pos"]) for sd, _ in panels)
    specs.append(dict(path=vid / "chase.mp4", panels=panels, cols=min(3, len(panels)), cell=3.6, dpi=110,
                      steps=frames, every=1, fps=30))
    print(f"rendering {len(specs)} videos", flush=True)
    sw.render_videos(specs)


def run_tracks(args):
    """Paths of the centre of mass with sources all round (top) and in a uniform field (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from swim_render import BG, GRID

    res = load(args.tag)
    probs = best(res["probs"], args.top)
    trials = Trials([p["rec"] for p in probs], res["time"], res["drag"], sw.DT, 1)
    angles = (np.arange(args.runs) + 0.5) * TAU / args.runs
    key = np.asarray(jax.random.PRNGKey(args.seed))
    colors = hsv_to_rgb(np.stack([angles / TAU, np.full(args.runs, 0.55), np.full(args.runs, 0.95)], 1))
    fig, axes = plt.subplots(2, len(probs), figsize=(2.6 * len(probs), 5.6), dpi=150, squeeze=False)
    fig.patch.set_facecolor(BG)
    for j, p in enumerate(probs):
        for i, sensor in enumerate(("on", "uniform")):
            g = jax.tree.map(lambda v: np.repeat(np.asarray(v)[None], args.runs, 0), p["genome"])
            src = trials.sources(np.full(args.runs, j), angles[:, None], START)[:, 0]
            sim = sf.simulate([trials.bodies[j]], np.zeros(args.runs, int), g, np.tile(key, (args.runs, 1)),
                              trials.steps[j], drag=trials.drag, source=src, sense=told(sensor))
            L = trials.length[j]
            tr = (sim["trace"][:, :, :2] - trials.centre[j]) / L
            s = (src[:, :2] - trials.centre[j]) / L
            ax = axes[i, j]
            ax.set_facecolor(BG)
            ax.add_patch(plt.Circle((0, 0), START, fill=False, color=GRID, lw=0.8))
            for k in range(args.runs):
                ax.plot(tr[k, :, 0], tr[k, :, 1], color=colors[k], lw=1.0, alpha=0.9)
                ax.plot(*tr[k, -1], "o", color=colors[k], ms=2.5)
            ax.scatter(s[:, 0], s[:, 1], s=26, c=colors, edgecolors="white", linewidths=0.5, zorder=3)
            body = (trials.bodies[j]["pos"] - trials.centre[j]) / L
            ax.scatter(body[:, 0], body[:, 1], s=2, color="#8899aa", zorder=2)
            lim = START * 1.45
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            v = p["verified"][START][sensor]
            ax.set_title(f"n={p['n']}, {NAMES[sensor]}\nends within 1 body length: {v['arrive']:.0%}",
                         color="#c8d0d8", fontsize=8)
    fig.suptitle(f"Centre-of-mass paths, one run per nutrient source (dots, {START} body lengths away; path and "
                 "source share a colour).\nEvery run starts from the same state. Bottom row: the same swimmers "
                 "with the same nutrient level everywhere, so no gradient.", color="#c8d0d8", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.0)
    out = sw.OUT / f"chemo_{args.tag}_tracks.png"
    fig.savefig(out, facecolor=BG)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = sw.common_args()
    s = sub.add_parser("search", parents=[common])
    s.add_argument("--tag", default="homing")
    s.add_argument("--from", default="coevo_net_cma.pkl", help="swim_coevo archive to take bodies and controllers from")
    s.add_argument("--resume", help="tag of an earlier chemotaxis search to carry on from")
    s.add_argument("--per-class", type=int, nargs=len(SIZE_BINS) - 1, default=(6, 6, 4, 3, 2),
                   help="bodies per size class")
    s.add_argument("--gens", type=int, default=300)
    s.add_argument("--lam", type=int, default=32)
    s.add_argument("--sigma", type=float, default=0.3)
    s.add_argument("--time", type=float, default=80.0, help="trial length for the smallest bodies")
    s.add_argument("--jitter", type=float, default=0.5, help="log-uniform spread of the source distance")
    s.add_argument("--dt", type=float, default=DT)
    s.add_argument("--rep-every", type=int, default=REP_EVERY)
    s.add_argument("--test-every", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("verify", parents=[common])
    v.add_argument("--tag", default="homing")
    v.add_argument("--directions", type=int, default=12)
    v.add_argument("--keys", type=int, default=4)
    v.add_argument("--dist", type=float, nargs="+", default=(START, 1.5, 4.0))
    v.add_argument("--seed", type=int, default=100)
    r = sub.add_parser("render", parents=[common])
    r.add_argument("--tag", default="homing")
    r.add_argument("--top", type=int, default=9, help="swimmers in the chase video")
    r.add_argument("--cols", type=int, default=4, help="swimmers in the all-round video")
    r.add_argument("--runs", type=int, default=8, help="runs per panel of the all-round video")
    r.add_argument("--jumps", type=int, default=4, help="places the source visits in the chase video")
    r.add_argument("--frames", type=int, default=600, help="frames per trial length")
    r.add_argument("--seed", type=int, default=7)
    t = sub.add_parser("tracks", parents=[common])
    t.add_argument("--tag", default="homing")
    t.add_argument("--top", type=int, default=6)
    t.add_argument("--runs", type=int, default=12)
    t.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    sw.configure(args)
    {"search": run_search, "verify": run_verify, "render": run_render, "tracks": run_tracks}[args.cmd](args)


if __name__ == "__main__":
    main()
