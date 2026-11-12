"""Speed optimisation of swimmer controllers with CMA-ES.

Seeds are the fastest elites of swim_qd results. Each problem is one
(seed swimmer, controller arm) pair with its own CMA-ES; all problems run in
one batch, since drag is passed per sim. Every genome is held at 8 channels,
and arms differ only in which parameters CMA-ES may move:

    base  the original 4-channel controller (position input frozen at zero)
    pos   + node position along / across the body axis
    pos8  + position input and 8 channels

Frozen channels have no coupling and no readout, so they are inert.

    uv run python swim_opt.py search --fluids water_sized stokes isotropic eel --per-fluid 2
    uv run python swim_opt.py render
"""

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import swim as sw
from swim_qd import describe

K8 = 8
ARMS = ("base", "pos", "pos8")
# rough per-field step sizes, so CMA-ES works in a roughly isotropic space
SCALE = dict(A=4.0, a_self=2.0, U=2.0, b=2.0, log_tau=0.5, r=0.3, r0=0.3, amp=0.1, P=1.0, S=1.0, C=4.0)


def _flatten(g):
    """Flat vector of the fields the genome has (a missing input, like C, is None)."""
    return np.concatenate([np.ravel(v) for v in g if v is not None])


def _layout(g):
    """Per-field shapes and offsets of the flat vector."""
    out, i = {}, 0
    for f in sw.Genome._fields:
        if getattr(g, f) is None:
            continue
        shape = np.shape(getattr(g, f))
        size = int(np.prod(shape))
        out[f] = (i, i + size, shape)
        i += size
    return out


def _unflatten(v, layout):
    """Flat vector(s) (..., D) -> Genome with leading batch dims."""
    lead = v.shape[:-1]
    return sw.Genome(**{f: v[..., a:b].reshape(lead + shape) for f, (a, b, shape) in layout.items()})


def free_mask(layout, arm, k0=4):
    """Which flat coordinates CMA-ES may move for an arm."""
    m = {}
    for f, (a, b, shape) in layout.items():
        mf = np.ones(shape, bool)
        if arm != "pos8" and f != "r0" and f != "amp":
            mf[k0:] = False           # channels beyond the original four
            if f == "A":
                mf[:, k0:] = False
        if arm == "base" and f == "P":
            mf[:] = False
        m[f] = mf.ravel()
    return np.concatenate([m[f] for f in layout])


def scale_vector(layout):
    return np.concatenate([np.full(b - a, SCALE[f]) for f, (a, b, _) in layout.items()])


class CMAES:
    """Standard (mu/mu_w, lambda)-CMA-ES, maximising."""

    def __init__(self, n, lam, sigma=0.5, seed=0):
        self.n, self.lam, self.mu = n, lam, lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.m, self.sigma = np.zeros(n), sigma
        self.C, self.B, self.D = np.eye(n), np.eye(n), np.ones(n)
        self.pc, self.ps, self.gen = np.zeros(n), np.zeros(n), 0
        self.rng = np.random.default_rng(seed)

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        return self.m + self.sigma * (z * self.D) @ self.B.T

    def tell(self, xs, f):
        y = (xs - self.m) / self.sigma
        ysel = y[np.argsort(-f)[:self.mu]]
        ymean = self.w @ ysel
        self.m = self.m + self.sigma * ymean
        inv_sqrt_c = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt_c @ ymean
        self.gen += 1
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chi_n
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (ysel.T * self.w) @ ysel)
        self.sigma *= np.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / self.chi_n - 1))
        d2, self.B = np.linalg.eigh((self.C + self.C.T) / 2)
        self.D = np.sqrt(np.maximum(d2, 1e-20))


def clip(g):
    return g._replace(log_tau=jnp.clip(g.log_tau, -1.5, 1.5), amp=jnp.clip(g.amp, 0.02, 0.45))


def make_eval(steps, scaled=False):
    """scaled: swim longer on bigger size buckets, as swim_coevo does."""
    from swim_coevo import STEP_SCALE

    def fn(body, genome, key, drag):
        n_steps = int(steps * STEP_SCALE[body.pos.shape[0]]) if scaled else steps
        trace, _, pos = sw.swim(body, clip(genome), key, n_steps, drag=drag)
        rel0 = body.pos * body.active[:, None]
        extent0 = jnp.sqrt((rel0 ** 2).sum(-1)).max()
        speed, straight, _, ok = describe(trace, pos, body.active, extent0)
        return speed, straight
    return fn


def load_seeds(fluids, per_fluid):
    """Fastest elites per fluid, on distinct bodies."""
    seeds = []
    for tag in fluids:
        with open(sw.OUT / f"qd_{tag}.pkl", "rb") as f:
            res = pickle.load(f)
        used = set()
        for e in sorted(res["archive"].values(), key=lambda e: -e["speed"]):
            if e["body"] in used:
                continue
            used.add(e["body"])
            g = res["growers"][e["body"]]
            seeds.append(dict(fluid=tag, drag=res["drag"], grower=g, n=g["n"], key=e["key"],
                              genome=sw.upgrade_genome(e["genome"], K8), qd_speed=e["speed"]))
            if len([s for s in seeds if s["fluid"] == tag]) == per_fluid:
                break
    return seeds


def load_coevo_seeds(tag, sizes, per_size):
    """Per swim_coevo size class: the most elongated bodies, plus the fastest body as a control.
    Each starts from the best controller found on it."""
    import swim_coevo as sc
    with open(sw.OUT / f"coevo_{tag}.pkl", "rb") as f:
        res = pickle.load(f)
    best_on = {}
    for c, e in res["archive"].items():
        if e["body"] not in best_on or best_on[e["body"]][1]["speed"] < e["speed"]:
            best_on[e["body"]] = (c, e)
    seeds = []
    for k in sizes:
        mine = [(c, e) for c, e in best_on.values() if c[0] == k]
        pick = sorted(mine, key=lambda ce: -ce[1]["aspect"])[:per_size]
        fastest = max(mine, key=lambda ce: ce[1]["speed"])
        pick += [fastest] if fastest not in pick else []
        for c, e in pick:
            rec = res["bodies"][e["body"]]
            seeds.append(dict(fluid=f"a{e['aspect']:.1f}", drag=res["drag"], grower=rec, n=rec["n"], key=e["key"],
                              genome=e["genome"], qd_speed=e["speed"], length=e["length"]))
    return seeds


def run_search(args):
    seeds = load_coevo_seeds(args.coevo, args.sizes, args.per_size) if args.coevo \
        else load_seeds(args.fluids, args.per_fluid)
    layout = _layout(seeds[0]["genome"])
    scale = scale_vector(layout)
    probs = [dict(seed=s, arm=a, x0=_flatten(s["genome"]), mask=free_mask(layout, a))
             for s in seeds for a in args.arms]
    for i, p in enumerate(probs):
        p["cma"] = CMAES(int(p["mask"].sum()), args.lam, args.sigma, seed=args.seed + i)
        p["best"], p["best_x"], p["hist"] = -np.inf, p["x0"].copy(), []
    P, L = len(probs), args.lam
    print(f"{len(seeds)} seeds x {len(args.arms)} arms = {P} problems, lambda {L}, "
          f"{P * L} sims/gen, {args.steps} steps", flush=True)

    bodies = sw.stack_bodies([p["seed"]["grower"]["body"] for p in probs for _ in range(L)])
    drags = np.repeat(np.array([p["seed"]["drag"] for p in probs], np.float32), L, 0)
    evaluate = make_eval(args.steps, scaled=bool(args.coevo))
    key = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    for gen in range(args.gens):
        key, kg = jax.random.split(key)
        # common random numbers: one CA initial state per problem per generation
        keys = np.repeat(np.asarray(jax.random.split(kg, P)), L, 0)
        us = [p["cma"].ask() for p in probs]
        xs = []
        for p, u in zip(probs, us):
            x = np.repeat(p["x0"][None], L, 0)
            x[:, p["mask"]] += u * scale[p["mask"]]
            xs.append(x)
        xs = np.concatenate(xs).astype(np.float32)
        speed, _ = sw.run_bucketed(evaluate, bodies, _unflatten(xs, layout), keys, drags)
        speed = np.where(np.isfinite(speed), speed, -1.0)
        for i, (p, u) in enumerate(zip(probs, us)):
            f = speed[i * L:(i + 1) * L]
            p["cma"].tell(u, f)
            if f.max() > p["best"]:
                p["best"], p["best_x"] = float(f.max()), xs[i * L + int(f.argmax())].copy()
            p["hist"].append(float(f.max()))
        if gen % args.log_every == 0 or gen == args.gens - 1:
            by_arm = {a: np.mean([p["best"] for p in probs if p["arm"] == a]) for a in args.arms}
            print(f"gen {gen:4d}  mean best by arm " + "  ".join(f"{a} {v:5.2f}" for a, v in by_arm.items())
                  + f"  {time.time() - t0:6.0f}s", flush=True)
            save(args, probs, layout)
    final_eval(args, probs, layout, evaluate)
    save(args, probs, layout)


def final_eval(args, probs, layout, evaluate, n_keys=8):
    """Re-score seed and best genome on fresh CA initial states (the search
    used one shared state per generation, so its best can be lucky)."""
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(10_000 + args.seed), n_keys))
    xs, bodies, drags, ks = [], [], [], []
    for p in probs:
        for x in (p["x0"], p["best_x"]):
            for k in keys:
                xs.append(x)
                bodies.append(p["seed"]["grower"]["body"])
                drags.append(p["seed"]["drag"])
                ks.append(k)
    speed, straight = sw.run_bucketed(
        evaluate, sw.stack_bodies(bodies), _unflatten(np.array(xs, np.float32), layout),
        np.array(ks), np.array(drags, np.float32))
    speed = speed.reshape(len(probs), 2, n_keys)
    straight = straight.reshape(len(probs), 2, n_keys)
    print(f"\n{'fluid':12s} {'n':>4s} {'arm':5s} {'seed':>6s} {'opt':>6s} {'x':>5s}  straight")
    for p, sp, st in zip(probs, speed, straight):
        p["seed_speed"], p["opt_speed"] = float(sp[0].mean()), float(sp[1].mean())
        p["opt_speed_min"], p["opt_straight"] = float(sp[1].min()), float(st[1].mean())
        s = p["seed"]
        bl = f"  BL/1000 {p['seed_speed'] / s['length']:.3f} -> {p['opt_speed'] / s['length']:.3f}" \
            if "length" in s else ""
        print(f"{s['fluid']:12s} {s['n']:4d} {p['arm']:5s} {p['seed_speed']:6.2f} {p['opt_speed']:6.2f} "
              f"{p['opt_speed'] / max(p['seed_speed'], 1e-3):5.1f}  {p['opt_straight']:.2f}{bl}")
    for a in args.arms:
        ps = [p for p in probs if p["arm"] == a]
        print(f"arm {a:5s}  mean seed {np.mean([p['seed_speed'] for p in ps]):5.2f}  "
              f"mean opt {np.mean([p['opt_speed'] for p in ps]):5.2f}")


def save(args, probs, layout):
    out = [dict(fluid=p["seed"]["fluid"], drag=p["seed"]["drag"], grower=p["seed"]["grower"],
                key=p["seed"]["key"], arm=p["arm"], hist=p["hist"],
                seed_genome=_unflatten(p["x0"], layout),
                genome=jax.tree.map(np.asarray, clip(_unflatten(p["best_x"], layout))),
                **{k: p[k] for k in ("seed_speed", "opt_speed", "opt_speed_min", "opt_straight") if k in p})
           for p in probs]
    with open(sw.OUT / f"opt_{args.tag}.pkl", "wb") as f:
        pickle.dump(out, f)


def run_render(args):
    with open(sw.OUT / f"opt_{args.tag}.pkl", "rb") as f:
        probs = pickle.load(f)
    vid = sw.OUT / "videos" / f"opt_{args.tag}"
    vid.mkdir(parents=True, exist_ok=True)
    # per seed swimmer: the seed controller beside the best arm
    by_seed = {}
    for p in probs:
        by_seed.setdefault((p["fluid"], p["grower"]["n"]), []).append(p)
    panels, specs = [], []
    for (fluid, n), ps in by_seed.items():
        best = max(ps, key=lambda p: p.get("opt_speed", -1))
        seed_sd = sw._simulate(best["grower"], best["seed_genome"], args.steps, best["key"], best["drag"])
        opt_sd = sw._simulate(best["grower"], best["genome"], args.steps, best["key"], best["drag"])
        panels += [(seed_sd, f"{fluid} n={n} seed  v={best['seed_speed']:.1f}"),
                   (opt_sd, f"{fluid} n={n} {best['arm']}  v={best['opt_speed']:.1f}")]
        specs.append(sw.swimmer_spec(vid / f"{fluid}_n{n}_{best['arm']}_v{best['opt_speed']:.0f}.mp4",
                                     opt_sd, args.steps, panels[-1][1]))
    specs.insert(0, sw.gallery_spec(vid / "before_after.mp4", panels, args.steps, cols=4))
    print(f"rendering {len(specs)} videos", flush=True)
    sw.render_videos(specs)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = sw.common_args()
    common.add_argument("--tag", default="speed")
    s = sub.add_parser("search", parents=[common])
    s.add_argument("--fluids", nargs="+", default=["water_sized", "stokes", "isotropic", "eel"])
    s.add_argument("--per-fluid", type=int, default=2)
    s.add_argument("--coevo", default=None, help="swim_coevo tag: optimise its elongated bodies instead")
    s.add_argument("--sizes", type=int, nargs="+", default=[3, 4], help="swim_coevo size classes")
    s.add_argument("--per-size", type=int, default=3)
    s.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    s.add_argument("--gens", type=int, default=150)
    s.add_argument("--lam", type=int, default=24)
    s.add_argument("--sigma", type=float, default=0.5)
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--log-every", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("render", parents=[common])
    r.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    sw.configure(args)
    {"search": run_search, "render": run_render}[args.cmd](args)


if __name__ == "__main__":
    main()
