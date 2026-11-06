"""Trial: refine CMA-ES swimmers with gradients through the simulation.

The objective is net centre-of-mass displacement over the last H steps of a
run, with gradients cut H steps before the end (truncated BPTT), averaged over
several CA initial states. Reports, per horizon H, how large and how
consistent the gradients are, then runs Adam from the CMA-ES optimum and
tracks the true 2000-step speed on held-out initial states.

    uv run python swim_grad.py --tag speed
"""

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import swim as sw
import swim_opt as so
from swim_qd import describe

T0 = 400  # transient, as in the search's scoring


def objective_fn(body, drag, layout, base, scale, mask, H):
    """u (free coordinates, scaled) -> mean displacement per 1000 steps over the last H steps."""
    def one(u, key):
        x = base.at[mask].add(u * scale[mask])
        g = so.clip(so._unflatten(x, layout))
        trace, _, _ = sw.swim(body, g, key, T0 + H, drag=drag, truncate_at=T0)
        d = trace[-1, :2] - trace[T0, :2]
        return jnp.sqrt((d ** 2).sum() + 1e-9) / (H / 1000.0)
    return lambda u, keys: jax.vmap(one, (None, 0))(u, keys).mean()


def true_speed_fn(body, drag, layout, base, scale, mask, steps=2000):
    def one(u, key):
        x = base.at[mask].add(u * scale[mask])
        trace, _, pos = sw.swim(body, so.clip(so._unflatten(x, layout)), key, steps, drag=drag)
        rel0 = body.pos * body.active[:, None]
        return describe(trace, pos, body.active, jnp.sqrt((rel0 ** 2).sum(-1)).max())[0]
    return jax.jit(lambda u, keys: jax.vmap(one, (None, 0))(u, keys))


def adam(grad_fn, u, keys_fn, steps, lr, on_step):
    m, v = jnp.zeros_like(u), jnp.zeros_like(u)
    for i in range(1, steps + 1):
        g = -grad_fn(u, keys_fn(i))                     # maximise
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        u = u - lr * (m / (1 - 0.9 ** i)) / (jnp.sqrt(v / (1 - 0.999 ** i)) + 1e-8)
        on_step(i, u)
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="speed")
    ap.add_argument("--horizons", type=int, nargs="+", default=[50, 100, 200, 400, 800, 1600])
    ap.add_argument("--adam-h", type=int, nargs="+", default=[100, 400])
    ap.add_argument("--adam-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--keys", type=int, default=8)
    ap.add_argument("--problems", type=int, default=0, help="limit for quick tests")
    args = ap.parse_args()
    if args.out_dir:
        sw.OUT = sw.Path(args.out_dir)

    with open(sw.OUT / f"opt_{args.tag}.pkl", "rb") as f:
        probs = pickle.load(f)
    # per seed swimmer, the arm CMA-ES did best with
    best = {}
    for p in probs:
        k = (p["fluid"], p["grower"]["n"])
        p["opt_speed"] = p.get("opt_speed", max(p["hist"]))  # mid-run snapshots lack it
        if k not in best or p["opt_speed"] > best[k]["opt_speed"]:
            best[k] = p
    probs = list(best.values())[:args.problems or None]
    held = jax.random.split(jax.random.PRNGKey(999), args.keys)
    results = []
    for p in probs:
        size = next(b for b in sw.BUCKETS if p["grower"]["body"]["active"].sum() <= b)
        body = sw.Body(**{k: jnp.asarray(v[:size]) for k, v in p["grower"]["body"].items()})
        drag = jnp.asarray(p["drag"], jnp.float32)
        base = jnp.asarray(so._flatten(p["genome"]), jnp.float32)
        layout = so._layout(p["genome"])
        scale = jnp.asarray(so.scale_vector(layout), jnp.float32)
        mask = np.flatnonzero(so.free_mask(layout, p["arm"]))
        u0 = jnp.zeros(len(mask))
        speed = true_speed_fn(body, drag, layout, base, scale, mask)
        s0 = float(speed(u0, held).mean())
        name = f"{p['fluid']} n={p['grower']['n']} {p['arm']}"
        print(f"\n== {name}  CMA-ES speed {s0:.2f} (held-out keys)", flush=True)

        # 1. gradient diagnostics per horizon
        diag = {}
        kd = jax.random.split(jax.random.PRNGKey(7), 2 * args.keys)
        for H in args.horizons:
            obj = objective_fn(body, drag, layout, base, scale, mask, H)
            vg = jax.jit(jax.value_and_grad(obj))
            t = time.time()
            f_a, g_a = vg(u0, kd[:args.keys])
            f_b, g_b = vg(u0, kd[args.keys:])
            cos = float(g_a @ g_b / (jnp.linalg.norm(g_a) * jnp.linalg.norm(g_b) + 1e-12))
            # does a small step along the gradient raise the objective as predicted?
            eps = 0.02
            d = g_a / (jnp.linalg.norm(g_a) + 1e-12)
            fd = float((jax.jit(obj)(u0 + eps * d, kd[:args.keys]) - f_a) / eps)
            diag[H] = dict(norm=float(jnp.linalg.norm(g_a)), cos=cos, fd=fd, secs=time.time() - t)
            print(f"  H={H:5d}  |grad| {diag[H]['norm']:10.3g}  cos(key sets) {cos:+.2f}  "
                  f"finite-diff slope {fd:+.3g} vs grad {diag[H]['norm']:.3g}  ({diag[H]['secs']:.0f}s)",
                  flush=True)

        # 2. Adam from the CMA-ES optimum
        runs = {}
        for H in args.adam_h:
            obj = objective_fn(body, drag, layout, base, scale, mask, H)
            grad = jax.jit(jax.grad(obj))
            curve = [(0, s0)]

            def log(i, u):
                if i % 10 == 0:
                    curve.append((i, float(speed(u, held).mean())))
            ks = lambda i: jax.random.split(jax.random.PRNGKey(100 + i), args.keys)
            t = time.time()
            u = adam(grad, u0, ks, args.adam_steps, args.lr, log)
            runs[H] = dict(curve=curve, u=np.asarray(u))
            print(f"  Adam H={H:4d}: true speed " + " -> ".join(f"{s:.2f}" for _, s in curve)
                  + f"  ({time.time() - t:.0f}s)", flush=True)
        results.append(dict(name=name, cma_speed=s0, diag=diag, adam=runs))

    with open(sw.OUT / f"grad_{args.tag}.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nsummary: change in true speed after Adam (held-out keys)")
    for r in results:
        print(f"  {r['name']:28s} CMA {r['cma_speed']:5.2f}  " + "  ".join(
            f"H={H}: {run['curve'][-1][1]:5.2f} (best {max(s for _, s in run['curve']):5.2f})"
            for H, run in r["adam"].items()))


if __name__ == "__main__":
    main()
