"""One video of the optimised swimmers growing side by side in a shared pool,
each setting off as soon as its own topology stops changing.

All swimmers run in one simulation: each body keeps its own controller and
fluid, and nodes of different bodies push apart on contact, so swimmers
bump and slide off each other. A body that is still growing is pinned at
its rest shape and is an obstacle. Swimmers start on a ring, rotated so
their usual direction of travel points outwards. Growth is replayed from
each grower's seed; at the handoff the grown layout morphs into the settled
rest shape the swim starts from.

    uv run python swim_pool.py --tag speed
"""

import argparse
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import swim as sw
from swim_render import BG, GRID, TRAIL, CHUNK, _concat

import jax
import jax.numpy as jnp

MORPH = 20       # frames from grown layout to rest shape
K_CONTACT = 300.0 # contact stiffness between a node and another body's edges
R_CONTACT = 0.5   # half the rest edge length: edges are thick rods, so bodies can't pass through


def pick_swimmers(probs):
    """The best controller arm per seed swimmer."""
    best = {}
    for p in probs:
        k = (p["fluid"], p["grower"]["n"])
        if k not in best or p["opt_speed"] > best[k]["opt_speed"]:
            best[k] = p
    return sorted(best.values(), key=lambda p: -p["opt_speed"])


def replay_growth(grower, blocks):
    params = jax.tree.map(jnp.asarray, grower["params"])
    key = jax.random.PRNGKey(grower["seed"])
    state = jax.tree.map(lambda x: x[0], sw.init_states(key[None], [grower["graph"]]))
    _, _, last, (_, pos, active, gra, nbrs, deg) = jax.jit(
        lambda p, s, k: sw.grow(p, s, k, blocks, record=True))(params, state, key)
    return int(last), *map(np.asarray, (pos, active, gra, nbrs, deg))


def procrustes(src, dst):
    """Rotation R minimising |(src - mean) @ R - (dst - mean)|."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def rot(phi):
    return np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])


def _push(p, a, b):
    """Contact force on points p (M, 2) from rods a-b (E, 2): (force (M, E, 2), position t along rod)."""
    ab = b - a
    t = jnp.clip(((p[:, None] - a) * ab).sum(-1) / ((ab ** 2).sum(-1) + 1e-9), 0.0, 1.0)
    d = p[:, None] - (a + t[..., None] * ab)
    r = jnp.sqrt((d ** 2).sum(-1) + 1e-9)
    return (K_CONTACT * jnp.clip(R_CONTACT - r, 0.0) / r)[..., None] * d, t


def contact(pos, af, live, seg, obs):
    """(S, n, 2) push on swimming nodes from the edges of other swimming bodies
    and from obstacle rods (bodies still growing, drawn as they are shown).

    seg = (a, b, owner, valid): edge end indices into the flattened nodes. The
    reaction on an edge is shared between its ends by where the contact lies.
    """
    S, n, _ = pos.shape
    p = pos.reshape(-1, 2)
    owner = jnp.repeat(jnp.arange(S), n)
    live_n = live[owner] * af.reshape(-1)
    sa, sb, s_owner, s_valid = seg
    f, t = _push(p, p[sa], p[sb])
    f = f * (live_n[:, None] * live[s_owner] * s_valid * (owner[:, None] != s_owner))[..., None]
    out = f.sum(1)
    out = out.at[sa].add(-((1 - t)[..., None] * f).sum(0))
    out = out.at[sb].add(-(t[..., None] * f).sum(0))
    fo, _ = _push(p, obs[0], obs[1])
    out = out + (fo * (live_n[:, None] * obs[2])[..., None]).sum(1)
    # and obstacle nodes (rod ends) push on swimming edges
    q, qv = jnp.concatenate(obs[:2]), jnp.concatenate([obs[2], obs[2]])
    fq, t = _push(q, p[sa], p[sb])
    fq = fq * (qv[:, None] * live[s_owner] * s_valid)[..., None]
    out = out.at[sa].add(-((1 - t)[..., None] * fq).sum(0))
    out = out.at[sb].add(-(t[..., None] * fq).sum(0))
    return out.reshape(S, n, 2)


def segments(bodies):
    """Undirected edges of every body as indices into the flattened node buffer."""
    S, n = bodies.active.shape
    per = [sw._edges(np.asarray(bodies.nbrs[s]), np.asarray(bodies.deg[s]),
                     np.asarray(bodies.active[s])) for s in range(S)]
    E = max(len(e) for e in per)
    sa, sb, owner, valid = (np.zeros((S, E), np.int32) for _ in range(4))
    for s, e in enumerate(per):
        sa[s, :len(e)], sb[s, :len(e)] = s * n + e[:, 0], s * n + e[:, 2]
        owner[s], valid[s, :len(e)] = s, 1
    return tuple(jnp.asarray(v.reshape(-1)) for v in (sa, sb, owner, valid))


@partial(jax.jit, static_argnums=(7,))
def pool_sim(bodies, genomes, keys, drags, start, seg, obs, every):
    """Joint swim of S bodies; body s is pinned until step start[s].

    obs: per-frame obstacle rods (a, b, valid), each (frames, E, ...).
    Returns per-frame (pos, CA channels 0-2, edge activation), each (frames, S, ...).
    """
    prep = jax.vmap(sw.swim_prep)(bodies, genomes)
    step_all = jax.vmap(sw.swim_step)

    def frame(carry, xs):
        f, obs_f = xs

        def step(carry, t):
            pos, vel, x = carry
            live = (t >= start).astype(jnp.float32)
            new = step_all(prep, genomes, drags, pos, vel, x,
                           contact(pos, prep["af"], live, seg, obs_f))
            keep = live[:, None, None] > 0
            return tuple(jnp.where(keep, a, b) for a, b in zip(new[:3], carry)), new[3] * keep

        carry, act = jax.lax.scan(step, carry, f * every + jnp.arange(every))
        return carry, (carry[0], carry[2][..., :3], act[-1])

    carry = jax.vmap(sw.swim_init)(bodies, genomes, keys)
    frames = obs[0].shape[0]
    return jax.lax.scan(frame, carry, (jnp.arange(frames), obs))[1]


def obstacles(growth, starts, total):
    """Rods of every body still growing or morphing, per frame, padded."""
    per = [[np.concatenate([g[f]["seg"] for g, s0 in zip(growth, starts) if f < s0]
                           or [np.zeros((0, 2, 2))])] for f in range(total)]
    E = max(max(len(p[0]) for p in per), 1)
    seg = np.zeros((total, E, 2, 2), np.float32)
    valid = np.zeros((total, E), np.float32)
    for f, (p,) in enumerate(per):
        seg[f, :len(p)], valid[f, :len(p)] = p, 1
    return jnp.asarray(seg[:, :, 0]), jnp.asarray(seg[:, :, 1]), jnp.asarray(valid)


def placement(p, blocks, per_frame, radius, phi):
    """Growth replay, handoff frame and pool placement of one swimmer."""
    gr = p["grower"]
    last, gpos, gact, ggra, gnbrs, gdeg = replay_growth(gr, blocks)
    stop = last + 1
    comp, _ = sw.body_component(gact[stop - 1], gnbrs[stop - 1], gdeg[stop - 1])
    assert len(comp) == gr["n"], f"regrown body has {len(comp)} nodes, stored {gr['n']}"
    # heading from a solo swim: rotate so net travel points away from the middle
    com = sw._simulate(gr, p["genome"], 3000, drag=p["drag"])["com"]
    travel = com[-1] - com[0]
    R = rot(np.arctan2(travel[1], travel[0])) @ rot(-phi)
    off = radius * np.array([np.cos(phi), np.sin(phi)])
    rest = gr["body"]["pos"][gr["body"]["active"]]
    G = gpos[stop - 1][comp]
    Rg = procrustes(G, rest)
    return dict(p=p, stop=stop, g_frames=(stop + per_frame - 1) // per_frame, per_frame=per_frame,
                growth=(gpos, gact, ggra, gnbrs, gdeg), comp=comp, G=G,
                to_pool=lambda x: x @ R + off,
                from_grown=lambda x: (x - G.mean(0)) @ Rg + rest.mean(0))


def growth_frames(pl, x0, edges):
    """Drawables for one swimmer's growth and morph, in pool coordinates."""
    gpos, gact, ggra, gnbrs, gdeg = pl["growth"]
    to_pool, from_grown, stop = pl["to_pool"], pl["from_grown"], pl["stop"]
    out = []
    for f in range(pl["g_frames"]):
        b = min(f * pl["per_frame"], stop - 1)
        a = gact[b]
        e = sw._edges(gnbrs[b], gdeg[b], a)
        e = e[a[gnbrs[b][e[:, 0], e[:, 1]]]]
        q = to_pool(from_grown(gpos[b]))
        rgb = 0.25 + 0.75 * np.clip(ggra[b], 0, 1)
        out.append(dict(seg=np.stack([q[e[:, 0]], q[e[:, 2]]], 1),
                        seg_rgb=0.5 * (rgb[e[:, 0]] + rgb[e[:, 2]]), seg_w=np.full(len(e), 0.6),
                        pts=q[a], pts_rgb=rgb[a]))
    act = pl["p"]["grower"]["body"]["active"][:len(x0)]
    rest = pl["p"]["grower"]["body"]["pos"][:len(x0)][act]
    g_rgb = 0.25 + 0.75 * np.clip(ggra[stop - 1][pl["comp"]], 0, 1)
    for f in range(MORPH):
        w = 0.5 - 0.5 * np.cos(np.pi * f / (MORPH - 1))
        q = np.zeros((len(act), 2))
        q[act] = to_pool((1 - w) * from_grown(pl["G"]) + w * rest)
        rgb = np.zeros((len(act), 3))
        rgb[act] = (1 - w) * g_rgb + w * (0.25 + 0.75 * np.clip(x0[act], 0, 1))
        out.append(dict(seg=np.stack([q[edges[:, 0]], q[edges[:, 2]]], 1),
                        seg_rgb=0.5 * (rgb[edges[:, 0]] + rgb[edges[:, 2]]),
                        seg_w=np.full(len(edges), 0.6), pts=q[act], pts_rgb=rgb[act]))
    return out


def run(swimmers, blocks, args, total):
    S = len(swimmers)
    phis = 2 * np.pi * np.arange(S) / S + np.pi / 2
    pls = [placement(p, blocks, args.per_frame, args.radius, phi) for p, phi in zip(swimmers, phis)]
    size = max(pl["p"]["grower"]["n"] for pl in pls)   # contact is O(S^2 n^2), so no spare slots
    k = max(pl["p"]["genome"].b.shape[-1] for pl in pls)
    bodies = []
    for pl in pls:
        b = {f: v[:size] for f, v in pl["p"]["grower"]["body"].items()}
        b["pos"] = np.where(b["active"][:, None], pl["to_pool"](b["pos"]), 0.0).astype(np.float32)
        bodies.append(b)
    bodies = sw.Body(*[jnp.asarray(np.stack([b[f] for b in bodies])) for f in sw.Body._fields])
    genomes = jax.tree.map(lambda *v: jnp.asarray(np.stack(v)),
                           *[sw.upgrade_genome(pl["p"]["genome"], k) for pl in pls])
    keys = jax.random.split(jax.random.PRNGKey(0), S)
    drags = jnp.asarray([pl["p"]["drag"] for pl in pls], jnp.float32)
    starts = np.array([pl["g_frames"] + MORPH for pl in pls])
    for pl, s0 in zip(pls, starts):
        print(f"{pl['p']['fluid']:12s} n={pl['p']['grower']['n']:<3d} sets off at {s0 / args.fps:.1f}s",
              flush=True)
    x0 = np.asarray(jax.vmap(sw.swim_init)(bodies, genomes, keys)[2])
    edges = [sw._edges(np.asarray(bodies.nbrs[s]), np.asarray(bodies.deg[s]),
                       np.asarray(bodies.active[s])) for s in range(S)]
    growth = [growth_frames(pl, x0[s, :, :3], edges[s]) for s, pl in enumerate(pls)]
    print(f"pool sim: {S} bodies x {size} nodes, {total * args.every} steps", flush=True)
    pos, x, act = map(np.asarray, pool_sim(bodies, genomes, keys, drags,
                                           jnp.asarray(starts * args.every), segments(bodies),
                                           obstacles(growth, starts, total), args.every))

    per = []
    for s, pl in enumerate(pls):
        active = np.asarray(bodies.active[s])
        fr, e = growth[s][:starts[s]], edges[s]
        com = (pos[:, s] * active[:, None]).sum(1) / active.sum()
        for f in range(starts[s], total):
            q = pos[f, s]
            rgb = 0.25 + 0.75 * np.clip(x[f, s], 0, 1)
            a = act[f, s][e[:, 0], e[:, 1]]
            fr.append(dict(seg=np.stack([q[e[:, 0]], q[e[:, 2]]], 1),
                           seg_rgb=0.5 * (rgb[e[:, 0]] + rgb[e[:, 2]]),
                           seg_w=0.5 + 2.5 * np.clip(-a, 0, 1), pts=q[active], pts_rgb=rgb[active],
                           trail=com[starts[s]:f + 1]))
        per.append(fr[:total])
    return list(zip(*per))


def camera(frames, aspect, min_half=14.0, pad=4.0, smooth=31):
    """Per-frame (cx, cy, half-height) framing every node, smoothed both ways."""
    box = []
    for fr in frames:
        pts = np.concatenate([s["pts"] for s in fr])
        lo, hi = pts.min(0) - pad, pts.max(0) + pad
        box.append([*(lo + hi) / 2, max((hi - lo)[1] / 2, (hi - lo)[0] / 2 / aspect, min_half)])
    box = np.array(box)
    # never zoom back in once the pool has spread out
    box[:, 2] = np.maximum.accumulate(box[:, 2])
    k = np.ones(smooth) / smooth
    padded = np.pad(box, ((smooth // 2, smooth // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, j], k, "valid") for j in range(3)], 1)


def _render_chunk(job):
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    frames, cams, size, fps, path = job
    W, H = size
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")
    grid = LineCollection([], colors=GRID, linewidths=0.6, zorder=0)
    ax.add_collection(grid)
    views = []
    for _ in frames[0]:
        e = LineCollection([], zorder=2)
        ax.add_collection(e)
        (tr,) = ax.plot([], [], color=TRAIL, lw=1.0, alpha=0.55, zorder=1)
        views.append((e, tr, ax.scatter([], [], linewidths=0, zorder=3)))
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    for fr, (cx, cy, h) in zip(frames, cams):
        w = h * W / H
        ax.set_xlim(cx - w, cx + w)
        ax.set_ylim(cy - h, cy + h)
        zoom = 14.0 / h   # marks shrink as the camera pulls back
        gs = 10.0
        xs = np.arange(np.floor((cx - w) / gs) * gs, cx + w, gs)
        ys = np.arange(np.floor((cy - h) / gs) * gs, cy + h, gs)
        grid.set_segments([((x, cy - h), (x, cy + h)) for x in xs]
                          + [((cx - w, y), (cx + w, y)) for y in ys])
        for s, (e, tr, nd) in zip(fr, views):
            e.set_segments(s["seg"])
            e.set_color(s["seg_rgb"])
            e.set_linewidth(s["seg_w"] * 1.6 * zoom)
            nd.set_offsets(s["pts"])
            nd.set_facecolor(s["pts_rgb"])
            nd.set_sizes(np.full(len(s["pts"]), 22 * zoom ** 2))
            trail = s.get("trail")
            tr.set_data(*(trail.T if trail is not None else ([], [])))
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(parents=[sw.common_args()])
    ap.add_argument("--tag", default="speed")
    ap.add_argument("--seconds", type=float, default=36.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--per-frame", type=int, default=1, help="growth blocks per frame")
    ap.add_argument("--every", type=int, default=6, help="swim steps per frame")
    ap.add_argument("--radius", type=float, default=18.0)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--limit", type=int, default=0, help="fewer swimmers, for quick tests")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sw.configure(args)

    with open(sw.OUT / f"opt_{args.tag}.pkl", "rb") as f:
        swimmers = pick_swimmers(pickle.load(f))[:args.limit or None]
    with open(sw.OUT / "growers.pkl", "rb") as f:
        blocks = pickle.load(f)["blocks"]
    total = int(args.seconds * args.fps)
    frames = run(swimmers, blocks, args, total)
    W, H = args.size
    cams = camera(frames, W / H)

    out = Path(args.out) if args.out else sw.OUT / "videos" / f"pool_{args.tag}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="swim_pool_"))
    jobs = [(frames[c:c + CHUNK], cams[c:c + CHUNK], (W, H), args.fps,
             str(tmp / f"{c // CHUNK:04d}.mp4")) for c in range(0, total, CHUNK)]
    print(f"rendering {total} frames in {len(jobs)} chunks", flush=True)
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as ex:
        parts = list(ex.map(_render_chunk, jobs))
    _concat(parts, out)
    tmp.rmdir()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
