"""Grown swimmers: GRA bodies that grow to a steady topology, then swim.

grow:  MAP-Elites over GRA params for bodies whose topology stops changing on
       its own. Node positions are relaxed with springs + repulsion as they grow.
swim:  topology and GRA states are frozen. A K-channel continuous CA runs on the
       nodes, with channels mixing only across edges, and sets each edge's
       rest length. The body moves through a drag fluid; fitness is net
       centre-of-mass travel.
render: mp4s of growth and swimming.

Runs on GPU or CPU. On CPU every core is exposed as a JAX device and batches
are sharded across them with pmap.

    uv run python swim.py grow --gens 30
    uv run python swim.py swim --gens 60
    uv run python swim.py render
    uv run python swim.py swim --drag 0.5 1.0 0.0 --out-dir outputs/swim_stokes
"""

import argparse
import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

# XLA's CPU backend runs a vmapped batch on only a couple of threads, so expose
# each core as a device for pmap. Must be set before jax is imported; it does
# nothing on GPU.
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={os.cpu_count()}")

import jax
import jax.numpy as jnp
import numpy as np

import sim
from swim_render import BG, render_videos

N = 256
sim.MAX_NODES = N
D = sim.MAX_DEGREE
TOPO_EVERY = sim.TOPO_EVERY

OUT = Path(__file__).parent / "outputs" / "swim"
ON_CPU = jax.default_backend() == "cpu"


def batched(fn):
    """vmap fn over a leading batch axis: jit on GPU, sharded across cores on CPU."""
    if not ON_CPU:
        return jax.jit(jax.vmap(fn))
    P = jax.local_device_count()
    f = jax.pmap(jax.vmap(fn))

    def call(*args):
        B = jax.tree.leaves(args)[0].shape[0]
        pad = (-B) % P
        if pad:
            args = jax.tree.map(lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, 0)]), args)
        args = jax.tree.map(lambda x: x.reshape((P, -1) + x.shape[1:]), args)
        return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:])[:B], f(*args))
    return call


# ── Growth with layout ───────────────────────────────────────────────────────

def _gather(nbrs, deg, active):
    safe = jnp.clip(nbrs, 0, nbrs.shape[0] - 1)
    valid = ((jnp.arange(D)[None] < deg[:, None]) & (nbrs >= 0)
             & active[safe] & active[:, None])
    return safe, valid


def relax(pos, nbrs, deg, active, dt=0.05, k=2.0, charge=0.3):
    """Overdamped spring + long-range repulsion step (unit rest length)."""
    safe, valid = _gather(nbrs, deg, active)
    d = pos[safe] - pos[:, None]
    dist = jnp.sqrt((d ** 2).sum(-1) + 1e-6)
    f = (k * valid * (dist - 1.0) / dist)[..., None] * d
    f = f.sum(1)
    diff = pos[:, None] - pos[None]
    r2 = (diff ** 2).sum(-1)
    pair = active[:, None] & active[None]
    f = f + charge * ((pair / (1.0 + r2))[..., None] * diff).sum(1)
    step = dt * f
    norm = jnp.sqrt((step ** 2).sum(-1, keepdims=True) + 1e-12)
    step = step * jnp.minimum(1.0, 0.2 / norm)
    return jnp.where(active[:, None], pos + step, pos)


def place_new(pos, prev_active, state, key):
    """New nodes start at the mean of their already-placed neighbours."""
    active = state.node_active
    new = active & ~prev_active
    safe, valid = _gather(state.neighbors, state.num_neighbors, active)
    known = prev_active & active
    for _ in range(3):
        w = valid & known[safe]
        cnt = w.sum(1)
        mean = (pos[safe] * w[..., None]).sum(1) / jnp.maximum(cnt, 1)[:, None]
        upd = new & ~known & (cnt > 0)
        pos = jnp.where(upd[:, None], mean, pos)
        known = known | upd
    jitter = 0.3 * jax.random.normal(key, pos.shape)
    return jnp.where(new[:, None], pos + jitter, pos)


def grow(params, state, key, num_blocks, record=False):
    """Run num_blocks * TOPO_EVERY GRA steps with one topology op per block."""
    pos = jax.random.normal(key, (N, 2)) * 2.0

    def inner(carry, _):
        state, pos = carry
        state = sim.apply_growth(state, params)
        pos = relax(pos, state.neighbors, state.num_neighbors, state.node_active)
        return (state, pos), None

    def block(carry, b):
        state, pos, last_change, key = carry
        (state, pos), _ = jax.lax.scan(inner, (state, pos), None, length=TOPO_EVERY)
        prev_active, prev_nb = state.node_active, state.neighbors
        state = sim.apply_collapses(sim.apply_divisions(state, params), params)
        key, k = jax.random.split(key)
        pos = place_new(pos, prev_active, state, k)
        changed = (jnp.any(state.neighbors != prev_nb)
                   | jnp.any(state.node_active != prev_active))
        last_change = jnp.where(changed, b, last_change)
        out = (state.num_active, pos, state.node_active, state.node_states,
               state.neighbors, state.num_neighbors) if record \
            else state.num_active
        return (state, pos, last_change, key), out

    init = (state, pos, jnp.array(0), key)
    (state, pos, last_change, _), trace = jax.lax.scan(
        block, init, jnp.arange(num_blocks))
    return state, pos, last_change, trace


def init_states(keys, graph_idx):
    graphs = sim.load_initial_graphs()
    states = []
    for k, g in zip(keys, graph_idx):
        states.append(sim.make_init_state(k, int(g) % len(graphs)))
    return jax.tree.map(lambda *x: jnp.stack(x), *states)


def mutate_params(params, key, scale=0.1):
    leaves, tree = jax.tree.flatten(params)
    ks = jax.random.split(key, len(leaves))
    leaves = [v + jax.random.normal(k, v.shape) * scale * jnp.maximum(jnp.abs(v).mean(), 0.01)
              for v, k in zip(leaves, ks)]
    p = jax.tree.unflatten(tree, leaves)
    return p._replace(
        kernel_mu=jnp.clip(p.kernel_mu, 0.0, 18.0),
        kernel_sigma=jnp.clip(p.kernel_sigma, 0.1, 4.0),
        growth_mu=jnp.clip(p.growth_mu, 0.0, 1.0),
        growth_sigma=jnp.clip(p.growth_sigma, 0.05, 1.0),
        state_dt=jnp.clip(p.state_dt, 0.001, 0.2),
        div_mu=jnp.clip(p.div_mu, 0.0, 1.0),
        div_sigma=jnp.clip(p.div_sigma, 0.05, 1.0),
        div_threshold=jnp.clip(p.div_threshold, 0.3, 0.99),
        div_prob=jnp.clip(p.div_prob, 0.0001, 0.05),
        death_mu=jnp.clip(p.death_mu, 0.0, 1.0),
        death_sigma=jnp.clip(p.death_sigma, 0.05, 1.0),
        death_threshold=jnp.clip(p.death_threshold, 0.3, 0.99),
        death_prob=jnp.clip(p.death_prob, 0.0001, 0.05),
    )


# ── Bodies: host-side cleanup of a grown graph ───────────────────────────────

class Body(NamedTuple):
    pos: np.ndarray     # (N, 2)
    active: np.ndarray  # (N,) bool
    nbrs: np.ndarray    # (N, D) int32, -1 empty
    deg: np.ndarray     # (N,) int32
    gra: np.ndarray     # (N, 3) frozen GRA channel states


def extract_body(pos, active, nbrs, deg, gra):
    """Largest connected component, mutual edges only, reindexed from 0.

    Edges must be mutual or spring forces don't cancel and the body can push
    itself along through its own bookkeeping.
    """
    import networkx as nx
    g = nx.Graph()
    idx = np.nonzero(active)[0]
    g.add_nodes_from(idx)
    for i in idx:
        for j in nbrs[i, :deg[i]]:
            if j >= 0 and j != i and active[j] and i in nbrs[j, :deg[j]]:
                g.add_edge(int(i), int(j))
    if g.number_of_nodes() == 0:
        return None
    comp = sorted(max(nx.connected_components(g), key=len))
    remap = {old: new for new, old in enumerate(comp)}
    n = len(comp)
    b_nbrs = np.full((N, D), -1, np.int32)
    b_deg = np.zeros(N, np.int32)
    for old in comp:
        i = remap[old]
        for j in g.neighbors(old):
            b_nbrs[i, b_deg[i]] = remap[j]
            b_deg[i] += 1
    b_pos = np.zeros((N, 2), np.float32)
    b_pos[:n] = pos[comp] - pos[comp].mean(0)
    b_gra = np.zeros((N, 3), np.float32)
    b_gra[:n] = gra[comp]
    return Body(b_pos, np.arange(N) < n, b_nbrs, b_deg, b_gra)


# Swim cost is dominated by O(n^2) repulsion over the node buffer, so sims run
# on a buffer trimmed to the smallest bucket that holds their body.
BUCKETS = (32, 64, 128, N)


def run_bucketed(fn, bodies, *args):
    """Apply fn(body, *args) per sim, grouping sims by body size.

    bodies is a stacked Body of numpy arrays with buffer N; args are batched
    pytrees. fn must return fixed-shape outputs (independent of buffer size).
    """
    n = bodies.active.sum(1)
    out = None
    lo = 0
    for size in BUCKETS:
        idx = np.nonzero((n > lo) & (n <= size))[0]
        lo = size
        if len(idx) == 0:
            continue
        sub = Body(*(jnp.asarray(x[idx][:, :size]) for x in bodies))
        sub_args = jax.tree.map(lambda x: jnp.asarray(x)[idx], args)
        res = jax.device_get(_batched_cache(fn)(sub, *sub_args))
        if out is None:
            out = jax.tree.map(lambda r: np.zeros((len(n),) + r.shape[1:], r.dtype), res)
        jax.tree.map(lambda o, r: o.__setitem__(idx, r), out, res)
    return out


_BATCHED = {}


def _batched_cache(fn):
    if fn not in _BATCHED:
        _BATCHED[fn] = batched(fn)
    return _BATCHED[fn]


def body_shape(body):
    p = body.pos[body.active]
    ev = np.sort(np.linalg.eigvalsh(np.cov(p.T)))
    aspect = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    return int(body.active.sum()), aspect


# ── Stage 1: grower search ───────────────────────────────────────────────────

N_BINS = np.geomspace(16, N, 11)
ASPECT_BINS = np.geomspace(1.0, 6.0, 9)


def cell_of(n, aspect):
    i = int(np.clip(np.searchsorted(N_BINS, n) - 1, 0, len(N_BINS) - 2))
    j = int(np.clip(np.searchsorted(ASPECT_BINS, aspect) - 1, 0, len(ASPECT_BINS) - 2))
    return i, j


def run_grow(args):
    OUT.mkdir(parents=True, exist_ok=True)
    B, blocks = args.batch, args.blocks
    steady_blocks = args.steady_blocks
    rng = jax.random.PRNGKey(args.seed)
    grow_b = batched(lambda p, s, k: grow(p, s, k, blocks)[:3])

    archive = {}  # cell -> dict(fitness, params, seed, graph, body)
    t0 = time.time()
    for gen in range(args.gens):
        rng, kp, ks, kg, km = jax.random.split(rng, 5)
        elites = list(archive.values())
        if gen < 2 or not elites:
            params = jax.vmap(sim.sample_params)(jax.random.split(kp, B))
            seeds = np.array(jax.random.randint(ks, (B,), 0, 2**30))
            graph = np.array(jax.random.randint(kg, (B,), 0, 14))
        else:
            pick = np.random.default_rng(gen).integers(0, len(elites), B)
            parents = jax.tree.map(lambda *x: jnp.stack(x), *[elites[i]["params"] for i in pick])
            params = jax.vmap(mutate_params)(parents, jax.random.split(km, B))
            seeds = np.array([elites[i]["seed"] for i in pick])
            graph = np.array([elites[i]["graph"] for i in pick])
            reseed = np.random.default_rng(gen + 1).random(B) < 0.3
            seeds = np.where(reseed, np.array(jax.random.randint(ks, (B,), 0, 2**30)), seeds)
        keys = jax.vmap(jax.random.PRNGKey)(jnp.array(seeds))
        states = init_states(keys, graph)
        state, pos, last_change = grow_b(params, states, keys)
        state, pos, last_change = jax.device_get((state, pos, last_change))

        n_new = 0
        for b in range(B):
            n_raw = int(state.num_active[b])
            if n_raw < 16 or n_raw > N - 8:
                continue
            body = extract_body(pos[b], state.node_active[b], state.neighbors[b],
                                state.num_neighbors[b], state.node_states[b])
            if body is None or body.active.sum() < 16:
                continue
            n, aspect = body_shape(body)
            steady = blocks - 1 - int(last_change[b])
            fitness = min(steady / steady_blocks, 1.0) + 0.001 * n
            cell = cell_of(n, aspect)
            if cell not in archive or archive[cell]["fitness"] < fitness:
                archive[cell] = dict(
                    fitness=fitness, steady=steady >= steady_blocks,
                    params=jax.tree.map(lambda x: np.asarray(x[b]), params),
                    seed=int(seeds[b]), graph=int(graph[b]), body=body._asdict(), n=n, aspect=aspect)
                n_new += 1
        n_steady = sum(e["steady"] for e in archive.values())
        print(f"gen {gen:3d}  cells {len(archive):3d}  steady {n_steady:3d}  "
              f"new {n_new:3d}  {time.time() - t0:6.0f}s", flush=True)
        with open(OUT / "growers.pkl", "wb") as f:
            pickle.dump(dict(archive=archive, blocks=blocks), f)


# ── Stage 2: a continuous CA drives the muscles ──────────────────────────────

K = 4  # channels of the muscle CA


class Genome(NamedTuple):
    A: jnp.ndarray       # (K, K) neighbour channel -> channel (the only cross-channel path)
    a_self: jnp.ndarray  # (K,)   per-channel self feedback within a node
    U: jnp.ndarray       # (K, 3) frozen GRA channels -> bias
    b: jnp.ndarray       # (K,)
    log_tau: jnp.ndarray # (K,)
    r: jnp.ndarray       # (K,)   channels -> muscle drive
    r0: jnp.ndarray      # ()
    amp: jnp.ndarray     # ()     max fractional change of rest length


def sample_genome(key):
    k = jax.random.split(key, 8)
    u = lambda kk, shape, lo, hi: lo + (hi - lo) * jax.random.uniform(kk, shape)
    return Genome(
        A=u(k[0], (K, K), -12.0, 12.0),
        a_self=u(k[1], (K,), -6.0, 6.0),
        U=u(k[2], (K, 3), -6.0, 6.0),
        b=u(k[3], (K,), -6.0, 3.0),
        log_tau=u(k[4], (K,), -1.0, 1.0),
        r=u(k[5], (K,), -1.0, 1.0),
        r0=u(k[6], (), -1.0, 1.0),
        amp=u(k[7], (), 0.1, 0.4),
    )


def mutate_genome(g, key, scale=0.15):
    leaves, tree = jax.tree.flatten(g)
    ks = jax.random.split(key, len(leaves))
    leaves = [v + jax.random.normal(k, v.shape) * scale * jnp.maximum(jnp.abs(v).mean(), 0.3)
              for v, k in zip(leaves, ks)]
    g = jax.tree.unflatten(tree, leaves)
    return g._replace(log_tau=jnp.clip(g.log_tau, -1.5, 1.5), amp=jnp.clip(g.amp, 0.02, 0.45))


DT = 0.02
K_SPRING = 30.0
C_INT = 1.0      # internal damping along springs
# growth forces scaled up together, so the grown shape stays the rest shape
CHARGE = K_SPRING * 0.3 / 2.0
C_T, C_N, C_Q = 0.1, 1.0, 1.0  # tangential, normal linear, normal quadratic drag


def set_drag(drag):
    """Set the fluid before anything is traced; swim() reads these at trace time."""
    global C_T, C_N, C_Q
    C_T, C_N, C_Q = (float(v) for v in drag)


def swim(body, g, key, steps, record_every=0):
    """Integrate the muscle CA + mass-spring body.

    Returns (trace, recordings, final pos); trace is (steps, 3) of centre of
    mass and body orientation (best-fit rotation from the rest shape). The
    buffer size comes from the body arrays, so a trimmed body runs cheaper.

    Every node runs the same rate rule. Channels only mix across edges, so a
    lone node relaxes to a fixed point and any rhythm needs a loop through
    neighbours.
    """
    active = body.active
    af = active.astype(jnp.float32)
    safe, valid = _gather(body.nbrs, body.deg, active)
    vf = valid.astype(jnp.float32)
    L0 = 1.0
    bias = body.gra @ g.U.T + g.b            # (N, K)
    tau = jnp.exp(g.log_tau)
    deg = jnp.maximum(vf.sum(1), 1.0)
    pair = (active[:, None] & active[None]).astype(jnp.float32)

    def repulsion(p):
        diff = p[:, None] - p[None]
        return CHARGE * ((pair / (1.0 + (diff ** 2).sum(-1)))[..., None] * diff).sum(1)


    def step(carry, t):
        pos, vel, x = carry
        nb = (x[safe] * vf[..., None]).sum(1) / deg[:, None]
        h = nb @ g.A.T + g.a_self * x + bias
        x = x + DT * (-x + jax.nn.sigmoid(h)) / tau
        m = jnp.tanh(4.0 * (x - 0.5) @ g.r + g.r0)
        act = 0.5 * (m[:, None] + m[safe])
        rest = L0 * (1.0 + g.amp * act)

        d = pos[safe] - pos[:, None]
        dist = jnp.sqrt((d ** 2).sum(-1) + 1e-6)
        u = d / dist[..., None]
        dv = vel[safe] - vel[:, None]
        f_mag = K_SPRING * (dist - rest) + C_INT * (dv * u).sum(-1)
        f = ((vf * f_mag)[..., None] * u).sum(1) + repulsion(pos)

        # resistive drag on each node from its incident edge segments
        vt = (vel[:, None] * u).sum(-1, keepdims=True) * u
        vn = vel[:, None] - vt
        vn_mag = jnp.sqrt((vn ** 2).sum(-1, keepdims=True) + 1e-9)
        drag = -(0.5 * dist * vf)[..., None] * (C_T * vt + C_N * vn + C_Q * vn_mag * vn)
        f = f + drag.sum(1)

        vel = (vel + DT * f) * af[:, None]
        pos = pos + DT * vel
        com = (pos * af[:, None]).sum(0) / af.sum()
        rel = (pos - com) * af[:, None]
        theta = jnp.arctan2((rest0[:, 0] * rel[:, 1] - rest0[:, 1] * rel[:, 0]).sum(),
                            (rest0 * rel).sum())
        rec = (pos, x[:, :3], act) if record_every else None
        return (pos, vel, x), (jnp.concatenate([com, theta[None]]), rec)

    n = body.pos.shape[0]
    rest0 = jnp.asarray(body.pos) * af[:, None]
    x0 = jax.random.uniform(key, (N, K))[:n]  # same init whatever the buffer size
    init = (jnp.asarray(body.pos), jnp.zeros((n, 2)), x0)
    (pos, _, _), (com, rec) = jax.lax.scan(step, init, jnp.arange(steps))
    return com, rec, pos


def swim_score(com):
    """Distance travelled after the transient, penalised for blowups."""
    T = com.shape[0]
    com = com[:, :2]
    start = com[T // 5]
    dist = jnp.sqrt(((com[-1] - start) ** 2).sum())
    ok = jnp.all(jnp.isfinite(com))
    return jnp.where(ok, dist, -1.0)


def settle(growers, steps=4000):
    """Relax each body under swim physics with muscles off; that is its rest shape."""
    bodies = stack_bodies([g["body"] for g in growers])
    off = jax.vmap(sample_genome)(jax.random.split(jax.random.PRNGKey(0), len(growers)))
    off = off._replace(amp=jnp.zeros(len(growers)))
    keys = jax.random.split(jax.random.PRNGKey(0), len(growers))
    pos = run_bucketed(_settle_fn(steps), bodies, off, keys)
    for g, p in zip(growers, pos):
        a = g["body"]["active"]
        g["body"] = dict(g["body"], pos=np.where(a[:, None], p - p[a].mean(0), 0.0).astype(np.float32))
    return growers


_SETTLE = {}


def _settle_fn(steps):
    if steps not in _SETTLE:
        def fn(b, g, k):
            pos = swim(b, g, k, steps)[2]
            return jnp.pad(pos, ((0, N - pos.shape[0]), (0, 0)))
        _SETTLE[steps] = fn
    return _SETTLE[steps]


def stack_bodies(bodies):
    """Stack body dicts into a Body of numpy arrays (buffer N)."""
    return Body(*[np.stack([b[f] for b in bodies]) for f in Body._fields])


def load_growers(path, min_n=16):
    with open(path, "rb") as f:
        arch = pickle.load(f)["archive"]
    return [e for e in arch.values() if e["steady"] and e["n"] >= min_n]


def run_swim(args):
    growers = load_growers(OUT / args.growers)
    growers.sort(key=lambda e: e["n"])
    if args.max_bodies and len(growers) > args.max_bodies:
        sel = np.linspace(0, len(growers) - 1, args.max_bodies).round().astype(int)
        growers = [growers[i] for i in sel]
    growers = settle(growers)
    nb = len(growers)
    per = max(args.batch // nb, 2)
    B = nb * per
    print(f"{nb} bodies x {per} genomes = {B} sims, {args.steps} steps each", flush=True)
    bodies = stack_bodies([growers[i // per]["body"] for i in range(B)])

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    genomes = jax.vmap(sample_genome)(jax.random.split(k, B))

    def score(b, g, k):
        return swim_score(swim(b, g, k, args.steps)[0])

    def evaluate(bodies, genomes, key):
        return run_bucketed(score, bodies, genomes, jax.random.split(key, B))

    best = [None] * nb  # (score, genome)
    t0 = time.time()
    for gen in range(args.gens):
        rng, ke, km = jax.random.split(rng, 3)
        scores = np.asarray(evaluate(bodies, genomes, ke))
        gnp = jax.device_get(genomes)
        for i in range(nb):
            sl = slice(i * per, (i + 1) * per)
            j = i * per + int(np.argmax(scores[sl]))
            if best[i] is None or scores[j] > best[i][0]:
                best[i] = (float(scores[j]), jax.tree.map(lambda x: x[j], gnp))
        # next population: per body, mutate from the top quarter (elitist);
        # slot 0 of each body is replaced by its best-ever genome below
        top = max(per // 4, 1)
        order = np.argsort(-scores.reshape(nb, per), 1)[:, :top] + np.arange(nb)[:, None] * per
        src = order[:, np.arange(per) % top].ravel()
        genomes = jax.tree.map(lambda x: x[src], gnp)
        bests = jax.tree.map(lambda *x: np.stack(x), *[b[1] for b in best])

        def put_best(g, b):
            g[::per] = b
            return g
        genomes = jax.tree.map(put_best, genomes, bests)
        kk = jax.random.split(km, B)
        mutated = jax.vmap(mutate_genome)(genomes, kk)
        keep = jnp.arange(B) % per == 0
        genomes = jax.tree.map(
            lambda a, b: jnp.where(keep.reshape((-1,) + (1,) * (a.ndim - 1)), a, b),
            genomes, mutated)
        top = sorted(b[0] for b in best)[::-1][:5]
        print(f"gen {gen:3d}  best {top[0]:7.2f}  top5 {np.round(top, 1)}  "
              f"median {np.median([b[0] for b in best]):6.2f}  {time.time() - t0:6.0f}s",
              flush=True)
        with open(OUT / args.out, "wb") as f:
            pickle.dump(dict(growers=growers, best=best, steps=args.steps,
                             drag=(C_T, C_N, C_Q)), f)


# ── Rendering ────────────────────────────────────────────────────────────────

def _edges(body_nbrs, body_deg, active):
    e = []
    for i in np.nonzero(active)[0]:
        for s in range(body_deg[i]):
            j = body_nbrs[i, s]
            if j > i:
                e.append((i, s, j))
    return np.array(e, dtype=int).reshape(-1, 3)


@partial(jax.jit, static_argnums=3)
def _record(body, g, key, steps):
    return swim(body, g, key, steps, 1)


def _simulate(grower, genome, steps, key=None):
    """Re-run one swimmer with per-step recording, on its bucket-sized buffer."""
    size = next(b for b in BUCKETS if grower["body"]["active"].sum() <= b)
    body = Body(**{k: jnp.asarray(v[:size]) for k, v in grower["body"].items()})
    g = jax.tree.map(jnp.asarray, genome)
    key = jax.random.PRNGKey(0) if key is None else jnp.asarray(key)
    com, (pos, x, act), _ = _record(body, g, key, steps)
    active = np.asarray(body.active)
    p0 = np.asarray(pos[0])[active]
    return dict(com=np.asarray(com[:, :2]), pos=np.asarray(pos), x=np.asarray(x),
                act=np.asarray(act), active=active,
                edges=_edges(np.asarray(body.nbrs), np.asarray(body.deg), active),
                extent=max(np.ptp(p0[:, 0]), np.ptp(p0[:, 1])) * 0.9 + 3.0)


def swimmer_spec(path, sd, steps, label=None, every=4, fps=30, size=6):
    return dict(path=path, panels=[(sd, label)], cols=1, cell=size, dpi=120,
                steps=steps, every=every, fps=fps)


def gallery_spec(path, panels, steps, cols=3, every=4, fps=30, cell=3.2):
    """panels: list of (sd, label)."""
    return dict(path=path, panels=panels, cols=cols, cell=cell, dpi=110,
                steps=steps, every=every, fps=fps)


def _write_video(fig, draw, frames, path, fps):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)
    for t in frames:
        draw(t)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    plt.close(fig)


def render_growth(grower, blocks, path, fps=30, size=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    params = jax.tree.map(jnp.asarray, grower["params"])
    key = jax.random.PRNGKey(grower["seed"])
    state = jax.tree.map(lambda x: x[0], init_states(key[None], [grower["graph"]]))
    _, _, last_change, (n, pos, active, gra, nbrs, deg) = jax.jit(
        lambda p, s, k: grow(p, s, k, blocks, record=True))(params, state, key)
    pos, active, gra, nbrs, deg = map(np.asarray, (pos, active, gra, nbrs, deg))

    stop = min(int(last_change) + 40, blocks)
    pf = pos[stop - 1][active[stop - 1]]
    c = pf.mean(0)
    extent = max(np.ptp(pf[:, 0]), np.ptp(pf[:, 1])) * 0.6 + 2.0
    fig, ax = plt.subplots(figsize=(size, size), dpi=120)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(0, 0, 1, 1)

    def draw(t):
        ax.clear()
        ax.set_facecolor(BG)
        a, p = active[t], pos[t]
        e = _edges(nbrs[t], deg[t], a)
        e = e[active[t][nbrs[t][e[:, 0], e[:, 1]]]]
        segs = np.stack([p[e[:, 0]], p[e[:, 2]]], 1)
        ax.add_collection(LineCollection(segs, colors="#56657a", linewidths=0.8, zorder=1))
        ax.scatter(p[a, 0], p[a, 1], s=14, c=np.clip(gra[t][a], 0, 1), linewidths=0, zorder=2)
        ax.set_xlim(c[0] - extent, c[0] + extent)
        ax.set_ylim(c[1] - extent, c[1] + extent)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.03, 0.96, f"step {t * TOPO_EVERY}   nodes {int(a.sum())}",
                transform=ax.transAxes, color="#aab", fontsize=9, va="top")

    frames = list(range(0, stop)) + [stop - 1] * fps
    _write_video(fig, draw, frames, path, fps)


def run_render(args):
    with open(OUT / args.swimmers, "rb") as f:
        res = pickle.load(f)
    set_drag(res.get("drag", (C_T, C_N, C_Q)))
    with open(OUT / "growers.pkl", "rb") as f:
        blocks = pickle.load(f)["blocks"]
    vid = OUT / "videos" / Path(args.swimmers).stem
    vid.mkdir(parents=True, exist_ok=True)
    order = np.argsort([-b[0] for b in res["best"]])[:args.top]
    print("simulating", flush=True)
    sds = [_simulate(res["growers"][i], res["best"][i][1], args.steps) for i in order]
    labels = [f"n={res['growers'][i]['n']}  d={res['best'][i][0]:.1f}" for i in order]
    names = [f"{rank:02d}_n{res['growers'][i]['n']}_d{res['best'][i][0]:.0f}"
             for rank, i in enumerate(order)]
    specs = [gallery_spec(vid / "gallery.mp4", list(zip(sds, labels))[:9], args.steps)]
    specs += [swimmer_spec(vid / f"{name}_swim.mp4", sd, args.steps) for sd, name in zip(sds, names)]
    print(f"rendering {len(specs)} videos", flush=True)
    render_videos(specs)
    if args.growth:
        for name, i in zip(names, order):
            print(f"rendering {name} growth", flush=True)
            render_growth(res["growers"][i], blocks, vid / f"{name}_grow.mp4")


def common_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--drag", type=float, nargs=3, default=(C_T, C_N, C_Q),
                    metavar=("CT", "CN", "CQ"),
                    help="tangential, normal linear, normal quadratic drag")
    return ap


def configure(args):
    global OUT
    OUT = args.out_dir
    set_drag(args.drag)
    print(f"backend {jax.default_backend()} ({jax.local_device_count()} devices)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = common_args()
    g = sub.add_parser("grow", parents=[common])
    g.add_argument("--gens", type=int, default=30)
    g.add_argument("--batch", type=int, default=256)
    g.add_argument("--blocks", type=int, default=400)
    g.add_argument("--steady-blocks", type=int, default=150)
    g.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("swim", parents=[common])
    s.add_argument("--gens", type=int, default=60)
    s.add_argument("--batch", type=int, default=1024)
    s.add_argument("--steps", type=int, default=2500)
    s.add_argument("--max-bodies", type=int, default=32)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--growers", default="growers.pkl")
    s.add_argument("--out", default="swimmers.pkl")
    r = sub.add_parser("render", parents=[common])
    r.add_argument("--top", type=int, default=6)
    r.add_argument("--steps", type=int, default=4000)
    r.add_argument("--growth", action="store_true")
    r.add_argument("--swimmers", default="swimmers.pkl")
    args = ap.parse_args()
    configure(args)
    {"grow": run_grow, "swim": run_swim, "render": run_render}[args.cmd](args)


if __name__ == "__main__":
    main()
