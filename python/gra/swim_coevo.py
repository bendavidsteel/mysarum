"""Bodies and controllers evolved together: MAP-Elites over body size x gait.

Each elite is a body (GRA growth params + seed + seed graph) and a controller
(muscle CA genome, with the node-position input). A child either keeps its
parent's body, regrows a mutated one, or borrows another elite's body; its
controller is always mutated. Rhythm must still emerge from the CA: there is
no clock input.

Fitness is net progress: displacement after the transient, in body lengths per
20 time units, so circling and thrashing on the spot score nothing and big
bodies aren't favoured just for being long. Gait descriptors come from the
muscle drive binned along the body's long axis:
  freq  dominant beat frequency, cycles per time unit
  wave  wavelengths of travelling wave along the body, signed by swimming
        direction: < 0 retrograde (wave runs back, as in eels), > 0 direct,
        ~0 the whole body pulses in phase

Sims run on the swim_fast kernel with a time step of DT and the repulsion force
refreshed every REP_EVERY steps; `verify` re-scores an archive with swim.DT and
repulsion every step.

    uv run python swim_coevo.py search --tag water --gens 200
    uv run python swim_coevo.py emit --tag water_cma --resume coevo_water.pkl
    uv run python swim_coevo.py verify --tag water_cma
    uv run python swim_coevo.py render --tag water_cma
"""

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import swim as sw
import swim_fast as sf
from swim_opt import CMAES, _flatten, _layout, _unflatten, clip, scale_vector

SIZE_BINS = (0, 30, 50, 80, 128, sw.N)
FREQ_BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, np.inf])
WAVE_BINS = np.array([-np.inf, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, np.inf])
ASPECT_BINS = np.array([0.0, 1.4, 2.2, np.inf])  # rest-shape elongation (sqrt of PCA eigenvalue ratio)
# bigger bodies beat slower, so they swim for longer per evaluation (keyed by size bucket)
STEP_SCALE = {32: 1.0, 64: 1.6, 128: 2.4, sw.N: 3.2}
K = 8  # CA channels
DT = 0.04        # search time step (swim.DT is 0.02)
REP_EVERY = 2    # steps between refreshes of the repulsion force


class Evaluator:
    """Scores batches of (body id, controller, CA key) on the swim_fast kernel."""

    def __init__(self, time, drag, dt, rep_every):
        self.time, self.drag, self.dt, self.rep_every = time, np.asarray(drag, np.float32), dt, rep_every
        self.prep = {}   # body id -> kernel constants

    def __call__(self, store, bid, genomes, keys):
        self.prep = {b: p for b, p in self.prep.items() if b in store}
        ids, idx = np.unique(bid, return_inverse=True)
        for b in ids:
            if b not in self.prep:
                self.prep[b] = sf.prep_body(store[b]["body"])
        bodies = [self.prep[b] for b in ids]
        scale = np.array([STEP_SCALE[next(s for s in sw.BUCKETS if p["n"] <= s)] for p in bodies])
        g = jax.tree.map(np.asarray, sw.Genome(*genomes))
        g = g._replace(log_tau=np.clip(g.log_tau, -1.5, 1.5), amp=np.clip(g.amp, 0.02, 0.45))
        sim = sf.simulate(bodies, idx, g, np.asarray(keys), (self.time / self.dt * scale[idx]).astype(np.int64),
                          drag=self.drag, dt=self.dt, rep_every=self.rep_every)
        d = sf.describe(sim, bodies, idx)
        return dict(speed=np.where(d["ok"], d["net"], -1.0), **{k: d[k] for k in (
            "path", "freq", "wave", "rhythm", "straight", "revs", "length", "ok")})


def aspect_of(rec):
    return sw.body_shape(sw.Body(**rec["body"]))[1]


def cell_of(n, aspect, freq, wave):
    return (int(np.searchsorted(SIZE_BINS, n, "left") - 1),
            int(np.clip(np.searchsorted(ASPECT_BINS, aspect, "right") - 1, 0, len(ASPECT_BINS) - 2)),
            int(np.clip(np.searchsorted(FREQ_BINS, freq, "right") - 1, 0, len(FREQ_BINS) - 2)),
            int(np.clip(np.searchsorted(WAVE_BINS, wave, "right") - 1, 0, len(WAVE_BINS) - 2)))


class Grower:
    """Grows batches of mutated bodies; fixed batch size so it compiles once."""

    def __init__(self, blocks, steady_blocks, batch):
        self.blocks, self.steady, self.batch = blocks, steady_blocks, batch
        self.fn = sw.batched(lambda p, s, k: sw.grow(p, s, k, blocks)[:3])

    def __call__(self, pick, rng, key):
        """pick: batch-size list of parent body records. Returns (index, settled record) per success."""
        G = self.batch
        pp = jax.tree.map(lambda *x: jnp.stack(x), *[jax.tree.map(jnp.asarray, b["params"]) for b in pick])
        km, ks = jax.random.split(key)
        params = jax.vmap(sw.mutate_params)(pp, jax.random.split(km, G))
        seeds = np.array([b["seed"] for b in pick])
        reseed = rng.random(G) < 0.3
        seeds = np.where(reseed, np.asarray(jax.random.randint(ks, (G,), 0, 2 ** 30)), seeds)
        graphs = np.array([b["graph"] for b in pick])
        keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
        state, pos, last = jax.device_get(self.fn(params, sw.init_states(keys, graphs), keys))
        params = jax.device_get(params)
        out, idx = [], []
        for b in range(G):
            if self.blocks - 1 - int(last[b]) < self.steady:
                continue
            n_raw = int(state.num_active[b])
            if n_raw < 16 or n_raw > sw.N - 8:
                continue
            body = sw.extract_body(pos[b], state.node_active[b], state.neighbors[b],
                                   state.num_neighbors[b], state.node_states[b])
            if body is None or body.active.sum() < 16:
                continue
            out.append(dict(params=jax.tree.map(lambda x: np.asarray(x[b]), params), seed=int(seeds[b]),
                            graph=int(graphs[b]), body=body._asdict(), n=int(body.active.sum())))
            idx.append(b)
        return list(zip(idx, sf.settle(out, 2000))) if out else []


def regrow(recs, blocks):
    """Regrow body records from their GRA params, seed and seed graph, under the current physics.

    For records laid out under other physics: growth doesn't depend on node
    positions, so each body comes back as the same graph with a new rest shape.
    """
    fn = sw.batched(lambda p, s, k: sw.grow(p, s, k, blocks)[:3])
    params = jax.tree.map(lambda *x: jnp.stack(x), *[jax.tree.map(jnp.asarray, r["params"]) for r in recs])
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray([r["seed"] for r in recs]))
    state, pos, _ = jax.device_get(fn(params, sw.init_states(keys, np.array([r["graph"] for r in recs])), keys))
    out = []
    for b, r in enumerate(recs):
        body = sw.extract_body(pos[b], state.node_active[b], state.neighbors[b],
                               state.num_neighbors[b], state.node_states[b])
        out.append(dict(r, body=body._asdict(), n=int(body.active.sum())))
    return sf.settle(out, 4000)


def run_search(args):
    out_path = sw.OUT / f"coevo_{args.tag}.pkl"
    drag = (sw.C_T, sw.C_N, sw.C_Q)
    evaluate = Evaluator(args.time, drag, args.dt, args.rep_every)
    with open(sw.OUT / args.growers, "rb") as f:
        blocks = pickle.load(f)["blocks"]
    init = sf.settle([dict(params=e["params"], seed=e["seed"], graph=e["graph"], body=e["body"], n=e["n"])
                      for e in sw.load_growers(sw.OUT / args.growers)], 2000)
    for r in init:
        r["aspect"] = aspect_of(r)
    store = dict(enumerate(init))   # body id -> record
    next_id = len(store)
    grower = Grower(blocks, args.steady_blocks, args.grow_batch)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    archive = {}
    B = args.batch
    seeds = []   # (body id, controller) from an earlier swim_qd run on the same growers
    if args.seed_from:
        with open(sw.OUT / args.seed_from, "rb") as f:
            qd = pickle.load(f)
        by_origin = {(r["seed"], r["graph"]): b for b, r in store.items()}
        for e in qd["archive"].values():
            g = qd["growers"][e["body"]]
            if (g["seed"], g["graph"]) in by_origin:
                seeds.append((by_origin[g["seed"], g["graph"]], sw.upgrade_genome(e["genome"], K)))
        print(f"seeding generation 0 with {len(seeds)} controllers from {args.seed_from}", flush=True)
    print(f"{len(store)} seed bodies, batch {B}, time {args.time}, drag {drag}", flush=True)
    t0 = time.time()
    for gen in range(args.gens):
        key, kg, km, kk, kb = jax.random.split(key, 5)
        rand = jax.vmap(lambda k: sw.sample_genome(k, K))(jax.random.split(kg, B))
        elites = list(archive.values())
        if gen < args.init_gens:
            ids = list(store)
            bid = np.array([ids[i] for i in rng.integers(0, len(ids), B)])
            genomes, n_grown = rand, 0
            if gen == 0 and seeds:
                m = min(len(seeds), B)
                bid[:m] = [b for b, _ in seeds[:m]]
                genomes = jax.tree.map(lambda r, *x: r.at[:m].set(jnp.stack(x)), rand,
                                       *[g for _, g in seeds[:m]])
        else:
            pick = rng.integers(0, len(elites), B)
            bid = np.array([elites[p]["body"] for p in pick])
            parents = jax.tree.map(lambda *x: jnp.stack(x), *[elites[p]["genome"] for p in pick])
            genomes = jax.vmap(sw.mutate_genome)(parents, jax.random.split(km, B))
            fresh = rng.random(B) < 0.05
            genomes = jax.tree.map(
                lambda a, b: jnp.where(fresh.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), genomes, rand)
            # borrow another elite's body
            hop = rng.random(B) < args.hop
            bid = np.where(hop, bid[rng.permutation(B)], bid)
            # regrow mutated bodies from elite bodies; their children keep the parent's controller line
            new = grower([store[b] for b in bid[:args.grow_batch]], rng, kb)
            for i, rec in new:
                rec["aspect"] = aspect_of(rec)
                store[next_id] = rec
                bid[i] = next_id
                next_id += 1
            n_grown = len(new)
        gnp, knp = jax.device_get((genomes, jax.random.split(kk, B)))
        res = evaluate(store, bid, gnp, knp)
        n_new = 0
        for i in np.nonzero(res["ok"] & (res["speed"] > 0))[0]:
            n = store[bid[i]]["n"]
            cell = cell_of(n, store[bid[i]]["aspect"], res["freq"][i], res["wave"][i])
            if cell not in archive or archive[cell]["speed"] < res["speed"][i]:
                archive[cell] = dict({k: float(v[i]) for k, v in res.items()}, body=int(bid[i]), n=n,
                                     key=knp[i], genome=jax.tree.map(lambda x: x[i], gnp))
                n_new += 1
        # keep only bodies the archive uses (and the seed bodies, for restarts)
        used = {e["body"] for e in archive.values()}
        store = {b: r for b, r in store.items() if b in used or b < len(init)}
        sp = np.array([e["speed"] for e in archive.values()])
        sizes = np.bincount([c[0] for c in archive], minlength=len(SIZE_BINS) - 1)
        best = [max([e["speed"] for c, e in archive.items() if c[0] == k], default=0) for k in range(len(sizes))]
        print(f"gen {gen:3d}  cells {len(archive):3d} {sizes}  new {n_new:4d}  grown {n_grown:2d}  "
              f"best/size {np.round(best, 2)}  ok {res['ok'].mean():.2f}  {time.time() - t0:6.0f}s", flush=True)
        with open(out_path, "wb") as f:
            pickle.dump(dict(archive=archive, bodies={e["body"]: store[e["body"]] for e in archive.values()},
                             time=args.time, dt=args.dt, rep_every=args.rep_every, drag=tuple(drag), blocks=blocks), f)


# ── CMA-ME search ────────────────────────────────────────────────────────────

class Emitter:
    """A CMA-ES over one body's controller, started from an elite's genome.

    Ranked as an improvement emitter (CMA-ME): samples that open a new cell
    first, then those that improve a cell by how much, then the rest. A newly
    grown body is protected for its first few generations, ranked by its own
    speed, so its controller can catch up with tuned incumbents before it
    has to compete.
    """

    def __init__(self, body, genome, lam, sigma, seed, protect=0):
        self.body, self.x0 = body, _flatten(genome)
        self.cma = CMAES(len(self.x0), lam, sigma, seed)
        self.protect, self.age, self.stale, self.best = protect, 0, 0, -np.inf

    def ask(self, scale):
        self.u = self.cma.ask()
        return self.x0 + self.u * scale

    def tell(self, speed, status, gain):
        protected = self.age < self.protect
        f = speed if protected else np.where(status == 2, 100 + speed, np.where(status == 1, 10 + gain, speed - 10))
        self.cma.tell(self.u, f)
        self.age += 1
        self.best = max(self.best, float(speed.max()))
        self.stale = 0 if protected or (status > 0).any() else self.stale + 1

    def done(self, max_age, patience):
        return self.age >= max_age or self.stale >= patience or self.cma.sigma < 0.02


def run_emit(args):
    """CMA-ME over bodies x controllers, warm-started from an earlier coevo archive."""
    out_path = sw.OUT / f"coevo_{args.tag}.pkl"
    with open(sw.OUT / args.resume, "rb") as f:
        prev = pickle.load(f)
    drag = prev["drag"]
    blocks = prev["blocks"]
    evaluate = Evaluator(args.time, drag, args.dt, args.rep_every)
    store = {b: prev["bodies"][b] for b in sorted({e["body"] for e in prev["archive"].values()})}
    if args.regrow:
        store = dict(zip(store, regrow(list(store.values()), blocks)))
        print(f"regrew {len(store)} bodies", flush=True)
    store = {b: dict(rec, aspect=aspect_of(rec)) for b, rec in store.items()}
    next_id = max(store) + 1
    grower = Grower(blocks, args.steady_blocks, args.grow_batch)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    archive = {}

    def insert(bid, res, genomes, keys):
        """Insert each result; returns per-sim status (2 new cell, 1 improved, 0 not) and gain."""
        status, gain = np.zeros(len(bid), int), np.zeros(len(bid))
        for i in np.nonzero(res["ok"] & (res["speed"] > 0))[0]:
            rec = store[bid[i]]
            cell = cell_of(rec["n"], rec["aspect"], res["freq"][i], res["wave"][i])
            old = archive.get(cell)
            if old is None or old["speed"] < res["speed"][i]:
                status[i] = 1 if old else 2
                gain[i] = res["speed"][i] - (old["speed"] if old else 0)
                archive[cell] = dict({k: float(v[i]) for k, v in res.items()}, body=int(bid[i]), n=rec["n"],
                                     aspect=rec["aspect"], key=keys[i],
                                     genome=jax.tree.map(lambda x: np.asarray(x[i]), genomes))
        return status, gain

    # generation 0: re-score the old elites, and random controllers on every body
    old = list(prev["archive"].values())
    bid = np.array([e["body"] for e in old])
    genomes = jax.tree.map(lambda *x: np.stack(x), *[sw.upgrade_genome(e["genome"], K) for e in old])
    keys = np.stack([e["key"] for e in old])
    insert(bid, evaluate(store, bid, genomes, keys), genomes, keys)
    print(f"re-scored {len(old)} elites into {len(archive)} cells", flush=True)
    if args.init_random:
        bid = np.repeat(list(store), args.init_random)
        key, kg, kk = jax.random.split(key, 3)
        genomes = jax.device_get(jax.vmap(lambda k: sw.sample_genome(k, K))(jax.random.split(kg, len(bid))))
        keys = np.asarray(jax.random.split(kk, len(bid)))
        insert(bid, evaluate(store, bid, genomes, keys), genomes, keys)
        print(f"{len(bid)} random controllers: {len(archive)} cells", flush=True)

    layout = _layout(sw.upgrade_genome(old[0]["genome"], K))
    scale = scale_vector(layout)
    L, E = args.lam, args.emitters
    nursery = []   # freshly grown bodies waiting for an emitter: (body id, parent genome)
    n_grown = n_spawned = 0

    def spawn():
        nonlocal next_id, n_grown, n_spawned
        n_spawned += 1
        elites = list(archive.values())
        if rng.random() < args.new_body:
            if not nursery:
                parents = [elites[i] for i in rng.integers(0, len(elites), args.grow_batch)]
                for i, rec in grower([store[p["body"]] for p in parents], rng, jax.random.PRNGKey(rng.integers(2 ** 31))):
                    rec["aspect"] = aspect_of(rec)
                    store[next_id] = rec
                    nursery.append((next_id, parents[i]["genome"]))
                    next_id += 1
                    n_grown += 1
            if nursery:
                b, g = nursery.pop(int(rng.integers(len(nursery))))
                return Emitter(b, g, L, args.sigma, int(rng.integers(2 ** 31)), args.protect)
        e = elites[rng.integers(len(elites))]
        return Emitter(e["body"], e["genome"], L, args.sigma, int(rng.integers(2 ** 31)))

    emitters = [spawn() for _ in range(E)]
    fresh = []   # retired new-body emitters: (body kept in archive, best speed / size-class best)
    t0 = time.time()
    for gen in range(args.gens):
        key, kk = jax.random.split(key)
        # common random numbers: one CA initial state per emitter per generation
        keys = np.repeat(np.asarray(jax.random.split(kk, E)), L, 0)
        xs = np.concatenate([em.ask(scale) for em in emitters]).astype(np.float32)
        genomes = jax.tree.map(np.asarray, clip(_unflatten(xs, layout)))
        bid = np.repeat([em.body for em in emitters], L)
        res = evaluate(store, bid, genomes, keys)
        speed = res["speed"]
        status, gain = insert(bid, res, genomes, keys)
        for j, em in enumerate(emitters):
            sl = slice(j * L, (j + 1) * L)
            em.tell(speed[sl], status[sl], gain[sl])
        n_new, n_imp = int((status == 2).sum()), int((status == 1).sum())
        sizes = np.bincount([c[0] for c in archive], minlength=len(SIZE_BINS) - 1)
        best = [max([e["speed"] for c, e in archive.items() if c[0] == k], default=0) for k in range(len(sizes))]
        in_archive = {e["body"] for e in archive.values()}
        for j, em in enumerate(emitters):
            if not em.done(args.max_age, args.patience):
                continue
            if em.protect:   # how did the new body fare, relative to the best of its size class?
                rec = store[em.body]
                fresh.append((em.body in in_archive, em.best / max(best[cell_of(rec["n"], 1, 0, 0)[0]], 1e-6)))
            emitters[j] = spawn()
        used = in_archive | {em.body for em in emitters} | {b for b, _ in nursery}
        store = {b: r for b, r in store.items() if b in used}
        n_bodies = len({e["body"] for e in archive.values()})
        print(f"gen {gen:3d}  cells {len(archive):3d} {sizes}  new {n_new:3d}  imp {n_imp:3d}  bodies {n_bodies:3d}  "
              f"grown {n_grown:4d}  emitters {n_spawned:4d}  new bodies kept {sum(k for k, _ in fresh)}/{len(fresh)} "
              f"rel {np.median([r for _, r in fresh]) if fresh else 0:.2f}  best/size {np.round(best, 2)}  "
              f"{time.time() - t0:6.0f}s", flush=True)
        with open(out_path, "wb") as f:
            pickle.dump(dict(archive=archive, bodies={e["body"]: store[e["body"]] for e in archive.values()},
                             time=args.time, dt=args.dt, rep_every=args.rep_every, drag=tuple(drag), blocks=blocks), f)


# ── Rendering ────────────────────────────────────────────────────────────────

def gait_name(e):
    w = e["wave"]
    kind = "pulsing" if abs(w) < 0.2 else ("retrograde" if w < 0 else "direct") + f" wave {abs(w):.1f}"
    return f"{kind}, {e['freq']:.2f}/t"


def pick_diverse(archive, k):
    """Fastest elite per size class, then farthest-point fill over (size, freq, wave)."""
    cells = list(archive)
    speed = np.array([archive[c].get("robust", archive[c]["speed"]) for c in cells])   # verified, if verify has run
    d = np.array([[c[0] / 4, np.log2(max(archive[c]["freq"], 0.02)) / 5, archive[c]["wave"] / 2] for c in cells])
    chosen = []
    for s in sorted({c[0] for c in cells}):
        chosen.append(max((i for i, c in enumerate(cells) if c[0] == s), key=lambda i: speed[i]))
    while len(chosen) < min(k, len(cells)):
        dist = np.min(((d[:, None] - d[chosen][None]) ** 2).sum(-1), 1)
        score = dist * speed ** 0.5
        chosen.append(int(np.argmax(score)))
    return [cells[i] for i in chosen[:k]]


def run_render(args):
    with open(sw.OUT / f"coevo_{args.tag}.pkl", "rb") as f:
        res = pickle.load(f)
    # a verified archive is shown with the integration its speeds were verified with
    verified = all("robust" in e for e in res["archive"].values())
    dt, rep_every = (sw.DT, 1) if verified else (res["dt"], res["rep_every"])
    steps, every = int(args.time / dt), max(int(round(0.08 / dt)), 1)
    vid = sw.OUT / "videos" / f"coevo_{args.tag}"
    vid.mkdir(parents=True, exist_ok=True)
    cells = pick_diverse(res["archive"], args.top)
    print("simulating", flush=True)
    panels, names = [], []
    for rank, c in enumerate(cells):
        e = res["archive"][c]
        sd = sf.recording(res["bodies"][e["body"]], sw.upgrade_genome(e["genome"], K), steps, e["key"],
                          res["drag"], dt, rep_every)
        panels.append((sd, f"n={e['n']}  {e.get('robust', e['speed']):.2f} BL  {gait_name(e)}"))
        names.append(f"{rank:02d}_n{e['n']}_{e.get('robust', e['speed']):.2f}bl")
        print(f"  {names[-1]}  {gait_name(e)}", flush=True)
    specs = [sw.gallery_spec(vid / "gallery.mp4", panels[:9], steps, every=every)]
    specs += [sw.swimmer_spec(vid / f"{name}.mp4", sd, steps, label, every=every)
              for (sd, label), name in zip(panels, names)]
    print(f"rendering {len(specs)} videos", flush=True)
    sw.render_videos(specs)


def run_verify(args):
    """Re-score an archive's elites with the reference integration: swim.DT, repulsion every step.

    Each elite is scored from the CA initial state it was found with and from
    args.keys fresh ones; the mean over the fresh ones is saved in the archive
    as `robust`, which render then uses to pick and label swimmers.
    """
    path = sw.OUT / f"coevo_{args.tag}.pkl"
    with open(path, "rb") as f:
        res = pickle.load(f)
    cells = list(res["archive"])
    elites = [res["archive"][c] for c in cells]
    bid = np.array([e["body"] for e in elites])
    genomes = jax.tree.map(lambda *x: np.stack(x), *[e["genome"] for e in elites])
    evaluate = Evaluator(res["time"], res["drag"], sw.DT, 1)
    ref = evaluate(res["bodies"], bid, genomes, np.stack([e["key"] for e in elites]))
    fresh = [np.asarray(jax.random.split(jax.random.PRNGKey(args.seed + r), len(elites))) for r in range(args.keys)]
    robust = np.mean([np.maximum(evaluate(res["bodies"], bid, genomes, k)["speed"], 0) for k in fresh], 0)
    stored = np.array([e["speed"] for e in elites])
    same = np.array([cell_of(e["n"], e["aspect"], f, w) == c
                     for c, e, f, w in zip(cells, elites, ref["freq"], ref["wave"])])
    print(f"search: dt {res['dt']}, repulsion every {res['rep_every']} steps; reference: dt {sw.DT}, every step; "
          f"robust: reference, mean of {args.keys} fresh CA initial states")
    print("size class  elites  median search / reference / robust   reference/search q10 q50 q90   same gait cell"
          "   best search -> reference, robust   best robust")
    for k in range(len(SIZE_BINS) - 1):
        m = np.array([c[0] == k for c in cells])
        if not m.any():
            continue
        ratio = np.quantile(ref["speed"][m] / stored[m], [0.1, 0.5, 0.9])
        i = np.argmax(stored[m])
        print(f"{SIZE_BINS[k]:4d}-{SIZE_BINS[k + 1]:<4d}   {m.sum():5d}  {np.median(stored[m]):13.3f} / "
              f"{np.median(ref['speed'][m]):.3f} / {np.median(robust[m]):.3f}   {ratio[0]:18.2f} {ratio[1]:.2f} {ratio[2]:.2f}   "
              f"{same[m].mean():14.2f}   {stored[m][i]:11.3f} -> {ref['speed'][m][i]:.3f}, {robust[m][i]:.3f}   "
              f"{robust[m].max():11.3f}")
    print(f"all: search vs reference correlation {np.corrcoef(stored, ref['speed'])[0, 1]:.3f}, "
          f"search vs robust {np.corrcoef(stored, robust)[0, 1]:.3f}, ok {ref['ok'].mean():.2f}")
    for e, r, s in zip(elites, ref["speed"], robust):
        e.update(reference=float(r), robust=float(s))
    with open(path, "wb") as f:
        pickle.dump(res, f)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = sw.common_args()
    s = sub.add_parser("search", parents=[common])
    s.add_argument("--tag", default="water")
    s.add_argument("--gens", type=int, default=200)
    s.add_argument("--init-gens", type=int, default=3)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--grow-batch", type=int, default=64)
    s.add_argument("--steady-blocks", type=int, default=150)
    s.add_argument("--hop", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--growers", default="growers.pkl")
    s.add_argument("--seed-from", default=None, help="swim_qd pickle whose elites seed generation 0")
    m = sub.add_parser("emit", parents=[common])
    m.add_argument("--tag", default="water_cma")
    m.add_argument("--resume", default="coevo_water.pkl", help="coevo pickle whose elites start the archive")
    m.add_argument("--gens", type=int, default=300)
    m.add_argument("--emitters", type=int, default=16)
    m.add_argument("--lam", type=int, default=32)
    m.add_argument("--sigma", type=float, default=0.3)
    m.add_argument("--new-body", type=float, default=0.5, help="chance a new emitter starts on a newly grown body")
    m.add_argument("--protect", type=int, default=10, help="generations a new body is ranked by its own speed")
    m.add_argument("--max-age", type=int, default=60)
    m.add_argument("--patience", type=int, default=8)
    m.add_argument("--grow-batch", type=int, default=64)
    m.add_argument("--steady-blocks", type=int, default=150)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--regrow", action="store_true",
                   help="regrow the resumed bodies under the current physics (archives from before the repulsion cutoff)")
    m.add_argument("--init-random", type=int, default=0, help="random controllers scored per body at the start")
    for p in (s, m):
        p.add_argument("--time", type=float, default=50.0, help="swim duration of the smallest size class")
        p.add_argument("--dt", type=float, default=DT)
        p.add_argument("--rep-every", type=int, default=REP_EVERY)
    r = sub.add_parser("render", parents=[common])
    r.add_argument("--tag", default="water")
    r.add_argument("--top", type=int, default=12)
    r.add_argument("--time", type=float, default=60.0)
    v = sub.add_parser("verify", parents=[common])
    v.add_argument("--tag", default="water_cma")
    v.add_argument("--keys", type=int, default=4, help="fresh CA initial states per elite")
    v.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    sw.configure(args)
    {"search": run_search, "emit": run_emit, "render": run_render, "verify": run_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
