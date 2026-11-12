"""Native CPU kernel for the swim simulation (numba), 11-14x faster than XLA on CPU.

The same model as swim.swim, in float32. Each sim runs on its body's exact
node count and undirected edge list (no size buckets, no degree padding), with
structure-of-arrays inner loops that LLVM vectorises, one sim per thread task.
swim.swim stays the reference; `python swim_fast.py check` compares the two.

The all-pairs repulsion is most of a step on big bodies and is a soft, slowly
varying force, so `rep_every` > 1 refreshes it only every few steps and holds
it in between (multiple time stepping); rep_every = 1 is swim.swim.

    simulate(bodies, body_idx, genomes, keys, steps)  -> traces (+ recordings)
    describe(sim, bodies, body_idx)                   -> speed and gait descriptors
    settle(recs)                                      -> bodies relaxed to rest
    recording(rec, genome, steps)                     -> one sim in the renderer's form
"""

import argparse
import math
import pickle
import time

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np

import swim as sw

GAIT_BINS = 8
TRACE_DT = 0.08      # time between trace samples
WINDOW_T = 4.0       # path length is measured between centre-of-mass samples this far apart
T_REF = 20.0         # speeds are per this much time (1000 steps at dt = 0.02)

# reassociation lets reductions vectorise; NaN/inf semantics stay, blowups must be detected
FASTMATH = {"reassoc", "contract", "arcp", "nsz", "afn"}
F = np.float32
FAR = F(3e4)          # where padding nodes are parked, out of reach of the repulsion


@nb.njit(inline="always", fastmath=FASTMATH, error_model="numpy")
def _tanh(v):
    """tanh to float32 rounding (rational fit on a clamped range), branch-free so loops vectorise."""
    v = min(max(v, F(-7.90531110763549805)), F(7.90531110763549805))
    v2 = v * v
    p = F(-2.76076847742355e-16)
    p = p * v2 + F(2.00018790482477e-13)
    p = p * v2 + F(-8.60467152213735e-11)
    p = p * v2 + F(5.12229709037114e-08)
    p = p * v2 + F(1.48572235717979e-05)
    p = p * v2 + F(6.37261928875436e-04)
    p = p * v2 + F(4.89352455891786e-03)
    q = F(1.19825839466702e-06)
    q = q * v2 + F(1.18534705686654e-04)
    q = q * v2 + F(2.26843463243900e-03)
    q = q * v2 + F(4.89352518554385e-03)
    return v * p / q


@nb.njit(inline="always")
def _gather(out, src, tab, count):
    """out[i] = sum over the table rows w of src[tab[w, i]], a node's gathers sharing a loop."""
    W = tab.shape[0]
    w = W % 2
    if w == 1:
        for i in range(count):
            out[i] = src[tab[0, i]]
    else:
        for i in range(count):
            out[i] = F(0.0)
    if (W - w) % 4 == 2:
        for i in range(count):
            out[i] += src[tab[w, i]] + src[tab[w + 1, i]]
        w += 2
    while w < W:
        for i in range(count):
            out[i] += src[tab[w, i]] + src[tab[w + 1, i]] + src[tab[w + 2, i]] + src[tab[w + 3, i]]
        w += 4


@nb.njit(inline="always")
def _gather_row(out, src, c, tab, count):
    """_gather on row c of two 2-d arrays (no views: they cost more than the loop on small bodies)."""
    W = tab.shape[0]
    w = W % 2
    if w == 1:
        for i in range(count):
            out[c, i] = src[c, tab[0, i]]
    else:
        for i in range(count):
            out[c, i] = F(0.0)
    if (W - w) % 4 == 2:
        for i in range(count):
            out[c, i] += src[c, tab[w, i]] + src[c, tab[w + 1, i]]
        w += 2
    while w < W:
        for i in range(count):
            out[c, i] += src[c, tab[w, i]] + src[c, tab[w + 1, i]] + src[c, tab[w + 2, i]] + src[c, tab[w + 3, i]]
        w += 4


@nb.njit(parallel=True, cache=True, nogil=True)
def _kernel(noff, eoff, toff, width, ei, ej, tab_n, tab_e, tab_h, len0, pos0, gra, axes, inv_deg, gbin, bin_w,
            body, A, a_self, U, b, log_tau, r, r0, amp, P, S, C, src, src_steps, drag, x0, steps,
            dt, k_spring, c_int, charge, inv_rc2, by_len, every, rep_every, rec_every,
            trace, final, rec_pos, rec_x, rec_act):
    for s in nb.prange(body.shape[0]):
        _sim(s, noff, eoff, toff, width, ei, ej, tab_n, tab_e, tab_h, len0, pos0, gra, axes, inv_deg, gbin, bin_w,
             body, A, a_self, U, b, log_tau, r, r0, amp, P, S, C, src, src_steps, drag, x0, steps,
             dt, k_spring, c_int, charge, inv_rc2, by_len, every, rep_every, rec_every,
             trace, final, rec_pos, rec_x, rec_act)


@nb.njit(fastmath=FASTMATH, error_model="numpy", cache=True, nogil=True)
def _sim(s, noff, eoff, toff, width, ei, ej, tab_n, tab_e, tab_h, len0, pos0, gra, axes, inv_deg, gbin, bin_w,
         body, A, a_self, U, b, log_tau, r, r0, amp, P, S, C, src, src_steps, drag, x0, steps,
         dt, k_spring, c_int, charge, inv_rc2, by_len, every, rep_every, rec_every,
         trace, final, rec_pos, rec_x, rec_act):
    """One sim. Every loop is written to vectorise: zero-based counters, one store per loop (two in
    the repulsion), node sums gathered through padded tables of unsigned indices rather than
    scattered from edges, and no array views in the step. Node and edge counts are padded to
    whole vectors (nv, ev) so no loop has a scalar tail: padding nodes sit far away with no
    edges, padding edges join node 0 to itself, and nothing gathers from either."""
    K = b.shape[1]
    bins = bin_w.shape[1]
    zero, half, one = F(0.0), F(0.5), F(1.0)
    bd = body[s]
    n0, n = noff[bd], noff[bd + 1] - noff[bd]
    e0, E = eoff[bd], eoff[bd + 1] - eoff[bd]
    W = width[bd]
    nv, ev = (n + 7) // 8 * 8, (E + 7) // 8 * 8
    ii, jj = np.zeros(ev, np.uint32), np.zeros(ev, np.uint32)
    for e in range(E):
        ii[e], jj[e] = ei[e0 + e], ej[e0 + e]
    # row w of each table: per node, its w-th neighbour / edge / half-edge (edge e at its i end,
    # ev + e at its j end), or the zero dummy slot past the end of the source array
    tn, te = np.full((W, nv), nv, np.uint32), np.full((W, nv), ev, np.uint32)
    th = np.full((W, nv), 2 * ev, np.uint32)
    for w in range(W):
        for i in range(n):
            q = toff[bd] + w * n + i
            if tab_n[q] < n:
                tn[w, i] = tab_n[q]
            if tab_e[q] < E:
                te[w, i] = tab_e[q]
            if tab_h[q] < E:
                th[w, i] = tab_h[q]
            elif tab_h[q] < 2 * E:
                th[w, i] = ev + tab_h[q] - E
    ideg = np.zeros(nv, F)
    c_t, c_n, c_q = drag[s, 0], drag[s, 1], drag[s, 2]
    r0_s = r0[s]

    px, py = np.full(nv, FAR, F), np.full(nv, FAR, F)
    rx, ry = np.empty(n, F), np.empty(n, F)
    for i in range(n):
        px[i], py[i] = pos0[n0 + i, 0], pos0[n0 + i, 1]
        rx[i], ry[i] = px[i], py[i]
        ideg[i] = inv_deg[n0 + i]
    vx, vy = np.zeros(nv, F), np.zeros(nv, F)
    fx, fy, fb = np.empty(nv, F), np.empty(nv, F), np.empty(nv, F)
    gx, gy = np.empty(nv, F), np.empty(nv, F)
    x, nbx = np.zeros((K, nv + 1), F), np.empty((K, nv), F)
    bias = np.zeros((K, nv), F)
    h, stretch, m, drive = np.empty(nv, F), np.empty(nv, F), np.empty(nv, F), np.empty(nv, F)
    conc = np.zeros(nv, F)
    sensing = src_steps[s] > 0
    ex, ey, ed, er = np.empty(ev, F), np.empty(ev, F), np.empty(ev, F), np.empty(ev, F)
    est, ea = np.zeros(ev + 1, F), np.zeros(ev + 1, F)
    vxi, vyi, vxj, vyj = np.empty(ev, F), np.empty(ev, F), np.empty(ev, F), np.empty(ev, F)
    vti, vtj, fm = np.empty(ev, F), np.empty(ev, F), np.empty(ev, F)
    # per half-edge: force along the edge, drag weight, force components
    ha, hb = np.empty(2 * ev, F), np.zeros(2 * ev + 1, F)
    hx, hy = np.zeros(2 * ev + 1, F), np.zeros(2 * ev + 1, F)
    inv_l0, stroke = np.zeros(ev, F), np.zeros(ev, F)
    for e in range(E):
        inv_l0[e] = one / len0[e0 + e]
        stroke[e] = amp[s] * len0[e0 + e] if by_len else amp[s]
    rate, r4 = np.empty(K, F), np.empty(K, F)
    for c in range(K):
        rate[c] = dt * F(math.exp(-log_tau[s, c]))
        r4[c] = F(4.0) * r[s, c]
        for i in range(n):
            x[c, i] = x0[s, i, c]
            bias[c, i] = gra[n0 + i, 0] * U[s, c, 0] + gra[n0 + i, 1] * U[s, c, 1] \
                + gra[n0 + i, 2] * U[s, c, 2] + b[s, c] \
                + axes[n0 + i, 0] * P[s, c, 0] + axes[n0 + i, 1] * P[s, c, 1]

    alive = True
    for t in range(steps[s]):
        # edge geometry
        for e in range(ev):
            ex[e] = px[jj[e]] - px[ii[e]]
        for e in range(ev):
            ey[e] = py[jj[e]] - py[ii[e]]
        for e in range(ev):
            ed[e] = math.sqrt(ex[e] * ex[e] + ey[e] * ey[e] + F(1e-6))
        for e in range(ev):
            er[e] = one / ed[e]
        for e in range(ev):
            ex[e] *= er[e]
        for e in range(ev):
            ey[e] *= er[e]
        for e in range(ev):
            est[e] = ed[e] * inv_l0[e] - one
        est[ev] = zero

        # muscle CA: neighbour mean, own state, fixed bias and the stretch of the node's edges
        _gather(stretch, est, te, nv)
        for i in range(nv):
            stretch[i] *= ideg[i]
        for c in range(K):
            _gather_row(nbx, x, c, tn, nv)
        # nutrient at each node: 1 at the source, falling linearly to 0 at its range; the source
        # steps through its path, src_steps steps at each point (padding nodes are out of range)
        if sensing:
            q = min(t // src_steps[s], src.shape[1] - 1)
            sx, sy, inv_range = src[s, q, 0], src[s, q, 1], one / src[s, q, 2]
            for i in range(nv):
                dx, dy = px[i] - sx, py[i] - sy
                conc[i] = max(one - math.sqrt(dx * dx + dy * dy + F(1e-6)) * inv_range, zero)
            w_own, w_mean, w_const = src[s, q, 3], src[s, q, 4], src[s, q, 5]
            if w_own != one or w_mean != zero or w_const != zero:   # what the nodes are told instead
                mean = zero
                for i in range(n):
                    mean += conc[i]
                mean /= n
                for i in range(nv):
                    conc[i] = w_own * conc[i] + w_mean * mean + w_const
        for i in range(nv):
            m[i] = r0_s
        for c in range(K):
            a_s, s_c, c_c, rc, r4c = a_self[s, c], S[s, c], C[s, c], rate[c], r4[c]
            for i in range(nv):
                h[i] = zero
            k = 0
            while k + 4 <= K:
                a0, a1, a2, a3 = A[s, c, k], A[s, c, k + 1], A[s, c, k + 2], A[s, c, k + 3]
                for i in range(nv):
                    h[i] += a0 * nbx[k, i] + a1 * nbx[k + 1, i] + a2 * nbx[k + 2, i] + a3 * nbx[k + 3, i]
                k += 4
            while k < K:
                a0 = A[s, c, k]
                for i in range(nv):
                    h[i] += a0 * nbx[k, i]
                k += 1
            # nbx holds the old neighbour sums of every channel, so x can be updated in place
            for i in range(nv):
                v = h[i] * ideg[i] + a_s * x[c, i] + bias[c, i] + s_c * stretch[i] + c_c * conc[i]
                x[c, i] += rc * (half + half * _tanh(half * v) - x[c, i])
            for i in range(nv):
                m[i] += r4c * (x[c, i] - half)
        for i in range(nv):
            m[i] = _tanh(m[i])

        # springs with muscle-set rest lengths and internal damping
        for e in range(ev):
            ea[e] = half * (m[ii[e]] + m[jj[e]])
        ea[ev] = zero
        for e in range(ev):
            vxi[e] = vx[ii[e]]
        for e in range(ev):
            vyi[e] = vy[ii[e]]
        for e in range(ev):
            vxj[e] = vx[jj[e]]
        for e in range(ev):
            vyj[e] = vy[jj[e]]
        for e in range(ev):
            vti[e] = vxi[e] * ex[e] + vyi[e] * ey[e]
        for e in range(ev):
            vtj[e] = vxj[e] * ex[e] + vyj[e] * ey[e]
        for e in range(ev):
            fm[e] = k_spring * (ed[e] - one - stroke[e] * ea[e]) + c_int * (vtj[e] - vti[e])
        # drag on each end of each edge segment: -w (c_t v_t + (c_n + c_q |v_n|) v_n), which is
        # (w (c_n + c_q |v_n|) - w c_t) v_t u - w (c_n + c_q |v_n|) v
        for e in range(ev):
            ux, uy = vxi[e] - vti[e] * ex[e], vyi[e] - vti[e] * ey[e]
            hb[e] = half * ed[e] * (c_n + c_q * math.sqrt(ux * ux + uy * uy + F(1e-9)))
        for e in range(ev):
            ux, uy = vxj[e] - vtj[e] * ex[e], vyj[e] - vtj[e] * ey[e]
            hb[ev + e] = half * ed[e] * (c_n + c_q * math.sqrt(ux * ux + uy * uy + F(1e-9)))
        for e in range(ev):
            ha[e] = fm[e] + (hb[e] - half * ed[e] * c_t) * vti[e]
        for e in range(ev):
            ha[ev + e] = (hb[ev + e] - half * ed[e] * c_t) * vtj[e] - fm[e]
        for e in range(ev):
            hx[e] = ha[e] * ex[e]
        for e in range(ev):
            hy[e] = ha[e] * ey[e]
        for e in range(ev):
            hx[ev + e] = ha[ev + e] * ex[e]
        for e in range(ev):
            hy[ev + e] = ha[ev + e] * ey[e]
        _gather(fx, hx, th, nv)
        _gather(fy, hy, th, nv)
        _gather(fb, hb, th, nv)

        # repulsion: every real node pushes all nodes closer than the cutoff. It is soft and
        # slowly varying, so it can be refreshed every rep_every steps and held in between
        if t % rep_every == 0:
            for j in range(nv):
                gx[j] = zero
            for j in range(nv):
                gy[j] = zero
            for i in range(n):
                xi, yi = px[i], py[i]
                for j in range(nv):
                    dx, dy = px[j] - xi, py[j] - yi
                    r2 = dx * dx + dy * dy
                    cut = max(one - r2 * inv_rc2, zero)
                    wr = charge * cut * cut / (one + r2)
                    gx[j] += wr * dx
                    gy[j] += wr * dy
        for i in range(nv):
            fx[i] += gx[i] - fb[i] * vx[i]
        for i in range(nv):
            fy[i] += gy[i] - fb[i] * vy[i]

        for i in range(nv):
            vx[i] += dt * fx[i]
        for i in range(nv):
            vy[i] += dt * fy[i]
        for i in range(nv):
            px[i] += dt * vx[i]
        for i in range(nv):
            py[i] += dt * vy[i]

        if (t + 1) % every == 0:
            q = (t + 1) // every - 1
            cx, cy = zero, zero
            for i in range(n):
                cx += px[i]
                cy += py[i]
            cx /= F(n)
            cy /= F(n)
            if not (abs(cx) < F(1e6) and abs(cy) < F(1e6)):
                alive = False
                break
            sn, cs = zero, zero
            for i in range(n):
                qx, qy = px[i] - cx, py[i] - cy
                sn += rx[i] * qy - ry[i] * qx
                cs += rx[i] * qx + ry[i] * qy
            trace[s, q, 0], trace[s, q, 1] = cx, cy
            trace[s, q, 2] = math.atan2(sn, cs)
            # mean muscle drive in bins along the body axis
            for g in range(bins):
                trace[s, q, 3 + g] = zero
            _gather(drive, ea, te, nv)
            for i in range(n):
                g = gbin[n0 + i]
                trace[s, q, 3 + g] += drive[i] * ideg[i] * bin_w[bd, g]
        if rec_every > 0 and t % rec_every == 0:
            q = t // rec_every
            for i in range(n):
                rec_pos[s, q, i, 0], rec_pos[s, q, i, 1] = px[i], py[i]
                for c in range(3):
                    rec_x[s, q, i, c] = x[c, i]
            for e in range(E):
                rec_act[s, q, e] = ea[e]
    if alive:
        for i in range(n):
            final[s, i, 0], final[s, i, 1] = px[i], py[i]


# ── Bodies ───────────────────────────────────────────────────────────────────

_CONSTS = {}


def _body_consts(size):
    """Position input, long axis and length, computed as swim.py does on the body's size bucket."""
    if size not in _CONSTS:
        def fn(pos, af):
            w = af / af.sum()
            c = pos - (pos * w[:, None]).sum(0)
            _, vec = jnp.linalg.eigh((c * w[:, None]).T @ c)
            a0 = vec[:, -1]
            proj = c @ a0
            a0 = a0 * jnp.sign((proj ** 3 * w).sum() + 1e-12)
            proj = jnp.where(af > 0, c @ a0, jnp.nan)
            return sw.body_axes(pos, af), a0, jnp.nanmax(proj) - jnp.nanmin(proj)
        _CONSTS[size] = jax.jit(fn)
    return _CONSTS[size]


def prep_body(body):
    """Per-body constants of the kernel, from a body dict (buffer sw.N, nodes packed first)."""
    active = np.asarray(body["active"])
    n = int(active.sum())
    assert active[:n].all(), "body nodes must be packed at the start of the buffer"
    edges = sw._edges(np.asarray(body["nbrs"]), np.asarray(body["deg"]), active)
    pos = np.asarray(body["pos"], F)
    size = next(s for s in sw.BUCKETS if n <= s)
    axes, a0, length = jax.device_get(_body_consts(size)(pos[:size], active[:size].astype(F)))
    i, j = edges[:, 0], edges[:, 2]
    E = len(edges)
    deg = np.bincount(np.concatenate([i, j]), minlength=n)
    # padded per-node tables of neighbours, edges and half-edges (edge e at its i end, E + e at its j end)
    ends = np.concatenate([i, j])
    order = np.argsort(ends, kind="stable")
    slot = np.arange(2 * E) - np.searchsorted(ends[order], ends[order])
    tabs = []
    for fill, val in ((n, np.concatenate([j, i])), (E, np.tile(np.arange(E), 2)), (2 * E, np.arange(2 * E))):
        tab = np.full((max(deg.max(), 1), n), fill, np.int32)
        tab[slot, ends[order]] = val[order]
        tabs.append(tab)
    gbin = np.clip(((axes[:n, 0] + 1) / 2 * GAIT_BINS).astype(np.int32), 0, GAIT_BINS - 1)
    return dict(n=n, pos=pos[:n], gra=np.asarray(body["gra"], F)[:n], axes=axes[:n], edges=edges,
                tab_n=tabs[0].ravel(), tab_e=tabs[1].ravel(), tab_h=tabs[2].ravel(), width=tabs[0].shape[0],
                len0=np.sqrt(((pos[j] - pos[i]) ** 2).sum(-1) + 1e-6).astype(F),
                inv_deg=(1.0 / np.maximum(deg, 1)).astype(F), gbin=gbin,
                bin_w=(1.0 / np.maximum(np.bincount(gbin, minlength=GAIT_BINS), 1)).astype(F),
                a0=a0, length=float(length), extent=float(np.sqrt((pos[:n] ** 2).sum(-1)).max()))


_X0 = jax.jit(lambda key, k: jax.random.uniform(key, (sw.N, k)), static_argnums=1)


def initial_states(keys, k):
    """(S, N, k) CA initial states from JAX keys, the draw sw.swim_init makes; repeated keys are drawn once."""
    uniq, inv = np.unique(np.asarray(keys), axis=0, return_inverse=True)
    return np.stack([np.asarray(_X0(u, k)) for u in uniq])[inv.ravel()]


def simulate(bodies, body_idx, genomes, keys, steps, drag=None, dt=None, rep_every=1, record=False,
             source=None, source_steps=None, sense=(1.0, 0.0, 0.0)):
    """Run S sims. bodies: prepared bodies (prep_body); body_idx (S,) indexes them.

    genomes is a Genome with leading S; keys (S, 2) JAX keys for the CA initial
    state; steps an int or (S,); drag (3,) or (S, 3), default the module fluid;
    rep_every: steps between refreshes of the repulsion force (1 is swim.swim).
    source: nutrient source (x, y, range) per sim, (S, 3), or a path of them
    (S, Q, 3) held for source_steps steps each (int or (S,)), the last one to
    the end; nodes sense the concentration through Genome.C. sense, (3,) or (S, 3):
    a node is told own * its concentration + mean * the body's mean + const
    (for ablations; swim.swim only has the default).
    Returns dict(trace (S, T, 3 + GAIT_BINS) sampled every `every` steps and NaN
    after a blowup or past a sim's own length, pos (S, n_max, 2), every, dt),
    plus recordings of every record-th step (rec_pos, rec_x, rec_act per edge) if record.
    """
    dt = sw.DT if dt is None else dt
    S = len(body_idx)
    g = jax.tree.map(lambda v: np.ascontiguousarray(v, F), sw.Genome(*genomes))
    K = g.b.shape[1]
    steps = np.broadcast_to(np.asarray(steps, np.int64), (S,)).copy()
    drag = np.broadcast_to(np.asarray((sw.C_T, sw.C_N, sw.C_Q) if drag is None else drag, F), (S, 3)).copy()
    every = max(int(round(TRACE_DT / dt)), 1)
    n = np.array([b["n"] for b in bodies])
    ne = np.array([len(b["edges"]) for b in bodies])
    cat = lambda f, dtype=F: np.ascontiguousarray(np.concatenate([b[f] for b in bodies]), dtype)
    edges = np.concatenate([b["edges"] for b in bodies])
    n_max = int(n[body_idx].max())
    trace = np.full((S, int(steps.max()) // every, 3 + GAIT_BINS), np.nan, F)
    final = np.full((S, n_max, 2), np.nan, F)
    R = -(-int(steps.max()) // int(record)) if record else 0
    rec = np.zeros((S, R, n_max, 2), F), np.zeros((S, R, n_max, 3), F), np.zeros((S, R, int(ne.max())), F)
    if source is None:
        src, src_steps = np.ones((S, 1, 6), F), np.zeros(S, np.int64)
        chem = np.zeros((S, K), F)
    else:
        src = np.asarray(source, F).reshape(S, -1, 3)
        sense = np.broadcast_to(np.asarray(sense, F).reshape(-1, 1, 3), src.shape)
        src = np.ascontiguousarray(np.concatenate([src, sense], 2))
        src_steps = np.broadcast_to(np.asarray(steps.max() if source_steps is None else source_steps, np.int64),
                                    (S,)).copy()
        chem = np.zeros((S, K), F) if g.C is None else g.C
    nb.set_parallel_chunksize(1)
    width = np.array([b["width"] for b in bodies])
    _kernel(np.concatenate([[0], np.cumsum(n)]), np.concatenate([[0], np.cumsum(ne)]),
            np.concatenate([[0], np.cumsum(n * width)]), width,
            edges[:, 0].astype(np.int32), edges[:, 2].astype(np.int32),
            cat("tab_n", np.int32), cat("tab_e", np.int32), cat("tab_h", np.int32), cat("len0"), cat("pos"),
            cat("gra"), cat("axes"), cat("inv_deg"), cat("gbin", np.int32),
            np.stack([b["bin_w"] for b in bodies]),
            np.asarray(body_idx, np.int64), g.A, g.a_self, g.U, g.b, g.log_tau, g.r, g.r0, g.amp, g.P,
            F(sw.SENSE) * g.S, chem, src, src_steps, drag, initial_states(keys, K), steps,
            F(dt), F(sw.K_SPRING), F(sw.C_INT), F(sw.CHARGE), F(1.0 / sw.R_CUT ** 2),
            sw.STROKE_BY_LENGTH, every, rep_every, int(record), trace, final, *rec)
    out = dict(trace=trace, pos=final, every=every, dt=dt, steps=steps, rep_every=rep_every)
    if record:
        out.update(rec_pos=rec[0], rec_x=rec[1], rec_act=rec[2])
    return out


def describe(sim, bodies, body_idx):
    """Speed, path and gait descriptors of each sim, as numpy arrays.

    net       net displacement after the transient, body lengths per T_REF
    path      path length (WINDOW_T sampling), body lengths per T_REF
    straight  net displacement / path length
    revs      body revolutions per T_REF
    freq      dominant beat frequency of the muscle drive, cycles per time unit
    rhythm    share of drive power at that frequency
    wave      wavelengths of travelling wave along the body, signed by the
              direction of travel: < 0 retrograde, > 0 direct, ~0 pulsing
    """
    S = len(body_idx)
    out = {k: np.zeros(S) for k in ("net", "path", "straight", "revs", "freq", "rhythm", "wave", "length")}
    out["ok"] = np.zeros(S, bool)
    sample_t = sim["every"] * sim["dt"]
    win = max(int(round(WINDOW_T / sample_t)), 1)
    body_idx = np.asarray(body_idx)
    length = np.array([b["length"] for b in bodies])[body_idx]
    extent = np.array([b["extent"] for b in bodies])[body_idx]
    a0 = np.stack([b["a0"] for b in bodies])[body_idx]
    n = np.array([b["n"] for b in bodies])[body_idx]
    for steps in np.unique(sim["steps"]):
        idx = np.nonzero(sim["steps"] == steps)[0]
        tr = sim["trace"][idx, :steps // sim["every"]].astype(np.float64)
        T = tr.shape[1]
        t0 = T // 5
        c = tr[:, t0::win, :2]
        path = np.sqrt(((c[:, 1:] - c[:, :-1]) ** 2).sum(-1)).sum(1)
        d = tr[:, -1, :2] - tr[:, t0, :2]
        disp = np.sqrt((d ** 2).sum(-1))
        dth = np.diff(tr[:, t0:, 2], axis=1)
        dth = (dth + np.pi) % (2 * np.pi) - np.pi
        span = (T - 1 - t0) * sample_t / T_REF
        pos = sim["pos"][idx].astype(np.float64)
        on = np.arange(pos.shape[1])[None] < n[idx, None]
        radius = np.sqrt(np.where(on, ((pos - tr[:, -1, None, :2]) ** 2).sum(-1), 0.0)).max(1)
        ok = np.isfinite(tr).all((1, 2)) & np.isfinite(radius) & (radius < 3.0 * extent[idx] + 4.0)

        sig = tr[:, t0:, 3:]
        X = np.fft.rfft(sig - sig.mean(1, keepdims=True), axis=1)       # (s, F, bins)
        Pw = (np.abs(X) ** 2).sum(-1)
        Pw[:, 0] = 0.0
        k = np.argmax(Pw, 1)
        rows = np.arange(len(idx))
        near = Pw[rows, k - 1] + Pw[rows, k] + Pw[rows, np.minimum(k + 1, Pw.shape[1] - 1)]
        Z = X[rows, k]
        step = np.angle((Z[:, 1:] * np.conj(Z[:, :-1])).sum(1))
        th = tr[:, t0:, 2]
        ax = np.stack([np.cos(th) * a0[idx, :1] - np.sin(th) * a0[idx, 1:],
                       np.sin(th) * a0[idx, :1] + np.cos(th) * a0[idx, 1:]], -1)
        heading = (ax * (d / (disp[:, None] + 1e-9))[:, None]).sum(-1).mean(1)

        out["net"][idx] = np.where(ok, disp / span / length[idx], -1.0)
        out["path"][idx] = np.where(ok, path / ((c.shape[1] - 1) * win * sample_t / T_REF) / length[idx], -1.0)
        out["straight"][idx] = disp / (path + 1e-6)
        out["revs"][idx] = np.abs(dth.sum(1)) / (2 * np.pi) / span
        out["freq"][idx] = k / ((T - t0) * sample_t)
        out["rhythm"][idx] = near / (Pw.sum(1) + 1e-12)
        out["wave"][idx] = -step * (GAIT_BINS - 1) / (2 * np.pi) * np.sign(heading + 1e-9)
        out["ok"][idx] = ok
    out["length"] = length
    with np.errstate(invalid="ignore"):
        for k in ("straight", "revs", "freq", "rhythm", "wave"):
            out[k] = np.nan_to_num(out[k])
    return out


def settle(recs, steps=2000):
    """Relax each body record under swim physics with muscles off; that is its rest shape."""
    bodies = [prep_body(r["body"]) for r in recs]
    off = jax.device_get(jax.vmap(sw.sample_genome)(jax.random.split(jax.random.PRNGKey(0), len(recs))))
    off = off._replace(amp=np.zeros(len(recs), F))
    keys = np.zeros((len(recs), 2), np.uint32)
    pos = simulate(bodies, np.arange(len(recs)), off, keys, steps)["pos"]
    for r, b, p in zip(recs, bodies, pos):
        new = np.zeros((sw.N, 2), F)
        new[:b["n"]] = p[:b["n"]] - p[:b["n"]].mean(0)
        r["body"] = dict(r["body"], pos=new)
    return recs


def recording(rec, genome, steps, key=None, drag=None, dt=None, rep_every=1, source=None, source_steps=None):
    """Re-run one swimmer recording every step, in the form the renderers take (as sw._simulate)."""
    body = prep_body(rec["body"])
    g = jax.tree.map(lambda v: np.asarray(v)[None], sw.Genome(*genome))
    key = np.zeros(2, np.uint32) if key is None else np.asarray(key)
    sim = simulate([body], [0], g, key[None], steps, drag=drag, dt=dt, rep_every=rep_every, record=True,
                   source=None if source is None else np.asarray(source)[None], source_steps=source_steps)
    n, edges = body["n"], body["edges"]
    act = np.zeros((steps, n, sw.D), F)
    act[:, edges[:, 0], edges[:, 1]] = sim["rec_act"][0]
    pos = sim["rec_pos"][0]
    return dict(com=pos.mean(1), pos=pos, x=sim["rec_x"][0], act=act, active=np.ones(n, bool), edges=edges,
                extent=max(np.ptp(pos[0, :, 0]), np.ptp(pos[0, :, 1])) * 0.9 + 3.0)


# ── Check against the JAX reference, and timing ──────────────────────────────

def _test_set(path, count, seed=0):
    """Bodies and controllers from a swim_coevo result, spread over body size."""
    with open(path, "rb") as f:
        res = pickle.load(f)
    elites = sorted(res["archive"].values(), key=lambda e: e["n"])
    elites = [elites[i] for i in np.linspace(0, len(elites) - 1, count).round().astype(int)]
    recs = [res["bodies"][e["body"]] for e in elites]
    genomes = [sw.upgrade_genome(e["genome"], 8, chem=True) for e in elites]
    rng = np.random.default_rng(seed)
    for g in genomes:   # old controllers have no stretch or nutrient input; give them both so they are exercised
        g.S[:] = rng.uniform(-3, 3, g.S.shape)
        g.C[:] = rng.uniform(-6, 6, g.C.shape)
    return recs, jax.tree.map(lambda *x: np.stack(x), *genomes), np.stack([e["key"] for e in elites])


def run_check(args):
    recs, genomes, keys = _test_set(sw.OUT / args.archive, args.count)
    bodies = [prep_body(r["body"]) for r in recs]
    idx = np.arange(len(recs))
    # a nutrient source two body lengths away, in a different direction for each body
    ang = np.linspace(0, 2 * np.pi, len(recs), endpoint=False)
    length = np.array([b["length"] for b in bodies])
    source = np.stack([2 * length * np.cos(ang), 2 * length * np.sin(ang), 4 * length], 1).astype(F)
    for steps in (200, args.steps):
        fast = simulate(bodies, idx, genomes, keys, steps, source=source)
        ref = sw.run_bucketed(lambda b, g, k, s: sw.swim(b, g, k, steps, gait_bins=GAIT_BINS, source=s)[0],
                              sw.stack_bodies([r["body"] for r in recs]), genomes, keys, source)
        ref = ref[:, fast["every"] - 1::fast["every"]]
        err = np.abs(fast["trace"] - ref)
        move = np.sqrt(((ref[:, -1, :2] - ref[:, 0, :2]) ** 2).sum(-1))
        print(f"{steps} steps: centre of mass differs by median {np.median(err[:, -1, :2].max(-1)):.2g}, "
              f"max {err[:, -1, :2].max():.2g} (bodies moved {np.median(move):.2f} median); "
              f"muscle drive differs by median {np.median(err[:, -1, 3:].max(-1)):.2g}, max {err[:, -1, 3:].max():.2g}")


def run_bench(args):
    recs, genomes, keys = _test_set(sw.OUT / args.archive, args.count)
    bodies = [prep_body(r["body"]) for r in recs]
    n = np.array([b["n"] for b in bodies])
    simulate(bodies[:1], [0], jax.tree.map(lambda v: v[:1], genomes), keys[:1], 10)
    lo = 0
    for size in sw.BUCKETS:
        pick = np.nonzero((n > lo) & (n <= size))[0]
        lo = size
        if len(pick) == 0:
            continue
        idx = np.resize(pick, args.sims)
        sub = jax.tree.map(lambda v: v[idx], genomes)
        simulate(bodies, idx, sub, keys[idx], 10)
        t0 = time.time()
        simulate(bodies, idx, sub, keys[idx], args.steps)
        dt = time.time() - t0
        print(f"bucket {size:3d} (mean n {n[idx].mean():5.1f}): {dt / (args.sims * args.steps) * 1e6:6.2f} us per "
              f"body-step, {args.sims * args.steps / dt / 1e3:6.0f}k body-steps/s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = sw.common_args()
    for name in ("check", "bench"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--archive", default="coevo_water_cma2.pkl")
        p.add_argument("--count", type=int, default=32)
        p.add_argument("--steps", type=int, default=1000)
        p.add_argument("--sims", type=int, default=128)
    args = ap.parse_args()
    sw.configure(args)
    {"check": run_check, "bench": run_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
