"""Neural graph cellular automaton that drives growth and colours the surface.

Every vertex carries a state S in [-1, 1]^C. Each update a vertex perceives its
own state, the mean of its neighbours' states and local cues (geodesic
distance, mean curvature, height, normal tilt, boundary, static noise, light
exposure with self-shading), and
its individual's MLP pulls the state toward a new value:

    S ← S + dt·(tanh(W2·relu(W1·p + b1) + b2) − S),  then per-channel diffusion

Channels 0–2 drive the sheet (growth rate, anisotropy along ∇distance,
spontaneous curvature), 3–5 are surface RGB, 6 steers growth toward the light
(phototropism). 7 is a polarity field: its intrinsic gradient is a growth axis,
with signed strength from 8 (auxin-like; a growth zone elongates along its own
axis). 9 sets local bending stiffness (ribs, veins), 10 spontaneous curvature
across the polarity axis (tubes, funnels, scrolls), 11 modulates accretion,
12 perforates: faces where it peaks above its surroundings are cut (lace, fenestrae, sea-fan
mesh), and the new rims count as boundary. 13–16 are surface material for the
renderer: iridescence, thin-film thickness, translucency, gloss; 17 is surface
height, which the renderer turns into bump shading. The rest are hidden; they diffuse at a range of fixed rates so the CA has
short- and long-range signals to pattern with. Cues also include ambient
(hemisphere) exposure. Growth, polarity and the hidden channels are also
perceived through spherical-harmonic filters (degrees 1 and 2, world frame, as
in MeshNCA): their gradient and anisotropy around the vertex, so the CA can
tell up from down and lay patterns along a direction. The `sh_local` gene measures
them in a frame on the surface instead (normal and ∇distance), so patterns follow
the form's own axes. As in MeshNCA, each vertex updates with probability
`fire` per step (asynchronous, symmetry-breaking); fire = 1 is synchronous. New vertices from edge splits inherit the endpoint
mean like any vertex field.

Genome = physics genes in [0,1] followed by raw MLP weights (unbounded).
"""

import math

import numpy as np
import torch

from . import genome as PG
from .sim import Params

C = 21
HID = 24
N_CUE = 8
N_SH = 8  # real spherical harmonics, degrees 1 and 2
CH_GROW, CH_ANISO, CH_CURL, CH_TROPIC = 0, 1, 2, 6
CH_POL, CH_PSTR, CH_STIFF, CH_ACURL, CH_ACC, CH_HOLE = 7, 8, 9, 10, 11, 12
CH_MAT = slice(13, 17)  # iridescence, film thickness, translucency, gloss
CH_BUMP = 17  # surface height for bump shading
N_OUT = 18  # channels with a fixed meaning; the rest are hidden
CH_RGB = slice(3, 6)
# The polarity channel diffuses faster so its gradient is smooth enough to steer by.
DIFF = [0.05] * 7 + [0.2] + [0.05] * 10 + list(np.linspace(0.0, 0.5, C - N_OUT))
SH_CH = [CH_GROW, CH_POL] + list(range(N_OUT, C))
K_SH = len(SH_CH) * N_SH
P_IN = 2 * C + N_CUE + K_SH
SHAPES = [("W1", (P_IN, HID)), ("b1", (HID,)), ("W2", (HID, C)), ("b2", (C,))]
N_W = sum(math.prod(s) for _, s in SHAPES)


class NCA:
    CH_GROW, CH_ANISO, CH_CURL, CH_TROPIC = CH_GROW, CH_ANISO, CH_CURL, CH_TROPIC
    CH_POL, CH_PSTR, CH_STIFF, CH_ACURL, CH_ACC, CH_HOLE = CH_POL, CH_PSTR, CH_STIFF, CH_ACURL, CH_ACC, CH_HOLE

    def __init__(self, weights, dt, gain, fire=None, sh_local=None, device="cuda"):
        """weights (B, N_W) raw genome values; dt, gain (B,) per individual; fire (B,)
        update probability per step; sh_local (B,) bool, SH in the surface frame."""
        w = torch.as_tensor(np.asarray(weights), dtype=torch.float32, device=device)
        gain = torch.as_tensor(np.asarray(gain), dtype=torch.float32, device=device)
        self.dt = torch.as_tensor(np.asarray(dt), dtype=torch.float32, device=device)
        self.B = w.shape[0]
        k = 0
        p = {}
        for name, shp in SHAPES:
            n = math.prod(shp)
            p[name] = w[:, k:k + n].reshape(self.B, *shp)
            k += n
        g = gain[:, None, None]
        self.W1 = p["W1"] * g / math.sqrt(P_IN)
        self.W2 = p["W2"] * g / math.sqrt(HID)
        self.b1 = p["b1"] * 0.5
        self.b2 = p["b2"] * 0.5
        self.diff = torch.tensor(DIFF, dtype=torch.float32, device=device)
        ones = np.ones(self.B)
        self.fire = torch.as_tensor(np.asarray(ones if fire is None else fire), dtype=torch.float32, device=device)
        self.sh_local = torch.as_tensor(np.asarray(ones * 0 if sh_local is None else sh_local), dtype=torch.bool,
                                        device=device)

    def init_state(self, sim):
        s = 0.1 * torch.randn(sim.pos.shape[0], C, device=sim.dev, generator=sim.gen)
        s[:, :N_OUT] = 0.0
        return s

    def cues(self, sim):
        ind = sim.ind
        n = torch.nn.functional.normalize(sim.vertex_normals(), dim=1)
        z = sim.pos[:, 2] - sim.centers[ind, 2]
        return torch.stack([
            torch.exp(-sim.dist / 6.0),
            sim.mean_curvature().clamp(-2, 2) * 0.5,
            torch.tanh(z / 10.0),
            n[:, 2],
            sim.topo.boundary.float(),
            sim.vf["noise"],
            sim.light(),
            sim.ambient(),
        ], 1)

    @staticmethod
    def _harm(d):
        """(.., 8) degree-1 and degree-2 real harmonics of unit vectors d (.., 3)."""
        x, y, z = d.unbind(-1)
        r3 = math.sqrt(3)
        return torch.stack([x, y, z, r3 * x * y, r3 * y * z, 0.5 * (3 * z * z - 1), r3 * x * z,
                            0.5 * r3 * (x * x - y * y)], -1)

    def _frames(self, sim):
        """(N, 3, 3) rows = the frame each vertex measures directions in: world axes, or
        for sh_local individuals (∇distance, normal × it, normal) on the surface."""
        N = sim.pos.shape[0]
        R = torch.eye(3, device=sim.dev).expand(N, 3, 3).clone()
        loc = self.sh_local[sim.ind]
        if not loc.any():
            return R
        e = sim.topo.edges
        n = torch.nn.functional.normalize(sim.vertex_normals(), dim=1)
        dx = sim.pos[e[:, 1]] - sim.pos[e[:, 0]]
        dd = (sim.dist[e[:, 1]] - sim.dist[e[:, 0]])[:, None] * dx / (dx * dx).sum(1, keepdim=True).clamp(min=1e-9)
        g = torch.zeros(N, 3, device=sim.dev).index_add_(0, e[:, 0], dd).index_add_(0, e[:, 1], dd)
        # Where distance is flat, fall back to world up, then x, projected onto the surface.
        g = g + 1e-3 * torch.tensor([0.0, 0.0, 1.0], device=sim.dev) + 1e-5 * torch.tensor([1.0, 0.0, 0.0], device=sim.dev)
        t1 = torch.nn.functional.normalize(g - (g * n).sum(1, keepdim=True) * n, dim=1)
        t2 = torch.cross(n, t1, dim=1)
        R[loc] = torch.stack([t1, t2, n], 1)[loc]
        return R

    def _sh_basis(self, sim):
        """(E, 8) harmonics of each edge's direction as seen from each end, in that end's frame."""
        e = sim.topo.edges
        d = torch.nn.functional.normalize(sim.pos[e[:, 1]] - sim.pos[e[:, 0]], dim=1)
        R = self._frames(sim)
        return self._harm(torch.einsum("eij,ej->ei", R[e[:, 0]], d)), \
            self._harm(torch.einsum("eij,ej->ei", R[e[:, 1]], -d))

    @staticmethod
    def _sh(sim, Y, S):
        """Mean over neighbours of Y(direction) * (s_j - s_i) for the SH channels, taken
        on the neighbour-smoothed state so mesh-scale noise isn't amplified."""
        e = sim.topo.edges
        Ya, Yb = Y
        D = S[e[:, 1]][:, SH_CH] - S[e[:, 0]][:, SH_CH]
        out = torch.zeros(S.shape[0], K_SH, device=S.device)
        out.index_add_(0, e[:, 0], (Ya[:, :, None] * D[:, None]).reshape(len(e), K_SH))
        out.index_add_(0, e[:, 1], (Yb[:, :, None] * -D[:, None]).reshape(len(e), K_SH))
        # Neighbour differences are ~30x smaller than the states; bring them to the same scale.
        return torch.tanh(20.0 * out / sim.topo.valence.clamp(min=1).float()[:, None])

    def _mlp(self, x, ind):
        order = torch.argsort(ind, stable=True)
        row = ind[order]
        cnt = torch.bincount(ind, minlength=self.B)
        start = torch.cumsum(cnt, 0) - cnt
        col = torch.arange(len(ind), device=x.device) - start[row]
        X = x.new_zeros(self.B, int(cnt.max()), x.shape[1])
        X[row, col] = x[order]
        h = torch.relu(torch.baddbmm(self.b1[:, None], X, self.W1))
        y = torch.tanh(torch.baddbmm(self.b2[:, None], h, self.W2))
        out = torch.empty(len(ind), C, device=x.device)
        out[order] = y[row, col]
        return out

    def step(self, sim, n, live=False):
        """`live`: run for every individual, grown or not (film.py animates settled forms)."""
        if n <= 0:
            return
        ind = sim.ind
        cue = self.cues(sim)
        rate = (self.dt * (1.0 if live else sim.active))[ind][:, None]
        fire = self.fire[ind][:, None]
        S = sim.vf["nca"]
        Y = self._sh_basis(sim)
        for _ in range(n):
            nb = S + sim._graph_lap(S)
            y = self._mlp(torch.cat([S, nb, cue, self._sh(sim, Y, nb)], 1), ind)
            go = torch.rand(fire.shape, device=S.device, generator=sim.gen) < fire
            S = S + rate * go * (y - S)
            S = (S + (rate > 0) * self.diff * sim._graph_lap(S)).clamp(-1, 1)
        sim.vf["nca"] = S

    @staticmethod
    def color(S):
        # Kept off pure black/white/primaries so surfaces read as pigment, not plastic.
        return 0.5 + 0.35 * S[:, CH_RGB]

    @staticmethod
    def material(S):
        """(N, 5) iridescence, film thickness, translucency, gloss, height in [0, 1]. The
        effects start above a channel value of 0.2, so a quiet channel is matte;
        height 0.5 is flat."""
        m = ((S[:, CH_MAT] - 0.2) / 0.8).clamp(0, 1)
        m[:, 1] = 0.5 * (S[:, CH_MAT.start + 1] + 1)
        return torch.cat([m, 0.5 * (S[:, CH_BUMP:CH_BUMP + 1] + 1)], 1)


# ── genome ──

GENES = [
    "seed", "k_bend", "curl", "g_base", "g_nca", "aniso", "nca_aniso", "nca_curl",
    "foundation", "pressure", "gravity", "k_stretch", "rate", "nca_dt", "gain",
    "nca_tropic", "g_light",
    "nca_polar", "nca_stiff", "nca_acurl", "g_smooth", "plastic",
    "accrete", "acc_thresh", "acc_curv", "nca_acc", "g_ambient",
    "acc_gauss", "acc_bias", "acc_smooth", "acc_light", "acc_tension", "g_curv",
    "acc_field", "acc_up", "acc_solid", "nca_hole", "nca_fire", "sh_local",
]
D_PHYS = len(GENES)
D = D_PHYS + N_W
SEEDS = PG.SEEDS + ["stalk"]
SEED_RADIUS = dict(PG.SEED_RADIUS, stalk=6)
_lin, _log, _gate = PG._lin, PG._log, PG._gate


def decode(g):
    """-> (seed kind, Params, dict(weights, dt, gain))."""
    g = np.asarray(g, dtype=float)
    x = dict(zip(GENES, np.clip(g[:D_PHYS], 0, 1)))
    seed = SEEDS[min(int(x["seed"] * len(SEEDS)), len(SEEDS) - 1)]
    rate = _log(x["rate"], 0.5, 2.0)
    p = Params(
        k_stretch=_log(x["k_stretch"], 8, 40),
        k_bend=_log(x["k_bend"], 0.5, 20),
        curl=_gate(x["curl"], 0.3, lambda u: _lin(u, -0.12, 0.12)),
        g_base=rate * _lin(x["g_base"], 0.02, 0.1),
        g_nca=rate * _lin(x["g_nca"], 0.1, 0.9),
        aniso=_lin(x["aniso"], -0.6, 0.8),
        nca_aniso=_lin(x["nca_aniso"], 0.0, 0.8),
        nca_curl=_gate(x["nca_curl"], 0.3, lambda u: _lin(u, 0.0, 0.15)),
        foundation_k=_gate(x["foundation"], 0.4, lambda u: _log(u, 0.005, 0.1)) if seed == "sphere" else 0.0,
        foundation_r=float(SEED_RADIUS["sphere"]),
        pressure=_gate(x["pressure"], 0.3, lambda u: _lin(u, -0.01, 0.04)),
        gravity=_gate(x["gravity"], 0.5, lambda u: _lin(u, 0.0, 0.05)),
        nca_tropic=_gate(x["nca_tropic"], 0.3, lambda u: _lin(u, 0.2, 0.8)),
        g_light=rate * _gate(x["g_light"], 0.4, lambda u: _lin(u, 0.05, 0.4)),
        ground=1.0 if seed in ("hemisphere", "stalk") else 0.0,
        nca_polar=_gate(x["nca_polar"], 0.3, lambda u: _lin(u, 0.2, 0.9)),
        nca_stiff=_gate(x["nca_stiff"], 0.3, lambda u: _lin(u, 0.3, 2.3)),
        nca_acurl=_gate(x["nca_acurl"], 0.3, lambda u: _lin(u, 0.03, 0.3)),
        g_smooth=_lin(x["g_smooth"], 0.0, 10.0),
        plastic=_gate(x["plastic"], 0.5, lambda u: _log(u, 1.0, 20.0)),
        accrete=_gate(x["accrete"], 0.5, lambda u: _log(u, 2.0, 25.0)),
        acc_thresh=_lin(x["acc_thresh"], 0.0, 0.75),
        acc_curv=_lin(x["acc_curv"], -2.0, 6.0),
        acc_gauss=_lin(x["acc_gauss"], 0.0, 20.0),
        acc_bias=_lin(x["acc_bias"], -1.5, 1.0),
        acc_smooth=_log(x["acc_smooth"], 2.0, 20.0),
        acc_light=_lin(x["acc_light"], 0.0, 1.0),
        acc_tension=_log(x["acc_tension"], 0.02, 0.2),
        nca_acc=_lin(x["nca_acc"], 0.0, 1.5),
        g_ambient=rate * _gate(x["g_ambient"], 0.4, lambda u: _lin(u, 0.05, 0.4)),
        # Area growth at convex tips widens them until they split.
        g_curv=rate * _gate(x["g_curv"], 0.4, lambda u: _lin(u, 0.02, 0.2)),
        # Diffusion-limited supply (dgrow.field): screened gaps let branches diverge.
        acc_field=_gate(x["acc_field"], 0.3, lambda u: _lin(u, 0.3, 1.0)),
        acc_up=_lin(x["acc_up"], 0.0, 1.0),
        acc_solid=_lin(x["acc_solid"], 0.0, 1.0),
        # Perforation is destructive, so it is off in most of the gene's range.
        nca_hole=_gate(x["nca_hole"], 0.6, lambda u: _lin(u, 0.15, 0.6)),
    )
    ctrl = dict(weights=g[D_PHYS:], dt=_log(x["nca_dt"], 0.05, 0.5), gain=_lin(x["gain"], 0.8, 2.5),
                # Asynchronous updates; off (synchronous) in the lower part of the range.
                fire=1.0 - _gate(x["nca_fire"], 0.4, lambda u: _lin(u, 0.0, 0.5)),
                sh_local=x["sh_local"] > 0.5)
    return seed, p, ctrl


def _n_w(c, k):
    return (2 * c + N_CUE + k) * HID + HID + HID * c + c


# Past (channels, SH inputs) layouts, newest first, and where each next one inserted
# its new fixed channels: v4 added materials at 13, v5 SH inputs, v6 height at 17.
LAYOUTS = [(C, K_SH), (20, K_SH), (20, 0), (16, 0)]
NEW_AT = {(20, K_SH): 17, (20, 0): None, (16, 0): 13}


def _step_up(w, c, k):
    """Weights in layout (c, k) -> the next layout: new channels / SH inputs zero."""
    p = 2 * c + N_CUE + k
    W1 = w[:p * HID].reshape(p, HID)
    b1, W2, b2 = w[p * HID:p * HID + HID], w[p * HID + HID:p * HID + HID + HID * c].reshape(HID, c), \
        w[p * HID + HID + HID * c:]
    o = NEW_AT[(c, k)]
    a = {13: 4, 17: 1}.get(o, 0)
    pad = lambda x, axis: np.insert(x, [o] * a, 0.0, axis=axis) if a else x
    k2 = K_SH if o is None else k
    W1 = np.concatenate([pad(W1[:c], 0), pad(W1[c:2 * c], 0), W1[2 * c:2 * c + N_CUE], W1[2 * c + N_CUE:],
                         np.zeros((k2 - k, HID))])
    # NCA scales W1 by 1/sqrt(inputs); keep the effective weights.
    W1 = W1 * math.sqrt(len(W1) / p)
    return np.concatenate([W1.ravel(), b1, pad(W2, 1).ravel(), pad(b2, 0)]), c + a, k2


def upgrade(g):
    """Genome from before later physics genes, state channels or SH inputs were
    added: new genes off (0); weights into and out of new channels and from the
    SH inputs 0, so the old form grows, and renders, unchanged."""
    g = np.asarray(g, dtype=float)
    c, k = next(ck for ck in LAYOUTS if 0 < len(g) - _n_w(*ck) <= D_PHYS)
    d = len(g) - _n_w(c, k)
    w = g[d:]
    while (c, k) != (C, K_SH):
        w, c, k = _step_up(w, c, k)
    return np.concatenate([g[:d], np.zeros(D_PHYS - d), w])


def random(n, rng):
    return np.concatenate([rng.random((n, D_PHYS)), rng.normal(0, 1, (n, N_W))], 1)


def _clip(c):
    c[:, :D_PHYS] = np.clip(c[:, :D_PHYS], 0, 1)
    return c


def mutate(parents, rng, sigma=0.08, p_reset=0.05, w_scale=2.5):
    c = parents.copy()
    c[:, :D_PHYS] = PG.mutate(parents[:, :D_PHYS], rng, sigma, p_reset)
    c[:, D_PHYS:] += rng.normal(0, sigma * w_scale, (len(c), N_W))
    return c


def iso_dd(a, b, rng, s1=0.02, s2=0.2):
    return _clip(a + rng.normal(0, s1, a.shape) + s2 * rng.normal(0, 1, (len(a), 1)) * (b - a))
