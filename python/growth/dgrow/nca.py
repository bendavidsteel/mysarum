"""Neural graph cellular automaton that drives growth and colours the surface.

Every vertex carries a state S in [-1, 1]^C. Each update a vertex perceives its
own state, the mean of its neighbours' states and local cues (geodesic
distance, mean curvature, height, normal tilt, boundary, static noise, light
exposure with self-shading), and
its individual's MLP pulls the state toward a new value:

    S ← S + dt·(tanh(W2·relu(W1·p + b1) + b2) − S),  then per-channel diffusion

Channels 0–2 drive the sheet (growth rate, anisotropy along ∇distance,
spontaneous curvature), 3–5 are surface RGB, 6 steers growth toward the light
(phototropism), the rest are hidden. Hidden channels diffuse at a range
of fixed rates so the CA has short- and long-range signals to pattern with.
New vertices from edge splits inherit the endpoint mean like any vertex field.

Genome = physics genes in [0,1] followed by raw MLP weights (unbounded).
"""

import math

import numpy as np
import torch

from . import genome as PG
from .sim import Params

C = 12
HID = 24
N_CUE = 7
P_IN = 2 * C + N_CUE
CH_GROW, CH_ANISO, CH_CURL, CH_TROPIC = 0, 1, 2, 6
CH_RGB = slice(3, 6)
DIFF = [0.05] * 7 + list(np.linspace(0.0, 0.5, C - 7))
SHAPES = [("W1", (P_IN, HID)), ("b1", (HID,)), ("W2", (HID, C)), ("b2", (C,))]
N_W = sum(math.prod(s) for _, s in SHAPES)


class NCA:
    CH_GROW, CH_ANISO, CH_CURL, CH_TROPIC = CH_GROW, CH_ANISO, CH_CURL, CH_TROPIC

    def __init__(self, weights, dt, gain, device="cuda"):
        """weights (B, N_W) raw genome values; dt, gain (B,) per individual."""
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

    def init_state(self, sim):
        s = 0.1 * torch.randn(sim.pos.shape[0], C, device=sim.dev, generator=sim.gen)
        s[:, :7] = 0.0
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
        ], 1)

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

    def step(self, sim, n):
        if n <= 0:
            return
        ind = sim.ind
        cue = self.cues(sim)
        rate = (self.dt * sim.active)[ind][:, None]
        S = sim.vf["nca"]
        for _ in range(n):
            nb = S + sim._graph_lap(S)
            y = self._mlp(torch.cat([S, nb, cue], 1), ind)
            S = S + rate * (y - S)
            S = (S + (rate > 0) * self.diff * sim._graph_lap(S)).clamp(-1, 1)
        sim.vf["nca"] = S

    @staticmethod
    def color(S):
        # Kept off pure black/white/primaries so surfaces read as pigment, not plastic.
        return 0.5 + 0.35 * S[:, CH_RGB]


# ── genome ──

GENES = [
    "seed", "k_bend", "curl", "g_base", "g_nca", "aniso", "nca_aniso", "nca_curl",
    "foundation", "pressure", "gravity", "k_stretch", "rate", "nca_dt", "gain",
    "nca_tropic", "g_light",
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
    )
    ctrl = dict(weights=g[D_PHYS:], dt=_log(x["nca_dt"], 0.05, 0.5), gain=_lin(x["gain"], 0.8, 2.5))
    return seed, p, ctrl


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
