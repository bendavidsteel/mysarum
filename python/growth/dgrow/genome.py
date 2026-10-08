"""Genome in [0,1]^D -> (seed shape, Params). One genome spans every growth mechanism."""

import math

import numpy as np

from .sim import Params

SEEDS = ["disc", "sphere", "hemisphere"]
SEED_RADIUS = {"disc": 10, "sphere": 12, "hemisphere": 9}


def _lin(x, lo, hi):
    return lo + (hi - lo) * x


def _log(x, lo, hi):
    return math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * x)


def _gate(x, thresh, f):
    # Lower `thresh` of the gene's range switches the mechanism off entirely.
    return 0.0 if x < thresh else f((x - thresh) / (1 - thresh))


GENES = [
    "seed", "k_bend", "curl", "g_base", "g_edge", "edge_lambda", "aniso", "g_curv",
    "g_morph", "gs_feed", "gs_kill", "g_height", "foundation", "pressure", "gravity",
    "g_noise", "k_stretch", "rate",
]
D = len(GENES)


def decode(g):
    g = np.clip(np.asarray(g, dtype=float), 0, 1)
    x = dict(zip(GENES, g))
    seed = SEEDS[min(int(x["seed"] * len(SEEDS)), len(SEEDS) - 1)]
    rate = _log(x["rate"], 0.5, 2.0)
    p = Params(
        k_stretch=_log(x["k_stretch"], 8, 40),
        k_bend=_log(x["k_bend"], 0.5, 20),
        curl=_gate(x["curl"], 0.3, lambda u: _lin(u, -0.12, 0.12)),
        g_base=rate * _lin(x["g_base"], 0.0, 0.3),
        g_edge=rate * _lin(x["g_edge"], 0.0, 0.8),
        edge_lambda=_log(x["edge_lambda"], 1.0, 12.0),
        aniso=_lin(x["aniso"], -0.6, 0.8),
        g_curv=rate * _gate(x["g_curv"], 0.3, lambda u: _lin(u, -0.3, 0.8)),
        g_morph=rate * _gate(x["g_morph"], 0.4, lambda u: _lin(u, 0.1, 0.6)),
        gs_feed=_lin(x["gs_feed"], 0.022, 0.055),
        gs_kill=_lin(x["gs_kill"], 0.057, 0.065),
        g_height=rate * _gate(x["g_height"], 0.3, lambda u: _lin(u, -0.2, 0.4)),
        foundation_k=_gate(x["foundation"], 0.4, lambda u: _log(u, 0.005, 0.1)) if seed == "sphere" else 0.0,
        foundation_r=float(SEED_RADIUS["sphere"]),
        pressure=_gate(x["pressure"], 0.3, lambda u: _lin(u, -0.01, 0.04)),
        gravity=_gate(x["gravity"], 0.5, lambda u: _lin(u, 0.0, 0.05)),
        g_noise=_lin(x["g_noise"], 0.0, 0.8),
        ground=1.0 if seed == "hemisphere" else 0.0,
    )
    # Guarantee some net growth so no evaluation is wasted on a static seed.
    if p.g_base + p.g_edge < 0.08 * rate:
        p.g_base += 0.08 * rate
    return seed, p


def random(n, rng):
    return rng.random((n, D))


def mutate(parents, rng, sigma=0.08, p_reset=0.05):
    c = parents + rng.normal(0, sigma, parents.shape)
    reset = rng.random(parents.shape) < p_reset
    c[reset] = rng.random(reset.sum())
    return np.clip(c, 0, 1)


def iso_dd(a, b, rng, s1=0.02, s2=0.2):
    """Iso+line crossover (Vassiliades & Mouret 2018)."""
    return np.clip(a + rng.normal(0, s1, a.shape) + s2 * rng.normal(0, 1, (len(a), 1)) * (b - a), 0, 1)
