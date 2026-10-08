"""Seed meshes and hand-built growth presets modelled on real organisms."""

import numpy as np

from . import mesh as M
from .sim import Params


def seed_mesh(kind, radius):
    """kind: disc | sphere | hemisphere. Returns dict(pos, faces, pinned, source)."""
    if kind == "disc":
        v, f = M.make_disc(radius)
        # Organizer at the margin (boundary distance); no pins.
        return dict(pos=v, faces=f)
    if kind == "sphere":
        v, f = M.make_icosphere(radius)
        src = np.zeros(len(v))
        src[np.argmax(v[:, 2])] = 1  # apical pole
        return dict(pos=v, faces=f, source=src)
    if kind == "hemisphere":
        v, f = M.make_hemisphere(radius)
        pin = v[:, 2] < 0.05 * radius
        src = np.zeros(len(v))
        src[np.argmax(v[:, 2])] = 1
        return dict(pos=v, faces=f, pinned=pin.astype(float), source=src)
    if kind == "stalk":
        # Short closed-top cylinder rising from a pinned rim: a polyp/bud.
        v, f = M.make_hemisphere(radius)
        v = v.copy()
        v[:, 2] *= 2.5
        pin = v[:, 2] < 0.05 * radius
        src = np.zeros(len(v))
        src[np.argmax(v[:, 2])] = 1
        return dict(pos=v, faces=f, pinned=pin.astype(float), source=src)
    raise ValueError(kind)


# name -> (seed kind, radius, Params kwargs, growth ticks, dt per tick)
PRESETS = {
    # Margin growth decaying into the blade (Liang & Mahadevan 2009; Floraform).
    "lettuce": ("disc", 10, dict(g_edge=0.5, edge_lambda=4.0, k_bend=3.0), 900, 0.01),
    "kale": ("disc", 10, dict(g_edge=0.8, edge_lambda=1.5, k_bend=2.0), 700, 0.01),
    # Broad uniform growth with a stiff blade: few large saddle waves.
    "saddle": ("disc", 10, dict(g_base=0.15, g_edge=0.25, edge_lambda=8.0, k_bend=12.0), 900, 0.01),
    # Circumferential-only margin growth -> hyperbolic coral / frilly petal.
    "hyperbolic": ("disc", 10, dict(g_edge=0.4, edge_lambda=3.0, aniso=-0.5, k_bend=3.0), 900, 0.01),
    # Radial-only growth with natural curl -> trumpet / funnel.
    "trumpet": ("disc", 10, dict(g_edge=0.4, edge_lambda=6.0, aniso=0.8, curl=0.06, k_bend=4.0), 900, 0.01),
    # Cortex on an elastic core (Tallinen 2016): uniform growth, radial tether.
    "brain": ("sphere", 12, dict(g_base=0.25, foundation_k=0.05, foundation_r=12.0, k_bend=4.0), 600, 0.01),
    # Turing spots drive local growth on a shell -> knobbly/tuberculate.
    "turing": ("sphere", 12, dict(g_base=0.05, g_morph=0.4, k_bend=4.0, pressure=0.02,
                                  gs_feed=0.035, gs_kill=0.065), 700, 0.01),
    # Apical growth + curvature feedback on a pinned dome -> branching coral.
    "coral": ("hemisphere", 8, dict(g_base=0.02, g_curv=0.5, g_height=0.3, k_bend=2.0,
                                    gravity=0.0, ground=1.0), 900, 0.01),
}


def preset(name, **over):
    kind, r, kw, ticks, dt = PRESETS[name]
    kw = dict(kw, **over)
    return seed_mesh(kind, r), Params(**kw), ticks, dt
