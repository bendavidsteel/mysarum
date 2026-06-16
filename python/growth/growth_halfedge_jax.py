"""
growth_halfedge_jax.py — Differential growth on a half-edge mesh in JAX.

Combines:
- Half-edge data structure (2-manifold with boundary)
- Intrinsic triangulation (per-half-edge intrinsic rest lengths decoupled from 3D embedding)
- XPBD constraint projection (springs, dihedral bending, collision)
- Growth control via either:
    * "lenia"   — Chebyshev polynomial graph CA, scalar channel
    * "nca"     — MeshNCA-style MLP over per-vertex feature vectors
- nvdiffrast rendering
"""

import collections
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pygame
import torch

import nvdiffrast.torch as dr


# ── Constants ────────────────────────────────────────────────────────────────

EPSILON = 1e-6
# JIT-static array size. Per-frame cost scales with this, NOT the active
# vertex count (every kernel runs over the full padded arrays), so sizing it
# just above a sweep's real peak vertex count is the single biggest speed
# lever. Override per run with GROWTH_MAX_VERTICES (e.g. 5000 for a sweep that
# tops out ~3700 verts ≈ 2.4x faster than the 12000 default). Keep 12000 for
# open-ended runs on the 4 GB box.
MAX_VERTICES = int(os.environ.get("GROWTH_MAX_VERTICES", "12000"))
MAX_HALF_EDGES = MAX_VERTICES * 6
MAX_FACES = MAX_VERTICES * 2
MAX_CHEB_ORDER = 16
MAX_FAN_SIZE = 20

# Spatial hash for repulsion / collision
NUM_HASH_BUCKETS = 12289
SLOTS_PER_CELL = 12

# World bounds for enforce_boundary
WORLD_MIN = (0.0, 0.0, 0.0)
WORLD_MAX = (800.0, 800.0, 200.0)


# ── State ────────────────────────────────────────────────────────────────────

MeshState = collections.namedtuple('MeshState', [
    'vertex_idx',
    'vertex_half_edge',
    'vertex_pos',
    'vertex_state',
    'half_edge_idx',
    'half_edge_twin',
    'half_edge_dest',
    'half_edge_face',
    'half_edge_next',
    'half_edge_prev',
    'half_edge_intrinsic_len',
    'face_idx',
    'face_half_edge',
])


MeshParams = collections.namedtuple('MeshParams', [
    # XPBD / physics
    'spring_len',
    'compliance',
    'bending_compliance',
    # SOR under-relaxation applied to spring + bending corrections each Jacobi
    # iteration (matches the Rust port). <1 damps overshoot from the full
    # Jacobi step; 1.0 = no damping (old Python behaviour, prone to buckling).
    'relaxation',
    'repulsion_dist',
    'repulsion_strength',
    'bulge_strength',
    'stiffness',
    'damping',
    'dt',
    'max_edge_len',
    # Lenia CA (used when growth_mode == "lenia")
    'kernel_mu',
    'kernel_sigma',
    'growth_mu',
    'growth_sigma',
    # Common growth
    'growth_rate',
    'state_dt',
    # MeshNCA (used when growth_mode == "nca")
    'vertex_state_mlp_params',
    # General
    'num_colour_channels',
    # Phototropic / nature-inspired growth (growth_mode == "phototropic")
    #   ch0 = tissue (drives growth), ch1 = light (recomputed each step),
    #   ch2 = nutrient (diffuses from ground), optional ch3+ extra resources
    'light_dir',
    'light_pos',
    'light_decay_dist',
    'light_ambient',
    'gravity_strength',
    'tissue_decay',
    'tissue_saturation',
    'resource_diffusion',
    'resource_decay',
    'ground_z',
    'ground_source_value',
    'ground_pin_strength',
    # Anisotropic growth (applies across all growth modes; strength 0 = off)
    #   anisotropy_dir    arbitrary preferred-growth axis (need not be unit;
    #                     normalized internally). Unlike the Rust version's
    #                     three cardinal-axis choices, this is a free vec3.
    #   anisotropy_strength  in [0, 1]: 0 = isotropic, 1 = grow only along the
    #                     axis (edge growth scaled by |cos| of edge vs axis).
    'anisotropy_dir',
    'anisotropy_strength',
])


def default_params(**overrides):
    defaults = dict(
        spring_len=30.0,
        compliance=0.0,
        bending_compliance=0.001,
        relaxation=0.7,
        repulsion_dist=80.0,
        repulsion_strength=4.0,
        bulge_strength=5.0,
        stiffness=400.0,
        damping=10.0,
        dt=0.02,
        max_edge_len=50.0,
        kernel_mu=4.0,
        kernel_sigma=1.0,
        growth_mu=0.5,
        growth_sigma=0.2,
        growth_rate=1.0,
        state_dt=0.02,
        vertex_state_mlp_params={},
        num_colour_channels=1,
        light_dir=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        light_pos=jnp.array([400.0, 400.0, 800.0], dtype=jnp.float32),
        light_decay_dist=1e6,
        light_ambient=0.05,
        gravity_strength=0.0,
        tissue_decay=0.05,
        tissue_saturation=2.0,
        resource_diffusion=0.5,
        resource_decay=0.02,
        ground_z=0.0,
        ground_source_value=1.0,
        ground_pin_strength=0.15,
        anisotropy_dir=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        anisotropy_strength=0.0,
    )
    defaults.update(overrides)
    return MeshParams(**defaults)


def create_initial_state(state_dims=1, key=None):
    """Create empty mesh state with multi-channel vertex_state.

    state_dims=1 is typical for Lenia mode; NCA uses larger state_dims.
    """
    if key is None:
        vstate = jnp.zeros((MAX_VERTICES, state_dims), jnp.float32)
    else:
        vstate = jax.random.normal(key, (MAX_VERTICES, state_dims), jnp.float32)

    return MeshState(
        vertex_idx=jnp.full(MAX_VERTICES, -1, jnp.int32),
        vertex_half_edge=jnp.full(MAX_VERTICES, -1, jnp.int32),
        vertex_pos=jnp.zeros((MAX_VERTICES, 3), jnp.float32),
        vertex_state=vstate,
        half_edge_idx=jnp.full(MAX_HALF_EDGES, -1, jnp.int32),
        half_edge_twin=jnp.full(MAX_HALF_EDGES, -1, jnp.int32),
        half_edge_dest=jnp.full(MAX_HALF_EDGES, -1, jnp.int32),
        half_edge_face=jnp.full(MAX_HALF_EDGES, -1, jnp.int32),
        half_edge_next=jnp.full(MAX_HALF_EDGES, -1, jnp.int32),
        half_edge_prev=jnp.full(MAX_HALF_EDGES, -1, jnp.int32),
        half_edge_intrinsic_len=jnp.zeros(MAX_HALF_EDGES, jnp.float32),
        face_idx=jnp.full(MAX_FACES, -1, jnp.int32),
        face_half_edge=jnp.full(MAX_FACES, -1, jnp.int32),
    )


def make_mlp_params(key, state_dims, hidden_state_dims=32, num_mlp_layers=2, scale=0.1):
    """Build MLP weights for MeshNCA."""
    num_sph = 4  # spherical harmonics up to degree 1
    keys = jax.random.split(key, num_mlp_layers)
    params = {}
    for i in range(num_mlp_layers):
        if i == 0:
            in_dim = state_dims + num_sph * state_dims
        else:
            in_dim = hidden_state_dims
        out_dim = state_dims if i == num_mlp_layers - 1 else hidden_state_dims
        w = scale * jax.random.normal(keys[i], (in_dim, out_dim))
        b = jnp.zeros((out_dim,))
        params[f'layer{i+1}'] = (w, b)
    return params


# ── Mesh Builders ────────────────────────────────────────────────────────────

def _build_half_edges(positions, faces):
    """Build half-edge connectivity arrays from positions + triangle faces.

    Returns numpy arrays sized to MAX_HALF_EDGES, plus n_he, n_verts, n_faces.
    Pure CPU/numpy. Computes intrinsic edge lengths from extrinsic positions.
    """
    n_verts = len(positions)
    n_faces_input = len(faces)
    pos_np = np.array(positions, dtype=np.float32)

    edge_dict = {}
    he_list = []

    for fi, (v0, v1, v2) in enumerate(faces):
        he_start = len(he_list)
        for j, (src, dst) in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
            he_list.append({
                'dest': int(dst), 'face': fi, 'twin': -1,
                'next': he_start + (j + 1) % 3,
                'prev': he_start + (j + 2) % 3,
            })
            edge_dict[(int(src), int(dst))] = he_start + j

    boundary_edges = []
    for (src, dst), he in edge_dict.items():
        twin_key = (dst, src)
        if twin_key in edge_dict:
            he_list[he]['twin'] = edge_dict[twin_key]
        else:
            bhe = len(he_list)
            he_list.append({
                'dest': src, 'face': -1, 'twin': he,
                'next': -1, 'prev': -1,
            })
            he_list[he]['twin'] = bhe
            boundary_edges.append((dst, src, bhe))

    bnd_dict = {(s, d): i for s, d, i in boundary_edges}
    for src, dst, bhe in boundary_edges:
        for (ns, _nd), nbhe in bnd_dict.items():
            if ns == dst:
                he_list[bhe]['next'] = nbhe
                he_list[nbhe]['prev'] = bhe
                break

    n_he = len(he_list)
    assert n_he <= MAX_HALF_EDGES, f"Too many half-edges: {n_he} > {MAX_HALF_EDGES}"
    assert n_verts <= MAX_VERTICES, f"Too many vertices: {n_verts} > {MAX_VERTICES}"
    assert n_faces_input <= MAX_FACES, f"Too many faces: {n_faces_input} > {MAX_FACES}"

    he_dest = np.full(MAX_HALF_EDGES, -1, np.int32)
    he_face = np.full(MAX_HALF_EDGES, -1, np.int32)
    he_twin = np.full(MAX_HALF_EDGES, -1, np.int32)
    he_next = np.full(MAX_HALF_EDGES, -1, np.int32)
    he_prev = np.full(MAX_HALF_EDGES, -1, np.int32)
    he_idx = np.full(MAX_HALF_EDGES, -1, np.int32)
    he_intrinsic = np.zeros(MAX_HALF_EDGES, np.float32)

    for i, he in enumerate(he_list):
        he_idx[i] = i
        he_dest[i] = he['dest']
        he_face[i] = he['face']
        he_twin[i] = he['twin']
        he_next[i] = he['next']
        he_prev[i] = he['prev']

    for i in range(n_he):
        dst = he_dest[i]
        twin = he_twin[i]
        if dst < 0 or twin < 0:
            continue
        src = he_dest[twin]
        if src < 0:
            continue
        d = pos_np[dst] - pos_np[src]
        he_intrinsic[i] = float(np.linalg.norm(d))

    v_he = np.full(MAX_VERTICES, -1, np.int32)
    for (src, _dst), he in edge_dict.items():
        if v_he[src] == -1:
            v_he[src] = he

    f_idx = np.full(MAX_FACES, -1, np.int32)
    f_he = np.full(MAX_FACES, -1, np.int32)
    for fi in range(n_faces_input):
        f_idx[fi] = fi
        v0, v1, _v2 = faces[fi]
        f_he[fi] = edge_dict[(int(v0), int(v1))]

    return (he_idx, he_dest, he_face, he_twin, he_next, he_prev, he_intrinsic,
            v_he, f_idx, f_he, n_verts, n_faces_input, n_he)


def build_mesh_from_faces(positions, faces, state_dims=1, key=None):
    """Build MeshState from positions and triangle faces."""
    state = create_initial_state(state_dims=state_dims, key=key)

    (he_idx, he_dest, he_face, he_twin, he_next, he_prev, he_intrinsic,
     v_he, f_idx, f_he, n_verts, _, _) = _build_half_edges(positions, faces)

    state = state._replace(
        vertex_idx=state.vertex_idx.at[:n_verts].set(jnp.arange(n_verts)),
        vertex_pos=state.vertex_pos.at[:n_verts].set(
            jnp.array(positions, dtype=jnp.float32)
        ),
        vertex_half_edge=jnp.array(v_he),
        half_edge_idx=jnp.array(he_idx),
        half_edge_dest=jnp.array(he_dest),
        half_edge_face=jnp.array(he_face),
        half_edge_twin=jnp.array(he_twin),
        half_edge_next=jnp.array(he_next),
        half_edge_prev=jnp.array(he_prev),
        half_edge_intrinsic_len=jnp.array(he_intrinsic),
        face_idx=jnp.array(f_idx),
        face_half_edge=jnp.array(f_he),
    )

    return state


def make_disc(n_rings=5, spring_len=30.0, center=(400.0, 400.0, 0.0),
              state_dims=1, key=None):
    """Flat hexagonal disc."""
    cx, cy, cz = center
    positions = [[cx, cy, cz]]
    faces = []

    ring_counts = []
    for ring in range(n_rings):
        r = spring_len * (ring + 1)
        n = 6 * (ring + 1)
        ring_counts.append(n)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            positions.append([cx + r * np.cos(a), cy + r * np.sin(a), cz])

    def vidx(ring, seg):
        if ring == -1:
            return 0
        start = 1 + sum(ring_counts[:ring])
        return start + (seg % ring_counts[ring])

    for i in range(ring_counts[0]):
        faces.append([0, vidx(0, i), vidx(0, i + 1)])

    for ri in range(n_rings - 1):
        ni = ring_counts[ri]
        no = ring_counts[ri + 1]
        ii, oi = 0, 0
        while ii < ni or oi < no:
            vi = vidx(ri, ii)
            vin = vidx(ri, ii + 1)
            vo = vidx(ri + 1, oi)
            von = vidx(ri + 1, oi + 1)
            ia = ii / ni
            ian = (ii + 1) / ni
            oa = oi / no
            oan = (oi + 1) / no
            if oan < ian:
                faces.append([vi, vo, von])
                oi += 1
            else:
                faces.append([vi, vo, vin])
                if oa < ian:
                    faces.append([vin, vo, von])
                    oi += 1
                ii += 1

    return build_mesh_from_faces(
        np.array(positions), np.array(faces), state_dims=state_dims, key=key,
    )


def make_circle(width, height, params, n_rings=3, segments_inner=6,
                state_dims=1, key=None):
    """Concentric-ring disc centered on screen."""
    spring_len = float(params.spring_len)
    return make_disc(
        n_rings=n_rings,
        spring_len=spring_len,
        center=(width / 2, height / 2, 0.0),
        state_dims=state_dims,
        key=key,
    )


def make_hemisphere(width, height, params, n_lat=4, n_lon=10,
                    radius_multiplier=2.0, state_dims=1, key=None,
                    ground_z=0.0, closed=False):
    """UV hemisphere dome on z=ground_z.

    Top pole + (n_lat - 1) intermediate latitude rings + equator ring (last,
    on the ground). With `closed=False` the equator is an open boundary loop
    (the rim is pinned by callers); with `closed=True` a flat disc cap fans
    the equator ring to a centre vertex so the surface is watertight (a closed
    half-ball, flat side on the ground) — there is then no boundary, so the
    base is anchored by the ground-plane clamp + gravity rather than a pin.
    """
    cx, cy = width / 2, height / 2
    radius = float(params.spring_len) * radius_multiplier

    positions = [[cx, cy, ground_z + radius]]
    for lat_idx in range(n_lat):
        theta = (np.pi / 2) * (lat_idx + 1) / n_lat
        z = ground_z + radius * np.cos(theta)
        ring_radius = radius * np.sin(theta)
        for lon_idx in range(n_lon):
            phi = 2 * np.pi * lon_idx / n_lon
            x = cx + ring_radius * np.cos(phi)
            y = cy + ring_radius * np.sin(phi)
            positions.append([x, y, z])

    def vidx(lat, lon):
        if lat == -1:
            return 0
        return 1 + lat * n_lon + (lon % n_lon)

    faces = []
    for lon_idx in range(n_lon):
        faces.append([vidx(-1, 0), vidx(0, lon_idx), vidx(0, lon_idx + 1)])
    for lat_idx in range(n_lat - 1):
        for lon_idx in range(n_lon):
            v_tl = vidx(lat_idx, lon_idx)
            v_tr = vidx(lat_idx, lon_idx + 1)
            v_bl = vidx(lat_idx + 1, lon_idx)
            v_br = vidx(lat_idx + 1, lon_idx + 1)
            faces.append([v_tl, v_bl, v_tr])
            faces.append([v_tr, v_bl, v_br])

    if closed:
        # Flat bottom disc closing the equator. Built as concentric rings of
        # n_lon segments at decreasing radius (radial spacing ~ spring_len, like
        # the dome's latitude rings), sharing the equator ring as its outer ring
        # and fanning to a centre vertex — so cap edges are ~spring_len rather
        # than the radius-long slivers a single centre fan would make. Wound
        # opposite the dome so the cap's outward normal points down (-z).
        n_cap = max(1, int(round(radius / float(params.spring_len))))
        cap_ring_starts = []  # first vertex index of each interior cap ring
        for cap_idx in range(1, n_cap):
            ring_radius = radius * (1.0 - cap_idx / n_cap)
            cap_ring_starts.append(len(positions))
            for lon_idx in range(n_lon):
                phi = 2 * np.pi * lon_idx / n_lon
                positions.append([cx + ring_radius * np.cos(phi),
                                  cy + ring_radius * np.sin(phi), ground_z])
        center_idx = len(positions)
        positions.append([cx, cy, ground_z])

        def cap_vidx(cap_ring, lon):
            if cap_ring == 0:                 # outer ring == dome equator
                return vidx(n_lat - 1, lon)
            if cap_ring == n_cap:             # innermost == centre vertex
                return center_idx
            return cap_ring_starts[cap_ring - 1] + (lon % n_lon)

        for cap_idx in range(n_cap - 1):      # stitch ring cap_idx -> cap_idx+1
            for lon_idx in range(n_lon):
                v_tl = cap_vidx(cap_idx, lon_idx)
                v_tr = cap_vidx(cap_idx, lon_idx + 1)
                v_bl = cap_vidx(cap_idx + 1, lon_idx)
                v_br = cap_vidx(cap_idx + 1, lon_idx + 1)
                faces.append([v_tl, v_tr, v_bl])
                faces.append([v_tr, v_br, v_bl])
        for lon_idx in range(n_lon):          # innermost ring -> centre
            faces.append([cap_vidx(n_cap - 1, lon_idx),
                          cap_vidx(n_cap - 1, lon_idx + 1), center_idx])

    return build_mesh_from_faces(
        np.array(positions), np.array(faces),
        state_dims=state_dims, key=key,
    )


def make_sphere(width, height, params, n_lat=6, n_lon=8, radius_multiplier=2.0,
                state_dims=1, key=None):
    """Closed UV sphere."""
    cx, cy, cz = width / 2, height / 2, 0.0
    radius = float(params.spring_len) * radius_multiplier

    positions = [[cx, cy, cz + radius]]
    for lat_idx in range(n_lat):
        theta = np.pi * (lat_idx + 1) / (n_lat + 1)
        z = cz + radius * np.cos(theta)
        ring_radius = radius * np.sin(theta)
        for lon_idx in range(n_lon):
            phi = 2 * np.pi * lon_idx / n_lon
            x = cx + ring_radius * np.cos(phi)
            y = cy + ring_radius * np.sin(phi)
            positions.append([x, y, z])
    positions.append([cx, cy, cz - radius])
    south_pole = len(positions) - 1

    def vidx(lat, lon):
        if lat == -1:
            return 0
        if lat == n_lat:
            return south_pole
        return 1 + lat * n_lon + (lon % n_lon)

    faces = []
    for lon_idx in range(n_lon):
        faces.append([vidx(-1, 0), vidx(0, lon_idx), vidx(0, lon_idx + 1)])
    for lat_idx in range(n_lat - 1):
        for lon_idx in range(n_lon):
            v_tl = vidx(lat_idx, lon_idx)
            v_tr = vidx(lat_idx, lon_idx + 1)
            v_bl = vidx(lat_idx + 1, lon_idx)
            v_br = vidx(lat_idx + 1, lon_idx + 1)
            faces.append([v_tl, v_bl, v_tr])
            faces.append([v_tr, v_bl, v_br])
    for lon_idx in range(n_lon):
        faces.append([vidx(n_lat, 0), vidx(n_lat - 1, lon_idx + 1), vidx(n_lat - 1, lon_idx)])

    return build_mesh_from_faces(
        np.array(positions), np.array(faces), state_dims=state_dims, key=key,
    )


# ── Topology Operations ──────────────────────────────────────────────────────

def _get_he_ab(state, va, vb):
    """Find half-edge from va to vb by walking va's fan."""
    start = state.vertex_half_edge[va]

    def cond(carry):
        he, first = carry
        return (state.half_edge_dest[he] != vb) & (first | (he != start))

    def body(carry):
        he, _ = carry
        return state.half_edge_next[state.half_edge_twin[he]], False

    he, _ = jax.lax.while_loop(cond, body, (start, True))
    return he


@jax.jit
def add_external_triangle(state, va, vb):
    """Add a triangle outside boundary edge va->vb.

    Jitted: va/vb trace as dynamic int scalars, so this compiles once and each
    subsequent call is a single executable launch instead of an eager op-by-op
    dispatch with lax.cond/while_loop re-tracing (~600 ms/call unjitted).
    """
    he_ab = _get_he_ab(state, va, vb)
    new_f = jnp.max(state.face_idx) + 1
    new_v = jnp.max(state.vertex_idx) + 1
    new_he = jnp.max(state.half_edge_idx) + 1

    overflow = (new_f >= MAX_FACES) | (new_v >= MAX_VERTICES) \
        | (new_he + 4 >= MAX_HALF_EDGES)
    invalid = (he_ab == -1) | (state.half_edge_face[he_ab] != -1)

    return jax.lax.cond(
        overflow | invalid,
        lambda: state,
        lambda: _add_external_triangle(state, va, vb, he_ab, new_f, new_v, new_he),
    )


def _add_external_triangle(state, va, vb, he_ab, nf, nv, nhe):
    vc = nv
    he_bc, he_ca, he_cb, he_ac = nhe, nhe + 1, nhe + 2, nhe + 3
    he_b_next = state.half_edge_next[he_ab]
    he_a_prev = state.half_edge_prev[he_ab]
    edge_len = jnp.sqrt(jnp.sum((state.vertex_pos[vb] - state.vertex_pos[va]) ** 2) + EPSILON)
    z_perturb = (jnp.float32(nv % 97) / 97.0 - 0.5) * edge_len * 0.05
    new_pos = (state.vertex_pos[va] + state.vertex_pos[vb]) / 2
    new_pos = new_pos.at[2].add(z_perturb)
    new_state_val = (state.vertex_state[va] + state.vertex_state[vb]) / 2

    u = {}
    u['vertex_idx'] = state.vertex_idx.at[vc].set(vc)
    u['vertex_pos'] = state.vertex_pos.at[vc].set(new_pos)
    u['vertex_half_edge'] = state.vertex_half_edge.at[vc].set(he_ca)
    u['vertex_state'] = state.vertex_state.at[vc].set(new_state_val)
    u['face_idx'] = state.face_idx.at[nf].set(nf)
    u['face_half_edge'] = state.face_half_edge.at[nf].set(he_ab)
    idx4 = jnp.array([he_bc, he_ca, he_cb, he_ac])
    u['half_edge_idx'] = state.half_edge_idx.at[idx4].set(idx4)
    u['half_edge_face'] = state.half_edge_face \
        .at[jnp.array([he_ab, he_bc, he_ca, he_cb, he_ac])] \
        .set(jnp.array([nf, nf, nf, -1, -1]))
    u['half_edge_dest'] = state.half_edge_dest \
        .at[jnp.array([he_bc, he_ca, he_cb, he_ac])] \
        .set(jnp.array([vc, va, vb, vc]))
    u['half_edge_twin'] = state.half_edge_twin \
        .at[jnp.array([he_bc, he_cb, he_ca, he_ac])] \
        .set(jnp.array([he_cb, he_bc, he_ac, he_ca]))
    u['half_edge_next'] = state.half_edge_next \
        .at[jnp.array([he_ab, he_bc, he_ca, he_cb, he_ac, he_a_prev])] \
        .set(jnp.array([he_bc, he_ca, he_ab, he_b_next, he_cb, he_ac]))
    u['half_edge_prev'] = state.half_edge_prev \
        .at[jnp.array([he_ab, he_bc, he_ca, he_cb, he_ac, he_b_next])] \
        .set(jnp.array([he_ca, he_ab, he_bc, he_ac, he_a_prev, he_cb]))
    l_bc = jnp.sqrt(jnp.sum((state.vertex_pos[vb] - new_pos) ** 2) + EPSILON)
    l_ca = jnp.sqrt(jnp.sum((new_pos - state.vertex_pos[va]) ** 2) + EPSILON)
    u['half_edge_intrinsic_len'] = state.half_edge_intrinsic_len \
        .at[jnp.array([he_bc, he_cb, he_ca, he_ac])] \
        .set(jnp.array([l_bc, l_bc, l_ca, l_ca]))
    return state._replace(**u)


@jax.jit
def add_internal_edge_triangle(state, va, vb):
    """Split edge va->vb where one side is internal, other is boundary."""
    he_ab = _get_he_ab(state, va, vb)
    he_ba = state.half_edge_twin[he_ab]
    he_bc = state.half_edge_next[he_ab]
    he_ca = state.half_edge_next[he_bc]
    vc = state.half_edge_dest[he_bc]
    f_abc = state.half_edge_face[he_ab]

    vd = jnp.max(state.vertex_idx) + 1
    f_dbc = jnp.max(state.face_idx) + 1
    nhe = jnp.max(state.half_edge_idx) + 1
    he_db, he_cd, he_dc, he_da = nhe, nhe + 1, nhe + 2, nhe + 3
    he_ad = he_ab
    he_bd = he_ba
    f_adc = f_abc

    he_ba_next = state.half_edge_next[he_ba]
    he_ba_prev = state.half_edge_prev[he_ba]
    edge_len = jnp.sqrt(jnp.sum((state.vertex_pos[vb] - state.vertex_pos[va]) ** 2) + EPSILON)
    z_perturb = (jnp.float32(vd % 97) / 97.0 - 0.5) * edge_len * 0.05
    new_pos = (state.vertex_pos[va] + state.vertex_pos[vb]) / 2
    new_pos = new_pos.at[2].add(z_perturb)
    new_state_val = (state.vertex_state[va] + state.vertex_state[vb]) / 2

    l_ab = state.half_edge_intrinsic_len[he_ab]
    l_bc_i = state.half_edge_intrinsic_len[he_bc]
    l_ca_i = state.half_edge_intrinsic_len[he_ca]

    u = {}
    u['vertex_idx'] = state.vertex_idx.at[vd].set(vd)
    u['vertex_pos'] = state.vertex_pos.at[vd].set(new_pos)
    u['vertex_half_edge'] = state.vertex_half_edge.at[vd].set(he_db)
    u['vertex_state'] = state.vertex_state.at[vd].set(new_state_val)

    u['face_idx'] = state.face_idx.at[f_dbc].set(f_dbc)
    u['face_half_edge'] = state.face_half_edge \
        .at[jnp.array([f_dbc, f_adc])].set(jnp.array([he_bc, he_ca]))

    idx4 = jnp.array([he_db, he_da, he_dc, he_cd])
    u['half_edge_idx'] = state.half_edge_idx.at[idx4].set(idx4)
    u['half_edge_face'] = state.half_edge_face \
        .at[jnp.array([he_ad, he_db, he_da, he_dc, he_cd, he_ca, he_bc])] \
        .set(jnp.array([f_adc, f_dbc, -1, f_adc, f_dbc, f_adc, f_dbc]))
    u['half_edge_dest'] = state.half_edge_dest \
        .at[jnp.array([he_ad, he_db, he_bd, he_da, he_dc, he_cd])] \
        .set(jnp.array([vd, vb, vd, va, vc, vd]))
    u['half_edge_twin'] = state.half_edge_twin \
        .at[jnp.array([he_ad, he_db, he_bd, he_da, he_dc, he_cd])] \
        .set(jnp.array([he_da, he_bd, he_db, he_ad, he_cd, he_dc]))
    u['half_edge_next'] = state.half_edge_next \
        .at[jnp.array([he_ad, he_db, he_ba_prev, he_bd, he_da,
                        he_dc, he_cd, he_ca, he_bc])] \
        .set(jnp.array([he_dc, he_bc, he_bd, he_da, he_ba_next,
                        he_ca, he_db, he_ad, he_cd]))
    u['half_edge_prev'] = state.half_edge_prev \
        .at[jnp.array([he_ad, he_db, he_ba_next, he_da,
                        he_dc, he_cd, he_ca, he_bc])] \
        .set(jnp.array([he_ca, he_cd, he_da, he_bd,
                        he_ad, he_bc, he_dc, he_db]))
    l_ad = l_ab * 0.5
    l_db = l_ab * 0.5
    l_dc_sq = (2.0 * l_ca_i * l_ca_i + 2.0 * l_bc_i * l_bc_i - l_ab * l_ab) / 4.0
    l_dc = jnp.sqrt(jnp.maximum(l_dc_sq, 0.0))
    u['half_edge_intrinsic_len'] = state.half_edge_intrinsic_len \
        .at[jnp.array([he_ad, he_da, he_db, he_bd, he_dc, he_cd])] \
        .set(jnp.array([l_ad, l_ad, l_db, l_db, l_dc, l_dc]))
    return state._replace(**u)


@jax.jit
def add_internal_triangles(state, va, vb):
    """Split interior edge va->vb, creating 2 new faces."""
    he_ab = _get_he_ab(state, va, vb)
    he_ba = state.half_edge_twin[he_ab]

    return jax.lax.cond(
        (state.half_edge_face[he_ab] == -1) | (state.half_edge_face[he_ba] == -1),
        lambda: state,
        lambda: _add_internal_triangles(state, va, vb, he_ab),
    )


def _add_internal_triangles(state, va, vb, he_ab):
    he_ba = state.half_edge_twin[he_ab]
    vc = state.half_edge_dest[state.half_edge_next[he_ab]]
    vd = state.half_edge_dest[state.half_edge_next[he_ba]]
    f_abc = state.half_edge_face[he_ab]
    f_bad = state.half_edge_face[he_ba]

    ve = jnp.max(state.vertex_idx) + 1
    f_ebc = jnp.max(state.face_idx) + 1
    f_ead = f_ebc + 1
    nhe = jnp.max(state.half_edge_idx) + 1
    he_ec, he_ea, he_ce, he_eb, he_de, he_ed = \
        nhe, nhe + 1, nhe + 2, nhe + 3, nhe + 4, nhe + 5
    he_ae = he_ab
    he_be = he_ba
    f_aec = f_abc
    f_bed = f_bad

    he_bc = state.half_edge_next[he_ab]
    he_ca = state.half_edge_next[he_bc]
    he_ad = state.half_edge_next[he_ba]
    he_db = state.half_edge_next[he_ad]

    edge_len = jnp.sqrt(jnp.sum((state.vertex_pos[vb] - state.vertex_pos[va]) ** 2) + EPSILON)
    z_perturb = (jnp.float32(ve % 97) / 97.0 - 0.5) * edge_len * 0.05
    new_pos = (state.vertex_pos[va] + state.vertex_pos[vb]) / 2
    new_pos = new_pos.at[2].add(z_perturb)
    new_state_val = (state.vertex_state[va] + state.vertex_state[vb]) / 2

    l_ab = state.half_edge_intrinsic_len[he_ab]
    l_bc_i = state.half_edge_intrinsic_len[he_bc]
    l_ca_i = state.half_edge_intrinsic_len[he_ca]
    l_ad_i = state.half_edge_intrinsic_len[he_ad]
    l_db_i = state.half_edge_intrinsic_len[he_db]

    u = {}
    u['vertex_idx'] = state.vertex_idx.at[ve].set(ve)
    u['vertex_pos'] = state.vertex_pos.at[ve].set(new_pos)
    u['vertex_half_edge'] = state.vertex_half_edge.at[ve].set(he_eb)
    u['vertex_state'] = state.vertex_state.at[ve].set(new_state_val)

    f2 = jnp.array([f_ebc, f_ead])
    u['face_idx'] = state.face_idx.at[f2].set(f2)
    u['face_half_edge'] = state.face_half_edge \
        .at[jnp.array([f_aec, f_bed, f_ebc, f_ead])] \
        .set(jnp.array([he_ae, he_be, he_eb, he_ea]))

    idx6 = jnp.array([he_ec, he_ea, he_eb, he_ce, he_de, he_ed])
    u['half_edge_idx'] = state.half_edge_idx.at[idx6].set(idx6)

    u['half_edge_face'] = state.half_edge_face \
        .at[jnp.array([he_ec, he_eb, he_bc, he_ce, he_ea,
                        he_ad, he_de, he_be, he_ed])] \
        .set(jnp.array([f_aec, f_ebc, f_ebc, f_ebc, f_ead,
                        f_ead, f_ead, f_bed, f_bed]))
    u['half_edge_dest'] = state.half_edge_dest \
        .at[jnp.array([he_ae, he_ec, he_eb, he_ce, he_ea, he_de, he_be, he_ed])] \
        .set(jnp.array([ve, vc, vb, ve, va, ve, ve, vd]))
    u['half_edge_twin'] = state.half_edge_twin \
        .at[jnp.array([he_ae, he_ec, he_ce, he_eb, he_be, he_ea, he_de, he_ed])] \
        .set(jnp.array([he_ea, he_ce, he_ec, he_be, he_eb, he_ae, he_ed, he_de]))
    u['half_edge_next'] = state.half_edge_next \
        .at[jnp.array([he_ae, he_ec, he_ca, he_eb, he_bc, he_ce,
                        he_ea, he_ad, he_de, he_be, he_ed, he_db])] \
        .set(jnp.array([he_ec, he_ca, he_ae, he_bc, he_ce, he_eb,
                        he_ad, he_de, he_ea, he_ed, he_db, he_be]))
    u['half_edge_prev'] = state.half_edge_prev \
        .at[jnp.array([he_ec, he_ca, he_ae, he_eb, he_bc, he_ce,
                        he_ea, he_ad, he_de, he_be, he_ed, he_db])] \
        .set(jnp.array([he_ae, he_ec, he_ca, he_ce, he_eb, he_bc,
                        he_de, he_ea, he_ad, he_db, he_be, he_ed]))
    l_ae = l_ab * 0.5
    l_eb = l_ab * 0.5
    l_ec_sq = (2.0 * l_ca_i * l_ca_i + 2.0 * l_bc_i * l_bc_i - l_ab * l_ab) / 4.0
    l_ec = jnp.sqrt(jnp.maximum(l_ec_sq, 0.0))
    l_ed_sq = (2.0 * l_ad_i * l_ad_i + 2.0 * l_db_i * l_db_i - l_ab * l_ab) / 4.0
    l_ed = jnp.sqrt(jnp.maximum(l_ed_sq, 0.0))
    u['half_edge_intrinsic_len'] = state.half_edge_intrinsic_len \
        .at[jnp.array([he_ae, he_ea, he_eb, he_be, he_ec, he_ce, he_ed, he_de])] \
        .set(jnp.array([l_ae, l_ae, l_eb, l_eb, l_ec, l_ec, l_ed, l_ed]))
    return state._replace(**u)


def flip_edge(state, he_ab):
    """Flip interior edge using intrinsic planar unfold for new length."""
    he_ba = state.half_edge_twin[he_ab]
    return jax.lax.cond(
        (state.half_edge_face[he_ab] == -1) | (state.half_edge_face[he_ba] == -1),
        lambda: state,
        lambda: _flip_edge(state, he_ab, he_ba),
    )


@jax.jit
def _flip_edge(state, he_ab, he_ba):
    va = state.half_edge_dest[he_ba]
    vb = state.half_edge_dest[he_ab]
    he_bc = state.half_edge_next[he_ab]
    he_ca = state.half_edge_next[he_bc]
    he_ad = state.half_edge_next[he_ba]
    he_db = state.half_edge_next[he_ad]
    f_abc = state.half_edge_face[he_ab]
    f_bad = state.half_edge_face[he_ba]

    lab = state.half_edge_intrinsic_len[he_ab]
    lbc = state.half_edge_intrinsic_len[he_bc]
    lca = state.half_edge_intrinsic_len[he_ca]
    lad = state.half_edge_intrinsic_len[he_ad]
    ldb = state.half_edge_intrinsic_len[he_db]
    lcd = _intrinsic_flip_length(lab, lca, lbc, lad, ldb)

    he_dc, he_cd = he_ab, he_ba
    f_adc, f_bcd = f_abc, f_bad

    u = {}
    u['face_half_edge'] = state.face_half_edge \
        .at[jnp.array([f_adc, f_bcd])].set(jnp.array([he_ca, he_db]))
    u['vertex_half_edge'] = state.vertex_half_edge \
        .at[jnp.array([va, vb])].set(jnp.array([he_ad, he_bc]))
    u['half_edge_dest'] = state.half_edge_dest \
        .at[jnp.array([he_dc, he_cd])].set(jnp.array([jnp.int32(state.half_edge_dest[he_bc]),
                                                        jnp.int32(state.half_edge_dest[he_ad])]))
    u['half_edge_next'] = state.half_edge_next \
        .at[jnp.array([he_dc, he_cd, he_ca, he_ad, he_db, he_bc])] \
        .set(jnp.array([he_ca, he_db, he_ad, he_dc, he_bc, he_cd]))
    u['half_edge_prev'] = state.half_edge_prev \
        .at[jnp.array([he_dc, he_cd, he_ca, he_ad, he_db, he_bc])] \
        .set(jnp.array([he_ad, he_bc, he_dc, he_ca, he_cd, he_db]))
    u['half_edge_face'] = state.half_edge_face \
        .at[jnp.array([he_dc, he_cd, he_ca, he_ad, he_db, he_bc])] \
        .set(jnp.array([f_adc, f_bcd, f_adc, f_adc, f_bcd, f_bcd]))
    u['half_edge_intrinsic_len'] = state.half_edge_intrinsic_len \
        .at[jnp.array([he_dc, he_cd])].set(jnp.array([lcd, lcd]))
    return state._replace(**u)


def _intrinsic_flip_length(lab, lca, lbc, lad, ldb):
    """Intrinsic edge length of CD after flipping AB→CD via planar unfold."""
    cos_a_acb = (lab * lab + lca * lca - lbc * lbc) \
        / jnp.maximum(2.0 * lab * lca, EPSILON)
    sin_a_acb = jnp.sqrt(jnp.maximum(1.0 - cos_a_acb * cos_a_acb, 0.0))
    cx = lca * cos_a_acb
    cy = lca * sin_a_acb
    cos_a_adb = (lab * lab + lad * lad - ldb * ldb) \
        / jnp.maximum(2.0 * lab * lad, EPSILON)
    sin_a_adb = jnp.sqrt(jnp.maximum(1.0 - cos_a_adb * cos_a_adb, 0.0))
    dx = lad * cos_a_adb
    dy = -lad * sin_a_adb
    return jnp.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)


# ── Mesh Refinement ──────────────────────────────────────────────────────────

def _edge_flip_criterion(state, he):
    """Per-edge intrinsic-Delaunay criterion.

    Returns (should_flip, sin_sum). The canonical direction filter
    (dest < twin_dest) ensures each undirected edge is considered once.
    """
    twin = state.half_edge_twin[he]
    safe_twin = jnp.maximum(twin, 0)

    he_bc = state.half_edge_next[he]
    he_ad = state.half_edge_next[safe_twin]
    safe_bc = jnp.maximum(he_bc, 0)
    safe_ad = jnp.maximum(he_ad, 0)
    he_ca = state.half_edge_next[safe_bc]
    he_db = state.half_edge_next[safe_ad]
    safe_ca = jnp.maximum(he_ca, 0)
    safe_db = jnp.maximum(he_db, 0)

    lab = state.half_edge_intrinsic_len[he]
    lbc = state.half_edge_intrinsic_len[safe_bc]
    lca = state.half_edge_intrinsic_len[safe_ca]
    lad = state.half_edge_intrinsic_len[safe_ad]
    ldb = state.half_edge_intrinsic_len[safe_db]

    lab2 = lab * lab
    cos_c = (lca * lca + lbc * lbc - lab2) \
        / jnp.maximum(2.0 * lca * lbc, EPSILON)
    sin_c = jnp.sqrt(jnp.maximum(1.0 - cos_c * cos_c, 0.0))
    cos_d = (lad * lad + ldb * ldb - lab2) \
        / jnp.maximum(2.0 * lad * ldb, EPSILON)
    sin_d = jnp.sqrt(jnp.maximum(1.0 - cos_d * cos_d, 0.0))
    sin_sum = sin_c * cos_d + cos_c * sin_d

    dest = state.half_edge_dest[he]
    twin_dest = state.half_edge_dest[safe_twin]

    should_flip = (
        (state.half_edge_idx[he] >= 0)
        & (state.half_edge_face[he] != -1)
        & (twin >= 0)
        & (state.half_edge_face[safe_twin] != -1)
        & (he_bc >= 0) & (he_ad >= 0)
        & (he_ca >= 0) & (he_db >= 0)
        & (dest < twin_dest)
        & (sin_sum < -EPSILON)
    )
    return should_flip, sin_sum


REFINE_TOP_K = 128


def _refine_score_all(state):
    """Vectorized violation score for every half-edge.

    Mirrors _edge_flip_criterion but in batched form, written with explicit
    array ops (rather than vmap) so XLA can fuse the elementwise chain into
    fewer live intermediates.
    """
    he_twin = state.half_edge_twin
    safe_twin = jnp.maximum(he_twin, 0)

    he_next = state.half_edge_next
    he_bc = he_next
    he_ad = he_next[safe_twin]
    safe_bc = jnp.maximum(he_bc, 0)
    safe_ad = jnp.maximum(he_ad, 0)
    he_ca = he_next[safe_bc]
    he_db = he_next[safe_ad]
    safe_ca = jnp.maximum(he_ca, 0)
    safe_db = jnp.maximum(he_db, 0)

    he_ilen = state.half_edge_intrinsic_len
    lab = he_ilen
    lbc = he_ilen[safe_bc]
    lca = he_ilen[safe_ca]
    lad = he_ilen[safe_ad]
    ldb = he_ilen[safe_db]

    lab2 = lab * lab
    cos_c = (lca * lca + lbc * lbc - lab2) \
        / jnp.maximum(2.0 * lca * lbc, EPSILON)
    sin_c = jnp.sqrt(jnp.maximum(1.0 - cos_c * cos_c, 0.0))
    cos_d = (lad * lad + ldb * ldb - lab2) \
        / jnp.maximum(2.0 * lad * ldb, EPSILON)
    sin_d = jnp.sqrt(jnp.maximum(1.0 - cos_d * cos_d, 0.0))
    sin_sum = sin_c * cos_d + cos_c * sin_d

    he_face = state.half_edge_face
    he_dest = state.half_edge_dest
    twin_dest = he_dest[safe_twin]

    should_flip = (
        (state.half_edge_idx >= 0)
        & (he_face != -1)
        & (he_twin >= 0)
        & (he_face[safe_twin] != -1)
        & (he_bc >= 0) & (he_ad >= 0)
        & (he_ca >= 0) & (he_db >= 0)
        & (he_dest < twin_dest)
        & (sin_sum < -EPSILON)
    )
    return jnp.where(should_flip, -sin_sum, -jnp.inf)


@jax.jit
def refine_mesh(state):
    """Flip non-Delaunay edges using intrinsic edge lengths.

    Vectorized pass scores every half-edge by violation severity; top_k
    picks the REFINE_TOP_K worst; scan over those re-checks the criterion
    against carry state (since earlier flips can invalidate neighbours).
    Edges that newly violate mid-pass get caught next call to refine_mesh.
    """
    score = _refine_score_all(state)
    _, candidate_he = jax.lax.top_k(score, REFINE_TOP_K)

    def body(state, he):
        should_flip, _ = _edge_flip_criterion(state, he)
        state = jax.lax.cond(
            should_flip,
            lambda s: flip_edge(s, he),
            lambda s: s,
            state,
        )
        return state, None

    state, _ = jax.lax.scan(body, state, candidate_he)
    return state


@functools.partial(jax.jit, static_argnames=("max_splits",))
def _select_long_edges(state, params, key, max_splits):
    """Pick up to max_splits edges with avg intrinsic length >= max_edge_len.

    Pure JAX selection: avoids per-iter transfers of the full half-edge arrays
    and the MAX_HALF_EDGES-long Python scan. Returns small (max_splits,) arrays
    plus a host-int candidate of n_verts (for the limit check).

    Random ordering is preserved (relative to the old np.random shuffle): each
    eligible half-edge gets a uniform-random score, top_k picks the K with the
    highest score. Ineligible edges get score=-1 so they sort last.
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    unique = he_valid \
        & (state.half_edge_twin >= 0) \
        & (state.half_edge_dest < state.half_edge_dest[safe_twin])

    avg_ilen = (state.half_edge_intrinsic_len
                + state.half_edge_intrinsic_len[safe_twin]) * 0.5
    eligible = unique & (avg_ilen >= params.max_edge_len)

    noise = jax.random.uniform(key, (MAX_HALF_EDGES,))
    score = jnp.where(eligible, noise, -1.0)

    _, top_he = jax.lax.top_k(score, max_splits)
    top_twin = safe_twin[top_he]

    src_arr = state.half_edge_dest[top_twin]
    dst_arr = state.half_edge_dest[top_he]
    he_bnd_arr = state.half_edge_face[top_he] == -1
    twin_bnd_arr = state.half_edge_face[top_twin] == -1
    valid_arr = eligible[top_he]

    n_verts = jnp.sum(state.vertex_idx != -1)
    return src_arr, dst_arr, he_bnd_arr, twin_bnd_arr, valid_arr, n_verts


def split_long_edges(state, params, key, max_splits=20):
    """Split edges exceeding max_edge_len based on intrinsic lengths.

    Selection runs in JAX; mutations run on the host because each
    add_internal_triangle{,s} call depends on the state produced by the
    previous one (sequential graph rewrites).
    """
    key, sel_key = jax.random.split(key)
    src_j, dst_j, he_bnd_j, twin_bnd_j, valid_j, n_verts_j = \
        _select_long_edges(state, params, sel_key, max_splits)

    src_arr = np.asarray(src_j)
    dst_arr = np.asarray(dst_j)
    he_bnd_arr = np.asarray(he_bnd_j)
    twin_bnd_arr = np.asarray(twin_bnd_j)
    valid_arr = np.asarray(valid_j)
    n_verts = int(n_verts_j)

    for i in range(max_splits):
        if not valid_arr[i]:
            continue
        if n_verts >= MAX_VERTICES - 5:
            break
        src = int(src_arr[i])
        dst = int(dst_arr[i])
        he_bnd = bool(he_bnd_arr[i])
        twin_bnd = bool(twin_bnd_arr[i])
        if he_bnd or twin_bnd:
            if twin_bnd:
                state = add_internal_edge_triangle(state, src, dst)
            else:
                state = add_internal_edge_triangle(state, dst, src)
        else:
            state = add_internal_triangles(state, src, dst)
        n_verts += 1

    return state, key, n_verts


# ── External Forces (Predict Step) ──────────────────────────────────────────

@jax.jit
def compute_external_forces(pos, topo, params):
    """Bulge (boundary outward push) + Laplacian smoothing + gravity.

    Uses the precomputed XpbdTopo. The clipped (safe_*) indices match the raw
    half_edge_dest values on every valid boundary edge; invalid/non-boundary
    edges contribute zero (edge_n is masked, neighbor_sum uses he_valid), so
    the clip does not change the result.
    """
    active = topo.active[:, None].astype(jnp.float32)
    he_src = topo.he_src
    safe_dest = topo.safe_dest

    edge_vec = pos[safe_dest] - pos[he_src]
    next_dst = safe_dest[topo.safe_twin_next]
    next_vec = pos[next_dst] - pos[he_src]

    surface_n = jnp.cross(edge_vec, next_vec)
    edge_n = jnp.cross(edge_vec, surface_n)
    edge_n_len = jnp.linalg.norm(edge_n, axis=-1, keepdims=True)
    edge_n = edge_n / jnp.maximum(edge_n_len, EPSILON)
    edge_n = jnp.where(topo.boundary[:, None], edge_n, 0.0)

    bulge = jnp.zeros_like(pos)
    bulge = bulge.at[he_src].add(edge_n)
    bulge = bulge.at[safe_dest].add(edge_n)
    fnorm = jnp.linalg.norm(bulge, axis=-1, keepdims=True)
    bulge = bulge / jnp.maximum(fnorm, EPSILON)

    dst_pos = pos[safe_dest]
    neighbor_sum = jnp.zeros_like(pos).at[he_src].add(
        jnp.where(topo.he_valid[:, None], dst_pos, 0.0)
    )
    centroid = neighbor_sum / jnp.maximum(topo.degree_raw[:, None], 1.0)
    laplacian = centroid - pos

    gravity = jnp.zeros_like(pos)
    gravity = gravity.at[:, 2].set(-params.gravity_strength)
    gravity = gravity * active

    forces = params.bulge_strength * bulge \
        + params.stiffness * 0.5 * laplacian * active \
        + gravity
    return forces


@jax.jit
def predict_positions(state, forces, params):
    """Overdamped position prediction: p += (dt/damping) * f, clamped."""
    active = (state.vertex_idx != -1)[:, None].astype(jnp.float32)
    displacement = (params.dt / params.damping) * forces
    max_disp = params.spring_len * 0.25
    disp_len = jnp.linalg.norm(displacement, axis=-1, keepdims=True)
    displacement = jnp.where(disp_len > max_disp,
                              displacement * max_disp / disp_len,
                              displacement)
    return state.vertex_pos + displacement * active


# ── Spatial Hash for Collision ──────────────────────────────────────────────

_HASH_OFFSETS_3D = jnp.array(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=jnp.int32,
)
_EARLIER_MASK = jnp.tril(jnp.ones((27, 27), dtype=bool), k=-1)


def _hash_cells(cell_coords):
    s = cell_coords + 10000
    h = s[:, 0] * 73856093 + s[:, 1] * 19349663 + s[:, 2] * 83492791
    return jnp.abs(h) % NUM_HASH_BUCKETS


def _hash_cells_2d(cell_coords):
    s = cell_coords + 10000
    h = s[:, :, 0] * 73856093 + s[:, :, 1] * 19349663 + s[:, :, 2] * 83492791
    return jnp.abs(h) % NUM_HASH_BUCKETS


def _build_spatial_hash(pos, active, cell_size):
    cell_coords = jnp.floor(pos / cell_size).astype(jnp.int32)
    bucket = jnp.where(active, _hash_cells(cell_coords), NUM_HASH_BUCKETS)
    sort_order = jnp.argsort(bucket)
    sorted_bucket = bucket[sort_order]
    all_buckets = jnp.arange(NUM_HASH_BUCKETS, dtype=jnp.int32)
    bstart = jnp.searchsorted(sorted_bucket, all_buckets, side='left')
    bend = jnp.searchsorted(sorted_bucket, all_buckets, side='right')
    return sort_order, bstart, bend, cell_coords


def _gather_candidates(cell_coords, active, sort_order, bstart, bend):
    N = MAX_VERTICES
    nbr_cells = cell_coords[:, None, :] + _HASH_OFFSETS_3D[None, :, :]
    nbr_buckets = _hash_cells_2d(nbr_cells)

    nb_starts = bstart[nbr_buckets]
    nb_ends = bend[nbr_buckets]

    bucket_match = nbr_buckets[:, :, None] == nbr_buckets[:, None, :]
    has_dup = (bucket_match & _EARLIER_MASK[None, :, :]).any(axis=2)
    nb_sizes = jnp.where(has_dup, 0, nb_ends - nb_starts)

    local = jnp.arange(SLOTS_PER_CELL, dtype=jnp.int32)
    cand_sorted = nb_starts[:, :, None] + local[None, None, :]
    cand_sorted = jnp.clip(cand_sorted, 0, N - 1)
    cand_valid = local[None, None, :] < nb_sizes[:, :, None]

    cand_idx = sort_order[cand_sorted]
    K = SLOTS_PER_CELL * 27
    cand_idx = cand_idx.reshape(N, K)
    cand_valid = cand_valid.reshape(N, K)

    cand_active = active[cand_idx]
    not_self = cand_idx != jnp.arange(N)[:, None]
    mask = cand_valid & cand_active & not_self & active[:, None]

    return cand_idx, mask


@jax.jit
def build_neighbor_list(state):
    """(MAX_VERTICES, MAX_FAN_SIZE) of mesh-neighbor vertex indices (-1 pad).

    Used by collision to skip direct mesh neighbors.

    Vectorized via a stable sort on source vertex + a segmented rank, instead
    of a MAX_HALF_EDGES-long sequential scan. A stable argsort keeps half-edge
    index order within each source group, so for vertices with > MAX_FAN_SIZE
    neighbours the same first-MAX_FAN_SIZE are retained as the old scan kept.
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]

    # Group half-edges by source vertex; invalid ones get a sentinel key that
    # sorts after every real vertex so they cluster at the tail.
    src_key = jnp.where(he_valid, he_src, MAX_VERTICES)
    order = jnp.argsort(src_key)  # stable
    sorted_src = src_key[order]
    sorted_dst = safe_dest[order]
    sorted_valid = he_valid[order]

    # Segmented rank: position minus the start index of the current group.
    pos = jnp.arange(MAX_HALF_EDGES)
    is_new = jnp.concatenate(
        [jnp.array([True]), sorted_src[1:] != sorted_src[:-1]]
    )
    group_start = jax.lax.cummax(jnp.where(is_new, pos, 0))
    slot = pos - group_start

    do_write = sorted_valid & (sorted_src < MAX_VERTICES) & (slot < MAX_FAN_SIZE)

    # Route non-writes to a throwaway row at index MAX_VERTICES, then drop it.
    write_row = jnp.where(do_write, sorted_src, MAX_VERTICES)
    write_slot = jnp.where(do_write, slot, 0)
    neighbors = jnp.full((MAX_VERTICES + 1, MAX_FAN_SIZE), -1, jnp.int32)
    neighbors = neighbors.at[write_row, write_slot].set(sorted_dst)
    return neighbors[:MAX_VERTICES]


# ── XPBD Constraints ─────────────────────────────────────────────────────────

XpbdTopo = collections.namedtuple('XpbdTopo', [
    'he_valid', 'safe_twin', 'safe_dest', 'safe_twin_next',
    'he_src', 'opp_c', 'opp_d', 'interior', 'boundary', 'active',
    'degree_raw', 'bend_count_raw',
])


@jax.jit
def build_xpbd_topo(state):
    """Precompute the topology-derived index arrays, masks and degree counts
    shared by project_springs / project_bending / compute_external_forces.

    Topology (all half_edge_* arrays + vertex_idx) is constant for an entire
    batched_physics_step call — growth only mutates intrinsic lengths / state /
    positions, and splits/refines happen outside the step — so this is computed
    once and reused across every substep and XPBD iteration instead of being
    re-derived each time (the index gathers and the two degree scatter-adds are
    the bulk of project_springs / project_bending's non-position work).
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    safe_next = jnp.clip(state.half_edge_next, 0)
    he_src = safe_dest[safe_twin]
    safe_twin_next = jnp.clip(state.half_edge_next[safe_twin], 0)
    opp_c = safe_dest[safe_next]
    opp_d = safe_dest[safe_twin_next]

    he_face = state.half_edge_face
    twin_face = he_face[safe_twin]
    he_idx = jnp.arange(MAX_HALF_EDGES)
    interior = he_valid & (he_face >= 0) & (twin_face >= 0) \
        & (state.half_edge_twin >= 0) & (he_idx < state.half_edge_twin)
    boundary = (he_face == -1) & he_valid
    active = state.vertex_idx != -1

    degree_raw = jnp.zeros(MAX_VERTICES).at[he_src].add(he_valid.astype(jnp.float32))

    int_f = interior.astype(jnp.float32)
    bend_count_raw = jnp.zeros(MAX_VERTICES)
    bend_count_raw = bend_count_raw.at[he_src].add(int_f)
    bend_count_raw = bend_count_raw.at[safe_dest].add(int_f)
    bend_count_raw = bend_count_raw.at[opp_c].add(int_f)
    bend_count_raw = bend_count_raw.at[opp_d].add(int_f)

    return XpbdTopo(
        he_valid=he_valid, safe_twin=safe_twin, safe_dest=safe_dest,
        safe_twin_next=safe_twin_next, he_src=he_src, opp_c=opp_c, opp_d=opp_d,
        interior=interior, boundary=boundary, active=active,
        degree_raw=degree_raw, bend_count_raw=bend_count_raw,
    )


@jax.jit
def project_springs(pos, topo, rest, params):
    """XPBD spring: enforce dist = intrinsic rest length (Jacobi over half-edges).

    `topo` is the precomputed XpbdTopo; `rest` the per-substep rest lengths.
    """
    he_src = topo.he_src
    safe_dest = topo.safe_dest

    alpha_tilde = params.compliance / (params.dt * params.dt + EPSILON)

    src_pos = pos[he_src]
    dst_pos = pos[safe_dest]
    d = src_pos - dst_pos
    dist = jnp.sqrt(jnp.sum(d * d, axis=-1) + EPSILON)
    dist_safe = jnp.maximum(dist, EPSILON)

    C = dist - rest
    dlambda = -C / (2.0 + alpha_tilde)
    edge_corr = dlambda[:, None] * (d / dist_safe[:, None])

    max_corr = rest * 0.2
    corr_mag = jnp.linalg.norm(edge_corr, axis=-1, keepdims=True)
    edge_corr = jnp.where(corr_mag > max_corr[:, None],
                           edge_corr * max_corr[:, None] / corr_mag,
                           edge_corr)
    edge_corr = jnp.where(topo.he_valid[:, None], edge_corr, 0.0)

    correction = jnp.zeros_like(pos).at[he_src].add(edge_corr)
    degree = jnp.maximum(topo.degree_raw, 1.0)

    # SOR under-relaxation: scale the averaged Jacobi correction so chained
    # passes don't overshoot the rest configuration (see params.relaxation).
    active = topo.active[:, None].astype(jnp.float32)
    return pos + params.relaxation * (correction / degree[:, None]) * active


@jax.jit
def project_bending(pos, topo, params):
    """XPBD dihedral bending (Jacobi over unique interior edges)."""
    he_src = topo.he_src
    safe_dest = topo.safe_dest
    interior = topo.interior
    opp_c = topo.opp_c
    opp_d = topo.opp_d

    alpha_tilde = params.bending_compliance / (params.dt * params.dt + EPSILON)

    src_pos = pos[he_src]
    dst_pos = pos[safe_dest]
    diff = dst_pos - src_pos
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + EPSILON)
    dist_safe = jnp.maximum(dist, EPSILON)
    e_hat = diff / dist_safe[:, None]

    pc = pos[opp_c]
    pd = pos[opp_d]

    fn1 = jnp.cross(diff, pc - src_pos)
    fn2 = jnp.cross(src_pos - dst_pos, pd - dst_pos)
    a1 = jnp.sqrt(jnp.sum(fn1 * fn1, axis=-1) + EPSILON)
    a2 = jnp.sqrt(jnp.sum(fn2 * fn2, axis=-1) + EPSILON)
    n1 = fn1 / a1[:, None]
    n2 = fn2 / a2[:, None]

    cos_theta = jnp.sum(n1 * n2, axis=-1)
    sin_theta = jnp.sum(jnp.cross(n1, n2) * e_hat, axis=-1)
    theta = jnp.arctan2(
        jnp.where(interior, sin_theta, 0.0),
        jnp.where(interior, cos_theta, 1.0),
    )

    vc = src_pos - pc
    jc = dst_pos - pc
    area_c = jnp.sqrt(jnp.sum(jnp.cross(vc, jc) ** 2, axis=-1) + EPSILON)
    cot_c = jnp.clip(jnp.sum(vc * jc, axis=-1) / area_c, -5.0, 5.0)

    vd = src_pos - pd
    jd = dst_pos - pd
    area_d = jnp.sqrt(jnp.sum(jnp.cross(vd, jd) ** 2, axis=-1) + EPSILON)
    cot_d = jnp.clip(jnp.sum(vd * jd, axis=-1) / area_d, -5.0, 5.0)

    g_a = (cot_c[:, None] * n1 + cot_d[:, None] * n2) / dist_safe[:, None]
    g_c = dist_safe[:, None] * n1 / a1[:, None]
    g_d = dist_safe[:, None] * n2 / a2[:, None]
    g_b = -(g_a + g_c + g_d)

    sum_sq = jnp.sum(g_a * g_a, axis=-1) + jnp.sum(g_b * g_b, axis=-1) \
        + jnp.sum(g_c * g_c, axis=-1) + jnp.sum(g_d * g_d, axis=-1)

    dlambda = -theta / (sum_sq + alpha_tilde + EPSILON)
    dlambda = jnp.where(interior, dlambda, 0.0)

    int_mask = interior[:, None].astype(jnp.float32)
    corr_a = (dlambda[:, None] * g_a) * int_mask
    corr_b = (dlambda[:, None] * g_b) * int_mask
    corr_c = (dlambda[:, None] * g_c) * int_mask
    corr_d = (dlambda[:, None] * g_d) * int_mask

    correction = jnp.zeros_like(pos)
    correction = correction.at[he_src].add(corr_a)
    correction = correction.at[safe_dest].add(corr_b)
    correction = correction.at[opp_c].add(corr_c)
    correction = correction.at[opp_d].add(corr_d)

    count = jnp.maximum(topo.bend_count_raw, 1.0)

    avg_corr = correction / count[:, None]
    corr_mag = jnp.linalg.norm(avg_corr, axis=-1, keepdims=True)
    max_corr = params.spring_len * 0.1
    avg_corr = jnp.where(corr_mag > max_corr, avg_corr * max_corr / corr_mag, avg_corr)

    # SOR under-relaxation (matches Rust); damps the overshoot that otherwise
    # made stiff bending oscillate into sharper folds rather than flatten them.
    active = topo.active[:, None].astype(jnp.float32)
    return pos + params.relaxation * avg_corr * active


@jax.jit
def collision_candidates(state, sort_order, bstart, bend, cell_coords,
                         neighbor_list):
    """Candidate collision pairs + validity mask for the current cell layout.

    Depends only on the spatial hash (built once per substep) and topology, not
    on the live positions mutated during XPBD iterations — so it is computed
    once per substep and reused across every iteration of project_collision.
    """
    active = state.vertex_idx != -1
    cand_idx, mask = _gather_candidates(cell_coords, active, sort_order, bstart, bend)

    is_neighbor = jnp.any(
        cand_idx[:, :, None] == neighbor_list[:, None, :], axis=-1
    )
    mask = mask & ~is_neighbor
    return cand_idx, mask


@jax.jit
def project_collision(pos, cand_idx, mask, active, params):
    """XPBD collision: push apart non-neighbor vertices within repulsion_dist.

    `cand_idx` / `mask` come from collision_candidates; only the per-candidate
    distance and correction depend on `pos`, so this is the only part that has
    to rerun each XPBD iteration.
    """
    cand_pos = pos[cand_idx]
    diff = pos[:, None, :] - cand_pos
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + EPSILON)

    penetration = params.repulsion_dist - dist
    in_range = (penetration > 0) & mask

    dlambda = -penetration / 2.0
    ratio = penetration / params.repulsion_dist
    corr = -dlambda[:, :, None] * (diff / dist[:, :, None]) * ratio[:, :, None]

    corr_mag = jnp.linalg.norm(corr, axis=-1, keepdims=True)
    max_corr = params.repulsion_dist * 0.2
    corr = jnp.where(corr_mag > max_corr, corr * max_corr / corr_mag, corr)
    corr = jnp.where(in_range[:, :, None], corr, 0.0)

    num_collisions = jnp.sum(in_range.astype(jnp.float32), axis=1, keepdims=True)
    total_corr = corr.sum(axis=1)
    avg_corr = jnp.where(num_collisions > 0,
                          params.repulsion_strength * total_corr / jnp.maximum(num_collisions, 1.0),
                          0.0)

    active_f = active[:, None].astype(jnp.float32)
    return pos + avg_corr * active_f


# ── Lenia CA (Chebyshev) ────────────────────────────────────────────────────

def chebyshev_ring_coeffs(mu, sigma, max_order=MAX_CHEB_ORDER):
    """Chebyshev expansion coefficients for a Gaussian ring kernel."""
    k = np.arange(max_order)
    c_raw = np.exp(-(k - mu) ** 2 / (2 * sigma ** 2 + 1e-8))
    total = c_raw.sum()
    cumsum = np.cumsum(c_raw)
    order = int(np.searchsorted(cumsum, 0.95 * total)) + 1
    order = max(order, 2)
    c = np.zeros(max_order)
    c[:order] = c_raw[:order] / c_raw[:order].sum()
    return jnp.array(c, dtype=jnp.float32), order


@jax.jit
def chebyshev_ca_step(state, coeffs, params):
    """One Chebyshev graph CA step on channel 0 of vertex_state."""
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    he_src = state.half_edge_dest[safe_twin]
    he_dst = state.half_edge_dest
    v_active = (state.vertex_idx != -1).astype(jnp.float32)

    degree = jnp.zeros(MAX_VERTICES).at[he_src].add(he_valid.astype(jnp.float32))
    safe_degree = jnp.maximum(degree, 1.0)

    def W(f):
        s = jnp.zeros(MAX_VERTICES).at[he_src].add(
            jnp.where(he_valid, f[he_dst], 0.0)
        )
        return s / safe_degree

    s = state.vertex_state[:, 0]

    t_prev = s
    t_curr = W(s)
    result = coeffs[0] * t_prev + coeffs[1] * t_curr

    def step(carry, ck):
        tp, tc, res = carry
        tn = 2.0 * W(tc) - tp
        return (tc, tn, res + ck * tn), None

    (_, _, result), _ = jax.lax.scan(step, (t_prev, t_curr, result), coeffs[2:])

    x = result - params.growth_mu
    sigma = params.growth_sigma + EPSILON
    g = 2.0 * jnp.exp(-0.5 * x * x / (sigma * sigma)) - 1.0

    ch0 = s + params.state_dt * g
    ch0 = jnp.clip(ch0, 0.0, 1.0) * v_active
    new_state = state.vertex_state.at[:, 0].set(ch0)
    return state._replace(vertex_state=new_state)


# ── MeshNCA ──────────────────────────────────────────────────────────────────

def _rsh_cart_1(xyz):
    """Real spherical harmonics up to degree 1 (shape ..., 4)."""
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    return jnp.stack(
        [
            jnp.full(xyz.shape[:-1], 0.282094791773878),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
        ],
        -1,
    )


def _relu(x):
    return jnp.maximum(0.0, x)


@jax.jit
def nca_update(state, params):
    """MeshNCA update on full multi-channel vertex_state."""
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]
    he_dst = safe_dest

    edge_pos_src = state.vertex_pos[he_src]
    edge_pos_dst = state.vertex_pos[he_dst]
    edge_vec = edge_pos_dst - edge_pos_src
    edge_dist = jnp.linalg.norm(edge_vec, axis=-1, keepdims=True)
    edge_dir = edge_vec / (edge_dist + EPSILON)
    edge_dist_norm = edge_dist / params.spring_len

    sh = _rsh_cart_1(edge_dir)  # (E, 4)
    feature_diff = state.vertex_state[he_dst] - state.vertex_state[he_src]  # (E, C)
    edge_msg = (edge_dist_norm[..., None] * sh[..., None]
                * feature_diff[..., None, :])  # (E, 4, C)
    edge_msg = edge_msg.reshape(edge_msg.shape[0], -1)
    edge_msg = jnp.where(he_valid[:, None], edge_msg, 0.0)

    z = jnp.zeros((MAX_VERTICES, edge_msg.shape[-1]))
    z = z.at[he_src].add(edge_msg)

    s = state.vertex_state
    inp = jnp.concatenate([s, z], axis=-1)

    mlp = params.vertex_state_mlp_params
    num_layers = len(mlp)
    h = inp
    for i in range(num_layers):
        w, b = mlp[f'layer{i+1}']
        h = jnp.einsum('ij,jk->ik', h, w) + b
        if i < num_layers - 1:
            h = _relu(h)

    v_active = (state.vertex_idx != -1).astype(jnp.float32)[:, None]
    delta = h * params.state_dt * v_active
    new_state = s + delta
    return state._replace(vertex_state=new_state)


# ── Phototropic (light + resources + gravity) ───────────────────────────────
#
# Convention:
#   ch0 = tissue (drives growth via grow_intrinsic_lengths_bipolar)
#   ch1 = light  (recomputed every step from per-vertex normals)
#   ch2 = nutrient (diffuses on the graph + decays + boundary inflow at ground)
#   ch3+ = extra resource channels with the same diffuse/decay rule
#
# All dynamics are deterministic given params. No MLP. Diversity comes from
# the configurable params (light direction/position, gravity, decay rates,
# diffusion rate, etc.).


@jax.jit
def _vertex_normals_from_state(state):
    """Per-vertex normals from active-face triangle normals (unit, masked)."""
    pos = state.vertex_pos
    active_mask = state.face_idx != -1
    face_hes = jnp.where(active_mask, state.face_half_edge, 0)
    safe_next = state.half_edge_next
    safe_dest = state.half_edge_dest
    tri_he = jnp.stack([
        face_hes,
        safe_next[face_hes],
        safe_next[safe_next[face_hes]],
    ], -1)
    fv = safe_dest[tri_he]
    fv = jnp.where(active_mask[:, None], fv, 0)
    v0, v1, v2 = pos[fv[:, 0]], pos[fv[:, 1]], pos[fv[:, 2]]
    fn = jnp.cross(v1 - v0, v2 - v0)
    fn = jnp.where(active_mask[:, None], fn, 0.0)
    vn = jnp.zeros_like(pos)
    vn = vn.at[fv[:, 0]].add(fn)
    vn = vn.at[fv[:, 1]].add(fn)
    vn = vn.at[fv[:, 2]].add(fn)
    norm = jnp.linalg.norm(vn, axis=-1, keepdims=True)
    return vn / jnp.maximum(norm, EPSILON)


@jax.jit
def compute_light_channel(state, params):
    """Light intensity at each vertex from normal·light_dir + 1/r falloff.

    The light direction in params points *toward* the source. Returns a
    scalar in roughly [0, 1] per vertex, zero on the unlit side. Inactive
    vertices get 0.
    """
    active = (state.vertex_idx != -1).astype(jnp.float32)
    normals = _vertex_normals_from_state(state)
    light_dir = params.light_dir / (jnp.linalg.norm(params.light_dir) + EPSILON)
    dot = jnp.sum(normals * light_dir[None, :], axis=-1)
    diffuse = jnp.maximum(dot, 0.0)

    diff = state.vertex_pos - params.light_pos[None, :]
    d = jnp.linalg.norm(diff, axis=-1)
    falloff = params.light_decay_dist / (params.light_decay_dist + d)

    illum = params.light_ambient + (1.0 - params.light_ambient) * diffuse * falloff
    return illum * active


@jax.jit
def diffuse_channel(state, channel, rate):
    """One explicit graph-Laplacian smoothing step on `channel`.

    new[v] = (1 - rate) * old[v] + rate * mean(old[neighbours]).
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]
    he_dst = safe_dest
    s = state.vertex_state[:, channel]
    nbr_sum = jnp.zeros(MAX_VERTICES).at[he_src].add(
        jnp.where(he_valid, s[he_dst], 0.0)
    )
    degree = jnp.zeros(MAX_VERTICES).at[he_src].add(he_valid.astype(jnp.float32))
    avg = nbr_sum / jnp.maximum(degree, 1.0)
    new = (1.0 - rate) * s + rate * avg
    return state._replace(vertex_state=state.vertex_state.at[:, channel].set(new))


def boundary_source(state, channel, value, ground_z, band):
    """Set channel = value on vertices whose z lies within `band` of ground_z."""
    active = state.vertex_idx != -1
    on_ground = active & (state.vertex_pos[:, 2] < ground_z + band)
    new = jnp.where(on_ground, value, state.vertex_state[:, channel])
    return state._replace(vertex_state=state.vertex_state.at[:, channel].set(new))


def project_ground(pos, on_b, params, base_pos=None):
    """Pin boundary vertices and keep every vertex above the ground plane.
    `on_b` is the topology-derived boundary mask (get_on_boundary), precomputed
    once per substep since topology is fixed across XPBD iterations.

    Two pinning modes:
    - `base_pos is None` (default, open-rim shapes like discs): snap only the
      boundary's z to `ground_z`; x/y slide freely so the rim can ruffle/expand.
    - `base_pos` given (hemisphere's fixed bottom circle): restore boundary
      vertices to their stored x/y/z so the base circle is fully immovable,
      matching the Rust port's pinned bottom cap (which fixes all three axes,
      not just z).
    """
    if base_pos is not None:
        pos = jnp.where(on_b[:, None], base_pos, pos)
    else:
        pos = pos.at[:, 2].set(jnp.where(on_b, params.ground_z, pos[:, 2]))
    z = jnp.maximum(pos[:, 2], params.ground_z)
    return pos.at[:, 2].set(z)


# ── Intrinsic Length Growth ──────────────────────────────────────────────────

def _anisotropy_factor(state, safe_dest, he_src, params):
    """Per-half-edge growth multiplier from edge alignment with the preferred
    axis `params.anisotropy_dir` (an arbitrary vec3, normalized here).

    Mirrors the Rust growth.wgsl `mix(1, |dot|, strength)`: |cos| treats the
    two edge orientations symmetrically, and strength ramps from isotropic
    (1.0 everywhere, strength 0) to axis-only (|cos|, strength 1). The caller
    passes the same `safe_dest`/`he_src` it already gathered so we don't redo
    the twin/dest clipping.
    """
    edge = state.vertex_pos[safe_dest] - state.vertex_pos[he_src]
    elen = jnp.linalg.norm(edge, axis=1, keepdims=True)
    axis = params.anisotropy_dir / (jnp.linalg.norm(params.anisotropy_dir) + EPSILON)
    align = jnp.abs((edge / (elen + EPSILON)) @ axis)
    return 1.0 + params.anisotropy_strength * (align - 1.0)


@jax.jit
def grow_intrinsic_lengths(state, params, key, no_grow_v=None):
    """Grow intrinsic edge lengths based on source vertex state channel 0.

    Channel 0 is interpreted as a growth signal in [0, 1] (Lenia) or via
    sigmoid (NCA). Caller is responsible for any pre-clamping/saturation.

    `no_grow_v` (optional bool mask over vertices): half-edges whose source
    vertex is set never grow — used to freeze the hemisphere's pinned base
    circle, mirroring the Rust port's pinned vertices skipping growth.
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]

    s = state.vertex_state[he_src, 0]
    grow = jnp.maximum(2.0 * s - 1.0, 0.0) ** 2

    noise_per_vertex = 0.5 + jax.random.uniform(key, (MAX_VERTICES,))
    noise = noise_per_vertex[he_src]

    aniso = _anisotropy_factor(state, safe_dest, he_src, params)
    delta = grow * noise * params.state_dt * params.growth_rate * aniso
    delta = jnp.where(he_valid, delta, 0.0)
    if no_grow_v is not None:
        delta = jnp.where(no_grow_v[he_src], 0.0, delta)

    new_intrinsic = state.half_edge_intrinsic_len + delta
    new_intrinsic = jnp.minimum(new_intrinsic, params.spring_len * 3.0)
    return state._replace(half_edge_intrinsic_len=new_intrinsic)


@jax.jit
def grow_intrinsic_lengths_phototropic(state, params, key, no_grow_v=None):
    """Growth driven by ch0 (tissue) in [0, 1], no threshold.

    grow = tissue^1.5 (concave-down in [0,1], slow start, fast finish).
    Always positive: light-driven growth never shrinks, mirroring biology.

    `no_grow_v`: see grow_intrinsic_lengths (freezes pinned base vertices).
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]

    s = jnp.clip(state.vertex_state[he_src, 0], 0.0, 1.0)
    grow = s ** 1.5

    noise_per_vertex = 0.5 + jax.random.uniform(key, (MAX_VERTICES,))
    noise = noise_per_vertex[he_src]

    aniso = _anisotropy_factor(state, safe_dest, he_src, params)
    delta = grow * noise * params.state_dt * params.growth_rate * aniso
    delta = jnp.where(he_valid, delta, 0.0)
    if no_grow_v is not None:
        delta = jnp.where(no_grow_v[he_src], 0.0, delta)

    new_intrinsic = state.half_edge_intrinsic_len + delta
    new_intrinsic = jnp.minimum(new_intrinsic, params.spring_len * 3.0)
    return state._replace(half_edge_intrinsic_len=new_intrinsic)


def grow_intrinsic_lengths_bipolar(state, params, key, no_grow_v=None):
    """Bipolar growth: ch0 in [-1, 1] drives signed length change. Negative
    regions shrink; positive regions grow. Allows folds / buckling that the
    one-sided variant cannot reach.

    Lengths clamped to [0.3, 3.0] * spring_len to prevent degenerate edges.

    `no_grow_v`: see grow_intrinsic_lengths (freezes pinned base vertices).
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]

    s = state.vertex_state[he_src, 0]
    grow = jnp.sign(s) * jnp.abs(s) ** 1.5

    noise_per_vertex = 0.5 + jax.random.uniform(key, (MAX_VERTICES,))
    noise = noise_per_vertex[he_src]

    aniso = _anisotropy_factor(state, safe_dest, he_src, params)
    delta = grow * noise * params.state_dt * params.growth_rate * aniso
    delta = jnp.where(he_valid, delta, 0.0)
    if no_grow_v is not None:
        delta = jnp.where(no_grow_v[he_src], 0.0, delta)

    new_intrinsic = state.half_edge_intrinsic_len + delta
    new_intrinsic = jnp.clip(
        new_intrinsic, params.spring_len * 0.3, params.spring_len * 3.0,
    )
    return state._replace(half_edge_intrinsic_len=new_intrinsic)


# ── Combined XPBD Physics Step ──────────────────────────────────────────────

XPBD_ITERATIONS = 20  # static for JIT compatibility (matches the Rust port)


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def batched_physics_step(state, params, cheb_coeffs, key, n_substeps,
                         growth_mode, fix_boundary=False, pin_ground_cap=False):
    """Run n_substeps of XPBD + CA in a single JIT call.

    growth_mode: "lenia" or "nca" (static argument, controls trace).
    fix_boundary (static): when True (open-rim hemisphere), the topological
    boundary loop is the fixed bottom circle — its vertices are fully pinned in
    x/y/z to their position at the start of this call and never grow, matching
    the Rust port's pinned rim. When False (open-rim discs), only z is snapped
    to the ground plane so the rim can ruffle freely.
    pin_ground_cap (static): when True (closed hemisphere, which has no
    boundary), the pinned set is the whole flat bottom cap — every active vertex
    within `ground_pin_strength · spring_len` of the ground plane. Like Rust's
    pinned bottom cap, those vertices are frozen in x/y/z and their edges never
    grow/subdivide, so the closing circle and disc stay flat and static.
    """

    # Topology is constant across substeps (the scan carry holds only
    # positions / state / intrinsic lengths / key), so the neighbor list,
    # boundary mask and the shared topology bundle are computed once here
    # rather than inside every substep / XPBD iteration.
    neighbor_list = build_neighbor_list(state)
    topo = build_xpbd_topo(state)
    active_v = topo.active

    # The pinned-vertex mask: the topological boundary (open rim) or the whole
    # ground-plane cap (closed hemisphere). Pinned vertices are frozen to their
    # current x/y/z (base_pos) and their outgoing edges never grow (no_grow_v).
    if pin_ground_cap:
        cap_band = params.ground_pin_strength * params.spring_len
        on_b = active_v & (state.vertex_pos[:, 2] <= params.ground_z + cap_band)
    else:
        on_b = get_on_boundary(state)
    pin = fix_boundary or pin_ground_cap
    base_pos = state.vertex_pos if pin else None
    no_grow_v = on_b if pin else None

    def substep(carry, _):
        pos, vertex_state, he_intrinsic, key = carry

        st = state._replace(
            vertex_pos=pos,
            vertex_state=vertex_state,
            half_edge_intrinsic_len=he_intrinsic,
        )

        f_ext = compute_external_forces(pos, topo, params)
        pos = predict_positions(st, f_ext, params)

        sort_order, bstart, bend, cell_coords = \
            _build_spatial_hash(pos, active_v, params.repulsion_dist)
        # Candidate pairs depend on the cell layout (fixed for this substep)
        # and topology, not on the positions mutated below — gather once.
        cand_idx, coll_mask = collision_candidates(
            st, sort_order, bstart, bend, cell_coords, neighbor_list
        )

        # Rest lengths depend on intrinsic lengths (per-substep) but not on the
        # positions mutated by the XPBD iterations, so derive them once here.
        rest = (he_intrinsic + he_intrinsic[topo.safe_twin]) * 0.5

        def xpbd_iter(p, _):
            p = project_springs(p, topo, rest, params)
            p = project_bending(p, topo, params)
            p = project_collision(p, cand_idx, coll_mask, active_v, params)
            return p, None

        pos, _ = jax.lax.scan(xpbd_iter, pos, None, length=XPBD_ITERATIONS)

        pos = project_ground(pos, on_b, params, base_pos)
        st = st._replace(vertex_pos=pos)

        key, subkey = jax.random.split(key)

        if growth_mode == 'lenia':
            st = chebyshev_ca_step(st, cheb_coeffs, params)
            st = grow_intrinsic_lengths(st, params, subkey, no_grow_v)
        elif growth_mode == 'nca':
            st = nca_update(st, params)
            vs = st.vertex_state
            ch0 = jax.nn.sigmoid(vs[:, 0])
            ch0 = ch0 * (state.vertex_idx != -1).astype(jnp.float32)
            ch_rest = jnp.clip(vs[:, 1:], -5.0, 5.0)
            vs = jnp.concatenate([ch0[:, None], ch_rest], axis=1)
            st = st._replace(vertex_state=vs)
            st = grow_intrinsic_lengths(st, params, subkey, no_grow_v)
        elif growth_mode == 'nca_bipolar':
            st = nca_update(st, params)
            vs = st.vertex_state
            ch0 = jnp.tanh(vs[:, 0])
            ch0 = ch0 * (state.vertex_idx != -1).astype(jnp.float32)
            ch_rest = jnp.clip(vs[:, 1:], -5.0, 5.0)
            vs = jnp.concatenate([ch0[:, None], ch_rest], axis=1)
            st = st._replace(vertex_state=vs)
            st = grow_intrinsic_lengths_bipolar(st, params, subkey, no_grow_v)
        elif growth_mode == 'phototropic':
            active_f = (state.vertex_idx != -1).astype(jnp.float32)

            light = compute_light_channel(st, params)
            vs = st.vertex_state.at[:, 1].set(light)
            st = st._replace(vertex_state=vs)

            n_res = st.vertex_state.shape[1] - 2
            on_ground = (state.vertex_idx != -1) & (
                st.vertex_pos[:, 2] < params.ground_z + params.spring_len * 0.5
            )
            for ci in range(n_res):
                ch = 2 + ci
                s = st.vertex_state[:, ch]
                s = (1.0 - params.resource_decay * params.state_dt) * s
                st = st._replace(
                    vertex_state=st.vertex_state.at[:, ch].set(s),
                )
                st = diffuse_channel(st, ch, params.resource_diffusion)
                cur = st.vertex_state[:, ch]
                refreshed = jnp.where(
                    on_ground,
                    jnp.maximum(cur, params.ground_source_value),
                    cur,
                )
                st = st._replace(
                    vertex_state=st.vertex_state.at[:, ch].set(refreshed),
                )

            tissue = st.vertex_state[:, 0]
            nutrient = st.vertex_state[:, 2]
            production = light * nutrient
            head = 1.0 - tissue / params.tissue_saturation
            tissue_next = tissue + params.state_dt * (
                production * head - params.tissue_decay * tissue
            )
            tissue_next = jnp.clip(tissue_next, 0.0, 1.0) * active_f
            st = st._replace(
                vertex_state=st.vertex_state.at[:, 0].set(tissue_next),
            )

            st = grow_intrinsic_lengths_phototropic(st, params, subkey, no_grow_v)
        else:
            raise ValueError(f"Unknown growth_mode: {growth_mode}")

        return (st.vertex_pos, st.vertex_state,
                st.half_edge_intrinsic_len, key), None

    init = (state.vertex_pos, state.vertex_state,
            state.half_edge_intrinsic_len, key)
    (pos, vs, hil, key), _ = jax.lax.scan(
        substep, init, None, length=n_substeps
    )

    return state._replace(
        vertex_pos=pos, vertex_state=vs,
        half_edge_intrinsic_len=hil,
    ), key


# ── Diagnostics ──────────────────────────────────────────────────────────────

@jax.jit
def measure_dihedral_angles(state):
    """Absolute dihedral angles (degrees) for all interior edges + summary."""
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    safe_next = jnp.clip(state.half_edge_next, 0)
    he_src = safe_dest[safe_twin]

    pos = state.vertex_pos
    src_pos = pos[he_src]
    dst_pos = pos[safe_dest]
    diff = dst_pos - src_pos
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + EPSILON)
    e_hat = diff / jnp.maximum(dist[:, None], EPSILON)

    he_face = state.half_edge_face
    twin_face = he_face[safe_twin]
    he_idx = jnp.arange(MAX_HALF_EDGES)
    interior = he_valid & (he_face >= 0) & (twin_face >= 0) \
        & (state.half_edge_twin >= 0) & (he_idx < state.half_edge_twin)

    opp_c = safe_dest[safe_next]
    safe_twin_next = jnp.clip(state.half_edge_next[safe_twin], 0)
    opp_d = safe_dest[safe_twin_next]

    pc = pos[opp_c]
    pd = pos[opp_d]

    fn1 = jnp.cross(diff, pc - src_pos)
    fn2 = jnp.cross(src_pos - dst_pos, pd - dst_pos)
    a1 = jnp.sqrt(jnp.sum(fn1 * fn1, axis=-1) + EPSILON)
    a2 = jnp.sqrt(jnp.sum(fn2 * fn2, axis=-1) + EPSILON)
    n1 = fn1 / a1[:, None]
    n2 = fn2 / a2[:, None]

    cos_theta = jnp.sum(n1 * n2, axis=-1)
    sin_theta = jnp.sum(jnp.cross(n1, n2) * e_hat, axis=-1)
    theta = jnp.arctan2(
        jnp.where(interior, sin_theta, 0.0),
        jnp.where(interior, cos_theta, 1.0),
    )

    abs_theta_deg = jnp.abs(theta) * (180.0 / jnp.pi)
    abs_theta_deg = jnp.where(interior, abs_theta_deg, 0.0)

    n_interior = jnp.sum(interior.astype(jnp.int32))
    max_angle = jnp.max(abs_theta_deg)
    mean_angle = jnp.sum(abs_theta_deg) / jnp.maximum(n_interior.astype(jnp.float32), 1.0)
    return abs_theta_deg, max_angle, mean_angle, n_interior


def detect_boundary(state):
    """Boolean (MAX_VERTICES,) marking vertices touching a boundary half-edge."""
    boundary = np.zeros(MAX_VERTICES, dtype=bool)
    he_face = np.array(state.half_edge_face)
    he_dest = np.array(state.half_edge_dest)
    he_twin = np.array(state.half_edge_twin)
    he_valid = np.array(state.half_edge_idx != -1)

    for i in range(MAX_HALF_EDGES):
        if he_valid[i] and he_face[i] == -1:
            src = he_dest[max(he_twin[i], 0)]
            dst = he_dest[i]
            if 0 <= src < MAX_VERTICES:
                boundary[src] = True
            if 0 <= dst < MAX_VERTICES:
                boundary[dst] = True
    return jnp.array(boundary)


@jax.jit
def get_on_boundary(state):
    """Mark every active vertex that touches at least one boundary half-edge."""
    he_valid = state.half_edge_idx != -1
    on_boundary_he = (state.half_edge_face == -1) & he_valid
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]

    on_b = jnp.zeros(MAX_VERTICES, dtype=bool)
    on_b = on_b.at[safe_dest].max(on_boundary_he)
    on_b = on_b.at[he_src].max(on_boundary_he)
    return on_b & (state.vertex_idx != -1)


# ── Rendering (nvdiffrast) ───────────────────────────────────────────────────

@jax.jit
def extract_mesh_for_nvdiffrast(state):
    """Extract verts, faces, n_active_faces for rendering."""
    verts = state.vertex_pos
    active_mask = state.face_idx != -1
    face_half_edges = jnp.where(active_mask, state.face_half_edge, 0)

    tri_he = jnp.stack([
        face_half_edges,
        state.half_edge_next[face_half_edges],
        state.half_edge_next[state.half_edge_next[face_half_edges]],
    ], -1)
    faces = state.half_edge_dest[tri_he]
    faces = jnp.where(active_mask[:, None], faces, 0)
    n_active = jnp.sum(active_mask)
    return verts, faces, n_active


@jax.jit
def compute_vertex_normals_jax(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = jnp.cross(v1 - v0, v2 - v0)
    vn = jnp.zeros_like(verts)
    vn = vn.at[faces[:, 0]].add(face_normals)
    vn = vn.at[faces[:, 1]].add(face_normals)
    vn = vn.at[faces[:, 2]].add(face_normals)
    norm = jnp.linalg.norm(vn, axis=-1, keepdims=True)
    return vn / jnp.maximum(norm, 1e-8)


@jax.jit
def compute_vertex_colors_jax(state):
    """Vertex colors from channel 0 (Lenia-style blue→yellow) or first 3 channels (NCA)."""
    state_rgb = state.vertex_state[:, :3]
    colors = 0.5 * (jnp.tanh(state_rgb) + 1.0)
    pad = 3 - colors.shape[1]
    if pad > 0:
        colors = jnp.pad(colors, ((0, 0), (0, pad)), mode='edge')
    return colors


def jax_to_torch_gpu(arr):
    return torch.from_dlpack(jax.dlpack.to_dlpack(arr))


class JAXNvdiffrastRenderer:
    def __init__(self, width, height, device='cuda'):
        self.width = width
        self.height = height
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.glctx = dr.RasterizeCudaContext()
        self.center_x = width / 2
        self.center_y = height / 2
        self.proj_scale = 2.0 / max(width, height)
        self.light_dir = torch.tensor([0.3, 0.3, 1.0], device=self.device, dtype=torch.float32)
        self.light_dir = self.light_dir / torch.norm(self.light_dir)
        self.ambient = 0.25
        self.diffuse = 0.75

    def render(self, state, rot=None, flip_z=True):
        """Render the mesh to an (H, W, 3) uint8 image.

        `rot` (optional 3x3): a rotation applied to the centered geometry
        before the orthographic projection, so callers can orbit the camera
        (see `turntable_rot` / `render_turntable_gif`). The default view looks
        down +z (top-down); pass a turntable rotation for a readable 3/4 view.
        Framing uses the rotation-invariant bounding sphere so the form does
        not zoom in/out as it spins.

        `flip_z` (default True): negate the world z axis before projecting, so
        the grown dome reads the way we want on screen (apex pointing up). This
        only affects rendering — the simulation/export geometry is untouched.
        """
        verts_j, faces_j, n_active = extract_mesh_for_nvdiffrast(state)
        normals_j = compute_vertex_normals_jax(verts_j, faces_j)
        colors_j = compute_vertex_colors_jax(state)
        jax.block_until_ready(verts_j)

        verts = jax_to_torch_gpu(verts_j).to(self.device)
        faces = jax_to_torch_gpu(faces_j).to(torch.int32).to(self.device)
        normals = jax_to_torch_gpu(normals_j).to(self.device)
        vertex_colors = jax_to_torch_gpu(colors_j).to(self.device)
        n = int(n_active)
        faces = faces[:n].contiguous()

        # Auto-fit the orthographic camera to the live mesh: center on the
        # bounding-box midpoint, scale by the bounding-sphere radius (which is
        # rotation invariant, so an orbiting turntable keeps a steady framing).
        if n > 0:
            used = faces.flatten().unique()
            used_v = verts[used]
            center = 0.5 * (used_v.min(dim=0).values + used_v.max(dim=0).values)
            radius = (used_v - center).norm(dim=1).max()
            scale = 1.7 / (2.0 * radius + 1e-6)  # fill ~85% of the frame
        else:
            center = torch.tensor([self.center_x, self.center_y, 0.0],
                                  device=self.device)
            scale = self.proj_scale

        centered = verts - center
        if flip_z:
            # Reflect through the ground plane (negate world z). For a spinning
            # turntable this reads as "flipped upside down"; the doubled faces
            # keep both sides lit so the reflected winding is invisible.
            flip = torch.tensor([1.0, 1.0, -1.0], device=self.device,
                                 dtype=verts.dtype)
            centered = centered * flip
            normals = normals * flip
        if rot is not None:
            R = torch.as_tensor(rot, device=self.device, dtype=verts.dtype)
            centered = centered @ R.T
            normals = normals @ R.T
        verts_clip = centered * scale
        verts_homo = torch.cat([verts_clip, torch.ones_like(verts_clip[:, :1])], dim=-1)
        verts_homo = verts_homo.unsqueeze(0).contiguous()

        with torch.no_grad():
            faces_back = faces[:, [0, 2, 1]]
            faces_double = torch.cat([faces_back, faces], dim=0)
            normals_double = torch.cat([-normals, normals], dim=0)
            colors_double = torch.cat([vertex_colors, vertex_colors], dim=0)

            rast_out, _ = dr.rasterize(self.glctx, verts_homo, faces_double, (self.height, self.width))
            normals_interp, _ = dr.interpolate(normals_double.unsqueeze(0).contiguous(),
                                               rast_out, faces_double)
            colors_interp, _ = dr.interpolate(colors_double.unsqueeze(0).contiguous(),
                                              rast_out, faces_double)

            ni = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)
            light_dir = self.light_dir.view(1, 1, 1, 3)
            # Two-sided diffuse (|n·l|): the mesh is closed and we orbit all the
            # way under it, so back-facing surfaces (e.g. the downward-facing
            # bottom cap) must still be lit — otherwise the flat base renders
            # black against the background and reads as a hole.
            ndotl = torch.abs(torch.sum(ni * light_dir, dim=-1, keepdim=True))
            mesh_color = colors_interp * (self.ambient + self.diffuse * ndotl)
            mesh_color = dr.antialias(mesh_color, rast_out, verts_homo, faces_double)

            tri_id = rast_out[..., 3:4]
            mask = (tri_id > 0).float()
            bg = torch.full_like(mesh_color, 0.1)
            color = mesh_color * mask + bg * (1 - mask)
            color = (color[0] * 255).clamp(0, 255).to(torch.uint8)

        return color.cpu().numpy()


def turntable_rot(azimuth, elevation=0.35):
    """3x3 rotation for an orthographic turntable camera orbiting +z-up.

    `azimuth` spins about the world z (growth) axis; `elevation` (radians)
    tilts the camera above the horizon (default ~20°). The returned matrix
    maps world coordinates into view space where +x=screen-right,
    +y=screen-up, +z=toward-camera, so the hemisphere's z axis reads as
    "up" on screen instead of the default top-down view.
    """
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    ce, se = np.cos(elevation), np.sin(elevation)
    # Camera direction (center -> camera): orbit in azimuth, lifted by el.
    cam = np.array([ce * ca, ce * sa, se], dtype=np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(world_up, cam)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(cam, right)
    return np.stack([right, up, cam], axis=0).astype(np.float32)


def render_turntable_gif(renderer, state, path, n_frames=48, elevation=0.0,
                         elev_amp=0.9, duration=80, loop=0):
    """Render `state` as a rotating turntable GIF (one full revolution).

    The camera elevation oscillates sinusoidally over the revolution
    (`elevation` midline ± `elev_amp` radians, one full cycle so it loops
    seamlessly): it rises to look down on the top, then dips below the horizon
    to look up at the underside, so a single GIF shows both faces of the mesh.
    Set `elev_amp=0` for a classic fixed-elevation turntable.
    """
    import imageio.v3 as iio
    frames = [
        renderer.render(
            state,
            rot=turntable_rot(
                2 * np.pi * i / n_frames,
                elevation + elev_amp * np.sin(2 * np.pi * i / n_frames),
            ),
        )
        for i in range(n_frames)
    ]
    iio.imwrite(path, frames, duration=duration, loop=loop)
    return path


def save_mesh_ply(state, path, include_colors=True):
    """Export the live mesh (active faces only, re-indexed) to a binary PLY.

    Vertex colors come from channel 0..2 of the per-vertex state (the same
    mapping the renderer uses). Writes binary little-endian PLY directly so
    there is no third-party mesh-IO dependency. Returns (n_verts, n_faces).
    """
    verts_j, faces_j, n_active = extract_mesh_for_nvdiffrast(state)
    n_active = int(n_active)
    faces_np = np.asarray(faces_j)[:n_active]
    used = np.unique(faces_np)
    remap = -np.ones(int(verts_j.shape[0]), np.int64)
    remap[used] = np.arange(len(used))
    verts = np.asarray(verts_j)[used].astype('<f4')
    faces = remap[faces_np].astype('<i4')
    nv, nf = len(verts), len(faces)

    header = [b"ply", b"format binary_little_endian 1.0",
              b"element vertex %d" % nv,
              b"property float x", b"property float y", b"property float z"]
    if include_colors:
        header += [b"property uchar red", b"property uchar green",
                   b"property uchar blue"]
    header += [b"element face %d" % nf,
               b"property list uchar int vertex_indices", b"end_header"]

    # Interleave per-vertex records as a structured array.
    if include_colors:
        colors_j = compute_vertex_colors_jax(state)
        vc = (np.asarray(colors_j)[used] * 255).clip(0, 255).astype(np.uint8)
        vrec = np.empty(nv, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                   ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        vrec['x'], vrec['y'], vrec['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
        vrec['r'], vrec['g'], vrec['b'] = vc[:, 0], vc[:, 1], vc[:, 2]
    else:
        vrec = verts

    frec = np.empty(nf, dtype=[('n', 'u1'), ('v', '<i4', 3)])
    frec['n'] = 3
    frec['v'] = faces

    with open(path, 'wb') as f:
        f.write(b"\n".join(header) + b"\n")
        f.write(vrec.tobytes())
        f.write(frec.tobytes())
    return nv, nf


def enforce_boundary(state):
    pos = jnp.clip(
        state.vertex_pos,
        a_min=jnp.array(WORLD_MIN, dtype=jnp.float32),
        a_max=jnp.array(WORLD_MAX, dtype=jnp.float32),
    )
    return state._replace(vertex_pos=pos)


# ── Main Loop ────────────────────────────────────────────────────────────────

def seed_boundary_state(state, channel=0, value=1.0):
    """Set channel `channel` of vertex_state to `value` on boundary vertices."""
    on_b = get_on_boundary(state)
    new_ch = jnp.where(on_b, value, state.vertex_state[:, channel])
    return state._replace(vertex_state=state.vertex_state.at[:, channel].set(new_ch))


def seed_random_state(state, key, channel=0, low=0.0, high=1.0):
    """Set channel `channel` of vertex_state to random uniform [low, high] on
    active vertices (vertex_idx != -1). Used for closed meshes (e.g. sphere)
    where there is no boundary to seed."""
    active = state.vertex_idx != -1
    rnd = jax.random.uniform(key, (MAX_VERTICES,), minval=low, maxval=high)
    new_ch = jnp.where(active, rnd, state.vertex_state[:, channel])
    return state._replace(vertex_state=state.vertex_state.at[:, channel].set(new_ch))


def main():
    width, height = 800, 800
    key = jax.random.PRNGKey(42)

    # Pick a growth mode here:
    #   "lenia": scalar Chebyshev graph CA
    #   "nca":   MeshNCA with MLP
    growth_mode = 'lenia'

    if growth_mode == 'lenia':
        state_dims = 1
        mlp_params = {}
    else:
        state_dims = 8
        key, mlp_key = jax.random.split(key)
        mlp_params = make_mlp_params(mlp_key, state_dims,
                                     hidden_state_dims=32, num_mlp_layers=2)

    params = default_params(vertex_state_mlp_params=mlp_params)
    state = make_circle(width, height, params, n_rings=5, state_dims=state_dims)
    state = seed_boundary_state(state, channel=0, value=1.0)

    cheb_coeffs, cheb_order = chebyshev_ring_coeffs(
        params.kernel_mu, params.kernel_sigma
    )
    print(f"Chebyshev order: {cheb_order}")

    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    renderer = JAXNvdiffrastRenderer(width, height)

    frame = 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        key, subkey = jax.random.split(key)
        state, key = batched_physics_step(
            state, params, cheb_coeffs, subkey, 1, growth_mode
        )

        if frame % 10 == 0:
            state, key, n_verts = split_long_edges(state, params, key)
            if frame % 20 == 0:
                state = refine_mesh(state)
            n_faces = int(jnp.sum(state.face_idx != -1))
            if frame % 50 == 0:
                _, max_angle, mean_angle, _ = measure_dihedral_angles(state)
                print(f"Frame {frame}: {n_verts} verts, {n_faces} faces, "
                      f"max_angle={float(max_angle):.1f}°, "
                      f"mean_angle={float(mean_angle):.1f}°")

        image = renderer.render(state)
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)
        frame += 1

    pygame.quit()


if __name__ == '__main__':
    main()
