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

import jax
import jax.numpy as jnp
import numpy as np
import pygame
import torch

import nvdiffrast.torch as dr


# ── Constants ────────────────────────────────────────────────────────────────

EPSILON = 1e-6
MAX_VERTICES = 12000
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
])


def default_params(**overrides):
    defaults = dict(
        spring_len=30.0,
        compliance=0.0,
        bending_compliance=0.001,
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


def add_external_triangle(state, va, vb):
    """Add a triangle outside boundary edge va->vb."""
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
def compute_external_forces(state, params):
    """Bulge (boundary outward push) + Laplacian smoothing."""
    pos = state.vertex_pos
    active = (state.vertex_idx != -1)[:, None].astype(jnp.float32)

    boundary = (state.half_edge_face == -1) & (state.half_edge_idx != -1)
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    src_idx = state.half_edge_dest[safe_twin]
    dst_idx = state.half_edge_dest

    edge_vec = pos[dst_idx] - pos[src_idx]
    safe_twin_next = jnp.clip(state.half_edge_next[safe_twin], 0)
    next_dst = state.half_edge_dest[safe_twin_next]
    next_vec = pos[next_dst] - pos[src_idx]

    surface_n = jnp.cross(edge_vec, next_vec)
    edge_n = jnp.cross(edge_vec, surface_n)
    edge_n_len = jnp.linalg.norm(edge_n, axis=-1, keepdims=True)
    edge_n = edge_n / jnp.maximum(edge_n_len, EPSILON)
    edge_n = jnp.where(boundary[:, None], edge_n, 0.0)

    bulge = jnp.zeros_like(pos)
    bulge = bulge.at[src_idx].add(edge_n)
    bulge = bulge.at[dst_idx].add(edge_n)
    fnorm = jnp.linalg.norm(bulge, axis=-1, keepdims=True)
    bulge = bulge / jnp.maximum(fnorm, EPSILON)

    he_valid = state.half_edge_idx != -1
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]
    dst_pos = pos[safe_dest]

    neighbor_sum = jnp.zeros_like(pos).at[he_src].add(
        jnp.where(he_valid[:, None], dst_pos, 0.0)
    )
    degree = jnp.zeros(MAX_VERTICES).at[he_src].add(he_valid.astype(jnp.float32))
    centroid = neighbor_sum / jnp.maximum(degree[:, None], 1.0)
    laplacian = centroid - pos

    forces = params.bulge_strength * bulge \
        + params.stiffness * 0.5 * laplacian * active
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
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]

    neighbors = jnp.full((MAX_VERTICES, MAX_FAN_SIZE), -1, jnp.int32)
    count = jnp.zeros(MAX_VERTICES, jnp.int32)

    def add_edge(carry, he_i):
        nbrs, cnt = carry
        src = he_src[he_i]
        dst = safe_dest[he_i]
        valid = he_valid[he_i]
        slot = cnt[src]
        in_range = slot < MAX_FAN_SIZE
        do_write = valid & in_range
        nbrs = nbrs.at[src, slot].set(jnp.where(do_write, dst, nbrs[src, slot]))
        cnt = cnt.at[src].add(jnp.where(do_write, 1, 0))
        return (nbrs, cnt), None

    (neighbors, _), _ = jax.lax.scan(
        add_edge, (neighbors, count), jnp.arange(MAX_HALF_EDGES)
    )
    return neighbors


# ── XPBD Constraints ─────────────────────────────────────────────────────────

@jax.jit
def project_springs(pos, state, params):
    """XPBD spring: enforce dist = intrinsic rest length (Jacobi over half-edges)."""
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]

    alpha_tilde = params.compliance / (params.dt * params.dt + EPSILON)

    src_pos = pos[he_src]
    dst_pos = pos[safe_dest]
    d = src_pos - dst_pos
    dist = jnp.sqrt(jnp.sum(d * d, axis=-1) + EPSILON)
    dist_safe = jnp.maximum(dist, EPSILON)

    rest = (state.half_edge_intrinsic_len + state.half_edge_intrinsic_len[safe_twin]) * 0.5
    C = dist - rest
    dlambda = -C / (2.0 + alpha_tilde)
    edge_corr = dlambda[:, None] * (d / dist_safe[:, None])

    max_corr = rest * 0.2
    corr_mag = jnp.linalg.norm(edge_corr, axis=-1, keepdims=True)
    edge_corr = jnp.where(corr_mag > max_corr[:, None],
                           edge_corr * max_corr[:, None] / corr_mag,
                           edge_corr)
    edge_corr = jnp.where(he_valid[:, None], edge_corr, 0.0)

    correction = jnp.zeros_like(pos).at[he_src].add(edge_corr)
    degree = jnp.zeros(MAX_VERTICES).at[he_src].add(he_valid.astype(jnp.float32))
    degree = jnp.maximum(degree, 1.0)

    active = (state.vertex_idx != -1)[:, None].astype(jnp.float32)
    return pos + (correction / degree[:, None]) * active


@jax.jit
def project_bending(pos, state, params):
    """XPBD dihedral bending (Jacobi over unique interior edges)."""
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    safe_next = jnp.clip(state.half_edge_next, 0)
    he_src = safe_dest[safe_twin]

    he_face = state.half_edge_face
    twin_face = he_face[safe_twin]
    he_idx = jnp.arange(MAX_HALF_EDGES)
    interior = he_valid & (he_face >= 0) & (twin_face >= 0) \
        & (state.half_edge_twin >= 0) & (he_idx < state.half_edge_twin)

    alpha_tilde = params.bending_compliance / (params.dt * params.dt + EPSILON)

    src_pos = pos[he_src]
    dst_pos = pos[safe_dest]
    diff = dst_pos - src_pos
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + EPSILON)
    dist_safe = jnp.maximum(dist, EPSILON)
    e_hat = diff / dist_safe[:, None]

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

    count = jnp.zeros(MAX_VERTICES)
    int_f = interior.astype(jnp.float32)
    count = count.at[he_src].add(int_f)
    count = count.at[safe_dest].add(int_f)
    count = count.at[opp_c].add(int_f)
    count = count.at[opp_d].add(int_f)
    count = jnp.maximum(count, 1.0)

    avg_corr = correction / count[:, None]
    corr_mag = jnp.linalg.norm(avg_corr, axis=-1, keepdims=True)
    max_corr = params.spring_len * 0.1
    avg_corr = jnp.where(corr_mag > max_corr, avg_corr * max_corr / corr_mag, avg_corr)

    active = (state.vertex_idx != -1)[:, None].astype(jnp.float32)
    return pos + avg_corr * active


@jax.jit
def project_collision(pos, state, params, sort_order, bstart, bend,
                      cell_coords, neighbor_list):
    """XPBD collision: push apart non-neighbor vertices within repulsion_dist."""
    N = MAX_VERTICES
    active = state.vertex_idx != -1
    cand_idx, mask = _gather_candidates(cell_coords, active, sort_order, bstart, bend)

    is_neighbor = jnp.any(
        cand_idx[:, :, None] == neighbor_list[:, None, :], axis=-1
    )
    mask = mask & ~is_neighbor

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


# ── Intrinsic Length Growth ──────────────────────────────────────────────────

@jax.jit
def grow_intrinsic_lengths(state, params, key):
    """Grow intrinsic edge lengths based on source vertex state channel 0.

    Channel 0 is interpreted as a growth signal in [0, 1] (Lenia) or via
    sigmoid (NCA). Caller is responsible for any pre-clamping/saturation.
    """
    he_valid = state.half_edge_idx != -1
    safe_twin = jnp.clip(state.half_edge_twin, 0)
    safe_dest = jnp.clip(state.half_edge_dest, 0)
    he_src = safe_dest[safe_twin]

    s = state.vertex_state[he_src, 0]
    grow = jnp.maximum(2.0 * s - 1.0, 0.0) ** 2

    noise_per_vertex = 0.5 + jax.random.uniform(key, (MAX_VERTICES,))
    noise = noise_per_vertex[he_src]

    delta = grow * noise * params.state_dt * params.growth_rate
    delta = jnp.where(he_valid, delta, 0.0)

    new_intrinsic = state.half_edge_intrinsic_len + delta
    new_intrinsic = jnp.minimum(new_intrinsic, params.spring_len * 3.0)
    return state._replace(half_edge_intrinsic_len=new_intrinsic)


# ── Combined XPBD Physics Step ──────────────────────────────────────────────

XPBD_ITERATIONS = 10  # static for JIT compatibility


@functools.partial(jax.jit, static_argnums=(4, 5))
def batched_physics_step(state, params, cheb_coeffs, key, n_substeps, growth_mode):
    """Run n_substeps of XPBD + CA in a single JIT call.

    growth_mode: "lenia" or "nca" (static argument, controls trace).
    """

    def substep(carry, _):
        pos, vertex_state, he_intrinsic, key = carry

        st = state._replace(
            vertex_pos=pos,
            vertex_state=vertex_state,
            half_edge_intrinsic_len=he_intrinsic,
        )

        f_ext = compute_external_forces(st, params)
        pos = predict_positions(st, f_ext, params)

        active_v = state.vertex_idx != -1
        sort_order, bstart, bend, cell_coords = \
            _build_spatial_hash(pos, active_v, params.repulsion_dist)
        neighbor_list = build_neighbor_list(st)

        def xpbd_iter(p, _):
            p = project_springs(p, st._replace(vertex_pos=p), params)
            p = project_bending(p, st._replace(vertex_pos=p), params)
            p = project_collision(p, st, params,
                                  sort_order, bstart, bend, cell_coords,
                                  neighbor_list)
            return p, None

        pos, _ = jax.lax.scan(xpbd_iter, pos, None, length=XPBD_ITERATIONS)

        st = st._replace(vertex_pos=pos)

        if growth_mode == 'lenia':
            st = chebyshev_ca_step(st, cheb_coeffs, params)
        elif growth_mode == 'nca':
            st = nca_update(st, params)
            # ch0 → sigmoid (needed in [0,1] by grow_intrinsic_lengths).
            # ch1..N: hard-clip wide enough not to interfere with normal dynamics
            # but tight enough to stop untrained random-MLP runaway.
            vs = st.vertex_state
            ch0 = jax.nn.sigmoid(vs[:, 0])
            ch0 = ch0 * (state.vertex_idx != -1).astype(jnp.float32)
            ch_rest = jnp.clip(vs[:, 1:], -5.0, 5.0)
            vs = jnp.concatenate([ch0[:, None], ch_rest], axis=1)
            st = st._replace(vertex_state=vs)
        else:
            raise ValueError(f"Unknown growth_mode: {growth_mode}")

        key, subkey = jax.random.split(key)
        st = grow_intrinsic_lengths(st, params, subkey)

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

    def render(self, state):
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

        verts_clip = verts.clone()
        verts_clip[:, 0] = (verts[:, 0] - self.center_x) * self.proj_scale
        verts_clip[:, 1] = (verts[:, 1] - self.center_y) * self.proj_scale
        verts_clip[:, 2] = verts[:, 2] * self.proj_scale
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
            ndotl = torch.clamp(torch.sum(ni * light_dir, dim=-1, keepdim=True), 0, 1)
            mesh_color = colors_interp * (self.ambient + self.diffuse * ndotl)
            mesh_color = dr.antialias(mesh_color, rast_out, verts_homo, faces_double)

            tri_id = rast_out[..., 3:4]
            mask = (tri_id > 0).float()
            bg = torch.full_like(mesh_color, 0.1)
            color = mesh_color * mask + bg * (1 - mask)
            color = (color[0] * 255).clamp(0, 255).to(torch.uint8)

        return color.cpu().numpy()


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
