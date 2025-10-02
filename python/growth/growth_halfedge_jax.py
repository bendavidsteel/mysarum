import collections
import functools

import jax
import jax.numpy as jnp
from jax import lax
import pygame
import numpy as np
from typing import NamedTuple


MAX_VERTICES = 1000
EPSILON = 0.000001

MeshState = collections.namedtuple('MeshState', [
    'face_idx',
    'face_half_edge',
    'vertex_idx',
    'vertex_half_edge',
    'vertex_pos',
    'half_edge_idx',
    'half_edge_twin',
    'half_edge_dest',
    'half_edge_face',
    'half_edge_next',
    'half_edge_prev',
    'vertex_suitability'
])

MeshParams = collections.namedtuple('MeshParams', [
    'spring_len',
    'elastic_constant',
    'repulsion_distance',
    'repulsion_strength',
    'bulge_strength',
    'planar_strength'
])

def create_initial_state(num_dims=2):
    """Create initial mesh state"""
    return MeshState(
        face_idx=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        face_half_edge=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        vertex_idx=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        vertex_half_edge=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        vertex_pos=jnp.full((MAX_VERTICES, num_dims), -1, dtype=jnp.float32),
        half_edge_idx=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        half_edge_twin=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        half_edge_dest=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        half_edge_face=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        half_edge_next=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        half_edge_prev=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        vertex_suitability=jnp.zeros((MAX_VERTICES,), dtype=jnp.float32)
    )

@jax.jit
def calculate_spring_force(state, params):
    """Calculate spring forces"""
    edges = jnp.stack([state.half_edge_dest, state.half_edge_dest[state.half_edge_prev]])
    edge_pos = state.vertex_pos[edges]
    edge_vector = edge_pos[0] - edge_pos[1]
    edge_lengths = jnp.linalg.norm(edge_vector, axis=1)
    
    safe_lengths = jnp.maximum(edge_lengths, EPSILON)
    edge_force = -1 * (edge_lengths[:, None] - params.spring_len) * params.elastic_constant * (edge_vector / safe_lengths[:, None])
    
    # Mask out invalid edges
    valid_mask = (state.half_edge_idx != -1) & (edges[0] != -1) & (edges[1] != -1)
    edge_force = jnp.where(valid_mask[:, None], edge_force, 0)
    
    # Accumulate forces at vertices
    vertex_force = jnp.zeros_like(state.vertex_pos)
    vertex_force = vertex_force.at[edges[0]].add(edge_force)
    vertex_force = vertex_force.at[edges[1]].add(-edge_force)
    
    return vertex_force

@jax.jit
def calculate_repulsion_force(state, params):
    """Calculate repulsion forces"""
    active_mask = state.vertex_idx != -1
    
    vertex_diff = state.vertex_pos[:, None, :] - state.vertex_pos[None, :, :]
    vertex_dist = jnp.sqrt(jnp.sum(vertex_diff ** 2, axis=-1) + EPSILON)
    
    dist_ratio = jnp.maximum(0, (params.repulsion_distance - vertex_dist) / params.repulsion_distance)
    repulsion_force = (dist_ratio ** 2)[:, :, None] * (vertex_diff / vertex_dist[:, :, None])
    
    # Mask out inactive vertices and self-interactions
    mask = jnp.outer(active_mask, active_mask) & ~jnp.eye(MAX_VERTICES, dtype=bool)
    repulsion_force = jnp.where(mask[:, :, None], repulsion_force, 0)
    
    repulsion_force_mag = jnp.linalg.norm(repulsion_force, axis=-1).sum(axis=1)
    repulsion_force_sum = repulsion_force.sum(axis=1)
    
    return repulsion_force_mag, repulsion_force_sum

@jax.jit
def calculate_planar_force(state, params):
    """Calculate planar forces to keep mesh flat"""
    # for each vertex, sum its neighbour positions and get its edge count
    vertex_sum = jnp.zeros_like(state.vertex_pos)
    edge_count = jnp.zeros((state.vertex_idx.shape[0],), dtype=int)
    start_half_edge = state.vertex_half_edge.copy()
    start_vertex = state.half_edge_dest[start_half_edge]
    vertex_a = start_vertex.copy()
    half_edge_a = start_half_edge.copy()
    iterating = jnp.ones_like(half_edge_a)
    
    def cond_func(carry):
        vertex_sum, edge_count, half_edge_a, vertex_a, iterating = carry
        return jnp.any(iterating)
    
    def body_func(carry):
        vertex_sum, edge_count, half_edge_a, vertex_a, iterating = carry
        half_edge_b = state.half_edge_next[state.half_edge_twin[half_edge_a]]
        vertex_b = state.half_edge_dest[half_edge_b]
        iterating = iterating & (vertex_b != start_vertex)
        vertex_sum += (state.vertex_pos[vertex_b] * iterating[:, np.newaxis])
        edge_count += iterating
        return (vertex_sum, edge_count, half_edge_b, vertex_b, iterating)

    vertex_sum, edge_count, _, _, _ = jax.lax.while_loop(
        cond_func,
        body_func,
        (vertex_sum, edge_count, half_edge_a, vertex_a, iterating)
    )

    neighbour_avg = vertex_sum / (edge_count[:, np.newaxis] + EPSILON)
    planar_force = (neighbour_avg - state.vertex_pos) * (state.vertex_idx != -1)[:, np.newaxis]
    return planar_force


@jax.jit
def calculate_bulge_force(state, params):
    """Calculate bulge forces for boundary edges"""
    # Find boundary half edges
    boundary_mask = (state.half_edge_face == -1) & (state.half_edge_idx != -1)
    
    # Get edge vectors
    edges = jnp.stack([state.half_edge_dest, state.half_edge_dest[state.half_edge_twin]])
    edge_pos = state.vertex_pos[edges]
    edge_vector = edge_pos[1] - edge_pos[0]
    
    # Calculate normals (rotate 90 degrees)
    edge_normal = jnp.stack([-edge_vector[:, 1], edge_vector[:, 0]], axis=-1)
    edge_normal = edge_normal / (jnp.linalg.norm(edge_normal, axis=-1, keepdims=True) + EPSILON)
    
    # Apply only to boundary edges
    edge_normal = jnp.where(boundary_mask[:, None], edge_normal, 0)
    
    # Accumulate at vertices
    vertex_force = jnp.zeros_like(state.vertex_pos)
    vertex_force = vertex_force.at[edges[0]].add(edge_normal)
    vertex_force = vertex_force.at[edges[1]].add(edge_normal)
    
    # Normalize
    force_norm = jnp.linalg.norm(vertex_force, axis=-1, keepdims=True)
    vertex_force = vertex_force / jnp.maximum(force_norm, EPSILON)
    
    return vertex_force

@jax.jit
def update_positions(state, params, dt=0.1):
    """Update vertex positions based on forces"""
    spring_force = calculate_spring_force(state, params)
    repulsion_mag, repulsion_force = calculate_repulsion_force(state, params)
    bulge_force = calculate_bulge_force(state, params)
    planar_force = calculate_planar_force(state, params)
    
    total_force = (spring_force + 
                   params.repulsion_strength * repulsion_force + 
                   params.bulge_strength * bulge_force +
                   params.planar_strength * planar_force)
    
    # Only update active vertices
    active_mask = state.vertex_idx != -1
    new_positions = state.vertex_pos + dt * total_force * active_mask[:, None]
    
    # Update suitability
    min_repulsion = jnp.min(jnp.where(repulsion_mag > 0, repulsion_mag, jnp.inf))
    min_repulsion = jnp.where(jnp.isfinite(min_repulsion), min_repulsion, 1.0)
    repulsion_norm = repulsion_mag / min_repulsion
    new_suitability = 1 / (1 + repulsion_norm ** 2) * active_mask.astype(jnp.float32)
    
    return state._replace(vertex_pos=new_positions, vertex_suitability=new_suitability), repulsion_mag

def make_first_triangle(state, width, height):
    """Initialize the first triangle"""
    # Vertices
    vertex_positions = jnp.array([
        [width/2, height/2],
        [width/2 + 40, height/2],
        [width/2 + 20, height/2 + 34.64]  # 40 * sqrt(3)/2
    ])
    
    vertex_idx = state.vertex_idx.at[:3].set(jnp.arange(3))
    vertex_pos = state.vertex_pos.at[:3].set(vertex_positions)
    
    # Half edges: inner triangle (0,1,2) and outer boundary (3,4,5)
    half_edge_idx = state.half_edge_idx.at[:6].set(jnp.arange(6))
    
    # Set twins
    half_edge_twin = state.half_edge_twin.at[0].set(3).at[3].set(0)
    half_edge_twin = half_edge_twin.at[1].set(5).at[5].set(1)
    half_edge_twin = half_edge_twin.at[2].set(4).at[4].set(2)
    
    # Set destinations
    half_edge_dest = state.half_edge_dest.at[0].set(1).at[1].set(2).at[2].set(0)
    half_edge_dest = half_edge_dest.at[3].set(0).at[4].set(2).at[5].set(1)
    
    # Set next/prev
    half_edge_next = state.half_edge_next.at[0].set(1).at[1].set(2).at[2].set(0)
    half_edge_next = half_edge_next.at[3].set(4).at[4].set(5).at[5].set(3)
    
    half_edge_prev = state.half_edge_prev.at[0].set(2).at[1].set(0).at[2].set(1)
    half_edge_prev = half_edge_prev.at[3].set(5).at[4].set(3).at[5].set(4)
    
    # Set face (0 for inner triangle, -1 for boundary)
    half_edge_face = state.half_edge_face.at[0].set(0).at[1].set(0).at[2].set(0)
    
    # Set vertex half edges
    vertex_half_edge = state.vertex_half_edge.at[0].set(0).at[1].set(1).at[2].set(2)
    
    # Set face
    face_idx = state.face_idx.at[0].set(0)
    face_half_edge = state.face_half_edge.at[0].set(0)
    
    return state._replace(
        vertex_idx=vertex_idx,
        vertex_pos=vertex_pos,
        vertex_half_edge=vertex_half_edge,
        half_edge_idx=half_edge_idx,
        half_edge_twin=half_edge_twin,
        half_edge_dest=half_edge_dest,
        half_edge_face=half_edge_face,
        half_edge_next=half_edge_next,
        half_edge_prev=half_edge_prev,
        face_idx=face_idx,
        face_half_edge=face_half_edge
    )

def get_vertex_half_edge(state, vertex_a, vertex_b):
    """Get half edge from vertex_a to vertex_b"""
    current_he = state.vertex_half_edge[vertex_a]
    # TODO replace with jax lax while loop
    half_edge_ab = jax.lax.while_loop(
        lambda he: jnp.logical_and(state.half_edge_dest[he] != vertex_b, he != state.vertex_half_edge[vertex_a]),
        lambda he: state.half_edge_next[state.half_edge_twin[he]],
        current_he
    )
    return half_edge_ab

def add_external_triangle(state, vertex_a, vertex_b):
    """Add an external triangle (non-JIT for simplicity)"""
    # Find half edge from a to b
    half_edge_ab = get_vertex_half_edge(state, vertex_a, vertex_b)

    # Get next available indices
    new_face_idx = jnp.max(state.face_idx) + 1
    new_vertex_idx = jnp.max(state.vertex_idx) + 1
    new_half_edge_idx = jnp.max(state.half_edge_idx) + 1
    
    return jax.lax.cond(
        jnp.logical_or(
            half_edge_ab == -1, 
            jnp.logical_or(
                state.half_edge_face[half_edge_ab] != -1, 
                jnp.logical_or(
                    new_face_idx >= MAX_VERTICES, 
                    jnp.logical_or(new_vertex_idx >= MAX_VERTICES, new_half_edge_idx + 4 >= MAX_VERTICES)
                )
            )
        ),
        lambda: state,
        lambda: _add_external_triangle(state, vertex_a, vertex_b, half_edge_ab, new_face_idx, new_vertex_idx, new_half_edge_idx),
    )

def _add_external_triangle(state, vertex_a, vertex_b, half_edge_ab, new_face_idx, new_vertex_idx, new_half_edge_idx):

    # Create new vertex at midpoint
    vertex_c = new_vertex_idx
    new_pos = (state.vertex_pos[vertex_a] + state.vertex_pos[vertex_b]) / 2
    
    # New half edges
    half_edge_bc = new_half_edge_idx
    half_edge_ca = new_half_edge_idx + 1
    half_edge_cb = new_half_edge_idx + 2
    half_edge_ac = new_half_edge_idx + 3
    
    # Get existing connectivity
    half_edge_b_next = state.half_edge_next[half_edge_ab]
    half_edge_a_prev = state.half_edge_prev[half_edge_ab]
    
    # Update arrays
    updates = {}
    
    # Add new vertex
    updates['vertex_idx'] = state.vertex_idx.at[vertex_c].set(vertex_c)
    updates['vertex_pos'] = state.vertex_pos.at[vertex_c].set(new_pos)
    updates['vertex_half_edge'] = state.vertex_half_edge.at[vertex_c].set(half_edge_ca)
    
    # Add new face
    updates['face_idx'] = state.face_idx.at[new_face_idx].set(new_face_idx)
    updates['face_half_edge'] = state.face_half_edge.at[new_face_idx].set(half_edge_ab)
    
    # Update existing half edge ab
    updates['half_edge_face'] = state.half_edge_face.at[half_edge_ab].set(new_face_idx)
    updates['half_edge_next'] = state.half_edge_next.at[half_edge_ab].set(half_edge_bc)
    updates['half_edge_prev'] = state.half_edge_prev.at[half_edge_ab].set(half_edge_ca)
    
    # Add new half edges
    updates['half_edge_idx'] = state.half_edge_idx.at[half_edge_bc].set(half_edge_bc)
    updates['half_edge_idx'] = updates['half_edge_idx'].at[half_edge_ca].set(half_edge_ca)
    updates['half_edge_idx'] = updates['half_edge_idx'].at[half_edge_cb].set(half_edge_cb)
    updates['half_edge_idx'] = updates['half_edge_idx'].at[half_edge_ac].set(half_edge_ac)
    
    # Set properties for new half edges
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_bc].set(new_face_idx)
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_ca].set(new_face_idx)
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_cb].set(-1)
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_ac].set(-1)
    
    updates['half_edge_dest'] = state.half_edge_dest.at[half_edge_bc].set(vertex_c)
    updates['half_edge_dest'] = updates['half_edge_dest'].at[half_edge_ca].set(vertex_a)
    updates['half_edge_dest'] = updates['half_edge_dest'].at[half_edge_cb].set(vertex_b)
    updates['half_edge_dest'] = updates['half_edge_dest'].at[half_edge_ac].set(vertex_c)
    
    updates['half_edge_twin'] = state.half_edge_twin.at[half_edge_bc].set(half_edge_cb)
    updates['half_edge_twin'] = updates['half_edge_twin'].at[half_edge_cb].set(half_edge_bc)
    updates['half_edge_twin'] = updates['half_edge_twin'].at[half_edge_ca].set(half_edge_ac)
    updates['half_edge_twin'] = updates['half_edge_twin'].at[half_edge_ac].set(half_edge_ca)
    
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_bc].set(half_edge_ca)
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_ca].set(half_edge_ab)
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_cb].set(half_edge_b_next)
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_ac].set(half_edge_cb)
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_a_prev].set(half_edge_ac)
    
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_bc].set(half_edge_ab)
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_ca].set(half_edge_bc)
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_cb].set(half_edge_ac)
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_ac].set(half_edge_a_prev)
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_b_next].set(half_edge_cb)
    
    # Create updated state
    return state._replace(**updates)

def add_internal_edge_triangle(state, vertex_a, vertex_b):
    # assert that there is a half edge from a to b and its twin is on the boundary
    half_edge_ab = get_vertex_half_edge(state, vertex_a, vertex_b)

    half_edge_ba = state.half_edge_twin[half_edge_ab]
    # assert state.half_edge_face[half_edge_ab] != -1, "Half edge ab is not internal"
    # assert state.half_edge_face[half_edge_ba] == -1, "Half edge ba is not on boundary"

    half_edge_bc = state.half_edge_next[half_edge_ab]
    half_edge_ca = state.half_edge_next[half_edge_bc]

    vertex_c = state.half_edge_dest[half_edge_bc]
    face_abc = state.half_edge_face[half_edge_ab]

    vertex_d = jnp.max(state.vertex_idx) + 1
    face_dbc = jnp.max(state.face_idx) + 1
    half_edge_ad = half_edge_ab
    half_edge_db = jnp.max(state.half_edge_idx) + 1
    half_edge_cd = half_edge_db + 1
    half_edge_dc = half_edge_cd + 1
    half_edge_bd = half_edge_ba
    half_edge_da = half_edge_dc + 1
    face_adc = face_abc

    face_idx = state.face_idx.at[face_dbc].set(face_dbc)
    face_half_edge = state.face_half_edge.at[face_dbc].set(half_edge_bc)
    face_half_edge = face_half_edge.at[face_adc].set(half_edge_ca)

    new_vertex_pos = jnp.mean(jnp.stack([state.vertex_pos[vertex_a], state.vertex_pos[vertex_b]], axis=0), axis=0)

    vertex_idx = state.vertex_idx.at[vertex_d].set(vertex_d)
    vertex_pos = state.vertex_pos.at[vertex_d].set(new_vertex_pos)
    vertex_half_edge = state.vertex_half_edge.at[vertex_d].set(half_edge_db)

    half_edge_face = state.half_edge_face.at[half_edge_ad].set(face_adc)
    half_edge_dest = state.half_edge_dest.at[half_edge_ad].set(vertex_d)
    half_edge_twin = state.half_edge_twin.at[half_edge_ad].set(half_edge_da)
    half_edge_next = state.half_edge_next.at[half_edge_ad].set(half_edge_dc)
    half_edge_prev = state.half_edge_prev.at[half_edge_ad].set(half_edge_ca)

    half_edge_idx = state.half_edge_idx.at[half_edge_db].set(half_edge_db)
    half_edge_face = half_edge_face.at[half_edge_db].set(face_dbc)
    half_edge_dest = half_edge_dest.at[half_edge_db].set(vertex_b)
    half_edge_twin = half_edge_twin.at[half_edge_db].set(half_edge_bd)
    half_edge_next = half_edge_next.at[half_edge_db].set(half_edge_bc)
    half_edge_prev = half_edge_prev.at[half_edge_db].set(half_edge_cd)

    half_edge_ba_next = state.half_edge_next[half_edge_ba]
    half_edge_ba_prev = state.half_edge_prev[half_edge_ba]
    half_edge_prev = half_edge_prev.at[half_edge_ba_next].set(half_edge_da)
    half_edge_next = half_edge_next.at[half_edge_ba_prev].set(half_edge_bd)

    half_edge_dest = half_edge_dest.at[half_edge_bd].set(vertex_d)
    half_edge_twin = half_edge_twin.at[half_edge_bd].set(half_edge_db)
    half_edge_next = half_edge_next.at[half_edge_bd].set(half_edge_da)

    half_edge_idx = half_edge_idx.at[half_edge_da].set(half_edge_da)
    half_edge_face = half_edge_face.at[half_edge_da].set(-1)
    half_edge_dest = half_edge_dest.at[half_edge_da].set(vertex_a)
    half_edge_twin = half_edge_twin.at[half_edge_da].set(half_edge_ad)
    half_edge_next = half_edge_next.at[half_edge_da].set(half_edge_ba_next)
    half_edge_prev = half_edge_prev.at[half_edge_da].set(half_edge_bd)

    half_edge_idx = half_edge_idx.at[half_edge_dc].set(half_edge_dc)
    half_edge_face = half_edge_face.at[half_edge_dc].set(face_adc)
    half_edge_dest = half_edge_dest.at[half_edge_dc].set(vertex_c)
    half_edge_twin = half_edge_twin.at[half_edge_dc].set(half_edge_cd)
    half_edge_next = half_edge_next.at[half_edge_dc].set(half_edge_ca)
    half_edge_prev = half_edge_prev.at[half_edge_dc].set(half_edge_ad)

    half_edge_idx = half_edge_idx.at[half_edge_cd].set(half_edge_cd)
    half_edge_face = half_edge_face.at[half_edge_cd].set(face_dbc)
    half_edge_dest = half_edge_dest.at[half_edge_cd].set(vertex_d)
    half_edge_twin = half_edge_twin.at[half_edge_cd].set(half_edge_dc)
    half_edge_next = half_edge_next.at[half_edge_cd].set(half_edge_db)
    half_edge_prev = half_edge_prev.at[half_edge_cd].set(half_edge_bc)

    half_edge_face = half_edge_face.at[half_edge_ca].set(face_adc)
    half_edge_next = half_edge_next.at[half_edge_ca].set(half_edge_ad)
    half_edge_prev = half_edge_prev.at[half_edge_ca].set(half_edge_dc)

    half_edge_face = half_edge_face.at[half_edge_bc].set(face_dbc)
    half_edge_next = half_edge_next.at[half_edge_bc].set(half_edge_cd)
    half_edge_prev = half_edge_prev.at[half_edge_bc].set(half_edge_db)

    state = state._replace(
        face_idx=face_idx,
        face_half_edge=face_half_edge,
        vertex_idx=vertex_idx,
        vertex_pos=vertex_pos,
        vertex_half_edge=vertex_half_edge,
        half_edge_idx=half_edge_idx,
        half_edge_face=half_edge_face,
        half_edge_dest=half_edge_dest,
        half_edge_twin=half_edge_twin,
        half_edge_next=half_edge_next,
        half_edge_prev=half_edge_prev
    )

    return state

def add_internal_triangles(state, vertex_a, vertex_b):
    # find half edge from a to b
    half_edge_ab = get_vertex_half_edge(state, vertex_a, vertex_b)
    half_edge_ba = state.half_edge_twin[half_edge_ab]

    return jax.lax.cond(
        jnp.logical_or(state.half_edge_face[half_edge_ab] == -1, state.half_edge_face[half_edge_ba] == -1),
        lambda: state,
        lambda: _add_internal_triangles(state, vertex_a, vertex_b, half_edge_ab)
    )

def _add_internal_triangles(state, vertex_a, vertex_b, half_edge_ab):

    half_edge_ba = state.half_edge_twin[half_edge_ab]
    vertex_c = state.half_edge_dest[state.half_edge_next[half_edge_ab]]
    vertex_d = state.half_edge_dest[state.half_edge_next[half_edge_ba]]
    # assert vertex_c != vertex_d, "Cannot add internal triangles when both faces are the same."

    face_abc = state.half_edge_face[half_edge_ab]
    face_bad = state.half_edge_face[half_edge_ba]
    face_aec = face_abc
    face_bed = face_bad

    half_edge_ae = half_edge_ab
    half_edge_be = half_edge_ba

    face_ebc = jnp.max(state.face_idx) + 1
    face_ead = face_ebc + 1

    vertex_e = jnp.max(state.vertex_idx) + 1
    vertex_idx = state.vertex_idx.at[vertex_e].set(vertex_e)
    vertex_pos = state.vertex_pos.at[vertex_e].set(jnp.mean(jnp.stack([state.vertex_pos[vertex_a], state.vertex_pos[vertex_b]], axis=0), axis=0))

    # create new face
    face_idx = state.face_idx.at[face_ebc].set(face_ebc)
    face_idx = face_idx.at[face_ead].set(face_ead)

    # edit face aec
    half_edge_ec = jnp.max(state.half_edge_idx) + 1
    half_edge_ea = half_edge_ec + 1
    half_edge_ce = half_edge_ea + 1
    half_edge_bc = state.half_edge_next[half_edge_ab]
    half_edge_ca = state.half_edge_next[half_edge_bc]
    half_edge_ad = state.half_edge_next[half_edge_ba]
    half_edge_db = state.half_edge_next[half_edge_ad]

    half_edge_dest = state.half_edge_dest.at[half_edge_ae].set(vertex_e)
    half_edge_next = state.half_edge_next.at[half_edge_ae].set(half_edge_ec)
    half_edge_twin = state.half_edge_twin.at[half_edge_ae].set(half_edge_ea)

    half_edge_idx = state.half_edge_idx.at[half_edge_ec].set(half_edge_ec)
    half_edge_twin = half_edge_twin.at[half_edge_ec].set(half_edge_ce)
    half_edge_face = state.half_edge_face.at[half_edge_ec].set(face_aec)
    half_edge_dest = half_edge_dest.at[half_edge_ec].set(vertex_c)
    half_edge_next = half_edge_next.at[half_edge_ec].set(half_edge_ca)
    half_edge_prev = state.half_edge_prev.at[half_edge_ec].set(half_edge_ae)

    half_edge_prev = half_edge_prev.at[half_edge_ca].set(half_edge_ec)

    # edit face ebc
    half_edge_eb = half_edge_ce + 1
    vertex_half_edge = state.vertex_half_edge.at[vertex_e].set(half_edge_eb)

    half_edge_idx = half_edge_idx.at[half_edge_eb].set(half_edge_eb)
    half_edge_twin = half_edge_twin.at[half_edge_eb].set(half_edge_be)
    half_edge_face = half_edge_face.at[half_edge_eb].set(face_ebc)
    half_edge_dest = half_edge_dest.at[half_edge_eb].set(vertex_b)
    half_edge_next = half_edge_next.at[half_edge_eb].set(half_edge_bc)
    half_edge_prev = half_edge_prev.at[half_edge_eb].set(half_edge_ce)

    half_edge_face = half_edge_face.at[half_edge_bc].set(face_ebc)
    half_edge_next = half_edge_next.at[half_edge_bc].set(half_edge_ce)
    half_edge_prev = half_edge_prev.at[half_edge_bc].set(half_edge_eb)

    half_edge_idx = half_edge_idx.at[half_edge_ce].set(half_edge_ce)
    half_edge_twin = half_edge_twin.at[half_edge_ce].set(half_edge_ec)
    half_edge_face = half_edge_face.at[half_edge_ce].set(face_ebc)
    half_edge_dest = half_edge_dest.at[half_edge_ce].set(vertex_e)
    half_edge_next = half_edge_next.at[half_edge_ce].set(half_edge_eb)
    half_edge_prev = half_edge_prev.at[half_edge_ce].set(half_edge_bc)

    # edit face ead
    half_edge_de = half_edge_eb + 1
    half_edge_ed = half_edge_de + 1

    half_edge_idx = half_edge_idx.at[half_edge_ea].set(half_edge_ea)
    half_edge_twin = half_edge_twin.at[half_edge_ea].set(half_edge_ae)
    half_edge_face = half_edge_face.at[half_edge_ea].set(face_ead)
    half_edge_dest = half_edge_dest.at[half_edge_ea].set(vertex_a)
    half_edge_next = half_edge_next.at[half_edge_ea].set(half_edge_ad)
    half_edge_prev = half_edge_prev.at[half_edge_ea].set(half_edge_de)

    half_edge_face = half_edge_face.at[half_edge_ad].set(face_ead)
    half_edge_next = half_edge_next.at[half_edge_ad].set(half_edge_de)
    half_edge_prev = half_edge_prev.at[half_edge_ad].set(half_edge_ea)

    half_edge_idx = half_edge_idx.at[half_edge_de].set(half_edge_de)
    half_edge_twin = half_edge_twin.at[half_edge_de].set(half_edge_ed)
    half_edge_face = half_edge_face.at[half_edge_de].set(face_ead)
    half_edge_dest = half_edge_dest.at[half_edge_de].set(vertex_e)
    half_edge_next = half_edge_next.at[half_edge_de].set(half_edge_ea)
    half_edge_prev = half_edge_prev.at[half_edge_de].set(half_edge_ad)

    # edit existing face bed
    half_edge_twin = half_edge_twin.at[half_edge_be].set(half_edge_eb)
    half_edge_face = half_edge_face.at[half_edge_be].set(face_bed)
    half_edge_dest = half_edge_dest.at[half_edge_be].set(vertex_e)
    half_edge_next = half_edge_next.at[half_edge_be].set(half_edge_ed)
    half_edge_prev = half_edge_prev.at[half_edge_be].set(half_edge_db)

    half_edge_idx = half_edge_idx.at[half_edge_ed].set(half_edge_ed)
    half_edge_twin = half_edge_twin.at[half_edge_ed].set(half_edge_de)
    half_edge_face = half_edge_face.at[half_edge_ed].set(face_bed)
    half_edge_dest = half_edge_dest.at[half_edge_ed].set(vertex_d)
    half_edge_next = half_edge_next.at[half_edge_ed].set(half_edge_db)
    half_edge_prev = half_edge_prev.at[half_edge_ed].set(half_edge_be)

    half_edge_next = half_edge_next.at[half_edge_db].set(half_edge_be)
    half_edge_prev = half_edge_prev.at[half_edge_db].set(half_edge_ed)

    face_half_edge = state.face_half_edge.at[face_ebc].set(half_edge_eb)
    face_half_edge = face_half_edge.at[face_ead].set(half_edge_ea)

    state = state._replace(
        vertex_idx=vertex_idx,
        vertex_pos=vertex_pos,
        vertex_half_edge=vertex_half_edge,
        face_idx=face_idx,
        face_half_edge=face_half_edge,
        half_edge_idx=half_edge_idx,
        half_edge_face=half_edge_face,
        half_edge_dest=half_edge_dest,
        half_edge_twin=half_edge_twin,
        half_edge_next=half_edge_next,
        half_edge_prev=half_edge_prev
    )

    return state

@jax.jit
def get_edge_count(state):
    """Count edges for each vertex using JAX operations"""
    complete = jnp.zeros((state.vertex_idx.shape[0],), dtype=bool)
    edge_count = jnp.zeros((state.vertex_idx.shape[0],), dtype=int)
    start_vertex = state.half_edge_dest[state.vertex_half_edge]
    this_half_edge = state.vertex_half_edge

    def cond_func(carry):
        complete, _, _, _ = carry
        return jnp.any(~complete)
    
    def body_func(carry):
        complete, edge_count, this_half_edge, start_vertex = carry
        next_half_edge = state.half_edge_next[state.half_edge_twin[this_half_edge]]
        next_vertex = state.half_edge_dest[next_half_edge]
        complete = complete | (next_vertex == start_vertex)
        this_half_edge = next_half_edge
        edge_count += ~complete
        return (complete, edge_count, this_half_edge, start_vertex)

    (_, edge_count, _, _) = jax.lax.while_loop(
        cond_func,
        body_func,
        (complete, edge_count, this_half_edge, start_vertex)
    )
    return edge_count

def flip_edge(state, half_edge_ab):
    """Flip an edge in the mesh"""
    half_edge_ba = state.half_edge_twin[half_edge_ab]
    
    # Check if edge can be flipped (both faces must exist)
    return jax.lax.cond(
        jnp.logical_or(
            state.half_edge_face[half_edge_ab] == -1, 
            state.half_edge_face[half_edge_ba] == -1
        ),
        lambda: state,
        lambda: _flip_edge(state, half_edge_ab, half_edge_ba)
    )
    
def _flip_edge(state, half_edge_ab, half_edge_ba):
    # Get vertices
    vertex_a = state.half_edge_dest[half_edge_ba]
    vertex_b = state.half_edge_dest[half_edge_ab]
    
    # Get surrounding half edges
    half_edge_bc = state.half_edge_next[half_edge_ab]
    vertex_c = state.half_edge_dest[half_edge_bc]
    half_edge_ca = state.half_edge_next[half_edge_bc]

    half_edge_ad = state.half_edge_next[half_edge_ba]
    vertex_d = state.half_edge_dest[half_edge_ad]
    half_edge_db = state.half_edge_next[half_edge_ad]
    
    # Get faces
    face_abc = state.half_edge_face[half_edge_ab]
    face_bad = state.half_edge_face[half_edge_ba]

    # After flip: faces will be adc and bcd
    face_adc = face_abc
    face_bcd = face_bad
    
    # The flipped edge becomes dc and cd
    half_edge_dc = half_edge_ab
    half_edge_cd = half_edge_ba
    
    # Build updates
    updates = {}
    
    # Update face references
    updates['face_half_edge'] = state.face_half_edge.at[face_adc].set(half_edge_ca)
    updates['face_half_edge'] = updates['face_half_edge'].at[face_bcd].set(half_edge_db)
    
    # Update vertex references (in case they pointed to the flipped edge)
    updates['vertex_half_edge'] = state.vertex_half_edge.at[vertex_a].set(half_edge_ad)
    updates['vertex_half_edge'] = updates['vertex_half_edge'].at[vertex_b].set(half_edge_bc)
    
    # Update the flipped edge
    updates['half_edge_dest'] = state.half_edge_dest.at[half_edge_dc].set(vertex_c)
    updates['half_edge_dest'] = updates['half_edge_dest'].at[half_edge_cd].set(vertex_d)
    
    updates['half_edge_next'] = state.half_edge_next.at[half_edge_dc].set(half_edge_ca)
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_cd].set(half_edge_db)
    
    updates['half_edge_prev'] = state.half_edge_prev.at[half_edge_dc].set(half_edge_ad)
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_cd].set(half_edge_bc)
    
    updates['half_edge_face'] = state.half_edge_face.at[half_edge_dc].set(face_adc)
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_cd].set(face_bcd)
    
    # Update surrounding half edges
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_ca].set(half_edge_ad)
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_ad].set(half_edge_dc)
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_db].set(half_edge_bc)
    updates['half_edge_next'] = updates['half_edge_next'].at[half_edge_bc].set(half_edge_cd)
    
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_ca].set(half_edge_dc)
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_ad].set(half_edge_ca)
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_db].set(half_edge_cd)
    updates['half_edge_prev'] = updates['half_edge_prev'].at[half_edge_bc].set(half_edge_db)
    
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_ca].set(face_adc)
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_ad].set(face_adc)
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_db].set(face_bcd)
    updates['half_edge_face'] = updates['half_edge_face'].at[half_edge_bc].set(face_bcd)
    
    return state._replace(**updates)

@jax.jit
def refine_mesh(state):
    """Refine mesh by flipping edges to optimize vertex valences"""
    # Find internal edges (both faces exist)
    edges = jnp.stack([state.half_edge_dest, state.half_edge_dest[state.half_edge_twin]], axis=-1)
    
    # get two other vertices
    vertex_c = state.half_edge_dest[state.half_edge_next]
    vertex_d = state.half_edge_dest[state.half_edge_next[state.half_edge_twin]]

    quad_half_edges = state.half_edge_idx
    edges_data = jnp.stack([
        edges[:, 0], 
        edges[:, 1], 
        vertex_c, 
        vertex_d, 
        quad_half_edges, 
        state.half_edge_face, 
        state.half_edge_face[state.half_edge_twin],
    ], axis=-1)

    # Process each internal edge
    state, _ = jax.lax.scan(check_flip_edge, state, edges_data)
    return state

def check_flip_edge(state, edge_data):

    edge = edge_data[:2]
    vertex_c, vertex_d, half_edge_idx, half_edge_face, half_edge_twin_face = edge_data[2:]

    return jax.lax.cond(
        (edge[0] < edge[1]) & (half_edge_face != -1) & (half_edge_twin_face != -1) & (half_edge_idx != -1),
        lambda: _check_flip_edge(state, edge, vertex_c, vertex_d, half_edge_idx),
        lambda: state
    ), None # No output needed for scan

def _check_flip_edge(state, edge, vertex_c, vertex_d, half_edge_idx):

    edge_count = get_edge_count(state)

    vertex_quad = jnp.array([edge[0], edge[1], vertex_c, vertex_d])
    vertex_quad_edge_counts = edge_count[vertex_quad]
    
    no_flip_valence_weights = jnp.full((4,), 6)
    flip_valence_weights = jnp.array([7, 7, 5, 5])

    # Get current valences
    no_flip_valence = jnp.sum(jnp.square(vertex_quad_edge_counts - no_flip_valence_weights))
    flip_valence = jnp.sum(jnp.square(vertex_quad_edge_counts - flip_valence_weights))
    
    
    state = jax.lax.cond(
        flip_valence < no_flip_valence,
        lambda: flip_edge(state, half_edge_idx),
        lambda: state
    )

    return state

@jax.jit
def maybe_generate_new_triangles(state, params, key, repulsion_mag):
    """Randomly generate new triangles"""
    key, subkey = jax.random.split(key)
    
    # Only generate with 10% probability
    return jax.lax.cond(
        jax.random.uniform(subkey) < 0.1,
        lambda: generate_new_triangles(state, params, key),
        lambda: (state, key),
    )
    

def generate_new_triangles(state, params, key):
    # Find active vertices

    key, subkey = jax.random.split(key)
    # Choose a random vertex weighted by suitability
    suitabilities = state.vertex_suitability
    probs = suitabilities / (jnp.sum(suitabilities) + EPSILON)
    probs *= (state.vertex_idx != -1)
    chosen_vertex = jax.random.choice(subkey, state.vertex_idx, p=probs)
    
    # Get its half edge and destination
    chosen_half_edge = state.vertex_half_edge[chosen_vertex]
    dest_vertex = state.half_edge_dest[chosen_half_edge]

    # Check if on boundary
    chosen_on_boundary = state.half_edge_face[chosen_half_edge] == -1
    twin_on_boundary = state.half_edge_face[state.half_edge_twin[chosen_half_edge]] == -1
    
    key, subkey = jax.random.split(key)

    state = jax.lax.cond(
        jnp.logical_and(chosen_on_boundary, twin_on_boundary),
        lambda: add_boundary_triangle(state, chosen_vertex, dest_vertex, chosen_on_boundary, twin_on_boundary, subkey),
        lambda: add_internal_triangles(state, chosen_vertex, dest_vertex),
    )

    return state, key

def add_boundary_triangle(state, chosen_vertex, dest_vertex, chosen_on_boundary, twin_on_boundary, subkey):
    # Add triangle based on boundary status
    state = jax.lax.cond(
        jax.random.uniform(subkey) < 0.5,
        lambda: jax.lax.cond(
            chosen_on_boundary,
            lambda: add_external_triangle(state, chosen_vertex, dest_vertex),
            lambda: add_external_triangle(state, dest_vertex, chosen_vertex)
        ),
        lambda: jax.lax.cond(
            twin_on_boundary,
            lambda: add_internal_edge_triangle(state, chosen_vertex, dest_vertex),
            lambda: add_internal_edge_triangle(state, dest_vertex, chosen_vertex)
        )
    )

    return state

def draw_pygame(state, screen):
    """Draw the mesh using pygame"""
    screen.fill((0, 0, 0))
    
    # Convert JAX arrays to numpy for drawing
    vertex_pos = np.array(state.vertex_pos)
    vertex_idx = np.array(state.vertex_idx)
    half_edge_dest = np.array(state.half_edge_dest)
    half_edge_prev = np.array(state.half_edge_prev)
    half_edge_idx = np.array(state.half_edge_idx)
    vertex_suitability = np.array(state.vertex_suitability)
    
    # Draw edges
    active_edges = half_edge_idx != -1
    edges = np.stack([half_edge_dest, half_edge_dest[half_edge_prev]])
    
    for i in np.where(active_edges)[0]:
        if edges[0, i] != -1 and edges[1, i] != -1:
            start = vertex_pos[edges[0, i]]
            end = vertex_pos[edges[1, i]]
            if start[0] >= 0 and end[0] >= 0:  # Valid positions
                pygame.draw.line(screen, (50, 50, 50), start.astype(int), end.astype(int), 2)
    
    # Draw vertices
    active_vertices = vertex_idx != -1
    for i in np.where(active_vertices)[0]:
        pos = vertex_pos[i]
        if pos[0] >= 0:  # Valid position
            suitability = vertex_suitability[i]
            color = int(255 * suitability)
            pygame.draw.circle(screen, (color, color, color), pos.astype(int), 5)
    
    pygame.display.flip()

def main():
    width, height = 1400, 1400
    
    # Initialize JAX
    key = jax.random.PRNGKey(42)
    
    # Create initial state and parameters
    state = create_initial_state(num_dims=2)
    params = MeshParams(
        spring_len=40.0,
        elastic_constant=0.1,
        repulsion_distance=200.0,
        repulsion_strength=2.0,
        bulge_strength=10.0,
        planar_strength=0.1
    )
    
    # Initialize first triangle
    state = make_first_triangle(state, width, height)
    
    # Add one external triangle
    state = add_external_triangle(state, 0, 2)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    
    running = True
    frame_count = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update physics
        state, repulsion_mag = update_positions(state, params)
        
        # Generate new triangles
        state, key = maybe_generate_new_triangles(state, params, key, repulsion_mag)
        
        # Refine mesh every 10 frames
        if frame_count % 10 == 0:
            state = refine_mesh(state)
        
        # Draw
        draw_pygame(state, screen)
        clock.tick(60)
        frame_count += 1
    
    pygame.quit()

if __name__ == '__main__':
    main()