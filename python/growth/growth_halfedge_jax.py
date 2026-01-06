import collections
import functools

import glm
import jax
import jax.numpy as jnp
import numpy as np
import pygame
import torch

import nvdiffrast.torch as dr


MAX_VERTICES = 1000
MAX_HALF_EDGES = MAX_VERTICES * 6  # Triangular mesh needs ~6 half-edges per vertex
MAX_FACES = MAX_VERTICES * 2  # Triangular mesh needs ~2 faces per vertex
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
    'vertex_state'
])

MeshParams = collections.namedtuple('MeshParams', [
    'spring_len',
    'elastic_constant',
    'repulsion_distance',
    'repulsion_strength',
    'bulge_strength',
    'planar_strength',
    'num_dims',
    'vertex_state_mlp_params',
    'num_colour_channels'
])

@jax.jit
def get_face_indices(state: MeshState, all_half_edges_with_faces):
    # Build triangles for all half-edges with faces
    triangle_half_edges = jnp.stack([
        all_half_edges_with_faces,
        state.half_edge_next[all_half_edges_with_faces],
        state.half_edge_next[state.half_edge_next[all_half_edges_with_faces]],
    ], -1)
    triangle_vertices = state.half_edge_dest[triangle_half_edges]

    # Front face indices (original order)
    indices_front = triangle_vertices.reshape(-1)

    # Back face indices (swap v1 and v2 to reverse winding)
    triangle_vertices_back = triangle_vertices[:, [0, 2, 1]]
    indices_back = triangle_vertices_back.reshape(-1)

    # Flatten to get index array
    indices = jnp.concatenate([indices_front, indices_back])
    return indices

def create_initial_state(subkey, num_dims=2, state_dims=4):
    """Create initial mesh state"""

    return MeshState(
        face_idx=jnp.full((MAX_FACES,), -1, dtype=jnp.int32),
        face_half_edge=jnp.full((MAX_FACES,), -1, dtype=jnp.int32),
        vertex_idx=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        vertex_half_edge=jnp.full((MAX_VERTICES,), -1, dtype=jnp.int32),
        vertex_pos=jnp.full((MAX_VERTICES, num_dims), -1, dtype=jnp.float32),
        half_edge_idx=jnp.full((MAX_HALF_EDGES,), -1, dtype=jnp.int32),
        half_edge_twin=jnp.full((MAX_HALF_EDGES,), -1, dtype=jnp.int32),
        half_edge_dest=jnp.full((MAX_HALF_EDGES,), -1, dtype=jnp.int32),
        half_edge_face=jnp.full((MAX_HALF_EDGES,), -1, dtype=jnp.int32),
        half_edge_next=jnp.full((MAX_HALF_EDGES,), -1, dtype=jnp.int32),
        half_edge_prev=jnp.full((MAX_HALF_EDGES,), -1, dtype=jnp.int32),
        vertex_state=jax.random.normal(subkey, (MAX_VERTICES, state_dims), dtype=jnp.float32)
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
def calculate_repulsion_force(state: MeshState, params: MeshParams):
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

    # TODO maybe no planar force for boundary vertices?
    
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
def calculate_bulge_force(state: MeshState, params: MeshParams):
    """Calculate bulge forces for boundary edges"""
    # Find boundary half edges
    boundary_mask = (state.half_edge_face == -1) & (state.half_edge_idx != -1)
    
    # Get edge vectors
    edges = jnp.stack([
        state.half_edge_dest[state.half_edge_twin],
        state.half_edge_dest,
    ])
    edge_pos = state.vertex_pos[edges]
    edge_vector = edge_pos[1] - edge_pos[0]

    next_edge = jnp.stack([
        state.half_edge_dest[state.half_edge_twin],
        state.half_edge_dest[state.half_edge_next[state.half_edge_twin]],

    ])
    next_edge_pos = state.vertex_pos[next_edge]
    next_edge_vector = next_edge_pos[1] - next_edge_pos[0]

    # calculate normals
    surface_normal = jnp.cross(edge_vector, next_edge_vector)
    edge_normal = jnp.cross(edge_vector, surface_normal)
    
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
    # min_repulsion = jnp.min(jnp.where(repulsion_mag > 0, repulsion_mag, jnp.inf))
    # min_repulsion = jnp.where(jnp.isfinite(min_repulsion), min_repulsion, 1.0)
    # repulsion_norm = repulsion_mag / min_repulsion
    # new_suitability = 1 / (1 + repulsion_norm ** 2) * active_mask.astype(jnp.float32)
    
    return state._replace(vertex_pos=new_positions), repulsion_mag

def make_first_triangle(state, width, height, params):
    """Initialize the first triangle"""
    # Vertices
    vertex_positions = jnp.array([
        [width/2, height/2, 0],
        [width/2 + params.spring_len, height/2, 0],
        [width/2 + params.spring_len/2, height/2 + params.spring_len * jnp.sqrt(3)/2, 0]
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

def make_circle(state, width, height, params, n_rings=3, segments_inner=6):
    """Initialize a circular disc mesh with multiple concentric rings

    Args:
        state: Initial mesh state
        width, height: Screen dimensions
        params: Mesh parameters
        n_rings: Number of concentric rings (default 3, giving radius ~3*spring_len)
        segments_inner: Number of segments in innermost ring (outer rings have proportionally more)
    """
    center_x, center_y = width/2, height/2
    spring_len = params.spring_len

    # Start building the mesh with center vertex
    vertex_positions_list = [jnp.array([center_x, center_y, 0])]

    # Create vertices for each ring
    ring_vertex_counts = []
    for ring_idx in range(n_rings):
        radius = spring_len * (ring_idx + 1)
        # Each ring has more segments proportional to its radius to maintain edge length
        n_segments = segments_inner * (ring_idx + 1)
        ring_vertex_counts.append(n_segments)

        angles = jnp.linspace(0, 2 * jnp.pi, n_segments, endpoint=False)
        x = center_x + radius * jnp.cos(angles)
        y = center_y + radius * jnp.sin(angles)
        z = jnp.zeros(n_segments)
        ring_positions = jnp.stack([x, y, z], axis=1)
        vertex_positions_list.append(ring_positions)

    # Concatenate all vertices
    vertex_positions = jnp.concatenate([jnp.expand_dims(vertex_positions_list[0], 0)] +
                                       [vertex_positions_list[i+1] for i in range(n_rings)])
    n_vertices = len(vertex_positions)

    # Initialize mesh arrays
    vertex_idx = state.vertex_idx.at[:n_vertices].set(jnp.arange(n_vertices))
    vertex_pos = state.vertex_pos.at[:n_vertices].set(vertex_positions)

    # Build connectivity using numpy for flexibility (will convert to JAX arrays)
    edges_list = []
    faces_list = []

    # Get vertex index for ring i, segment j
    def get_vertex_idx(ring, segment):
        if ring == -1:
            return 0  # Center vertex
        ring_start = 1 + sum(ring_vertex_counts[:ring])
        return ring_start + (segment % ring_vertex_counts[ring])

    # Connect center to first ring
    n_segments_ring0 = ring_vertex_counts[0]
    for i in range(n_segments_ring0):
        v0 = 0  # center
        v1 = get_vertex_idx(0, i)
        v2 = get_vertex_idx(0, i + 1)
        faces_list.append([v0, v1, v2])

    # Connect rings to each other
    for ring_idx in range(n_rings - 1):
        n_inner = ring_vertex_counts[ring_idx]
        n_outer = ring_vertex_counts[ring_idx + 1]

        # For each inner vertex, connect to outer ring
        inner_idx = 0
        outer_idx = 0

        while inner_idx < n_inner or outer_idx < n_outer:
            v_inner_curr = get_vertex_idx(ring_idx, inner_idx)
            v_inner_next = get_vertex_idx(ring_idx, inner_idx + 1)
            v_outer_curr = get_vertex_idx(ring_idx + 1, outer_idx)
            v_outer_next = get_vertex_idx(ring_idx + 1, outer_idx + 1)

            # Calculate angles to decide which triangle to add
            inner_angle_curr = inner_idx / n_inner
            inner_angle_next = (inner_idx + 1) / n_inner
            outer_angle_curr = outer_idx / n_outer
            outer_angle_next = (outer_idx + 1) / n_outer

            # Choose which diagonal to use
            if outer_angle_next < inner_angle_next:
                # Add triangle: inner_curr, outer_curr, outer_next
                faces_list.append([v_inner_curr, v_outer_curr, v_outer_next])
                outer_idx += 1
            else:
                # Add triangle: inner_curr, outer_curr, inner_next
                faces_list.append([v_inner_curr, v_outer_curr, v_inner_next])
                # If we're not done, also add the other triangle
                if outer_angle_curr < inner_angle_next:
                    faces_list.append([v_inner_next, v_outer_curr, v_outer_next])
                    outer_idx += 1
                inner_idx += 1

    # Now build half-edge structure from faces
    n_faces = len(faces_list)
    edge_dict = {}  # (v1, v2) -> half_edge_idx
    half_edge_list = []

    for face_idx, face in enumerate(faces_list):
        v0, v1, v2 = face
        face_edges = [(v0, v1), (v1, v2), (v2, v0)]
        face_half_edges = []

        for v_from, v_to in face_edges:
            he_idx = len(half_edge_list)
            half_edge_list.append({
                'dest': v_to,
                'face': face_idx,
                'twin': None,
                'next': None,
                'prev': None
            })
            edge_dict[(v_from, v_to)] = he_idx
            face_half_edges.append(he_idx)

        # Set next/prev within face
        for i in range(3):
            half_edge_list[face_half_edges[i]]['next'] = face_half_edges[(i + 1) % 3]
            half_edge_list[face_half_edges[i]]['prev'] = face_half_edges[(i - 1) % 3]

    # Create boundary half-edges and set twins
    boundary_half_edges = []
    for (v_from, v_to), he_idx in edge_dict.items():
        twin_key = (v_to, v_from)
        if twin_key in edge_dict:
            # Internal edge
            half_edge_list[he_idx]['twin'] = edge_dict[twin_key]
        else:
            # Boundary edge - create boundary half-edge
            boundary_he_idx = len(half_edge_list)
            half_edge_list.append({
                'dest': v_from,
                'face': -1,
                'twin': he_idx,
                'next': None,  # Will set later
                'prev': None   # Will set later
            })
            half_edge_list[he_idx]['twin'] = boundary_he_idx
            boundary_half_edges.append((v_to, v_from, boundary_he_idx))

    # Connect boundary half-edges
    boundary_dict = {(v_from, v_to): he_idx for v_from, v_to, he_idx in boundary_half_edges}
    for v_from, v_to, he_idx in boundary_half_edges:
        # Find next boundary half-edge (starts at v_to)
        for (v_next_from, v_next_to), next_he_idx in boundary_dict.items():
            if v_next_from == v_to:
                half_edge_list[he_idx]['next'] = next_he_idx
                half_edge_list[next_he_idx]['prev'] = he_idx
                break

    # Convert to JAX arrays
    n_half_edges = len(half_edge_list)
    half_edge_idx = state.half_edge_idx.at[:n_half_edges].set(jnp.arange(n_half_edges))
    half_edge_dest = state.half_edge_dest
    half_edge_face = state.half_edge_face
    half_edge_twin = state.half_edge_twin
    half_edge_next = state.half_edge_next
    half_edge_prev = state.half_edge_prev

    for he_idx, he_data in enumerate(half_edge_list):
        half_edge_dest = half_edge_dest.at[he_idx].set(he_data['dest'])
        half_edge_face = half_edge_face.at[he_idx].set(he_data['face'])
        half_edge_twin = half_edge_twin.at[he_idx].set(he_data['twin'])
        half_edge_next = half_edge_next.at[he_idx].set(he_data['next'])
        half_edge_prev = half_edge_prev.at[he_idx].set(he_data['prev'])

    # Set vertex half-edges (one outgoing half-edge per vertex)
    vertex_half_edge = state.vertex_half_edge
    for v_idx in range(n_vertices):
        # Find any half-edge with this vertex as source
        for (v_from, v_to), he_idx in edge_dict.items():
            if v_from == v_idx:
                vertex_half_edge = vertex_half_edge.at[v_idx].set(he_idx)
                break

    # Set face data
    face_idx = state.face_idx.at[:n_faces].set(jnp.arange(n_faces))
    face_half_edge_array = state.face_half_edge
    for f_idx, face in enumerate(faces_list):
        v0, v1, v2 = face
        he_idx = edge_dict[(v0, v1)]
        face_half_edge_array = face_half_edge_array.at[f_idx].set(he_idx)

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
        face_half_edge=face_half_edge_array
    )

def make_sphere(state, width, height, params, n_lat=6, n_lon=8, radius_multiplier=2.0):
    """Initialize a spherical mesh using UV sphere approach

    Args:
        state: Initial mesh state
        width, height: Screen dimensions
        params: Mesh parameters
        n_lat: Number of latitude divisions (excluding poles)
        n_lon: Number of longitude divisions
        radius_multiplier: Sphere radius as multiple of spring_len
    """
    center_x, center_y, center_z = width/2, height/2, 0.0
    radius = params.spring_len * radius_multiplier

    # Create vertices
    vertex_positions_list = []

    # Top pole
    vertex_positions_list.append(jnp.array([center_x, center_y, center_z + radius]))

    # Latitude rings (from top to bottom, excluding poles)
    for lat_idx in range(n_lat):
        # Angle from north pole (0 to pi)
        theta = jnp.pi * (lat_idx + 1) / (n_lat + 1)
        z = center_z + radius * jnp.cos(theta)
        ring_radius = radius * jnp.sin(theta)

        # Longitude divisions around the ring
        for lon_idx in range(n_lon):
            phi = 2 * jnp.pi * lon_idx / n_lon
            x = center_x + ring_radius * jnp.cos(phi)
            y = center_y + ring_radius * jnp.sin(phi)
            vertex_positions_list.append(jnp.array([x, y, z]))

    # Bottom pole
    vertex_positions_list.append(jnp.array([center_x, center_y, center_z - radius]))

    # Concatenate all vertices
    vertex_positions = jnp.stack(vertex_positions_list)
    n_vertices = len(vertex_positions)

    # Initialize mesh arrays
    vertex_idx = state.vertex_idx.at[:n_vertices].set(jnp.arange(n_vertices))
    vertex_pos = state.vertex_pos.at[:n_vertices].set(vertex_positions)

    # Helper function to get vertex index
    def get_vertex_idx(lat, lon):
        """Get vertex index for latitude ring and longitude position"""
        if lat == -1:  # North pole
            return 0
        elif lat == n_lat:  # South pole
            return 1 + n_lat * n_lon
        else:  # Regular ring
            return 1 + lat * n_lon + (lon % n_lon)

    # Build faces
    faces_list = []

    # Top cap - triangles connecting north pole to first ring
    for lon_idx in range(n_lon):
        v0 = get_vertex_idx(-1, 0)  # North pole
        v1 = get_vertex_idx(0, lon_idx)
        v2 = get_vertex_idx(0, lon_idx + 1)
        faces_list.append([v0, v1, v2])

    # Middle bands - quads between latitude rings (split into triangles)
    for lat_idx in range(n_lat - 1):
        for lon_idx in range(n_lon):
            # Vertices of the quad
            v_tl = get_vertex_idx(lat_idx, lon_idx)      # top-left
            v_tr = get_vertex_idx(lat_idx, lon_idx + 1)  # top-right
            v_bl = get_vertex_idx(lat_idx + 1, lon_idx)      # bottom-left
            v_br = get_vertex_idx(lat_idx + 1, lon_idx + 1)  # bottom-right

            # Split quad into two triangles
            faces_list.append([v_tl, v_bl, v_tr])
            faces_list.append([v_tr, v_bl, v_br])

    # Bottom cap - triangles connecting south pole to last ring
    for lon_idx in range(n_lon):
        v0 = get_vertex_idx(n_lat, 0)  # South pole
        v1 = get_vertex_idx(n_lat - 1, lon_idx + 1)
        v2 = get_vertex_idx(n_lat - 1, lon_idx)
        faces_list.append([v0, v1, v2])

    # Build half-edge structure from faces
    n_faces = len(faces_list)
    edge_dict = {}  # (v1, v2) -> half_edge_idx
    half_edge_list = []

    for face_idx, face in enumerate(faces_list):
        v0, v1, v2 = face
        face_edges = [(v0, v1), (v1, v2), (v2, v0)]
        face_half_edges = []

        for v_from, v_to in face_edges:
            he_idx = len(half_edge_list)
            half_edge_list.append({
                'dest': v_to,
                'face': face_idx,
                'twin': None,
                'next': None,
                'prev': None
            })
            edge_dict[(v_from, v_to)] = he_idx
            face_half_edges.append(he_idx)

        # Set next/prev within face
        for i in range(3):
            half_edge_list[face_half_edges[i]]['next'] = face_half_edges[(i + 1) % 3]
            half_edge_list[face_half_edges[i]]['prev'] = face_half_edges[(i - 1) % 3]

    # Set twins (all edges should have twins in a closed sphere)
    for (v_from, v_to), he_idx in edge_dict.items():
        twin_key = (v_to, v_from)
        if twin_key in edge_dict:
            half_edge_list[he_idx]['twin'] = edge_dict[twin_key]
        else:
            # This shouldn't happen for a closed sphere, but handle it gracefully
            print(f"Warning: No twin found for edge {v_from} -> {v_to}")
            half_edge_list[he_idx]['twin'] = he_idx  # Self-twin as fallback

    # Convert to JAX arrays
    n_half_edges = len(half_edge_list)
    half_edge_idx = state.half_edge_idx.at[:n_half_edges].set(jnp.arange(n_half_edges))
    half_edge_dest = state.half_edge_dest
    half_edge_face = state.half_edge_face
    half_edge_twin = state.half_edge_twin
    half_edge_next = state.half_edge_next
    half_edge_prev = state.half_edge_prev

    for he_idx, he_data in enumerate(half_edge_list):
        half_edge_dest = half_edge_dest.at[he_idx].set(he_data['dest'])
        half_edge_face = half_edge_face.at[he_idx].set(he_data['face'])
        half_edge_twin = half_edge_twin.at[he_idx].set(he_data['twin'])
        half_edge_next = half_edge_next.at[he_idx].set(he_data['next'])
        half_edge_prev = half_edge_prev.at[he_idx].set(he_data['prev'])

    # Set vertex half-edges (one outgoing half-edge per vertex)
    vertex_half_edge = state.vertex_half_edge
    for v_idx in range(n_vertices):
        # Find any half-edge with this vertex as source
        for (v_from, v_to), he_idx in edge_dict.items():
            if v_from == v_idx:
                vertex_half_edge = vertex_half_edge.at[v_idx].set(he_idx)
                break

    # Set face data
    face_idx = state.face_idx.at[:n_faces].set(jnp.arange(n_faces))
    face_half_edge_array = state.face_half_edge
    for f_idx, face in enumerate(faces_list):
        v0, v1, v2 = face
        he_idx = edge_dict[(v0, v1)]
        face_half_edge_array = face_half_edge_array.at[f_idx].set(he_idx)

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
        face_half_edge=face_half_edge_array
    )

def get_vertex_half_edge(state, vertex_a, vertex_b):
    """Get half edge from vertex_a to vertex_b"""
    start_he = state.vertex_half_edge[vertex_a]

    def cond_func(carry):
        he, first_iter = carry
        # Continue if: (dest != vertex_b) AND (not back at start OR first iteration)
        return jnp.logical_and(
            state.half_edge_dest[he] != vertex_b,
            jnp.logical_or(he != start_he, first_iter)
        )

    def body_func(carry):
        he, _ = carry
        next_he = state.half_edge_next[state.half_edge_twin[he]]
        return (next_he, False)

    half_edge_ab, _ = jax.lax.while_loop(
        cond_func,
        body_func,
        (start_he, True)
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
                    new_face_idx >= MAX_FACES,
                    jnp.logical_or(new_vertex_idx >= MAX_VERTICES, new_half_edge_idx + 4 >= MAX_HALF_EDGES)
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

    # Build updates
    updates = {}

    # Update vertex references
    updates['vertex_idx'] = state.vertex_idx.at[vertex_c].set(vertex_c)
    updates['vertex_pos'] = state.vertex_pos.at[vertex_c].set(new_pos)
    updates['vertex_half_edge'] = state.vertex_half_edge.at[vertex_c].set(half_edge_ca)

    # Update face references
    updates['face_idx'] = state.face_idx.at[new_face_idx].set(new_face_idx)
    updates['face_half_edge'] = state.face_half_edge.at[new_face_idx].set(half_edge_ab)

    # Update half edge indices
    half_edge_idx_arr = jnp.array([half_edge_bc, half_edge_ca, half_edge_cb, half_edge_ac])
    half_edge_idx_updates = jnp.array([half_edge_bc, half_edge_ca, half_edge_cb, half_edge_ac])
    updates['half_edge_idx'] = state.half_edge_idx.at[half_edge_idx_arr].set(half_edge_idx_updates)

    # Update half edge faces
    half_edge_face_idx = jnp.array([half_edge_ab, half_edge_bc, half_edge_ca, half_edge_cb, half_edge_ac])
    half_edge_face_updates = jnp.array([new_face_idx, new_face_idx, new_face_idx, -1, -1])
    updates['half_edge_face'] = state.half_edge_face.at[half_edge_face_idx].set(half_edge_face_updates)

    # Update half edge destinations
    half_edge_dest_idx = jnp.array([half_edge_bc, half_edge_ca, half_edge_cb, half_edge_ac])
    half_edge_dest_updates = jnp.array([vertex_c, vertex_a, vertex_b, vertex_c])
    updates['half_edge_dest'] = state.half_edge_dest.at[half_edge_dest_idx].set(half_edge_dest_updates)

    # Update half edge twins
    half_edge_twin_idx = jnp.array([half_edge_bc, half_edge_cb, half_edge_ca, half_edge_ac])
    half_edge_twin_updates = jnp.array([half_edge_cb, half_edge_bc, half_edge_ac, half_edge_ca])
    updates['half_edge_twin'] = state.half_edge_twin.at[half_edge_twin_idx].set(half_edge_twin_updates)

    # Update half edge next
    half_edge_next_idx = jnp.array([half_edge_ab, half_edge_bc, half_edge_ca, half_edge_cb, half_edge_ac, half_edge_a_prev])
    half_edge_next_updates = jnp.array([half_edge_bc, half_edge_ca, half_edge_ab, half_edge_b_next, half_edge_cb, half_edge_ac])
    updates['half_edge_next'] = state.half_edge_next.at[half_edge_next_idx].set(half_edge_next_updates)

    # Update half edge prev
    half_edge_prev_idx = jnp.array([half_edge_ab, half_edge_bc, half_edge_ca, half_edge_cb, half_edge_ac, half_edge_b_next])
    half_edge_prev_updates = jnp.array([half_edge_ca, half_edge_ab, half_edge_bc, half_edge_ac, half_edge_a_prev, half_edge_cb])
    updates['half_edge_prev'] = state.half_edge_prev.at[half_edge_prev_idx].set(half_edge_prev_updates)

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

    half_edge_ba_next = state.half_edge_next[half_edge_ba]
    half_edge_ba_prev = state.half_edge_prev[half_edge_ba]
    new_vertex_pos = jnp.mean(jnp.stack([state.vertex_pos[vertex_a], state.vertex_pos[vertex_b]], axis=0), axis=0)

    # Build updates
    updates = {}

    # Update face references
    face_idx_arr = jnp.array([face_dbc])
    face_idx_updates = jnp.array([face_dbc])
    updates['face_idx'] = state.face_idx.at[face_idx_arr].set(face_idx_updates)

    face_he_idx = jnp.array([face_dbc, face_adc])
    face_he_updates = jnp.array([half_edge_bc, half_edge_ca])
    updates['face_half_edge'] = state.face_half_edge.at[face_he_idx].set(face_he_updates)

    # Update vertex references
    updates['vertex_idx'] = state.vertex_idx.at[vertex_d].set(vertex_d)
    updates['vertex_pos'] = state.vertex_pos.at[vertex_d].set(new_vertex_pos)
    updates['vertex_half_edge'] = state.vertex_half_edge.at[vertex_d].set(half_edge_db)

    # Update half edge indices
    half_edge_idx_arr = jnp.array([half_edge_db, half_edge_da, half_edge_dc, half_edge_cd])
    half_edge_idx_updates = jnp.array([half_edge_db, half_edge_da, half_edge_dc, half_edge_cd])
    updates['half_edge_idx'] = state.half_edge_idx.at[half_edge_idx_arr].set(half_edge_idx_updates)

    # Update half edge faces
    half_edge_face_idx = jnp.array([half_edge_ad, half_edge_db, half_edge_da, half_edge_dc, half_edge_cd, half_edge_ca, half_edge_bc])
    half_edge_face_updates = jnp.array([face_adc, face_dbc, -1, face_adc, face_dbc, face_adc, face_dbc])
    updates['half_edge_face'] = state.half_edge_face.at[half_edge_face_idx].set(half_edge_face_updates)

    # Update half edge destinations
    half_edge_dest_idx = jnp.array([half_edge_ad, half_edge_db, half_edge_bd, half_edge_da, half_edge_dc, half_edge_cd])
    half_edge_dest_updates = jnp.array([vertex_d, vertex_b, vertex_d, vertex_a, vertex_c, vertex_d])
    updates['half_edge_dest'] = state.half_edge_dest.at[half_edge_dest_idx].set(half_edge_dest_updates)

    # Update half edge twins
    half_edge_twin_idx = jnp.array([half_edge_ad, half_edge_db, half_edge_bd, half_edge_da, half_edge_dc, half_edge_cd])
    half_edge_twin_updates = jnp.array([half_edge_da, half_edge_bd, half_edge_db, half_edge_ad, half_edge_cd, half_edge_dc])
    updates['half_edge_twin'] = state.half_edge_twin.at[half_edge_twin_idx].set(half_edge_twin_updates)

    # Update half edge next
    half_edge_next_idx = jnp.array([half_edge_ad, half_edge_db, half_edge_ba_prev, half_edge_bd, half_edge_da, half_edge_dc, half_edge_cd, half_edge_ca, half_edge_bc])
    half_edge_next_updates = jnp.array([half_edge_dc, half_edge_bc, half_edge_bd, half_edge_da, half_edge_ba_next, half_edge_ca, half_edge_db, half_edge_ad, half_edge_cd])
    updates['half_edge_next'] = state.half_edge_next.at[half_edge_next_idx].set(half_edge_next_updates)

    # Update half edge prev
    half_edge_prev_idx = jnp.array([half_edge_ad, half_edge_db, half_edge_ba_next, half_edge_da, half_edge_dc, half_edge_cd, half_edge_ca, half_edge_bc])
    half_edge_prev_updates = jnp.array([half_edge_ca, half_edge_cd, half_edge_da, half_edge_bd, half_edge_ad, half_edge_bc, half_edge_dc, half_edge_db])
    updates['half_edge_prev'] = state.half_edge_prev.at[half_edge_prev_idx].set(half_edge_prev_updates)

    return state._replace(**updates)

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

    # edit face aec
    half_edge_ec = jnp.max(state.half_edge_idx) + 1
    half_edge_ea = half_edge_ec + 1
    half_edge_ce = half_edge_ea + 1
    half_edge_bc = state.half_edge_next[half_edge_ab]
    half_edge_ca = state.half_edge_next[half_edge_bc]
    half_edge_ad = state.half_edge_next[half_edge_ba]
    half_edge_db = state.half_edge_next[half_edge_ad]

    # edit face ebc
    half_edge_eb = half_edge_ce + 1

    # edit face ead
    half_edge_de = half_edge_eb + 1
    half_edge_ed = half_edge_de + 1

    new_vertex_pos = jnp.mean(jnp.stack([state.vertex_pos[vertex_a], state.vertex_pos[vertex_b]], axis=0), axis=0)

    # Build updates
    updates = {}

    # Update vertex references
    updates['vertex_idx'] = state.vertex_idx.at[vertex_e].set(vertex_e)
    updates['vertex_pos'] = state.vertex_pos.at[vertex_e].set(new_vertex_pos)
    updates['vertex_half_edge'] = state.vertex_half_edge.at[vertex_e].set(half_edge_eb)

    # Update face references
    face_idx_arr = jnp.array([face_ebc, face_ead])
    face_idx_updates = jnp.array([face_ebc, face_ead])
    updates['face_idx'] = state.face_idx.at[face_idx_arr].set(face_idx_updates)

    face_he_idx = jnp.array([face_ebc, face_ead])
    face_he_updates = jnp.array([half_edge_eb, half_edge_ea])
    updates['face_half_edge'] = state.face_half_edge.at[face_he_idx].set(face_he_updates)

    # Update half edge indices
    half_edge_idx_arr = jnp.array([half_edge_ec, half_edge_ea, half_edge_eb, half_edge_ce, half_edge_de, half_edge_ed])
    half_edge_idx_updates = jnp.array([half_edge_ec, half_edge_ea, half_edge_eb, half_edge_ce, half_edge_de, half_edge_ed])
    updates['half_edge_idx'] = state.half_edge_idx.at[half_edge_idx_arr].set(half_edge_idx_updates)

    # Update half edge faces
    half_edge_face_idx = jnp.array([half_edge_ec, half_edge_eb, half_edge_bc, half_edge_ce, half_edge_ea, half_edge_ad, half_edge_de, half_edge_be, half_edge_ed])
    half_edge_face_updates = jnp.array([face_aec, face_ebc, face_ebc, face_ebc, face_ead, face_ead, face_ead, face_bed, face_bed])
    updates['half_edge_face'] = state.half_edge_face.at[half_edge_face_idx].set(half_edge_face_updates)

    # Update half edge destinations
    half_edge_dest_idx = jnp.array([half_edge_ae, half_edge_ec, half_edge_eb, half_edge_ce, half_edge_ea, half_edge_de, half_edge_be, half_edge_ed])
    half_edge_dest_updates = jnp.array([vertex_e, vertex_c, vertex_b, vertex_e, vertex_a, vertex_e, vertex_e, vertex_d])
    updates['half_edge_dest'] = state.half_edge_dest.at[half_edge_dest_idx].set(half_edge_dest_updates)

    # Update half edge twins
    half_edge_twin_idx = jnp.array([half_edge_ae, half_edge_ec, half_edge_ce, half_edge_eb, half_edge_be, half_edge_ea, half_edge_de, half_edge_ed])
    half_edge_twin_updates = jnp.array([half_edge_ea, half_edge_ce, half_edge_ec, half_edge_be, half_edge_eb, half_edge_ae, half_edge_ed, half_edge_de])
    updates['half_edge_twin'] = state.half_edge_twin.at[half_edge_twin_idx].set(half_edge_twin_updates)

    # Update half edge next
    half_edge_next_idx = jnp.array([half_edge_ae, half_edge_ec, half_edge_ca, half_edge_eb, half_edge_bc, half_edge_ce, half_edge_ea, half_edge_ad, half_edge_de, half_edge_be, half_edge_ed, half_edge_db])
    half_edge_next_updates = jnp.array([half_edge_ec, half_edge_ca, half_edge_ae, half_edge_bc, half_edge_ce, half_edge_eb, half_edge_ad, half_edge_de, half_edge_ea, half_edge_ed, half_edge_db, half_edge_be])
    updates['half_edge_next'] = state.half_edge_next.at[half_edge_next_idx].set(half_edge_next_updates)

    # Update half edge prev
    half_edge_prev_idx = jnp.array([half_edge_ec, half_edge_ca, half_edge_ae, half_edge_eb, half_edge_bc, half_edge_ce, half_edge_ea, half_edge_ad, half_edge_de, half_edge_be, half_edge_ed, half_edge_db])
    half_edge_prev_updates = jnp.array([half_edge_ae, half_edge_ec, half_edge_ca, half_edge_ce, half_edge_eb, half_edge_bc, half_edge_de, half_edge_ea, half_edge_ad, half_edge_db, half_edge_be, half_edge_ed])
    updates['half_edge_prev'] = state.half_edge_prev.at[half_edge_prev_idx].set(half_edge_prev_updates)

    return state._replace(**updates)

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
    
@jax.jit
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
    face_idx = jnp.array([face_adc, face_bcd])
    face_updates = jnp.array([half_edge_ca, half_edge_db])
    updates['face_half_edge'] = state.face_half_edge.at[face_idx].set(face_updates)
    
    # Update vertex references (in case they pointed to the flipped edge)
    vertex_idx = jnp.array([vertex_a, vertex_b])
    vertex_updates = jnp.array([half_edge_ad, half_edge_bc])
    updates['vertex_half_edge'] = state.vertex_half_edge.at[vertex_idx].set(vertex_updates)

    # Update the flipped edge
    half_edge_dest_idx = jnp.array([half_edge_dc, half_edge_cd])
    half_edge_dest_updates = jnp.array([vertex_c, vertex_d])
    updates['half_edge_dest'] = state.half_edge_dest.at[half_edge_dest_idx].set(half_edge_dest_updates)
    
    # Update surrounding half edges
    half_edge_next_idx = jnp.array([half_edge_dc, half_edge_cd, half_edge_ca, half_edge_ad, half_edge_db, half_edge_bc])
    half_edge_next_updates = jnp.array([half_edge_ca, half_edge_db, half_edge_ad, half_edge_dc, half_edge_bc, half_edge_cd])
    updates['half_edge_next'] = state.half_edge_next.at[half_edge_next_idx].set(half_edge_next_updates)
    
    half_edge_prev_idx = jnp.array([half_edge_dc, half_edge_cd, half_edge_ca, half_edge_ad, half_edge_db, half_edge_bc])
    half_edge_prev_updates = jnp.array([half_edge_ad, half_edge_bc, half_edge_dc, half_edge_ca, half_edge_cd, half_edge_db])
    updates['half_edge_prev'] = state.half_edge_prev.at[half_edge_prev_idx].set(half_edge_prev_updates)
    
    half_edge_face_idx = jnp.array([half_edge_dc, half_edge_cd, half_edge_ca, half_edge_ad, half_edge_db, half_edge_bc])
    half_edge_face_updates = jnp.array([face_adc, face_bcd, face_adc, face_adc, face_bcd, face_bcd])
    updates['half_edge_face'] = state.half_edge_face.at[half_edge_face_idx].set(half_edge_face_updates)
    
    return state._replace(**updates)

@jax.jit
def refine_mesh(state):
    """Refine mesh by flipping edges to optimize vertex valences"""
    edge_count = get_edge_count(state)

    # Find internal edges (both faces exist)
    edges = jnp.stack([state.half_edge_dest, state.half_edge_dest[state.half_edge_twin]], axis=-1)

    # get two other vertices
    vertex_c = state.half_edge_dest[state.half_edge_next]
    vertex_d = state.half_edge_dest[state.half_edge_next[state.half_edge_twin]]

    quad_half_edges = state.half_edge_idx

    # Pre-compute validity mask (optimization: compute once instead of per-edge in loop)
    valid_mask = (
        (edges[:, 0] < edges[:, 1]) &
        (state.half_edge_face != -1) &
        (state.half_edge_face[state.half_edge_twin] != -1) &
        (quad_half_edges != -1)
    )

    edges_data = jnp.stack([
        edges[:, 0],
        edges[:, 1],
        vertex_c,
        vertex_d,
        quad_half_edges,
        state.half_edge_face,
        state.half_edge_face[state.half_edge_twin],
        valid_mask.astype(jnp.int32),  # Add pre-computed validity
    ], axis=-1)

    # Process each internal edge
    (state, _), _ = jax.lax.scan(check_flip_edge, (state, edge_count), edges_data)
    return state

def check_flip_edge(state_and_counts, edge_data):

    state, edge_count = state_and_counts

    edge = edge_data[:2]
    vertex_c, vertex_d, half_edge_idx = edge_data[2], edge_data[3], edge_data[4]
    is_valid = edge_data[7] == 1  # Use pre-computed validity mask

    return jax.lax.cond(
        is_valid,
        lambda: _check_flip_edge(state, edge_count, edge, vertex_c, vertex_d, half_edge_idx),
        lambda: state_and_counts
    ), None # No output needed for scan

@jax.jit
def _check_flip_edge(state, edge_count, edge, vertex_c, vertex_d, half_edge_idx):

    vertex_quad = jnp.array([edge[0], edge[1], vertex_c, vertex_d])
    vertex_quad_edge_counts = edge_count[vertex_quad]
    
    no_flip_valence_weights = jnp.full((4,), 6)
    flip_valence_weights = jnp.array([7, 7, 5, 5])

    # Get current valences
    no_flip_valence = jnp.sum(jnp.square(vertex_quad_edge_counts - no_flip_valence_weights))
    flip_valence = jnp.sum(jnp.square(vertex_quad_edge_counts - flip_valence_weights))
    
    should_flip = flip_valence < no_flip_valence
    state = jax.lax.cond(
        should_flip,
        lambda: flip_edge(state, half_edge_idx),
        lambda: state
    )

    # Update edge counts: if we flipped, a and b lose one edge, c and d gain one edge
    edge_count_delta = jnp.zeros_like(edge_count)
    edge_count_delta = edge_count_delta.at[edge[0]].add(jnp.where(should_flip, -1, 0))
    edge_count_delta = edge_count_delta.at[edge[1]].add(jnp.where(should_flip, -1, 0))
    edge_count_delta = edge_count_delta.at[vertex_c].add(jnp.where(should_flip, 1, 0))
    edge_count_delta = edge_count_delta.at[vertex_d].add(jnp.where(should_flip, 1, 0))

    new_edge_count = edge_count + edge_count_delta

    return state, new_edge_count

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
    out = state.vertex_state[:, 0]
    # apply sigmoid
    probs = jax.nn.sigmoid(out)
    probs *= (state.vertex_idx != -1)
    divide_vertices = jax.random.bernoulli(subkey, probs)
    
    # Get its half edge and destination
    chosen_half_edges = state.vertex_half_edge
    chosen_vertices = state.half_edge_dest[state.half_edge_twin[chosen_half_edges]]
    dest_vertices = state.half_edge_dest[chosen_half_edges]

    # Check if on boundary
    chosen_on_boundary = state.half_edge_face[chosen_half_edges] == -1
    twin_on_boundary = state.half_edge_face[state.half_edge_twin[chosen_half_edges]] == -1
    
    key, subkey = jax.random.split(key)

    state, _ = jax.lax.scan(
        lambda s, args: generate_new_triangle(*args, s),
        state,
        (divide_vertices, chosen_vertices, dest_vertices, chosen_on_boundary, twin_on_boundary)
    )
    return state, key

def generate_new_triangle(divide_vertex, chosen_vertex, dest_vertex, chosen_on_boundary, twin_on_boundary, state):
    state = jax.lax.cond(
        divide_vertex,
        lambda: jax.lax.cond(
            jnp.logical_or(chosen_on_boundary, twin_on_boundary),
            lambda: add_boundary_triangle(state, chosen_vertex, dest_vertex, chosen_on_boundary, twin_on_boundary),
            lambda: add_internal_triangles(state, chosen_vertex, dest_vertex),
        ),
        lambda: state
    )

    return state, None

def add_boundary_triangle(state, chosen_vertex, dest_vertex, chosen_on_boundary, twin_on_boundary):
    # Add triangle based on boundary status
    # state = jax.lax.cond(
    #     jax.random.uniform(subkey) < 0.0,
    #     lambda: jax.lax.cond(
    #         chosen_on_boundary,
    #         lambda: add_external_triangle(state, chosen_vertex, dest_vertex),
    #         lambda: add_external_triangle(state, dest_vertex, chosen_vertex)
    #     ),
    #     lambda: jax.lax.cond(
    #         twin_on_boundary,
    #         lambda: add_internal_edge_triangle(state, chosen_vertex, dest_vertex),
    #         lambda: add_internal_edge_triangle(state, dest_vertex, chosen_vertex)
    #     )
    # )

    state = jax.lax.cond(
        twin_on_boundary,
        lambda: add_internal_edge_triangle(state, chosen_vertex, dest_vertex),
        lambda: add_internal_edge_triangle(state, dest_vertex, chosen_vertex)
    )

    return state

def rsh_cart_1(xyz):
    """Computes all real spherical harmonics up to degree 1.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,4) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
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


def relu(x):
    return jnp.maximum(0, x)

def update_vertex_state(state: MeshState, params: MeshParams):
    return ca_update(state, params)

def gaussian(x, mu, sigma):
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

def ca_update(state: MeshState, params: MeshParams):
    edges = jnp.stack([state.half_edge_dest, state.half_edge_dest[state.half_edge_twin]], axis=-1)
    edge_states = state.vertex_state[edges]
    u = jnp.zeros((state.vertex_state.shape[0], edge_states.shape[-1]))
    u = u.at[edges[:, 0]].add(edge_states[:, 1])
    u /= get_edge_count(state)[:, jnp.newaxis] + EPSILON

    state.vertex_state += params.dt * gaussian(u, params.mu, params.sigma)
    return state

def nca_update(state: MeshState, params: MeshParams):
    edges = jnp.stack([state.half_edge_dest, state.half_edge_dest[state.half_edge_twin]], axis=-1)
    edge_positions = state.vertex_pos[edges]
    edge_vectors = edge_positions[:, 1] - edge_positions[:, 0]
    edge_distances = jnp.linalg.norm(edge_vectors, axis=-1, keepdims=True)
    edge_unit_dir = edge_vectors / (edge_distances + EPSILON)
    edge_dist_norm = edge_distances / params.spring_len

    # get edge theta and phi in polar coordinates
    sh_coefs = rsh_cart_1(edge_unit_dir)

    edge_states = state.vertex_state[edges]
    feature_diff = edge_states[:, 1] - edge_states[:, 0]
    edge_message = edge_dist_norm[..., jnp.newaxis] * sh_coefs[..., jnp.newaxis] * feature_diff[..., jnp.newaxis, :]
    edge_message = edge_message.reshape(edge_message.shape[0], -1)
    z = jnp.zeros((state.vertex_state.shape[0], edge_message.shape[-1]))
    z = z.at[edges[:, 0]].add(edge_message)
    s = state.vertex_state
    s = jnp.concatenate([s, z], axis=-1)

    num_mlp_layers = len(params.vertex_state_mlp_params)
    for i in range(num_mlp_layers):
        w, b = params.vertex_state_mlp_params[f'layer{i+1}']
        s = relu(jnp.einsum('ij,jk->ik', s, w) + b)

    vertex_state = state.vertex_state + s

    return state._replace(vertex_state=vertex_state)


def generalized_eigh(A, B):
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.T
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = L_inv.T @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original

def get_natural_frequencies(state: MeshState, params: MeshParams):

    vertices = state.vertex_pos[state.vertex_idx != -1]
    edges = jnp.stack([
        state.half_edge_dest[state.half_edge_dest != -1], 
        state.half_edge_dest[state.half_edge_twin[state.half_edge_dest != -1]]
    ], axis=-1)

    return _get_natural_frequencies(vertices, edges, state, params)

@functools.partial(jax.jit, static_argnames=('params',))
def _get_natural_frequencies(vertices, edges, state: MeshState, params: MeshParams):
    M = jnp.diag(vertices.repeat(params.num_dims))
    K = jnp.zeros((state.vertex_idx.shape[0] * params.num_dims, state.vertex_idx.shape[0] * params.num_dims))
    
    edge_positions = state.vertex_pos[edges]
    edge_directions = edge_positions[:, 1] - edge_positions[:, 0]
    edge_lengths = jnp.linalg.norm(edge_directions, axis=-1, keepdims=True)
    edge_directions = edge_directions / (edge_lengths + EPSILON)

    K_spring = params.elastic_constant * jnp.einsum('ij,ik->ijk', edge_directions, edge_directions)
    K = K.at[edges[:, 0]*params.num_dims:(edges[:, 0]+1)*params.num_dims, edges[:, 0]*params.num_dims:(edges[:, 0]+1)*params.num_dims].add(K_spring)
    K = K.at[edges[:, 1]*params.num_dims:(edges[:, 1]+1)*params.num_dims, edges[:, 1]*params.num_dims:(edges[:, 1]+1)*params.num_dims].add(K_spring)
    K = K.at[edges[:, 0]*params.num_dims:(edges[:, 0]+1)*params.num_dims, edges[:, 1]*params.num_dims:(edges[:, 1]+1)*params.num_dims].add(-K_spring)
    K = K.at[edges[:, 1]*params.num_dims:(edges[:, 1]+1)*params.num_dims, edges[:, 0]*params.num_dims:(edges[:, 0]+1)*params.num_dims].add(-K_spring)

    # Solve generalized eigenvalue problem
    eigenvals, eigenvecs = generalized_eigh(K, M)
    natural_freqs = jnp.sqrt(eigenvals)
    return natural_freqs, eigenvecs

@jax.jit
def calculate_vertex_normals(state):
    # Filter to only active faces
    active_faces = state.face_idx != -1
    active_face_half_edges = jnp.where(active_faces, state.face_half_edge, 0)
    
    # Get triangle vertices - only for active faces
    triangle_half_edges = jnp.stack([
        active_face_half_edges,
        state.half_edge_next[active_face_half_edges],
        state.half_edge_next[state.half_edge_next[active_face_half_edges]],
    ], -1)
    triangle_vertices = state.half_edge_dest[triangle_half_edges]

    # Get vertex positions for all triangles
    triangle_positions = state.vertex_pos[triangle_vertices]

    # Calculate face normals using cross product
    edge1 = triangle_positions[:, 1] - triangle_positions[:, 0]
    edge2 = triangle_positions[:, 2] - triangle_positions[:, 0]
    face_normals = jnp.cross(edge1, edge2)

    # Mask out inactive faces
    face_normals = jnp.where(active_faces[:, None], face_normals, 0)

    vertex_normals = jnp.zeros_like(state.vertex_pos)
    vertex_normals = vertex_normals.at[triangle_vertices[:, 0]].add(face_normals)
    vertex_normals = vertex_normals.at[triangle_vertices[:, 1]].add(face_normals)
    vertex_normals = vertex_normals.at[triangle_vertices[:, 2]].add(face_normals)
    
    # Normalize
    vertex_normals = vertex_normals / (jnp.linalg.norm(vertex_normals, axis=1, keepdims=True) + EPSILON)
    return vertex_normals

def camera_matrix(width, height):
    # Camera position - looking at the mesh from above and in front
    eye = glm.vec3(width/2, height/2, 800)
    target = glm.vec3(width/2, height/2, 0)
    up = glm.vec3(0, 1, 0)
    
    view = glm.lookAt(eye, target, up)
    projection = glm.perspective(glm.radians(45.0), width/height, 0.1, 2000.0)
    
    return projection * view

@jax.jit
def get_indices(state, all_half_edges_with_faces):
    

    # Build triangles for all half-edges with faces
    triangle_half_edges = jnp.stack([
        all_half_edges_with_faces,
        state.half_edge_next[all_half_edges_with_faces],
        state.half_edge_next[state.half_edge_next[all_half_edges_with_faces]],
    ], -1)
    triangle_vertices = state.half_edge_dest[triangle_half_edges]

    # Front face indices (original order)
    indices_front = triangle_vertices.reshape(-1)

    # Back face indices (swap v1 and v2 to reverse winding)
    triangle_vertices_back = triangle_vertices[:, [0, 2, 1]]
    indices_back = triangle_vertices_back.reshape(-1)

    # Flatten to get index array
    indices = jnp.concatenate([indices_front, indices_back])
    return indices

def test(state):
    assert jnp.all(state.half_edge_dest[state.vertex_half_edge] == state.half_edge_dest[state.half_edge_next[state.half_edge_prev[state.vertex_half_edge]]])
    assert jnp.all(state.half_edge_face[state.half_edge_idx] == state.half_edge_face[state.half_edge_next[state.half_edge_idx]]), f"Half edge {jnp.where(state.half_edge_face[state.half_edge_idx] != state.half_edge_face[state.half_edge_next[state.half_edge_idx]])[0].tolist()} have different faces than their next half edges"
    assert jnp.all(state.half_edge_idx == state.half_edge_idx[state.half_edge_next[state.half_edge_prev[state.half_edge_idx]]]), f"Half edge {jnp.where(state.half_edge_idx != state.half_edge_idx[state.half_edge_next[state.half_edge_prev[state.half_edge_idx]]])[0].tolist()} do not loop back to themselves when going next then prev"

    # assert jnp.all(jnp.bitwise_xor(state.half_edge_dest[state.half_edge_next[state.vertex_half_edge]] == -1, state.half_edge_dest[state.half_edge_next[state.vertex_half_edge]] != state.half_edge_dest[state.half_edge_next[state.half_edge_twin[state.vertex_half_edge]]]))

    max_iter = 100
    complete = jnp.zeros((state.vertex_idx.shape[0],), dtype=bool)
    iter = 0
    start_vertex = state.half_edge_dest[state.vertex_half_edge]
    this_half_edge = state.vertex_half_edge
    def cond_func(carry):
        complete, iter, this_half_edge, start_vertex, max_iter = carry
        return ~jnp.all(complete) & (iter < max_iter)

    def body_func(carry):
        complete, iter, this_half_edge, start_vertex, max_iter = carry
        next_half_edge = state.half_edge_next[state.half_edge_twin[this_half_edge]]
        next_vertex = state.half_edge_dest[next_half_edge]
        complete = complete | (next_vertex == start_vertex)
        this_half_edge = next_half_edge
        iter += 1
        return (complete, iter, this_half_edge, start_vertex, max_iter)

    (complete, iter, this_half_edge, start_vertex, max_iter) = jax.lax.while_loop(
        cond_func,
        body_func,
        (complete, iter, this_half_edge, start_vertex, max_iter)
    )
    assert iter < max_iter, "Looping over half edges did not complete in max iterations"

def get_on_boundary(state: MeshState):
    # get vertices on external edges
    vertex_half_edge = state.vertex_half_edge
    finish_vertex_half_edge = vertex_half_edge.copy()
    start_vertex_half_edge = state.half_edge_next[state.half_edge_twin[vertex_half_edge]]
    on_boundary = jnp.logical_and(state.half_edge_face[finish_vertex_half_edge] == -1, finish_vertex_half_edge != -1)

    def cond_fun(carry):
        vertex_half_edge, _ = carry
        return jnp.any(vertex_half_edge != finish_vertex_half_edge)

    def body_fun(carry):
        vertex_half_edge, on_boundary = carry
        on_boundary = jnp.logical_or(on_boundary, jnp.logical_and(state.half_edge_face[vertex_half_edge] == -1, vertex_half_edge != -1))
        current_half_edge = state.half_edge_next[state.half_edge_twin[vertex_half_edge]]
        return (current_half_edge, on_boundary)

    (_, on_boundary) = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (start_vertex_half_edge, on_boundary)
    )
    return on_boundary

@jax.jit
def set_vertex_state_edge(state: MeshState):
    on_boundary = get_on_boundary(state)
    vertex_state = state.vertex_state.at[:, 0].set(jnp.where(on_boundary, 1.0, 0.0))
    return state._replace(vertex_state=vertex_state)

def create_initial_state_and_params(
        subkeys, 
        spring_len=40.0,
        elastic_constant=0.1,
        repulsion_distance=200.0,
        repulsion_strength=2.0,
        bulge_strength=10.0,
        planar_strength=0.1,
        num_dims=3, 
        state_dims=8, 
        hidden_state_dims=32, 
        num_mlp_layers=2,
        num_colour_channels=1,
        scale=0.1
    ):
    state = create_initial_state(subkeys[0], num_dims=num_dims, state_dims=state_dims)

    num_sph = 4 # 4 spherical harmonics up to degree 1
    mlp_params = {}
    for i in range(num_mlp_layers):
        if i == 0:
            in_dim = state_dims + num_sph * state_dims  
        else:
            in_dim = hidden_state_dims
        if i == num_mlp_layers - 1:
            out_dim = state_dims
        else:
            out_dim = hidden_state_dims
        w = scale * jax.random.normal(subkeys[i + 1], (in_dim, out_dim))
        b = jnp.zeros((out_dim,))
        mlp_params[f'layer{i+1}'] = (w, b)

    params = MeshParams(
        spring_len=spring_len,
        elastic_constant=elastic_constant,
        repulsion_distance=repulsion_distance,
        repulsion_strength=repulsion_strength,
        bulge_strength=bulge_strength,
        planar_strength=planar_strength,
        num_dims=num_dims,
        vertex_state_mlp_params=mlp_params,
        num_colour_channels=num_colour_channels
    )
    return state, params

def get_vertex_colour(state: MeshState, params: MeshParams):
    return  0.5 * jnp.tanh(state.vertex_state[:, :3]) + 0.5

def get_vertex_colour_grayscale(state: MeshState, params: MeshParams):
    grayscale = 0.5 * jnp.tanh(state.vertex_state[:, 0:1]) + 0.5
    return jnp.concatenate([grayscale, grayscale, grayscale], axis=-1)


def jax_to_torch_gpu(jax_array):
    """Zero-copy JAX to PyTorch conversion via DLPack (GPU only)."""
    return torch.from_dlpack(jax.dlpack.to_dlpack(jax_array))


@jax.jit
def extract_mesh_for_nvdiffrast(state: MeshState):
    """
    Extract mesh data from JAX half-edge structure for nvdiffrast.

    Returns:
        verts: JAX array (MAX_VERTICES, 3) vertex positions
        faces: JAX array (MAX_FACES, 3) triangle vertex indices
        n_active_faces: number of active faces
    """
    verts = state.vertex_pos

    # Get face half-edges for ALL face slots (use 0 as placeholder for inactive)
    active_mask = state.face_idx != -1
    face_half_edges = jnp.where(active_mask, state.face_half_edge, 0)

    # Build triangles from the representative half-edge
    triangle_half_edges = jnp.stack([
        face_half_edges,
        state.half_edge_next[face_half_edges],
        state.half_edge_next[state.half_edge_next[face_half_edges]],
    ], -1)

    # Get face vertex indices
    faces = state.half_edge_dest[triangle_half_edges]

    # For inactive faces, set to degenerate triangle (0, 0, 0)
    faces = jnp.where(active_mask[:, None], faces, 0)

    n_active_faces = jnp.sum(active_mask)

    return verts, faces, n_active_faces


@jax.jit
def compute_vertex_colors_jax(state: MeshState):
    """Compute vertex colors from vertex state.

    Maps vertex_state to RGB using tanh activation, taking first 3 channels.
    """
    # Use first 3 channels of vertex_state (or fewer if not available)
    state_rgb = state.vertex_state[:, :3]
    # Map to [0, 1] range using tanh
    colors = 0.5 * (jnp.tanh(state_rgb) + 1.0)
    # pad to 3 channels if fewer
    colors = jnp.pad(colors, ((0, 0), (0, 3 - colors.shape[1])), mode='edge')
    return colors


@jax.jit
def compute_vertex_normals_jax(verts, faces, n_active):
    """Compute vertex normals in JAX."""
    # Get triangle vertices
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Compute face normals
    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = jnp.cross(e1, e2)

    # Accumulate normals at vertices
    vertex_normals = jnp.zeros_like(verts)
    vertex_normals = vertex_normals.at[faces[:, 0]].add(face_normals)
    vertex_normals = vertex_normals.at[faces[:, 1]].add(face_normals)
    vertex_normals = vertex_normals.at[faces[:, 2]].add(face_normals)

    # Normalize
    norm = jnp.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    vertex_normals = vertex_normals / jnp.maximum(norm, 1e-8)

    return vertex_normals


class JAXNvdiffrastRenderer:
    def __init__(self, width, height, device='cuda'):
        self.width = width
        self.height = height
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"nvdiffrast Renderer using device: {self.device}")

        # Create nvdiffrast context
        self.glctx = dr.RasterizeCudaContext()

        # Pre-compute transformation matrices
        self.center_x = width / 2
        self.center_y = height / 2

        # Camera setup (orthographic projection)
        # Map screen coordinates to clip space [-1, 1]
        # Vertices at (0, 0) -> (-1, -1), vertices at (width, height) -> (1, 1)
        self.proj_scale = 2.0 / max(width, height)

        # Lighting parameters
        self.light_dir = torch.tensor([0.3, 0.3, 1.0], device=self.device, dtype=torch.float32)
        self.light_dir = self.light_dir / torch.norm(self.light_dir)
        self.ambient = 0.25
        self.diffuse = 0.75

        print(f"Renderer initialized: {width}x{height}")

    def render(self, state: MeshState):
        """
        Render mesh using nvdiffrast.

        Args:
            state: JAX MeshState

        Returns:
            numpy array (H, W, 3) RGB image as uint8
        """
        # Extract mesh data (in JAX) - JIT compiled
        verts_jax, faces_jax, n_active = extract_mesh_for_nvdiffrast(state)

        # Compute vertex normals and colors in JAX
        normals_jax = compute_vertex_normals_jax(verts_jax, faces_jax, n_active)
        colors_jax = compute_vertex_colors_jax(state)

        # Block until JAX computation is done
        jax.block_until_ready(verts_jax)
        jax.block_until_ready(faces_jax)
        jax.block_until_ready(normals_jax)
        jax.block_until_ready(colors_jax)

        # Zero-copy conversion to PyTorch
        verts = jax_to_torch_gpu(verts_jax)
        faces = jax_to_torch_gpu(faces_jax).to(torch.int32)
        normals = jax_to_torch_gpu(normals_jax)
        vertex_colors = jax_to_torch_gpu(colors_jax)

        # Only use active faces
        n_active_int = int(n_active)
        faces = faces[:n_active_int].contiguous()

        # Transform vertices to clip space [-1, 1]
        # Screen coords (0,0) to (width,height) -> clip coords (-1,-1) to (1,1)
        verts_clip = verts.clone()
        verts_clip[:, 0] = (verts[:, 0] - self.center_x) * self.proj_scale
        verts_clip[:, 1] = (verts[:, 1] - self.center_y) * self.proj_scale
        verts_clip[:, 2] = verts[:, 2] * self.proj_scale  # Keep Z in same scale

        # Convert to homogeneous coordinates [x, y, z, w]
        verts_homo = torch.cat([verts_clip, torch.ones_like(verts_clip[:, :1])], dim=-1)

        # Add batch dimension for nvdiffrast
        verts_homo = verts_homo.unsqueeze(0).contiguous()

        with torch.no_grad():
            # Create double-sided geometry with explicit normals for each side
            # Back faces: flipped winding, negated normals (pointing inward)
            # Front faces: original winding, original normals (pointing outward)
            faces_back = faces[:, [0, 2, 1]]
            faces_double = torch.cat([faces_back, faces], dim=0)

            # Duplicate normals: negated for back faces, original for front faces
            normals_back = -normals
            normals_double = torch.cat([normals_back, normals], dim=0)

            # Duplicate colors for both sides
            colors_double = torch.cat([vertex_colors, vertex_colors], dim=0)

            # Rasterize with double-sided faces
            rast_out, _ = dr.rasterize(self.glctx, verts_homo, faces_double, (self.height, self.width))

            # Interpolate normals (using duplicated normals with correct orientation per side)
            normals_batch = normals_double.unsqueeze(0).contiguous()
            normals_interp, _ = dr.interpolate(normals_batch, rast_out, faces_double)

            # Interpolate vertex colors
            colors_batch = colors_double.unsqueeze(0).contiguous()
            colors_interp, _ = dr.interpolate(colors_batch, rast_out, faces_double)

            # Normalize interpolated normals
            normals_shading = normals_interp / (torch.norm(normals_interp, dim=-1, keepdim=True) + 1e-8)

            # Compute lighting (simple Lambertian)
            light_dir = self.light_dir.view(1, 1, 1, 3)
            ndotl = torch.clamp(torch.sum(normals_shading * light_dir, dim=-1, keepdim=True), 0, 1)

            # Apply lighting to get mesh color
            mesh_color = colors_interp * (self.ambient + self.diffuse * ndotl)

            # Apply antialiasing for smooth silhouette edges
            mesh_color = dr.antialias(mesh_color, rast_out, verts_homo, faces_double)

            # Composite over background (triangle_id > 0 means a triangle was rasterized)
            tri_id = rast_out[..., 3:4]
            mask = (tri_id > 0).float()
            bg_color = torch.full_like(mesh_color, 0.1)
            color = mesh_color * mask + bg_color * (1 - mask)

            # Convert to uint8
            color = (color[0] * 255).clamp(0, 255).to(torch.uint8)

        return color.cpu().numpy()


def enforce_boundary(state):
    z_min, z_max = 0.0, 50
    vertex_pos = jnp.clip(
        state.vertex_pos,
        a_min=jnp.array([0.0, 0.0, z_min]),
        a_max=jnp.array([800.0, 800.0, z_max])
    )
    return state._replace(vertex_pos=vertex_pos)



def main():
    width, height = 500, 500
    
    # Initialize JAX
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 5)
    
    # Create initial state and parameters
    num_dims = 3
    state, params = create_initial_state_and_params(
        subkeys, num_dims=num_dims, state_dims=1, num_mlp_layers=1, bulge_strength=0
    )
    
    # Initialize first triangle
    state = make_circle(state, width, height, params)

    # get first triangle vertices
    v0, v1, v2 = state.vertex_pos[:3]

    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF, vsync=True)
    clock = pygame.time.Clock()

    # Initialize renderer
    renderer = JAXNvdiffrastRenderer(width, height)
    
    running = True
    frame_count = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update physics
        state, repulsion_mag = update_positions(state, params)
        state = update_vertex_state(state, params)

        # ensure first triangle vertices stay fixed
        vertex_pos = state.vertex_pos.at[jnp.array([0, 1, 2])].set(jnp.array([v0, v1, v2]))
        state = state._replace(vertex_pos=vertex_pos)
        
        # Generate new triangles
        state, key = maybe_generate_new_triangles(state, params, key, repulsion_mag)
        
        # Refine mesh every 10 frames
        if frame_count % 10 == 0:
            state = refine_mesh(state)
        
        # Render with nvdiffrast
        image_np = renderer.render(state)
        # Convert to pygame surface
        surface = pygame.surfarray.make_surface(image_np.swapaxes(0, 1))

        screen.blit(surface, (0, 0))
        pygame.display.flip()

        clock.tick(60)
        frame_count += 1
    
    pygame.quit()

if __name__ == '__main__':
    main()