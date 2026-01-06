import collections
import functools

import glm
import jax
import jax.numpy as jnp
import moderngl
import numpy as np
import pygame


from growth_halfedge_jax import (
    MeshParams, 
    MeshState,
    add_external_triangle,
    create_initial_state_and_params,
    refine_mesh, 
    make_first_triangle,
    calculate_spring_force,
    calculate_repulsion_force,
    add_boundary_triangle,
    add_internal_edge_triangle,
    add_internal_triangles,
    get_on_boundary,
    update_vertex_state,
    set_vertex_state_edge,
    EPSILON
)

@jax.jit
def calculate_planar_force_open(state, params):
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

    on_boundary = get_on_boundary(state)
    planar_force *= ~on_boundary[:, np.newaxis]

    return planar_force

@jax.jit
def update_positions(state: MeshState, params: MeshParams, dt=0.1):
    """Update vertex positions based on forces"""
    spring_force = calculate_spring_force(state, params)
    repulsion_mag, repulsion_force = calculate_repulsion_force(state, params)
    bulge_force = calculate_bulge_force_2d(state, params)
    planar_force = calculate_planar_force_open(state, params)

    total_force = (spring_force +
                   params.repulsion_strength * repulsion_force +
                   params.bulge_strength * bulge_force +
                   params.planar_strength * planar_force)

    # Only update active vertices
    active_mask = state.vertex_idx != -1
    new_positions = state.vertex_pos + dt * total_force * active_mask[:, None]
    
    return state._replace(vertex_pos=new_positions), repulsion_mag


def draw_pygame(state, screen):
    """Draw the mesh using pygame"""
    screen.fill((0, 0, 0))
    
    # Convert JAX arrays to numpy for drawing
    vertex_pos = np.array(state.vertex_pos)
    vertex_idx = np.array(state.vertex_idx)
    vertex_colour = np.array((0.5 * (jnp.tanh(state.vertex_state[:, :-1]) + 1) * 255).astype(np.uint8))
    half_edge_dest = np.array(state.half_edge_dest)
    half_edge_prev = np.array(state.half_edge_prev)
    half_edge_idx = np.array(state.half_edge_idx)
    # vertex_suitability = np.array(state.vertex_suitability)
    
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
        colour = vertex_colour[i]
        if len(colour) == 3:
            pygame.draw.circle(screen, (colour[0], colour[1], colour[2]), pos.astype(int), 5)
        else:
            pygame.draw.circle(screen, (colour[0], colour[0], colour[0]), pos.astype(int), 5)

    pygame.display.flip()

def make_first_triangle(state, width, height):
    """Initialize the first triangle"""
    # Vertices
    vertex_positions = jnp.array([
        [width/2, height/2],
        [width/2 + 40, height/2],
        [width/2 + 20, height/2 + 34.64]
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

@jax.jit
def calculate_bulge_force_2d(state: MeshState, params: MeshParams):
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

    edge_vector = jnp.stack([edge_vector[:, 0], edge_vector[:, 1], jnp.zeros_like(edge_vector[:, 0])], axis=-1)
    next_edge_vector = jnp.stack([next_edge_vector[:, 0], next_edge_vector[:, 1], jnp.zeros_like(next_edge_vector[:, 0])], axis=-1)

    # calculate normals
    surface_normal = jnp.cross(edge_vector, next_edge_vector)
    edge_normal = jnp.cross(edge_vector, surface_normal)[..., :2]
    
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

def generate_new_triangles(state, params, key):
    # Find active vertices

    key, subkey = jax.random.split(key)
    # Choose a random vertex weighted by suitability
    logits = state.vertex_state[:, -1]
    probs = jax.nn.softmax(logits, where=state.vertex_idx != -1)
    chosen_vertex = jax.random.choice(subkey, state.vertex_idx, p=probs)
    
    # Get its half edge and destination
    chosen_half_edge = state.vertex_half_edge[chosen_vertex]
    dest_vertex = state.half_edge_dest[chosen_half_edge]

    # Check if on boundary
    chosen_on_boundary = state.half_edge_face[chosen_half_edge] == -1
    twin_on_boundary = state.half_edge_face[state.half_edge_twin[chosen_half_edge]] == -1
    
    key, subkey = jax.random.split(key)

    state = jax.lax.cond(
        jnp.logical_or(chosen_on_boundary, twin_on_boundary),
        lambda: add_boundary_triangle(state, chosen_vertex, dest_vertex, chosen_on_boundary, twin_on_boundary, subkey),
        lambda: add_internal_triangles(state, chosen_vertex, dest_vertex),
    )

    return state, key

def add_boundary_triangle(state, chosen_vertex, dest_vertex, chosen_on_boundary, twin_on_boundary, subkey):
    # Add triangle based on boundary status
    state = jax.lax.cond(
        jax.random.uniform(subkey) < 0.5,
        lambda: state,
        lambda: jax.lax.cond(
            twin_on_boundary,
            lambda: add_internal_edge_triangle(state, chosen_vertex, dest_vertex),
            lambda: add_internal_edge_triangle(state, dest_vertex, chosen_vertex)
        )
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

@jax.jit
def boundary_conditions(state: MeshState, width, height):
    """Keep vertices within bounds"""
    vertex_pos = state.vertex_pos
    vertex_pos = jnp.clip(vertex_pos, min=0, max=jnp.array([width, height]))
    return state._replace(vertex_pos=vertex_pos)

def main():
    width, height = 1400, 1400
    
    # Initialize JAX
    key = jax.random.PRNGKey(0)
    
    # Create initial state and parameters
    num_dims = 2
    state_dims = 2
    hidden_state_dims = 4
    scale = 0.02
    key, *subkeys = jax.random.split(key, 4)
    state, params = create_initial_state_and_params(
        subkeys, 
        num_dims=num_dims, 
        state_dims=state_dims, 
        hidden_state_dims=hidden_state_dims, 
        scale=scale,
        elastic_constant=0.2,
        repulsion_distance=40.0,
        repulsion_strength=10.0,
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
        state = boundary_conditions(state, width, height)

        state = set_vertex_state_edge(state)
        state = update_vertex_state(state, params)
        
        # Generate new triangles
        state, key = maybe_generate_new_triangles(state, params, key, repulsion_mag)
        
        # Refine mesh every 10 frames
        if frame_count % 10 == 0:
            state = refine_mesh(state)

        if frame_count % 10 == 5:
            # natural_freqs, eigenvecs = get_natural_frequencies(state, params)
            pass
        
        # Draw
        # draw_pygame(state, screen)
        draw_pygame(state, screen)
        clock.tick(120)
        frame_count += 1
    
    pygame.quit()

if __name__ == '__main__':
    main()