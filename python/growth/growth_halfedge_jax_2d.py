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
    add_external_triangle,
    create_initial_state,
    maybe_generate_new_triangles,
    refine_mesh, 
    make_first_triangle,
    calculate_spring_force,
    calculate_repulsion_force,
    EPSILON
)

@jax.jit
def update_positions(state, params, dt=0.1):
    """Update vertex positions based on forces"""
    spring_force = calculate_spring_force(state, params)
    repulsion_mag, repulsion_force = calculate_repulsion_force(state, params)
    bulge_force = calculate_bulge_force_2d(state, params)
    
    total_force = (spring_force + 
                   params.repulsion_strength * repulsion_force + 
                   params.bulge_strength * bulge_force)
    
    # Only update active vertices
    active_mask = state.vertex_idx != -1
    new_positions = state.vertex_pos + dt * total_force * active_mask[:, None]
    
    # Update suitability
    min_repulsion = jnp.min(jnp.where(repulsion_mag > 0, repulsion_mag, jnp.inf))
    min_repulsion = jnp.where(jnp.isfinite(min_repulsion), min_repulsion, 1.0)
    repulsion_norm = repulsion_mag / min_repulsion
    new_suitability = 1 / (1 + repulsion_norm ** 2) * active_mask.astype(jnp.float32)
    
    return state._replace(vertex_pos=new_positions, vertex_suitability=new_suitability), repulsion_mag


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
def calculate_bulge_force_2d(state, params):
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
        state.half_edge_next[state.half_edge_twin],
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
        # draw_pygame(state, screen)
        draw_pygame(state, screen)
        clock.tick(60)
        frame_count += 1
    
    pygame.quit()

if __name__ == '__main__':
    main()