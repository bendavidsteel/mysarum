#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from growth_halfedge import Mesh as NumPyMesh
from growth_halfedge_jax import Mesh as JAXMesh

def create_test_meshes():
    """Create identical initial meshes for both implementations"""
    width, height = 1400, 1400

    numpy_mesh = NumPyMesh()
    numpy_mesh.make_first_triangle(width, height)
    numpy_mesh.add_external_triangle(0, 2)

    jax_mesh = JAXMesh()
    jax_mesh.make_first_triangle(width, height)
    jax_mesh.add_external_triangle(0, 2)

    return numpy_mesh, jax_mesh

def compare_spring_forces():
    """Compare spring force calculations"""
    print("=== Comparing Spring Forces ===")

    numpy_mesh, jax_mesh = create_test_meshes()

    # Calculate forces
    numpy_force = numpy_mesh.calculate_spring_force()

    # For JAX version, we need to call the function directly
    from growth_halfedge_jax import calculate_spring_force
    jax_force = calculate_spring_force(
        jax_mesh.vertex_pos,
        jax_mesh.half_edge_dest,
        jax_mesh.half_edge_prev,
        jax_mesh.half_edge_idx,
        jax_mesh.spring_len,
        jax_mesh.elastic_constant
    )

    print(f"NumPy force shape: {numpy_force.shape}")
    print(f"JAX force shape: {jax_force.shape}")
    print(f"NumPy force (first 5 vertices):\n{numpy_force[:5]}")
    print(f"JAX force (first 5 vertices):\n{jax_force[:5]}")

    # Check for differences
    diff = np.array(numpy_force) - np.array(jax_force)
    print(f"Max difference: {np.max(np.abs(diff))}")
    print(f"Mean difference: {np.mean(np.abs(diff))}")

    return numpy_force, jax_force

def compare_repulsion_forces():
    """Compare repulsion force calculations"""
    print("\n=== Comparing Repulsion Forces ===")

    numpy_mesh, jax_mesh = create_test_meshes()

    # Calculate forces
    numpy_mag, numpy_force = numpy_mesh.calculate_repulsion_force()

    from growth_halfedge_jax import calculate_repulsion_force
    jax_mag, jax_force = calculate_repulsion_force(
        jax_mesh.vertex_pos,
        jax_mesh.vertex_idx,
        jax_mesh.repulsion_distance
    )

    print(f"NumPy mag shape: {numpy_mag.shape}, force shape: {numpy_force.shape}")
    print(f"JAX mag shape: {jax_mag.shape}, force shape: {jax_force.shape}")

    print(f"NumPy mag (first 5): {numpy_mag[:5]}")
    print(f"JAX mag (first 5): {jax_mag[:5]}")

    mag_diff = np.array(numpy_mag) - np.array(jax_mag)
    force_diff = np.array(numpy_force) - np.array(jax_force)

    print(f"Magnitude max diff: {np.max(np.abs(mag_diff))}")
    print(f"Force max diff: {np.max(np.abs(force_diff))}")

    return (numpy_mag, numpy_force), (jax_mag, jax_force)

def compare_bulge_forces():
    """Compare bulge force calculations"""
    print("\n=== Comparing Bulge Forces ===")

    numpy_mesh, jax_mesh = create_test_meshes()

    numpy_force = numpy_mesh.calculate_bulge_force()

    from growth_halfedge_jax import calculate_bulge_force
    jax_force = calculate_bulge_force(
        jax_mesh.vertex_pos,
        jax_mesh.half_edge_idx,
        jax_mesh.half_edge_face,
        jax_mesh.half_edge_dest,
        jax_mesh.half_edge_twin
    )

    print(f"NumPy bulge force shape: {numpy_force.shape}")
    print(f"JAX bulge force shape: {jax_force.shape}")
    print(f"NumPy force (first 5 vertices):\n{numpy_force[:5]}")
    print(f"JAX force (first 5 vertices):\n{jax_force[:5]}")

    diff = np.array(numpy_force) - np.array(jax_force)
    print(f"Max difference: {np.max(np.abs(diff))}")

    return numpy_force, jax_force

def compare_mesh_states():
    """Compare mesh state after initialization"""
    print("\n=== Comparing Initial Mesh States ===")

    numpy_mesh, jax_mesh = create_test_meshes()

    # Compare key arrays
    arrays_to_compare = [
        ('vertex_idx', numpy_mesh.vertex_idx, jax_mesh.vertex_idx),
        ('vertex_pos', numpy_mesh.vertex_pos, jax_mesh.vertex_pos),
        ('half_edge_idx', numpy_mesh.half_edge_idx, jax_mesh.half_edge_idx),
        ('half_edge_dest', numpy_mesh.half_edge_dest, jax_mesh.half_edge_dest),
        ('half_edge_face', numpy_mesh.half_edge_face, jax_mesh.half_edge_face),
    ]

    for name, numpy_arr, jax_arr in arrays_to_compare:
        diff = np.array(numpy_arr) - np.array(jax_arr)
        max_diff = np.max(np.abs(diff))
        print(f"{name}: max difference = {max_diff}")
        if max_diff > 1e-10:
            print(f"  NumPy: {numpy_arr[:10]}")
            print(f"  JAX:   {jax_arr[:10]}")

def analyze_edge_filtering():
    """Analyze edge filtering differences"""
    print("\n=== Analyzing Edge Filtering ===")

    numpy_mesh, jax_mesh = create_test_meshes()

    # Check how edges are filtered in NumPy version
    numpy_edges, numpy_edge_pos = numpy_mesh.get_edge_positions()
    print(f"NumPy edges shape: {numpy_edges.shape}")
    print(f"NumPy edge positions shape: {numpy_edge_pos.shape}")

    # Check active edges
    active_edges_mask = (numpy_edges[0] != -1) & (numpy_edges[1] != -1)
    print(f"NumPy active edges: {np.sum(active_edges_mask)}")

    # Check JAX version
    jax_edges, jax_edge_pos = jax_mesh.get_edge_positions()
    print(f"JAX edges shape: {jax_edges.shape}")
    print(f"JAX edge positions shape: {jax_edge_pos.shape}")

def main():
    print("Testing JAX vs NumPy mesh implementations")
    print("=" * 50)

    try:
        compare_mesh_states()
        analyze_edge_filtering()
        compare_spring_forces()
        compare_repulsion_forces()
        compare_bulge_forces()

    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()