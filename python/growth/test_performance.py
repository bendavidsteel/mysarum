import time
import numpy as np
import jax
import jax.numpy as jnp

import growth_halfedge_surface as numpy_module
import growth_halfedge_jax as jax_module

def setup_numpy_mesh(num_triangles=10):
    """Create a NumPy mesh with specified number of triangles"""
    mesh = numpy_module.Mesh()
    mesh.make_first_triangle(800, 800)

    # Add some triangles
    for i in range(min(num_triangles, 5)):
        if i == 0:
            mesh.add_external_triangle(0, 2)
        else:
            try:
                mesh.add_external_triangle(0, i+2)
            except:
                break

    return mesh

def setup_jax_mesh(num_triangles=10):
    """Create a JAX mesh with specified number of triangles"""
    state = jax_module.create_initial_state(num_dims=2)
    params = jax_module.MeshParams(
        spring_len=40.0,
        elastic_constant=0.1,
        repulsion_distance=200.0,
        repulsion_strength=2.0,
        bulge_strength=10.0,
        planar_strength=0.1
    )

    state = jax_module.make_first_triangle(state, 800, 800)

    # Add some triangles
    for i in range(min(num_triangles, 5)):
        if i == 0:
            state = jax_module.add_external_triangle(state, 0, 2)
        else:
            state = jax_module.add_external_triangle(state, 0, i+2)

    return state, params

def time_function(func, warmup=5, iterations=50):
    """Time a function with warmup"""
    # Warmup
    for _ in range(warmup):
        func()

    # Time
    start = time.time()
    for _ in range(iterations):
        result = func()
    end = time.time()

    # For JAX, block until computation completes
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()

    return (end - start) / iterations

def benchmark_force_calculations():
    """Benchmark individual force calculation functions"""
    print("\n" + "="*70)
    print("Force Calculation Benchmarks")
    print("="*70)

    # Setup
    numpy_mesh = setup_numpy_mesh()
    jax_state, jax_params = setup_jax_mesh()

    # Spring force
    print("\n1. Spring Force Calculation:")
    numpy_time = time_function(lambda: numpy_mesh.calculate_spring_force())
    jax_time = time_function(lambda: jax_module.calculate_spring_force(jax_state, jax_params))
    print(f"   NumPy: {numpy_time*1000:.4f} ms")
    print(f"   JAX:   {jax_time*1000:.4f} ms")
    print(f"   Speedup: {numpy_time/jax_time:.2f}x")

    # Repulsion force
    print("\n2. Repulsion Force Calculation:")
    numpy_time = time_function(lambda: numpy_mesh.calculate_repulsion_force())
    jax_time = time_function(lambda: jax_module.calculate_repulsion_force(jax_state, jax_params))
    print(f"   NumPy: {numpy_time*1000:.4f} ms")
    print(f"   JAX:   {jax_time*1000:.4f} ms")
    print(f"   Speedup: {numpy_time/jax_time:.2f}x")

    # Bulge force
    print("\n3. Bulge Force Calculation:")
    numpy_time = time_function(lambda: numpy_mesh.calculate_bulge_force())
    jax_time = time_function(lambda: jax_module.calculate_bulge_force(jax_state, jax_params))
    print(f"   NumPy: {numpy_time*1000:.4f} ms")
    print(f"   JAX:   {jax_time*1000:.4f} ms")
    print(f"   Speedup: {numpy_time/jax_time:.2f}x")

    # Planar force
    print("\n4. Planar Force Calculation:")
    numpy_time = time_function(lambda: numpy_mesh.calculate_planar_force())
    jax_time = time_function(lambda: jax_module.calculate_planar_force(jax_state, jax_params))
    print(f"   NumPy: {numpy_time*1000:.4f} ms")
    print(f"   JAX:   {jax_time*1000:.4f} ms")
    print(f"   Speedup: {numpy_time/jax_time:.2f}x")

def benchmark_position_update():
    """Benchmark position update (all forces combined)"""
    print("\n" + "="*70)
    print("Position Update Benchmark")
    print("="*70)

    numpy_mesh = setup_numpy_mesh()
    jax_state, jax_params = setup_jax_mesh()

    def numpy_update():
        spring_force = numpy_mesh.calculate_spring_force()
        repulsion_mag, repulsion_force = numpy_mesh.calculate_repulsion_force()
        bulge_force = numpy_mesh.calculate_bulge_force()
        planar_force = numpy_mesh.calculate_planar_force()
        force = spring_force + numpy_mesh.repulsion_strength * repulsion_force + \
                numpy_mesh.bulge_strength * bulge_force + \
                numpy_mesh.planar_strength * planar_force
        return force

    print("\nCombined force calculation:")
    numpy_time = time_function(numpy_update)
    jax_time = time_function(lambda: jax_module.update_positions(jax_state, jax_params))
    print(f"   NumPy: {numpy_time*1000:.4f} ms")
    print(f"   JAX:   {jax_time*1000:.4f} ms")
    print(f"   Speedup: {numpy_time/jax_time:.2f}x")

def benchmark_edge_operations():
    """Benchmark edge counting and refinement"""
    print("\n" + "="*70)
    print("Edge Operations Benchmark")
    print("="*70)

    numpy_mesh = setup_numpy_mesh()
    jax_state, jax_params = setup_jax_mesh()

    # Edge counting
    print("\n1. Edge Counting:")
    numpy_time = time_function(lambda: numpy_mesh.get_edge_count(), warmup=2, iterations=10)
    jax_time = time_function(lambda: jax_module.get_edge_count(jax_state), warmup=2, iterations=10)
    print(f"   NumPy: {numpy_time*1000:.4f} ms")
    print(f"   JAX:   {jax_time*1000:.4f} ms")
    print(f"   Speedup: {numpy_time/jax_time:.2f}x")

    # Mesh refinement
    print("\n2. Mesh Refinement (includes edge counting):")
    numpy_time = time_function(lambda: numpy_mesh.refine_mesh(), warmup=2, iterations=10)
    jax_time = time_function(lambda: jax_module.refine_mesh(jax_state), warmup=2, iterations=10)
    print(f"   NumPy: {numpy_time*1000:.4f} ms")
    print(f"   JAX:   {jax_time*1000:.4f} ms")
    print(f"   Speedup: {numpy_time/jax_time:.2f}x")

def benchmark_overall():
    """Benchmark overall update cycle"""
    print("\n" + "="*70)
    print("Overall Update Cycle Benchmark")
    print("="*70)

    print("\nThis simulates a full update cycle with physics and occasional triangle addition")

    # NumPy version
    numpy_mesh = setup_numpy_mesh()
    def numpy_full_update():
        spring_force = numpy_mesh.calculate_spring_force()
        repulsion_mag, repulsion_force = numpy_mesh.calculate_repulsion_force()
        bulge_force = numpy_mesh.calculate_bulge_force()
        planar_force = numpy_mesh.calculate_planar_force()
        force = spring_force + numpy_mesh.repulsion_strength * repulsion_force + \
                numpy_mesh.bulge_strength * bulge_force + \
                numpy_mesh.planar_strength * planar_force
        numpy_mesh.vertex_pos += 0.1 * force
        numpy_mesh.refine_mesh()
        return force

    # JAX version
    jax_state, jax_params = setup_jax_mesh()
    def jax_full_update():
        state, _ = jax_module.update_positions(jax_state, jax_params)
        state = jax_module.refine_mesh(state)
        return state

    numpy_time = time_function(numpy_full_update, warmup=5, iterations=100)
    jax_time = time_function(jax_full_update, warmup=5, iterations=100)

    print(f"\n   NumPy: {numpy_time*1000:.4f} ms per update")
    print(f"   JAX:   {jax_time*1000:.4f} ms per update")
    print(f"   Speedup: {numpy_time/jax_time:.2f}x")

    if jax_time > numpy_time:
        print(f"\n   ⚠️  JAX is {jax_time/numpy_time:.2f}x SLOWER than NumPy")
    else:
        print(f"\n   ✓ JAX is {numpy_time/jax_time:.2f}x faster than NumPy")

def main():
    print("\n" + "="*70)
    print("Performance Comparison: NumPy vs JAX Half-Edge Mesh")
    print("="*70)

    # Run benchmarks
    benchmark_force_calculations()
    benchmark_position_update()
    benchmark_edge_operations()
    benchmark_overall()

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)

if __name__ == "__main__":
    main()