import collections
import functools

import einops
import jax
import jax.numpy as jp

# Define namedtuples for parameters and fields
Params = collections.namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = collections.namedtuple('Fields', 'U G R E')
SpatialHash = collections.namedtuple('SpatialHash', 'bin_indices particle_indices bin_offsets grid_size')

def peak_f(x, mu, sigma):
    """Compute the Gaussian peak function."""
    return jp.exp(-((x - mu) / sigma) ** 2)

def compute_spatial_hash(points, bin_size, map_size):
    """
    Create spatial hash for particles to optimize neighbor search.
    
    Args:
        points: array of shape (N, 2) containing particle positions
        bin_size: size of each spatial bin
        map_size: size of simulation domain
        
    Returns:
        SpatialHash containing bin assignments and offsets for efficient neighbor lookup
    """
    n_particles = points.shape[0]
    
    # Calculate grid dimensions
    grid_size = jp.array([
        jp.ceil(map_size / bin_size).astype(jp.int32),
        jp.ceil(map_size / bin_size).astype(jp.int32)
    ])
    
    # Compute bin coordinates for each particle
    bin_coords = jp.floor(points / bin_size).astype(jp.int32)
    bin_coords = jp.clip(bin_coords, 0, grid_size - 1)
    
    # Convert 2D bin coordinates to 1D bin indices
    bin_indices = bin_coords[:, 1] * grid_size[0] + bin_coords[:, 0]
    
    # Sort particles by bin index
    sort_indices = jp.argsort(bin_indices)
    sorted_bin_indices = bin_indices[sort_indices]
    
    # Compute bin offsets using cumulative counts
    total_bins = grid_size[0] * grid_size[1]
    bin_counts = jp.bincount(sorted_bin_indices, length=total_bins)
    bin_offsets = jp.concatenate([jp.array([0]), jp.cumsum(bin_counts)])
    
    return SpatialHash(
        bin_indices=sorted_bin_indices,
        particle_indices=sort_indices, 
        bin_offsets=bin_offsets,
        grid_size=grid_size
    )

def get_neighbor_bins(bin_coord, grid_size, periodic=True):
    """
    Get the 9 neighboring bin coordinates (including center) for a given bin.
    
    Args:
        bin_coord: 2D coordinate of center bin
        grid_size: dimensions of the spatial grid
        periodic: whether to use periodic boundary conditions
        
    Returns:
        array of shape (9, 2) containing neighbor bin coordinates
    """
    # Generate 3x3 neighborhood offsets
    offsets = jp.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],  [0, 0],  [0, 1], 
        [1, -1],  [1, 0],  [1, 1]
    ])
    
    neighbor_coords = bin_coord + offsets
    
    if periodic:
        # Wrap coordinates for periodic boundaries
        neighbor_coords = neighbor_coords % grid_size
    else:
        # Clamp coordinates for bounded domain
        neighbor_coords = jp.clip(neighbor_coords, 0, grid_size - 1)
    
    return neighbor_coords

def get_neighbor_particles(particle_idx, points, spatial_hash, bin_size, map_size, max_radius=None):
    """
    Get neighboring particles for a given particle using spatial hashing.
    
    Args:
        particle_idx: index of the particle to find neighbors for
        points: array of all particle positions
        spatial_hash: precomputed spatial hash structure
        bin_size: size of spatial bins
        map_size: size of simulation domain  
        max_radius: maximum interaction radius (for early culling)
        
    Returns:
        indices of neighboring particles and their distances
    """
    # Get particle position and bin coordinate
    pos = points[particle_idx]
    bin_coord = jp.floor(pos / bin_size).astype(jp.int32)
    bin_coord = jp.clip(bin_coord, 0, spatial_hash.grid_size - 1)
    
    # Get neighboring bin coordinates
    neighbor_bins = get_neighbor_bins(bin_coord, spatial_hash.grid_size, periodic=True)
    
    # Convert to 1D bin indices
    neighbor_bin_indices = neighbor_bins[:, 1] * spatial_hash.grid_size[0] + neighbor_bins[:, 0]
    
    # Collect particles from all neighboring bins
    neighbor_indices = []
    
    for bin_idx in neighbor_bin_indices:
        start_idx = spatial_hash.bin_offsets[bin_idx]
        end_idx = spatial_hash.bin_offsets[bin_idx + 1]
        
        # Get particles in this bin
        bin_particles = spatial_hash.particle_indices[start_idx:end_idx]
        neighbor_indices.append(bin_particles)
    
    # Concatenate all neighbor indices
    if len(neighbor_indices) > 0:
        all_neighbors = jp.concatenate(neighbor_indices)
    else:
        all_neighbors = jp.array([], dtype=jp.int32)
    
    # Remove self-interaction
    all_neighbors = all_neighbors[all_neighbors != particle_idx]
    
    if len(all_neighbors) == 0:
        return jp.array([], dtype=jp.int32), jp.array([])
    
    # Compute distances to neighbors
    neighbor_positions = points[all_neighbors]
    diff = neighbor_positions - pos
    
    # Handle periodic boundaries
    diff = diff - jp.round(diff / map_size) * map_size
    distances = jp.sqrt(jp.sum(diff**2, axis=1))
    
    # Optional: filter by maximum radius
    if max_radius is not None:
        valid_mask = distances <= max_radius
        all_neighbors = all_neighbors[valid_mask]
        distances = distances[valid_mask]
    
    return all_neighbors, distances

def precomputed_r_fields_f(p: Params, species, s, r):
    """Calculate the fields U, G, R, and E based on parameters and points.
    x: shape (2,) array of coordinates
    points: shape (N, 2) array of particle positions"""
    mu_k = p.mu_k[s, species] # shape (num_points, num_kernels)
    sigma_k = p.sigma_k[s, species] # shape (num_points, num_kernels)
    u = jax.vmap(peak_f, in_axes=(None, 1, 1))(r, mu_k, sigma_k) # shape (num_kernel, num_points)
    w_k = p.w_k[s] # shape (num_kernel,)
    U = (u * einops.rearrange(w_k, "s -> s 1")).sum() # shape ()
    mu_g = p.mu_g[s]
    sigma_g = p.sigma_g[s]
    G = jax.vmap(peak_f, in_axes=(None, 0, 0))(U, mu_g, sigma_g).sum()
    c_rep = p.c_rep[s, species]
    R = (c_rep / 2 * (1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)

def fields_f(p: Params, points, species, x, s, map_size):
    """Calculate the fields U, G, R, and E based on parameters and points.
    x: shape (2,) array of coordinates
    points: shape (N, 2) array of particle positions"""
    diff = x - points  # shape (N, N, D)
    diff -= jp.round(diff / map_size) * map_size # periodic boundary
    r = jp.sqrt(jp.square(diff).sum(-1).clip(1e-10))  # shape (num_pairs,)
    mu_k = p.mu_k[s, species] # shape (num_points, num_kernels)
    sigma_k = p.sigma_k[s, species] # shape (num_points, num_kernels)
    u = jax.vmap(peak_f, in_axes=(None, 1, 1))(r, mu_k, sigma_k) # shape (num_kernel, num_points)
    w_k = p.w_k[s, species] # shape (num_points, num_kernel)
    U = (u * einops.rearrange(w_k, "p k -> k p")).sum() # shape ()
    mu_g = p.mu_g[s]
    sigma_g = p.sigma_g[s]
    G = jax.vmap(peak_f, in_axes=(None, 0, 0))(U, mu_g, sigma_g).sum()
    c_rep = p.c_rep[s, species]
    R = (c_rep / 2 * (1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)

def evo_fields_and_update_f(p: Params, particles, species, x, s, key, current_particle_idx):
    """Calculate the fields U, G, R, and E based on parameters and points.
    Also evolve species and parameters based on nearby particle interactions.
    x: shape (2,) array of coordinates
    particles: shape (N, 2) array of particle positions
    species: shape (N, num_species_dims) array of species parameters  
    s: shape (num_species_dims,) current particle's species embedding
    key: JAX random key for stochastic selection
    current_particle_idx: index of current particle in the arrays
    Returns: (Fields, (new_species, new_param_idx)) where new_param_idx is donor particle index"""
    r = jp.sqrt(jp.square(x - particles).sum(-1).clip(1e-10))  # shape (num_pairs,)
    
    # Compute weighted species parameters based on all particles
    mu_k = jp.dot(p.mu_k, species) # shape (num_particles, num_kernels)
    sigma_k = jp.dot(p.sigma_k, species) # shape (num_particles, num_kernels)
    u = jax.vmap(peak_f, in_axes=(None, 1, 1))(r, mu_k, sigma_k) # shape (num_kernel, num_particles)
    w_k = jp.dot(p.w_k, species) # shape (num_particles, num_kernel)
    U = (u * einops.rearrange(w_k, "p k -> k p")).sum() # shape ()
    mu_g = jp.dot(p.mu_g, species) # shape (num_growth_funcs,)
    sigma_g = jp.dot(p.sigma_g, species) # shape (num_growth_funcs,)
    G = jax.vmap(peak_f, in_axes=(None, 0, 0))(U, mu_g, sigma_g).sum()
    c_rep = jp.dot(p.c_rep, species) # shape (num_particles,)
    R = (c_rep / 2 * (1.0 - r).clip(0.0) ** 2).sum()

    # Evolution mechanism: probabilistic selection from interacting particles
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    
    # Use computed u values as interaction strength
    interaction_strength = u.sum(axis=0)  # shape (num_particles,) - sum over kernels
    
    # Parameter similarity in species space
    species_diff = jp.linalg.norm(species - s[None, :], axis=1)  # shape (num_particles,)
    similarity_weights = jp.exp(-species_diff / 1.0)  # more similar = higher weight
    
    # Combined weights: kernel interaction strength * parameter similarity
    interaction_weights = interaction_strength * similarity_weights
    
    # Add self-retention bias - current particle has higher probability to keep its parameters
    self_retention_weight = 3.0  # bias toward keeping own parameters
    interaction_weights = interaction_weights.at[current_particle_idx].multiply(self_retention_weight)
    
    # Normalize to create categorical distribution
    interaction_weights = interaction_weights / (interaction_weights.sum() + 1e-10)
    
    # Evolution probability - how often to actually evolve vs stay the same
    evolution_prob = 0.1
    should_evolve = jax.random.uniform(subkey1) < evolution_prob
    
    # Select donor particle based on interaction weights
    donor_idx = jax.random.categorical(subkey2, jp.log(interaction_weights + 1e-10))
    
    # Get new species from selected donor
    new_species = jax.lax.cond(
        should_evolve,
        lambda: species[donor_idx],
        lambda: s
    )
    
    # Add mutation noise occasionally
    noise_prob = 0.05
    should_mutate = jax.random.uniform(subkey3) < noise_prob
    noise_scale = 0.1
    
    noise = jax.random.normal(subkey3, s.shape) * noise_scale
    new_species = jax.lax.cond(
        should_mutate,
        lambda: new_species + noise,
        lambda: new_species
    )
    
    # Return donor index - the calling function can use this to update parameters
    new_param_idx = jax.lax.cond(
        should_evolve,
        lambda: donor_idx,
        lambda: current_particle_idx
    )
    
    return Fields(U, G, R, E=R - G), (new_species, new_param_idx)

def direct_motion_f(params, points, species, map_size):
    """Compute motion vector field using analytical gradients"""
    diff = points[:, None] - points  # shape (N, N, D)
    diff -= jp.round(diff / map_size) * map_size # periodic boundary
    r2 = jp.square(diff).sum(-1)  # shape (N, N)
    r = jp.sqrt(r2.clip(1e-10))  # shape (N, N)
    r_unit = diff / r[..., None]  # shape (N, N, D)
    
    # Compute U first
    mu_k = params.mu_k[species[:, None], species]  # shape (N, N, K)
    sigma_k = params.sigma_k[species[:, None], species]  # shape (N, N, K)
    w_k = params.w_k[species[:, None], species]  # shape (N, N, K)
    
    r_expanded = r[..., None]  # shape (N, N, 1)
    u_kernel = w_k * jp.exp(-jp.square((r_expanded - mu_k) / sigma_k))  # shape (N, N, K)
    U = u_kernel.sum((-1, -2))  # shape (N,)
    
    # Gradient of U wrt r (unchanged)
    dU_dr = -2 * (r_expanded - mu_k) / jp.square(sigma_k) * u_kernel # shape (N, N, K)
    grad_U = (dU_dr[..., None] * r_unit[..., None, :]).sum((-3, -2))  # shape (N, D)

    # Now compute gradients
    # First R gradient
    c_rep = params.c_rep[species[:, None], species]  # shape (N, N)
    mask = (r < 1.0)
    dR = c_rep * (1.0 - r) * mask
    grad_R = (dR[..., None] * r_unit).sum(1)  # sum over j
    
    # G gradient - need to handle S and G dimensions
    mu_g = params.mu_g[species]  # shape (N, G)
    sigma_g = params.sigma_g[species]  # shape (N, G)
    
    # Expand U for broadcasting with mu_g and sigma_g
    U_expanded = U[:, None]  # shape (N, 1)
    
    # Calculate dG_dU accounting for G dimensions
    dG_dU = 2 * (U_expanded - mu_g) / jp.square(sigma_g) * \
            jp.exp(-jp.square((U_expanded - mu_g) / sigma_g))  # shape (N, G)
    
    # Sum over G dimensions to get shape (N,)
    dG_dU = dG_dU.sum(1)  # shape (N,)
    
    # Now dG_dU is shape (N,), so expand for multiplication with grad_U
    grad_G = dG_dU[:, None] * grad_U  # shape (N, D)
    
    return grad_R - grad_G

def simple_fields_f(p: Params, points, x):
    """Calculate the fields U, G, R, and E based on parameters and points."""
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))
    U = peak_f(r, p.mu_k, p.sigma_k).sum() * p.w_k
    G = peak_f(U, p.mu_g, p.sigma_g)
    R = p.c_rep / 2 * ((1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)

def compute_dist_matrix(points):
    """Compute upper triangle of distance matrix efficiently"""
    n = points.shape[0]
    # Get indices for upper triangle (excluding diagonal)
    i, j = jp.triu_indices(n, k=1)
    
    # Compute distances only for upper triangle
    diff = points[i] - points[j]  # shape (num_pairs, 2)
    r = jp.sqrt(jp.square(diff).sum(-1).clip(1e-10))  # shape (num_pairs,)
    
    # Create full matrix using symmetry
    dist_matrix = jp.zeros((n, n))
    dist_matrix = dist_matrix.at[i, j].set(r)
    dist_matrix = dist_matrix.at[j, i].set(r)  # mirror across diagonal
    
    return dist_matrix

def all_fields_f(params, points, species):
    """Compute the fields U, G, R, and E for all points."""
    dist_matrix = compute_dist_matrix(points)
    return jax.vmap(functools.partial(fields_f, params, species))(species, dist_matrix)

def motion_f(params, points, species):
    """Compute the motion vector field as the negative gradient of the energy."""

    grad_E = jax.grad(lambda p, s: fields_f(params, points, species, p, s).E)
    return -jax.vmap(grad_E)(points, species)

def motion_and_energy_f(params, points, species, map_size):
    """Compute the motion vector field as the negative gradient of the energy."""

    grad_E = jax.value_and_grad(lambda p, s: fields_f(params, points, species, p, s, map_size).E)
    energy, grad = jax.vmap(grad_E)(points, species)
    return -grad, energy

def step_and_energy_f(params, points, species, dt, map_size):
    """Perform a step and return the energy"""
    f, e = motion_and_energy_f(params, points, species)
    points += dt * f
    points -= jp.floor(points/map_size) * map_size  # periodic boundary
    return points, e

def step_f(params, points, species, dt, map_size):
    """Perform a single Euler integration step."""
    points += dt * direct_motion_f(params, points, species, map_size)
    points -= jp.floor(points/map_size) * map_size  # periodic boundary
    return points

def evo_step_f(params, particles, species, dt, map_size, key):
    """Perform a single Euler integration step with evolution updates."""
    motion, (new_species_array, new_param_indices) = evo_motion_f(params, particles, species, map_size, key)
    
    # Update particle positions
    particles += dt * motion
    particles -= jp.floor(particles/map_size) * map_size  # periodic boundary
    
    # Update species
    species = new_species_array
    
    # Update parameters by copying from donor particles
    new_params = Params(
        mu_k=params.mu_k[new_param_indices],
        sigma_k=params.sigma_k[new_param_indices], 
        w_k=params.w_k[new_param_indices],
        mu_g=params.mu_g[new_param_indices],
        sigma_g=params.sigma_g[new_param_indices],
        c_rep=params.c_rep[new_param_indices]
    )
    
    return new_params, particles, species

def evo_motion_f(params, particles, species, map_size, key):
    """Compute the motion vector field as the negative gradient of the energy.
    Also returns evolution updates for species and parameters."""
    # Generate keys for each particle
    keys = jax.random.split(key, len(particles))
    particle_indices = jp.arange(len(particles))
    
    # Define function that returns energy and evolution updates
    def energy_and_evolution(x, s, k, idx):
        fields, (new_species, new_param_idx) = evo_fields_and_update_f(params, particles, species, x, s, k, idx)
        return fields.E, (new_species, new_param_idx)
    
    # Get gradients and evolution updates for each particle
    grad_fn = jax.value_and_grad(energy_and_evolution, argnums=0, has_aux=True)
    
    # Apply to all particles
    (energies, evolution_updates), gradients = jax.vmap(grad_fn)(particles, species, keys, particle_indices)
    
    return -gradients, evolution_updates

def simple_motion_f(params, points):
    """Compute the motion vector field as the negative gradient of the energy."""
    grad_E = jax.grad(lambda x: simple_fields_f(params, points, x).E)
    return -jax.vmap(grad_E)(points)

def simple_step_f(params, points, dt):
    """Perform a single Euler integration step."""
    # max_f = 20
    # return points + dt * jp.clip(motion_f(params, points, species), -max_f, max_f)
    return points + dt * simple_motion_f(params, points)

def point_fields_f(params, points, species):
    return jax.vmap(functools.partial(fields_f, params, points, species))(points, species)

def total_energy_f(params, points, species):
    return point_fields_f(params, points, species).E.sum()



def draw_multi_species_particles(trajectory, map_size, species=None, num_species=None, start=-16000, offset=2000, bloom=False):
    # Create a color map for different species
    
    # Generate distinct colors for each species
    angles = jp.linspace(0, 2 * jp.pi, num_species, endpoint=False)
    colors = jp.stack([
        jp.sin(angles) * 0.5 + 0.5,
        jp.sin(angles + 2 * jp.pi / 3) * 0.5 + 0.5,
        jp.sin(angles + 4 * jp.pi / 3) * 0.5 + 0.5
    ], axis=1)

    # Pre-compute particle colors based on species
    particle_colors = colors[species]
    return _draw_particles(trajectory, map_size, particle_colors, start=start, offset=offset, bloom=bloom)

def draw_particles(trajectory, map_size, start=-16000, offset=2000, bloom=False):
    particle_colors = jp.ones((trajectory.shape[1], 3))
    return _draw_particles(trajectory, map_size, particle_colors, start=start, offset=offset, bloom=bloom)

def _draw_particles(trajectory, map_size, particle_colors, start=-16000, offset=2000, bloom=False):
    """
    Create an animation of particle trajectories using JAX and imageio.
    Optimized to minimize loops and leverage JAX's parallel processing.
    
    Args:
        trajectory: JAX array of shape (num_frames, num_particles, num_dims)
                   containing particle positions at each time step
    """
    
    # Subsample frames - keep 1 in 20
    subsampled_trajectory = trajectory[start::offset]
    num_frames = subsampled_trajectory.shape[0]
    num_particles = subsampled_trajectory.shape[1]
    
    # Define image dimensions and particle rendering parameters
    img_size = 800
    particle_radius = 3
    radius_sq = particle_radius ** 2

    bloom_radius = particle_radius * 4
    bloom_radius_sq = bloom_radius ** 2

    @jax.jit
    def hit_color_func(dists_sq):
        return particle_colors[jp.argmin(dists_sq)]
            
    @jax.jit
    def bloom_color_func(dists_sq):
        min_idx = jp.argmin(dists_sq)
        min_dist_sq = dists_sq[min_idx]
        bloom_color = particle_colors[min_idx]
        return bloom_color * (1 - jp.clip(min_dist_sq / bloom_radius_sq, 0, 1)) ** 3

    @jax.jit
    def no_hit_color_func(dists_sq):
        return jp.zeros(3)
    
    # Alternative approach: use a splatting technique
    @jax.jit
    def render_frame(positions):
        """
        Render particles using a splatting technique.
        
        This creates an alpha mask for each particle and composites them onto the image.
        """
        # Scale positions to image coordinates
        scaled_pos = (positions / map_size) * img_size
        scaled_pos = jp.clip(scaled_pos, 0, img_size - 1)
        
        # Create a grid of coordinates
        x_indices = jp.arange(img_size)
        y_indices = jp.arange(img_size)
        X, Y = jp.meshgrid(x_indices, y_indices)
        coords = jp.stack([X, Y], axis=-1)  # Shape: (img_size, img_size, 2)
        
        # for each pixel, compute color
        def process_pixel(coord):
            dists_sq = jp.sum((coord - scaled_pos) ** 2, axis=-1)
            min_dist = jp.min(dists_sq)

            # bloom_pred = jp.bitwise_and((min_dist <= bloom_radius_sq), bloom)
            bloom_pred = min_dist <= bloom_radius_sq
            bloom_func = lambda dists_sq: jax.lax.cond(bloom_pred, bloom_color_func, no_hit_color_func, dists_sq)
            return jax.lax.cond(min_dist <= radius_sq, hit_color_func, bloom_func, dists_sq)

        image_fn = jax.vmap(jax.vmap(process_pixel, in_axes=0), in_axes=1)
        image = image_fn(coords)
        
        return (image * 255).astype(jp.uint8)
    
    frames = jax.vmap(render_frame)(subsampled_trajectory)
    return frames

@functools.partial(jax.jit, static_argnames=['map_size', 'num_species', 'start', 'offset'])
def draw_multi_species_particles(trajectory, map_size, species=None, num_species=None, 
                                   start=-16000, offset=2000):
    """
    Wrapper to create three orthogonal view animations with colored species for 3D particles.
    """
    # Create a color map for different species
    angles = jp.linspace(0, 2 * jp.pi, num_species, endpoint=False)
    colors = jp.stack([
        jp.sin(angles) * 0.5 + 0.5,
        jp.sin(angles + 2 * jp.pi / 3) * 0.5 + 0.5,
        jp.sin(angles + 4 * jp.pi / 3) * 0.5 + 0.5
    ], axis=1)
    
    # Pre-compute particle colors based on species
    particle_colors = colors[species]
    
    return _draw_particles(trajectory, map_size, particle_colors, 
                                  start=start, offset=offset)


@functools.partial(jax.jit, static_argnames=('map_size', 'img_size', 'particle_radius', 'start', 'offset'))
def draw_particles_3d_views(trajectory, map_size, particle_colors, img_size=800, 
                            particle_radius=5, start=-16000, offset=2000):
    """
    Create three animations of 3D particle trajectories from orthogonal viewing angles.
    
    Args:
        trajectory: JAX array of shape (num_frames, num_particles, 3)
                   containing 3D particle positions at each time step
        map_size: Size of the simulation space (assumed cubic)
        particle_colors: Colors for each particle of shape (num_particles, 3)
        img_size: Size of the output images (square)
        particle_radius: Radius of particles in pixels
        start: Starting frame index
        offset: Frame sampling interval
        
    Returns:
        Tuple of three video arrays (xy_view, xz_view, yz_view)
    """
    # Subsample frames
    subsampled_trajectory = trajectory[start::offset]
    num_frames = subsampled_trajectory.shape[0]
    num_particles = subsampled_trajectory.shape[1]
    
    # Create three trajectory sets for different orthogonal projections
    # XY projection (top view)
    xy_view_traj = subsampled_trajectory  # Take just X and Y coordinates
    
    # XZ projection (side view)
    xz_view_traj = jp.stack([
        subsampled_trajectory[:, :, 0],  # X coordinate
        subsampled_trajectory[:, :, 2],  # Z coordinate
        subsampled_trajectory[:, :, 1]
    ], axis=-1)
    
    # YZ projection (front view)
    yz_view_traj = jp.stack([
        subsampled_trajectory[:, :, 1],  # Y coordinate
        subsampled_trajectory[:, :, 2],  # Z coordinate
        subsampled_trajectory[:, :, 0]
    ], axis=-1)
    
    # Define the rendering function
    @jax.jit
    def render_frame(positions):
        """
        Render particles using a splatting technique.
        """
        # Scale positions to image coordinates
        positions, depth = positions[..., :2], positions[..., 2]
        idx_order = jp.argsort(depth, descending=True)
        positions, depth = positions[idx_order, :], depth[idx_order]
        scaled_pos = (positions / map_size) * img_size
        scaled_pos = jp.clip(scaled_pos, 0, img_size - 1)
        
        # Start with a black background
        image = jp.zeros((img_size, img_size, 3))
        
        # Create a grid of coordinates
        x_indices = jp.arange(img_size)
        y_indices = jp.arange(img_size)
        X, Y = jp.meshgrid(x_indices, y_indices)
        coords = jp.stack([X, Y], axis=-1)  # Shape: (img_size, img_size, 2)
        
        # For each particle, compute influence on each pixel
        def process_particle(image, particle_idx):
            pos = scaled_pos[particle_idx]
            this_depth = depth[particle_idx]
            color = particle_colors[particle_idx]
            
            # Calculate distance from each pixel to the particle
            dist_squared = jp.sum((coords - pos)**2, axis=-1)
            this_particle_radius = particle_radius * (1 - 0.5 * this_depth / map_size)

            # Create a mask for pixels affected by this particle
            mask = dist_squared <= (this_particle_radius**2)
            mask = mask[:, :, jp.newaxis]  # Add channel dimension

            this_color = (1 - 0.5 * this_depth / map_size) * color + (0.5 * this_depth / map_size) * jp.zeros((3,))
            
            # Update image: where mask is True, use particle color
            return jp.where(mask, this_color, image)
        
        # Scan through all particles
        image, _ = jax.lax.scan(
            lambda img, idx: (process_particle(img, idx), None),
            image,
            jp.arange(num_particles)
        )
        
        return (image * 255).astype(jp.uint8)
    
    # Render all frames for each view using vmap
    xy_frames = jax.vmap(render_frame)(xy_view_traj)
    xz_frames = jax.vmap(render_frame)(xz_view_traj)
    yz_frames = jax.vmap(render_frame)(yz_view_traj)
    
    return jp.stack([xy_frames, xz_frames, yz_frames])

def draw_multi_species_particles_3d(trajectory, map_size, species=None, num_species=None, 
                                   start=-16000, offset=2000):
    """
    Wrapper to create three orthogonal view animations with colored species for 3D particles.
    """
    # Create a color map for different species
    angles = jp.linspace(0, 2 * jp.pi, num_species, endpoint=False)
    colors = jp.stack([
        jp.sin(angles) * 0.5 + 0.5,
        jp.sin(angles + 2 * jp.pi / 3) * 0.5 + 0.5,
        jp.sin(angles + 4 * jp.pi / 3) * 0.5 + 0.5
    ], axis=1)
    
    # Pre-compute particle colors based on species
    particle_colors = colors[species]
    
    return draw_particles_3d_views(trajectory, map_size, particle_colors, 
                                  start=start, offset=offset)

def draw_particles_3d(trajectory, map_size, start=-16000, offset=2000):
    """
    Wrapper to create three orthogonal view animations with default particle colors for 3D particles.
    """
    particle_colors = jp.zeros((trajectory.shape[1], 3))
    return draw_particles_3d_views(trajectory, map_size, particle_colors, 
                                  start=start, offset=offset)

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt'))
def multi_step_scan(params, initial_points, species, dt, num_steps, map_size):
    def scan_step(carry, _):
        points = carry
        points = step_f(params, points, species, dt, map_size)
        return points, points

    final_points, trajectory = jax.lax.scan(
        scan_step,
        initial_points,
        xs=None,
        length=num_steps
    )

    return final_points, trajectory

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt'))
def evo_multi_step_scan(params, initial_particles, species, dt, num_steps, map_size, key):
    def scan_step(carry, step_key):
        params_t, particles, species_t = carry
        new_params, new_particles, new_species = evo_step_f(params_t, particles, species_t, dt, map_size, step_key)
        return (new_params, new_particles, new_species), new_particles

    # Generate keys for each step
    step_keys = jax.random.split(key, num_steps)
    
    (final_params, final_particles, final_species), trajectory = jax.lax.scan(
        scan_step,
        (params, initial_particles, species),
        step_keys
    )

    return (final_params, final_particles, final_species), trajectory

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt', 'map_size'))
def multi_step_scan_with_force(params, initial_points, species, dt, num_steps, map_size):
    def scan_step(carry, _):
        points = carry
        f = direct_motion_f(params, points, species, map_size)
        points += dt * f
        points -= jp.floor(points/map_size) * map_size  # periodic boundary
        return points, (points, f)

    final_points, (trajectory, force) = jax.lax.scan(
        scan_step,
        initial_points,
        xs=None,
        length=num_steps
    )

    return final_points, (trajectory, force)

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt', 'map_size'))
def multi_step_scan_with_force_and_energy(params, initial_points, species, dt, num_steps, map_size):
    def scan_step(carry, _):
        points = carry
        f, e = motion_and_energy_f(params, points, species, map_size)
        points += dt * f
        points -= jp.floor(points/map_size) * map_size  # periodic boundary
        return points, (points, f, e)

    final_points, (trajectory, force, energy) = jax.lax.scan(
        scan_step,
        initial_points,
        xs=None,
        length=num_steps
    )

    return final_points, (trajectory, force, energy)
