import collections
import functools

import einops
import jax
import jax.numpy as jp

# Define namedtuples for parameters and fields
Params = collections.namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep map_size')
Fields = collections.namedtuple('Fields', 'U G R E')

def peak_f(x, mu, sigma):
    """Compute the Gaussian peak function."""
    return jp.exp(-((x - mu) / sigma) ** 2)

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

def fields_f(p: Params, points, species, x, s):
    """Calculate the fields U, G, R, and E based on parameters and points.
    x: shape (2,) array of coordinates
    points: shape (N, 2) array of particle positions"""
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))  # shape (num_pairs,)
    mu_k = p.mu_k[s, species] # shape (num_points, num_kernels)
    sigma_k = p.sigma_k[s, species] # shape (num_points, num_kernels)
    u = jax.vmap(peak_f, in_axes=(None, 1, 1))(r, mu_k, sigma_k) # shape (num_kernel, num_points)
    w_k = p.w_k[s, species] # shape (num_kernel,)
    U = (u * einops.rearrange(w_k, "s -> s 1")).sum() # shape ()
    mu_g = p.mu_g[s]
    sigma_g = p.sigma_g[s]
    G = jax.vmap(peak_f, in_axes=(None, 0, 0))(U, mu_g, sigma_g).sum()
    c_rep = p.c_rep[s, species]
    R = (c_rep / 2 * (1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)

def soft_fields_f(p: Params, points, species, x, s):
    """Calculate the fields U, G, R, and E based on parameters and points.
    x: shape (2,) array of coordinates
    points: shape (N, 2) array of particle positions"""
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))  # shape (num_pairs,)
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


def direct_U_f(params: Params, points, species):
    diff = points[:, None] - points  # shape (N, N, 2)
    r2 = jp.square(diff).sum(-1)  # shape (N, N)
    r = jp.sqrt(r2.clip(1e-10))  # shape (N, N)
    r_unit = diff / r[..., None]  # shape (N, N, 2)
    
    # Compute U first
    mu_k = params.mu_k[species[:, None], species]  # shape (N, N, K)
    sigma_k = params.sigma_k[species[:, None], species]  # shape (N, N, K)
    w_k = params.w_k[species]  # shape (N, K)
    
    r_expanded = r[..., None]  # shape (N, N, 1)
    u_kernel = jp.exp(-jp.square((r_expanded - mu_k) / sigma_k))  # shape (N, N, K)
    U = (u_kernel * w_k[:, None, :]).sum((-1, -2))  # shape (N,)
    return U

def direct_grad_G_f(params, points, species):
    """Compute motion vector field using analytical gradients"""
    diff = points[:, None] - points  # shape (N, N, 2)
    r2 = jp.square(diff).sum(-1)  # shape (N, N)
    r = jp.sqrt(r2.clip(1e-10))  # shape (N, N)
    r_unit = diff / r[..., None]  # shape (N, N, 2)
    
    # Compute U first
    mu_k = params.mu_k[species[:, None], species]  # shape (N, N, K)
    sigma_k = params.sigma_k[species[:, None], species]  # shape (N, N, K)
    w_k = params.w_k[species]  # shape (N, K)
    
    r_expanded = r[..., None]  # shape (N, N, 1)
    u_kernel = jp.exp(-jp.square((r_expanded - mu_k) / sigma_k))  # shape (N, N, K)
    U = (u_kernel * w_k[:, None, :]).sum((-1, -2))  # shape (N,)
    
    # G gradient - need to handle S and G dimensions
    mu_g = params.mu_g[species]  # shape (N, S, G)
    sigma_g = params.sigma_g[species]  # shape (N, S, G)
    
    # Expand U for broadcasting with mu_g and sigma_g
    U_expanded = U[:, None, None]  # shape (N, 1, 1)
    
    # Calculate dG_dU accounting for S and G dimensions
    dG_dU = 2 * (U_expanded - mu_g) / jp.square(sigma_g) * \
            jp.exp(-jp.square((U_expanded - mu_g) / sigma_g))  # shape (N, S, G)
    
    # Sum over S and G dimensions to get shape (N,)
    dG_dU = dG_dU.sum((1, 2))  # shape (N,)
    
    # Gradient of U wrt r (unchanged)
    dU_dr = w_k[:, None, :] * -2 * (r_expanded - mu_k) / jp.square(sigma_k) * u_kernel
    grad_U = (dU_dr * r_unit).sum((-2))  # shape (N, 2)
    
    # Now dG_dU is shape (N,), so expand for multiplication with grad_U
    grad_G = dG_dU[:, None] * grad_U  # shape (N, 2)
    
    return -grad_G

def direct_grad_R_f(params, points, species):
    """Compute motion vector field using analytical gradients"""
    diff = points[:, None] - points  # shape (N, N, 2)
    r2 = jp.square(diff).sum(-1)  # shape (N, N)
    r = jp.sqrt(r2.clip(1e-10))  # shape (N, N)
    r_unit = diff / r[..., None]  # shape (N, N, 2)
    
    # Now compute gradients
    # First R gradient
    c_rep = params.c_rep[species[:, None], species]  # shape (N, N)
    mask = (r < 1.0)
    dR = c_rep * (1.0 - r) * mask
    grad_R = (dR[..., None] * r_unit).sum(1)  # sum over j
    
    return -grad_R

def direct_grad_U_f(params, points, species):
    """Compute motion vector field using analytical gradients"""
    diff = points[:, None] - points  # shape (N, N, 2)
    r2 = jp.square(diff).sum(-1)  # shape (N, N)
    r = jp.sqrt(r2.clip(1e-10))  # shape (N, N)
    r_unit = diff / r[..., None]  # shape (N, N, 2)
    
    # Compute U first
    mu_k = params.mu_k[species[:, None], species]  # shape (N, N, K)
    sigma_k = params.sigma_k[species[:, None], species]  # shape (N, N, K)
    w_k = params.w_k[species]  # shape (N, K)
    
    r_expanded = r[..., None]  # shape (N, N, 1)
    u_kernel = jp.exp(-jp.square((r_expanded - mu_k) / sigma_k))  # shape (N, N, K)
    
    # Gradient of U wrt r (unchanged)
    dU_dr = w_k[:, None, :] * -2 * (r_expanded - mu_k) / jp.square(sigma_k) * u_kernel # shape (N, N, K)
    grad_U = (dU_dr * r_unit).sum((-2))  # shape (N, 2)
    
    return grad_U

def direct_motion_f(params, points, species):
    """Compute motion vector field using analytical gradients"""
    diff = points[:, None] - points  # shape (N, N, D)
    map_size = params.map_size
    diff -= jp.round(diff / map_size) * map_size # periodic boundary
    r2 = jp.square(diff).sum(-1)  # shape (N, N)
    r = jp.sqrt(r2.clip(1e-10))  # shape (N, N)
    r_unit = diff / r[..., None]  # shape (N, N, D)
    
    # Compute U first
    mu_k = params.mu_k[species[:, None], species]  # shape (N, N, K)
    sigma_k = params.sigma_k[species[:, None], species]  # shape (N, N, K)
    w_k = params.w_k[species[:, None], species]  # shape (N, N, K)
    
    r_expanded = r[..., None]  # shape (N, N, 1)
    u_kernel = jp.exp(-jp.square((r_expanded - mu_k) / sigma_k))  # shape (N, N, K)
    U = (u_kernel * w_k).sum((-1, -2))  # shape (N,)
    
    # Now compute gradients
    # First R gradient
    c_rep = params.c_rep[species[:, None], species]  # shape (N, N)
    mask = (r < 1.0)
    dR = c_rep * (1.0 - r) * mask
    grad_R = (dR[..., None] * r_unit).sum(1)  # sum over j
    
    # G gradient - need to handle S and G dimensions
    mu_g = params.mu_g[species]  # shape (N, S, G)
    sigma_g = params.sigma_g[species]  # shape (N, S, G)
    
    # Expand U for broadcasting with mu_g and sigma_g
    U_expanded = U[:, None, None]  # shape (N, 1, 1)
    
    # Calculate dG_dU accounting for S and G dimensions
    dG_dU = 2 * (U_expanded - mu_g) / jp.square(sigma_g) * \
            jp.exp(-jp.square((U_expanded - mu_g) / sigma_g))  # shape (N, S, G)
    
    # Sum over S and G dimensions to get shape (N,)
    dG_dU = dG_dU.sum((1, 2))  # shape (N,)
    
    # Gradient of U wrt r (unchanged)
    dU_dr = w_k * -2 * (r_expanded - mu_k) / jp.square(sigma_k) * u_kernel # shape (N, N, K)
    grad_U = (dU_dr[..., None] * r_unit[..., None, :]).sum((-3, -2))  # shape (N, D)
    # grad_U = (dU_dr * r_unit).sum((-2))  # shape (N, 2)
    
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

def soft_motion_f(params, points, species):
    """Compute the motion vector field as the negative gradient of the energy."""

    grad_E = jax.grad(lambda p, s: soft_fields_f(params, points, species, p, s).E)
    return -jax.vmap(grad_E)(points, species)

def step_f(params, points, species, dt):
    """Perform a single Euler integration step."""
    points += dt * direct_motion_f(params, points, species)
    points -= jp.floor(points/params.map_size) * params.map_size  # periodic boundary
    return points

def soft_step_f(params, points, species, dt):
    """Perform a single Euler integration step."""
    return points + dt * soft_motion_f(params, points, species)


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



def draw_multi_species_particles(trajectory, map_size, species=None, num_species=None, start=-16000, offset=2000):
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
    return _draw_particles(trajectory, map_size, particle_colors, start=start, offset=offset)

def draw_particles(trajectory, map_size, start=-16000, offset=2000):
    particle_colors = jp.ones((trajectory.shape[1], 3))
    return _draw_particles(trajectory, map_size, particle_colors, start=start, offset=offset)

def _draw_particles(trajectory, map_size, particle_colors, start=-16000, offset=2000):
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
            color = particle_colors[particle_idx]
            
            # Calculate distance from each pixel to the particle
            dist_squared = jp.sum((coords - pos)**2, axis=-1)
            
            # Create a mask for pixels affected by this particle
            mask = dist_squared <= (particle_radius**2)
            mask = mask[:, :, jp.newaxis]  # Add channel dimension
            
            # Update image: where mask is True, use particle color
            return jp.where(mask, color, image)
        
        # Scan through all particles
        image, _ = jax.lax.scan(
            lambda img, idx: (process_particle(img, idx), None),
            image,
            jp.arange(num_particles)
        )
        
        return (image * 255).astype(jp.uint8)
    
    frames = jax.vmap(render_frame)(subsampled_trajectory)
    return frames

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
def multi_step_scan(params, initial_points, species, dt, num_steps):
    def scan_step(carry, _):
        points = carry
        points = step_f(params, points, species, dt)
        return points, points

    final_points, trajectory = jax.lax.scan(
        scan_step,
        initial_points,
        xs=None,
        length=num_steps
    )

    return final_points, trajectory

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt'))
def multi_step_scan_with_force(params, initial_points, species, dt, num_steps):
    def scan_step(carry, _):
        points = carry
        f = direct_motion_f(params, points, species)
        points += dt * f
        points -= jp.floor(points/params.map_size) * params.map_size  # periodic boundary
        return points, (points, f)

    final_points, (trajectory, force) = jax.lax.scan(
        scan_step,
        initial_points,
        xs=None,
        length=num_steps
    )

    return final_points, (trajectory, force)
