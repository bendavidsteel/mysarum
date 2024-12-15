import collections
import functools

import einops
import jax
import jax.numpy as jp

# Define namedtuples for parameters and fields
Params = collections.namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = collections.namedtuple('Fields', 'U G R E')

def peak_f(x, mu, sigma):
    """Compute the Gaussian peak function."""
    return jp.exp(-((x - mu) / sigma) ** 2)

def fields_f(p: Params, points, species, x, s):
    """Calculate the fields U, G, R, and E based on parameters and points.
    x: shape (2,) array of coordinates
    points: shape (N, 2) array of particle positions"""
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10)) # shape (num_points,)
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

def simple_fields_f(p: Params, points, x):
    """Calculate the fields U, G, R, and E based on parameters and points."""
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))
    U = peak_f(r, p.mu_k, p.sigma_k).sum() * p.w_k
    G = peak_f(U, p.mu_g, p.sigma_g)
    R = p.c_rep / 2 * ((1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)

def motion_f(params, points, species):
    """Compute the motion vector field as the negative gradient of the energy."""
    grad_E = jax.grad(lambda x, s: fields_f(params, points, species, x, s).E)
    return -jax.vmap(grad_E)(points, species)

def step_f(params, points, species, dt):
    """Perform a single Euler integration step."""
    # max_f = 20
    # return points + dt * jp.clip(motion_f(params, points, species), -max_f, max_f)
    return points + dt * motion_f(params, points, species)

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
