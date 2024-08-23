from collections import namedtuple

import einops
import jax
import jax.numpy as jp

# Define namedtuples for parameters and fields
Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')

@jax.jit
def peak_f(x, mu, sigma):
    """Compute the Gaussian peak function."""
    return jp.exp(-((x - mu) / sigma) ** 2)

@jax.jit
def fields_f(p: Params, points, species, x, s):
    """Calculate the fields U, G, R, and E based on parameters and points."""
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))
    mu_k = p.mu_k[s, species]
    sigma_k = p.sigma_k[s, species]
    u = jax.vmap(peak_f, in_axes=(None, 1, 1))(r, mu_k, sigma_k)
    w_k = p.w_k[s]
    U = (u * einops.rearrange(w_k, "s -> s 1")).sum()
    mu_g = p.mu_g[s, species]
    sigma_g = p.sigma_g[s, species]
    G = jax.vmap(peak_f, in_axes=(None, 1, 1))(U, mu_g, sigma_g).sum()
    c_rep = p.c_rep[s, species]
    R = (c_rep / 2 * (1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)

def motion_f(params, points, species):
    """Compute the motion vector field as the negative gradient of the energy."""
    grad_E = jax.grad(lambda x, s: fields_f(params, points, species, x, s).E)
    return -jax.vmap(grad_E)(points, species)

@jax.jit
def step_f(params, points, species, dt):
    """Perform a single Euler integration step."""
    max_f = 20
    return points + dt * jp.clip(motion_f(params, points, species), -max_f, max_f)
