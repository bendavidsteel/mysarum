import base64
import functools
import hashlib
import json
import itertools
import os

import jax
import jax.numpy as jnp
import tqdm

from particle_lenia import Params, fields_f, step_f

def make_hash_sha256(o):
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b64encode(hasher.digest()).decode()

def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o

def create_energy_field(points, params, grid_size=48):
    x = jnp.linspace(-grid_size/2, grid_size/2, grid_size)
    y = jnp.linspace(-grid_size/2, grid_size/2, grid_size)
    XX, YY = jnp.meshgrid(x, y)
    grid_points = jnp.stack([XX, YY], axis=-1)
    energy_field = jax.vmap(lambda p: fields_f(params, points, p).E)(grid_points.reshape(-1, 2)).reshape(grid_size, grid_size)
    return energy_field[None, None, :, :] # Add batch and channel dimensions

@jax.jit
def compute_system_energy(points, params):
    return jnp.sum(jax.vmap(lambda p: fields_f(params, points, p).E)(points))

def is_system_stable(energy_history, threshold=0.01):
    if len(energy_history) < 10:
        return False
    recent_energies = energy_history[-10:]
    energy_change = (max(recent_energies) - min(recent_energies)) / jnp.mean(recent_energies)
    return energy_change < threshold

def scan_step(dt, params, species, points, _):
    points = step_f(params, points, species, dt)
    return points, points

def generate_lenia_data(key, data_dir_path, param_ranges, param_lists, max_steps=1000, num_runs_per_point=100):
    # Create parameter grid
    param_grid = list(itertools.product(*[param_list for _, param_list in param_lists]))
    param_names = [param_name for param_name, _ in param_lists]
    
    pbar = tqdm.tqdm(total=len(param_grid) * num_runs_per_point)
    for i, param_values in enumerate(param_grid):
        get_param_val = lambda p_name: [param_value for param_name, param_value in zip(param_names, param_values) if param_name == p_name][0]
        num_particles = get_param_val('num_particles')
        size = get_param_val('size')
        dt = get_param_val('dt')
        num_species = get_param_val('num_species')
        num_kernels = get_param_val('num_kernels')
        num_growth_funcs = get_param_val('num_growth_funcs')
        num_dims = get_param_val('num_dims')
        for _ in range(num_runs_per_point):
            pbar.update(1)
            key, *subkeys = jax.random.split(key, num=7)
            mu_k = jax.random.uniform(subkeys[0], (num_species, num_species, num_kernels), minval=param_ranges['mu_k'][0], maxval=param_ranges['mu_k'][1])
            sigma_k = jax.random.uniform(subkeys[1], (num_species, num_species, num_kernels), minval=param_ranges['sigma_k'][0], maxval=param_ranges['sigma_k'][1])

            mu_g = jax.random.uniform(subkeys[2], (num_species, num_species, num_growth_funcs), minval=param_ranges['mu_g'][0], maxval=param_ranges['mu_g'][1])
            sigma_g = jax.random.uniform(subkeys[3], (num_species, num_species, num_growth_funcs), minval=param_ranges['sigma_g'][0], maxval=param_ranges['sigma_g'][1])

            w_k = jax.random.uniform(subkeys[4], (num_species, num_kernels), minval=param_ranges['w_k'][0], maxval=param_ranges['w_k'][1])
            c_rep = jax.random.uniform(subkeys[5], (num_species, num_species), minval=param_ranges['c_rep'][0], maxval=param_ranges['c_rep'][1])
            params = Params(mu_k=mu_k, sigma_k=sigma_k, w_k=w_k, mu_g=mu_g, sigma_g=sigma_g, c_rep=c_rep)
            
            json_params = {k: v.tolist() for k, v in params._asdict().items()}
            run_params = {
                'num_particles': num_particles.item(),
                'size': size.item(),
                'dt': dt.item(),
                'num_species': num_species.item(),
                'num_kernels': num_kernels.item(),
                'num_growth_funcs': num_growth_funcs.item(),
                'params': json_params
            }

            run_hash = make_hash_sha256(run_params)
            run_path = os.path.join(data_dir_path, f"run_{run_hash}")
            if os.path.exists(run_path):
                continue

            points = (jax.random.uniform(key, (num_particles, num_dims)) - 0.5) * size
            species = jax.random.randint(key, (num_particles,), 0, num_species)
            
            points, points_history = jax.lax.scan(functools.partial(scan_step, dt, params, species), points, None, length=max_steps)

            # save energy fields
            os.makedirs(run_path, exist_ok=True)
            jnp.save(os.path.join(run_path, 'points_history.npy'), jnp.array(points_history))
            
            with open(os.path.join(run_path, 'params.json'), 'w') as f:
                json.dump(run_params, f)

def main():
    # Define parameter ranges for grid search
    param_ranges = {
        'mu_k': jnp.array([-10.0, 10.0]),
        'sigma_k': jnp.array([-10.0, 10.0]),
        'w_k': jnp.array([-0.1, 0.1]),
        'mu_g': jnp.array([-1.0, 1.0]),
        'sigma_g': jnp.array([-1.0, 1.0]),
        'c_rep': jnp.array([-10.0, 10.0])
    }
    
    param_lists = [
        ('num_particles', jnp.array([100, 1000])),
        ('num_species', jnp.array([1, 2, 4, 8])),
        ('num_kernels', jnp.array([1, 2, 4, 8])),
        ('num_growth_funcs', jnp.array([1, 2, 4, 8])),
        ('size', jnp.array([20, 100])),
        ('dt', jnp.array([0.1, 0.01, 0.001])),
        ('num_dims', jnp.array([3, 2]))
    ]
    key = jax.random.PRNGKey(8)
    
    generate_lenia_data(key, 'lenia_data', param_ranges, param_lists)

if __name__ == '__main__':
    main()