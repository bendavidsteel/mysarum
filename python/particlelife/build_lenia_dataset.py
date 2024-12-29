import base64
import functools
import hashlib
import json
import itertools
import os

import jax
import jax.numpy as jnp
import tqdm

from particle_lenia import Params, fields_f, step_f, total_energy_f

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
def compute_system_energy(points, species, params):
    return total_energy_f(params, points, species)

def is_system_stable(energy_history, threshold=0.01):
    if len(energy_history) < 10:
        return False
    recent_energies = energy_history[-10:]
    energy_change = (max(recent_energies) - min(recent_energies)) / jnp.mean(recent_energies)
    return energy_change < threshold

def scan_step(dt, params, species, state, _):
    points = state['points']
    new_points = step_f(params, points, species, dt)
    energy = compute_system_energy(new_points, species, params)
    return {'points': new_points, 'energy': energy}, {'points': new_points, 'energy': energy}

def check_steady_state(energy_history, min_steps_without_new_min=100):
    if len(energy_history) < min_steps_without_new_min + 1:
        return False
    
    recent_min = jnp.min(energy_history[-min_steps_without_new_min:])
    global_min = jnp.min(energy_history)
    return recent_min >= global_min

def generate_lenia_data(key, data_dir_path, param_ranges, param_lists, max_steps=10000, num_runs_per_point=100):
    os.makedirs(data_dir_path, exist_ok=True)
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
            
            # Initialize parameters as before
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
                'num_dims': num_dims.item(),
                'size': size.item(),
                'dt': dt.item(),
                'num_species': num_species.item(),
                'num_kernels': num_kernels.item(),
                'num_growth_funcs': num_growth_funcs.item(),
                'params': json_params
            }

            run_hash = make_hash_sha256(run_params)
            run_hash = run_hash.replace('/', '_')
            run_path = os.path.join(data_dir_path, f"run_{run_hash}")
            if os.path.exists(run_path):
                continue

            points = (jax.random.uniform(key, (num_particles, num_dims)) - 0.5) * size
            species = jax.random.randint(key, (num_particles,), 0, num_species)
            
            # Initialize state and history
            state = {'points': points, 'energy': compute_system_energy(points, species, params)}
            points_history = None
            energy_history = None
            steady_state_found = False
            post_steady_state_steps = 0
            
            # Run simulation in chunks of 100 steps
            chunk_size = 100
            total_steps = 0
            
            while total_steps < max_steps:
                # Run chunk
                state, history = jax.lax.scan(
                    functools.partial(scan_step, dt, params, species),
                    state,
                    None,
                    length=chunk_size
                )
                
                # Extract history
                chunk_points = history['points']
                chunk_energies = history['energy']
                if points_history is None:
                    points_history = chunk_points
                    energy_history = chunk_energies
                else:
                    points_history = jnp.concatenate([points_history, chunk_points], axis=0)
                    energy_history = jnp.concatenate([energy_history, chunk_energies], axis=0)
                
                total_steps += chunk_size
                
                # Check for steady state
                if not steady_state_found and check_steady_state(energy_history):
                    steady_state_found = True
                
                # If in steady state, count additional steps
                if steady_state_found:
                    post_steady_state_steps += chunk_size
                    if post_steady_state_steps >= 1000:
                        break
            
            # Subsample the last 1000 steps (or all steps if simulation ended early)
            final_points = jnp.array(points_history[-1000::5])  # Take every 5th step from last 1000
            
            # Save data
            os.makedirs(run_path, exist_ok=True)
            jnp.save(os.path.join(run_path, 'points_history.npy'), final_points)
            
            with open(os.path.join(run_path, 'params.json'), 'w') as f:
                json.dump({
                    **run_params,
                    'reached_steady_state': steady_state_found,
                    'total_steps': total_steps,
                    'final_energy': float(state['energy'])
                }, f)

def main():
    # Define parameter ranges for grid search
    param_ranges = {
        'mu_k': jnp.array([0.1, 10.0]),
        'sigma_k': jnp.array([0.1, 10.0]),
        'w_k': jnp.array([-0.1, 0.1]),
        'mu_g': jnp.array([-1.0, 1.0]),
        'sigma_g': jnp.array([0.01, 1.0]),
        'c_rep': jnp.array([0.1, 10.0])
    }
    
    param_lists = [
        ('num_particles', jnp.array([200])),
        ('num_species', jnp.array([1, 2, 4, 8])),
        ('num_kernels', jnp.array([1, 2, 4, 8])),
        ('num_growth_funcs', jnp.array([1, 2, 4, 8])),
        ('size', jnp.array([20, 100])),
        ('dt', jnp.array([0.1])),
        ('num_dims', jnp.array([2]))
    ]
    key = jax.random.PRNGKey(8)
    
    generate_lenia_data(key, 'new_lenia_data', param_ranges, param_lists, num_runs_per_point=400)

if __name__ == '__main__':
    main()