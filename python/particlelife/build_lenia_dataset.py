import os
import json
import itertools

import jax
import jax.numpy as jnp
import tqdm

from particle_lenia import Params, fields_f, step_f

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

def generate_lenia_data(data_dir_path, param_ranges, max_steps=1000, num_particles=1000, max_stable_steps=100):
    # Create parameter grid
    param_grid = list(itertools.product(*param_ranges.values()))
    
    for i, param_values in enumerate(tqdm.tqdm(param_grid)):
        params = Params(**dict(zip(param_ranges.keys(), param_values)))
        
        key = jax.random.PRNGKey(i)  # Use index as seed for reproducibility
        points = (jax.random.uniform(key, (num_particles, 2)) - 0.5) * 24.0
        
        energy_history = []
        for step in range(max_steps):
            points = step_f(params, points, 0.1)
            energy = compute_system_energy(points, params)
            energy_history.append(energy)
            
            if is_system_stable(energy_history):
                break
        
        energy_fields = []
        for step in range(max_stable_steps):
            points = step_f(params, points, 0.1)
            energy = compute_system_energy(points, params)
            energy_history.append(energy)
            energy_field = create_energy_field(points, params)
            energy_fields.append(energy_field)
        
        # save energy fields
        run_path = os.path.join(data_dir_path, f"run_{i}")
        os.makedirs(run_path, exist_ok=True)
        jnp.save(os.path.join(run_path, 'energy_fields.npy'), jnp.stack(energy_fields))
        jnp.save(os.path.join(run_path, 'energy_history.npy'), jnp.array(energy_history))
        with open(os.path.join(run_path, 'params.json'), 'w') as f:
            json.dump(params._asdict(), f)

def main():
    # Define parameter ranges for grid search
    param_ranges = {
        'mu_k': jnp.linspace(0.1, 10.0, 5),
        'sigma_k': jnp.linspace(0.1, 10.0, 5),
        'w_k': jnp.linspace(0.001, 0.1, 5),
        'mu_g': jnp.linspace(0.1, 1.0, 5),
        'sigma_g': jnp.linspace(0.01, 1.0, 5),
        'c_rep': jnp.linspace(0.1, 10.0, 5)
    }
    
    generate_lenia_data('lenia_data', param_ranges)

if __name__ == '__main__':
    main()