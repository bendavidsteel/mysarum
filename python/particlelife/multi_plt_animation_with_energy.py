from collections import namedtuple

import einops
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import numpy as np

from particle_lenia import Params, step_f, fields_f, total_energy_f

def main():
    key = jax.random.PRNGKey(8)

    # Initial parameters
    method = 3
    if method == 1:
        num_points = 500
        map_size = 50
        param_config = {
            "species": [
                {
                    "kernels": [
                        {
                            "mu": 1.5,
                            "sigma": 1.0
                        },
                        {
                            "mu": 3.0,
                            "sigma": 1.2
                        }
                    ],
                    "growth_funcs": [
                        {
                            "mu": 0.6,
                            "sigma": 0.15
                        },
                        {
                            "mu": 1.8,
                            "sigma": 0.5
                        }
                    ]
                },
                {
                    "kernels": [
                        {
                            "mu": 1.1,
                            "sigma": 1.4
                        },
                        {
                            "mu": 2.2,
                            "sigma": 0.9
                        }
                    ],
                    "growth_funcs": [
                        {
                            "mu": 0.4,
                            "sigma": 0.2
                        },
                        {
                            "mu": 2.4,
                            "sigma": 0.8
                        }
                    ]
                }
            ]
        }
        num_species = len(param_config['species'])
        max_kernels = max([len(s['kernels']) for s in param_config['species']])
        max_growth_funcs = max([len(s['growth_funcs']) for s in param_config['species']])

        mu_k = np.zeros((num_species, max_kernels))
        sigma_k = np.zeros((num_species, max_kernels))

        mu_g = np.zeros((num_species, max_growth_funcs))
        sigma_g = np.zeros((num_species, max_growth_funcs))

        for i, species in enumerate(param_config['species']):
            for j, kernel in enumerate(species['kernels']):
                mu_k[i, j] = kernel['mu']
                sigma_k[i, j] = kernel['sigma']
            for j, growth_func in enumerate(species['growth_funcs']):
                mu_g[i, j] = growth_func['mu']
                sigma_g[i, j] = growth_func['sigma']

        mu_k = jp.array(mu_k)
        sigma_k = jp.array(sigma_k)
        mu_g = jp.array(mu_g)
        sigma_g = jp.array(sigma_g)

    elif method == 2:
        num_points = 500
        num_species = 4
        num_kernels = 2
        num_growth_funcs = 2
        map_size = 50
        mu_k = jax.random.uniform(key, (num_species, num_species, num_kernels), minval=2.0, maxval=5.0)
        sigma_k = jax.random.uniform(key, (num_species, num_species, num_kernels), minval=0.5, maxval=2.0)

        mu_g = jax.random.uniform(key, (num_species, num_species, num_growth_funcs), minval=-2.0, maxval=2.0)
        sigma_g = jax.random.uniform(key, (num_species, num_species, num_growth_funcs), minval=0.01, maxval=1.0)

        w_k = jax.random.uniform(key, (num_species, num_kernels), minval=-0.04, maxval=0.04)
        c_rep = jax.random.uniform(key, (num_species, num_species), minval=0.1, maxval=1.0)
    elif method == 3:
        # params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
        mu_k = jp.array(4.0).reshape(1, 1, 1)
        sigma_k = jp.array(1.0).reshape(1, 1, 1)
        w_k = jp.array(0.022).reshape(1, 1)
        mu_g = jp.array(0.6).reshape(1, 1, 1)
        sigma_g = jp.array(0.15).reshape(1, 1, 1)
        c_rep = jp.array(1.0).reshape(1, 1)
        num_points = 200
        num_species = 1
        num_dims = 2
        map_size = 20
        points = jax.random.uniform(key, [num_points, num_dims], minval=-0.5, maxval=0.5) * map_size
        dt = 0.1
    
    params = Params(mu_k=mu_k, sigma_k=sigma_k, w_k=w_k, mu_g=mu_g, sigma_g=sigma_g, c_rep=c_rep)
    
    num_dims = 2
    points = jax.random.uniform(key, [num_points, num_dims], minval=-0.5, maxval=0.5) * map_size
    species = jax.random.randint(key, [num_points], 0, num_species)
    dt = 0.1

    jit = True
    if jit:
        _fields_f = jax.jit(fields_f)
        _step_f = jax.jit(step_f)
        _total_energy_f = jax.jit(total_energy_f)
    else:
        _fields_f = fields_f
        _step_f = step_f
        _total_energy_f = total_energy_f

    # Set up the figure and axis
    fig, axes = plt.subplots(ncols=2, figsize=(8,8))
    ax = axes[0]
    scat = ax.scatter(points[:, 0], points[:, 1], s=20, c=species)
    line, = axes[1].plot([], [], lw=2)
    axes[1].set_ylim(-100, 100)
    axes[1].set_xlim(0, 9000)

    ax.set_xlim(-map_size/2, map_size/2)
    ax.set_ylim(-map_size/2, map_size/2)

    # Update function
    def update(frame):
        nonlocal points
        current_params = Params(
            mu_k=params.mu_k,
            sigma_k=params.sigma_k, 
            w_k=params.w_k, 
            mu_g=params.mu_g, 
            sigma_g=params.sigma_g, 
            c_rep=params.c_rep
        )
            
        points = _step_f(current_params, points, species, dt)
        points = jp.clip(points, -map_size/2, map_size/2)
        scat.set_offsets(points)

        # Update energy plot
        energy = _total_energy_f(current_params, points, species)
        line.set_xdata(np.append(line.get_xdata(), frame))
        line.set_ydata(np.append(line.get_ydata(), energy))
        # axes[1].set_xlim(0, frame + 1)
        
        return scat, line

    # Animation
    ani = FuncAnimation(fig, update, interval=2, blit=True)

    # Show the plot with sliders
    plt.show()

if __name__ == "__main__":
    main()
