from collections import namedtuple

import einops
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import numpy as np

from particle_lenia import Params, step_f, fields_f

def main():
    key = jax.random.PRNGKey(7)

    # Initial parameters
    method = 2
    if method == 1:
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
        num_species = 10
        num_kernels = 3
        num_growth_funcs = 3
        mu_k = jax.random.uniform(key, (num_species, num_species, num_kernels), minval=0.01, maxval=5.0)
        sigma_k = jax.random.uniform(key, (num_species, num_species, num_kernels), minval=0.05, maxval=2.0)

        mu_g = jax.random.uniform(key, (num_species, num_species, num_growth_funcs), minval=0.1, maxval=5.0)
        sigma_g = jax.random.uniform(key, (num_species, num_species, num_growth_funcs), minval=0.05, maxval=2.0)

        w_k = jax.random.uniform(key, (num_species, num_kernels), minval=-0.1, maxval=0.1)
        c_rep = jax.random.uniform(key, (num_species, num_species), minval=10.0, maxval=30.0)
    
    params = Params(mu_k=mu_k, sigma_k=sigma_k, w_k=w_k, mu_g=mu_g, sigma_g=sigma_g, c_rep=c_rep)
    num_points = 2000
    num_dims = 2
    map_size = 24
    points = jax.random.uniform(key, [num_points, num_dims]) * map_size
    species = jax.random.randint(key, [num_points], 0, num_species)
    dt = 0.001

    # Set up the figure and axis
    if num_dims == 2:
        fig, ax = plt.subplots(figsize=(8,8))
    elif num_dims == 3:
        fix, ax = plt.subplot()
    add_gui_controls = False
    if add_gui_controls:
        plt.subplots_adjust(left=0.25, bottom=0.35)
    scat = ax.scatter(points[:, 0], points[:, 1], s=5, c=species)

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)

    add_field = False
    if add_field:
        # Create initial E field image
        X, Y = jp.meshgrid(jp.linspace(0, map_size, 100), jp.linspace(0, map_size, 100))
        grid_points = jp.stack([X, Y], axis=-1).reshape(-1, 2)
        E_field = jax.vmap(lambda p: fields_f(params, points, p).E)(grid_points).reshape(100, 100)
        im = ax.imshow(E_field, extent=(0, map_size, 0, map_size), origin='lower', cmap='viridis', alpha=0.5)

    if add_gui_controls:
        # Slider axes
        axcolor = 'lightgoldenrodyellow'
        ax_mu_k = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
        ax_sigma_k = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
        ax_w_k = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        ax_mu_g = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        ax_sigma_g = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        ax_c_rep = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)

        # Sliders
        s_mu_k = Slider(ax_mu_k, 'mu_k', 0.1, 10.0, valinit=params.mu_k)
        s_sigma_k = Slider(ax_sigma_k, 'sigma_k', 0.1, 10.0, valinit=params.sigma_k)
        s_w_k = Slider(ax_w_k, 'w_k', 0.001, 0.1, valinit=params.w_k)
        s_mu_g = Slider(ax_mu_g, 'mu_g', 0.1, 1.0, valinit=params.mu_g)
        s_sigma_g = Slider(ax_sigma_g, 'sigma_g', 0.01, 1.0, valinit=params.sigma_g)
        s_c_rep = Slider(ax_c_rep, 'c_rep', 0.1, 10.0, valinit=params.c_rep)

        # Toggle button for E field visualization
        ax_button = plt.axes([0.8, 0.95, 0.1, 0.04])
        button = Button(ax_button, 'Toggle E field')

        show_E_field = [False]

    # Update function
    def update(frame):
        nonlocal points
        if add_gui_controls:
            current_params = Params(
                mu_k=s_mu_k.val, 
                sigma_k=s_sigma_k.val, 
                w_k=s_w_k.val, 
                mu_g=s_mu_g.val, 
                sigma_g=s_sigma_g.val, 
                c_rep=s_c_rep.val
            )
        else:
            current_params = Params(
                mu_k=params.mu_k,
                sigma_k=params.sigma_k, 
                w_k=params.w_k, 
                mu_g=params.mu_g, 
                sigma_g=params.sigma_g, 
                c_rep=params.c_rep
            )
            
        points = step_f(current_params, points, species, dt)
        points = jp.clip(points, 0, map_size)
        scat.set_offsets(points)
        
        if add_field:
            if show_E_field[0]:
                E_field = jax.vmap(lambda p: fields_f(current_params, points, p).E)(grid_points).reshape(100, 100)
                im.set_data(E_field)
            else:
                im.set_data(jp.zeros((100, 100)))
            return scat, im
        else:
            return scat,

    if add_field:
        # Toggle E field display
        def toggle_E_field(event):
            show_E_field[0] = not show_E_field[0]
            im.set_visible(show_E_field[0])

        button.on_clicked(toggle_E_field)

    # Animation
    ani = FuncAnimation(fig, update, interval=2, blit=True)

    # Show the plot with sliders
    plt.show()

if __name__ == "__main__":
    main()
