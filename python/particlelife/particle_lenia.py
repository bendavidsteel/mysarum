from collections import namedtuple

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# Define namedtuples for parameters and fields
Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')

@jax.jit
def peak_f(x, mu, sigma):
    """Compute the Gaussian peak function."""
    return jp.exp(-((x - mu) / sigma) ** 2)

@jax.jit
def fields_f(p: Params, points, x):
    """Calculate the fields U, G, R, and E based on parameters and points."""
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))
    U = peak_f(r, p.mu_k, p.sigma_k).sum() * p.w_k
    G = peak_f(U, p.mu_g, p.sigma_g)
    R = p.c_rep / 2 * ((1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)

def motion_f(params, points):
    """Compute the motion vector field as the negative gradient of the energy."""
    grad_E = jax.grad(lambda x: fields_f(params, points, x).E)
    return -jax.vmap(grad_E)(points)

@jax.jit
def step_f(params, points, dt):
    """Perform a single Euler integration step."""
    return points + dt * motion_f(params, points)

def main():
    # Initial parameters
    params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    key = jax.random.PRNGKey(20)
    num_points = 1000
    points = (jax.random.uniform(key, [num_points, 2]) - 0.5) * 12.0
    dt = 0.1

    # Set up the figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
    scat = ax.scatter(points[:, 0], points[:, 1], s=1)

    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)

    # Create initial E field image
    X, Y = jp.meshgrid(jp.linspace(-12, 12, 100), jp.linspace(-12, 12, 100))
    grid_points = jp.stack([X, Y], axis=-1).reshape(-1, 2)
    E_field = jax.vmap(lambda p: fields_f(params, points, p).E)(grid_points).reshape(100, 100)
    im = ax.imshow(E_field, extent=(-12, 12, -12, 12), origin='lower', cmap='viridis', alpha=0.5)

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
        current_params = Params(
            mu_k=s_mu_k.val, 
            sigma_k=s_sigma_k.val, 
            w_k=s_w_k.val, 
            mu_g=s_mu_g.val, 
            sigma_g=s_sigma_g.val, 
            c_rep=s_c_rep.val
        )
        points = step_f(current_params, points, dt)
        scat.set_offsets(points)
        
        if show_E_field[0]:
            E_field = jax.vmap(lambda p: fields_f(current_params, points, p).E)(grid_points).reshape(100, 100)
            im.set_data(E_field)
        return scat, im

    # Toggle E field display
    def toggle_E_field(event):
        show_E_field[0] = not show_E_field[0]
        im.set_visible(show_E_field[0])

    button.on_clicked(toggle_E_field)

    # Animation
    ani = FuncAnimation(fig, update, frames=range(1000), interval=20, blit=True)

    # Show the plot with sliders
    plt.show()

if __name__ == "__main__":
    main()
