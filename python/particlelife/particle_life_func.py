
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from particle_life_original import force_graph as old_force_graph
from particle_life import force_graph as new_force_graph
from particle_life import e_force_graph as new_force_graph_deriv

def main():

    fig, ax = plt.subplots()
    r = jnp.linspace(0, 1, 100)
    a = 1.0
    rmax = 0.5
    repulsion_dist = 0.2
    repulsion = 10.0
    x = jnp.array([0.0])
    xs = r[:, None, None]

    ax.plot(r, jax.vmap(old_force_graph, in_axes=(0, None, None, None, None))(r, rmax, a, repulsion_dist, repulsion), label="Old force")
    ax.plot(r, jax.vmap(new_force_graph, in_axes=(None, 0, None, None, None, None))(x, xs, a, rmax, repulsion_dist, repulsion), label="New force")
    ax.plot(r, jax.vmap(new_force_graph_deriv, in_axes=(None, 0, None, None, None, None))(x, xs, a, rmax, repulsion_dist, repulsion), label="New force via derivative")
    ax.set_xlabel("r")
    ax.set_ylabel("Force")
    ax.set_title("Force graph")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()