import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from pythonosc import udp_client

from particle_lenia import peak_f

@jax.jit
def force_graph(r, rmax, alpha, repulsion_dist, repulsion):
    first = jnp.maximum(repulsion_dist - r, 0.) * repulsion
    second = alpha * jnp.maximum(1 - abs(2 * r - rmax- repulsion_dist) / (rmax - repulsion_dist), 0)
    cond_first = (r < repulsion_dist)
    cond_second = (repulsion_dist < r) & (r < rmax)
    return jnp.where(cond_first, first, jnp.where(cond_second, second, 0.))

@jax.jit
def calc_force(x1, x2, c1, c2, param_alpha, param_rmax, repulsion_dist, repulsion):
    r = x2 - x1
    r -= jnp.round(r) # periodic boundary
    # r = jax.lax.select(r > 0.5, r-1, jax.lax.select(r < -0.5, r+1, r))
    alpha, rmax = param_alpha[c1, c2], param_rmax[c1]
    rlen = jnp.linalg.norm(r)
    rdir = r / (rlen + 1e-8)
    flen = force_graph(rlen, rmax, alpha, repulsion_dist, repulsion)
    return rdir * flen

def compute_step(x, v, species, mass, half_life, dt, alpha, rmax, repulsion_dist, repulsion):
    f = jax.vmap(
        jax.vmap(calc_force, in_axes=(None, 0, None, 0, None, None, None, None)),
        in_axes=(0, None, 0, None, None, None, None, None)
    )(x, x, species, species, alpha, rmax, repulsion_dist, repulsion)
    
    acc = f.sum(axis=-2) / mass[:, None]
    mu = (0.5) ** (dt / half_life[:, None])
    v = mu * v + acc * dt
    x = x + v * dt
    x -= jnp.floor(x) # periodic boundary
    return x, v

def create_multi_step(species, mass, half_life, dt, alpha, rmax, repulsion_dist, repulsion):
    @jax.jit
    def multi_step(carry, _):
        x, v = carry
        x, v = compute_step(x, v, species, mass, half_life, dt, alpha, rmax, repulsion_dist, repulsion)
        return (x, v), x
    return multi_step

class ParticleLife:
    def __init__(self, num_particles, num_species, size=100, dt=0.001, steps_per_frame=10):
        self.num_particles = num_particles
        self.num_species = num_species
        self.size = size
        self.dt = dt
        self.steps_per_frame = steps_per_frame
        
        key = jax.random.PRNGKey(11)
        key, *subkeys = jax.random.split(key, 5)
        
        # Initialize random positions and velocities
        self.positions = jax.random.uniform(subkeys[0], (num_particles, 2), minval=0, maxval=1)
        self.velocities = jnp.zeros((num_particles, 2))
        
        # Randomly assign species
        self.species = jax.random.randint(subkeys[1], (num_particles,), 0, num_species)
        
        # Parameters
        self.mass = jnp.full((num_species,), 0.02)
        self.half_life = jnp.full((num_species,), 0.001)
        self.rmax = jnp.full((num_species,), 0.5)
        self.alpha = jnp.eye(num_species) + 0.12 * jnp.roll(jnp.eye(num_species), 1, axis=0) - 0.1 * jnp.roll(jnp.eye(num_species), -1, axis=0)
        #jax.random.normal(subkeys[2], (num_species, num_species))
        
        repulsion_dist = 0.1
        repulsion = -10.0

        # Create the multi_step function with fixed parameters
        self.multi_step = create_multi_step(
            self.species, self.mass[self.species], self.half_life[self.species], self.dt,
            self.alpha, self.rmax[self.species], repulsion_dist, repulsion
        )
        
    def step(self):
        # Run multiple steps using scan
        (self.positions, self.velocities), positions = jax.lax.scan(
            self.multi_step,
            (self.positions, self.velocities),
            None,
            length=self.steps_per_frame
        )
        return self.positions

def main():
    # Simulation parameters
    num_particles = 1000
    num_species = 6
    size = 1
    steps_per_frame = 10  # Number of physics steps per frame

    client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
    
    # Create simulation
    sim = ParticleLife(num_particles, num_species, size, steps_per_frame=steps_per_frame)

    # send initial species data to set up synths
    for i in range(num_particles):
        client.send_message("/species", [i, int(sim.species[i])])
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(sim.positions[:, 0], sim.positions[:, 1],
                        c=sim.species, cmap='tab10', s = 5)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    
    # Animation update function
    def update(frame):
        positions = sim.step()
        scatter.set_offsets(positions)

        

        return scatter,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=None,
                        interval=1, blit=True)
    
    plt.show()

if __name__ == "__main__":
    main()