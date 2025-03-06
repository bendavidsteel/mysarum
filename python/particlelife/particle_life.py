import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from particle_lenia import peak_f

@jax.jit
def force_graph(x, xs, a, rmax, repulsion_dist, repulsion):
    diff = xs - x
    # diff -= jnp.round(diff) # periodic boundary
    r = jnp.sqrt(jnp.square(diff).sum(-1).clip(1e-10))
    w_k = a
    sigma = (rmax - repulsion_dist) / 3
    mu = repulsion_dist + (rmax - repulsion_dist) / 2
    U = peak_f(r, mu, sigma) * w_k
    R = repulsion * (repulsion_dist - r).clip(0.0)
    dir = diff / (r[:, jnp.newaxis] + 1e-10)
    return (dir * (U-R)[:, jnp.newaxis]).sum(0)

@jax.jit
def e_force_graph(x, xs, a, rmax, repulsion_dist, repulsion):
    mu = repulsion_dist + (rmax - repulsion_dist) / 2
    sigma = (rmax - repulsion_dist) / 7
    def field(x):
        diff = xs - x
        diff -= jnp.round(diff) # periodic boundary
        r = jnp.sqrt(jnp.square(diff).sum(-1).clip(1e-10))
        U = (-0.17 * a / (1 + jnp.exp(-(r - mu) / sigma))).sum()
        R = (repulsion / 2) * ((repulsion_dist - r).clip(0.0) ** 2).sum()
        return U-R
    return jax.grad(field)(x)

@jax.jit
def calc_force(x, xs, c1, c2, param_alpha, param_rmax, repulsion_dist, repulsion):
    alpha, rmax = param_alpha[c1, c2], param_rmax[c1]
    f = e_force_graph(x, xs, alpha, rmax, repulsion_dist, repulsion)
    return f

@jax.jit
def compute_step(x, v, species, mass, half_life, dt, alpha, rmax, repulsion_dist, repulsion):
    f = jax.vmap(
        calc_force,
        in_axes=(0, None, 0, None, None, None, None, None)
    )(x, x, species, species, alpha, rmax, repulsion_dist, repulsion)
    
    acc = f / mass[:, jnp.newaxis]
    mu = (0.5) ** (dt / half_life)
    v = mu[:, jnp.newaxis] * v + acc * dt
    x = x + v * dt
    x -= jnp.floor(x)  # periodic boundary
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
        
        key = jax.random.PRNGKey(14)
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
        self.beta = jnp.full((num_species,), 0.2)
        self.alpha = jnp.eye(num_species) + 0.12 * jnp.roll(jnp.eye(num_species), 1, axis=1) - 0.1 * jnp.roll(jnp.eye(num_species), -1, axis=1)
        #jax.random.normal(subkeys[2], (num_species, num_species))
        
        repulsion_dist = 0.1
        repulsion = 10.0

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
    
    # Create simulation
    sim = ParticleLife(num_particles, num_species, size, steps_per_frame=steps_per_frame)
    
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