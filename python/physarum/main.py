import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_gaussian_kernel(size, sigma):
    """ Create a 2D Gaussian kernel. """
    x = jnp.arange(-size // 2 + 1., size // 2 + 1.)
    y = x[:, None]
    x2_y2 = x ** 2 + y ** 2
    gaussian_kernel = jnp.exp(-x2_y2 / (2 * sigma ** 2))
    return gaussian_kernel / jnp.sum(gaussian_kernel)


def initialize_agents(num_agents, grid_size, num_species):
    key = jax.random.PRNGKey(0)
    positions = jax.random.uniform(key, (num_agents, 2), minval=0, maxval=grid_size, dtype=jnp.float16)
    directions = jax.random.uniform(key, (num_agents,), minval=0, maxval=2 * jnp.pi, dtype=jnp.float16)
    species = jax.random.randint(key, (num_agents,), minval=0, maxval=num_species, dtype=jnp.int8)
    return positions, directions, species

def initialize_chemical_grid(grid_size, num_channels):
    return jnp.zeros((grid_size, grid_size, num_channels), dtype=jnp.float16)

def move_agents(positions, directions, speed, grid_size, chemical_grid, sensor_angle, sensor_distance):
    # Calculate sensor positions: front, left, and right
    front_offsets = sensor_distance * jnp.stack([jnp.cos(directions), jnp.sin(directions)], axis=-1)
    left_offsets = sensor_distance * jnp.stack([jnp.cos(directions + sensor_angle), jnp.sin(directions + sensor_angle)], axis=-1)
    right_offsets = sensor_distance * jnp.stack([jnp.cos(directions - sensor_angle), jnp.sin(directions - sensor_angle)], axis=-1)

    front_sensors = (positions + front_offsets) % grid_size
    left_sensors = (positions + left_offsets) % grid_size
    right_sensors = (positions + right_offsets) % grid_size

    # Sense the chemical concentration at each sensor
    front_concentration = chemical_grid[front_sensors[:, 0].astype(jnp.int32), front_sensors[:, 1].astype(jnp.int32)]
    left_concentration = chemical_grid[left_sensors[:, 0].astype(jnp.int32), left_sensors[:, 1].astype(jnp.int32)]
    right_concentration = chemical_grid[right_sensors[:, 0].astype(jnp.int32), right_sensors[:, 1].astype(jnp.int32)]

    # Determine steering direction
    steer_left = left_concentration > front_concentration
    steer_right = right_concentration > front_concentration
    new_directions = directions + sensor_angle * steer_left - sensor_angle * steer_right

    # Move agents
    dx = speed * jnp.cos(new_directions)
    dy = speed * jnp.sin(new_directions)
    new_positions = (positions + jnp.stack([dx, dy], axis=-1)) % grid_size

    return new_positions, new_directions

def update_chemical_grid(chemical_grid, positions, decay_rate, deposit_amount):
    # Deposit chemical
    quantized_positions = jnp.round(positions).astype(int)
    chemical_grid = chemical_grid.at[quantized_positions[:, 0], quantized_positions[:, 1]].add(deposit_amount).clip(0, 1)

    # Apply decay and diffusion
    # x = jnp.linspace(-3, 3, 3)
    # window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
    window = jnp.array([[0.0625, 0.125, 0.0625],
                        [0.125, 0.25, 0.125],
                        [0.0625, 0.125, 0.0625]], dtype=jnp.float16)
    smooth_image = jsp.signal.convolve(chemical_grid, window, mode='same')
    chemical_grid = smooth_image * (1 - decay_rate)
    return chemical_grid

class Physarum:
    def __init__(self):
        # Simulation parameters
        num_agents = 10000
        num_species = 1
        num_channels = 1
        self.grid_size = 1000
        self.speed = 1.0
        self.decay_rate = 0.1
        self.deposit_amount = 5.0
        self.sensor_angle = jnp.pi / 3
        self.sensor_distance = 5

        # Initialize simulation
        self.positions, self.directions, self.species = initialize_agents(num_agents, self.grid_size, num_species)
        self.chemical_grid = initialize_chemical_grid(self.grid_size, num_channels)

        self.fig, ax = plt.subplots(figsize=(10, 10))
        self.chemical_image = ax.imshow(self.chemical_grid, cmap='viridis', origin='lower', vmin=0., vmax=1.)

    def update(self, frame):
        # Simulation loop
        self.positions, self.directions = move_agents(self.positions, self.directions, self.speed, self.grid_size, self.chemical_grid, self.sensor_angle, self.sensor_distance)
        self.chemical_grid = update_chemical_grid(self.chemical_grid, self.positions, self.decay_rate, self.deposit_amount)

        self.chemical_image.set_data(self.chemical_grid)

        return self.chemical_image,

    def draw(self):
        ani = FuncAnimation(self.fig, self.update, frames=10000, interval=0, blit=True)
        plt.show()

def main():
    physarum = Physarum()
    physarum.draw()

if __name__ == '__main__':
    main()