import imageio
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
import numpy as np
import soundcard as sc
from vispy import app, scene, visuals


def create_gaussian_kernel(size, sigma):
    """ Create a 2D Gaussian kernel. """
    x = jnp.arange(-size // 2 + 1., size // 2 + 1.)
    y = x[:, None]
    x2_y2 = x ** 2 + y ** 2
    gaussian_kernel = jnp.exp(-x2_y2 / (2 * sigma ** 2))
    return gaussian_kernel / jnp.sum(gaussian_kernel)

class Physarum:
    def __init__(self, num_agents=1000000, grid_size=1000, num_species=1, num_channels=1, speed=1.0, decay_rate=0.1, deposit_amount=5.0, sensor_angle=jnp.pi / 3, sensor_distance=5):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_species = num_species
        self.num_channels = num_channels
        self.speed = speed
        self.decay_rate = decay_rate
        self.deposit_amount = deposit_amount
        self.sensor_angle = sensor_angle
        self.sensor_distance = sensor_distance

        self.positions, self.directions, self.species = self.initialize_agents()
        self.chemical_grid = self.initialize_chemical_grid()

        self._move_agents = jax.jit(self.move_agents)
        self._update_chemical_grid = jax.jit(self.update_chemical_grid)

    def update(self):
        self.positions, self.directions = self._move_agents(self.positions, self.directions, self.chemical_grid)
        self.chemical_grid = self._update_chemical_grid(self.chemical_grid, self.positions)

    def get_display(self):
        return np.array(self.chemical_grid[0])  # Convert JAX array to NumPy for compatibility with Vispy

    def initialize_agents(self):
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, (self.num_agents, 2), minval=0, maxval=self.grid_size, dtype=jnp.float16)
        directions = jax.random.uniform(key, (self.num_agents,), minval=0, maxval=2 * jnp.pi, dtype=jnp.float16)
        species = jax.random.randint(key, (self.num_agents,), minval=0, maxval=self.num_species, dtype=jnp.int8)
        return positions, directions, species

    def initialize_chemical_grid(self):
        return jnp.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=jnp.float16)

    def move_agents(self, positions, directions, chemical_grid):
        # Calculate sensor positions: front, left, and right
        front_offsets = self.sensor_distance * jnp.stack([jnp.cos(directions), jnp.sin(directions)], axis=-1)
        left_offsets = self.sensor_distance * jnp.stack([jnp.cos(directions + self.sensor_angle), jnp.sin(directions + self.sensor_angle)], axis=-1)
        right_offsets = self.sensor_distance * jnp.stack([jnp.cos(directions - self.sensor_angle), jnp.sin(directions - self.sensor_angle)], axis=-1)

        front_sensors = jnp.round((positions + front_offsets) % self.grid_size)
        left_sensors = jnp.round((positions + left_offsets) % self.grid_size)
        right_sensors = jnp.round((positions + right_offsets) % self.grid_size)

        # Sense the chemical concentration at each sensor
        front_concentration = chemical_grid[:, front_sensors[:, 0].astype(jnp.int32), front_sensors[:, 1].astype(jnp.int32)]
        left_concentration = chemical_grid[:, left_sensors[:, 0].astype(jnp.int32), left_sensors[:, 1].astype(jnp.int32)]
        right_concentration = chemical_grid[:, right_sensors[:, 0].astype(jnp.int32), right_sensors[:, 1].astype(jnp.int32)]

        # Determine steering direction
        steer_left = left_concentration > front_concentration
        steer_right = right_concentration > front_concentration
        new_directions = (directions + self.sensor_angle * steer_left - self.sensor_angle * steer_right).sum(axis=0)

        # Move agents
        dx = self.speed * jnp.cos(new_directions)
        dy = self.speed * jnp.sin(new_directions)
        new_positions = (positions + jnp.stack([dx, dy], axis=-1)) % self.grid_size

        return new_positions, new_directions

    def update_chemical_grid(self, chemical_grid, positions):
        # Deposit chemical
        quantized_positions = jnp.round(positions).astype(int)
        chemical_grid = chemical_grid.at[:, quantized_positions[:, 0], quantized_positions[:, 1]].add(self.deposit_amount).clip(0, 1)

        # Apply decay and diffusion
        # x = jnp.linspace(-3, 3, 3)
        # window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
        window = jnp.array([[0.0625, 0.125, 0.0625],
                            [0.125, 0.25, 0.125],
                            [0.0625, 0.125, 0.0625]], dtype=jnp.float16)
        smooth_image = lax.conv(chemical_grid.reshape((chemical_grid.shape[0], 1, chemical_grid.shape[1], chemical_grid.shape[2])),    # lhs = NCHW image tensor
                window.reshape((1, 1, window.shape[0], window.shape[1])), # rhs = OIHW conv kernel tensor
                (1, 1),  # window strides
                'SAME') # padding mode
        smooth_image = smooth_image.reshape(chemical_grid.shape)
        return smooth_image * (1 - self.decay_rate)
    

class ReactionDiffusionPhysarum(Physarum):
    def __init__(self, kill_rate=0.0545, feed_rate=0.0367, diffusion_rate=0.16):
        self.kill_rate = kill_rate
        self.feed_rate = feed_rate
        self.diffusion_rate = diffusion_rate

        super().__init__(num_channels=2)

    def move_agents(self, positions, directions, chemical_grid):
        # write the move_agents function here
        pass

    def update_chemical_grid(self, chemical_grid, positions):
        # write the update_chemical_grid function here
        pass

class BoidsPhysarum(Physarum):
    def __init__(self, alignment_weight=1.0, cohesion_weight=1.0, separation_weight=1.0):
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.separation_weight = separation_weight

        super().__init__()

    def move_agents(self, positions, directions, chemical_grid):
        # write the move_agents function here
        pass

    def update_chemical_grid(self, chemical_grid, positions):
        # write the update_chemical_grid function here
        pass


class VisualizeSimulation:
    def __init__(self):
        # Initial setup
        key = random.PRNGKey(0)

        self.record = False

        # Audio setup
        self.default_speaker = sc.default_speaker()
        self.fs = 44100  # Sample rate
        self.blocksize = 16000
        self.num_channels = self.default_speaker.channels
        self.phase = 0  # To keep track of phase between updates

        # Initialize simulation
        num_agents = 1000000
        num_species = 1
        num_channels = 1
        grid_size = 1000
        speed = 1.0
        decay_rate = 0.1
        deposit_amount = 5.0
        sensor_angle = jnp.pi / 3
        sensor_distance = 5
        self.simulation = Physarum(num_agents, grid_size, num_species, num_channels, speed, decay_rate, deposit_amount, sensor_angle, sensor_distance)

        self.canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000))
        self.canvas.show()

        view = self.canvas.central_widget.add_view()

        im = self.simulation.get_display()
        self.image = scene.visuals.Image(im, parent=view.scene, clim=(0, 1))
        
        # Configure the view
        # view.camera = 'panzoom'
        # view.camera.aspect = 1.0

        # start audio system
        self.player = self.default_speaker.player(self.fs, self.blocksize)
        self.player.__enter__()  # Manually enter the context

        if self.record:
            self.writer = imageio.get_writer('physarum.gif', duration=0.1)
        self.idx = 0

        # Update the image data periodically
        display_timer = app.Timer()
        display_timer.connect(self.update_image)
        display_timer.start(0)

        audio_timer = app.Timer()
        audio_timer.connect(self.audio_playback)
        audio_timer.start(0)

    def generate_audio_data(self):
        # Example method to generate audio data
        length = self.fs // 10  # Generate 0.1 seconds of audio at a time, for example
        t = np.arange(length) + self.phase
        audio_data = np.sin(2 * np.pi * t * 440.0 / self.fs).astype(np.float32)
        self.phase += length  # Update phase

        if self.num_channels > 1:
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, self.num_channels))
        return audio_data

    def update_image(self, event):
        self.idx += 1
        self.simulation.update()
        im = self.simulation.get_display()
        self.image.set_data(im)
        self.canvas.update()
        if self.record:
            im = self.canvas.render()
            self.writer.append_data(im)

    def generate_audio_data(self):
        # Example method to generate audio data
        length = self.fs // 10  # Generate 0.1 seconds of audio at a time, for example
        t = np.arange(length) + self.phase
        audio_data = np.sin(2 * np.pi * t * 440.0 / self.fs).astype(np.float32)
        self.phase += length  # Update phase

        if self.num_channels > 1:
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, self.num_channels))
        return audio_data

    def audio_playback(self):
        # Method to continuously generate and play audio data
        audio_data = self.generate_audio_data()
        self.player.play(audio_data)

    def close(self):
        if self.record:
            self.writer.close()
        self.canvas.close()
        self.player.__exit__(None, None, None)  # Manually exit the context

def main():
    try:
        visual = VisualizeSimulation()
        app.run()
    finally:
        visual.close()

if __name__ == '__main__':
    main()