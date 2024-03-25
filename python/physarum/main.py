import multiprocessing
import threading
import time

import imageio
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
import numpy as np
# import soundcard as sc
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

        self.agents = self.initialize_agents()
        self.chemical_grid = self.initialize_chemical_grid()

        jit = True
        if jit:
            self._move_agents = jax.jit(self.move_agents)
            self._update_chemical_grid = jax.jit(self.update_chemical_grid)
            self._update = jax.jit(self.update_sim)
            # TODO jit audio update
        else:
            self._move_agents = self.move_agents
            self._update_chemical_grid = self.update_chemical_grid
            self._update = self.update_sim

    def update_sim(self, agents, chemical_grid):
        agents = self._move_agents(agents, chemical_grid)
        chemical_grid = self._update_chemical_grid(chemical_grid, agents)
        return agents, chemical_grid
    
    def update(self):
        self.agents, self.chemical_grid = self._update(self.agents, self.chemical_grid)

    def get_display(self):
        return np.array(self.chemical_grid[0])  # Convert JAX array to NumPy for compatibility with Vispy
    
    def get_audio(self, t):
        max_sense = self.agents[:, self.alookup['max_sense']]
        phase = self.agents[:, self.alookup['phase']]
        max_sense = max_sense.reshape((1, -1))
        phase = phase.reshape((1, -1))
        t = t.reshape((-1, 1))
        num_sources = phase.shape[1]
        min_pitch = 40
        max_pitch = 400
        pitch = min_pitch + (max_pitch - min_pitch) * max_sense
        # TODO goes OOM, fix
        signals = (1 / num_sources) * jnp.sin(2 * jnp.pi * t * pitch + phase)
        signal = jnp.sum(signals, axis=1)
        return np.array(signal).astype(np.float32)

    def initialize_agents(self):
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, (self.num_agents, 2), minval=0, maxval=self.grid_size, dtype=jnp.float16)
        directions = jax.random.uniform(key, (self.num_agents, 1), minval=0, maxval=2 * jnp.pi, dtype=jnp.float16)
        species = jax.random.randint(key, (self.num_agents,1), minval=0, maxval=self.num_species, dtype=jnp.int8)
        max_sense = jnp.zeros((self.num_agents, 1), dtype=jnp.float16)
        phase = jax.random.uniform(key, (self.num_agents, 1), minval=0, maxval=2*jnp.pi, dtype=jnp.float16)
        cols = [positions, directions, species, max_sense, phase]
        col_names = ['positions', 'directions', 'species', 'max_sense', 'phase']
        assert len(cols) == len(col_names)
        self.alookup = {}
        col_idx = 0
        for col_name, col in zip(col_names, cols):
            col_width = col.shape[1]
            if col_width == 1:
                self.alookup[col_name] = col_idx
            else:
                self.alookup[col_name] = slice(col_idx, col_idx+col_width)
            col_idx += col_width
        agents = jnp.concatenate(cols, axis=1)
        return agents

    def initialize_chemical_grid(self):
        return jnp.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=jnp.float16)

    def move_agents(self, agents, chemical_grid):
        positions = agents[:, self.alookup['positions']]
        directions = agents[:, self.alookup['directions']]
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

        agents = agents.at[:, self.alookup['positions']].set(new_positions)
        agents = agents.at[:, self.alookup['directions']].set(new_directions)

        return agents

    def update_chemical_grid(self, chemical_grid, agents):
        positions = agents[:, self.alookup['positions']]
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


def audio_timer(get_audio, fs=44100, blocksize=16000, interval=1):
    import soundcard as sc
    default_speaker = sc.default_speaker()
    num_channels = default_speaker.channels
    with default_speaker.player(fs, channels=num_channels, blocksize=blocksize) as player:
        next_call = time.time()
        while True:
            audio_data = get_audio(fs, num_channels)
            player.play(audio_data)
            next_call = next_call + interval
            time.sleep(max(next_call - time.time(), 0))

class VisualizeSimulation:
    def __init__(self):
        # Initial setup
        key = random.PRNGKey(0)

        self.record = False

        # Audio setup
        # self.default_speaker = sc.default_speaker()
        # self.fs = 44100  # Sample rate
        # self.blocksize = 16000
        # self.num_channels = self.default_speaker.channels
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
        
        if self.record:
            self.writer = imageio.get_writer('physarum.gif', duration=0.1)
        self.idx = 0

        # Update the image data periodically
        self.display_timer = app.Timer()
        self.display_timer.connect(self.update_image)
        self.display_timer.start()

        use_process = False
        kwargs = {'target': audio_timer, 'args': (self.generate_audio_data,), 'kwargs': {'interval':self.display_timer.interval}}
        if use_process:
            self.audio_timer = multiprocessing.Process(**kwargs)
        else:
            self.audio_timer = threading.Thread(**kwargs)
        self.audio_timer.start()

    def update_image(self, event):
        self.idx += 1
        self.simulation.update()
        im = self.simulation.get_display()
        self.image.set_data(im)
        self.canvas.update()
        # audio_data = self.generate_audio_data()
        # self.player.play(audio_data)
        if self.record:
            im = self.canvas.render()
            self.writer.append_data(im)

    def generate_audio_data(self, fs, num_channels):
        # Example method to generate audio data
        length = fs // 10  # Generate 0.1 seconds of audio at a time, for example
        t = np.arange(length) + self.phase
        audio_data = self.simulation.get_audio(t / fs)
        self.phase += length  # Update phase

        if num_channels > 1:
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, num_channels))
        return audio_data


    def close(self):
        if self.record:
            self.writer.close()
        self.display_timer.stop()
        self.audio_timer.close()
        self.canvas.close()
        self.player.__exit__(None, None, None)  # Manually exit the context

def main():
    visual = None
    try:
        visual = VisualizeSimulation()
        app.run()
    except:
        raise
    finally:
        if hasattr(visual, 'close'):
            visual.close()

if __name__ == '__main__':
    main()