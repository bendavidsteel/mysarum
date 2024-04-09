from collections import namedtuple
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

def simple_colour_wheel(num_colours):
    colour_points = np.linspace(0., 1., num_colours)
    key_colours = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    key_points = np.linspace(0., 1., len(key_colours))
    colours = []
    for point in colour_points:
        for i in range(len(key_points) - 1):
            if key_points[i] <= point < key_points[i+1]:
                scaled_point = (point - key_points[i]) / (key_points[i+1] - key_points[i])
                colours.append((1 - scaled_point) * key_colours[i] + scaled_point * key_colours[i+1])
                break
        else:
            i = len(key_points) - 2
            scaled_point = (point - key_points[i]) / (key_points[i+1] - key_points[i])
            colours.append((1 - scaled_point) * key_colours[i] + scaled_point * key_colours[i+1])

    assert len(colours) == num_colours
    return np.array(colours)


class DSCL:
    def __init__(self, random_key, jit=True, num_agents=1000000, grid_size=1000, num_species=1, num_chemicals=1, speed=1.0, decay_rate=0.1, deposit_amount=5.0, sensor_angle=jnp.pi / 3, sensor_distance=5):
        self.random_key = random_key
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_species = num_species
        self.num_chemicals = num_chemicals
        self.decay_rate = decay_rate
        self.deposit_amount = deposit_amount

        self.agents = self.initialize_agents()
        self.chemical_grid = self.initialize_chemical_grid()

        self.colours = jnp.array(simple_colour_wheel(self.num_chemicals))

        if jit:
            self._move_agents = jax.jit(self.move_agents)
            self._update_chemical_grid = jax.jit(self.update_chemical_grid)
            self._update = jax.jit(self.update_sim)
            self._display = jax.jit(self.display_sim)
            # TODO jit audio update
        else:
            self._move_agents = self.move_agents
            self._update_chemical_grid = self.update_chemical_grid
            self._update = self.update_sim
            self._display = self.display_sim

    def update_sim(self, agents, chemical_grid):
        agents = self._move_agents(agents, chemical_grid)
        chemical_grid = self._update_chemical_grid(chemical_grid, agents)
        return agents, chemical_grid
    
    def update(self):
        self.agents, self.chemical_grid = self._update(self.agents, self.chemical_grid)

    def get_display(self):
        return np.array(self._display(self.chemical_grid, self.colours))
    
    def display_sim(self, chemical_grid, colours):
        colours = jnp.expand_dims(chemical_grid, 1) * jnp.expand_dims(jnp.expand_dims(colours, -1), -1)
        colour_im = jnp.sum(colours, axis=0)
        return jnp.moveaxis(colour_im, 0, -1).astype(np.float32)
    
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
        raise NotImplementedError

    def initialize_chemical_grid(self):
        return jnp.zeros((self.num_chemicals, self.grid_size, self.grid_size), dtype=jnp.float32)

    def get_sense_reward(self, senses, agents):
        return senses

    def move_agents(self, agents, chemical_grid):
        raise NotImplementedError

    def deposit(self, chemical_grid, quantized_positions, agents):
        return chemical_grid.at[:, quantized_positions[:, 0], quantized_positions[:, 1]].add(self.deposit_amount).clip(0, 1)

    def update_chemical_grid(self, chemical_grid, agents):
        positions = agents[:, self.alookup['positions']]
        # Deposit chemical
        quantized_positions = jnp.round(positions).astype(int)
        chemical_grid = self.deposit(chemical_grid, quantized_positions, agents)

        # Apply decay and diffusion
        # x = jnp.linspace(-3, 3, 3)
        # window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
        window = jnp.array([[0.0625, 0.125, 0.0625],
                            [0.125, 0.25, 0.125],
                            [0.0625, 0.125, 0.0625]], dtype=jnp.float32)
        # window = jnp.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]], dtype=jnp.float32)
        smooth_image = lax.conv(chemical_grid.reshape((chemical_grid.shape[0], 1, chemical_grid.shape[1], chemical_grid.shape[2])),    # lhs = NCHW image tensor
                window.reshape((1, 1, window.shape[0], window.shape[1])), # rhs = OIHW conv kernel tensor
                (1, 1),  # window strides
                'SAME') # padding mode
        smooth_image = smooth_image.reshape(chemical_grid.shape)
        return smooth_image * (1 - self.decay_rate)
    
class Physarum(DSCL):
    def __init__(self, *args, speed=1.0, sensor_angle=jnp.pi / 3, sensor_distance=5, **kwargs):
        
        self.speed = speed
        self.sensor_angle = sensor_angle
        self.sensor_distance = sensor_distance

        super().__init__(*args, **kwargs)

    def initialize_agents(self):
        positions = jax.random.uniform(self.random_key, (self.num_agents, 2), minval=0, maxval=self.grid_size, dtype=jnp.float32)
        directions = jax.random.uniform(self.random_key, (self.num_agents, 1), minval=0, maxval=2 * jnp.pi, dtype=jnp.float32)
        species = jax.random.randint(self.random_key, (self.num_agents,1), minval=0, maxval=self.num_species, dtype=jnp.int8)
        max_sense = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        phase = jax.random.uniform(self.random_key, (self.num_agents, 1), minval=0, maxval=2*jnp.pi, dtype=jnp.float32)
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

    def move_agents(self, agents, chemical_grid):
        positions = agents[:, self.alookup['positions']]
        directions = agents[:, self.alookup['directions']]
        # Calculate sensor positions: front, left, and right
        # TODO vectorize these 3 calcs into 1
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

        front_reward, left_reward, right_reward = self.get_sense_reward([front_concentration, left_concentration, right_concentration], agents)

        # Determine steering direction
        steer_left = left_reward > front_reward
        steer_right = right_reward > front_reward
        new_directions = (directions + self.sensor_angle * steer_left - self.sensor_angle * steer_right).mean(axis=0)

        # Move agents
        dx = self.speed * jnp.cos(new_directions)
        dy = self.speed * jnp.sin(new_directions)
        new_positions = (positions + jnp.stack([dx, dy], axis=-1)) % self.grid_size

        agents = agents.at[:, self.alookup['positions']].set(new_positions)
        agents = agents.at[:, self.alookup['directions']].set(new_directions)

        return agents


class ParticleSystem(DSCL):
    def __init__(self, *args, force_factor=1.0, num_sensors=8, sensor_distance=5, **kwargs):
        
        self.force_factor = force_factor
        self.num_sensors = num_sensors
        self.sensor_distance = sensor_distance

        super().__init__(*args, **kwargs)

    def initialize_agents(self):
        positions = jax.random.uniform(self.random_key, (self.num_agents, 2), minval=0, maxval=self.grid_size, dtype=jnp.float32)
        velocity = jnp.zeros((self.num_agents, 2), dtype=jnp.float32)
        species = jax.random.randint(self.random_key, (self.num_agents,1), minval=0, maxval=self.num_species, dtype=jnp.int8)
        max_sense = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        phase = jax.random.uniform(self.random_key, (self.num_agents, 1), minval=0, maxval=2*jnp.pi, dtype=jnp.float32)
        cols = [positions, velocity, species, max_sense, phase]
        col_names = ['positions', 'velocity', 'species', 'max_sense', 'phase']
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

    def move_agents(self, agents, chemical_grid):
        positions = agents[:, self.alookup['positions']]
        velocity = agents[:, self.alookup['velocity']]
        # Calculate sensor positions: around the particle
        angles = jnp.linspace(0., 2 * jnp.pi, self.num_sensors + 1)[:-1]
        offsets = self.sensor_distance * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        # sensors = jnp.round((positions + jnp.expand_dims(offsets, 1)) % self.grid_size)
        # concentrations = chemical_grid[:, sensors[..., 0].astype(jnp.int32), sensors[..., 1].astype(jnp.int32)]
        # concentrations = jnp.moveaxis(concentrations, 0, 1)
        # # Sense the chemical concentration at each sensor
        # sensor_rewards = self.get_sense_reward(concentrations, agents)
        def fields_f(position, agent):
            sensors = jnp.round((position + offsets) % self.grid_size).astype(jnp.int32)
            sensor_concentrations = chemical_grid[:, sensors[..., 0], sensors[..., 1]]
            sensor_concentrations = jnp.moveaxis(sensor_concentrations, 0, 1)
            pos_sensor = jnp.round(position).astype(jnp.int32) % self.grid_size
            pos_concentration = chemical_grid[:, pos_sensor[0], pos_sensor[1]]
            # Sense the chemical concentration at each sensor
            sensor_rewards = self.get_sense_reward(sensor_concentrations, agent)
            pos_reward = self.get_sense_reward(pos_concentration, agent)
            slope = jnp.expand_dims(offsets, 1) * jnp.expand_dims(sensor_rewards - pos_reward, 2) / self.sensor_distance
            reward = (slope * position).sum()
            return reward
        
        def motion_f(points, agents):
            grad = jax.grad(lambda pos, agent : fields_f(pos, agent))
            return -jax.vmap(grad)(points, agents)

        force = motion_f(positions, agents)
        # Move agents
        # compute forces along sensors and concentrations
        # forces = jnp.expand_dims(rewards, 2) * jnp.expand_dims(jnp.expand_dims(offsets, 1), -1)
        # # get means along forces and concentrations
        # force = jnp.mean(jnp.mean(forces, axis=0), axis=0).T
        # force += 0.1 * jax.random.uniform(self.random_key, force.shape, minval=-1.0, maxval=1.0, dtype=jnp.float32)
        new_velocity = (1 - self.force_factor) * velocity + self.force_factor * force
        new_positions = (positions + velocity) % self.grid_size

        agents = agents.at[:, self.alookup['positions']].set(new_positions)
        agents = agents.at[:, self.alookup['velocity']].set(new_velocity)

        return agents

class ReactionDiffusionPhysarum(Physarum):
    def __init__(self, kill_rate=0.0545, feed_rate=0.0367, diffusion_rate=0.16):
        self.kill_rate = kill_rate
        self.feed_rate = feed_rate
        self.diffusion_rate = diffusion_rate

        super().__init__(num_chemicals=2)

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

class ParticleLife(ParticleSystem):
    def __init__(self, random_key, num_species=5, beta=0.3, **kwargs):
        if 'num_chemicals' in kwargs:
            del kwargs['num_chemicals']
        self.beta = beta
        self.chemical_factors = jax.random.uniform(random_key, (num_species, num_species), minval=-1.0, maxval=1.0)
        super().__init__(random_key, num_species=num_species, num_chemicals=num_species, **kwargs)

    def get_sense_reward(self, senses, agents):
        # get species
        species = jnp.round(agents[..., self.alookup['species']]).astype(int)
        # look up species chemical mapping
        chemical_factors = self.chemical_factors[species].T
        # return senses * chemical_factors
        reward = jnp.where(
            senses > self.beta,
            (senses - self.beta) / (self.beta - 1.),
            senses
        )
        reward = jnp.where(
            jnp.logical_and(senses < self.beta, senses > self.beta / 2.),
            (chemical_factors * 2.0) * (1 - senses / self.beta),
            reward
        )
        reward = jnp.where(
            senses < self.beta / 2.,
            (chemical_factors * 2.0) * (senses / self.beta),
            reward
        )
        return reward
    
    def deposit(self, chemical_grid, quantized_positions, agents):
        species = jnp.round(agents[:, self.alookup['species']]).astype(int)
        return chemical_grid.at[species, quantized_positions[:, 0], quantized_positions[:, 1]].add(self.deposit_amount).clip(0, 1)

class Lenia(DSCL):
    def __init__(self, *args, num_sensors=8, sensor_distance=5, **kwargs):
        
        self.num_sensors = num_sensors
        self.sensor_distance = sensor_distance

        super().__init__(*args, **kwargs)

    def initialize_agents(self):
        positions = jax.random.uniform(self.random_key, (self.num_agents, 2), minval=0, maxval=self.grid_size, dtype=jnp.float32)
        species = jax.random.randint(self.random_key, (self.num_agents,1), minval=0, maxval=self.num_species, dtype=jnp.int8)
        max_sense = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        phase = jax.random.uniform(self.random_key, (self.num_agents, 1), minval=0, maxval=2*jnp.pi, dtype=jnp.float32)
        cols = [positions, species, max_sense, phase]
        col_names = ['positions', 'species', 'max_sense', 'phase']
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

    def move_agents(self, agents, chemical_grid):
        positions = agents[:, self.alookup['positions']]
        # Calculate sensor positions: around the particle
        angles = jnp.linspace(0., 2 * jnp.pi, self.num_sensors + 1)[:-1]
        offsets = self.sensor_distance * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
        Fields = namedtuple('Fields', 'U G R E')

        def peak_f(x, mu, sigma):
            return jnp.exp(-((x-mu)/sigma)**2)

        def fields_f(p: Params, x):
            sensors = jnp.round((x + jnp.expand_dims(offsets, 1)) % self.grid_size)
            # TODO reformulate r to be a differentiable function of x using bilinear interpolation
            r = chemical_grid[:, sensors[..., 0].astype(jnp.int32), sensors[..., 1].astype(jnp.int32)]
            U = peak_f(r, p.mu_k, p.sigma_k).sum()*p.w_k
            G = peak_f(U, p.mu_g, p.sigma_g)
            R = p.c_rep/2 * ((1.0-r).clip(0.0)**2).sum()
            return Fields(U, G, R, E=R-G)

        def motion_f(params, points):
            grad_E = jax.grad(lambda x : fields_f(params, x).E)
            return -jax.vmap(grad_E)(points)

        # Move agents
        dt = 0.1
        params = Params(mu_k=0.0, sigma_k=0.1, w_k=1.0, mu_g=0.5, sigma_g=0.1, c_rep=1.0)
        force = motion_f(params, positions)
        new_positions = (positions + dt * force) % self.grid_size

        agents = agents.at[:, self.alookup['positions']].set(new_positions)

        return agents

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
        random_key = random.PRNGKey(1)

        self.record = True

        # Audio setup
        # self.default_speaker = sc.default_speaker()
        # self.fs = 44100  # Sample rate
        # self.blocksize = 16000
        # self.num_channels = self.default_speaker.channels
        self.phase = 0  # To keep track of phase between updates

        # Initialize simulation
        jit = True
        num_agents = 10000
        num_species = 1
        num_chemicals = 1
        grid_size = 1000
        speed = 1.0
        decay_rate = 0.01
        deposit_amount = 1.0
        sensor_angle = jnp.pi / 3
        sensor_distance = 3
        kwargs = {}
        physarum_type = 'particle_life'
        if physarum_type == 'physarum':
            simulation_cls = Physarum
        elif physarum_type == 'particle_life':
            num_species = 10
            kwargs['beta'] = 0.5
            simulation_cls = ParticleLife
        elif physarum_type == 'particle_system':
            simulation_cls = ParticleSystem
        elif physarum_type == 'lenia':
            simulation_cls = Lenia
        self.simulation = simulation_cls(
            random_key,
            jit=jit,
            num_agents=num_agents, 
            grid_size=grid_size, 
            num_species=num_species, 
            num_chemicals=num_chemicals, 
            speed=speed, 
            decay_rate=decay_rate, 
            deposit_amount=deposit_amount, 
            sensor_angle=sensor_angle, 
            sensor_distance=sensor_distance,
            **kwargs
        )

        self.canvas = scene.SceneCanvas(keys='interactive', size=(grid_size, grid_size))
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

        do_audio = False
        if do_audio:
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
        if hasattr(self, 'audio_timer'):
            self.audio_timer.close()
        self.canvas.close()

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