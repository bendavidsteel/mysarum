import einops
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

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
    def __init__(self, random_key, jit=True, num_dims=2, num_agents=1000000, grid_size=1000, num_species=1, num_chemicals=1, speed=1.0, decay_rate=0.1, deposit_amount=5.0, sensor_angle=jnp.pi / 3, sensor_distance=5):
        self.random_key = random_key
        self.num_dims = num_dims
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_species = num_species
        self.num_chemicals = num_chemicals
        self.decay_rate = decay_rate
        self.deposit_amount = deposit_amount

        self.agents = self.initialize_agents()
        self.chemical_grid = self.initialize_chemical_grid()

        self.colours = jnp.array(simple_colour_wheel(self.num_chemicals))

        window = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
        # get outer product a num_dims times
        for _ in range(num_dims - 1):
            window = jnp.outer(window, window.T)
        self.window = einops.rearrange(window, '... -> 1 1 ...')

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
    
    def get_intensity(self):
        return np.array(self.chemical_grid)
    
    def display_sim(self, chemical_grid, colours):
        colours = jnp.expand_dims(chemical_grid, 1) * jnp.expand_dims(jnp.expand_dims(colours, -1), -1)
        colour_im = jnp.sum(colours, axis=0)
        max_intensity = 1.0
        return (255.0 * jnp.moveaxis(colour_im, 0, -1) / max_intensity).astype(np.uint8)
    
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
        return jnp.zeros((self.num_chemicals, *[self.grid_size for _ in range(self.num_dims)]), dtype=jnp.float32)

    def get_sense_reward(self, senses, agents):
        return senses
    
    def get_summed_sense_reward(self, reward):
        return reward

    def move_agents(self, agents, chemical_grid):
        raise NotImplementedError

    def deposit(self, chemical_grid, quantized_positions, agents):
        indexer = tuple([slice(None)] + [quantized_positions[:, i] for i in range(quantized_positions.shape[1])])
        return chemical_grid.at[indexer].add(self.deposit_amount).clip(0, 1)

    def update_chemical_grid(self, chemical_grid, agents):
        positions = agents[:, self.alookup['positions']]
        # Deposit chemical
        quantized_positions = jnp.round(positions).astype(int)
        chemical_grid = self.deposit(chemical_grid, quantized_positions, agents)

        # Apply decay and diffusion
        # x = jnp.linspace(-3, 3, 3)
        # window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
        
        # window = jnp.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]], dtype=jnp.float32)
        smooth_image = lax.conv(einops.rearrange(chemical_grid, 'c ... -> c 1 ...'),    # lhs = NCHW image tensor
                self.window, # rhs = OIHW conv kernel tensor
                tuple([1 for _ in range(self.num_dims)]),  # window strides
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
        positions = jax.random.uniform(self.random_key, (self.num_agents, self.num_dims), minval=0, maxval=self.grid_size, dtype=jnp.float32)
        velocity = jnp.zeros((self.num_agents, self.num_dims), dtype=jnp.float32)
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
            self.alookup[col_name] = slice(col_idx, col_idx+col_width)
            col_idx += col_width
        agents = jnp.concatenate(cols, axis=1)
        return agents

    def move_agents(self, agents, chemical_grid):
        positions = agents[:, self.alookup['positions']]
        velocity = agents[:, self.alookup['velocity']]
        # Calculate sensor positions: around the particle
        
        assert self.num_dims <= 2, "Yet to implement 3D sensor positions"
        if self.num_dims == 1:
            offsets = jnp.array([self.sensor_distance, -self.sensor_distance]).reshape(-1, 1)
        elif self.num_dims == 2:
            angles = jnp.linspace(0., 2 * jnp.pi, self.num_sensors + 1)[:-1]
            offsets = self.sensor_distance * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        def fields_f(position, agent):
            # TODO stack these operations for faster computation
            sensors = jnp.round((position + offsets) % self.grid_size).astype(jnp.int32)
            indexer = tuple([slice(None)] + [sensors[:, i] for i in range(sensors.shape[1])])
            sensor_concentrations = chemical_grid[indexer]
            sensor_concentrations = einops.rearrange(sensor_concentrations, 'c s -> s c')
            pos_sensor = jnp.round(position).astype(jnp.int32) % self.grid_size
            indexer = tuple([slice(None)] + [pos_sensor[i] for i in range(pos_sensor.shape[0])])
            pos_concentration = chemical_grid[indexer]

            # Sense the chemical concentration at each sensor
            sensor_rewards = self.get_sense_reward(sensor_concentrations, agent)
            pos_reward = self.get_sense_reward(pos_concentration, agent)
            slope = einops.rearrange(offsets / jnp.linalg.norm(offsets, axis=1, keepdims=True), 's d -> s 1 d') * einops.rearrange(sensor_rewards - pos_reward, 's c -> s c 1') / self.sensor_distance
            intercept = einops.rearrange(pos_reward, 'c -> c 1') - slope * position
            slope = jax.lax.stop_gradient(slope)
            intercept = jax.lax.stop_gradient(intercept)
            reward = (slope * position + intercept).sum()
            return reward
        
        def motion_f(points, agents):
            grad = jax.grad(lambda pos, agent : fields_f(pos, agent))
            return jax.vmap(grad)(points, agents)

        force = motion_f(positions, agents)
        # Move agents
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
            jnp.logical_and(senses <= self.beta, senses > self.beta / 2.),
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

def peak_f(x, mu, sigma):
    return jnp.exp(-((x-mu)/sigma)**2)

def repel_f(x):
    return x

class Lenia(ParticleSystem):
    def __init__(self, *args, num_sensors=8, sensor_distance=8, inner_sensor_gap=3, **kwargs):
        
        self.num_sensors = num_sensors
        self.outer_sensor_distance = sensor_distance
        self.inner_sensor_distance = sensor_distance - inner_sensor_gap

        self.mu_k = 0.01
        self.sigma_k = 0.05
        self.mu_g = 0.05
        self.sigma_g = 0.01
        self.c_rep = 0.01

        super().__init__(*args, **kwargs)
    
    def move_agents(self, agents, chemical_grid):
        positions = agents[:, self.alookup['positions']]
        velocity = agents[:, self.alookup['velocity']]
        # Calculate sensor positions: around the particle
        if self.num_dims == 1:
            directions = jnp.array([1.0, -1.0]).reshape(-1, 1)
        elif self.num_dims == 2:
            angles = jnp.linspace(0., 2 * jnp.pi, self.num_sensors + 1)[:-1]
            directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        outer_offsets = self.outer_sensor_distance * directions
        inner_offsets = self.inner_sensor_distance * directions
        epsilon = 1e-6

        def fields_f(position, agent):
            # TODO stack these operations for faster computation
            outer_sensors = jnp.round(position + outer_offsets).astype(jnp.int32) % self.grid_size
            indexer = tuple([slice(None)] + [outer_sensors[..., i] for i in range(outer_sensors.shape[-1])])
            outer_sensor_concentrations = chemical_grid[indexer]
            outer_sensor_concentrations = einops.rearrange(outer_sensor_concentrations, 'c s -> s c')
            inner_sensors = jnp.round(position + inner_offsets).astype(jnp.int32) % self.grid_size
            indexer = tuple([slice(None)] + [inner_sensors[..., i] for i in range(inner_sensors.shape[-1])])
            inner_sensor_concentrations = chemical_grid[indexer]

            # Sense the chemical concentration at each sensor
            outer_sensor_u = peak_f(outer_sensor_concentrations, self.mu_k, self.sigma_k)
            inner_sensor_u = peak_f(inner_sensor_concentrations, self.mu_k, self.sigma_k)
            slope_u = directions * (outer_sensor_u - inner_sensor_u) / (self.outer_sensor_distance - self.inner_sensor_distance)
            intercept_u = inner_sensor_u - slope_u * position
            slope_u = jax.lax.stop_gradient(slope_u)
            intercept_u = jax.lax.stop_gradient(intercept_u)
            u = (slope_u * (position + inner_offsets) + intercept_u).sum()

            # get the growth field
            g = peak_f(u, self.mu_g, self.sigma_g)

            # determine the repulsion field
            outer_sensor_r = repel_f(outer_sensor_concentrations)
            inner_sensor_r = repel_f(inner_sensor_concentrations)
            slope_r = directions * (outer_sensor_r - inner_sensor_r) / (self.outer_sensor_distance - self.inner_sensor_distance)
            intercept_r = inner_sensor_u - slope_r * position
            slope_r = jax.lax.stop_gradient(slope_r)
            intercept_r = jax.lax.stop_gradient(intercept_r)
            r = self.c_rep * (slope_r * (position + inner_offsets) + intercept_r).sum()
            return r - g
        
        def motion_f(points, agents):
            grad = jax.grad(lambda pos, agent : fields_f(pos, agent))
            return jax.vmap(grad)(points, agents)

        force = motion_f(positions, agents)
        # Move agents
        new_velocity = (1 - self.force_factor) * velocity + self.force_factor * force
        new_positions = (positions + velocity) % self.grid_size

        agents = agents.at[:, self.alookup['positions']].set(new_positions)
        agents = agents.at[:, self.alookup['velocity']].set(new_velocity)

        return agents