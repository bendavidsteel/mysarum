# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This flocking task is based on the following colab notebook:
https://github.com/google/jax-md/blob/main/notebooks/flocking.ipynb
"""


from functools import partial
from typing import Dict, Tuple

from flax.struct import dataclass
import jax
from jax import vmap
import jax.numpy as jnp
from jax.numpy import ndarray
import jax.scipy as jsp
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from evojax.task.base import VectorizedTask, BDExtractor
from evojax.task.base import TaskState

import rewards

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


def sample_position(key: jnp.ndarray, n: jnp.ndarray, map_size, n_dim) -> jnp.ndarray:
    return jax.random.uniform(key, shape=(n, n_dim), minval=0.0, maxval=map_size)


def unpack_act(action: jnp.ndarray) -> jnp.ndarray:
    d_theta, acceleration, desposit_amount = action[..., 0], action[..., 1], action[..., 2]
    return d_theta, acceleration, desposit_amount


def pack_state(position: jnp.ndarray, species: jnp.ndarray) -> jnp.ndarray:
    # flatten to 1d array to allow packing
    position = position.reshape((-1,))
    species = species.reshape((-1,))
    return jnp.concatenate([position, species])

def unpack_state(state: jnp.ndarray, num_particles: int, n_dims: int) -> jnp.ndarray:
    position = state[..., :num_particles * n_dims].reshape((num_particles, n_dims))
    species = state[..., num_particles * n_dims:].reshape((num_particles,)).astype(jnp.int32)
    return position, species


def pack_obs(state: jnp.ndarray, num_agents, sense_dist, map_size) -> jnp.ndarray:
    positions, directions, concentrations = unpack_state(state, num_agents, map_size)
    # calculation concentrations around each agent
    def get_concentration(position):
        pos = jnp.round(position).astype(int)
        return jax.lax.dynamic_slice(concentrations, (pos[0] - sense_dist, pos[1] - sense_dist), (2 * sense_dist, 2 * sense_dist))
    concentration_region = vmap(get_concentration)(positions)
    obs = jnp.concatenate([concentration_region.reshape((num_agents, -1)), directions.reshape((num_agents, 1))], axis=-1)
    return obs


def normal(theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)


def update_state(state, action, num_steps, num_particles, n_dims):
    # all_positions, species = unpack_state(action, num_particles, n_dims)
    return action.reshape(num_steps, num_particles, n_dims)

def get_reward(all_positions, max_steps: jnp.int32, reward_type, maximise_reward: bool, num_agents: jnp.int32, n_dims: jnp.int32):
    reward = rewards.multi_scale_fft_entropy(all_positions) + 0.5 * rewards.jitter(all_positions, order=4)

    reward = jax.lax.cond(
        maximise_reward,
        lambda x: x,
        lambda x: -x,
        reward)
    # reward = jax.lax.cond(
    #     True,
    #     lambda x: x,
    #     lambda x: x * (state.steps / max_steps) ** 2,
    #     reward)
    # reward = jax.lax.cond(
    #     state.steps >= max_steps - 5,
    #     lambda x: reward_func(x),
    #     lambda x: 0.0,
    #     concentrations
    # )
    return reward


def to_pillow_coordinate(position, width, height):
    # Fix the format for drawing with pillow.
    return jnp.stack([position[:, 0] * width,
                      (1.0 - position[:, 1]) * height], axis=1)


def rotate(px, py, cx, cy, angle):
    R = jnp.array([[jnp.cos(angle), jnp.sin(angle)],
                   [-jnp.sin(angle), jnp.cos(angle)]])
    u = jnp.array([px - cx, py - cy])
    x, y = jnp.dot(R, u) + jnp.array([cx, cy])
    return x, y

def render_single(positions, species, size):
    frame_scale = 10
    image = Image.new('RGB', (frame_scale * size, frame_scale * size), 'white')
    draw = ImageDraw.Draw(image)
    radius = frame_scale * size / 200
    colors = matplotlib.cm.get_cmap('tab10', len(species))
    if positions.shape[1] == 2:
        # Draw each particle
        for (x, y), specie in zip(positions, species):
            color = colors(specie)
            color = tuple(int(255 * c) for c in color[:3])
            
            # Draw circle
            x *= frame_scale
            y *= frame_scale
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=color, outline='black')
    
    elif positions.shape[1] == 3:
        # Sort particles by z-coordinate for proper depth rendering
        positions = [(x, y, z) for x, y, z in positions]
        sorted_particles = sorted(positions, key=lambda p: p[2], reverse=True)
        
        # Parameters for perspective projection
        focal_length = frame_scale * 500
        for x, y, z in sorted_particles:

            x *= frame_scale
            y *= frame_scale
            z *= frame_scale

            # Apply perspective projection
            scale = focal_length / (focal_length + z)
            proj_x = int(x * scale + (1 - scale) * size/2)
            proj_y = int(y * scale + (1 - scale) * size/2)
            proj_radius = int(radius * scale)
            
            # Calculate color based on depth
            intensity = int(255 * (1 - z/1000))  # Closer particles are brighter
            color = (intensity, intensity, intensity)
            
            # Draw circle with perspective
            draw.ellipse([proj_x-proj_radius, proj_y-proj_radius,
                            proj_x+proj_radius, proj_y+proj_radius],
                        fill=color, outline='black')
    else:
        raise ValueError("Position must have 2 or 3 dimensions")
        
    return image


class ParticleLeniaTask(VectorizedTask):

    def __init__(
            self,
            max_steps: int = 150,
            num_particles: int = 1000,
            map_size: int = 100,
            n_dims: int = 2,
            num_species: int = 1,
            reward_type: str = 'fft_entropy',
            maximise_reward: bool = True,
            jit=True
            ):
        self.max_steps = max_steps
        self.num_particles = num_particles
        self.map_size = map_size
        self.n_dims = n_dims
        self.obs_shape = tuple([0]) # we can save VRAM space by just using the state as the observation
        self.act_shape = tuple([num_particles, n_dims])

        def reset_fn(key):
            next_key, *subkeys = jax.random.split(key, num=4)
            position = sample_position(subkeys[0], num_particles, map_size, n_dims)
            species = jax.random.randint(subkeys[1], (num_particles,), 0, num_species, dtype=jnp.int32)
            state = pack_state(position, species)
            return State(obs=jnp.zeros(()),
                         state=state,
                         steps=jnp.zeros((), dtype=jnp.int32),
                         key=next_key)
        
        if jit:
            self._reset_fn = jax.jit(jax.vmap(reset_fn))
        else:
            self._reset_fn = jax.vmap(reset_fn)

        def step_fn(state, action):
            # new_state = update_state(state, action, num_particles, n_dims)
            all_positions = action.reshape(max_steps, num_particles, n_dims)
            new_state = jnp.zeros_like(state.state)
            new_obs = jnp.zeros(())
            new_steps = jnp.int32(state.steps + 1)
            new_steps = jnp.int32(state.steps + 1)
            next_key, _ = jax.random.split(state.key)
            reward = get_reward(all_positions, max_steps, reward_type, maximise_reward, num_particles, n_dims)
            done = True # one step
            return State(obs=new_obs,
                         state=new_state,
                         steps=new_steps,
                         key=next_key), reward, done
        
        if jit:
            self._step_fn = jax.jit(jax.vmap(step_fn))
        else:
            self._step_fn = jax.vmap(step_fn)

    def step(self, state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

    def reset(self, key: jnp.ndarray):
        return self._reset_fn(key)

    def render(self, state: State, task_id: int) -> Image:
        return render_single(state.state[task_id], self.num_particles, self.n_dims)
    

class PhysarumBDExtractor(BDExtractor):
    def init_state(self, extended_task_state):
        return {'state': extended_task_state}
    
    def update(self, extended_task_state, action: jax.Array, reward, done):
        return extended_task_state
    
    def summarize(self, extended_task_state):
        return super().summarize(extended_task_state)
