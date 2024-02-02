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

from PIL import Image, ImageDraw
from functools import partial
from typing import Tuple

import jax
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jsp
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


def sample_position(key: jnp.ndarray, n: jnp.ndarray, map_size) -> jnp.ndarray:
    return jax.random.uniform(key, shape=(n, 2), minval=0.0, maxval=map_size)


def sample_theta(key: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    return jax.random.uniform(key, shape=(n, 1), maxval=2. * jnp.pi)


def unpack_act(action: jnp.ndarray) -> jnp.ndarray:
    d_theta, d_speed, desposit_amount = action[..., 0], action[..., 1], action[..., 2]
    return d_theta, d_speed, desposit_amount


def pack_state(position: jnp.ndarray, theta: jnp.ndarray, concentrations: jnp.ndarray) -> jnp.ndarray:
    # flatten to 1d array to allow packing
    position = position.reshape((-1,))
    theta = theta.reshape((-1,))
    concentrations = concentrations.reshape((-1,))
    return jnp.concatenate([position, theta, concentrations])

def unpack_state(state: jnp.ndarray, num_agents: int, map_size: int) -> jnp.ndarray:
    position = state[:num_agents * 2].reshape((num_agents, 2))
    theta = state[num_agents * 2:num_agents * 3].reshape((num_agents,))
    concentrations = state[num_agents * 3:].reshape((map_size, map_size))
    return position, theta, concentrations


def displacement(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    dR = p1 - p2
    return jnp.mod(dR + 0.5, 1) - 0.5


def map_product(displacement):
    return vmap(vmap(displacement, (0, None), 0), (None, 0), 0)


def calc_distance(dR: jnp.ndarray) -> jnp.ndarray:
    dr = jnp.sqrt(jnp.sum(dR**2, axis=-1))
    return dr


def select_xy(xy: jnp.ndarray, ix: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(xy, ix, axis=0)


_select_xy = jax.vmap(select_xy, in_axes=(None, 0))


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


def update_state(state, action, num_agents, decay_rate, map_size):
    position, theta, concentrations = unpack_state(state.state, num_agents, map_size)
    action = jnp.concatenate([action, jnp.ones_like(action)], axis=1)
    d_theta, speed, deposit_amount = unpack_act(action)
    new_theta = jnp.mod(theta + d_theta, 2 * jnp.pi)
    N = normal(new_theta)
    speed = speed.reshape((-1, 1))
    new_position = jnp.mod(position + speed * N, map_size)
    quantized_positions = jnp.round(position).astype(int)
    concentrations = concentrations.at[quantized_positions[:, 0], quantized_positions[:, 1]].add(deposit_amount).clip(0, 1)

    # Apply decay and diffusion
    # x = jnp.linspace(-3, 3, 3)
    # window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
    window = jnp.array([[0.0625, 0.125, 0.0625],
                        [0.125, 0.25, 0.125],
                        [0.0625, 0.125, 0.0625]], dtype=jnp.float16)
    concentrations = jsp.signal.convolve(concentrations, window, mode='same')
    concentrations = concentrations * (1 - decay_rate)

    new_state = pack_state(new_position, new_theta, concentrations)
    return new_state


def calc_entropy(concentrations: jnp.ndarray) -> jnp.ndarray:
    # calculate the multi-scale entropy of the image
    # progressively downsample the image and calculate the entropy
    # of each scale
    ent = 0
    for i in range(1, 4):
        # downsample the image
        window = jnp.array([[0.0625, 0.125, 0.0625],
                            [0.125, 0.25, 0.125],
                            [0.0625, 0.125, 0.0625]], dtype=jnp.float16)
        concentrations = jsp.signal.convolve(concentrations, window, mode='same')
        concentrations = concentrations[::2, ::2]

        # get the spectrum of the image
        spectrum = jnp.fft.fft2(concentrations)
        spectrum = jnp.abs(spectrum)
        spec_sum = spectrum.sum()
        spec_sum_d = jnp.where(spec_sum == 0.0, 1.0, spec_sum)
        spectrum = jnp.where(spec_sum == 0.0, 0.0, spectrum / spec_sum_d)
        spectrum = spectrum.flatten()

        # calculate the entropy
        epsilon = 1e-12
        ent += -(spectrum * jnp.log(spectrum + epsilon)).sum()

    return ent


def get_reward(state: State, max_steps: jnp.int32, reward_type: jnp.int32, num_agents: jnp.int32, map_size: jnp.int32):
    position, theta, concentrations = unpack_state(state.state, num_agents, map_size)
    reward = calc_entropy(concentrations)
    reward = jax.lax.cond(
        reward_type == 0,
        lambda x: -x,
        lambda x: -x * (state.steps / max_steps) ** 2,
        reward)
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

def render_single(obs_single, num_agents, map_size):
    _, _, concentration = unpack_state(obs_single, num_agents, map_size)
    return concentration


class PhysarumTask(VectorizedTask):

    def __init__(
            self,
            max_steps: int = 150,
            reward_type: int = 0,  # (0: as it is, 1: increase rewards for late step)
            action_type: int = 0,   # (0: theta, 1: theta/speed)
            num_agents: int = 1000,
            map_size: int = 100,
            sense_dist: int = 5,
            decay_rate: float = 0.1,
            jit=True
            ):
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.map_size = map_size
        self.obs_shape = tuple([num_agents, (2 * sense_dist) ** 2 + 1])
        self.act_shape = tuple([num_agents, 3])

        def reset_fn(key):
            next_key, key = jax.random.split(key)
            position = sample_position(key, num_agents, map_size)
            theta = sample_theta(key, num_agents)
            concentrations = jnp.zeros((map_size, map_size), dtype=jnp.float16)
            state = pack_state(position, theta, concentrations)
            return State(obs=pack_obs(state, num_agents, sense_dist, map_size),
                         state=state,
                         steps=jnp.zeros((), dtype=jnp.int32),
                         key=next_key)
        
        if jit:
            self._reset_fn = jax.jit(jax.vmap(reset_fn))
        else:
            self._reset_fn = jax.vmap(reset_fn)

        def step_fn(state, action):
            new_state = update_state(state, action, num_agents, decay_rate, map_size)
            new_obs = pack_obs(new_state, num_agents, sense_dist, map_size)
            new_steps = jnp.int32(state.steps + 1)
            next_key, _ = jax.random.split(state.key)
            reward = get_reward(state, max_steps, reward_type, num_agents, map_size)
            done = jnp.where(max_steps <= new_steps, True, False)
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
        return render_single(state.state[task_id], self.num_agents, self.map_size)
