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

from PIL import Image
from functools import partial
from typing import Dict, Tuple

from flax.struct import dataclass
import jax
from jax import vmap
import jax.numpy as jnp
from jax.numpy import ndarray
import jax.scipy as jsp
import numpy as np

from evojax.task.base import VectorizedTask, BDExtractor
from evojax.task.base import TaskState

import rewards

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
    d_theta, acceleration, desposit_amount = action[..., 0], action[..., 1], action[..., 2]
    return d_theta, acceleration, desposit_amount


def pack_state(position: jnp.ndarray, speed: jnp.ndarray, theta: jnp.ndarray, concentrations: jnp.ndarray) -> jnp.ndarray:
    # flatten to 1d array to allow packing
    position = position.reshape((-1,))
    speed = speed.reshape((-1,))
    theta = theta.reshape((-1,))
    concentrations = concentrations.reshape((-1,))
    return jnp.concatenate([position, speed, theta, concentrations])

def unpack_state(state: jnp.ndarray, num_agents: int, map_size: int) -> jnp.ndarray:
    position = state[..., :num_agents * 2].reshape((num_agents, 2))
    speed = state[..., num_agents * 2:num_agents * 3].reshape((num_agents,))
    theta = state[..., num_agents * 3:num_agents * 4].reshape((num_agents,))
    concentration = state[..., num_agents * 4:].reshape((map_size, map_size))
    return position, speed, theta, concentration


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
    position, speed, theta, concentrations = unpack_state(state.state, num_agents, map_size)
    action = jnp.concatenate([action, jnp.ones_like(action)], axis=1)
    d_theta, acceleration, deposit_amount = unpack_act(action)
    new_theta = jnp.mod(theta + d_theta, 2 * jnp.pi)
    unit_vel = normal(new_theta)
    new_speed = speed + acceleration
    new_speed = new_speed.reshape((-1, 1)).clip(0.5, 1.5)
    new_position = jnp.mod(position + new_speed * unit_vel, map_size)
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

    new_state = pack_state(new_position, new_speed, new_theta, concentrations)
    return new_state

def get_reward(state: State, max_steps: jnp.int32, reward_type, maximise_reward: bool, num_agents: jnp.int32, map_size: jnp.int32):
    _, _, _, concentrations = unpack_state(state.state, num_agents, map_size)
    if reward_type == 'mse':
        reward = rewards.get_multiscale_entropy(concentrations)
    elif reward_type == 'energy':
        reward = rewards.get_energy(concentrations)
    elif reward_type == 'diff_sine':
        reward = rewards.get_diff_sine(concentrations, state.steps)
    elif reward_type == 'random_circle_diff':
        reward = rewards.get_random_circle_diff(concentrations, random_key=state.key)

    reward = jax.lax.cond(
        maximise_reward,
        lambda x: x,
        lambda x: -x,
        reward)
    reward = jax.lax.cond(
        True,
        lambda x: x,
        lambda x: x * (state.steps / max_steps) ** 2,
        reward)
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

def render_single(obs_single, num_agents, map_size):
    _, _, _, concentration = unpack_state(obs_single, num_agents, map_size)
    image = (np.array(concentration) * 255).astype(np.uint8)
    pillow_image = Image.fromarray(image, mode='L')
    return pillow_image


class PhysarumTask(VectorizedTask):

    def __init__(
            self,
            max_steps: int = 150,
            num_agents: int = 1000,
            map_size: int = 100,
            decay_rate: float = 0.1,
            reward_type: str = 'mse',
            maximise_reward: bool = True,
            jit=True
            ):
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.map_size = map_size
        self.obs_shape = tuple([0]) # we can save VRAM space by just using the state as the observation
        self.act_shape = tuple([num_agents, 3])

        def reset_fn(key):
            next_key, key = jax.random.split(key)
            position = sample_position(key, num_agents, map_size)
            speed = jnp.ones((num_agents,), dtype=jnp.float16)
            theta = sample_theta(key, num_agents)
            concentrations = jnp.zeros((map_size, map_size), dtype=jnp.float16)
            state = pack_state(position, speed, theta, concentrations)
            return State(obs=jnp.zeros(()),
                         state=state,
                         steps=jnp.zeros((), dtype=jnp.int32),
                         key=next_key)
        
        if jit:
            self._reset_fn = jax.jit(jax.vmap(reset_fn))
        else:
            self._reset_fn = jax.vmap(reset_fn)

        def step_fn(state, action):
            new_state = update_state(state, action, num_agents, decay_rate, map_size)
            new_obs = jnp.zeros(())
            new_steps = jnp.int32(state.steps + 1)
            next_key, _ = jax.random.split(state.key)
            reward = get_reward(state, max_steps, reward_type, maximise_reward, num_agents, map_size)
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
    

class PhysarumBDExtractor(BDExtractor):
    def init_state(self, extended_task_state):
        return {'state': extended_task_state}
    
    def update(self, extended_task_state, action: jax.Array, reward, done):
        return extended_task_state
    
    def summarize(self, extended_task_state):
        return super().summarize(extended_task_state)
