import argparse
import logging
import os
import shutil
from typing import Tuple


from flax import linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm

from evojax import Trainer, util
from evojax.algo import PGPE, ARS_native
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger, get_params_format_fn

from physarum_task import PhysarumTask


class PhysarumPolicyNetwork(PolicyNetwork):
    def __init__(self, num_agents, patch_size, jit=True, logger=None):
        if logger is None:
            self._logger = create_logger(name='PhysarumPolicyNetwork')
        else:
            self._logger = logger

        self.num_agents = num_agents
        self.patch_size = patch_size

        params = FrozenDict({
            'sensor_distance': random.uniform(random.PRNGKey(0), (1,), minval=0.1, maxval=10.),
            'sensor_angle': random.uniform(random.PRNGKey(1), (1,), minval=0.1, maxval=jnp.pi / 2),
        })

        self.num_params, format_params_fn = get_params_format_fn(params)
        self._format_params_fn = jax.vmap(format_params_fn)

        # Wrap the action computation in jax.vmap for vectorization
        if jit:
            self._get_actions_fn = jax.jit(jax.vmap(self._compute_actions))
        else:
            self._get_actions_fn = jax.vmap(self._compute_actions)

    def _compute_actions(self, concentrations, directions, params):
        front_offsets, left_offsets, right_offsets = self._get_offsets(directions, params)

        # concentrations are centered on positions
        positions = jnp.stack([self.patch_size / 2, self.patch_size / 2], axis=-1)
        front_sensors = positions + front_offsets
        left_sensors = positions + left_offsets
        right_sensors = positions + right_offsets

        def get_concentration(concentrations, sensors):
            return concentrations[sensors[0].astype(jnp.int32), sensors[1].astype(jnp.int32)]
        
        get_concentration = jax.vmap(get_concentration)
        front_concentration = get_concentration(concentrations, front_sensors)
        left_concentration = get_concentration(concentrations, left_sensors)
        right_concentration = get_concentration(concentrations, right_sensors)

        sensor_angle = params['sensor_angle']
        steer_left = left_concentration > front_concentration
        steer_right = right_concentration > front_concentration
        d_theta = sensor_angle * steer_left - sensor_angle * steer_right

        speed = jnp.ones_like(d_theta)
        deposit = jnp.ones_like(d_theta)

        actions = jnp.stack([d_theta, speed, deposit], axis=-1)

        return actions

    def _get_offsets(self, directions, params):
        sensor_distance = params['sensor_distance']
        sensor_angle = params['sensor_angle']
        front_offsets = sensor_distance * jnp.concatenate([jnp.cos(directions), jnp.sin(directions)], axis=-1)
        left_offsets = sensor_distance * jnp.concatenate([jnp.cos(directions + sensor_angle), jnp.sin(directions + sensor_angle)], axis=-1)
        right_offsets = sensor_distance * jnp.concatenate([jnp.cos(directions - sensor_angle), jnp.sin(directions - sensor_angle)], axis=-1)
        return front_offsets, left_offsets, right_offsets

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        if self.num_params > 0:
            params = self._format_params_fn(params)

        concentrations, directions = t_states.obs[..., :-1], t_states.obs[..., -1:]
        concentrations = jnp.reshape(concentrations, (concentrations.shape[:-1] + (self.patch_size, self.patch_size)))

        actions = self._get_actions_fn(concentrations, directions, params)

        # Update policy state if necessary
        new_p_states = p_states

        return actions, new_p_states

class PhysarumVisualize:
    def __init__(self, test_task, policy, best_params, max_steps=100, jit=True):
        self.max_steps = max_steps
        self.test_task = test_task

        if jit:
            task_reset_fn = jax.jit(test_task.reset)
            policy_reset_fn = jax.jit(policy.reset)
            self.step_fn = jax.jit(test_task.step)
            self.action_fn = jax.jit(policy.get_actions)
        else:
            task_reset_fn = test_task.reset
            policy_reset_fn = policy.reset
            self.step_fn = test_task.step
            self.action_fn = policy.get_actions

        best_params = best_params.reshape(1, *best_params.shape)
        self.best_params = best_params

        key = jax.random.PRNGKey(0)[None, :]

        self.task_state = task_reset_fn(key)
        self.policy_state = policy_reset_fn(self.task_state)  

        self.fig, ax = plt.subplots(figsize=(10, 10))
        self.chemical_image = ax.imshow(test_task.render(self.task_state, 0), cmap='viridis', origin='lower', vmin=0., vmax=1.)

    def update(self, t):
        assert self.task_state.obs.ndim == 3
        num_tasks, num_agents, num_obs = self.task_state.obs.shape
        action, self.policy_state = self.action_fn(self.task_state, self.best_params, self.policy_state)
        self.task_state, reward, done = self.step_fn(self.task_state, action)
        self.chemical_image.set_data(self.test_task.render(self.task_state, 0))
        return self.chemical_image,

    def draw(self, file_path):
        ani = FuncAnimation(self.fig, self.update, frames=tqdm.tqdm(range(self.max_steps)), interval=25, blit=True)
        ani.save(file_path)


def main():
    debug = False
    jit = True
    init_std = 0.095
    center_lr = 0.011
    std_lr = 0.054
    max_iter = 100
    log_interval = 10
    test_interval = 100
    num_tests = 100
    n_repeats = 1
    pop_size = 4
    seed = 42
    algo = 'ars'

    log_dir = './log/physarum'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='Physarum', log_dir=log_dir, debug=debug)

    logger.info('EvoJAX Physarum')
    logger.info('=' * 30)

    max_steps = 1000
    num_agents = 500
    map_size = 100
    sense_dist = 5
    train_task = PhysarumTask(max_steps=max_steps, num_agents=num_agents, map_size=map_size, sense_dist=sense_dist, jit=jit)
    test_task = PhysarumTask(max_steps=max_steps, num_agents=num_agents, map_size=map_size, sense_dist=sense_dist, jit=jit)
    policy = PhysarumPolicyNetwork(num_agents, patch_size=sense_dist * 2, logger=logger, jit=jit)

    if max_iter > 0 and policy.num_params > 0:
            
        if algo == 'pgpe':
            solver = PGPE(
                pop_size=pop_size,
                param_size=policy.num_params,
                optimizer='adam',
                center_learning_rate=center_lr,
                stdev_learning_rate=std_lr,
                init_stdev=init_std,
                logger=logger,
                seed=seed,
            )
        elif algo == 'ars':
            solver = ARS_native(
                param_size=policy.num_params,
                pop_size=pop_size,
                elite_ratio=0.1,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}.")

        # Train.
        trainer = Trainer(
            policy=policy,
            solver=solver,
            train_task=train_task,
            test_task=test_task,
            max_iter=max_iter,
            log_interval=log_interval,
            test_interval=test_interval,
            n_evaluations=10,
            n_repeats=n_repeats,
            use_for_loop=True,
            test_n_repeats=num_tests,
            seed=seed,
            log_dir=log_dir,
            logger=logger,
        )
        trainer.run(demo_mode=False)

        # Test the final model.
        src_file = os.path.join(log_dir, 'best.npz')
        tar_file = os.path.join(log_dir, 'model.npz')
        shutil.copy(src_file, tar_file)
        trainer.model_dir = log_dir
        trainer.run(demo_mode=True)

    # Visualize the policy.
    if max_iter > 0:
        best_params = trainer.solver.best_params[None, :]
    else:
        best_params = jnp.zeros((policy.num_params))

    physarum = PhysarumVisualize(test_task, policy, best_params, jit=jit, max_steps=max_steps)

    gif_file = os.path.join(log_dir, 'physarum.gif')
    physarum.draw(gif_file)
    logger.info('GIF saved to {}.'.format(gif_file))


if __name__ == '__main__':
    main()