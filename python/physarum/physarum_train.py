import argparse
import logging
import os
import shutil
from typing import Tuple


from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from evojax import Trainer, util
from evojax.algo import PGPE
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger, get_params_format_fn

from physarum_task import PhysarumTask


class PhysarumPolicyNetwork(PolicyNetwork):
    def __init__(self, num_agents, sensor_distance, sensor_angle, patch_size, jit=True, logger=None):
        if logger is None:
            self._logger = create_logger(name='PhysarumPolicyNetwork')
        else:
            self._logger = logger

        self.num_agents = num_agents
        self.sensor_distance = sensor_distance
        self.sensor_angle = sensor_angle
        self.patch_size = patch_size

        # Assuming we don't have trainable parameters for this policy
        # Otherwise, initialize and format them here
        self.num_params = 0

        # If there are parameters, replace None with the appropriate function
        self._format_params_fn = None

        # Wrap the action computation in jax.vmap for vectorization
        if jit:
            self._get_actions_fn = jax.jit(jax.vmap(self._compute_actions))
        else:
            self._get_actions_fn = jax.vmap(self._compute_actions)

    def _compute_actions(self, concentrations, directions):
        front_offsets, left_offsets, right_offsets = self._get_offsets(directions)

        # concentrations are centered on positions
        positions = jnp.stack([self.patch_size / 2, self.patch_size / 2], axis=-1)
        front_sensors = positions + front_offsets
        left_sensors = positions + left_offsets
        right_sensors = positions + right_offsets

        front_concentration = concentrations[front_sensors[:, 0].astype(jnp.int32), front_sensors[:, 1].astype(jnp.int32)]
        left_concentration = concentrations[left_sensors[:, 0].astype(jnp.int32), left_sensors[:, 1].astype(jnp.int32)]
        right_concentration = concentrations[right_sensors[:, 0].astype(jnp.int32), right_sensors[:, 1].astype(jnp.int32)]

        steer_left = left_concentration > front_concentration
        steer_right = right_concentration > front_concentration
        d_theta = self.sensor_angle * steer_left - self.sensor_angle * steer_right

        speed = jnp.ones_like(d_theta)
        deposit = jnp.ones_like(d_theta)

        actions = jnp.concatenate([d_theta, speed, deposit])

        return actions

    def _get_offsets(self, directions):
        front_offsets = self.sensor_distance * jnp.stack([jnp.cos(directions), jnp.sin(directions)], axis=-1)
        left_offsets = self.sensor_distance * jnp.stack([jnp.cos(directions + self.sensor_angle), jnp.sin(directions + self.sensor_angle)], axis=-1)
        right_offsets = self.sensor_distance * jnp.stack([jnp.cos(directions - self.sensor_angle), jnp.sin(directions - self.sensor_angle)], axis=-1)
        return front_offsets, left_offsets, right_offsets

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        if self._format_params_fn is not None:
            params = self._format_params_fn(params)

        concentrations, directions = t_states.obs[:, :-1], t_states.obs[:, -1:]
        concentrations = jnp.reshape(concentrations, (self.num_agents, self.patch_size, self.patch_size))

        actions = self._get_actions_fn(concentrations, directions)

        # Update policy state if necessary
        new_p_states = p_states

        return actions, new_p_states

class PhysarumVisualize:
    def __init__(self, test_task, policy, best_params, jit=True):
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

        self.best_params = best_params

        key = jax.random.PRNGKey(0)[None, :]

        self.task_state = task_reset_fn(key)
        self.policy_state = policy_reset_fn(self.task_state)  

        self.fig, ax = plt.subplots(figsize=(10, 10))
        self.chemical_image = ax.imshow(test_task.render(self.task_state, 0), cmap='viridis', origin='lower', vmin=0., vmax=1.)

    def update(self, t):
        num_tasks, num_agents = self.task_state.obs.shape[:2]
        self.task_state = self.task_state.replace(
            obs=self.task_state.obs.reshape((-1, *self.task_state.obs.shape[2:])))
        action, self.policy_state = self.action_fn(self.task_state, self.best_params, self.policy_state)
        action = action.reshape(num_tasks, num_agents, *action.shape[1:])
        self.task_state = self.task_state.replace(
            obs=self.task_state.obs.reshape(
                num_tasks, num_agents, *self.task_state.obs.shape[1:]))
        self.task_state, reward, done = self.step_fn(self.task_state, action)
        self.chemical_image.set_data(self.test_task.render(self.task_state, 0))
        return self.chemical_image,

    def draw(self, file_path):
        ani = FuncAnimation(self.fig, self.update, frames=100, blit=True)
        ani.save(file_path)


def main():
    debug = False
    jit = True
    init_std = 0.095
    center_lr = 0.011
    std_lr = 0.054
    max_iter = 0
    log_interval = 10
    test_interval = 100
    num_tests = 100
    n_repeats = 64
    seed = 42

    log_dir = './log/physarum'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='Physarum', log_dir=log_dir, debug=debug)

    logger.info('EvoJAX Physarum')
    logger.info('=' * 30)

    max_steps = 500
    num_agents = 1000
    map_size = 200
    sense_dist = 5
    train_task = PhysarumTask(max_steps=max_steps, num_agents=num_agents, map_size=map_size, sense_dist=sense_dist, jit=jit)
    test_task = PhysarumTask(max_steps=max_steps, num_agents=num_agents, map_size=map_size, sense_dist=sense_dist, jit=jit)
    policy = PhysarumPolicyNetwork(num_agents, sensor_distance=5, sensor_angle=jnp.pi / 3, patch_size=sense_dist * 2, logger=logger, jit=jit)

    if max_iter > 0:
        solver = PGPE(
            pop_size=4,
            param_size=policy.num_params,
            optimizer='adam',
            center_learning_rate=center_lr,
            stdev_learning_rate=std_lr,
            init_stdev=init_std,
            logger=logger,
            seed=seed,
        )

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
        best_params = jnp.repeat(
            trainer.solver.best_params[None, :], num_agents, axis=0)
    else:
        best_params = jnp.zeros((num_agents, policy.num_params))

    physarum = PhysarumVisualize(test_task, policy, best_params, jit=jit)

    gif_file = os.path.join(log_dir, 'physarum.gif')
    physarum.draw(gif_file)
    logger.info('GIF saved to {}.'.format(gif_file))


if __name__ == '__main__':
    main()