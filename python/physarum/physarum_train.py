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
import PIL
import tqdm

from evojax import Trainer, util
from evojax.algo import PGPE, ARS_native, MAPElites, CMA_ES_JAX
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.policy.mlp import MLP
from evojax.task.base import TaskState
from evojax.util import create_logger, get_params_format_fn

from physarum_task import PhysarumTask, unpack_state


class PhysarumPolicyNetwork(PolicyNetwork):
    def __init__(
            self, 
            num_agents, 
            map_size, 
            max_sensor_dist,
            method='mlp', 
            jit=True, 
            logger=None
        ):
        if logger is None:
            self._logger = create_logger(name='PhysarumPolicyNetwork')
        else:
            self._logger = logger

        self.num_agents = num_agents
        self.map_size = map_size
        self.max_sensor_dist = max_sensor_dist
        self.max_sensor_angle = jnp.pi / 8
        self.min_acceleration = -0.1
        self.max_acceleration = 0.1
        self.min_deposit = 0.0
        self.max_deposit = 5.0
        self.min_turn = -jnp.pi
        self.max_turn = jnp.pi

        param_dict = {}

        self.method = method
        if self.method == 'mlp':
            action_hidden_dims = [3]
            num_sensors = 4
            num_outputs = 3
            output_act_fn = 'tanh'
            self.action_model = MLP(feat_dims=action_hidden_dims, out_dim=num_outputs, out_fn=output_act_fn)
            action_params = self.action_model.init(random.PRNGKey(0), jnp.ones([1, num_sensors]))
            param_dict['action_mlp'] = action_params
            num_sensor_inputs = 1
            sensor_hidden_dims = [3]
            self.sensor_model = MLP(feat_dims=sensor_hidden_dims, out_dim=num_sensors, out_fn=output_act_fn)
            sensor_params = self.sensor_model.init(random.PRNGKey(0), jnp.ones([1, num_sensor_inputs]))
            param_dict['sensor_mlp'] = sensor_params
        else:
            param_dict['sensor_distance'] = random.uniform(random.PRNGKey(0), (1,), minval=0.1, maxval=self.max_sensor_dist)
            param_dict['sensor_angle'] = random.uniform(random.PRNGKey(1), (1,), minval=0.1, maxval=self.max_sensor_angle)

        params = FrozenDict(param_dict)

        self.num_params, format_params_fn = get_params_format_fn(params)
        self._format_params_fn = jax.vmap(format_params_fn)

        # Wrap the action computation in jax.vmap for vectorization
        self._get_actions_fn = jax.vmap(self._compute_actions)
        if jit:
            self._get_actions_fn = jax.jit(self._get_actions_fn)

    def _compute_actions(self, state, params):
        positions, speed, directions, concentration = unpack_state(state, self.num_agents, self.map_size)

        if self.method == 'mlp':
            sensor_params = params['sensor_mlp']
            sensor_inputs = jnp.stack([speed], axis=-1)
            sensor_specs = self.sensor_model.apply(sensor_params, sensor_inputs)

            sensor_specs = sensor_specs.reshape(sensor_inputs.shape[0], -1, 2)

            # get offsets
            sensor_dist = self.max_sensor_dist * sensor_specs[..., 0]
            sensor_dist = jnp.expand_dims(jnp.clip(sensor_dist, 0., self.max_sensor_dist), -1)
            # map 0 to 1, to 0 to pi
            sensor_angle = self.max_sensor_angle * sensor_specs[..., 1]
            sensor_angle = jnp.clip(sensor_angle, 0., self.max_sensor_angle)

            dirs = jnp.expand_dims(directions, -1)
            right_offsets = sensor_dist * jnp.stack([jnp.cos(dirs + sensor_angle), jnp.sin(dirs + sensor_angle)], axis=-1)
            left_offsets = sensor_dist * jnp.stack([jnp.cos(dirs - sensor_angle), jnp.sin(dirs - sensor_angle)], axis=-1)
            offsets = jnp.concatenate([right_offsets, left_offsets], axis=-2) 

        elif self.method == 'default':
            sensor_dist = (self.max_sensor_dist / 2) * (1 + params['sensor_distance'])
            sensor_dist = jnp.clip(sensor_dist, 0., self.max_sensor_dist)
            # map -1 to 1, to 0 to pi
            sensor_angle = (self.max_sensor_angle / 2) * (1 + params['sensor_angle']) 
            sensor_angle = jnp.clip(sensor_angle, 0., self.max_sensor_angle)

            front_offsets, left_offsets, right_offsets = self._get_offsets(directions, sensor_dist, sensor_angle)
            offsets = jnp.stack([front_offsets, left_offsets, right_offsets], axis=-1)

        # concentrations are centered on positions
        sensor_positions = positions.reshape(-1, 1, 2) + offsets
        sensor_positions = sensor_positions.astype(jnp.int32)

        def get_concentration(sensors):
            return concentration[sensors[0], sensors[1]]
        get_concentration = jax.vmap(jax.vmap(get_concentration))

        # front_concentration = get_concentration(front_sensors)
        # left_concentration = get_concentration(left_sensors)
        # right_concentration = get_concentration(right_sensors)
        concentrations = get_concentration(sensor_positions)

        if self.method == 'default':
            front_concentration = concentrations[..., 0]
            left_concentration = concentrations[..., 1]
            right_concentration = concentrations[..., 2]
            actions = self._default_concentration_action(front_concentration, left_concentration, right_concentration, sensor_angle)
        elif self.method == 'mlp':
            mlp_params = params['action_mlp']
            actions = self.action_model.apply(mlp_params, concentrations)
            # scale the actions to the correct range
            actions = (actions + 1) / 2
            actions *= jnp.array([
                self.max_turn - self.min_turn,
                self.max_acceleration - self.min_acceleration,
                self.max_deposit - self.min_deposit
            ])
            actions += jnp.array([self.min_turn, self.min_acceleration, self.min_deposit])
        else:
            raise ValueError(f"Unknown action method: {self.action_method}")

        return actions
    
    def _default_concentration_action(self, front_concentration, left_concentration, right_concentration, sensor_angle):
        steer_left = left_concentration > front_concentration
        steer_right = right_concentration > front_concentration
        d_theta = sensor_angle * steer_left - sensor_angle * steer_right

        speed = jnp.ones_like(d_theta)
        deposit = jnp.ones_like(d_theta)

        actions = jnp.stack([d_theta, speed, deposit], axis=-1)
        return actions

    def _get_offsets(self, directions, sensor_distance, sensor_angle):
        front_offsets = sensor_distance * jnp.stack([jnp.cos(directions), jnp.sin(directions)], axis=-1)
        left_offsets = sensor_distance * jnp.stack([jnp.cos(directions + sensor_angle), jnp.sin(directions + sensor_angle)], axis=-1)
        right_offsets = sensor_distance * jnp.stack([jnp.cos(directions - sensor_angle), jnp.sin(directions - sensor_angle)], axis=-1)
        return front_offsets, left_offsets, right_offsets

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        if self.num_params > 0:
            params = self._format_params_fn(params)

        actions = self._get_actions_fn(t_states.state, params)

        # Update policy state if necessary
        new_p_states = p_states

        return actions, new_p_states

def visualize(test_task, policy, best_params, file_path, max_steps=100, jit=True):

    if jit:
        task_reset_fn = jax.jit(test_task.reset)
        policy_reset_fn = jax.jit(policy.reset)
        step_fn = jax.jit(test_task.step)
        action_fn = jax.jit(policy.get_actions)
    else:
        task_reset_fn = test_task.reset
        policy_reset_fn = policy.reset
        step_fn = test_task.step
        action_fn = policy.get_actions

    best_params = best_params.reshape(1, *best_params.shape)
    best_params = best_params

    key = jax.random.PRNGKey(0)[None, :]

    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)  

    imgs = [test_task.render(task_state, 0)]

    for _ in tqdm.tqdm(range(max_steps)):
        action, policy_state = action_fn(task_state, best_params, policy_state)
        task_state, reward, done = step_fn(task_state, action)
        imgs.append(test_task.render(task_state, 0))

    imgs[0].save(file_path, save_all=True, append_images=imgs[1:], optimize=False, duration=40, loop=0)


def main():
    debug = False
    jit = True
    init_std = 0.6
    center_lr = 0.4
    std_lr = 0.4
    max_iter = 50
    log_interval = 10
    test_interval = 100
    num_tests = 1
    n_evaluations = 1
    n_repeats = 1
    pop_size = 128
    elite_ratio = 0.1
    seed = 42
    algo = 'cma'

    log_dir = './log/physarum'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='Physarum', log_dir=log_dir, debug=debug)

    logger.info('EvoJAX Physarum')
    logger.info('=' * 30)

    reward_type = 'random_circle_diff'
    maximise_reward = False

    max_steps = 500
    num_agents = 1000
    map_size = 100
    sense_dist = 20
    train_task = PhysarumTask(
        max_steps=max_steps, 
        num_agents=num_agents, 
        map_size=map_size,
        reward_type=reward_type,
        maximise_reward=maximise_reward,
        jit=jit
    )
    test_task = PhysarumTask(
        max_steps=max_steps, 
        num_agents=num_agents, 
        map_size=map_size,
        reward_type=reward_type,
        maximise_reward=maximise_reward,
        jit=jit
    )

    policy = PhysarumPolicyNetwork(
        num_agents, 
        map_size, 
        sense_dist,
        method='mlp',
        logger=logger, 
        jit=jit
    )

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
                elite_ratio=elite_ratio,
            )
        elif algo == 'mapelites':
            solver = MAPElites(
                param_size=policy.num_params,
                pop_size=pop_size,
                bd_extractor=None,
            )
        elif algo == 'cma':
            solver = CMA_ES_JAX(
                param_size=policy.num_params,
                pop_size=pop_size,
                seed=seed,
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
            n_evaluations=n_evaluations,
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
        best_params = trainer.solver.best_params[None, :]
    else:
        best_params = jnp.zeros((policy.num_params))
        # best_params = jnp.array([-0.2, 0.0])

    gif_file = os.path.join(log_dir, 'physarum.gif')
    visualize(test_task, policy, best_params, gif_file, jit=jit, max_steps=max_steps)
    logger.info('GIF saved to {}.'.format(gif_file))


if __name__ == '__main__':
    main()