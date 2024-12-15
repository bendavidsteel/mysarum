import functools
import multiprocessing
import os
import shutil
from typing import Tuple

import cv2
from flax import linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from evojax import Trainer, util
from evojax.algo import PGPE, ARS_native, MAPElites, CMA_ES_JAX
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.policy.mlp import MLP
from evojax.task.base import TaskState
from evojax.util import create_logger, get_params_format_fn

from particle_lenia_task import ParticleLeniaTask, unpack_state, render_single
from particle_lenia import step_f, fields_f, total_energy_f, Params

def scan_step(current_params, species, dt, positions, _):
    new_positions = step_f(current_params, positions, species, dt)
    return new_positions, new_positions

def compute_actions(state, params, num_particles, n_dims, dt, num_steps):
    positions, species = unpack_state(state, num_particles, n_dims)
    current_params = Params(**params)
    final_positions, all_positions = jax.lax.scan(functools.partial(scan_step, current_params, species, dt), positions, None, length=num_steps)
    return all_positions.reshape(-1)

def process_single_frame(idx, action, species, render_single, map_size):
    imtemp = render_single(action[idx], species, map_size)
    return cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR)

class ParticleLeniaPolicyNetwork(PolicyNetwork):
    def __init__(
            self, 
            key,
            num_particles=100,
            map_size=100,
            n_dims=2,
            max_steps=1000,
            jit=True, 
            logger=None
        ):
        if logger is None:
            self._logger = create_logger(name='ParticlePolicyNetwork')
        else:
            self._logger = logger

        key, *subkeys = jax.random.split(key, 10)
        self.num_particles = num_particles
        self.map_size = map_size
        self.n_dims = n_dims
        self.dt = 0.1
        self.max_steps = max_steps

        num_species = 1
        num_kernels = 1
        num_growth_funcs = 1
        mu_k = jax.random.uniform(subkeys[0], (num_species, num_species, num_kernels), minval=-5.0, maxval=5.0)
        sigma_k = jax.random.uniform(subkeys[1], (num_species, num_species, num_kernels), minval=0.5, maxval=2.0)

        mu_g = jax.random.uniform(subkeys[2], (num_species, num_species, num_growth_funcs), minval=-2.0, maxval=2.0)
        sigma_g = jax.random.uniform(subkeys[3], (num_species, num_species, num_growth_funcs), minval=0.01, maxval=1.0)

        w_k = jax.random.uniform(subkeys[4], (num_species, num_kernels), minval=-0.04, maxval=0.04)
        c_rep = jax.random.uniform(subkeys[5], (num_species, num_species), minval=0.1, maxval=1.0)

        param_dict = {
            'mu_k': mu_k,
            'sigma_k': sigma_k,
            'w_k': w_k,
            'mu_g': mu_g,
            'sigma_g': sigma_g,
            'c_rep': c_rep
        }

        params = FrozenDict(param_dict)

        self._step_f = jax.jit(step_f)

        self.num_params, format_params_fn = get_params_format_fn(params)
        self._format_params_fn = jax.vmap(format_params_fn)

        # Wrap the action computation in jax.vmap for vectorization
        self._get_actions_fn = jax.vmap(compute_actions, in_axes=(0, 0, None, None, None, None), out_axes=0)
        if jit:
            self._get_actions_fn = jax.jit(self._get_actions_fn, static_argnames=['num_particles', 'n_dims', 'dt', 'num_steps'])

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        if self.num_params > 0:
            params = self._format_params_fn(params)

        actions = self._get_actions_fn(t_states.state, params, self.num_particles, self.n_dims, self.dt, self.max_steps)

        # Update policy state if necessary
        new_p_states = p_states

        return actions, new_p_states

def visualize(key, test_task, policy, best_params, file_path, max_steps=100, jit=True):

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

    task_state = task_reset_fn(key[None, :])
    policy_state = policy_reset_fn(task_state)  

    initial_position, species = unpack_state(task_state.state, test_task.num_particles, test_task.n_dims)

    action, policy_state = action_fn(task_state, best_params, policy_state)
    action = action.reshape(max_steps, test_task.num_particles, test_task.n_dims)

    # Setup video writer
    fps=30
    frame_size = (10 * test_task.map_size, 10 * test_task.map_size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(file_path, fourcc, fps, frame_size)
    
    # Create partial function with fixed arguments
    process_frame = functools.partial(process_single_frame, 
                          action=np.array(action), 
                          species=np.array(species),
                          render_single=render_single,
                          map_size=test_task.map_size)
    
    # Process frames in parallel
    parallel = False
    if parallel:
        with multiprocessing.Pool() as pool:
            frames = list(tqdm.tqdm(
                pool.imap(process_frame, range(len(action))), 
                total=len(action)
            ))
    else:
        frames = []
        for idx in tqdm.tqdm(range(len(action))):
            frame = process_frame(idx)
            frames.append(frame)
    
    # Write frames to video
    for frame in frames:
        video.write(frame)
    
    video.release()


def main():
    debug = False
    jit = True
    use_for_loop = True
    init_std = 0.6
    center_lr = 0.4
    std_lr = 0.4
    max_iter = 100
    log_interval = 10
    test_interval = 100
    num_tests = 1
    n_evaluations = 1
    n_repeats = 1
    pop_size = 64
    elite_ratio = 0.1
    seed = 42
    algo = 'pgpe'

    key = jax.random.PRNGKey(seed)

    log_dir = './log/particle_lenia'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='Particle Lenia', log_dir=log_dir, debug=debug)

    logger.info('EvoJAX Particle Lenia')
    logger.info('=' * 30)

    reward_type = 'fft_entropy'
    maximise_reward = False

    max_steps = 1000
    num_particles = 100
    map_size = 100
    num_species = 1
    n_dims = 3
    train_task = ParticleLeniaTask(
        num_particles=num_particles,
        map_size=map_size,
        max_steps=max_steps, 
        n_dims=n_dims,
        num_species=num_species,
        reward_type=reward_type,
        maximise_reward=maximise_reward,
        jit=jit
    )
    test_task = ParticleLeniaTask(
        num_particles=num_particles,
        map_size=map_size,
        max_steps=max_steps, 
        n_dims=n_dims,
        num_species=num_species,
        reward_type=reward_type,
        maximise_reward=maximise_reward,
        jit=jit
    )

    key, subkey = jax.random.split(key)
    policy = ParticleLeniaPolicyNetwork(
        subkey,
        num_particles=num_particles,
        map_size=map_size,
        n_dims=n_dims,
        num_species=num_species,
        max_steps=max_steps,
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
            debug=debug,
            use_for_loop=use_for_loop,
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

    gif_file = os.path.join(log_dir, 'particle_lenia.mp4')
    key, subkey = jax.random.split(key)
    visualize(subkey, test_task, policy, best_params, gif_file, jit=jit, max_steps=max_steps)
    logger.info('MP4 saved to {}.'.format(gif_file))


if __name__ == '__main__':
    main()