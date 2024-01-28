import argparse
import os
import shutil
import jax
import jax.numpy as jnp

from evojax.task import flocking
from evojax.policy.mlp import MLPPolicy
from evojax.algo import PGPE
from evojax import Trainer
from evojax import util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hidden-size', type=int, default=100, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=64, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=1000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=100, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr', type=float, default=0.011, help='Center learning rate.')
    parser.add_argument(
        '--std-lr', type=float, default=0.054, help='Std learning rate.')
    parser.add_argument(
        '--init-std', type=float, default=0.095, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/flocking'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='Flocking', log_dir=log_dir, debug=config.debug)

    logger.info('EvoJAX Flocking')
    logger.info('=' * 30)


    seed = 42
    neighbor_num = 5

    rollout_key = jax.random.PRNGKey(seed=seed)

    reset_key, rollout_key = jax.random.split(rollout_key, 2)
    reset_key = reset_key[None, :] 

    train_task = flocking.FlockingTask(150)
    test_task = flocking.FlockingTask(150)

    policy = MLPPolicy(
        input_dim=neighbor_num*3,
        hidden_dims=[60, 60],
        output_dim=1,
        logger=logger,
    )

    solver = PGPE(
        pop_size=64,
        param_size=policy.num_params,
        optimizer='adam',
        center_learning_rate=0.05,
        seed=seed,
    )

    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=150,
        log_interval=10,
        test_interval=30,
        n_repeats=5,
        n_evaluations=10,
        seed=seed,
        log_dir=log_dir,
        logger=logger,
    )
    _ = trainer.run()


    def render(task, algo, policy):
        """Render the learned policy."""

        task_reset_fn = jax.jit(task.reset)
        policy_reset_fn = jax.jit(policy.reset)
        step_fn = jax.jit(task.step)
        act_fn = jax.jit(policy.get_actions)

        params = algo.best_params[None, :]
        task_s = task_reset_fn(jax.random.PRNGKey(seed=seed)[None, :])
        policy_s = policy_reset_fn(task_s)

        images = [flocking.FlockingTask.render(task_s, 0)]
        done = False
        step = 0
        reward = 0
        while not done:
            act, policy_s = act_fn(task_s, params, policy_s)
            task_s, r, done = step_fn(task_s, act)
            step += 1
            reward = reward + r
            images.append(flocking.FlockingTask.render(task_s, 0))
        print('reward={}'.format(reward))
        return images


    def save_images_as_gif(images, file_name):
        images[0].save(file_name,
            save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

    imgs = render(test_task, solver, policy)
    save_images_as_gif(imgs, 'flocking.gif')


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)