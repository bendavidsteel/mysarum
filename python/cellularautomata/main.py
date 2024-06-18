from collections.abc import Iterable
import functools
import os

import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tqdm

from jax_nca.dataset import ImageDataset
from jax_nca.nca import NCA

from dataset import RhythmDataset
from nca import DiscreteNCA
from trainer import Trainer, cross_entropy_loss, mse_loss

def main():
    # uncomment this to enable jax gpu preallocation, might lead to memory issues
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    img_size = 8
    dataset = RhythmDataset(img_size=img_size)

    # ### NCA
    # - num_hidden_channels = 16
    # - num_target_channels = 3
    # - cell_fire_rate = 1.0 (100% chance for cells to be updated)
    # - alpha_living_threshold = 0.1 (threshold for cells to be alive)

    nca = DiscreteNCA(5)

    trainer = Trainer(dataset, nca, loss=cross_entropy_loss, n_damage=0)
    trainer.train(200, batch_size=128, seed=10, lr=1e-3, min_steps=64, max_steps=96)

    # #### Get current state from trainer

    state = trainer.state

    # save
    nca.save(state.params, "saved_params")

    params = nca.load("saved_params")

    def render_nca_steps(nca, params, shape = (64, 64), num_steps = 2):
        nca_seed = nca.create_seed(nca.num_target_channels, shape=shape, batch_size=1)
        rng = jax.random.PRNGKey(0)
        _, outputs = nca.multi_step(params, nca_seed, rng, num_steps=num_steps)
        stacked = jnp.squeeze(jnp.stack(outputs))
        rgbs = np.array(nca.to_rgb(stacked))

        frames = []
        for r in rgbs:
            img = Image.fromarray((r * 255).astype(np.uint8))
            frames.append(img)
        frames[0].save('nca.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

    render_nca_steps(nca, params, shape=(img_size, img_size), num_steps=256)


if __name__ == "__main__":
    main()