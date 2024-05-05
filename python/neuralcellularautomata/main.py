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

from jax_nca.trainer import SamplePool, create_train_state, get_tensorboard_logger, flatten
from jax_nca.dataset import ImageDataset
from jax_nca.nca import NCA
from jax_nca.utils import make_circle_masks

def clip_grad_norm(grad):
    factor = 1.0 / (
        jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.linalg.norm, grad))))
        + 1e-8
    )
    return jax.tree_util.tree_map((lambda x: x * factor), grad)


@functools.partial(jax.jit, static_argnames=("apply_fn", "num_steps"))
def train_step(
    apply_fn, state, seeds: jnp.array, targets: jnp.array, num_steps: int, rng
):
    def mse_loss(pred, y):
        squared_diff = jnp.square(pred - y)
        return jnp.mean(squared_diff, axis=[-3, -2, -1])

    def loss_fn(params):
        def forward(carry, inp):
            carry = apply_fn({"params": params}, carry, rng)
            return carry, carry

        x, outs = jax.lax.scan(forward, seeds, None, length=num_steps)
        rgb, a = x[..., :3], jnp.clip(x[..., 3:4], 0.0, 1.0)
        rgb = jnp.clip(1.0 - a + rgb, 0.0, 1.0)

        outs = jnp.transpose(outs, [1, 0, 2, 3, 4])
        subset = outs[:, -8:]  # B 12 H W C
        return jnp.mean(
            jax.vmap(mse_loss)(subset[..., :4], jnp.expand_dims(targets, 1))
        ), (x, rgb)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grads = clip_grad_norm(grads)
    updated, rgb = aux
    return state.apply_gradients(grads=grads), loss, grads, updated, rgb

class SamplePool:
    def __init__(self, max_size: int = 1000, img_shape=(64, 64, 4)):
        self.max_size = max_size
        self.pool = jnp.zeros((max_size, *img_shape))
        self.empty = jnp.ones((max_size,), dtype=bool)

    def __getitem__(self, idx):
        return self.pool[idx]

    def __setitem__(self, idx, v):
        self.pool = self.pool.at[idx].set(v)
        self.empty = self.empty.at[idx].set(False)

    def sample(self, rng, num_samples: int):
        indices = jax.random.randint(rng, (num_samples,), 0, self.max_size)
        return self.pool[indices], self.empty[indices], indices

class Trainer:
    def __init__(self, dataset, nca, pool_size: int = 1024, n_damage: int = 0):
        self.dataset = dataset
        self.img_shape = self.dataset.img_shape
        self.nca = nca
        self.pool_size = pool_size
        self.n_damage = n_damage
        self.state = None

    def train(
        self,
        num_epochs,
        batch_size: int = 8,
        seed: int = 10,
        lr: float = 0.001,
        min_steps: int = 64,
        max_steps: int = 96,
    ):
        

        writer = get_tensorboard_logger("EMOJITrainer")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        self.state = create_train_state(init_rng, self.nca, lr, self.dataset.img_shape)

        default_seed = self.nca.create_seed(
            self.nca.num_hidden_channels,
            self.nca.num_target_channels,
            shape=self.img_shape[:-1],
            batch_size=1,
        )[0]
        pool = SamplePool(self.pool_size, img_shape=default_seed.shape)

        bar = tqdm.tqdm(np.arange(num_epochs))
        for i in bar:
            num_steps = int(jax.random.randint(rng, (), min_steps, max_steps))
            samples, empty, indices = pool.sample(rng, batch_size)
            samples = jax.lax.select(
                einops.repeat(empty, "b -> b h w c", h=default_seed.shape[0], w=default_seed.shape[1], c=default_seed.shape[2]),
                einops.repeat(default_seed, "h w c -> b h w c", b=batch_size), 
                samples
            )
            samples = samples.at[0].set(default_seed)
            batch = samples
            if self.n_damage > 0:
                damage = (
                    1.0
                    - make_circle_masks(
                        int(self.n_damage), self.img_shape[0], self.img_shape[1]
                    )[..., None]
                )
                batch[-self.n_damage :] *= damage

            targets, rgb_targets = self.dataset.get_batch(batch_size)

            self.state, loss, grads, outputs, rgb_outputs = train_step(
                self.nca.apply,
                self.state,
                batch,
                targets,
                num_steps=num_steps,
                rng=rng,
            )

            # grad_dict = {k: dict(grads[k]) for k in grads.keys()}
            # grad_dict = flatten(grad_dict)

            # grad_dict = {
            #     k: {kk: np.sum(vv).item() for kk, vv in v.items()}
            #     for k, v in grad_dict.items()
            # }
            # grad_dict = flatten(grad_dict)

            pool[indices] = outputs

            bar.set_description("Loss: {}".format(loss.item()))

            self.emit_metrics(
                writer,
                i,
                batch,
                rgb_outputs,
                rgb_targets,
                loss.item(),
                # metrics=grad_dict,
            )

        return self.state

    def emit_metrics(
        self, train_writer, i: int, batch, outputs, targets, loss, metrics={}
    ):
        train_writer.add_scalar("loss", loss, i)
        # train_writer.add_scalar("log10(loss)", math.log10(loss), i)
        train_writer.add_images("batch", self.nca.to_rgb(batch), i, dataformats="NHWC")
        train_writer.add_images("outputs", outputs, i, dataformats="NHWC")
        train_writer.add_images("targets", targets, i, dataformats="NHWC")
        for k in metrics:
            train_writer.add_scalar(k, metrics[k], i)


def main():
    # uncomment this to enable jax gpu preallocation, might lead to memory issues
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    img_size = 32
    dataset = ImageDataset(emoji='🦎', img_size=img_size)

    # ### NCA
    # - num_hidden_channels = 16
    # - num_target_channels = 3
    # - cell_fire_rate = 1.0 (100% chance for cells to be updated)
    # - alpha_living_threshold = 0.1 (threshold for cells to be alive)

    nca = NCA(8, 3, trainable_perception=False, cell_fire_rate=1.0, alpha_living_threshold=0.1)



    trainer = Trainer(dataset, nca, n_damage=0)
    trainer.train(1000, batch_size=16, seed=10, lr=2e-4, min_steps=64, max_steps=96)

    # #### Get current state from trainer

    state = trainer.state

    # save
    nca.save(state.params, "saved_params")

    params = nca.load("saved_params")

    def render_nca_steps(nca, params, shape = (64, 64), num_steps = 2):
        nca_seed = nca.create_seed(nca.num_hidden_channels, nca.num_target_channels, shape=shape, batch_size=1)
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