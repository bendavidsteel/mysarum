import functools

import einops
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import tqdm

from jax_nca.trainer import create_train_state, get_tensorboard_logger, make_circle_masks

def clip_grad_norm(grad):
    factor = 1.0 / (
        jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.linalg.norm, grad))))
        + 1e-8
    )
    return jax.tree_util.tree_map((lambda x: x * factor), grad)

def mse_loss(pred, y):
    squared_diff = jnp.square(pred - y)
    return jnp.mean(squared_diff, axis=[-3, -2, -1])

def cross_entropy_loss(pred, y):
    return -jnp.sum(y * jnp.log(pred + 1e-8)) # TODO check if this is correct

@functools.partial(jax.jit, static_argnames=("apply_fn", "loss", "num_steps"))
def train_step(
    apply_fn, loss, state, seeds: jnp.array, targets: jnp.array, channel_target: jnp.array, num_steps: int, rng
):
    def loss_fn(params):
        def forward(carry, inp):
            carry = apply_fn({"params": params}, carry, rng)
            return carry, carry

        x, outs = jax.lax.scan(forward, seeds, None, length=num_steps)

        outs = einops.rearrange(outs, "s b h w c -> b s h w c")
        subset = outs[:, -targets.shape[1]:]  # only use the last 8 steps for loss
        return jnp.mean(
            jax.vmap(loss)(subset[..., channel_target], targets)
        ), x

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updated), grads = grad_fn(state.params)
    grads = clip_grad_norm(grads)
    return state.apply_gradients(grads=grads), loss, grads, updated

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

def create_train_state(rng, nca, learning_rate, shape):
    nca_seed = nca.create_seed(
        nca.num_target_channels, shape=shape[:-1], batch_size=1
    )
    """Creates initial `TrainState`."""
    params = nca.init(rng, nca_seed, rng)["params"]
    tx = optax.chain(
        # optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=nca.apply, params=params, tx=tx)

class Trainer:
    def __init__(self, dataset, nca, loss=mse_loss, pool_size: int = 1024, n_damage: int = 0):
        self.dataset = dataset
        self.loss = loss
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
            self.nca.num_target_channels,
            shape=self.img_shape[:-1],
            batch_size=1,
        )[0]
        pool = SamplePool(self.pool_size, img_shape=default_seed.shape)

        bar = tqdm.tqdm(jnp.arange(num_epochs))
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

            targets, channel_targets = self.dataset.get_batch(batch_size)

            self.state, loss, grads, outputs = train_step(
                self.nca.apply,
                self.loss,
                self.state,
                batch,
                targets,
                channel_targets,
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

            bar.set_description(f"Loss: {loss.item():.4f}")

            # self.emit_metrics(
            #     writer,
            #     i,
            #     batch,
            #     loss.item(),
            #     # metrics=grad_dict,
            # )

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
