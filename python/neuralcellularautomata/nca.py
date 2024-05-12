import functools
from typing import Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import serialization
from jax import lax

from jax_nca.nca import TrainablePerception, nca_multi_step

class UpdateNet(nn.Module):
    num_channels: int
    num_conv_features: int = 4
    num_layers: int = 3

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Conv(
                features=self.num_conv_features, kernel_size=(1, 1), strides=1, padding="VALID"
            )(x)
            x = nn.relu(x)
        x = nn.Conv(
            features=self.num_channels,
            kernel_size=(1, 1),
            strides=1,
            padding="VALID",
            kernel_init=jax.nn.initializers.zeros,
            use_bias=False,
        )(x)
        x = nn.softmax(x)
        return x

class DiscreteNCA(nn.Module):
    num_target_channels: int
    alpha_living_threshold: float = 0.5

    """
        num_hidden_channels: Number of hidden channels for each cell to use
        num_target_channels: Number of target channels to be used
        alpha_living_threshold: threshold to determine whether a cell lives or dies
        cell_fire_rate: probability that a cell receives an update per step
        trainable_perception: if true, instead of using sobel filters use a trainable conv net
        alpha: scalar value to be multiplied to updates
    """

    @classmethod
    def create_seed(
        cls,
        num_target_channels: int,
        shape: Tuple[int] = (48, 48),
        batch_size: int = 1,
    ):
        seed = jnp.zeros((batch_size, *shape, num_target_channels + 1))
        w, h = seed.shape[1], seed.shape[2]
        return seed.at[:, w // 2, h // 2, -2:-1].set(1.0)

    def setup(self):
        num_channels = self.num_target_channels + 1
        self.perception = TrainablePerception(num_channels)
        self.update_net = UpdateNet(num_channels, num_conv_features=8, num_layers=2)

    def alive(self, x):
        # check if final channel is on (meaning the cell is dead)
        return (
            nn.max_pool(
                1 - x[..., -1:], window_shape=(3, 3), strides=(1, 1), padding="SAME"
            ) > self.alpha_living_threshold
        )

    def __call__(self, x, rng):
        pre_life_mask = self.alive(x)

        perception_out = self.perception(x)
        update = jnp.reshape(self.update_net(perception_out), x.shape)

        x = x + update

        post_life_mask = self.alive(x)

        life_mask = pre_life_mask & post_life_mask
        life_mask = life_mask.astype(float)

        return x * life_mask

    def save(self, params, path: str):
        bytes_output = serialization.to_bytes(params)
        with open(path, "wb") as f:
            f.write(bytes_output)

    def load(self, path: str):
        nca_seed = self.create_seed(
            self.num_target_channels, batch_size=1
        )
        rng = jax.random.PRNGKey(0)
        init_params = self.init(rng, nca_seed, rng)["params"]
        with open(path, "rb") as f:
            bytes_output = f.read()
        return serialization.from_bytes(init_params, bytes_output)

    def multi_step(self, params, current_state: jnp.array, rng, num_steps: int = 2):
        return nca_multi_step(self.apply, params, current_state, rng, num_steps)

    def to_rgb(self, x: jnp.array):
        channels, a = x[..., :-1], jnp.clip(x[..., -1:], 0.0, 1.0)
        channel_colours = jnp.linspace(0.3, 1.0, self.num_target_channels) * jnp.ones((3, self.num_target_channels))
        rgb = jnp.matmul(channels, einops.rearrange(channel_colours, 'c s -> s c'))
        rgb = jnp.clip(rgb - a, 0.0, 1.0) # if a is 1, then the cell is dead, so we cell is black
        return rgb
