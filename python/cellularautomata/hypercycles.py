import einops
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def growth(U, A):
    return 0 + ((jnp.roll(A, 1, axis=-1) == 1.0) & (U >= 1.0)) - ((A == 1.0) & (jnp.roll(U, -1, axis=-1) >= 1.0))

class Hypercycles:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        size = 64
        self.states = 2
        self.num_channels = 4
        initial = np.random.randint(self.num_channels, size=(size, size))
        self.A = jnp.zeros((size, size, self.num_channels))
        self.A = jnp.where(initial[:, :, None] == jnp.arange(self.num_channels), 1.0, 0.0)
        self.K = np.asarray([[1,1,1], [1,0,1], [1,1,1]]).astype(np.float32)
        # self.K = self.K / np.sum(self.K)
        self.im = self.ax.imshow(jnp.dot(self.A, jnp.linspace(0, 1, self.num_channels)), cmap='viridis')


    def update(self, frame):
        U = lax.conv_general_dilated(
            einops.rearrange(self.A, 'h w c -> c 1 h w'),
            einops.repeat(self.K, 'h w -> i o h w', o=1, i=1),
            window_strides=(1, 1),
            padding='SAME',
        )
        U = einops.rearrange(U, 'c 1 h w -> h w c')
        self.A = jnp.clip(self.A + growth(U, self.A), 0, 1)

        self.im.set_data(jnp.dot(self.A, jnp.linspace(0, 1, self.num_channels)))
        return self.im,

def main():
    hypercycles = Hypercycles()
    ani = FuncAnimation(hypercycles.fig, hypercycles.update, blit=True, interval=500)#int(1000 / 60))
    plt.show()

if __name__ == '__main__':
    main()