
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def growth(U):
  return 0 + ((U>=0.20)&(U<=0.25)) - ((U<=0.19)|(U>=0.33))

class Lenia:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        size = 64
        self.T = 10
        self.A = np.random.rand(size, size)
        self.K = np.asarray([[1,1,1], [1,0,1], [1,1,1]])
        self.K = self.K / np.sum(self.K)
        self.im = self.ax.imshow(self.A, cmap='viridis')


    def update(self, frame):
        U = jsp.signal.convolve2d(self.A, self.K, mode='same')
        self.A = jnp.clip(self.A + 1 / self.T * growth(U), 0, 1)

        self.im.set_data(self.A)
        return self.im,

def main():
    hypercycles = Lenia()
    ani = FuncAnimation(hypercycles.fig, hypercycles.update, blit=True, interval=int(1000 / 60))
    plt.show()

if __name__ == '__main__':
    main()