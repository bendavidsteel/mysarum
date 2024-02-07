import random

import jax
import matplotlib.pyplot as plt
import numpy as np

def generate_ring_image(image_size):
    image = np.zeros((image_size, image_size))
    n = 10
    for _ in range(n):
        r = random.uniform(image_size/10, image_size/2)
        width = random.normalvariate(image_size/40, image_size/40)
        x = random.uniform(0, image_size)
        y = random.uniform(0, image_size)
        for i in range(image_size):
            for j in range(image_size):
                dist = ((x - i) ** 2 + (y - j) ** 2) ** (1/2)
                if (dist > (r - (width/2))) and (dist <= (r + (width/2))):
                    image[i,j] = 1

    return image

def generate_circle_image(image_size):
    image = np.zeros((image_size, image_size))
    n = 10
    for _ in range(n):
        r = random.normalvariate(image_size/10, image_size/10)
        x = random.uniform(0, image_size)
        y = random.uniform(0, image_size)
        for i in range(image_size):
            for j in range(image_size):
                dist = ((x - i) ** 2 + (y - j) ** 2) ** (1/2)
                if dist <= r:
                    image[i,j] = 1

    return image

def generate_flat_image(image_size):
    return np.full((image_size, image_size), random.uniform(0, 1))

def generate_rect_image(image_size):
    image = np.zeros((image_size, image_size))
    n = 10
    for _ in range(n):
        w = random.normalvariate(image_size/10, image_size/10)
        h = random.normalvariate(image_size/10, image_size/10)
        x = random.uniform(0, image_size)
        y = random.uniform(0, image_size)
        for i in range(image_size):
            for j in range(image_size):
                if (i < (x + w/2) and i > (x - w/2) and j < (y + w/2) and j > (y - w/2)):
                    image[i,j] = 1

    return image

def generate_line_image(image_size):
    image = np.zeros((image_size, image_size))
    n = 10
    for _ in range(n):
        x1 = random.uniform(0, image_size)
        y1 = random.uniform(0, image_size)
        x2 = random.uniform(0, image_size)
        y2 = random.uniform(0, image_size)
        pass

    return image

def generate_noise_image(image_size):
    return np.zeros((image_size, image_size))

def main():
    # generate circle images
    fig, axes = plt.subplots(nrows=2, ncols=3)

    image_size = 100
    axes[0][0].matshow(generate_ring_image(image_size))
    axes[0][1].matshow(generate_circle_image(image_size))
    axes[0][2].matshow(generate_rect_image(image_size))
    axes[1][0].matshow(generate_line_image(image_size))
    axes[1][1].matshow(generate_noise_image(image_size))
    axes[1][2].matshow(generate_flat_image(image_size))

    plt.show()

if __name__ == '__main__':
    main()