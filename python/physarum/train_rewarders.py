import logging
import random

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

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





class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
    return train_ds, test_ds


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(
    num_epochs: int, workdir: str
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.

    Returns:
        The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.key(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, batch_size, input_rng
        )
        _, test_loss, test_accuracy = apply_model(
            state, test_ds['image'], test_ds['label']
        )

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,'
            ' test_accuracy: %.2f'
            % (
                epoch,
                train_loss,
                train_accuracy * 100,
                test_loss,
                test_accuracy * 100,
            )
        )

    return state

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

    classifier_names = ['ring', 'circle', 'rect', 'line', 'noise']
    for classifier_name in classifier_names:
        

if __name__ == '__main__':
    main()