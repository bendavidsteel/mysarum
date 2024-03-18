import jax
import jax.numpy as jnp
import jax.scipy as jsp

def get_multiscale_entropy(concentrations: jnp.ndarray) -> jnp.ndarray:
    # calculate the multi-scale entropy of the image
    # progressively downsample the image and calculate the entropy
    # of each scale
    ent = 0
    num_scale = 5
    for _ in range(num_scale):
        # downsample the image
        window = jnp.array([[0.0625, 0.125, 0.0625],
                            [0.125, 0.25, 0.125],
                            [0.0625, 0.125, 0.0625]], dtype=jnp.float16)
        concentrations = jsp.signal.convolve(concentrations, window, mode='same')
        concentrations = concentrations[::2, ::2]

        # get the spectrum of the image
        spectrum = jnp.fft.fft2(concentrations)
        spectrum = jnp.abs(spectrum)
        spec_sum = spectrum.sum()
        spec_sum_d = jnp.where(spec_sum == 0.0, 1.0, spec_sum)
        spectrum = jnp.where(spec_sum == 0.0, 0.0, spectrum / spec_sum_d)
        spectrum = spectrum.flatten()

        # calculate the entropy
        epsilon = 1e-12
        ent += -(spectrum * jnp.log(spectrum + epsilon)).sum()

    ent /= (num_scale * spectrum.size)

    # we want to maximise entropy
    return ent


def get_energy(concentrations: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(concentrations) / concentrations.size


def get_diff_sine(concentrations: jnp.ndarray, state_steps: jnp.ndarray) -> jnp.ndarray:
    desired_value = 0.5 * (jnp.sin(state_steps / 50) + 1)
    concentrations_mean = jnp.mean(concentrations)
    return jnp.abs(concentrations_mean - desired_value)


def get_random_circle_diff(concentrations: jnp.ndarray, random_key) -> jnp.ndarray:
    # generate a random circle
    
    

    # calculate the difference between the random circle and the concentrations
    return jnp.sum(jnp.abs(concentrations - circle)) / concentrations.size

def generate_circle(size, random_key):
    circle = jnp.zeros((size, size))
    x = jax.random.uniform(random_key, shape=(), minval=0, maxval=size[0])
    y = jax.random.uniform(random_key, shape=(), minval=0, maxval=size[1])
    r = jax.random.uniform(random_key, shape=(), minval=0, maxval=size[0] / 4)

    # Create a grid of coordinates
    I, J = jnp.meshgrid(jnp.arange(size[0]), jnp.arange(size[1]), indexing='ij')
    
    # Compute the mask for the circle
    mask = ((I - x) ** 2 + (J - y) ** 2) < (r ** 2)
    
    # Use the mask to set values in the circle array
    circle = jnp.where(mask, 1, circle)

    return circle
