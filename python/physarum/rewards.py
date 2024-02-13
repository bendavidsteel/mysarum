import jax.numpy as jnp
import jax.scipy as jsp

def get_multiscale_entropy(concentrations: jnp.ndarray) -> jnp.ndarray:
    # calculate the multi-scale entropy of the image
    # progressively downsample the image and calculate the entropy
    # of each scale
    ent = 0
    num_scale = 10
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