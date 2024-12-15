import jax
import jax.numpy as jnp

@jax.jit
def fft_entropy_filtered(positions: jnp.ndarray, min_period: int = 200, max_period: int = 1000) -> jnp.ndarray:
    """
    Calculate entropy of FFT coefficients with frequency filtering.
    
    Args:
        positions: Array of shape (timesteps, particles, n_dims)
        min_period: Minimum period length to consider (in timesteps)
        max_period: Maximum period length to consider (in timesteps)
    
    Returns:
        Filtered entropy value
    """
    assert len(positions.shape) == 3, "Must have timesteps, particles, n_dims shape"
    timesteps, particles, n_dims = positions.shape
    
    # Compute FFT along time axis
    fft = jnp.fft.fft(positions, axis=0)
    power = jnp.abs(fft) ** 2
    
    # Calculate frequencies
    freqs = jnp.fft.fftfreq(timesteps)
    
    # Create frequency mask for desired range
    # Convert periods to frequencies
    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    
    # Create mask for desired frequency range
    freq_mask = (jnp.abs(freqs) >= min_freq) & (jnp.abs(freqs) <= max_freq)
    freq_mask = freq_mask[:, jnp.newaxis, jnp.newaxis]  # Match dimensions
    
    # Apply mask and add small constant for numerical stability
    epsilon = 1e-10
    masked_power = jnp.where(freq_mask, power + epsilon, epsilon)
    
    # Calculate entropy only for masked frequencies
    log_power = jnp.log(masked_power)
    entropy = -jnp.sum(masked_power * log_power)
    
    # Normalize by number of frequencies considered
    num_freqs = jnp.sum(freq_mask)
    normalized_entropy = entropy / (num_freqs * particles * n_dims)
    
    return normalized_entropy

@jax.jit
def multi_scale_fft_entropy(positions: jnp.ndarray, 
                          period_ranges: list = [(100, 200), (200, 500), (500, 1000)],
                          weights: list = [1.0, 1.0, 1.0]) -> jnp.ndarray:
    """
    Calculate weighted entropy across multiple frequency bands.
    
    Args:
        positions: Array of shape (timesteps, particles, n_dims)
        period_ranges: List of (min_period, max_period) tuples
        weights: Weight for each frequency band
    
    Returns:
        Weighted sum of entropies across frequency bands
    """
    entropies = []
    for (min_period, max_period), weight in zip(period_ranges, weights):
        entropy = fft_entropy_filtered(positions, min_period, max_period)
        entropies.append(weight * entropy)
    
    return sum(entropies)

@jax.jit
def jitter(positions: jnp.ndarray, order: int = 2) -> jnp.ndarray:
    """
    Penalize jittering of particles.
    
    Args:
        positions: Array of shape (timesteps, particles, n_dims)
    
    Returns:
        Jitter penalty value
    """
    assert len(positions.shape) == 3, "Must have timesteps, particles, n_dims shape"
    timesteps, particles, n_dims = positions.shape
    
    # Calculate difference between consecutive timesteps
    diff = jnp.diff(positions, axis=0)
    
    # Calculate squared distance
    squared_diff = jnp.sum(diff ** order, axis=-1)
    
    # Sum across timesteps and particles
    jitter = jnp.sum(squared_diff)
    
    return jitter