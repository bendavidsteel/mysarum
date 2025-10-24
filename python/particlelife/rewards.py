import functools

import jax
import jax.numpy as jnp
import numpy as np
import torch
# import transformers

@functools.partial(jax.jit, static_argnames=['base_n_freqs'])
def fft_peak(energies, base_n_freqs: int = 10):
    energies -= jnp.mean(energies, axis=0, keepdims=True)
    energies *= jnp.hanning(energies.shape[0])[:, None]
    fft = jnp.fft.rfft(energies, axis=0)
    fft_freqs = jnp.fft.rfftfreq(energies.shape[0])
    fft = fft[:base_n_freqs]  # remove the zero frequency component and take the first base_n_freqs
    fft_power = jnp.abs(fft)
    fft_power_q3 = jnp.percentile(fft_power, 75, axis=1)
    fft_power_diff = jnp.diff(fft_power_q3)

    return jnp.max(fft_power_diff)  # return the maximum increase in power

@jax.jit
def fft_entropy_filtered(energies: jnp.ndarray, min_period: int = 200, max_period: int = 1000) -> jnp.ndarray:
    """
    Calculate entropy of FFT coefficients with frequency filtering.
    
    Args:
        positions: Array of shape (timesteps, particles, n_dims)
        min_period: Minimum period length to consider (in timesteps)
        max_period: Maximum period length to consider (in timesteps)
    
    Returns:
        Filtered entropy value
    """
    assert len(energies.shape) == 2, "Must have timesteps, particles shape"
    timesteps, particles = energies.shape
    
    # Compute FFT along time axis
    fft = jnp.fft.fft(energies, axis=0)
    power = jnp.abs(fft) ** 2
    
    # Calculate frequencies
    freqs = jnp.fft.fftfreq(timesteps)
    
    # Create frequency mask for desired range
    # Convert periods to frequencies
    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    
    # Create mask for desired frequency range
    freq_mask = (jnp.abs(freqs) >= min_freq) & (jnp.abs(freqs) <= max_freq)
    freq_mask = freq_mask[:, jnp.newaxis]  # Match dimensions
    
    # Apply mask and add small constant for numerical stability
    epsilon = 1e-10
    masked_power = jnp.where(freq_mask, power + epsilon, epsilon)
    
    # Calculate entropy only for masked frequencies
    log_power = jnp.log(masked_power)
    entropy = -jnp.sum(masked_power * log_power)
    
    # Normalize by number of frequencies considered
    num_freqs = jnp.sum(freq_mask)
    normalized_entropy = entropy / (num_freqs * particles)
    
    return normalized_entropy

@jax.jit
def position_fft_entropy_filtered(positions: jnp.ndarray, min_period: int = 200, max_period: int = 1000) -> jnp.ndarray:
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
def position_multi_scale_fft_entropy(positions: jnp.ndarray, 
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
        entropy = position_fft_entropy_filtered(positions, min_period, max_period)
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

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Equivalent to tensor.norm(p=2, dim=-1, keepdim=True) for executorch compatibility"""
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor

# class CLIPLifelikeClassifier:
#     def __init__(self):
#         print("Loading CLIP model for classification...")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#         self.model = transformers.AutoModel.from_pretrained(
#             "openai/clip-vit-base-patch32",
#             device_map=self.device,
#             torch_dtype=torch.float16
#         )

#         # Initialize processor for image embeddings
#         self.processor = transformers.AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

#         # Life-like prompts
#         self.life_like_prompts = [
#             "living organisms",
#             "multicellular organisms",
#             "plants",
#             "roots",
#             "fungi",
#             "mould",
#             "plankton",
#             "deep sea creatures"
#         ]

#         # Encode text prompts once
#         print("Encoding classification prompts...")
#         inputs = self.tokenizer(self.life_like_prompts, padding=True, return_tensors="pt")
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         self.text_features = self.model.get_text_features(**inputs)

#     def get_image_embeddings(self, images):
#         """Get CLIP embeddings for images"""
#         if len(images) == 0:
#             return np.zeros((0, 512))
#         inputs = self.processor(images=images, return_tensors="pt")
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         return self.model.get_image_features(**inputs).cpu().detach().numpy()

#     def classify_lifelikeness(self, embeddings):
#         """Classify life-likeness from pre-computed embeddings"""
#         if len(embeddings) == 0:
#             return np.array([])

#         batch_size = embeddings.shape[0]

#         # Convert to torch tensor
#         image_embeds = torch.tensor(embeddings).to(self.device)
#         text_embeds = self.text_features.unsqueeze(0).expand(batch_size, -1, -1)

#         # Normalize features
#         image_embeds = image_embeds / _get_vector_norm(image_embeds)
#         text_embeds = text_embeds / _get_vector_norm(text_embeds)

#         # Compute cosine similarity
#         logits_per_text = torch.einsum("btd,bkd->bkt", text_embeds, image_embeds)
#         logits_per_text = logits_per_text * self.model.logit_scale.exp().to(text_embeds.device)

#         return torch.mean(logits_per_text, dim=-1).cpu().detach().numpy()

# def clip_lifelikeness_reward(trajectory, map_size, species, num_species, classifier=None, start=-3000, offset=1000):
#     """
#     Calculate CLIP-based life-likeness reward for a trajectory.

#     Args:
#         trajectory: JAX array of shape (timesteps, particles, dims)
#         map_size: Size of the simulation map
#         species: Particle species assignments
#         num_species: Number of species
#         classifier: CLIPLifelikeClassifier instance
#         start: Start frame for rendering
#         offset: Frame sampling offset

#     Returns:
#         Life-likeness score and embedding descriptor
#     """
#     from particle_lenia import draw_multi_species_particles

#     if classifier is None:
#         classifier = CLIPLifelikeClassifier()

#     try:
#         # Render frames for analysis
#         video_frames = draw_multi_species_particles(
#             trajectory, map_size, species,
#             num_species=num_species, start=start, offset=offset
#         )

#         if video_frames.shape[0] == 0:
#             return 0.0, np.zeros(512)

#         # Get CLIP embeddings
#         np_images = [np.array(frame) for frame in video_frames]
#         embeddings = classifier.get_image_embeddings(np_images)

#         # Classify life-likeness
#         life_scores = classifier.classify_lifelikeness(embeddings)

#         # Use mean of max life-likeness scores as fitness
#         max_life_scores = np.mean(life_scores, axis=-1)  # Average across text prompts
#         fitness = float(np.mean(max_life_scores))  # Average across frames

#         # Use the embedding from the most life-like frame as descriptor
#         best_frame_idx = np.argmax(max_life_scores)
#         descriptor = embeddings[best_frame_idx] / np.linalg.norm(embeddings[best_frame_idx])

#         return fitness, descriptor

#     except Exception as e:
#         print(f"CLIP evaluation failed: {e}")
#         return 0.0, np.zeros(512)