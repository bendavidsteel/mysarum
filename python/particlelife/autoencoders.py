import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

def get_dtype(use_fp16: bool):
    """Helper to get dtype based on precision flag."""
    return jnp.float16 if use_fp16 else jnp.float32


class PointCloudNNEncoder(nn.Module):
    """
    Encoder network that takes a point cloud and outputs a latent vector.
    Input shape: (batch_size, seq_len, num_points, 3)
    Output shape: (batch_size, latent_dim)
    """
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # First pass through point-wise MLPs
        x = nn.Dense(16)(x)  # (batch_size, seq_len, num_points, 16)
        x = nn.gelu(x)
        x = nn.Dense(32)(x)
        x = nn.gelu(x)
        
        # Max pooling over points to get global features
        x = jnp.max(x, axis=-2)  # (batch_size, seq_len, 256)

        # Project to latent space
        x = nn.Dense(64)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.latent_dim)(x)
        
        x = x.mean(axis=1)  # Average over sequence length

        return x

class PointCloudNNDecoder(nn.Module):
    """
    Decoder network that takes a latent vector and outputs a point cloud.
    Input shape: (batch_size, latent_dim)
    Output shape: (batch_size, num_points, 3)
    """
    seq_len: int
    num_points: int
    num_dims: int

    @nn.compact
    def __call__(self, z):
        # Expand latent vector
        x = nn.Dense(64)(z)
        x = nn.gelu(x)
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        
        # Reshape to get multiple points
        x = nn.Dense(self.seq_len * self.num_points * self.num_dims)(x)
        x = jnp.reshape(x, (-1, self.seq_len, self.num_points, self.num_dims))
        
        return x

class PointCloudNNAutoencoder(nn.Module):
    """
    Complete autoencoder for point clouds.
    """
    latent_dim: int
    seq_len: int
    num_points: int
    num_dims: int

    def setup(self):
        self.encoder = PointCloudNNEncoder(latent_dim=self.latent_dim)
        self.decoder = PointCloudNNDecoder(seq_len=self.seq_len, num_points=self.num_points, num_dims=self.num_dims)

    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

class PointTransformerLayer(nn.Module):
    """
    Single Point Transformer Layer implementing self-attention for point clouds.
    """
    num_neighbors: int
    hidden_dim: int
    use_fp16: bool = False
    
    @nn.compact
    def __call__(self, x, pos):
        dtype = get_dtype(self.use_fp16)
        batch_size, seq_len, num_points, _ = x.shape
        
        # Linear projections for query, key, value
        dense_init = nn.initializers.variance_scaling(2.0, 'fan_in', 'truncated_normal')
        q = nn.Dense(self.hidden_dim, dtype=dtype, kernel_init=dense_init)(x)  # (B, S, N, H)
        k = nn.Dense(self.hidden_dim, dtype=dtype, kernel_init=dense_init)(x)  # (B, S, N, H)
        v = nn.Dense(self.hidden_dim, dtype=dtype, kernel_init=dense_init)(x)  # (B, S, N, H)
        
        # Position encodings
        pos_enc = nn.Dense(self.hidden_dim, dtype=jnp.float32)(pos)  # (B, S, N, H)
        
        # Compute pairwise distances
        pos_diff = jnp.expand_dims(pos, axis=-2) - jnp.expand_dims(pos, axis=-3)  # (B, S, N, N, D)
        dist = jnp.sum(pos_diff ** 2, axis=-1)  # (B, S, N, N)
        
        # Find k nearest neighbors
        _, neighbor_idx = jax.lax.top_k(-dist, self.num_neighbors)  # (B, S, N, K)
        
        # Gather neighbors for queries, keys, values, and positions
        def gather_neighbors(a):
            return jax.vmap(jax.vmap(lambda ak, an: jax.vmap(lambda en: ak[en])(an)))(a, neighbor_idx)
        
        k_neighbors = gather_neighbors(k)      # (B, S, N, K, H)
        v_neighbors = gather_neighbors(v)      # (B, S, N, K, H)
        p_neighbors = gather_neighbors(pos_enc)  # (B, S, N, K, H)
        
        # Query with position encoding
        q = q + pos_enc  # (B, S, N, H)
        q = jnp.expand_dims(q, axis=-2)  # (B, S, N, 1, H)
        
        # Relative position attention
        pos_diff_enc = nn.Dense(self.hidden_dim)(
            gather_neighbors(pos) - jnp.expand_dims(pos, axis=-2)
        )  # (B, S, N, K, H)
        
        # Attention weights
        scale = jnp.sqrt(self.hidden_dim).astype(dtype)
        attn = q * (k_neighbors + p_neighbors + pos_diff_enc) / scale  # (B, S, N, K, H)
        attn = nn.Dense(1)(attn).squeeze(-1)  # (B, S, N, K)
        attn = nn.softmax(attn, axis=-1)
        
        # Weighted sum
        out = jnp.sum(jnp.expand_dims(attn, -1) * v_neighbors, axis=-2)  # (B, S, N, H)
        
        # Output projection
        out = nn.Dense(x.shape[-1])(out)  # (B, S, N, D)
        out = nn.gelu(out)
        
        # Residual connection
        return x + out
    
class EncoderTransformerLayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # sequence transformer layers
        k = nn.Dense(self.hidden_dim)(x)
        q = nn.Dense(self.hidden_dim)(x)
        v = nn.Dense(self.hidden_dim)(x)

        # Attention scores
        attn_weights = jnp.einsum('bth,bsh->bts', q, k)  # (batch, seq_len, seq_len)
        attn_weights = nn.softmax(attn_weights / jnp.sqrt(self.hidden_dim))
        out = jnp.einsum('bts,bsh->bth', attn_weights, v)  # (batch, seq_len, hidden_dim)

        # ff layer
        out = nn.Dense(self.hidden_dim)(out)
        out = nn.gelu(out)
        out = nn.Dense(self.hidden_dim)(out)

        x = x + out
        return x

class PointTransformerEncoder(nn.Module):
    """
    Point Transformer Encoder for processing point clouds.
    """
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    num_neighbors: int = 16
    num_dims: int = 3
    
    @nn.compact
    def __call__(self, points):
        # Split into coordinates and features
        pos = points[..., :self.num_dims]
        x = points
        
        # Initial feature embedding
        x = nn.Dense(self.hidden_dim)(x)
        
        # Stack transformer layers
        for _ in range(self.num_hidden_layers):
            x = PointTransformerLayer(
                num_neighbors=self.num_neighbors,
                hidden_dim=self.hidden_dim
            )(x, pos)
            x = nn.LayerNorm()(x)
        
        # Global pooling
        x = jnp.max(x, axis=-2)

        x = EncoderTransformerLayer(hidden_dim=self.hidden_dim)(x)
        x = nn.LayerNorm()(x)

        x = jnp.max(x, axis=-2)
        
        return x

class PointTransformerDecoder(nn.Module):
    """
    Decoder for generating point clouds from encoded features.
    """
    num_points: int
    seq_len: int
    num_dims: int = 3
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        # Initial expansion to sequence dimension
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = x[:, jnp.newaxis, :]  # (batch, 1, hidden_dim)
        
        # Create positional encodings for maximum sequence length
        position = jnp.arange(self.seq_len + 1)[:, None]  # +1 because we include initial token
        div_term = jnp.exp(jnp.arange(0, self.hidden_dim, 2) * (-jnp.log(10000.0) / self.hidden_dim))
        pos_enc = jnp.zeros((self.seq_len + 1, self.hidden_dim))
        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))
        pos_enc = jnp.array(pos_enc)[None, :, :]  # (1, seq_len+1, hidden_dim)
        
        for idx in range(self.seq_len):
            # Add positional encoding to current sequence
            x_pos = x + pos_enc[:, :x.shape[1]]
            
            # Query from last position with positional encoding
            q = nn.Dense(self.hidden_dim)(x_pos[:, -1:, :])
            # Keys and values include positional encoding
            k = nn.Dense(self.hidden_dim)(x_pos)
            v = nn.Dense(self.hidden_dim)(x_pos)
            
            attn_weights = jnp.einsum('bth,bsh->bts', q, k)
            attn_weights = nn.softmax(attn_weights / jnp.sqrt(self.hidden_dim))
            out = jnp.einsum('bts,bsh->bth', attn_weights, v)
            
            out = nn.gelu(out)
            out = nn.Dense(self.hidden_dim)(out)
            
            out = x[:,:1,:] + out
            x = jnp.concatenate([x, out], axis=1)
        
        x = x[:, 1:, :]  # Remove initial token

        # generate points
        # Generate coarse structure first
        coarse_points = nn.Dense(self.num_points // 4 * self.num_dims)(x)
        coarse_points = jnp.reshape(coarse_points, (-1, self.seq_len, self.num_points // 4, self.num_dims))

        # Generate features for refinement
        features = nn.Dense(self.hidden_dim)(coarse_points)
        features = nn.LayerNorm()(features)
        features = nn.gelu(features)

        # Generate local point patches around coarse points
        local_offsets = nn.Dense(4 * self.num_dims)(features)
        local_offsets = jnp.reshape(local_offsets, (-1, self.seq_len, self.num_points // 4, 4, self.num_dims))

        # Combine coarse points with local details
        coarse_points_expanded = jnp.expand_dims(coarse_points, -2)
        fine_points = coarse_points_expanded + local_offsets
        x = jnp.reshape(fine_points, (-1, self.seq_len, self.num_points, self.num_dims))
        
        return x

class PointTransformerAutoencoder(nn.Module):
    """
    Complete Point Transformer Autoencoder
    """
    num_points: int
    seq_len: int
    num_dims: int
    encoder_dim: int = 64
    encoder_num_layers: int = 3
    decoder_dim: int = 64
    latent_dim: int = 32
    num_neighbors: int = 16
    
    def setup(self):
        self.encoder = PointTransformerEncoder(
            hidden_dim=self.encoder_dim,
            num_hidden_layers=self.encoder_num_layers,
            num_neighbors=self.num_neighbors
        )
        self.latent_proj = nn.Dense(self.latent_dim)
        self.decoder = PointTransformerDecoder(
            hidden_dim=self.decoder_dim,
            num_points=self.num_points,
            num_dims=self.num_dims,
            seq_len=self.seq_len
        )
    
    def __call__(self, points):
        # Encode
        x = self.encoder(points)
        latent = self.latent_proj(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed
