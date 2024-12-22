import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

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
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        
        # Max pooling over points to get global features
        x = jnp.max(x, axis=-2)  # (batch_size, seq_len, 256)

        # Project to latent space
        x = nn.Dense(64)(x)
        x = nn.relu(x)
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
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
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
    
    @nn.compact
    def __call__(self, x, pos):
        batch_size, seq_len, num_points, _ = x.shape
        
        # Linear projections for query, key, value
        q = nn.Dense(self.hidden_dim)(x)  # (B, S, N, H)
        k = nn.Dense(self.hidden_dim)(x)  # (B, S, N, H)
        v = nn.Dense(self.hidden_dim)(x)  # (B, S, N, H)
        
        # Position encodings
        pos_enc = nn.Dense(self.hidden_dim)(pos)  # (B, S, N, H)
        
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
        attn = q * (k_neighbors + p_neighbors + pos_diff_enc)  # (B, S, N, K, H)
        attn = nn.Dense(1)(attn).squeeze(-1)  # (B, S, N, K)
        attn = nn.softmax(attn, axis=-1)
        
        # Weighted sum
        out = jnp.sum(jnp.expand_dims(attn, -1) * v_neighbors, axis=-2)  # (B, S, N, H)
        
        # Output projection
        out = nn.Dense(x.shape[-1])(out)  # (B, S, N, D)
        out = nn.relu(out)
        
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
        out = nn.relu(out)
        out = nn.Dense(self.hidden_dim)(out)

        x = x + out
        return x

class PointTransformerEncoder(nn.Module):
    """
    Point Transformer Encoder for processing point clouds.
    """
    hidden_dims: Sequence[int]
    num_neighbors: int = 16
    num_dims: int = 3
    
    @nn.compact
    def __call__(self, points):
        # Split into coordinates and features
        pos = points[..., :self.num_dims]
        x = points
        
        # Initial feature embedding
        x = nn.Dense(self.hidden_dims[0])(x)
        
        # Stack transformer layers
        for dim in self.hidden_dims:
            x = PointTransformerLayer(
                num_neighbors=self.num_neighbors,
                hidden_dim=dim
            )(x, pos)
            x = nn.LayerNorm()(x)
        
        # Global pooling
        x = jnp.max(x, axis=-2)

        x = EncoderTransformerLayer(hidden_dim=self.hidden_dims[-1])(x)
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
    hidden_dims: Sequence[int] = (512, 256, 128, 64)
    
    @nn.compact
    def __call__(self, x):

        x = nn.Dense(self.hidden_dims[-1])(x)
        x = nn.relu(x)

        x = x[:, jnp.newaxis, :]

        for idx in range(self.seq_len):
            q = nn.Dense(self.hidden_dims[-1])(x[:, -1:, :])
            k = nn.Dense(self.hidden_dims[-1])(x)
            v = nn.Dense(self.hidden_dims[-1])(x)

            attn_weights = jnp.einsum('bth,bsh->bts', q, k)  # (batch, 1, seq_len)
            attn_weights = nn.softmax(attn_weights / jnp.sqrt(self.hidden_dims[-1]))
            out = jnp.einsum('bts,bsh->bth', attn_weights, v)  # (batch, 1, hidden_dim)

            out = nn.relu(out)
            out = nn.Dense(self.hidden_dims[-1])(out)

            out = x[:,:1,:] + out

            x = jnp.concatenate([x, out], axis=1)

        x = x[:, 1:, :]

        # generate points
        x = nn.Dense(self.num_points * self.num_dims)(x)
        x = jnp.reshape(x, (-1, self.seq_len, self.num_points, self.num_dims))
        
        return x

class PointTransformerAutoencoder(nn.Module):
    """
    Complete Point Transformer Autoencoder
    """
    num_points: int
    seq_len: int
    num_dims: int
    encoder_dims: Sequence[int] = (64, 64)
    decoder_dims: Sequence[int] = (64, 64)
    latent_dim: int = 32
    num_neighbors: int = 16
    
    def setup(self):
        self.encoder = PointTransformerEncoder(
            hidden_dims=self.encoder_dims,
            num_neighbors=self.num_neighbors
        )
        self.latent_proj = nn.Dense(self.latent_dim)
        self.decoder = PointTransformerDecoder(
            hidden_dims=self.decoder_dims,
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
