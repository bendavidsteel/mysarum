import json
import os

import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm


def soft_min(x, axis, tau=1e-3):
    """
    Compute differentiable soft minimum using log-sum-exp trick.
    Lower tau values make it closer to true minimum but potentially less stable.
    """
    return -tau * jnp.log(jnp.mean(jnp.exp(-x / tau), axis=axis))

def chamfer_distance(x, y, tau=1e-3):
    """
    Compute differentiable Chamfer Distance between two point clouds.
    
    Args:
        x: Points of shape (batch_size, num_points_x, dim)
        y: Points of shape (batch_size, num_points_y, dim)
        tau: Temperature parameter for soft minimum
        
    Returns:
        Scalar Chamfer distance averaged over batch
    """
    # Compute squared norms directly
    x_norm_sq = jnp.sum(x**2, axis=-1)  # (batch_size, num_points_x)
    y_norm_sq = jnp.sum(y**2, axis=-1)  # (batch_size, num_points_y)
    
    # Compute cross terms
    zz = jnp.matmul(x, y.transpose((0, 2, 1)))  # (batch_size, num_points_x, num_points_y)
    
    # Expand norms for broadcasting
    rx = jnp.expand_dims(x_norm_sq, axis=2)  # (batch_size, num_points_x, 1)
    ry = jnp.expand_dims(y_norm_sq, axis=1)  # (batch_size, 1, num_points_y)
    
    # Compute squared distances
    P = rx + ry - 2 * zz  # (batch_size, num_points_x, num_points_y)
    
    # Compute soft minimum distances in both directions
    min_dist_xy = jnp.mean(soft_min(P, axis=2, tau=tau))
    min_dist_yx = jnp.mean(soft_min(P, axis=1, tau=tau))
    
    return min_dist_xy + min_dist_yx

import jax
import jax.numpy as jnp
import flax.linen as nn

class PointCloudEncoder(nn.Module):
    """
    Encoder network that takes a point cloud and outputs a latent vector.
    Input shape: (batch_size, num_points, 3)
    Output shape: (batch_size, latent_dim)
    """
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # First pass through point-wise MLPs
        x = nn.Dense(64)(x)  # (batch_size, num_points, 64)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
        # Max pooling over points to get global features
        x = jnp.max(x, axis=1)  # (batch_size, 256)
        
        # Project to latent space
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        
        return x

class PointCloudDecoder(nn.Module):
    """
    Decoder network that takes a latent vector and outputs a point cloud.
    Input shape: (batch_size, latent_dim)
    Output shape: (batch_size, num_points, 3)
    """
    num_points: int

    @nn.compact
    def __call__(self, z):
        # Expand latent vector
        x = nn.Dense(512)(z)
        x = nn.relu(x)
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        
        # Reshape to get multiple points
        x = nn.Dense(self.num_points * 3)(x)
        x = jnp.reshape(x, (-1, self.num_points, 3))
        
        return x

class PointCloudAutoencoder(nn.Module):
    """
    Complete autoencoder for point clouds.
    """
    latent_dim: int
    num_points: int

    def setup(self):
        self.encoder = PointCloudEncoder(latent_dim=self.latent_dim)
        self.decoder = PointCloudDecoder(num_points=self.num_points)

    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Example usage
def create_autoencoder(num_points=1024, latent_dim=128):
    model = PointCloudAutoencoder(latent_dim=latent_dim, num_points=num_points)
    
    # Initialize with dummy input
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, num_points, 3))
    params = model.init(key, dummy_input)
    
    return model, params

class GenerateCallback:

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def log_generations(self, model, state, logger, epoch):
        if epoch % self.every_n_epochs == 0:
            pass
            # reconst_imgs = model.apply({'params': state.params}, self.input_imgs)
            # reconst_imgs = jax.device_get(reconst_imgs)

            # # Plot and add to tensorboard
            # imgs = np.stack([self.input_imgs, reconst_imgs], axis=1).reshape(-1, *self.input_imgs.shape[1:])
            # imgs = jax_to_torch(imgs)
            # grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1,1))
            # logger.add_image("Reconstructions", grid, global_step=epoch)

class TrainerModule:
    def __init__(self, num_points, latent_dim, lr=1e-3, seed=42):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.lr = lr
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = PointCloudAutoencoder(num_points=self.num_points, latent_dim=self.latent_dim)
        # Prepare logging
        self.exmp_imgs = next(iter(val_loader))[0][:8]
        self.log_dir = os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}')
        self.generate_callback = GenerateCallback(self.exmp_imgs, every_n_epochs=50)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()

    def create_functions(self):
        # Training step (example)
        @jax.jit
        def train_step(batch):
            def loss_fn(params):
                reconstructed = self.model.apply(params, batch)
                # Here you could use Chamfer distance or OT-based loss
                loss = chamfer_distance(reconstructed, batch)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(self.state.params)
            updates, new_optimizer_state = self.state.tx.update(grads, self.state)
            new_params = optax.apply_updates(self.state.params, updates)

            return new_params, new_optimizer_state, loss

        self.train_step = jax.jit(train_step)
        # Eval function
        def eval_step(batch):
            reconstructed = self.model.apply(self.state.params, batch)
            # Here you could use Chamfer distance or OT-based loss
            return chamfer_distance(reconstructed, batch)
        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        params = self.model.init(init_rng, self.exmp_imgs)['params']
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=100,
            decay_steps=500*len(train_loader),
            end_value=1e-5
        )
        optimizer = optax.chain(
            optax.clip(1.0),  # Clip gradients at 1
            optax.adam(lr_schedule)
        )
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)

    def train_model(self, num_epochs=500):
        # Train model for defined number of epochs
        best_eval = 1e6
        for epoch_idx in tqdm.tqdm(range(1, num_epochs+1)):
            self.train_epoch(epoch=epoch_idx)
            if epoch_idx % 10 == 0:
                eval_loss = self.eval_model(val_loader)
                self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    self.save_model(step=epoch_idx)
                self.generate_callback.log_generations(self.model, self.state, logger=self.logger, epoch=epoch_idx)
                self.logger.flush()

    def train_epoch(self, epoch):
        # Train model for one epoch, and log avg loss
        losses = []
        for batch in train_loader:
            self.state, loss = self.train_step(batch)
            losses.append(loss)
        losses_np = np.stack(jax.device_get(losses))
        avg_loss = losses_np.mean()
        self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss = self.eval_step(self.state, batch)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f'cifar10_{self.latent_dim}_', step=step)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f'cifar10_{self.latent_dim}_')
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}.ckpt'))

def train_autoencoder(num_points, latent_dim):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(num_points=num_points, latent_dim=latent_dim)
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(num_epochs=500)
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)
    test_loss = trainer.eval_model(test_loader)
    # Bind parameters to model for easier inference
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, test_loss

# Transformations applied on each image => bring them into a numpy array
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = img / 255. * 2. - 1.
    return img

# For visualization, we might want to map JAX or numpy tensors back to PyTorch
def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    imgs = torch.from_numpy(imgs.astype(np.float32))
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class ParticleLeniaDataset(torch.utils.data.Dataset):
    def __init__(self, root, num_points, num_dims, limit=None):
        self.exs = []
        for root, dirs, files in os.walk(root, topdown=False):
            if 'params.json' not in files:
                continue
            config_path = os.path.join(root, 'params.json')
            data_path = os.path.join(root, 'points_history.npy')
            with open(config_path, 'r') as f:
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    continue
            if config['num_particles'] != num_points:
                continue
            data = np.load(data_path)
            if data.shape[2] != num_dims:
                continue
            self.exs.append((config_path, data_path))
            if limit is not None and len(self.exs) >= limit:
                break
        self.len = len(self.exs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        config_path, data_path = self.exs[idx]
        return np.load(data_path)


if __name__ == '__main__':
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "./lenia_data"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "./saved_models/particle_lenia_autoencoder"

    num_points = 100
    num_dims = 2

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = ParticleLeniaDataset(DATASET_PATH, num_points, num_dims, limit=100)
    train_set, val_set, test_set = torch.utils.data.random_split(train_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

    batch_size = 4
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate)

    latent_dim = 32
    trainer_ld, test_loss_ld = train_autoencoder(num_points, latent_dim)