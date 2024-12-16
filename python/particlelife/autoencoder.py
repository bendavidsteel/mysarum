import json
import os

from clu import metrics
import flax
from flax import linen as nn
from flax.training import checkpoints, train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ott.geometry import pointcloud
from ott.solvers import linear
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

@jax.jit
def reg_ot_cost(x: jnp.ndarray, y: jnp.ndarray) -> float:
    geom = pointcloud.PointCloud(x, y)
    ot = linear.solve(geom)
    return ot.reg_ot_cost


class PointCloudEncoder(nn.Module):
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

class PointCloudDecoder(nn.Module):
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

class PointCloudAutoencoder(nn.Module):
    """
    Complete autoencoder for point clouds.
    """
    latent_dim: int
    seq_len: int
    num_points: int
    num_dims: int

    def setup(self):
        self.encoder = PointCloudEncoder(latent_dim=self.latent_dim)
        self.decoder = PointCloudDecoder(seq_len=self.seq_len, num_points=self.num_points, num_dims=self.num_dims)

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


@flax.struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, rng, init_example, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, init_example)['params'] # initialize parameters by passing a template image
    tx = optax.adamw(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty())

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        reconstructed = state.apply_fn({'params': params}, batch)
        # Here you could use Chamfer distance or OT-based loss
        loss = jax.vmap(jax.vmap(reg_ot_cost))(reconstructed, batch)
        return jnp.mean(loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Eval function
@jax.jit
def eval_step(state, batch):
    reconstructed = state.apply_fn({'params': state.params}, batch)
    loss = jax.vmap(jax.vmap(reg_ot_cost))(reconstructed, batch)
    loss = jnp.mean(loss)
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state, loss

# class TrainerModule:
#     def __init__(self, num_points, latent_dim, lr=1e-3, seed=42):
#         super().__init__()
#         self.num_points = num_points
#         self.latent_dim = latent_dim
#         self.lr = lr
#         self.seed = seed
#         # Prepare logging
#         self.exmp_imgs = next(iter(val_loader))[0][:8]
#         self.log_dir = os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}')
#         self.generate_callback = GenerateCallback(self.exmp_imgs, every_n_epochs=50)
#         self.logger = SummaryWriter(log_dir=self.log_dir)
#         # Initialize model
#         self.init_model()



def train_model(state, log_dir, num_epochs=500):
    # Train model for defined number of epochs
    best_eval = 1e6
    for epoch_idx in range(1, num_epochs+1):
        state = train_epoch(state, epoch=epoch_idx)
        if epoch_idx % 10 == 0:
            state, eval_loss = eval_model(state, val_loader)
            print(f"Epoch {epoch_idx}, eval loss: {eval_loss:.4f}")
            # self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
            if eval_loss < best_eval:
                best_eval = eval_loss
                save_model(state, log_dir, step=epoch_idx)
            # self.generate_callback.log_generations(self.model, self.state, logger=self.logger, epoch=epoch_idx)
            # self.logger.flush()

def train_epoch(state, epoch):
    # Train model for one epoch, and log avg loss
    losses = []
    pbar = tqdm.tqdm(train_loader)
    pbar.set_description(f"Epoch {epoch}, loss: {0.0}")
    for batch in pbar:
        state, loss = train_step(state, batch)
        pbar.set_description(f"Epoch {epoch}, loss: {loss:.4f}")
        losses.append(loss)
    losses_np = np.stack(jax.device_get(losses))
    avg_loss = losses_np.mean()
    # self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
    return state

def eval_model(state, data_loader):
    # Test model on all images of a data loader and return avg loss
    losses = []
    batch_sizes = []
    for batch in data_loader:
        state, loss = eval_step(state, batch)
        losses.append(loss)
        batch_sizes.append(batch[0].shape[0])
    losses_np = np.stack(jax.device_get(losses))
    batch_sizes_np = np.stack(batch_sizes)
    avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
    return state, avg_loss

def save_model(state, log_dir, step=0):
    # Save current model at certain training iteration
    checkpoints.save_checkpoint(ckpt_dir=log_dir, target=state.params, prefix=f'plenia_{latent_dim}_', step=step)

def load_model(state, log_dir, pretrained=False):
    # Load model. We use different checkpoint for pretrained models
    if not pretrained:
        params = checkpoints.restore_checkpoint(ckpt_dir=log_dir, target=state.params, prefix=f'plenia_{latent_dim}_')
    else:
        params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'plenia_{latent_dim}.ckpt'), target=state.params)
    state = state.replace(params=params)
    return state
    
def checkpoint_exists():
    # Check whether a pretrained model exist for this autoencoder
    return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'plenia_{latent_dim}.ckpt'))

def train_autoencoder(log_dir, seq_len, num_points, latent_dim, num_dims):
    model = PointCloudAutoencoder(seq_len=seq_len, num_points=num_points, latent_dim=latent_dim, num_dims=num_dims)
    learning_rate = 1e-3
    momentum = 0.9
    # Initialize model
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_example = next(iter(val_loader))
    state = create_train_state(model, init_rng, init_example, learning_rate, momentum)
    # Initialize learning rate schedule and optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=100,
        decay_steps=500*len(train_loader),
        end_value=1e-5
    )
    del init_rng  # Must not be used anymore.
    # Create a trainer module with specified hyperparameters
    # trainer = TrainerModule(num_points=num_points, latent_dim=latent_dim)
    state = train_model(state, log_dir, num_epochs=500)
    state = load_model(state, log_dir)
    test_loss = eval_model(test_loader)
    # Bind parameters to model for easier inference
    model_bd = model.bind({'params': state.params})
    return model_bd, test_loss

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
    def __init__(self, root, num_points, num_dims, sample=8, limit=None):
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
        self.sample = sample

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        config_path, data_path = self.exs[idx]
        data = np.load(data_path)
        if self.sample:
            idx = np.linspace(0, data.shape[0]-1, self.sample).astype(int)
            data = data[idx]
        return data


if __name__ == '__main__':
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "./lenia_data"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = os.path.join(os.getcwd(), "./saved_models/particle_lenia_autoencoder")
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    seq_len = 8
    num_points = 100
    num_dims = 2

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = ParticleLeniaDataset(DATASET_PATH, num_points, num_dims, sample=seq_len, limit=100)
    train_set, val_set, test_set = torch.utils.data.random_split(train_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

    batch_size = 4
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate)

    latent_dim = 32
    trainer_ld, test_loss_ld = train_autoencoder(CHECKPOINT_PATH, seq_len, num_points, latent_dim, num_dims)