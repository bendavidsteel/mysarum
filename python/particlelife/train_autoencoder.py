import json
import os

from clu import metrics
import flax
from flax import linen as nn
from flax.training import train_state
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from ott.geometry import pointcloud
from ott.solvers import linear
import torch
from torchvision import transforms
import tqdm
import wandb

from autoencoders import PointCloudNNAutoencoder, PointTransformerAutoencoder

@jax.jit
def reg_ot_cost(x: jnp.ndarray, y: jnp.ndarray) -> float:
    geom = pointcloud.PointCloud(x, y)
    ot = linear.solve(geom)
    return ot.reg_ot_cost

# Example usage
def create_autoencoder(model_type, **kwargs):
    if model_type == 'nn':
        model = PointCloudNNAutoencoder(**kwargs)
    elif model_type == 'transformer':
        model = PointTransformerAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

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

def create_train_state(module, rng, init_example, num_steps):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.array(init_example))['params'] # initialize parameters by passing a template image
    momentum = 0.9
    # Initialize learning rate schedule and optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=100,
        decay_steps=num_steps,
        end_value=1e-5
    )
    max_grad_norm = 1.0  # adjust this value based on your needs
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(lr_schedule, momentum)
    )
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



def train_model(state, train_loader, val_loader, checkpoint_manager, save_args, num_epochs=500, eval_every=1):
    # Train model for defined number of epochs
    best_eval = 1e6
    for epoch_idx in range(1, num_epochs+1):
        state = train_epoch(state, train_loader, epoch=epoch_idx)
        if epoch_idx % eval_every == 0:
            state, eval_loss = eval_model(state, val_loader)
            print(f"Epoch {epoch_idx}, eval loss: {eval_loss:.4f}")
            wandb.log({"eval_loss": eval_loss})
            # self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
            if eval_loss < best_eval:
                best_eval = eval_loss
                ckpt = {'model': state}
                checkpoint_manager.save(epoch_idx, ckpt, args=save_args)
            # self.generate_callback.log_generations(self.model, self.state, logger=self.logger, epoch=epoch_idx)
            # self.logger.flush()

    return state

def train_epoch(state, train_loader, epoch):
    # Train model for one epoch, and log avg loss
    losses = []
    pbar = tqdm.tqdm(train_loader)
    pbar.set_description(f"Epoch {epoch}, loss: {0.0}")
    for batch in pbar:
        state, loss = train_step(state, batch)
        wandb.log({"train_loss": loss})
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

def load_model(state, checkpoint_manager: orbax.checkpoint.CheckpointManager):
    # Load model. We use different checkpoint for pretrained models
    restore_args = orbax.checkpoint.args.StandardRestore(state)
    latest_step = checkpoint_manager.latest_step()
    target = {'model': state}
    restored_state = checkpoint_manager.restore(latest_step, args=restore_args, items=target)
    return restored_state
    
def train_autoencoder(train_loader, val_loader, test_loader, config):
    model = create_autoencoder(
        config.model.model_type,
        seq_len=config.params.num_samples,
        num_points=config.params.num_points,
        latent_dim=config.params.latent_dim,
        num_dims=config.params.num_dims,
        encoder_dim=config.params.encoder_dim,
        encoder_num_layers=config.params.encoder_num_layers,
        decoder_dim=config.params.decoder_dim,
    )
    # Initialize model
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_example = next(iter(val_loader))
    # log model architecture to wandb
    rng, t_rng = jax.random.split(rng)
    wandb.config.update({"model_architecture": model.tabulate(t_rng, jnp.array(init_example))})
    
    state = create_train_state(model, init_rng, init_example, len(train_loader) * config.params.num_epochs)
    
    del init_rng  # Must not be used anymore.

    # Set up the checkpointer.
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_path = os.path.join(this_dir_path, config.paths.checkpoint)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_path, options=options)
    save_args = orbax.checkpoint.args.StandardSave(state)

    state = train_model(state, train_loader, val_loader, checkpoint_manager, save_args, num_epochs=config.params.num_epochs)
    state = load_model(state, checkpoint_manager)
    state, test_loss = eval_model(state, test_loader)
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
    
def train_collate(batch):
    batch = numpy_collate(batch)
    # TODO randomly rotate the point cloud
    # Random rotation
    theta = np.random.uniform(0, 2 * np.pi, size=batch.shape[0])
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]).transpose(2, 0, 1)
    batch = batch @ rotation_matrix[:, np.newaxis, :, :]

    # TODO data augment by changing the sampled seq points
    return batch


class ParticleLeniaDataset(torch.utils.data.Dataset):
    def __init__(self, root, num_points, num_dims, sample=8, limit=None, transform=None):
        self.exs = []
        if limit is not None:
            total = limit
        else:
            total = len(list(os.listdir(root)))
        pbar = tqdm.tqdm(total=total, desc="Loading dataset")
        for root, dirs, files in os.walk(root, topdown=False):
            pbar.update(1)
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
            if 'num_dims' not in config:
                data = np.load(data_path)
                num_dims = data.shape[-1]
                config['num_dims'] = num_dims
                with open(config_path, 'w') as f:
                    json.dump(config, f)
            if config['num_dims'] != num_dims:
                continue
            self.exs.append((config_path, data_path))
            if limit is not None and len(self.exs) >= limit:
                break
        self.len = len(self.exs)
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        config_path, data_path = self.exs[idx]
        data = np.load(data_path)
        if self.sample:
            idx = np.linspace(0, data.shape[0]-1, self.sample).astype(int)
            data = data[idx]
        if self.transform:
            data = self.transform(data)
        # TODO label as rotated point cloud
        return data

def normalize(x):
    # normalizing per time step
    return (x - x.mean(axis=(1,2)).reshape(-1, 1, 1)) / x.std(axis=(1,2)).reshape(-1, 1, 1)

@hydra.main(config_path="conf", config_name="config")
def main(config):
    # TODO hook up learning rate scheduler


    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    dataset_path = os.path.join(this_dir_path, config.paths.dataset)
    # Path to the folder where the pretrained models are saved
    checkpoint_path = os.path.join(this_dir_path, config.paths.checkpoint)
    os.makedirs(checkpoint_path, exist_ok=True)

    wandb.init(
        # set the wandb project where this run will be logged
        project="particle-lenia-autoencoder",

        # track hyperparameters and run metadata
        config=dict(config)
    )

    transform = transforms.Compose([
        transforms.Lambda(normalize),  # Center
    ])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = ParticleLeniaDataset(
        dataset_path, 
        config.params.num_points, 
        config.params.num_dims, 
        sample=config.params.num_samples, 
        limit=None,
        transform=transform
    )
    train_set, val_set, test_set = torch.utils.data.random_split(train_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

    num_workers = 1
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=config.params.batch_size, 
        shuffle=True, 
        drop_last=False, 
        pin_memory=True, 
        num_workers=num_workers, 
        collate_fn=train_collate, 
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=config.params.batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers, 
        collate_fn=numpy_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=config.params.batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers, 
        collate_fn=numpy_collate
    )

    trainer_ld, test_loss_ld = train_autoencoder(train_loader, val_loader, test_loader, config)
    # wandb.finish()

if __name__ == '__main__':
    main()