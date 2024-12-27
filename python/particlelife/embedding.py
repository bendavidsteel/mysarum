import concurrent.futures
import itertools
import json
import os

from flax.training import train_state
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import torch
from torchvision import transforms
import tqdm

from autoencoders import PointCloudNNAutoencoder, PointTransformerAutoencoder

# Example usage
def create_autoencoder(model_type, **kwargs):
    if model_type == 'nn':
        model = PointCloudNNAutoencoder(**kwargs)
    elif model_type == 'transformer':
        model = PointTransformerAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def model_inference(model, params, data_loader):
    # Test model on all images of a data loader and return avg loss
    embeddings = []
    apply_fn = jax.jit(nn.apply(lambda m, x: m.embed(x), model))
    for batch in tqdm.tqdm(data_loader, desc='Inference'):
        batch = jnp.array(batch)
        embedding = apply_fn({'params': params}, batch)
        embeddings.append(embedding)
    embeddings = jnp.concatenate(embeddings)
    return embeddings

def load_model(checkpoint_manager: orbax.checkpoint.CheckpointManager):
    # Load model. We use different checkpoint for pretrained models
    restore_args = orbax.checkpoint.args.StandardRestore()
    latest_step = checkpoint_manager.latest_step()
    restored_state = checkpoint_manager.restore(latest_step, args=restore_args, items=None)
    return restored_state['params']
    
def get_embeddings(loader, config):
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
    init_example = next(iter(loader))
    
    del init_rng  # Must not be used anymore.

    # Set up the checkpointer.
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_path = os.path.join(this_dir_path, config.paths.checkpoint)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_path, options=options)

    params = load_model(checkpoint_manager)
    embeddings = model_inference(model, params, loader)
    # Bind parameters to model for easier inference
    
    return embeddings

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
class DataCollator:
    def __init__(self, sample, period):
        self.sample = sample
        self.period = period

    def test_collate(self, batch):
        batch = numpy_collate(batch)
        idx = np.arange(0, self.sample * self.period, self.period)
        return batch[:, idx, :]


class ParticleLeniaDataset(torch.utils.data.Dataset):
    def __init__(self, root, num_points, num_dims, tail=40, limit=None, transform=None):
        self.num_points = num_points
        self.num_dims = num_dims

        processed = 0
        self.exs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            pbar = tqdm.tqdm(desc="Loading dataset")
            
            for root_tuple in os.walk(root, topdown=False):
                if limit and processed >= limit:
                    break
                    
                future = executor.submit(self.process_directory, root_tuple)
                futures[future] = root_tuple
                processed += 1
                
                # Process completed tasks
                for future in list(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    if result is not None:
                        self.exs.append(result)
                    futures.pop(future)
                    pbar.update(1)
            
            # Process remaining tasks
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    self.exs.append(result)
                pbar.update(1)
        
        self.len = len(self.exs)
        self.tail = tail
        self.transform = transform

    def process_directory(self, root_dir_tuple):
        root, _, files = root_dir_tuple
        if 'params.json' not in files:
            return None
            
        config_path = os.path.join(root, 'params.json')
        data_path = os.path.join(root, 'points_history.npy')
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            if config['num_particles'] != self.num_points:
                return None
                
            if 'num_dims' not in config:
                data = np.load(data_path)
                config['num_dims'] = data.shape[-1]
                with open(config_path, 'w') as f:
                    json.dump(config, f)
                    
            if config['num_dims'] != self.num_dims:
                return None
                
            return (config_path, data_path)
            
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        config_path, data_path = self.exs[idx]
        data = np.load(data_path)
        data = data[-self.tail:, :, :]
        if self.transform:
            data = self.transform(data)
        # TODO label as rotated point cloud
        return data

def normalize(x):
    # normalizing per time step
    return (x - x.mean(axis=(1,2)).reshape(-1, 1, 1)) / x.std(axis=(1,2)).reshape(-1, 1, 1)

@hydra.main(config_path="conf", config_name="config")
def main(config):

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    dataset_path = os.path.join(this_dir_path, config.paths.dataset)
    # Path to the folder where the pretrained models are saved
    checkpoint_path = os.path.join(this_dir_path, config.paths.checkpoint)
    os.makedirs(checkpoint_path, exist_ok=True)

    # set random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.Lambda(normalize),  # Center
    ])

    # Loading the training dataset. We need to split it into a training and validation part
    dataset = ParticleLeniaDataset(
        dataset_path, 
        config.params.num_points, 
        config.params.num_dims, 
        tail=(config.params.num_samples + 1) * config.params.period,
        limit=None,
        transform=transform
    )
    collator = DataCollator(config.params.num_samples, config.params.period)

    num_workers = 4
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.params.batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers, 
        collate_fn=collator.test_collate
    )

    embeddings = get_embeddings(loader, config)

    jnp.save('embeddings.npy', embeddings)

if __name__ == '__main__':
    main()