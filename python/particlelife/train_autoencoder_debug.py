import concurrent.futures
import itertools
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

from train_autoencoder import ParticleLeniaDataset, DataCollator, normalize, reg_ot_cost, create_autoencoder, create_train_state

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

def getitem(dir_name, tail, transform):
    data_path = os.path.join('/home/ndg/users/bsteel2/repos/mysarum/python/particlelife', 'lenia_data', dir_name, 'points_history.npy')
    with open(data_path, 'rb') as f:
        data = np.load(f)
    data = data[-tail:, :, :]
    if transform:
        data = transform(data)
    # TODO label as rotated point cloud
    return data

@hydra.main(config_path="conf", config_name="config")
def main(config):
    # TODO hook up learning rate scheduler


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
    train_dataset = ParticleLeniaDataset(
        dataset_path, 
        config.params.num_points, 
        config.params.num_dims, 
        tail=(config.params.num_samples + 1) * config.params.period,
        limit=None,
        transform=transform
    )


    train_set, val_set, test_set = torch.utils.data.random_split(train_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

    collator = DataCollator(config.params.num_samples, config.params.period)

    num_workers = 0
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=config.params.batch_size, 
        shuffle=True, 
        drop_last=False, 
        pin_memory=True, 
        num_workers=num_workers, 
        collate_fn=collator.train_collate, 
        persistent_workers=num_workers > 0
    )

    # dir_names = ['run_ahNKX5ufQFR/r9V9BAcnncyrCqfyCGzaI0khq5D4Fw8=', 'run_dpA8SodhWcRPMtjkVza4o7gW5hbUPdn39kQDlI5zqBo=', 'run_FdIbLrTRliszuOtI0f6tNMfHfT/Xhn3M9leJieuzNlU=', 'run_jlaO3UpKzT7zvJDNbNyPwf0cxFFu+780UTio+mz7K1w=', 'run_lsoRT4+HmWcrsNLfytO/yjZvCwxqB347t/pi0IvakZI=', 'run_zpUHwszi9jplTwqZ3HWzYzlyNODsasqMzthR7mnI7jQ=', 'run_2X+GAUWO8vL5gzyB1fkMxsONvWovV2qnjz6XBTcuzOI=', 'run_Wl8th8dG3OMwIsK2lsptKcTeNRhvUTeD1aDgrZ2lxP4=', 'run_ezZmvB03XulYdwDq01zmbbzG3WaKtxcUiX4xiHW8zSo=', 'run_42790/5U+QL5fzu2M2T9CGyH1LR5BD3ex/q0cLejDYk=', 'run_OZCygviiENax7JmqBk4FfZ8C0qrrzVk7ykML2eItIQE=', 'run_SgwkRqfSoXIF9ilrfrkJG6IfjoZH14MXh+eCBFWlp4A=', 'run_JKtEh6wiD+nTpzzZ05Fa+xLVRdjPtatgZUUXoUvQyq8=', 'run_Rxd8/CVxIBTR6CMhLvDWGj+eSlc/CXaAd1xgPZn1Bbg=', 'run_9478pzMaFQQYTi0F/q7uIgUuaooG6d4w/3mifIw+/3c=', 'run_W0eFwNPDBj7tsNsL41i80VvOrgoLiThfGSNgwMrXW5c=', 'run_0t0GtWGHHFlTDn/98Vdo4WKKP5zPpMZ9PxD69lbmUk4=', 'run_TtL/C9pzxJTqfVzlxGhfy29pKE91+UDyWMICauiaxOE=', 'run_bgcKVKsACMFRdoyLiPGmW9z8tpREhPFMRb71lB59KjU=', 'run_O/kJxN9vUXC2c85l1Epx/HetqHWX6Gi7oRshpk3sEHc=', 'run_YkNya9E64scLBzfY6taEZqhYmdQWZYKX+e4kDXiueBI=', 'run_6L/wd9NmBDkkuHqSIZ8CW1gd4MlgnxQxzafB4DsXH4s=', 'run_+XzJS0zxW01+KsxtYCFX0+d0fTr/j3HQRaIAPoGWRT0=', 'run_Q9AzZvO/yVWONBYW1MfT/Jlmhq/hkkZ5z7crUZ7ppQ8=', 'run_LHr28Z/p1BdGPGux480VBV10PPFebaobk4INR4CKEn8=', 'run_vEdSdGdcyUC111g3/ptc6raNkeDKppgDJvl+G2h+IPU=', 'run_e8OoIOSmDItTGhxRMEoX2ogK7AMf79V6nJQKcUeGTG0=', 'run_L7khhSpA2yAehGPwrhK1RlngMyZVfbdQVG0Odftuews=', 'run_ndsfdoyN8cySFRtvEFxj6L04qxkpFx2+Gv3nrC8o0L4=', 'run_cea5vNnric2yQ5RnRx8IH+nnaCF1DvxEvh9vmC3/cjo=', 'run_JQVdsF86b5ty8bRlSb68tRRGuZw331f3LevDx+3HSjQ=', 'run_Pvek+TvoCgQWBcCol4yLRZPj+fxFJaQsN+6XHe5/NRg=']
    # exs = [getitem(dir_name, (config.params.num_samples + 1) * config.params.period, transform) for dir_name in dir_names]
    # batch = collator.train_collate(exs)

    jax.config.update("jax_debug_nans", True)
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
    init_example = next(iter(train_loader))
    state = create_train_state(model, init_rng, init_example, 1 * config.params.num_epochs)
    
    
    del init_rng  # Must not be used anymore.

    jax.config.update("jax_debug_nans", True)
    for idx, batch in enumerate(train_loader):
        if idx == 182:
            state, loss = train_step(state, batch)
        if idx == 181:
            pass

    
    # state, loss = train_step(state, batch)


if __name__ == '__main__':
    main()