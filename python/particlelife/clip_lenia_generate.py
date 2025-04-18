import collections
import functools
import os

import av
import einops
import imageio.v2 as iio
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
import transformers

from particle_lenia import step_f, Params, multi_step_scan_with_force, draw_particles, draw_multi_species_particles

class Embedder:
    def __init__(self):
        self.processor = transformers.AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def get_embeddings(self, images):
        inputs = self.processor(images=images, return_tensors="np")
        image_features = self.model.get_image_features(**inputs)
        return image_features

def to_storable(n_t):
    d = n_t._asdict()
    d = [{k: np.array(v[i]).tolist() for k, v in d.items()} for i in range(n_t.map_size.shape[0])]
    return d

@functools.partial(jax.jit, static_argnames=('num_particles', 'map_size', 'num_species', 'num_kernels', 'num_growth_funcs'))
def generate_lenia_video(key, num_particles, map_size, num_species, num_kernels, num_growth_funcs):
    # Initial parameters
    
    key, *subkeys = jax.random.split(key, 12)
    mu_k = jax.random.uniform(subkeys[0], (num_species, num_species, num_kernels), minval=1.0, maxval=5.0)
    sigma_k = jax.random.uniform(subkeys[1], (num_species, num_species, num_kernels), minval=0.5, maxval=3.0)

    mu_g = jax.random.uniform(subkeys[3], (num_species, num_growth_funcs), minval=-2.0, maxval=2.0)
    sigma_g = jax.random.uniform(subkeys[4], (num_species, num_growth_funcs), minval=0.1, maxval=1.0)

    w_k = jax.random.uniform(subkeys[6], (num_species, num_species, num_kernels), minval=-0.05, maxval=0.05)
    c_rep = jax.random.uniform(subkeys[7], (num_species, num_species), minval=0.5, maxval=3.0)

    dt = 0.1
    
    params = Params(
        mu_k=mu_k, 
        sigma_k=sigma_k, 
        w_k=w_k, 
        mu_g=mu_g, 
        sigma_g=sigma_g, 
        c_rep=c_rep,
        map_size=map_size
    )
    
    num_dims = 2
    key, *subkeys = jax.random.split(key, 3)
    x = jax.random.uniform(subkeys[0], [num_particles, num_dims], minval=0, maxval=map_size)
    species = jax.random.randint(subkeys[1], [num_particles], 0, num_species)

    carry, (trajectory, force) = multi_step_scan_with_force(params, x, species, dt, 20000)
    max_force = jp.max(jp.linalg.norm(force, axis=-1))
    # draw trajectory

    video = draw_multi_species_particles(trajectory, map_size, species, num_species, start=-3000, offset=1000)
    return params, video, max_force

def main():
    write_video = False

    embeddr = Embedder()
    
    batch_params = []
    batch_max_force = []
    batch_img_features = []
    embed_path = './data/particle_lenia_clip_embeddings.parquet.zstd'
    embed_backup_path = './data/particle_lenia_clip_embeddings_backup.parquet.zstd'
    if not os.path.exists(embed_path):
        df = pl.DataFrame()
    else:
        try:
            df = pl.read_parquet(embed_path)
        except:
            df = pl.read_parquet(embed_backup_path)
    pbar = tqdm()

    df_max_force = 20.0
    df = df.filter(pl.col('max_force') < df_max_force)

    key = jax.random.PRNGKey(len(df))

    batch_size = 10

    while True:
        key, *subkeys = jax.random.split(key, 6)
        num_species = int(jax.random.randint(subkeys[0], (), 1, 8))
        num_kernels = int(jax.random.randint(subkeys[1], (), 1, 3))
        num_growth_funcs = int(jax.random.randint(subkeys[2], (), 1, 3))
        num_particles = int(jax.random.choice(subkeys[3], jp.array([100, 200, 400]), ()))
        map_size = 20

        all_params, videos, max_force = jax.vmap(generate_lenia_video, in_axes=(0, None, None, None, None, None))(jax.random.split(key, batch_size), num_particles, map_size, num_species, num_kernels, num_growth_funcs)

        all_params = to_storable(all_params)
        all_params = [all_params[i] for i in range(batch_size) if max_force[i] < df_max_force]
        videos = videos[max_force < df_max_force]
        max_force = max_force[max_force < df_max_force]

        if len(all_params) == 0:
            continue

        if write_video:
            # Choose which rendering method to use
            # Process frames and save to video
            print("Rendering frames...")
            for video in videos:
                file_path = './outputs/particle_lenia.mp4'
                w = iio.get_writer(file_path, format='FFMPEG', mode='I', fps=30)#,
                                #    codec='h264_vaapi',
                                #    output_params=['-vaapi_device',
                                #                   '/dev/dri/renderD128',
                                #                   '-vf',
                                #                   'format=gray|nv12,hwupload'],
                                #    pixelformat='vaapi_vld')
                for frame in np.array(video):
                    w.append_data(frame)
                w.close()

        np_images = []
        for video in videos:
            for image in video:
                np_images.append(np.array(image))


        img_features = embeddr.get_embeddings(np_images)

        batch_params.extend(all_params)
        batch_img_features.extend([img_features[i*videos[0].shape[0]:i*videos[0].shape[0]+videos[0].shape[0]] for i in range(batch_size)])
        batch_max_force.extend(max_force.tolist())

        pbar.update(1)

        if len(batch_params) > batch_size * 2:
            new_df = pl.from_dict({
                'params': batch_params, 
                'img_features': batch_img_features,
                'num_particles': [num_particles] * len(batch_params),
                'max_force': batch_max_force
            }, schema_overrides={'img_features': pl.Array(pl.Float32, (3, 512))})
            new_df = new_df.filter(pl.col('max_force') < df_max_force)
            if len(new_df) > 0:
                df = pl.concat([df, new_df], how='diagonal_relaxed')
                batch_params = []
                batch_img_features = []
                batch_max_force = []

        if len(df) % (10 * batch_size) == 0:
            df.write_parquet(embed_path)
        if len(df) % (100 * batch_size) == 0:
            df.write_parquet(embed_backup_path)

if __name__ == '__main__':
    main()
