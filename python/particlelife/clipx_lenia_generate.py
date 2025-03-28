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

from particle_lenia import step_f, Params, multi_step_scan, draw_particles, draw_multi_species_particles

class Embedder:
    def __init__(self):
        self.processor = transformers.AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.model = transformers.AutoModel.from_pretrained("microsoft/xclip-base-patch32")

    def get_embeddings(self, videos):
        inputs = self.processor(videos=videos, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        pixel_values = inputs["pixel_values"]
        output_attentions = self.model.config.output_attentions
        output_hidden_states = self.model.config.output_hidden_states
        return_dict = self.model.config.use_return_dict

        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=False,
            return_dict=return_dict,
        )

        video_embeds = vision_outputs[1]
        video_embeds = self.model.visual_projection(video_embeds)

        cls_features = video_embeds.view(batch_size, num_frames, -1)

        mit_outputs = self.model.mit(
            cls_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        video_embeds = mit_outputs[1]

        img_features = vision_outputs[0][:, 1:, :]
        img_features = self.model.prompts_visual_layernorm(img_features)
        img_features = img_features @ self.model.prompts_visual_projection
        img_features = img_features.view(batch_size, num_frames, -1, video_embeds.shape[-1])
        img_features = img_features.mean(dim=1, keepdim=False)

        video_embeds = video_embeds.cpu().detach().numpy()
        img_features = img_features.cpu().detach().numpy()

        return video_embeds, img_features

def to_storable(n_t):
    d = n_t._asdict()
    d = [{k: np.array(v[i]).tolist() for k, v in d.items()} for i in range(n_t.map_size.shape[0])]
    return d

@functools.partial(jax.jit, static_argnames=('num_particles', 'map_size', 'num_species', 'num_kernels', 'num_growth_funcs'))
def generate_lenia_video(key, num_particles, map_size, num_species, num_kernels, num_growth_funcs):
    # Initial parameters
    
    key, *subkeys = jax.random.split(key, 12)
    mu_k = jax.random.uniform(subkeys[0], (num_species, num_species, num_kernels), minval=2.0, maxval=5.0)
    sigma_k = jax.random.uniform(subkeys[1], (num_species, num_species, num_kernels), minval=0.5, maxval=3.0)

    mu_g = jax.random.uniform(subkeys[3], (num_species, num_growth_funcs), minval=-2.0, maxval=2.0)
    sigma_g = jax.random.uniform(subkeys[4], (num_species, num_growth_funcs), minval=0.05, maxval=1.0)

    w_k = jax.random.uniform(subkeys[6], (num_species, num_species, num_kernels), minval=-0.1, maxval=0.1)
    c_rep = jax.random.uniform(subkeys[7], (num_species, num_species), minval=0.1, maxval=2.0)

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

    carry, trajectory = multi_step_scan(params, x, species, dt, 20000)

    # draw trajectory

    video = draw_multi_species_particles(trajectory, map_size, species, num_species, start=-8000, offset=1000)
    return params, video

def main():
    write_video = False

    key = jax.random.PRNGKey(7)

    embeddr = Embedder()
    
    batch_params = []
    batch_embeddings = []
    batch_img_features = []
    embed_path = './data/particle_lenia_embeddings.parquet.zstd'
    embed_backup_path = './data/particle_lenia_embeddings_backup.parquet.zstd'
    if not os.path.exists(embed_path):
        df = pl.DataFrame()
    else:
        try:
            df = pl.read_parquet(embed_path)
        except:
            df = pl.read_parquet(embed_backup_path)
    pbar = tqdm()

    batch_size = 10

    while True:
        key, *subkeys = jax.random.split(key, 6)
        num_species = int(jax.random.randint(subkeys[0], (), 1, 8))
        num_kernels = int(jax.random.randint(subkeys[1], (), 1, 3))
        num_growth_funcs = int(jax.random.randint(subkeys[2], (), 1, 3))
        num_particles = int(jax.random.choice(subkeys[3], jp.array([100, 200, 400]), ()))
        map_size = 20

        all_params, videos = jax.vmap(generate_lenia_video, in_axes=(0, None, None, None, None, None))(jax.random.split(key, batch_size), num_particles, map_size, num_species, num_kernels, num_growth_funcs)

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

        torch_videos = []
        for video in videos:
            torch_video = []
            for image in video:
                torch_video.append(torch.tensor(np.array(image)))
            torch_videos.append(torch_video)


        video_features, img_features = embeddr.get_embeddings(torch_videos)

        batch_params.extend(to_storable(all_params))
        batch_embeddings.extend(list(video_features))
        batch_img_features.extend(list(img_features))

        pbar.update(1)

        if len(batch_params) == 10:
            df = pl.concat([df, pl.from_dict({
                'params': batch_params, 
                'video_embedding': batch_embeddings, 
                'img_features': batch_img_features,
                'num_particles': [num_particles] * batch_size
            })], how='diagonal_relaxed')
            batch_params = []
            batch_embeddings = []
            batch_img_features = []

        df.write_parquet(embed_path)
        if len(df) % 100 == 0:
            df.write_parquet(embed_backup_path)

if __name__ == '__main__':
    main()
