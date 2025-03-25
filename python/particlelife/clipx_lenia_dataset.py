import collections
import functools
import json
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
from tqdm import tqdm
import transformers

from particle_lenia import draw_particles


def to_storable(n_t):
    d = n_t._asdict()
    d = {k: np.array(v) for k, v in d.items()}
    return d

def main():
    write_video = False

    key = jax.random.PRNGKey(7)

    processor = transformers.AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model = transformers.AutoModel.from_pretrained("microsoft/xclip-base-patch32")

    batch_params = []
    batch_embeddings = []
    batch_dir_paths = []

    df_path = './data/lenia_dataset_embeddings.parquet.zstd'
    if os.path.exists(df_path):
        df = pl.read_parquet(df_path)
    else:
        df = pl.DataFrame()

    batch_size = 10

    dir_names = os.listdir('./lenia_data')
    dir_names.remove('run_')
    dir_paths = [os.path.join('./lenia_data', dir_name) for dir_name in dir_names]
    dir_paths += [os.listdir(os.path.join('./lenia_data/run_', dir_name)) for dir_name in os.listdir('./lenia_data/run_')]
        
    for dir_path in tqdm(dir_paths):
        while not os.path.exists(os.path.join(dir_path, 'params.json')):
            dir_path = os.path.join(dir_path, os.listdir(dir_path)[0])

        if 'dir_paths' in df and dir_path in df['dir_paths']:
            continue

        try:
            with open(os.path.join(dir_path, 'params.json')) as f:
                params = json.load(f)

            if params['num_dims'] != 2:
                continue

            trajectories = jp.load(os.path.join(dir_path, 'points_history.npy'))

            video = draw_particles(trajectories, start=-800, offset=100)

            if write_video:
                # Choose which rendering method to use
                # Process frames and save to video
                print("Rendering frames...")
                
                file_path = './outputs/particle_motus.mp4'
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

            inputs = processor(videos=list(video), return_tensors="pt")

            video_features = model.get_video_features(**inputs)

            video_features = video_features.detach().numpy().squeeze(0)
            batch_params.append(params)
            batch_embeddings.append(video_features)
            batch_dir_paths.append(dir_path)

            if len(batch_params) == 10:
                df = pl.concat([df, pl.DataFrame({'params': batch_params, 'embedding': batch_embeddings, 'dir_path': batch_dir_paths})], how='diagonal_relaxed')
                batch_params = []
                batch_embeddings = []
                batch_dir_paths = []

                df.write_parquet(df_path)
        except Exception as ex:
            print(ex)
            continue

if __name__ == "__main__":
    main()
