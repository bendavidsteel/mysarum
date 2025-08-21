import os
import subprocess

import cv2
import jax
import jax.numpy as jp
import matplotlib.cm as cm
import numpy as np
import polars as pl
import scipy.io.wavfile
from tqdm import tqdm

from particle_lenia import Params, multi_step_scan_with_force_and_energy, fancy_draw_particles, sonify_particles, generate_colors
from viz_clusters import get_life_like_scores

SAMPLE_RATE = 44100

def main():
    num_dims = 2

    # Output directory for individual videos
    output_dir = f'./lifelike_videos/{num_dims}d'
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of examples to save
    num_examples = 16  # You can adjust this number as needed
    
    # Animation parameters
    fps = 24
    total_frames = 400   # Total number of animation frames
    
    # Simulation parameters
    dt = 0.1
    
    # Classification threshold for life-likeness (can be set lower to include more examples if needed)
    life_like_threshold = 0.3
    
    # Load the dataframe with embeddings
    if num_dims == 3:
        df_path = './data/particle_lenia_3d_clip_embeddings.parquet.zstd'
    elif num_dims == 2:
        df_path = './data/particle_lenia_clip_embeddings1.parquet.zstd'
    else:
        raise ValueError("num_dims must be either 2 or 3")
    
    df = pl.read_parquet(df_path)
    print(f"Loaded dataframe with {len(df)} examples from {df_path}")
    df = df.sort('median_force').tail(int(0.3 * len(df)))

    most_life_like_embedding_idx, max_life_like_scores, life_like_indices, embeddings = get_life_like_scores(df, life_like_threshold)
    df = df.with_columns(pl.Series(name='max_life_like_score', values=max_life_like_scores))

    # Select the top N most life-like examples
    print(f"Selecting top {num_examples} most life-like examples...")
    
    # Sort indices by their life-like scores in descending order
    df = df.sort('max_life_like_score', descending=True).head(num_examples)
    print(f"Selected {len(df)} examples with life scores ranging from {df['max_life_like_score'].min():.3f} to {df['max_life_like_score'].max():.3f}")

    # Store exemplars and their scores
    
    print("Initializing simulations for each exemplar...")
    for i, exemplar in tqdm(enumerate(df.to_dicts()), desc="Simulating exemplars"):
        # Get the parameters from the dataframe
        params_dict = exemplar['params']

        # Extract parameters
        mu_k = np.array(params_dict['mu_k'])
        sigma_k = np.array(params_dict['sigma_k'])
        w_k = np.array(params_dict['w_k'])
        mu_g = np.array(params_dict['mu_g'])
        sigma_g = np.array(params_dict['sigma_g'])
        c_rep = np.array(params_dict['c_rep'])
        map_size = params_dict.get('map_size', 20)
        
        # Determine dimensions from parameter shapes
        num_species = mu_k.shape[0]  # First dimension of mu_k gives num_species
        
        # Convert to JAX arrays
        mu_k = jp.array(mu_k)
        sigma_k = jp.array(sigma_k)
        w_k = jp.array(w_k)
        mu_g = jp.array(mu_g)
        sigma_g = jp.array(sigma_g)
        c_rep = jp.array(c_rep)
        # Make sure map_size is a scalar value
        map_size = 20
        
        # Create Params object
        params = Params(
            mu_k=mu_k,
            sigma_k=sigma_k,
            w_k=w_k,
            mu_g=mu_g,
            sigma_g=sigma_g,
            c_rep=c_rep
        )
        
        # Get the number of particles from the dataframe if available
        num_particles = exemplar['num_particles']
        max_force = exemplar['max_force']

        # Generate initial particles and species
        key = jax.random.PRNGKey(42)  # Using exemplar_idx as part of seed for reproducibility
        key, *subkeys = jax.random.split(key, 3)
        
        points = jax.random.uniform(subkeys[0], [num_particles, num_dims], minval=0.0, maxval=1.0) * map_size
        species = jax.random.randint(subkeys[1], [num_particles], 0, num_species)
        
        num_frames = fps * 15
        
        # Create color map for species
        colours = generate_colors(num_species)

        _, (trajectories, force, energy) = multi_step_scan_with_force_and_energy(params, points, species, dt, num_frames, map_size)

        device = jax.devices('cpu')[0]

        with jax.default_device(device):
            audio = sonify_particles(trajectories, energy, force, species, num_species, map_size, fps=fps, sample_rate=SAMPLE_RATE)
            audio = np.asarray(audio)

        # Save the animation as MP4
        life_score = exemplar['max_life_like_score']
        output_filename = os.path.join(output_dir, f'example_{i+1}_score_{life_score:.4f}.mp4')
        
        temp_video = './outputs/temp_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'h264' if available
        width = 800
        height = 800
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        batch_size = 1
        pbar = tqdm(total=num_frames, desc='Writing frames')
        for i in range(0, num_frames, batch_size):
            batch_frames = fancy_draw_particles(
                trajectories[i:i+batch_size],
                energy[i:i+batch_size],
                map_size,
                species,
                colours,
                start=0,
                offset=1
            )
            for j in range(batch_frames.shape[0]):
                pbar.update(1)
                frame = np.asarray(batch_frames[j])
                # Convert to BGR if needed (OpenCV uses BGR)
                if frame.shape[-1] == 3:  # RGB to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
        out.release()

        # Write audio to temporary file
        temp_audio = './outputs/temp_audio.wav'
        scipy.io.wavfile.write(temp_audio, SAMPLE_RATE, audio.T)

        # Mux video and audio with ffmpeg
        subprocess.run([
            'ffmpeg', '-i', temp_video, '-i', temp_audio,
            '-c:v', 'copy', '-c:a', 'aac', '-shortest',
            output_filename, '-y'
        ])
    
    print(f"Completed processing {len(df)} top life-like examples. Files saved to {output_dir}")

if __name__ == "__main__":
    main()