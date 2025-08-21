import os
import evoc
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import jax
import jax.numpy as jp
import torch
import transformers
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from particle_lenia import Params, multi_step_scan

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor

def calculate_diversity_score(embeddings):
    """
    Calculate diversity score based on pairwise distances between embeddings.
    Higher score means more diverse set of embeddings.
    """
    if len(embeddings) <= 1:
        return 0
    
    # Calculate pairwise distances
    pairwise_distances = pdist(embeddings)
    # Return average distance as diversity score
    return np.mean(pairwise_distances)

def select_diverse_subset(embeddings, scores, k, initial_indices=None):
    """
    Select a diverse subset of k embeddings that also have high scores.
    Uses a greedy approach to maximize diversity.
    
    Args:
        embeddings: Array of embeddings
        scores: Array of lifelikeness scores
        k: Number of embeddings to select
        initial_indices: Optional list of indices to start with
    
    Returns:
        List of selected indices
    """
    if len(embeddings) <= k:
        return list(range(len(embeddings)))
    
    # Normalize scores to [0, 1]
    norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
    
    # Calculate pairwise distances between all embeddings
    distances = squareform(pdist(embeddings))
    
    selected = []
    # Start with highest scoring examples if no initial indices provided
    if initial_indices is None or len(initial_indices) == 0:
        # Take top 3 by score to ensure we have high quality examples
        top_by_score = np.argsort(norm_scores)[-3:]
        selected.extend(top_by_score)
    else:
        selected.extend(initial_indices)
    
    # Make sure we don't exceed k
    selected = selected[:k]
    
    # Greedy selection for remaining slots
    remaining = k - len(selected)
    if remaining > 0:
        candidates = list(set(range(len(embeddings))) - set(selected))
        
        for _ in range(remaining):
            if not candidates:
                break
                
            # For each candidate, compute the minimum distance to any selected point
            # Weight this by the normalized score to favor higher scoring examples
            best_score = -1
            best_idx = -1
            
            for idx in candidates:
                if len(selected) == 0:
                    min_dist = 1.0  # If nothing selected yet, use a default value
                else:
                    min_dist = np.min([distances[idx, sel_idx] for sel_idx in selected])
                
                # Combined score: balance between diversity and lifelikeness
                # Adjustable weights (0.7 for diversity, 0.3 for lifelikeness)
                combined_score = 0.7 * min_dist + 0.3 * norm_scores[idx]
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                candidates.remove(best_idx)
    
    return selected

def get_life_like_scores(df, life_like_threshold):

    # Initialize CLIP model for life-like classification
    print("Loading CLIP model for classification...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = transformers.AutoModel.from_pretrained("openai/clip-vit-base-patch32", device_map=device, torch_dtype=torch.float16)
    
    # Define prompts for life-like classification
    life_like_prompts = [
        "living organisms",
        # "biological cells",
        # "a cell",
        # "chemical reaction",
        "multicellular organisms",
        "plants",
        "roots",
        "fungi",
        "mould",
        "plankton",
        "deep sea creatures"
    ]
    
    # Encode the prompts
    print("Encoding classification prompts...")
    inputs = tokenizer(life_like_prompts, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_features = model.get_text_features(**inputs)
    
    # Extract embeddings and image features from the dataframe
    embeddings = df['img_features'].to_numpy().astype(np.float16)
    
    # Classify all instances for life-likeness
    print("Classifying instances for life-likeness...")
    batch_size = 32
    life_like_scores = []
    
    # Classification function
    def classify_life_likeness(batch_embeddings):
        image_embeds = torch.tensor(batch_embeddings).to(device)
        
        # Get text embeddings
        text_embeds = text_features
        batch_size = batch_embeddings.shape[0]
        text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # normalized features
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        # cosine similarity as logits
        logits_per_text = torch.einsum("btd,bkd->bkt", text_embeds, image_embeds)
        logits_per_text = logits_per_text * model.logit_scale.exp().to(text_embeds.device)
        
        return torch.mean(logits_per_text, dim=-1).cpu().detach().numpy()
    
    # Process in batches
    for i in tqdm(range(0, len(embeddings), batch_size)):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_scores = classify_life_likeness(batch_embeddings)
        life_like_scores.extend(batch_scores)
    
    # Convert to numpy array and get max of snapshots
    life_like_scores = np.array(life_like_scores)
    most_life_like_embedding_idx = np.argmax(life_like_scores, axis=-1)
    max_life_like_scores = np.mean(life_like_scores, axis=-1)
    
    # Filter instances with life-like score above threshold
    life_like_indices = np.where(max_life_like_scores > life_like_threshold)[0]
    print(f"Found {len(life_like_indices)} instances with life-like score > {life_like_threshold}")
    
    if len(life_like_indices) == 0:
        print("No life-like instances found. Try lowering the threshold.")
        return
    
    return most_life_like_embedding_idx, max_life_like_scores, life_like_indices, embeddings

def main():
    # Maximum number of clusters to visualize
    max_clusters_to_show = 16  # You can adjust this number as needed
    
    # Animation parameters
    fps = 24
    save_animation = True
    
    # Simulation parameters
    dt = 0.1
    total_frames = 400   # Total number of animation frames
    
    # Classification threshold for life-likeness
    life_like_threshold = 0.3
    max_force = 5
    min_max_force = 0.5

    num_dims = 2
    
    # Load the dataframe with embeddings
    if num_dims == 3:
        df_path = './data/particle_lenia_3d_clip_embeddings.parquet.zstd'
    elif num_dims == 2:
        df_path = './data/particle_lenia_clip_embeddings1.parquet.zstd'
    else:
        raise ValueError("num_dims must be either 2 or 3")
    
    df = pl.read_parquet(df_path)
    df = df.filter((pl.col('max_force') < max_force) & (pl.col('max_force') > min_max_force))

    most_life_like_embedding_idx, max_life_like_scores, life_like_indices, embeddings = get_life_like_scores(df, life_like_threshold)

    # Filter embeddings to only include the most life-like snapshot for each instance
    filtered_embeddings = np.take_along_axis(embeddings, most_life_like_embedding_idx.reshape(-1, 1, 1), 1).squeeze(1)[life_like_indices]
    filtered_scores = max_life_like_scores[life_like_indices]
    
    # Cluster the filtered embeddings using EVoC
    print("Clustering life-like instances...")
    clusterer = evoc.EVoC(base_min_cluster_size=1, n_neighbors=15)
    cluster_labels = clusterer.fit_predict(filtered_embeddings)
    
    # Get the number of clusters and their members
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)
    print(f"Found {num_clusters} clusters")
    
    # Count samples per cluster and calculate average life-likeness per cluster
    cluster_counts = {}
    cluster_avg_scores = {}
    cluster_embeddings = {}
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise points
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_counts[cluster_id] = np.sum(cluster_mask)
        cluster_avg_scores[cluster_id] = np.mean(filtered_scores[cluster_mask])
        cluster_embeddings[cluster_id] = filtered_embeddings[cluster_mask]
    
    print("Samples per cluster:", cluster_counts)
    print("Average life-likeness per cluster:", {k: f"{v:.3f}" for k, v in cluster_avg_scores.items()})
    
    # Sort clusters by average life-likeness score
    valid_clusters = [c for c in unique_clusters if c != -1]
    
    # If we have more clusters than we can display, select a diverse subset
    if len(valid_clusters) > max_clusters_to_show:
        print(f"Found {len(valid_clusters)} valid clusters, selecting {max_clusters_to_show} most life-like clusters")
        
        # First, select top 5 clusters by average life-likeness score
        clusters_to_show = sorted(valid_clusters, key=lambda c: cluster_avg_scores[c], reverse=True)[:max_clusters_to_show]
        
        print(f"Selected {len(clusters_to_show)} clusters")
    else:
        # Sort by average life-likeness if we don't need to select a subset
        clusters_to_show = sorted(valid_clusters, key=lambda c: cluster_avg_scores[c], reverse=True)
    
    # Find exemplars for each selected cluster using membership strengths
    exemplars = []
    selected_cluster_counts = {}
    selected_cluster_scores = {}
    
    # Get membership strengths for each point
    membership_strengths = clusterer.membership_strengths_

    for i, cluster_id in enumerate(clusters_to_show):
        # Find all points in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) > 0:
            # Get the membership strengths for this cluster
            cluster_strengths = membership_strengths[cluster_indices]
            
            # Find the point with the highest membership strength
            strongest_point_idx = cluster_indices[np.argmax(cluster_strengths)]
            
            # Map back to the original index in the dataframe
            original_idx = life_like_indices[strongest_point_idx]
            exemplars.append(original_idx)
            selected_cluster_counts[i] = cluster_counts[cluster_id]
            selected_cluster_scores[i] = cluster_avg_scores[cluster_id]
    
    # Initialize parameters and initial states for each exemplar
    params_list = []
    positions_list = []
    species_list = []
    map_sizes = []
    num_species_list = []
    max_force_list = []
    
    print("Initializing simulations for each exemplar...")
    for exemplar_idx in tqdm(exemplars):
        # Get the parameters from the dataframe
        params_dict = df['params'][int(exemplar_idx)]
        
        # Extract parameters
        mu_k = np.array(params_dict['mu_k'])
        sigma_k = np.array(params_dict['sigma_k'])
        w_k = np.array(params_dict['w_k'])
        mu_g = np.array(params_dict['mu_g'])
        sigma_g = np.array(params_dict['sigma_g'])
        c_rep = np.array(params_dict['c_rep'])
        map_size = 20
        
        # Determine dimensions from parameter shapes
        num_species = mu_k.shape[0]  # First dimension of mu_k gives num_species
        num_kernels = mu_k.shape[2] if len(mu_k.shape) > 2 else 1
        num_growth_funcs = mu_g.shape[1] if len(mu_g.shape) > 1 else 1
        
        # Convert to JAX arrays
        mu_k = jp.array(mu_k)
        sigma_k = jp.array(sigma_k)
        w_k = jp.array(w_k)
        mu_g = jp.array(mu_g)
        sigma_g = jp.array(sigma_g)
        c_rep = jp.array(c_rep)
        # Make sure map_size is a scalar value
        map_size = jp.array(map_size) if isinstance(map_size, (int, float)) else jp.array(map_size[0] if isinstance(map_size, list) else 20)
        
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
        num_particles = df['num_particles'][int(exemplar_idx)]
        max_force = df['max_force'][int(exemplar_idx)]
        
        # Generate initial particles and species
        key = jax.random.PRNGKey(42 + int(exemplar_idx))  # Using exemplar_idx as part of seed for reproducibility
        key, *subkeys = jax.random.split(key, 3)
        
        x = jax.random.uniform(subkeys[0], [num_particles, num_dims], minval=0, maxval=map_size)
        species = jax.random.randint(subkeys[1], [num_particles], 0, num_species)
        
        # Store the parameters and initial states
        params_list.append(params)
        positions_list.append(np.array(x))
        species_list.append(np.array(species))
        map_sizes.append(float(map_size))
        num_species_list.append(num_species)
        max_force_list.append(max_force)
    
    # Calculate grid dimensions for the plot
    grid_size = int(np.ceil(np.sqrt(len(clusters_to_show))))
    
    # Create the figure and axes for the grid
    plt.style.use('dark_background')
    if num_dims == 3:
        fig = plt.figure(figsize=(12, 12))
        axes = np.empty((grid_size, grid_size), dtype=object)
        for i in range(grid_size):
            for j in range(grid_size):
                ax = fig.add_subplot(grid_size, grid_size, i * grid_size + j + 1, projection='3d')
                axes[i, j] = ax
    elif num_dims == 2:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    # plt.suptitle(f"Life-like Cluster Exemplars", fontsize=16)
    
    # Flatten the axes array for easier indexing
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Convert to list if it's a single axis
    
    # Initialize scatter plots
    scatters = []
    for i, ax in enumerate(axes):
        if i < len(positions_list):
            # Set up the plot area
            map_size = map_sizes[i]
            ax.set_xlim(0, map_size)
            ax.set_ylim(0, map_size)
            avg_cluster_score = selected_cluster_scores[i]
            
            # Create title with cluster info
            # ax.set_title(f"Cluster #{clusters_to_show[i]} (n={selected_cluster_counts[i]})\n"
            #              f"Life Score: {life_score:.2f}, Avg: {avg_cluster_score:.2f}",
            #              fontsize=8)
            ax.axis('off')
            
            # Create color map for species
            num_species = num_species_list[i]
            colors = cm.rainbow(np.linspace(0, 1, num_species))
            
            # Get initial positions and species
            initial_positions = positions_list[i]
            species = species_list[i]
            
            # Create a list of scatter plots, one for each species
            species_scatters = []
            for s in range(num_species):
                species_mask = species == s
                if np.any(species_mask):
                    # Create a scatter plot for this species
                    if num_dims == 3:
                        scatter = ax.scatter(
                            initial_positions[species_mask, 0], 
                            initial_positions[species_mask, 1],
                            initial_positions[species_mask, 2],
                            color=colors[s],
                            s=10,  # Point size
                            alpha=0.8
                        )
                    elif num_dims == 2:
                        scatter = ax.scatter(
                            initial_positions[species_mask, 0], 
                            initial_positions[species_mask, 1],
                            color=colors[s],
                            s=10,  # Point size
                            alpha=0.8
                        )
                    species_scatters.append((scatter, species_mask))
                else:
                    species_scatters.append((None, None))
            
            scatters.append(species_scatters)
        else:
            # Hide unused subplots
            ax.axis('off')
    
    # Animation update function
    def update_frame(frame_idx):
        updates = []
        
        # Update each simulation for steps_per_frame steps
        for i, species_scatters in enumerate(scatters):
            if i < len(positions_list):
                # Get current positions and parameters
                positions = positions_list[i]
                params = params_list[i]
                species = species_list[i]
                max_force = max_force_list[i]

                # Number of simulation steps per animation frame
                steps_per_frame = max(int(35 - 4 * max_force), 1)
                
                # Run simulation steps using multi_step_scan
                carry, trajectory = multi_step_scan(params, jp.array(positions), jp.array(species), dt, steps_per_frame, map_size)
                
                # Update stored positions (using the last frame from trajectory)
                positions_list[i] = np.array(trajectory[-1])
                
                # Update scatter plots
                for (scatter, species_mask) in species_scatters:
                    if scatter is not None:
                        if num_dims == 3:
                            scatter._offsets3d = (
                                positions_list[i][species_mask, 0], 
                                positions_list[i][species_mask, 1],
                                positions_list[i][species_mask, 2]
                            )
                        elif num_dims == 2:
                            scatter.set_offsets(positions_list[i][species_mask])
                        updates.append(scatter)
        
        # Add a progress indicator
        # frame_progress = f"Frame: {frame_idx+1}/{total_frames}"
        # plt.suptitle(f"Life-like Cluster Exemplars", 
        #             fontsize=16)
        
        return updates
    
    # Create animation
    print("Creating animation...")
    animation = FuncAnimation(
        fig, 
        update_frame, 
        frames=total_frames,
        interval=1000/fps,  # interval in milliseconds
        blit=True
    )
    
    plt.tight_layout()
    
    # Save the animation as MP4
    if save_animation:
        try:
            print("Saving animation to MP4...")
            from matplotlib.animation import FFMpegWriter
            from tqdm import tqdm as tqdm_progress
            
            # Create a progress bar for animation saving
            progress_bar = tqdm_progress(total=total_frames, desc="Saving animation")
            
            # Callback function to update progress bar
            def progress_callback(i, n):
                progress_bar.update(1)
            
            # Create writer with progress callback
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=5000)
            animation.save(f'life_like_diverse_clusters_animation_max_force_{max_force}.mp4', 
                           writer=writer, progress_callback=progress_callback)
            
            # Close progress bar
            progress_bar.close()
            print("Animation saved successfully.")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            print("Trying to save as GIF instead...")
            try:
                # For GIF saving, wrap with a manual progress bar
                print("Saving as GIF. This may take a while...")
                with tqdm_progress(total=1, desc="Saving GIF") as pbar:
                    animation.save('life_like_diverse_clusters_animation.gif', writer='pillow', fps=fps)
                    pbar.update(1)
                print("Animation saved as GIF successfully.")
            except Exception as e:
                print(f"Failed to save GIF animation: {e}")
    
    # Show the animation
    print("Displaying animation. Close the window when done viewing.")
    plt.show()

if __name__ == "__main__":
    main()