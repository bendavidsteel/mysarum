import concurrent.futures
import itertools
import json
import os
from typing import Tuple, List

import hdbscan
import hydra
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms
import tqdm
import umap

from train_autoencoder import ParticleLeniaDataset, DataCollator, normalize

def process_embeddings(embeddings: np.ndarray, 
                        file_paths: List[str],
                      n_components: int = 2,
                      min_cluster_size: int = 5,
                      min_samples: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process embeddings through dimensionality reduction and clustering.
    
    Args:
        embeddings: Input embeddings array
        n_components: Number of UMAP components
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        
    Returns:
        Tuple of (reduced embeddings, cluster labels)
    """
    # remove nans
    file_paths = np.array(file_paths)
    file_paths = file_paths[~np.isnan(embeddings).any(axis=1)]
    embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # remove nans
    file_paths = file_paths[~np.isnan(embeddings_scaled).any(axis=1)]
    embeddings_scaled = embeddings_scaled[~np.isnan(embeddings_scaled).any(axis=1)]
    
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_components=n_components, 
                       random_state=42,
                       min_dist=0.1,
                       n_neighbors=30)
    embeddings_2d = reducer.fit_transform(embeddings_scaled)
    
    # Cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples,
                               metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings_2d)
    
    return embeddings_2d, cluster_labels, file_paths

def find_cluster_representatives(embeddings_2d: np.ndarray,
                               cluster_labels: np.ndarray,
                               original_data: np.ndarray) -> List[np.ndarray]:
    """
    Find representative sequences for each cluster by selecting the point
    closest to each cluster center.
    
    Args:
        embeddings_2d: Reduced dimensionality embeddings
        cluster_labels: Cluster assignments
        original_data: Original Lenia pattern sequences
        
    Returns:
        List of representative sequences for each cluster
    """
    unique_clusters = np.unique(cluster_labels)
    representatives = []
    
    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise points
            continue
            
        # Get cluster points
        cluster_mask = cluster_labels == cluster
        cluster_points = embeddings_2d[cluster_mask]
        
        # Find cluster center
        center = np.mean(cluster_points, axis=0)
        
        # Find point closest to center
        distances = np.linalg.norm(cluster_points - center, axis=1)
        representative_idx = np.where(cluster_mask)[0][np.argmin(distances)]
        
        representatives.append(original_data[representative_idx])
        
    return representatives

def create_animation(sequence: np.ndarray, 
                    filename: str,
                    fps: int = 10) -> None:
    """
    Create an animated gif of a Lenia pattern sequence.
    
    Args:
        sequence: Pattern sequence array of shape (timesteps, num_points, num_dims)
        filename: Output filename
        fps: Frames per second
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        points = sequence[frame]
        ax.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.6, s=50)
        ax.set_xlim(points[:, 0].min() - 0.1, points[:, 0].max() + 0.1)
        ax.set_ylim(points[:, 1].min() - 0.1, points[:, 1].max() + 0.1)
        ax.set_title(f'Frame {frame}')
        
    anim = FuncAnimation(fig, update, frames=len(sequence), 
                        interval=1000//fps)
    
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close()

def visualize_clusters(embeddings_2d: np.ndarray,
                      cluster_labels: np.ndarray,
                      filename: str = 'clusters.png') -> None:
    """
    Create a scatter plot of the clustered embeddings.
    
    Args:
        embeddings_2d: Reduced dimensionality embeddings
        cluster_labels: Cluster assignments
        filename: Output filename
    """
    plt.figure(figsize=(10, 10))
    
    # Plot points
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=cluster_labels, cmap='Spectral',
                         alpha=0.6, s=50)
    
    # Add colorbar
    plt.colorbar(scatter)
    plt.title('UMAP projection of Lenia patterns')
    plt.savefig(filename)
    plt.close()

def process(embeddings, file_paths, output_dir='output'):
    """
    Main function to process embeddings and create visualizations.
    
    Args:
        embeddings: Embedding vectors from the autoencoder
        val_data: Original validation set data
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process embeddings
    print("Processing embeddings...")
    embeddings_2d, cluster_labels, file_paths = process_embeddings(embeddings, file_paths)
    assert len(embeddings_2d) == len(cluster_labels)
    assert len(embeddings_2d) == len(file_paths)
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    np.save(os.path.join(this_dir_path, 'embeddings_2d.npy'), embeddings_2d)
    np.save(os.path.join(this_dir_path, 'cluster_labels.npy'), cluster_labels)
    np.save(os.path.join(this_dir_path, 'file_paths.npy'), file_paths)
    
    # Visualize clusters
    # print("Creating cluster visualization...")
    # visualize_clusters(embeddings_2d, cluster_labels,
    #                   os.path.join(output_dir, 'clusters.png'))
    
    # # Find and animate representatives
    # print("Creating animations for cluster representatives...")
    # representatives = find_cluster_representatives(embeddings_2d,
    #                                             cluster_labels,
    #                                             val_data)
    
    # for i, rep in enumerate(representatives):
    #     filename = os.path.join(output_dir, f'cluster_{i}_representative.gif')
    #     create_animation(rep, filename)
        
    # print(f"Processing complete. Found {len(representatives)} clusters.")
    # print(f"Results saved in {output_dir}")

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

    limit = None
    # Loading the training dataset. We need to split it into a training and validation part
    dataset = ParticleLeniaDataset(
        dataset_path, 
        config.params.num_points, 
        config.params.num_dims, 
        tail=(config.params.num_samples + 1) * config.params.period,
        limit=limit,
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

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    embeddings = np.load(os.path.join(this_dir_path, 'embeddings.npy'))
    if limit:
        embeddings = embeddings[:limit]
    output_dir = 'lenia_visualizations'
    process(embeddings, dataset.exs, output_dir)

if __name__ == '__main__':
    main()