
import pandas as pd
import polars as pl
import umap
import numpy as np

import datamapplot
import toponymy

TOOLTIP_CSS = """
    maxWidth: '280px',
      backgroundColor: '#1a1a1a',
      color: '#ffffff',
      padding: '12px',
      borderRadius: '8px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
      border: '1px solid #333'"""


def main():
    # Use your existing parquet file
    parquet_file_path = "./data/particle_lenia_clip_embeddings.parquet.zstd"
    
    print("Processing parquet file with UMAP...")
    sample_size = 10000  # Default sample size for testing
    df = pl.read_parquet(parquet_file_path, n_rows=sample_size)
    
    # For testing, take a smaller sample
    print(f"Original dataset size: {len(df)}")
    df = df.sample(n=sample_size, seed=42)
    print(f"Using sample size: {len(df)}")
    
    if sample_size < 5:
        umap_embeddings = np.random.uniform(size=(sample_size, 2))
        layer_labels = [np.array(['0'] * sample_size)]

    else:
        # Extract embeddings and compute UMAP
        embeddings = np.stack(df['img_features'].to_list())
        if embeddings.ndim == 3:  # If embeddings have extra dimension, take mean
            # embeddings = embeddings.mean(axis=1)
            embeddings = embeddings[:, 0, :]  # Assuming shape is (n_samples, 1, n_features)
        
        print(f"Embeddings shape: {embeddings.shape}")
        print("Running UMAP...")
        
        # Perform UMAP dimensionality reduction
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, verbose=True)
        umap_embeddings = umap_reducer.fit_transform(embeddings)

        clusterer = toponymy.ToponymyClusterer(
            min_clusters=4,
            verbose=True
        )
        clusterer.fit(clusterable_vectors=umap_embeddings, embedding_vectors=embeddings, show_progress_bar=True)
        layer_labels = [l.cluster_labels.astype(str) for l in clusterer.cluster_layers_]

    custom_js_path = './webgpu/simulation.js'
    with open(custom_js_path, 'r') as f:
        custom_js = f.read()

    format_content_path = './webgpu/format_content.js'
    with open(format_content_path, 'r') as f:
        format_content_js = f.read()

    dynamic_tooltip = {
        'identifier_js': '({ index }) => index',
        'fetch_js': 'async (index) => datamap?.metaData?.params?.[index]',
        'format_js': format_content_js,
        'loading_js': '(identifier) => `Loading ...`',
        'error_js': '(error, identifier) => `Error loading data: ${error.message}`'
    }

    plot = datamapplot.create_interactive_plot(
        umap_embeddings,
        *layer_labels,
        noise_label='-1',
        tooltip_css=TOOLTIP_CSS,
        extra_point_data=pd.DataFrame(df.select('params').to_dicts()),
        dynamic_tooltip=dynamic_tooltip,
        on_click='startSimulation("{params}")',
        custom_js=custom_js,
        minify_deps=False,
        search_field=""
    )
    plot.save('./datamap.html')

# Example usage:
if __name__ == "__main__":
    main()