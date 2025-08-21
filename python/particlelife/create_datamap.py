import concurrent.futures

import datamapplot
import pacmap
import pandas as pd
import polars as pl
import numpy as np
import toponymy
import torch
import transformers
from tqdm import tqdm

TOOLTIP_CSS = """
    maxWidth: '280px',
      backgroundColor: '#1a1a1a',
      color: '#ffffff',
      padding: '12px',
      borderRadius: '8px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
      border: '1px solid #333'"""

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor

def get_life_like_scores(df):

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
    max_life_like_scores = np.mean(life_like_scores, axis=-1)
    
    return max_life_like_scores

def _get_sim(mu_k_i, mu_k_j, sigma_k_i, sigma_k_j, w_k_i, w_k_j, mu_g_i, mu_g_j, sigma_g_i, sigma_g_j):
    kernel_params_i = np.stack([mu_k_i, sigma_k_i, w_k_i], axis=-1)
    kernel_params_j = np.stack([mu_k_j, sigma_k_j, w_k_j], axis=-1)
    growth_params_i = np.stack([mu_g_i, sigma_g_i], axis=-1)
    growth_params_j = np.stack([mu_g_j, sigma_g_j], axis=-1)
    num_species_i = kernel_params_i.shape[0]
    num_species_j = kernel_params_j.shape[0]
    num_kernels_i = kernel_params_i.shape[2]
    num_kernels_j = kernel_params_j.shape[2]
    num_growth_funcs_i = growth_params_i.shape[1]
    num_growth_funcs_j = growth_params_j.shape[1]
    kernel_sim = np.sqrt(np.sum(np.square(kernel_params_i[:, np.newaxis, :, np.newaxis, :, np.newaxis] - kernel_params_j[np.newaxis, :, np.newaxis, :, np.newaxis]), axis=-1))
    growth_sim = np.sqrt(np.sum(np.square(growth_params_i[:, np.newaxis, :, np.newaxis] - growth_params_j[np.newaxis, :, np.newaxis, :]), axis=-1))

    max_kernel_axis = ()
    max_growth_axis = ()
    if num_species_i < num_species_j:
        max_kernel_axis = (0, 2)
        max_growth_axis = (0,)
    else:
        max_kernel_axis = (1, 3)
        max_growth_axis = (1,)

    if num_kernels_i < num_kernels_j:
        max_kernel_axis += (4,)
    else:
        max_kernel_axis += (5,)

    if num_growth_funcs_i < num_growth_funcs_j:
        max_growth_axis += (2,)
    else:
        max_growth_axis += (3,)

    kernel_sim = np.mean(np.min(kernel_sim, axis=max_kernel_axis))
    growth_sim = np.mean(np.min(growth_sim, axis=max_growth_axis))

    return (kernel_sim + growth_sim) / 2

def get_sim(params_i, params_j):
    mu_k_i = np.array(params_i['mu_k'])
    mu_k_j = np.array(params_j['mu_k'])
    sigma_k_i = np.array(params_i['sigma_k'])
    sigma_k_j = np.array(params_j['sigma_k'])
    w_k_i = np.array(params_i['w_k'])
    w_k_j = np.array(params_j['w_k'])
    mu_g_i = np.array(params_i['mu_g'])
    mu_g_j = np.array(params_j['mu_g'])
    sigma_g_i = np.array(params_i['sigma_g'])
    sigma_g_j = np.array(params_j['sigma_g'])
    return _get_sim(mu_k_i, mu_k_j, sigma_k_i, sigma_k_j, w_k_i, w_k_j, mu_g_i, mu_g_j, sigma_g_i, sigma_g_j)

def main():
    # Use your existing parquet file
    parquet_file_path = "./data/particle_lenia_clip_embeddings1.parquet.zstd"
    
    print("Processing parquet file with UMAP...")
    sample_size = 2000
    df = pl.read_parquet(parquet_file_path, n_rows=sample_size * 10)

    df = df.sort(pl.col('median_force'), descending=True).head(sample_size * 4)
    life_like_scores = get_life_like_scores(df)
    df = df.with_columns(pl.Series(name='life_like_score', values=life_like_scores))
    # life_like_threshold = 0.5
    df = df.sort(pl.col('life_like_score')).tail(sample_size)

    df = df.with_columns([
        pl.col('params').struct.field('mu_k').list.len().alias('num_species'),
        pl.col('params').struct.field('mu_k').list.get(0).list.get(0).list.len().alias('num_kernels'),
        pl.col('params').struct.field('mu_g').list.get(0).list.len().alias('num_growth_funcs')
    ])

    param_df = df.select('params').with_row_index()
    arg_list = param_df.join(param_df, how='cross')\
        .filter(pl.col('index') < pl.col('index_right'))\
        .select([pl.col('index', 'index_right', 'params', 'params_right')])\
        .rows()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(lambda p: get_sim(p[2], p[3]), arg_list), total=len(arg_list), desc='Computing pairwise similarities'))

    A = np.zeros((sample_size, sample_size))
    for (i, j, _, _), sim in zip(arg_list, results):
        A[i, j] = sim

    A += np.triu(A, k=1).T  # Make the matrix symmetric
    nearest_neighbours = np.argsort(A, axis=1)[:, 1:11]
    edges = []
    for i in range(sample_size):
        for j in nearest_neighbours[i]:
            edges.append((i, j))
    edge_df = pd.DataFrame(edges, columns=['source', 'target']).drop_duplicates()

    life_like_scores = df['life_like_score'].to_numpy()

    oscillator_scores = ((df['median_freq'] > 0) * df['median_freq_power'].log1p()).to_numpy()

    num_kernels = df['num_kernels'].to_numpy()
    num_species = df['num_species'].to_numpy()
    num_growth_funcs = df['num_growth_funcs'].to_numpy()

    marker_size_array = (df['median_force'] / df['median_force'].min()).log1p().log1p().to_numpy()
    
    if df.shape[0] < 5:
        umap_embeddings = np.random.uniform(size=(df.shape[0], 2))
        layer_labels = [np.array(['-1'] * df.shape[0])]

    else:
        # Extract embeddings and compute UMAP
        embeddings = np.stack(df['img_features'].to_list())
        if embeddings.ndim == 3:  # If embeddings have extra dimension, take mean
            # embeddings = embeddings.mean(axis=1)
            # get the last frame if it's a sequence
            embeddings = embeddings[:, -1, :]  # Assuming shape is (n_samples, 1, n_features)
        
        print(f"Embeddings shape: {embeddings.shape}")
        print("Running UMAP...")
        
        # Perform UMAP dimensionality reduction
        umap_reducer = pacmap.PaCMAP(n_components=2, verbose=True)
        umap_embeddings = umap_reducer.fit_transform(embeddings)

        clusterer = toponymy.ToponymyClusterer(
            min_clusters=4,
            verbose=True
        )
        clusterer.fit(clusterable_vectors=umap_embeddings, embedding_vectors=embeddings, show_progress_bar=True)
        layer_labels = [l.cluster_labels.astype(str) for l in clusterer.cluster_layers_]
        # layer_labels = [np.array(['-1'] * df.shape[0])]

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
        marker_size_array=marker_size_array,
        noise_label='-1',
        darkmode=True,
        tooltip_css=TOOLTIP_CSS,
        title="Particle Lenia Explorer",
        sub_title="Points positioned in behaviour space, joined by edges in parameter space. Hover over points to see behaviour, click to open in large simulation.",
        extra_point_data=pd.DataFrame(df.select('params').to_dicts()),
        edge_bundle=True,
        edge_bundle_keywords={"color_map_nn": 100, "edges": edge_df},
        edge_width=0.1,
        dynamic_tooltip=dynamic_tooltip,
        on_click="window.open(`/particle_lenia.html?params=${JSON.stringify(hoverData.params[index])}`)",
        custom_js=custom_js,
        minify_deps=False,
        search_field="",
        colormap_rawdata=[life_like_scores, oscillator_scores, num_species, num_kernels, num_growth_funcs],
        colormap_metadata=[
            {"field": "lifelike", "description": "Life-like score", "cmap": "viridis", "kind": "continuous"},
            {"field": "oscillators", "description": "Oscillator score", "cmap": "plasma", "kind": "continuous"},
            {"field": "num_species", "description": "Number of species", "cmap": "jet", "kind": "categorical"},
            {"field": "num_kernels", "description": "Number of kernels", "cmap": "jet", "kind": "categorical"},
            {"field": "num_growth_funcs", "description": "Number of growth functions", "cmap": "jet", "kind": "categorical"},
        ]
    )
    plot.save('./datamap.html')

# Example usage:
if __name__ == "__main__":
    main()