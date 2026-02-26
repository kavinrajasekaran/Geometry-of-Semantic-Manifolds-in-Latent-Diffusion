"""
reducer.py — Applies PCA, t-SNE, and UMAP to reduce the large tensor dimensionality.
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

import config

def apply_dimensionality_reduction():
    in_file = os.path.join(config.OUTPUT_DIR, "extracted_latents.npz")
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"{in_file} not found. Please run extractor.py first.")
        
    print(f"Loading data from {in_file}...")
    data = np.load(in_file, allow_pickle=True)
    embeddings = data["embeddings"]
    bottlenecks = data["bottlenecks"] if "bottlenecks" in data else None
    images = data["images"] if "images" in data else None
    labels = data["labels"]
    steps = data["steps"]
    
    # ─── 1. PCA (Noise Reduction) ────────────────────────────────────────────
    # High-dimensional manifolds are noisy. Reducing down to ~50 dimensions captures 
    # the maximum variance and stabilizes t-SNE/UMAP.
    n_components = min(50, len(embeddings))
    print(f"Applying PCA on Spatial Latents (reducing to {n_components} components)...")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    bottlenecks_pca = None
    if bottlenecks is not None:
        n_bn = min(50, len(bottlenecks))
        print(f"Applying PCA on U-Net Bottleneck (reducing to {n_bn} components)...")
        pca_bn = PCA(n_components=n_bn, random_state=42)
        bottlenecks_pca = pca_bn.fit_transform(bottlenecks)
    
    images_pca = None
    if images is not None:
        print(f"Applying PCA on Images (reducing to {n_components} components)...")
        pca_img = PCA(n_components=n_components, random_state=42)
        images_pca = pca_img.fit_transform(images)
    
    # ─── 2. t-SNE ────────────────────────────────────────────────────────────
    perplexity = min(30, len(embeddings) - 1)
    print(f"Applying t-SNE on Spatial Latents (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
    tsne_proj = tsne.fit_transform(embeddings_pca)
    
    tsne_proj_bn = None
    if bottlenecks_pca is not None:
        print(f"Applying t-SNE on U-Net Bottleneck (perplexity={perplexity})...")
        tsne_bn = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
        tsne_proj_bn = tsne_bn.fit_transform(bottlenecks_pca)
    
    tsne_proj_img = None
    if images_pca is not None:
        print(f"Applying t-SNE on Images (perplexity={perplexity})...")
        tsne_img = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
        tsne_proj_img = tsne_img.fit_transform(images_pca)
    
    # ─── 3. UMAP ─────────────────────────────────────────────────────────────
    # UMAP preserves global topology well (great for seeing the distances between classes).
    umap_proj = None
    umap_proj_img = None
    if _HAS_UMAP:
        print("Applying UMAP on Latents...")
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)
        umap_proj = reducer.fit_transform(embeddings_pca)
        
        if images_pca is not None:
            print("Applying UMAP on Images...")
            reducer_img = UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)
            umap_proj_img = reducer_img.fit_transform(images_pca)
    else:
        print("⚠ UMAP not installed. Only falling back to t-SNE.")
        if images_pca is not None:
            umap_proj_img = tsne_proj_img
        
    # ─── 4. Per-Timestep Reduction ────────────────────────────────────────────
    # Running t-SNE on ALL timesteps mixed together dilutes clustering signal.
    # Per-timestep plots show how clusters EMERGE over time.
    unique_steps = sorted(set(steps))
    per_step_tsne = {}
    per_step_tsne_img = {}
    per_step_tsne_bn = {}
    
    for s in unique_steps:
        mask = steps == s
        subset = embeddings[mask]
        n_sub = min(50, len(subset))
        
        if len(subset) < 4:
            continue
            
        perp = min(5, len(subset) - 1)
        
        print(f"  Per-timestep t-SNE on Latents at step {s} ({len(subset)} samples, perplexity={perp})...")
        pca_sub = PCA(n_components=min(n_sub, len(subset)), random_state=42)
        sub_pca = pca_sub.fit_transform(subset)
        tsne_sub = TSNE(n_components=2, random_state=42, perplexity=perp, init='random', learning_rate='auto')
        per_step_tsne[s] = tsne_sub.fit_transform(sub_pca)
        
        if images is not None:
            subset_img = images[mask]
            print(f"  Per-timestep t-SNE on Images at step {s}...")
            pca_sub_img = PCA(n_components=min(n_sub, len(subset_img)), random_state=42)
            sub_pca_img = pca_sub_img.fit_transform(subset_img)
            tsne_sub_img = TSNE(n_components=2, random_state=42, perplexity=perp, init='random', learning_rate='auto')
            per_step_tsne_img[s] = tsne_sub_img.fit_transform(sub_pca_img)
        
        if bottlenecks is not None:
            subset_bn = bottlenecks[mask]
            print(f"  Per-timestep t-SNE on Bottleneck at step {s}...")
            pca_sub_bn = PCA(n_components=min(n_sub, len(subset_bn)), random_state=42)
            sub_pca_bn = pca_sub_bn.fit_transform(subset_bn)
            tsne_sub_bn = TSNE(n_components=2, random_state=42, perplexity=perp, init='random', learning_rate='auto')
            per_step_tsne_bn[s] = tsne_sub_bn.fit_transform(sub_pca_bn)
    
    # Save the lowered-dimensional projections
    save_dict = {
        "tsne": tsne_proj,
        "umap": umap_proj if _HAS_UMAP else tsne_proj,
        "labels": labels,
        "steps": steps
    }
    
    if tsne_proj_bn is not None:
        save_dict["bottleneck_tsne"] = tsne_proj_bn
    
    if tsne_proj_img is not None:
        save_dict["images_tsne"] = tsne_proj_img
        save_dict["images_umap"] = umap_proj_img
    
    # Save per-timestep projections
    for s in per_step_tsne:
        save_dict[f"tsne_step_{s}"] = per_step_tsne[s]
    for s in per_step_tsne_img:
        save_dict[f"images_tsne_step_{s}"] = per_step_tsne_img[s]
    for s in per_step_tsne_bn:
        save_dict[f"bottleneck_tsne_step_{s}"] = per_step_tsne_bn[s]
        
    out_file = os.path.join(config.OUTPUT_DIR, "reduced_latents.npz")
    np.savez(out_file, **save_dict)
    print(f"✓ Reduction complete. Data saved to {out_file}")

if __name__ == "__main__":
    apply_dimensionality_reduction()
