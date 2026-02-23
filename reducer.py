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
    labels = data["labels"]
    steps = data["steps"]
    
    # ─── 1. PCA (Noise Reduction) ────────────────────────────────────────────
    # High-dimensional manifolds are noisy. Reducing down to ~50 dimensions captures 
    # the maximum variance and stabilizes t-SNE/UMAP.
    n_components = min(50, len(embeddings))
    print(f"Applying PCA (reducing to {n_components} components)...")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # ─── 2. t-SNE ────────────────────────────────────────────────────────────
    # t-SNE preserves local topology (perfect for looking at how clusters form locally).
    perplexity = min(30, len(embeddings) - 1)
    print(f"Applying t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
    tsne_proj = tsne.fit_transform(embeddings_pca)
    
    # ─── 3. UMAP ─────────────────────────────────────────────────────────────
    # UMAP preserves global topology well (great for seeing the distances between classes).
    umap_proj = None
    if _HAS_UMAP:
        print("Applying UMAP...")
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)
        umap_proj = reducer.fit_transform(embeddings_pca)
    else:
        print("⚠ UMAP not installed. Only falling back to t-SNE.")
        
    # Save the lowered-dimensional projections
    out_file = os.path.join(config.OUTPUT_DIR, "reduced_latents.npz")
    np.savez(
        out_file, 
        tsne=tsne_proj, 
        umap=umap_proj if _HAS_UMAP else tsne_proj,  # fallback if umap is missing 
        labels=labels, 
        steps=steps
    )
    print(f"✓ Reduction complete. Data saved to {out_file}")

if __name__ == "__main__":
    apply_dimensionality_reduction()
