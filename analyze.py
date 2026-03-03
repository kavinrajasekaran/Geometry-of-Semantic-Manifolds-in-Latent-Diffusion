import os
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import config

def compute_nn_purity(embeddings, labels):
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    correct = 0
    for i in range(len(labels)):
        neighbor_idx = indices[i, 1]
        if labels[i] == labels[neighbor_idx]:
            correct += 1
    return correct / len(labels)

def run_analysis():
    in_file = os.path.join(config.OUTPUT_DIR, "extracted_latents.npz")
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"{in_file} not found. Run extractor.py first.")
    
    data = np.load(in_file, allow_pickle=True)
    embeddings = data["embeddings"]
    bottlenecks = data["bottlenecks"] if "bottlenecks" in data else None
    images = data["images"] if "images" in data else None
    labels = data["labels"]
    steps = data["steps"]
    
    unique_steps = sorted(set(steps))
    classes = sorted(set(labels))
    
    print("Clustering Quality Analysis")
    print(f"Classes: {classes}")
    print(f"Total samples: {len(labels)}")
    print(f"Tracked timesteps: {unique_steps}")
    print()
    
    representations = {"Spatial Latent": embeddings}
    if bottlenecks is not None:
        representations["U-Net Bottleneck"] = bottlenecks
    if images is not None:
        representations["Image Pixels"] = images
    
    for rep_name, rep_data in representations.items():
        print(f"\n{'─' * 60}")
        print(f"  {rep_name} (raw dim = {rep_data.shape[1]:,})")
        print(f"{'─' * 60}")
        print(f"  {'Step':>6}  {'Silhouette':>12}  {'NN Purity':>12}  {'Quality':>15}")
        print(f"  {'────':>6}  {'──────────':>12}  {'─────────':>12}  {'───────':>15}")
        
        for s in unique_steps:
            mask = steps == s
            subset = rep_data[mask]
            subset_labels = labels[mask]
            
            # Need at least 2 classes and more samples than classes
            unique_in_step = set(subset_labels)
            if len(unique_in_step) < 2 or len(subset) < 4:
                print(f"  {s:>6}  {'(too few samples)':>30}")
                continue
            
            # PCA first to stabilize metrics
            n_pca = min(50, len(subset), subset.shape[1])
            pca = PCA(n_components=n_pca, random_state=42)
            subset_pca = pca.fit_transform(subset)
            
            sil = silhouette_score(subset_pca, subset_labels)
            purity = compute_nn_purity(subset_pca, subset_labels)
            
            if sil > 0.5:
                quality = "STRONG ✓✓"
            elif sil > 0.25:
                quality = "MODERATE ✓"
            elif sil > 0.0:
                quality = "WEAK"
            else:
                quality = "NONE ✗"
            
            print(f"  {s:>6}  {sil:>12.4f}  {purity:>12.1%}  {quality:>15}")
        
        # Also compute on ALL timesteps combined
        n_pca_all = min(50, len(rep_data), rep_data.shape[1])
        pca_all = PCA(n_components=n_pca_all, random_state=42)
        all_pca = pca_all.fit_transform(rep_data)
        sil_all = silhouette_score(all_pca, labels)
        purity_all = compute_nn_purity(all_pca, labels)
        
        print(f"  {'ALL':>6}  {sil_all:>12.4f}  {purity_all:>12.1%}")
    


if __name__ == "__main__":
    run_analysis()
