#!/usr/bin/env python3
"""
feature_extractor.py — Extract and visualize features from pre-trained unsupervised models.

This script replaces the custom DCAE by:
1. Loading a pre-trained self-supervised model (DINO ViT-S/16, fallback to ResNet50).
2. Extracting high-dimensional embeddings from the CIFAR-10 test set.
3. Reducing embeddings to 2D using t-SNE and UMAP.
4. Visualizing the latent space and color-coding by ground-truth labels.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

# CIFAR-10 class names for the legend
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# ─── 1. Dataset Handling ─────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128, data_root: str = "./data") -> DataLoader:
    """
    Creates a DataLoader for the CIFAR-10 test set using ImageNet normalization.
    """
    # Standard ImageNet transforms required by pre-trained models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return test_loader


# ─── 2. Model Loading ────────────────────────────────────────────────

def load_model(device: torch.device) -> nn.Module:
    """
    Attempts to load DINO ViT-S/16. Falls back to ResNet50 if unavailable.
    """
    try:
        print("Attempting to load pre-trained DINO (dino_vits16)...")
        # Load DINO from torch.hub
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
        print("✓ Successfully loaded DINO ViT-S/16.")
    except Exception as e:
        print(f"⚠ Failed to load DINO: {e}")
        print("Falling back to pre-trained ResNet50 feature extractor...")
        # Load standard ResNet50
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # We want the features right before the final fully connected layer (avgpool)
        # ResNet50's final pooling layer is named 'avgpool'
        model = create_feature_extractor(base_model, return_nodes={'avgpool': 'features'})
        
        # Wrap it so it just returns the tensor instead of a dict
        class ResNetFeatureWrapper(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor
            def forward(self, x):
                features = self.extractor(x)['features']
                # shape is (B, 2048, 1, 1), we need to flatten to (B, 2048)
                return torch.flatten(features, 1)
                
        model = ResNetFeatureWrapper(model)
        print("✓ Successfully loaded ResNet50 fallback.")

    model = model.to(device)
    model.eval()
    return model


# ─── 3. Feature Extraction Loop ──────────────────────────────────────

@torch.no_grad()
def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Iterates over the dataloader and extracts embeddings for all images.
    Returns (embeddings_array, labels_array).
    """
    print(f"\nExtracting features from {len(dataloader.dataset)} images...")
    embeddings_list = []
    labels_list = []

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        
        # Forward pass to get embeddings
        emb = model(images)
        
        embeddings_list.append(emb.cpu().numpy())
        labels_list.append(targets.numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    # Concatenate all batches
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    print(f"✓ Feature extraction complete. Shape: {all_embeddings.shape}")
    return all_embeddings, all_labels


# ─── 4. Dimensionality Reduction & Visualization ─────────────────────

def visualize_embeddings(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    method: str = 'tsne', 
    output_dir: str = './outputs'
) -> None:
    """
    Reduces embeddings to 2D and generates a color-coded scatter plot.
    """
    print(f"\nApplying dimensionality reduction using {method.upper()}...")
    
    if method == 'umap' and _HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    else:
        if method == 'umap' and not _HAS_UMAP:
            print("⚠ UMAP not installed. Falling back to t-SNE.")
            method = 'tsne'
        # PCA init is significantly faster and more stable for t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')

    projections = reducer.fit_transform(embeddings)

    print(f"Generating scatter plot for {method.upper()}...")
    plt.figure(figsize=(10, 8))
    
    # Use seaborn for a prettier scatter plot with a distinct color palette
    palette = sns.color_palette("tab10", 10)
    sns.scatterplot(
        x=projections[:, 0], 
        y=projections[:, 1], 
        hue=[CIFAR10_CLASSES[l] for l in labels],
        hue_order=CIFAR10_CLASSES,
        palette=palette,
        s=15,    # size of points
        alpha=0.8,
        linewidth=0
    )
    
    plt.title(f"Pre-trained Model Latent Space ({method.upper()}) — CIFAR-10", fontsize=14, fontweight='bold')
    plt.xlabel(f"{method.upper()} Axis 1", fontsize=12)
    plt.ylabel(f"{method.upper()} Axis 2", fontsize=12)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"pretrained_{method}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {method.upper()} visualization to: {out_path}")


# ─── Main Pipeline ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract and visualize features from pre-trained models.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save plots.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    test_loader = get_dataloaders(batch_size=args.batch_size)

    # 2. Load Model
    model = load_model(device)

    # 3. Extract Features
    embeddings, labels = extract_features(model, test_loader, device)

    # 4. Visualize
    visualize_embeddings(embeddings, labels, method='tsne', output_dir=args.output_dir)
    if _HAS_UMAP:
        visualize_embeddings(embeddings, labels, method='umap', output_dir=args.output_dir)

    print("\n✓ Pipeline completed successfully.")

if __name__ == "__main__":
    main()
