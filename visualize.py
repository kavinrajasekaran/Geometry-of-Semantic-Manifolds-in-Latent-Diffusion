"""
Visualization module for DCAE.

Provides two primary functions:
  1. plot_reconstructions  — side-by-side Original vs Reconstructed images
  2. plot_latent_space     — 2-D scatter of the latent space using UMAP or t-SNE
  3. plot_loss_curves      — training & test loss over epochs
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from config import Config
from model import DCAE

# Try importing UMAP; fall back to t-SNE if not installed
try:
    from umap import UMAP

    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


# ─── Helpers ──────────────────────────────────────────────────────────
def _to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [0,1] to (H, W, C) numpy for plotting."""
    img = tensor.detach().cpu().numpy()
    if img.shape[0] == 1:
        return img.squeeze(0)  # grayscale → (H, W)
    return np.transpose(img, (1, 2, 0))  # (C,H,W) → (H,W,C)


# ─── 1. Reconstruction comparison ────────────────────────────────────
@torch.no_grad()
def plot_reconstructions(
    model: DCAE,
    test_loader,
    cfg: Config,
    device: torch.device,
    save: bool = True,
) -> None:
    """
    Display `cfg.num_vis_images` original/reconstructed pairs in a grid.
    """
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[: cfg.num_vis_images].to(device)
    x_hat, _ = model(images)

    n = cfg.num_vis_images
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))

    for i in range(n):
        # Original
        ax = axes[0, i]
        img = _to_numpy_img(images[i])
        ax.imshow(img, cmap="gray" if cfg.in_channels == 1 else None)
        ax.axis("off")
        if i == 0:
            ax.set_title("Original", fontsize=10, fontweight="bold")

        # Reconstruction
        ax = axes[1, i]
        rec = _to_numpy_img(x_hat[i])
        ax.imshow(rec, cmap="gray" if cfg.in_channels == 1 else None)
        ax.axis("off")
        if i == 0:
            ax.set_title("Reconstructed", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"DCAE Reconstructions  —  {cfg.dataset.upper()}  |  latent_dim={cfg.latent_dim}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save:
        os.makedirs(cfg.output_dir, exist_ok=True)
        path = os.path.join(cfg.output_dir, "reconstructions.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Reconstruction figure saved → {path}")

    plt.show()


# ─── 2. Latent-space visualisation ───────────────────────────────────
@torch.no_grad()
def plot_latent_space(
    model: DCAE,
    test_loader,
    cfg: Config,
    device: torch.device,
    save: bool = True,
) -> None:
    """
    Collect latent vectors and labels, project to 2-D with UMAP or
    t-SNE, then create a colour-coded scatter plot.
    """
    model.eval()
    latents, labels = [], []
    collected = 0

    for images, targets in test_loader:
        images = images.to(device)
        z = model.encode(images)
        latents.append(z.cpu().numpy())
        labels.append(targets.numpy())
        collected += images.size(0)
        if collected >= cfg.num_vis_samples:
            break

    latents = np.concatenate(latents, axis=0)[: cfg.num_vis_samples]
    labels = np.concatenate(labels, axis=0)[: cfg.num_vis_samples]

    # Dimensionality reduction to 2-D
    method = cfg.vis_method.lower()
    if method == "umap" and _HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=cfg.seed, n_neighbors=15)
        title_method = "UMAP"
    else:
        if method == "umap" and not _HAS_UMAP:
            print("  ⚠  umap-learn not installed; falling back to t-SNE.")
        reducer = TSNE(n_components=2, random_state=cfg.seed, perplexity=30)
        title_method = "t-SNE"

    print(f"  → Projecting {len(latents)} latent vectors with {title_method} …")
    projection = reducer.fit_transform(latents)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    n_classes = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", n_classes)

    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels,
        cmap=cmap,
        s=8,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_classes), pad=0.02)
    cbar.set_label("Class", fontsize=11)

    ax.set_title(
        f"Latent Space ({title_method})  —  {cfg.dataset.upper()}  |  "
        f"latent_dim={cfg.latent_dim}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(f"{title_method}-1", fontsize=11)
    ax.set_ylabel(f"{title_method}-2", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(cfg.output_dir, exist_ok=True)
        path = os.path.join(cfg.output_dir, "latent_space.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Latent-space figure saved → {path}")

    plt.show()


# ─── 3. Loss curves ──────────────────────────────────────────────────
def plot_loss_curves(history: dict, cfg: Config, save: bool = True) -> None:
    """Plot training and test MSE loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["test_loss"], label="Test Loss", linewidth=2, linestyle="--")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(
        f"Training Curves  —  {cfg.dataset.upper()}  |  latent_dim={cfg.latent_dim}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(cfg.output_dir, exist_ok=True)
        path = os.path.join(cfg.output_dir, "loss_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Loss-curve figure saved → {path}")

    plt.show()
