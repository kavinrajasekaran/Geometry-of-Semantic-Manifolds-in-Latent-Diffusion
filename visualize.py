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

    plt.close(fig)


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

    plt.close(fig)


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

    plt.close(fig)


# ─── 4. Latent interpolation ─────────────────────────────────────────
@torch.no_grad()
def plot_interpolations(
    model: DCAE,
    test_loader,
    cfg: Config,
    device: torch.device,
    save: bool = True,
    steps: int = 10,
) -> None:
    """
    Interpolate between pairs of latent vectors to visualize the
    smoothness and learned structure of the bottleneck manifold.
    """
    model.eval()
    images, _ = next(iter(test_loader))
    
    num_rows = 5
    # Ensure we don't request more than batch size
    num_rows = min(num_rows, images.size(0) // 2)
    images = images[:num_rows * 2].to(device)
    
    fig, axes = plt.subplots(num_rows, steps, figsize=(steps * 1.5, num_rows * 1.5))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(num_rows):
        img_a = images[2 * i : 2 * i + 1]
        img_b = images[2 * i + 1 : 2 * i + 2]
        
        z_a = model.encode(img_a)
        z_b = model.encode(img_b)
        
        alphas = torch.linspace(0, 1, steps=steps).to(device)
        z_interp = z_a * (1 - alphas.view(-1, 1)) + z_b * alphas.view(-1, 1)
        
        recons = model.decode(z_interp)
        
        for j in range(steps):
            ax = axes[i, j]
            rec = _to_numpy_img(recons[j])
            ax.imshow(rec, cmap="gray" if cfg.in_channels == 1 else None)
            ax.axis("off")
            if i == 0:
                if j == 0:
                    ax.set_title("Source", fontsize=10, fontweight="bold")
                elif j == steps - 1:
                    ax.set_title("Target", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Latent Interpolation ({steps} steps) — {cfg.dataset.upper()} | latent_dim={cfg.latent_dim}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save:
        os.makedirs(cfg.output_dir, exist_ok=True)
        path = os.path.join(cfg.output_dir, "interpolations.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Latent interpolation figure saved → {path}")

    plt.close(fig)


# ─── 5. Feature inspection ───────────────────────────────────────────
def plot_filters(model: DCAE, cfg: Config, save: bool = True) -> None:
    """
    Extract and visualize the weights of the first convolutional layer
    to see the edge/texture detectors the model learned.
    """
    first_conv = model.encoder.blocks[0][0]
    weight = first_conv.weight.data.cpu()  # (out_c, in_c, H, W)
    out_c, in_c, H, W = weight.shape
    
    n_filters = min(out_c, 32)
    import math
    cols = 8
    rows = math.ceil(n_filters / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = axes.flatten()
    
    for i in range(n_filters):
        ax = axes[i]
        filt = weight[i].numpy() # (in_c, H, W)
        
        f_min, f_max = filt.min(), filt.max()
        filt = (filt - f_min) / (f_max - f_min + 1e-8)
        
        if in_c == 1:
            ax.imshow(filt[0], cmap="gray")
        else:
            filt = np.transpose(filt, (1, 2, 0)) # (H, W, c)
            ax.imshow(filt)
        ax.axis("off")
        
    for j in range(n_filters, len(axes)):
        axes[j].axis("off")
        
    fig.suptitle(
        f"First-Layer Convolutional Filters — {cfg.dataset.upper()}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save:
        os.makedirs(cfg.output_dir, exist_ok=True)
        path = os.path.join(cfg.output_dir, "filters.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Feature inspection (filters) figure saved → {path}")

    plt.close(fig)
