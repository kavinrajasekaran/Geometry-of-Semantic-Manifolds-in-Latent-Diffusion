#!/usr/bin/env python3
"""
main.py — Entry point for training and evaluating the DCAE.

Usage examples
--------------
  # Train on CIFAR-10 with default settings
  python main.py

  # Train on MNIST with a smaller bottleneck
  python main.py --dataset mnist --latent_dim 32 --num_epochs 20

  # Use t-SNE instead of UMAP
  python main.py --vis_method tsne

  # Skip training and only visualise from an existing checkpoint
  python main.py --eval_only --checkpoint ./outputs/dcae_checkpoint.pt
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Config, DCAE
from visualize import (
    plot_latent_space,
    plot_loss_curves,
    plot_reconstructions,
    plot_interpolations,
    plot_filters,
)

# ─── Data Loading ─────────────────────────────────────────────────────

def _build_transform(cfg: Config) -> transforms.Compose:
    """Return a transform pipeline that normalizes pixels to [0, 1]."""
    xforms = [transforms.ToTensor()]  # uint8 → float32 [0,1]
    if cfg.dataset == "mnist":
        # Pad 28×28 → 32×32 so spatial dims are powers of 2
        # (cleaner after 3 rounds of stride-2 pooling: 32→16→8→4)
        xforms.append(transforms.Pad(2))
    return transforms.Compose(xforms)


def get_dataloaders(cfg: Config):
    """Return (train_loader, test_loader) for the configured dataset."""
    transform = _build_transform(cfg)

    if cfg.dataset == "cifar10":
        train_ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=transform)
    elif cfg.dataset == "mnist":
        train_ds = datasets.MNIST(root=cfg.data_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=cfg.data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ─── Training Loop ────────────────────────────────────────────────────

def train_one_epoch(
    model: DCAE, loader, criterion: nn.Module, optimiser: Adam, device: torch.device,
) -> float:
    """Train for one epoch; return mean loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, _ in loader:
        images = images.to(device)

        x_hat, _ = model(images)
        loss = criterion(x_hat, images)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: DCAE, loader, criterion: nn.Module, device: torch.device,
) -> float:
    """Evaluate on test/val set; return mean loss."""
    model.eval()
    running_loss = 0.0
    n_batches = 0

    for images, _ in loader:
        images = images.to(device)
        x_hat, _ = model(images)
        loss = criterion(x_hat, images)
        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def train_model(
    model: DCAE, train_loader, test_loader, cfg: Config, device: torch.device,
) -> dict:
    """Full training procedure with cosine-annealing LR schedule."""
    if cfg.reconstruction_loss.lower() == "l1":
        criterion = nn.L1Loss()
        loss_name = "L1 Loss"
    else:
        criterion = nn.MSELoss()
        loss_name = "MSE Loss"

    optimiser = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg.num_epochs, eta_min=1e-6)

    history = {"train_loss": [], "test_loss": [], "loss_name": loss_name}

    print(f"\n{'═' * 60}")
    print(f"  Training DCAE  |  dataset={cfg.dataset}  |  latent_dim={cfg.latent_dim}")
    print(f"  device={device}  |  epochs={cfg.num_epochs}  |  batch_size={cfg.batch_size}")
    print(f"  lr={cfg.learning_rate}  |  scheduler=CosineAnnealing | loss={loss_name}")
    print(f"{'═' * 60}\n")

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch:3d}/{cfg.num_epochs}  │  "
            f"train_loss={train_loss:.6f}  │  "
            f"test_loss={test_loss:.6f}  │  "
            f"lr={lr_now:.2e}  │  "
            f"time={dt:.1f}s"
        )

    # Save checkpoint
    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir, "dcae_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "history": history,
        },
        ckpt_path,
    )
    print(f"\n  ✓ Checkpoint saved → {ckpt_path}")

    return history


# ─── Argument parser ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep Convolutional Auto-Encoder (DCAE)")

    # Dataset / model
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--encoder_channels", type=int, nargs=3, default=[32, 64, 128])
    p.add_argument("--reconstruction_loss", type=str, default="l1", choices=["mse", "l1"])

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # Visualisation
    p.add_argument("--vis_method", type=str, default="umap", choices=["umap", "tsne"])
    p.add_argument("--num_vis_images", type=int, default=16)
    p.add_argument("--num_vis_samples", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default="./outputs")

    # Eval-only mode
    p.add_argument("--eval_only", action="store_true", help="Skip training, load checkpoint")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")

    return p.parse_args()


# ─── Seed everything ─────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Build Config from CLI args
    cfg = Config(
        dataset=args.dataset,
        latent_dim=args.latent_dim,
        encoder_channels=tuple(args.encoder_channels),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        reconstruction_loss=args.reconstruction_loss,
        seed=args.seed,
        vis_method=args.vis_method,
        num_vis_images=args.num_vis_images,
        num_vis_samples=args.num_vis_samples,
        output_dir=args.output_dir,
    )

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, test_loader = get_dataloaders(cfg)

    # Model
    model = DCAE(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_params:,}")
    print(f"  Encoder channels: {cfg.encoder_channels}")
    print(f"  Latent dim:       {cfg.latent_dim}")
    print(f"  Input shape:      ({cfg.in_channels}, {cfg.image_size}, {cfg.image_size})")

    if args.eval_only:
        # ── Evaluation only ──────────────────────────────────────────
        ckpt_path = args.checkpoint or os.path.join(cfg.output_dir, "dcae_checkpoint.pt")
        print(f"\n  Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        history = ckpt.get("history", None)

        if history:
            plot_loss_curves(history, cfg)
    else:
        # ── Training ─────────────────────────────────────────────────
        history = train_model(model, train_loader, test_loader, cfg, device)
        plot_loss_curves(history, cfg)

    # ── Visualisation ────────────────────────────────────────────────
    plot_reconstructions(model, test_loader, cfg, device)
    plot_latent_space(model, test_loader, cfg, device)
    plot_interpolations(model, test_loader, cfg, device)
    plot_filters(model, cfg)

    print(f"\n  ✓ All outputs saved to {cfg.output_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
