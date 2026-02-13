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

import numpy as np
import torch

from config import Config
from dataset import get_dataloaders
from model import DCAE
from train import train
from visualize import plot_latent_space, plot_loss_curves, plot_reconstructions


# ─── Argument parser ─────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep Convolutional Auto-Encoder (DCAE)")

    # Dataset / model
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--encoder_channels", type=int, nargs=3, default=[32, 64, 128])

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=30)
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
        history = train(model, train_loader, test_loader, cfg, device)
        plot_loss_curves(history, cfg)

    # ── Visualisation ────────────────────────────────────────────────
    plot_reconstructions(model, test_loader, cfg, device)
    plot_latent_space(model, test_loader, cfg, device)

    print(f"\n  ✓ All outputs saved to {cfg.output_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
