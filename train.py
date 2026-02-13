"""
Training loop for the Deep Convolutional Auto-Encoder.

Uses MSE reconstruction loss and the Adam optimiser, as requested.
Supports GPU acceleration when available.
"""

import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from config import Config
from model import DCAE


def train_one_epoch(
    model: DCAE,
    loader,
    criterion: nn.Module,
    optimiser: Adam,
    device: torch.device,
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

    return running_loss / n_batches


@torch.no_grad()
def evaluate(
    model: DCAE,
    loader,
    criterion: nn.Module,
    device: torch.device,
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

    return running_loss / n_batches


def train(
    model: DCAE,
    train_loader,
    test_loader,
    cfg: Config,
    device: torch.device,
) -> dict:
    """
    Full training procedure.

    Returns
    -------
    history : dict
        {"train_loss": [...], "test_loss": [...]}
    """
    criterion = nn.MSELoss()
    optimiser = Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = {"train_loss": [], "test_loss": []}

    print(f"\n{'═' * 60}")
    print(f"  Training DCAE  |  dataset={cfg.dataset}  |  latent_dim={cfg.latent_dim}")
    print(f"  device={device}  |  epochs={cfg.num_epochs}  |  batch_size={cfg.batch_size}")
    print(f"{'═' * 60}\n")

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        print(
            f"  Epoch {epoch:3d}/{cfg.num_epochs}  │  "
            f"train_loss={train_loss:.6f}  │  "
            f"test_loss={test_loss:.6f}  │  "
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
