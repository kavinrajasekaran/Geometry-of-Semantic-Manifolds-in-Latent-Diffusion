"""
Configuration module for Deep Convolutional Auto-Encoder (DCAE).

Centralizes all hyperparameters and settings so you can easily
swap datasets, adjust the bottleneck size, or tune training.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """All-in-one configuration for DCAE training and evaluation."""

    # ── Dataset ───────────────────────────────────────────────────────
    dataset: str = "cifar10"          # "cifar10" | "mnist"
    data_root: str = "./data"         # download / cache directory
    num_workers: int = 4

    # ── Model ─────────────────────────────────────────────────────────
    latent_dim: int = 128             # size of the flattened bottleneck
    encoder_channels: Tuple[int, ...] = (32, 64, 128)  # conv channels per layer
    # decoder_channels is automatically mirrored from encoder_channels

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 30
    seed: int = 42

    # ── Visualization ─────────────────────────────────────────────────
    num_vis_images: int = 16          # images shown in reconstruction grid
    vis_method: str = "umap"          # "umap" | "tsne"
    num_vis_samples: int = 5000       # samples used for latent-space plot
    output_dir: str = "./outputs"     # where figures are saved

    # ── Derived (set automatically) ───────────────────────────────────
    in_channels: int = field(init=False)      # 1 for MNIST, 3 for CIFAR-10
    image_size: int = field(init=False)        # 28 for MNIST, 32 for CIFAR-10

    def __post_init__(self):
        if self.dataset == "mnist":
            self.in_channels = 1
            self.image_size = 28
        elif self.dataset == "cifar10":
            self.in_channels = 3
            self.image_size = 32
        else:
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. Choose 'mnist' or 'cifar10'."
            )
