"""
Deep Convolutional Auto-Encoder (DCAE) — PyTorch implementation.

Architecture follows the principles from Turchenko & Luczak (2016):
  • Symmetric encoder–decoder structure
  • Encoder:  Conv → BN → ReLU → Conv → BN → ReLU → MaxPool  (× 3 blocks)
  • Bottleneck:  flatten → Linear → latent_dim  (explicit low-dim representation)
  • Decoder:  Linear → unflatten → ConvTranspose+BN+ReLU (× 3 blocks) → Sigmoid

The model is fully parameterised by `Config` so you can change the
dataset, number of encoder channels, or bottleneck dimensionality
without touching this file.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    """All-in-one configuration for DCAE training and evaluation."""

    # ── Dataset ───────────────────────────────────────────────────────
    dataset: str = "cifar10"          # "cifar10" | "mnist"
    data_root: str = "./data"         # download / cache directory
    num_workers: int = 0

    # ── Model ─────────────────────────────────────────────────────────
    latent_dim: int = 256             # size of the flattened bottleneck (increased for clearer reconstructions)
    encoder_channels: Tuple[int, ...] = (32, 64, 128)  # conv channels per layer

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 30
    seed: int = 42
    reconstruction_loss: str = "l1"   # Added explicitly to allow toggling L1 for sharper reconstructions

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


def _enc_block(in_c: int, out_c: int) -> nn.Sequential:
    """Encoder block: double conv (3×3) + BN + ReLU, then MaxPool to halve spatial dims."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def _dec_block(in_c: int, out_c: int, final: bool = False) -> nn.Sequential:
    """Decoder block: ConvTranspose (upsample ×2) + double conv + BN + ReLU.
    If `final`, the last activation is Sigmoid instead of ReLU (output in [0,1])."""
    layers = [
        nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
    ]
    if final:
        layers.append(nn.Sigmoid())
    else:
        layers += [nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════════════
#  Encoder
# ═══════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    """
    Three encoder blocks with double convolutions and MaxPool.

    Input  : (B, C_in, 32, 32)
    Output : (B, latent_dim)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.encoder_channels  # e.g. (32, 64, 128)
        in_c = cfg.in_channels     # 1 or 3

        self.blocks = nn.Sequential(
            _enc_block(in_c, ch[0]),   # 32×32 → 16×16
            _enc_block(ch[0], ch[1]),  # 16×16 → 8×8
            _enc_block(ch[1], ch[2]),  # 8×8   → 4×4
        )

        # Spatial size after 3× pool-by-2 on a 32×32 input = 4×4
        self._flat_dim = ch[2] * 4 * 4

        # Bottleneck projection  →  explicit latent vector
        self.fc = nn.Sequential(
            nn.Linear(self._flat_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = x.view(x.size(0), -1)      # flatten
        z = self.fc(x)                  # latent vector
        return z


# ═══════════════════════════════════════════════════════════════════════
#  Decoder
# ═══════════════════════════════════════════════════════════════════════
class Decoder(nn.Module):
    """
    Symmetric mirror of the encoder using transposed convolutions.

    Input  : (B, latent_dim)
    Output : (B, C_in, 32, 32)   values in [0, 1]
    """

    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.encoder_channels   # (32, 64, 128)
        in_c = cfg.in_channels

        self._spatial_ch = ch[2]    # channels at 4×4 spatial map

        # Map latent vector back to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim, ch[2] * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            _dec_block(ch[2], ch[1]),              # 4×4  → 8×8
            _dec_block(ch[1], ch[0]),              # 8×8  → 16×16
            _dec_block(ch[0], in_c, final=True),   # 16×16 → 32×32, Sigmoid
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self._spatial_ch, 4, 4)  # unflatten
        x = self.blocks(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Full Auto-Encoder
# ═══════════════════════════════════════════════════════════════════════
class DCAE(nn.Module):
    """
    Deep Convolutional Auto-Encoder.

    Exposes `.encode(x)` and `.decode(z)` for convenient access to
    latent representations and reconstructions independently.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent vector z for a batch of images."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from latent vectors."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """Full forward pass: encode → decode.  Returns (reconstruction, latent)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
