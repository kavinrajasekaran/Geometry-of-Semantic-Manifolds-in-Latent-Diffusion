"""
Deep Convolutional Auto-Encoder (DCAE) — PyTorch implementation.

Architecture follows the principles from Turchenko & Luczak (2016):
  • Symmetric encoder–decoder structure
  • Encoder:  Conv → ReLU → MaxPool  (× 3 layers)
  • Bottleneck:  flatten → Linear → latent_dim  (explicit low-dim representation)
  • Decoder:  Linear → unflatten → ConvTranspose+ReLU (× 3 layers) → Sigmoid

The model is fully parameterised by `Config` so you can change the
dataset, number of encoder channels, or bottleneck dimensionality
without touching this file.
"""

import torch
import torch.nn as nn

from config import Config


# ═══════════════════════════════════════════════════════════════════════
#  Encoder
# ═══════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    """
    Three convolutional blocks:  Conv2d → BatchNorm → ReLU → MaxPool2d.

    Input  : (B, C_in, 32, 32)
    Output : (B, latent_dim)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.encoder_channels  # e.g. (32, 64, 128)
        in_c = cfg.in_channels     # 1 or 3

        self.conv_blocks = nn.Sequential(
            # Block 1: (B, in_c, 32, 32) → (B, ch[0], 16, 16)
            nn.Conv2d(in_c, ch[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (B, ch[0], 16, 16) → (B, ch[1], 8, 8)
            nn.Conv2d(ch[0], ch[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (B, ch[1], 8, 8) → (B, ch[2], 4, 4)
            nn.Conv2d(ch[1], ch[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Spatial size after 3× pool-by-2 on a 32×32 input = 4×4
        self._flat_dim = ch[2] * 4 * 4

        # Bottleneck projection  →  explicit latent vector
        self.fc = nn.Linear(self._flat_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
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
        ch = cfg.encoder_channels  # (32, 64, 128)
        in_c = cfg.in_channels

        self._spatial_ch = ch[2]  # channels at 4×4 spatial map

        # Map latent vector back to spatial feature map
        self.fc = nn.Linear(cfg.latent_dim, ch[2] * 4 * 4)

        self.deconv_blocks = nn.Sequential(
            # Block 1: (B, ch[2], 4, 4) → (B, ch[1], 8, 8)
            nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),

            # Block 2: (B, ch[1], 8, 8) → (B, ch[0], 16, 16)
            nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),

            # Block 3: (B, ch[0], 16, 16) → (B, in_c, 32, 32)
            nn.ConvTranspose2d(ch[0], in_c, kernel_size=2, stride=2),
            nn.Sigmoid(),  # output in [0, 1] to match normalised input
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self._spatial_ch, 4, 4)  # unflatten
        x = self.deconv_blocks(x)
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
