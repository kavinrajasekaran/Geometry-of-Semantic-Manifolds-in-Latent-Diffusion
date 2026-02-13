"""
Dataset loader for DCAE.

Supports CIFAR-10 and MNIST out of the box.  Adding a new dataset
only requires extending the `get_dataloaders` function.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config


def _build_transform(cfg: Config) -> transforms.Compose:
    """Return a transform pipeline that normalizes pixels to [0, 1]."""
    xforms = [transforms.ToTensor()]  # uint8 → float32 [0,1]
    if cfg.dataset == "mnist":
        # Pad 28×28 → 32×32 so spatial dims are powers of 2
        # (cleaner after 3 rounds of stride-2 pooling: 32→16→8→4)
        xforms.append(transforms.Pad(2))
    return transforms.Compose(xforms)


def get_dataloaders(cfg: Config):
    """
    Return (train_loader, test_loader) for the configured dataset.

    Parameters
    ----------
    cfg : Config
        Project configuration object.

    Returns
    -------
    train_loader, test_loader : DataLoader
    """
    transform = _build_transform(cfg)

    if cfg.dataset == "cifar10":
        train_ds = datasets.CIFAR10(
            root=cfg.data_root, train=True, download=True, transform=transform
        )
        test_ds = datasets.CIFAR10(
            root=cfg.data_root, train=False, download=True, transform=transform
        )
    elif cfg.dataset == "mnist":
        train_ds = datasets.MNIST(
            root=cfg.data_root, train=True, download=True, transform=transform
        )
        test_ds = datasets.MNIST(
            root=cfg.data_root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
