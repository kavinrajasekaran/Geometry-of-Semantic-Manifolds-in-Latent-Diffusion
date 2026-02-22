"""
Sparsity Test: Shrink the bottleneck until the model "breaks".
This experiment trains the DCAE on several bottleneck sizes and
plots the final reconstruction error (test loss) vs. latent dimension.
"""

import os
import torch
import matplotlib.pyplot as plt

from config import Config
from dataset import get_dataloaders
from model import DCAE
from train import train
from main import seed_everything

def run_sparsity_experiment():
    # Define our bottleneck sizes. A good range covers both sufficient capacity and severe constriction.
    latent_dims = [128, 64, 32, 16, 8, 4, 2]
    
    cfg = Config()
    cfg.num_epochs = 10  # Shorter training for multi-run experiment to save time
    
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_dataloaders(cfg)
    
    final_losses = []
    
    print(f"=== Running Sparsity Test on {cfg.dataset.upper()} ===")
    print(f"Testing dimensions: {latent_dims}")
    
    for dim in latent_dims:
        print(f"\n--- Training with latent_dim = {dim} ---")
        cfg.latent_dim = dim
        model = DCAE(cfg).to(device)
        
        # Train
        history = train(model, train_loader, test_loader, cfg, device)
        final_test_loss = history["test_loss"][-1]
        final_losses.append(final_test_loss)
        
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(latent_dims, final_losses, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # We invert x-axis so we visualize "shrinking" from left to right
    plt.gca().invert_xaxis()
    plt.xscale('log', base=2)
    plt.xticks(latent_dims, labels=[str(d) for d in latent_dims])
    
    
    plt.xlabel("Latent Dimension (Bottleneck Size)", fontsize=12)
    plt.ylabel("Final Test MSE Loss", fontsize=12)
    plt.title(f"Sparsity Test (Bottleneck Shrinkage) — {cfg.dataset.upper()}", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, "sparsity_test.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Sparsity test figure saved → {out_path}")
    
    try:
        plt.show()
    except Exception:
        pass # ignore if running headless

if __name__ == "__main__":
    run_sparsity_experiment()
