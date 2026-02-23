#!/usr/bin/env python3
"""
main.py — Orchestrator script for The Geometry of Latent Diffusion Models.
Run this script to execute the full pipeline.
"""

from extractor import run_extraction
from reducer import apply_dimensionality_reduction
from visualizer import plot_visualizations

def main():
    print("=" * 60)
    print(" THE GEOMETRY OF LATENT DIFFUSION MODELS ".center(60))
    print("=" * 60)
    
    print("\n[PHASE 1] Extracting Diffusion Latents...")
    run_extraction()
    
    print("\n[PHASE 2] Applying Dimensionality Reduction...")
    apply_dimensionality_reduction()
    
    print("\n[PHASE 3] Generating Visualizations...")
    plot_visualizations()

if __name__ == "__main__":
    main()
