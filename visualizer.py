"""
visualizer.py — Creates publication-ready semantic scatter and trajectory plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config

def plot_visualizations():
    in_file = os.path.join(config.OUTPUT_DIR, "reduced_latents.npz")
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"{in_file} not found. Please run reducer.py first.")
        
    data = np.load(in_file, allow_pickle=True)
    tsne_proj = data["tsne"]
    umap_proj = data["umap"]
    labels = data["labels"]
    steps = data["steps"]
    
    classes = list(config.TEXT_PROMPTS.keys())
    palette = sns.color_palette("tab10", len(classes))
    
    # Dictionary of projection names to arrays
    projections = {"t-SNE": tsne_proj}
    if umap_proj is not None and len(umap_proj.shape) == 2:
        projections["UMAP"] = umap_proj

    # ─── 1. Overall Scatter (Clustering over all timesteps) ────────────────
    print("Generating Overall Scatter plots...")
    for name, proj in projections.items():
        plt.figure(figsize=(10, 8))
        
        sns.scatterplot(
            x=proj[:, 0], 
            y=proj[:, 1], 
            hue=labels,
            hue_order=classes,
            style=steps,          # Differentiate step shapes
            palette=palette,
            s=80, alpha=0.8, edgecolor='w'
        )
        
        plt.title(f"{name} Projection of Latent Manifold (All Timesteps)", fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        out_path = os.path.join(config.OUTPUT_DIR, f"{name.lower()}_overall.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved {name} overall scatter to {out_path}")

    # ─── 2. Line Trajectories (Tracking generations over time) ──────────────
    print("Generating Latent Trajectory plots...")
    
    # Because we loop Class > Prompt > Step, the steps for a single image generation
    # are continuously chunked together in our arrays.
    pts_per_prompt = len(config.TRACKING_INTERVALS)
    
    for name, proj in projections.items():
        plt.figure(figsize=(10, 8))
        
        # Draw background scatter very faintly
        sns.scatterplot(
            x=proj[:, 0], 
            y=proj[:, 1], 
            hue=labels,
            hue_order=classes,
            palette=palette,
            s=30, alpha=0.2, legend=False
        )
        
        # Draw continuous paths
        for i in range(0, len(proj), pts_per_prompt):
            traj = proj[i : i + pts_per_prompt]
            cls_name = labels[i]
            c = palette[classes.index(cls_name)]
            
            # The actual path over timesteps
            plt.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.6, linewidth=1.5)
            
            # Pure Noise State (Start) - marked with circle
            plt.scatter(traj[0, 0], traj[0, 1], color=c, marker='o', s=80, edgecolor='k', zorder=5)
            # Final Image State (End) - marked with bold X
            plt.scatter(traj[-1, 0], traj[-1, 1], color=c, marker='X', s=150, edgecolor='k', zorder=5)
            
        # Manually create legend elements for clarity
        import matplotlib.lines as mlines
        legend_elements = [
            mlines.Line2D([], [], color=palette[i], marker='s', linestyle='None',
                          markersize=10, label=classes[i].capitalize()) for i in range(len(classes))
        ]
        legend_elements.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                          markersize=10, label='Start (Isotropic Noise)'))
        legend_elements.append(mlines.Line2D([], [], color='gray', marker='X', linestyle='None',
                          markersize=12, label='End (Semantic Crystal)'))

        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"{name} Latent Trajectories Over Denoisings", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out_path = os.path.join(config.OUTPUT_DIR, f"{name.lower()}_trajectories.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved {name} trajectory map to {out_path}")

    print("✓ Visualizations completed successfully.")

if __name__ == "__main__":
    plot_visualizations()
