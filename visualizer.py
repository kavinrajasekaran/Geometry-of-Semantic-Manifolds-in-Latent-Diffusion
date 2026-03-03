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
    images_tsne_proj = data["images_tsne"] if "images_tsne" in data else None
    images_umap_proj = data["images_umap"] if "images_umap" in data else None
    bottleneck_tsne_proj = data["bottleneck_tsne"] if "bottleneck_tsne" in data else None
    labels = data["labels"]
    steps = data["steps"]
    
    # Dynamically find all unique classes that were saved during extraction
    # We use a stable sort or keep order from config where possible, appending any new ones
    unique_labels = list(dict.fromkeys(labels))
    classes = []
    for c in config.TEXT_PROMPTS.keys():
        if c in unique_labels:
            classes.append(c)
    for c in unique_labels:
        if c not in classes:
            classes.append(c)
            
    palette = sns.color_palette("tab10", len(classes))
    
    # Dictionary of projection names to arrays
    projections = {"Spatial Latent t-SNE": tsne_proj}
    if umap_proj is not None and len(umap_proj.shape) == 2:
        projections["Spatial Latent UMAP"] = umap_proj

    if bottleneck_tsne_proj is not None:
        projections["U-Net Bottleneck t-SNE"] = bottleneck_tsne_proj

    if images_tsne_proj is not None:
        projections["Image t-SNE"] = images_tsne_proj
    if images_umap_proj is not None and len(images_umap_proj.shape) == 2:
        projections["Image UMAP"] = images_umap_proj

    # 1. Overall Scatter (Clustering over all timesteps)
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
        
        space_type = "Image Pixel" if "Image" in name else "Latent"
        proj_type = name.split(" ")[-1]
        plt.title(f"{proj_type} Projection of {space_type} Manifold (All Timesteps)", fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        out_name = name.lower().replace(" ", "-")
        out_path = os.path.join(config.OUTPUT_DIR, f"{out_name}_overall.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved {name} overall scatter to {out_path}")

    # 2. Line Trajectories (Tracking generations over time)
    print("Generating Latent Trajectory plots...")
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
        space_type = "Image" if "Image" in name else "Latent"
        proj_type = name.split(" ")[-1]
        plt.title(f"{proj_type} {space_type} Trajectories Over Denoisings", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out_name = name.lower().replace(" ", "-")
        out_path = os.path.join(config.OUTPUT_DIR, f"{out_name}_trajectories.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved {name} trajectory map to {out_path}")

    # 3. Per-Timestep Grid (Cluster Emergence Over Time)
    print("Generating Per-Timestep cluster emergence plots...")
    
    unique_steps = sorted(set(steps))
    step_labels_at = {}
    for s in unique_steps:
        mask = steps == s
        step_labels_at[s] = labels[mask]
    
    for space_name in ["Spatial Latent", "U-Net Bottleneck", "Image"]:
        if space_name == "Spatial Latent":
            prefix = "tsne_step_"
        elif space_name == "U-Net Bottleneck":
            prefix = "bottleneck_tsne_step_"
        else:
            prefix = "images_tsne_step_"
        
        available_steps = [s for s in unique_steps if f"{prefix}{s}" in data]
        if not available_steps:
            continue
        
        n_plots = len(available_steps)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, s in enumerate(available_steps):
            ax = axes[idx]
            proj_s = data[f"{prefix}{s}"]
            labels_s = step_labels_at[s]
            
            for ci, cls in enumerate(classes):
                cls_mask = labels_s == cls
                if cls_mask.any():
                    ax.scatter(proj_s[cls_mask, 0], proj_s[cls_mask, 1],
                              color=palette[ci], label=cls.capitalize(),
                              s=80, alpha=0.85, edgecolor='w')
            
            ax.set_title(f"Step {s}", fontsize=13, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.legend(fontsize=8, loc='best')
        
        # Hide unused axes
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f"{space_name} Manifold Clustering Over Denoising Steps", fontsize=16, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        out_path = os.path.join(config.OUTPUT_DIR, f"{space_name.lower()}_per_step_grid.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {space_name} per-step grid to {out_path}")

    print("Visualizations completed successfully.")

if __name__ == "__main__":
    plot_visualizations()
