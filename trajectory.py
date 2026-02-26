"""
trajectory.py — Computes trajectory curvature metrics for each prompt's
denoising path through the latent space.

Measures:
  1. Sinuosity: Total path length / straight-line distance.
     A value of 1.0 means a perfectly straight path.
     Higher values indicate more wandering/curved trajectories.

  2. Angular Deviation: The average angle (in degrees) between consecutive
     path segments. 0° = perfectly straight, 180° = complete reversal.

All computations are done in PCA-reduced space (50D) for numerical stability,
not in the 2D t-SNE space, since t-SNE distorts real distances.
"""

import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import config

def compute_angle(v1, v2):
    """Compute angle in degrees between two vectors."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def run_trajectory_analysis():
    in_file = os.path.join(config.OUTPUT_DIR, "extracted_latents.npz")
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"{in_file} not found. Run extractor.py first.")
    
    data = np.load(in_file, allow_pickle=True)
    embeddings = data["embeddings"]
    bottlenecks = data["bottlenecks"] if "bottlenecks" in data else None
    labels = data["labels"]
    steps = data["steps"]
    
    unique_steps = sorted(set(steps))
    pts_per_prompt = len(unique_steps)
    
    # We need to figure out total number of prompts
    # Data is arranged as: [prompt0_step0, prompt0_step10, ..., prompt0_step49, prompt1_step0, ...]
    total_prompts = len(embeddings) // pts_per_prompt
    
    representations = {"Spatial Latent": embeddings}
    if bottlenecks is not None:
        representations["U-Net Bottleneck"] = bottlenecks
    
    print("=" * 70)
    print("TRAJECTORY CURVATURE ANALYSIS")
    print("=" * 70)
    print(f"Total prompts: {total_prompts}")
    print(f"Steps per trajectory: {pts_per_prompt} ({unique_steps})")
    print()
    
    for rep_name, rep_data in representations.items():
        # PCA to reduce dimensionality for stable distance computation
        n_pca = min(50, len(rep_data), rep_data.shape[1])
        pca = PCA(n_components=n_pca, random_state=42)
        rep_pca = pca.fit_transform(rep_data)
        
        print(f"\n{'─' * 60}")
        print(f"  {rep_name} (PCA-reduced to {n_pca}D)")
        print(f"{'─' * 60}")
        print(f"  {'Prompt':>40}  {'Class':>10}  {'Sinuosity':>10}  {'Avg Angle':>10}")
        print(f"  {'──────':>40}  {'─────':>10}  {'─────────':>10}  {'─────────':>10}")
        
        class_sinuosities = {}
        class_angles = {}
        prompt_data = []
        
        for p in range(total_prompts):
            start = p * pts_per_prompt
            end = start + pts_per_prompt
            traj = rep_pca[start:end]
            label = labels[start]
            
            # Compute segment lengths
            segments = np.diff(traj, axis=0)
            segment_lengths = np.linalg.norm(segments, axis=1)
            total_path_length = np.sum(segment_lengths)
            
            # Straight-line distance (start to end)
            straight_dist = np.linalg.norm(traj[-1] - traj[0])
            
            # Sinuosity
            if straight_dist > 1e-10:
                sinuosity = total_path_length / straight_dist
            else:
                sinuosity = float('inf')
            
            # Angular deviation between consecutive segments
            angles = []
            for i in range(len(segments) - 1):
                if np.linalg.norm(segments[i]) > 1e-10 and np.linalg.norm(segments[i+1]) > 1e-10:
                    angle = compute_angle(segments[i], segments[i+1])
                    angles.append(angle)
            
            avg_angle = np.mean(angles) if angles else 0.0
            
            # Get the prompt text from config
            prompt_idx_in_class = 0
            count = 0
            prompt_text = "unknown"
            for cls, prompts in config.TEXT_PROMPTS.items():
                for pt in prompts:
                    if count == p:
                        prompt_text = pt[:38]
                    count += 1
            
            print(f"  {prompt_text:>40}  {label:>10}  {sinuosity:>10.2f}  {avg_angle:>9.1f}°")
            
            prompt_data.append({
                "label": label,
                "sinuosity": sinuosity,
                "avg_angle": avg_angle,
                "prompt": prompt_text
            })
            
            if label not in class_sinuosities:
                class_sinuosities[label] = []
                class_angles[label] = []
            class_sinuosities[label].append(sinuosity)
            class_angles[label].append(avg_angle)
        
        # Print class-level summaries
        print(f"\n  {'Class Averages':>40}")
        print(f"  {'─' * 40}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
        for cls in sorted(class_sinuosities.keys()):
            mean_sin = np.mean(class_sinuosities[cls])
            mean_ang = np.mean(class_angles[cls])
            print(f"  {'[' + cls.upper() + ']':>40}  {'':>10}  {mean_sin:>10.2f}  {mean_ang:>9.1f}°")
        
        # ─── Plot: Sinuosity by Class ────────────────────────────────────────
        classes = sorted(class_sinuosities.keys())
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of sinuosity
        ax = axes[0]
        x_pos = range(len(classes))
        means = [np.mean(class_sinuosities[c]) for c in classes]
        stds = [np.std(class_sinuosities[c]) for c in classes]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6'][:len(classes)],
                      edgecolor='white', linewidth=1.5, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
        ax.set_ylabel("Sinuosity (path length / straight distance)", fontsize=11)
        ax.set_title(f"{rep_name}: Trajectory Sinuosity", fontsize=13, fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfectly Straight')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Bar chart of angular deviation
        ax = axes[1]
        means_ang = [np.mean(class_angles[c]) for c in classes]
        stds_ang = [np.std(class_angles[c]) for c in classes]
        bars = ax.bar(x_pos, means_ang, yerr=stds_ang, capsize=5, 
                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6'][:len(classes)],
                      edgecolor='white', linewidth=1.5, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
        ax.set_ylabel("Average Angular Deviation (degrees)", fontsize=11)
        ax.set_title(f"{rep_name}: Trajectory Angular Deviation", fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        safe_name = rep_name.lower().replace(" ", "_").replace("-", "_")
        out_path = os.path.join(config.OUTPUT_DIR, f"trajectory_curvature_{safe_name}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved curvature plot to {out_path}")
        
        # ─── Plot: Per-step displacement magnitude ───────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for cls in classes:
            cls_displacements = []
            for p in range(total_prompts):
                if labels[p * pts_per_prompt] != cls:
                    continue
                start = p * pts_per_prompt
                traj = rep_pca[start:start + pts_per_prompt]
                disps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                cls_displacements.append(disps)
            
            if cls_displacements:
                mean_disps = np.mean(cls_displacements, axis=0)
                std_disps = np.std(cls_displacements, axis=0)
                step_transitions = [f"{unique_steps[i]}→{unique_steps[i+1]}" 
                                   for i in range(len(unique_steps) - 1)]
                ax.plot(step_transitions, mean_disps, 'o-', label=cls.capitalize(), 
                        linewidth=2, markersize=7)
                ax.fill_between(step_transitions,
                               mean_disps - std_disps,
                               mean_disps + std_disps,
                               alpha=0.1)
        
        ax.set_xlabel("Step Transition", fontsize=12)
        ax.set_ylabel("Displacement Magnitude (PCA-space)", fontsize=12)
        ax.set_title(f"{rep_name}: Step-by-Step Displacement", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path2 = os.path.join(config.OUTPUT_DIR, f"displacement_per_step_{safe_name}.png")
        plt.savefig(out_path2, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved displacement plot to {out_path2}")
    
    print(f"\n{'=' * 70}")
    print("INTERPRETATION:")
    print("  Sinuosity ≈ 1.0: Trajectory is a straight line (direct path)")
    print("  Sinuosity >> 1.0: Trajectory wanders/curves significantly")
    print("  Angular deviation ≈ 0°: Path continues in same direction each step")
    print("  Angular deviation >> 90°: Path makes sharp turns between steps")
    print("  Compare classes: Do some categories take more direct paths than others?")
    print("  Compare mixed prompts: Does 'cat on a car' have higher sinuosity?")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    run_trajectory_analysis()
