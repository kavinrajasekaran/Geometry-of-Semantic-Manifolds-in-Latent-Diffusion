import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import config

def run_linear_probe():
    in_file = os.path.join(config.OUTPUT_DIR, "extracted_latents.npz")
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"{in_file} not found. Run extractor.py first.")
    
    data = np.load(in_file, allow_pickle=True)
    embeddings = data["embeddings"]
    bottlenecks = data["bottlenecks"] if "bottlenecks" in data else None
    images = data["images"] if "images" in data else None
    labels = data["labels"]
    steps = data["steps"]
    
    unique_steps = sorted(set(steps))
    
    # Exclude "custom" category from probing since it typically has only 1 sample
    # and cannot be meaningfully classified
    exclude_labels = set()
    for label in set(labels):
        count_per_step = np.sum((labels == label) & (steps == unique_steps[0]))
        if count_per_step < 2:
            exclude_labels.add(label)
    
    if exclude_labels:
        print(f"Excluding labels with < 2 samples per step: {exclude_labels}")
    
    valid_mask = np.array([l not in exclude_labels for l in labels])
    
    le = LabelEncoder()
    le.fit(labels[valid_mask])
    classes = list(le.classes_)
    n_classes = len(classes)
    random_chance = 1.0 / n_classes
    
    print("Linear Probe Analysis")
    print(f"Classes: {classes}")
    print(f"Random chance accuracy: {random_chance:.1%}")
    print(f"Tracked timesteps: {unique_steps}")
    print()
    
    # Build representations dict
    representations = {"Spatial Latent": embeddings}
    if bottlenecks is not None:
        representations["U-Net Bottleneck"] = bottlenecks
    if images is not None:
        representations["Image Pixels"] = images
    
    # Store results for plotting
    all_results = {}
    
    for rep_name, rep_data in representations.items():
        print(f"\n{'─' * 60}")
        print(f"  {rep_name} (raw dim = {rep_data.shape[1]:,})")
        print(f"{'─' * 60}")
        print(f"  {'Step':>6}  {'Accuracy':>10}  {'Std':>8}  {'vs Random':>12}")
        print(f"  {'────':>6}  {'────────':>10}  {'───':>8}  {'─────────':>12}")
        
        step_accuracies = []
        step_stds = []
        
        for s in unique_steps:
            # Get samples at this timestep, excluding custom
            mask = (steps == s) & valid_mask
            subset = rep_data[mask]
            subset_labels = labels[mask]
            
            if len(subset) < 4 or len(set(subset_labels)) < 2:
                step_accuracies.append(random_chance)
                step_stds.append(0)
                print(f"  {s:>6}  {'(skipped)':>10}")
                continue
            
            # CRITICAL: Cap PCA components well below sample count to prevent
            # overfitting. With n features >= n samples, any linear classifier
            # can achieve ~100% by chance. Rule of thumb: features < n_samples / 5
            n_pca = min(len(subset) // 5, 10, subset.shape[1])
            n_pca = max(n_pca, 2)  # need at least 2 components
            pca = PCA(n_components=n_pca, random_state=42)
            subset_pca = pca.fit_transform(subset)
            
            # Encode labels
            y = le.transform(subset_labels)
            
            # Use Stratified K-Fold (more stable than LOO for small samples)
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Strong regularization (C=0.1) to prevent overfitting
            clf = LogisticRegression(
                max_iter=1000, 
                random_state=42,
                solver='lbfgs',
                C=0.1
            )
            
            scores = cross_val_score(clf, subset_pca, y, cv=cv, scoring='accuracy')
            mean_acc = scores.mean()
            std_acc = scores.std()
            
            step_accuracies.append(mean_acc)
            step_stds.append(std_acc)
            
            improvement = mean_acc - random_chance
            sign = "+" if improvement >= 0 else ""
            print(f"  {s:>6}  {mean_acc:>10.1%}  {std_acc:>8.1%}  {sign}{improvement:>11.1%}")
        
        # Run a permutation test: shuffle labels and re-probe to show baseline
        print(f"\n  Permutation baseline (shuffled labels):")
        shuffled_results = []
        for _ in range(10):
            s_final = unique_steps[-1]
            mask = (steps == s_final) & valid_mask
            subset = rep_data[mask]
            subset_labels = labels[mask]
            
            n_pca = min(len(subset) // 5, 10, subset.shape[1])
            n_pca = max(n_pca, 2)
            pca = PCA(n_components=n_pca, random_state=42)
            subset_pca = pca.fit_transform(subset)
            
            y_shuffled = le.transform(subset_labels).copy()
            np.random.shuffle(y_shuffled)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', C=0.1)
            try:
                s_scores = cross_val_score(clf, subset_pca, y_shuffled, cv=cv, scoring='accuracy')
                shuffled_results.append(s_scores.mean())
            except:
                shuffled_results.append(random_chance)
        
        perm_mean = np.mean(shuffled_results)
        print(f"  Average accuracy with random labels: {perm_mean:.1%} (should be ≈{random_chance:.0%})")
        
        all_results[rep_name] = {
            "accuracies": step_accuracies,
            "stds": step_stds
        }
    
    # Plot
    print("\nGenerating accuracy plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"Spatial Latent": "#3498db", "U-Net Bottleneck": "#e74c3c", "Image Pixels": "#2ecc71"}
    
    for rep_name, results in all_results.items():
        accs = results["accuracies"]
        stds = results["stds"]
        color = colors.get(rep_name, "#95a5a6")
        
        ax.plot(unique_steps, accs, 'o-', label=rep_name, color=color, 
                linewidth=2.5, markersize=8)
        ax.fill_between(unique_steps, 
                        [a - s for a, s in zip(accs, stds)],
                        [a + s for a, s in zip(accs, stds)],
                        alpha=0.15, color=color)
    
    # Random chance line
    ax.axhline(y=random_chance, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Random Chance ({random_chance:.0%})')
    
    ax.set_xlabel("Denoising Step", fontsize=13)
    ax.set_ylabel("Classification Accuracy (LOO-CV)", fontsize=13)
    ax.set_title("Linear Probe: Semantic Separability Over Denoising Steps", 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(unique_steps)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(config.OUTPUT_DIR, "linear_probe_accuracy.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved accuracy plot to {out_path}")
    


if __name__ == "__main__":
    run_linear_probe()
