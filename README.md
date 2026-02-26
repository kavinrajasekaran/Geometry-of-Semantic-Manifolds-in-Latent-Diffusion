# The Geometry of Semantic Manifolds in Latent Diffusion

A PyTorch `diffusers` implementation to visualize the internal trajectory of how images crystallize from pure isotropic Gaussian noise within Stable Diffusion. This project extracts, flattens, and applies Unsupervised Learning dimensionality reduction techniques (PCA, t-SNE, UMAP) to the $16,384$-dimensional internal U-Net Bottleneck Tensors and corresponding VAE Image outputs to visualize the exact moment explicit "semantic manifolds" (e.g. Cats, Cars, Cities) violently branch away from overlapping noise.

## 🚀 Features

- **Latent Interception**: Custom PyTorch hooks directly into `runwayml/stable-diffusion-v1-5` to extract massive tensors mid-generation across explicit denosing timesteps (e.g. `t=50, 40, ... 0`).
- **Parallel Output Dimensionality Reduction**: Separately flattens, PCA-compresses, and Non-Linearly embeds both internal mathematical **Latents** and final **Image Pixels**. 
- **Publication-Ready Visualizations**: 
  - **Overall Scatter**: Views the fully reduced manifolds across all timesteps to prove distinct semantic clustering.
  - **Trajectory Maps**: Maps explicit generation paths starting from a central noise distribution and traversing into isolated conceptual groups.
- **Dynamic Extensibility**: By default tracks Cats, Cars, Cities, Landscapes, and Portraits, but supports interactive custom user prompts directly via CLI!

## 📦 Installation
```bash
git clone https://github.com/your-username/latent-diffusion-geometry.git
cd latent-diffusion-geometry
pip install -r requirements.txt
```

## 🛠 Usage
This project operates in a three-stage sequential timeline. Follow these steps in order:

### 1. Extraction (Interception & Tensor Saving)
Run the script to initialize Stable Diffusion, optionally input a custom test prompt through the CLI to track alongside the defaults, and write the intercepted massive tensors to `.npz` arrays.
```bash
python3 extractor.py
```

### 2. Dimensionality Reduction (PCA / t-SNE / UMAP)
Compress the highly-noisy extracted latents and image pixels.
```bash
python3 reducer.py
```

### 3. Visualization
Generate and save all trajectory and cluster plots to the `outputs/` folder.
```bash
python3 visualizer.py
```

## 📂 Project Structure

```
.
├── config.py          # ⚙️ Constants, tracked timesteps, and default prompt definitions
├── extractor.py       # 🧠 Injects diffusers hooks and intercepts U-Net tensors to .npz
├── reducer.py         # 📉 Unsupervised learning pipeline (PCA/t-SNE/UMAP) 
├── visualizer.py      # 📊 Translates reduced coordinates into Trajectory Scatterplots
└── requirements.txt   # 📦 Python dependencies
```
