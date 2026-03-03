# Latent Diffusion Geometry

This project visualizes how images are generated from noise in Stable Diffusion. It extracts the internal U-Net bottleneck tensors and VAE outputs, then applies PCA, t-SNE, and UMAP to visualize the semantic clusters (like cats, cars, and cities) at different denoising steps.

## Features

- **Latent Interception**: Extracts intermediate tensors during Stable Diffusion generation.
- **Dimensionality Reduction**: Compresses the tensors using PCA, t-SNE, or UMAP.
- **Visualizations**: Plots the reduced data across timesteps to show how clusters form, along with generating trajectory maps.
- **Custom Prompts**: Allows adding custom prompts through the CLI.

## Installation
```bash
git clone https://github.com/your-username/latent-diffusion-geometry.git
cd latent-diffusion-geometry
pip install -r requirements.txt
```

## Usage
Run the scripts in the following order:

### 1. Extraction
Runs Stable Diffusion and saves the tensors to an `.npz` file. You can enter a custom prompt here if you want.
```bash
python3 extractor.py
```

### 2. Dimensionality Reduction
Applies the dimensionality reduction techniques to the extracted latents.
```bash
python3 reducer.py
```

### 3. Visualization
Generates the plots and saves them to the `outputs/` folder.
```bash
python3 visualizer.py
```

## Project Structure

```
.
├── config.py          # Settings and prompts
├── extractor.py       # Extracts tensors from Stable Diffusion
├── reducer.py         # Dimensionality reduction (PCA/t-SNE/UMAP) 
├── visualizer.py      # Generates plots
└── requirements.txt   # Dependencies
```
