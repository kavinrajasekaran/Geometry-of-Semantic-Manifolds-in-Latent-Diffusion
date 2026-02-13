# Deep Convolutional Auto-Encoder (DCAE)

A modular PyTorch implementation of a Deep Convolutional Auto-Encoder, following the architectural principles described in [Turchenko & Luczak (2015)](https://arxiv.org/abs/1512.01596).

This project implements a symmetric Encoder-Decoder architecture with a dedicated flattened bottleneck layer, allowing for explicit latent space visualization and manipulation.

## 🚀 Features

- **Modular Design**: Easily swap datasets (CIFAR-10, MNIST), adjust bottleneck size, or modify network depth via configuration.
- **Explicit Bottleneck**: A flattened linear layer between the encoder and decoder serves as the compressed latent representation.
- **Visualization Suite**:
  - **Reconstructions**: Side-by-side comparison of original vs. reconstructed images.
  - **Latent Space**: 2D projection of the latent vectors using **UMAP** or **t-SNE**, colored by class.
  - **Training Curves**: Real-time tracking of Train/Test MSE loss.
- **PyTorch Native**: Built with `torch`, `torchvision`, and `torch.nn` modules.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/dcae-pytorch.git
   cd dcae-pytorch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

The `main.py` script is the single entry point for training and evaluation.

### Quick Start (CIFAR-10)
Train a standard model on CIFAR-10 with a 128-dimensional bottleneck:
```bash
python main.py
```

### Custom Configuration
Train on MNIST with a smaller 32-dimensional bottleneck for 20 epochs:
```bash
python main.py --dataset mnist --latent_dim 32 --num_epochs 20
```

### Visualization Options
Use **t-SNE** instead of UMAP for the latent space scatter plot:
```bash
python main.py --vis_method tsne
```

### Evaluation Only
Skip training and load a pre-trained checkpoint to generate plots:
```bash
python main.py --eval_only --checkpoint outputs/dcae_checkpoint.pt
```

## 📂 Project Structure

```
.
├── config.py          # ⚙️ Central configuration (Hyperparameters, Paths)
├── dataset.py         # 💾 Data loading logic (CIFAR-10, MNIST)
├── model.py           # 🧠 DCAE Architecture (Encoder, Decoder, Bottleneck)
├── train.py           # 🔄 Training loop & Checkpointing
├── visualize.py       # 📊 Plotting utilities (Reconstructions, UMAP/t-SNE)
├── main.py            # 🚀 Entry point (CLI argument parsing)
└── requirements.txt   # 📦 Python dependencies
```

## 📐 Architecture Details

- **Encoder**: 3 blocks of `Conv2d` → `BatchNorm` → `ReLU` → `MaxPool2d`. Downsamples input (e.g., 32x32) to a 4x4 feature map.
- **Bottleneck**: Flattened 4x4 maps projected to a linear `latent_dim` vector.
- **Decoder**: Symmetric mirror. Projects latent vector back to 4x4, then `ConvTranspose2d` (Upsampling) layers to recover original spatial dimensions.
- **Output**: `Sigmoid` activation ensures pixel values are in `[0, 1]`.

## 📄 Reference
Turchenko, V., & Luczak, A. (2015). *Creation of a Deep Convolutional Auto-Encoder in Caffe*. arXiv preprint arXiv:1512.01596.
