# Hybrid CNN-RNN Architectures for Facial Expression Recognition (FER)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](#)
[![Weights & Biases](https://img.shields.io/badge/W&B-Experiment_Tracking-FFBE00.svg?logo=weightsandbiases&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](#)

## Abstract
This repository contains a PyTorch-based framework for Facial Expression Recognition (FER) utilizing hybrid spatio-temporal neural networks. Building upon the methodologies of hybrid CNN-RNN algorithms, this framework extracts high-level spatial feature representations using deep Convolutional Neural Networks (CNNs) and captures temporal dependencies/feature correlations using Recurrent Neural Networks (LSTMs/GRUs). 

The codebase is highly modular, supporting automated data pipelines, MLOps tracking, layer-freezing ablation studies, and post-hoc model interpretability via Grad-CAM.

## Architectural Configurations
The framework supports three primary architectural branches:

1. **InceptionV3 + LSTM (`--model inception`)**: 
   * **Spatial**: Pre-trained InceptionV3 (auxiliary logits disabled, adaptive pooling applied).
   * **Temporal**: 2-layer LSTM (Input: 2048, Hidden: 512 -> 128).
   * **Input**: Upsampled to 299x299. Ideal for high-capacity feature extraction on complex datasets like CK+.
2. **MobileNetV2 + LSTM (`--model mobilenet`)**:
   * **Spatial**: Pre-trained MobileNetV2.
   * **Temporal**: 2-layer LSTM (Input: 1280, Hidden: 512 -> 128).
   * **Input**: 224x224. Optimized for edge-deployment and lightweight inference.
3. **Custom CNN + GRU (`--model custom`)**:
   * **Spatial**: 3-block VGG-style CNN trained from scratch with Global Average Pooling (GAP) replacing dense flatten operations to prevent parameter explosion.
   * **Temporal**: 1-layer GRU (Input: 128, Hidden: 128).
   * **Input**: 48x48. Specifically architected for the native resolution of the FER2013 dataset.

## Advanced Features & MLOps

* **Automated Data Provisioning**: Integrates `kagglehub` to dynamically fetch, extract, and cache FER2013 and CK+ datasets. Handles both pre-split directories and dynamic 80/20 validation splitting with seeded generators for reproducibility.
* **Weights & Biases (W&B) Integration**: Full telemetry logging including Train/Val Loss, Macro-F1, Recall, and Accuracy. Automatically tracks hyperparameters and uploads model artifacts (`.pth`) and learning curves.
* **Model Interpretability (Grad-CAM)**: Automatically generates and logs Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps at the end of training. Hooks into the terminal spatial convolutions to visualize the spatial attention dictating the RNN's classification.
* **Hardware Agnosticism**: Dynamic device mapping across `cuda` (NVIDIA), `mps` (Apple Silicon), and `cpu`.
* **Robust Optimization**: Implements explicit gradient clipping (`torch.nn.utils.clip_grad_norm_`) and custom step-based learning rate decay to stabilize RNN backpropagation.

## Project Structure
```text
├── config.py                 # Argparse configurations and hyperparameter definitions
├── train.py                  # Main training loop, MLOps hooks, and evaluation metrics
├── run_experiments.sh        # Bash pipeline for automated ablation studies
├── models/
│   ├── custom_cnn_rnn.py     # VGG-style CNN + GRU (from scratch)
│   ├── inceptionv3_rnn.py    # InceptionV3 + LSTM (Transfer Learning)
│   └── mobilenetv2_rnn.py    # MobileNetV2 + LSTM (Transfer Learning)
└── utils/
    ├── data_loader.py        # Kagglehub fetching, transformations, and loaders
    └── training_utils.py     # Gradient clipping and custom LR decay
```

## Setup & Dependencies

```bash
git clone https://github.com/adelovelace/CV.git
cd CV

# Create environment (Python 3.10+ recommended)
conda create -n cv_env python=3.10
conda activate cv_env

# Install dependencies
pip install -r requirements.txt

# Authenticate MLOps Dashboard
wandb login
```

## Usage & Execution

### 1. Single Experiment Execution
The `train.py` script is fully controlled via CLI arguments defined in `config.py`.

```bash
# Fine-tune Inception on CK+ (Freezing the first 112 layers)
python train.py --model inception --dataset ckplus --frozen 112 --epoch 150 --lr 5e-5

# Train Custom CNN-GRU from scratch on FER2013
python train.py --model custom --dataset fer2013 --batchsize 64
```

### 2. Automated Pipeline (Ablation Studies)
To evaluate the effect of transfer learning depth, the repository includes a bash pipeline (`run_experiments.sh`) that executes a grid search over models, datasets, and frozen layer counts.

```bash
chmod +x run_experiments.sh
nohup ./run_experiments.sh > pipeline_output.log 2>&1 &
```
*The script sequentially executes fine-tuning with 30, 60, and 80 frozen layers across both backbones and datasets, followed by baseline training of the custom model.*

## Evaluation Metrics & Artifacts
Upon meeting early stopping criteria or reaching maximum epochs, the framework outputs:
1. **Model Weights**: `best_<dataset>_frozen<n>_best.pth` saved locally and uploaded to W&B registry.
2. **Classification Report**: Terminal output detailing Precision, Recall, and F1-score per emotion class.
3. **Confusion Matrix**: Seaborn-generated heatmap of class predictions.
4. **Grad-CAM Visualizations**: 8 validation images with overlaid spatial gradients indicating feature importance.

---
*MIT License | Copyright (c) 2026*
