# AffectiveVision: Advanced Spatio-Temporal Architectures for Facial Expression Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](#)
[![Weights & Biases](https://img.shields.io/badge/W&B-Experiment_Tracking-FFBE00.svg?logo=weightsandbiases&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](#)

## Abstract
**AffectiveVision** is a comprehensive, PyTorch-based Deep Learning framework designed for high-performance Facial Expression Recognition (FER). It classifies facial expressions into seven core discrete affective states: *Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise*. 

Moving beyond traditional static-image classification, this framework implements hybrid spatio-temporal neural networks. It utilizes deep Convolutional Neural Networks (CNNs) to extract high-dimensional spatial feature maps, and passes those sequential features through Recurrent Neural Networks (LSTMs/GRUs) to model complex temporal dependencies.

The codebase is meticulously engineered for rigorous Machine Learning research, supporting strict hardware determinism, automated MLOps telemetry, dynamic data provisioning, and post-hoc interpretability. Furthermore, it introduces robust asynchronous inference engines for evaluating video-based affective datasets and live local environments.

## Core Architectural Configurations
The framework is highly modular, supporting three primary architectural paradigms:

### 1. InceptionV3 + LSTM (`--model inception`)
* **Spatial Backbone**: Pre-trained InceptionV3. Auxiliary auxiliary computing blocks are explicitly disabled to clean the computational graph. Uses factorized 1x1 to 3x3 convolutions to achieve multi-scale spatial feature extraction.
* **Temporal Modeling**: A highly parameterized 2-layer LSTM. (Input: 2048-dim, Hidden Layers: 512 $\rightarrow$ 128-dim).
* **Target Domain**: Input dimensions are dynamically upsampled to $299 \times 299$. This model is heavily parameterized and optimized for high-capacity datasets with complex intra-class variances (e.g., CK+).

### 2. MobileNetV2 + LSTM (`--model mobilenet`)
* **Spatial Backbone**: Pre-trained MobileNetV2 utilizing depthwise separable convolutions to drastically reduce the parameter footprint while maintaining dense feature representations.
* **Temporal Modeling**: 2-layer LSTM (Input: 1280-dim, Hidden Layers: 512 $\rightarrow$ 128-dim).
* **Target Domain**: $224 \times 224$ input resolution. Architected specifically for lightweight execution, real-time latency requirements, and edge-device deployment.

### 3. Custom VGG-Style CNN + GRU (`--model custom`)
* **Spatial Backbone**: A bespoke 3-block CNN trained from scratch. Each block utilizes $3 \times 3$ convolutions, Batch Normalization, and Max Pooling. Crucially, it replaces dense flatten operations with Global Average Pooling (GAP) prior to the recurrent layer to prevent parameter explosion and overfitting.
* **Temporal Modeling**: 1-layer GRU (Gated Recurrent Unit). (Input: 128-dim, Hidden: 128-dim). GRUs are selected here over LSTMs to accelerate gradient flow given the model is trained entirely from scratch.
* **Target Domain**: Native $48 \times 48$ resolution. Explicitly optimized to train on the raw, unaltered FER2013 dataset dimensions to avoid upsampling artifacts.

## Advanced Engineering & Features

* **Strict Reproducibility & Determinism (`--seed`)**: Implements absolute global seed locking across Python's native `random`, NumPy, and PyTorch (CPU/GPU). Forces deterministic `cuDNN` benchmarking algorithms to guarantee 100% identical computational replication across independent runs.
* **Dynamic MLOps Telemetry**: Deeply integrated with Weights & Biases (W&B). Automatically routes artifact nomenclature, logs learning curves, and tracks advanced metrics (Macro-F1, Precision, Recall) crucial for the imbalanced nature of affective datasets.
* **Visual Interpretability (Grad-CAM)**: Temporarily disables PyTorch's optimized `cuDNN` RNN engine in `eval()` mode to bypass standard backwards-pass restrictions. This allows the system to hook into the terminal spatial convolutions and compute Gradient-weighted Class Activation Mappings (Grad-CAM), directly visualizing the focal points of the network's attention (e.g., eyes, mouth).
* **Temporal Video Aggregation (`video_evaluator.py`)**: Extends evaluation to continuous video datasets (e.g., DepVidMood). It utilizes temporal frame decimation (processing $n$-th frames to optimize I/O) and dynamic bounding-box spatial sorting to isolate the primary subject. It concludes inference via sequence-level majority voting.
* **Real-Time Asynchronous Inference (`live_demo.py`)**: A local deployment architecture leveraging `cv2.CascadeClassifier`. It features an asynchronous threading model that decouples visual inference from blocking Text-to-Speech (TTS) computational threads, guaranteeing zero-latency video rendering.

## Project Directory Structure
```text
├── config.py                 # Argparse definitions, hyperparameters, and routing configurations
├── train.py                  # Core training loop, early stopping logic, MLOps hooks, and evaluation metrics
├── run_experiments.sh        # Bash shell pipeline for automated, unattended ablation studies
├── live_demo.py              # Real-time multi-threaded webcam inference with local TTS feedback
├── video_evaluator.py        # Video sequence inference and temporal frame decimation engine
├── download_videos.py        # Automated Kaggle API data provisioning for video datasets
├── models/
│   ├── custom_cnn_rnn.py     # Custom VGG-Style CNN + GRU architecture
│   ├── inceptionv3_rnn.py    # InceptionV3 + LSTM transfer learning module
│   └── mobilenetv2_rnn.py    # MobileNetV2 + LSTM transfer learning module
└── utils/
    ├── data_loader.py        # Reproducible random_split transforms, normalizations, and dataloaders
    └── training_utils.py     # Gradient clipping boundaries and custom step-based LR decay rules
```

## Setup & Dependencies

```bash
# Clone the repository
git clone https://github.com/adelovelace/AffectiveVision.git
cd AffectiveVision

# Create a virtual environment (Python 3.10+ recommended)
conda create -n cv_env python=3.10
conda activate cv_env

# Install core dependencies 
# (Note: Use opencv-python-headless if executing on Linux clusters/Singularity)
pip install -r requirements.txt
pip install pyttsx3 # Required for auditory feedback in local live_demo.py

# Authenticate the MLOps Dashboard
wandb login
```

## Usage & Execution Documentation

### 1. Single Experiment Execution
The `train.py` script is governed entirely via the CLI arguments defined in `config.py`. Missing datasets (FER2013/CK+) are dynamically provisioned via the Kaggle API upon execution.

```bash
# Fine-tune Inception on CK+ (Freezing the first 112 layers, locking hardware seed to 42)
python train.py --model inception --dataset ckplus --frozen 112 --seed 42 --epoch 150 --lr 5e-5

# Train the lightweight Custom CNN-GRU from scratch on FER2013
python train.py --model custom --dataset fer2013 --batchsize 64
```

### 2. Automated Ablation Pipeline
Execute a fully unattended grid search sweeping across architectural backbones, datasets, and frozen layer variants (30, 60, 80 layers).
```bash
chmod +x run_experiments.sh
nohup ./run_experiments.sh > pipeline_output.log 2>&1 &
```

### 3. Video Sequence Inference
Automate the downloading and processing of raw `.mp4/.avi` datasets utilizing the temporal sequence aggregation engine.
```bash
python download_videos.py  # Fetches DepVidMood from the Kaggle registry
python video_evaluator.py  # Executes face extraction, frame decimation, and majority-vote inference
```

### 4. Real-Time Hardware Inference
*Note: This script requires a local machine with accessible camera and audio peripherals.*
```bash
python live_demo.py
```

---
*MIT License | Copyright (c) 2026*
