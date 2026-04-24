# Detection of Human Emotions using Hybrid CNN-RNN

## Overview
This repository provides a PyTorch implementation of a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) architecture for Facial Expression Recognition (FER). It builds upon the methodologies described in the paper *"Detection of human emotions through facial expressions using hybrid convolutional neural network-recurrent neural network algorithm"*.

The framework extracts spatial features using pre-trained **MobileNetV2** or **InceptionV3** models and processes them through an LSTM-based RNN to classify human emotions.

## Features
* **Hybrid Architecture**: Combines CNNs (MobileNetV2 / InceptionV3) with RNNs (LSTM) for robust feature extraction and classification.
* **Automated Data Management**: Seamlessly downloads and prepares Kaggle datasets (FER2013 and CK+) using `kagglehub` with automatic train/validation splitting.
* **Hardware Acceleration**: Automatically detects and utilizes NVIDIA GPUs (`cuda`), Apple Silicon (`mps`), or defaults to `cpu`.
* **Robust Training Setup**: Features gradient clipping, custom learning rate decay, and comprehensive experiment logging/checkpointing.
* **Configurable Parameters**: Easily manage hyperparameters and directories through `config.py` or command-line arguments.

## Project Structure
```text
├── config.py                 # Hyperparameters and system configuration
├── train.py                  # Main training loop with logging and checkpointing
├── models/
│   ├── mobilenetv2_rnn.py    # MobileNetV2 + LSTM model definition
│   ├── inceptionv3_rnn.py    # InceptionV3 + LSTM model definition
├── utils/
│   ├── data_loader.py        # Dataset downloading, transformations, and loaders
│   ├── training_utils.py     # Custom functions for LR decay and gradient clipping
└── README.md
```

## Setup & Installation

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/adelovelace/CV.git
cd CV
```

### 2. Environment Setup
It is recommended to use a virtual environment (like `venv` or `conda`).

**Option A: Using `venv`**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

**Option B: Using `conda`**
```bash
conda create -n fer_env python=3.10
conda activate fer_env
```

### 3. Install Dependencies
You will need PyTorch and other supporting libraries. Install them via `pip`:
```bash
pip install torch torchvision kagglehub
```
*(Note: If you have an NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch from the [official website](https://pytorch.org/get-started/locally/).)*

## Usage

### 1. Configuration
You can adjust the default hyperparameters directly in `config.py` or pass them as arguments when running the training script.

### 2. Training
To start training the model, simply run:
```bash
python train.py
```

**Example with command-line arguments:**
```bash
python train.py --epoch 150 --lr 0.0001 --batchsize 32 --trainsize 224 --gpu_id 0
```

### Arguments (from `config.py`)
* `--epoch`: Total number of training epochs (default: 180)
* `--lr`: Initial learning rate (default: 5e-5)
* `--batchsize`: Training batch size (default: 24)
* `--trainsize`: Input image resolution (e.g., 224 for MobileNet, 299 for Inception)
* `--clip`: Gradient clipping margin (default: 0.5)
* `--decay_rate`: Decay rate for learning rate (default: 0.1)
* `--decay_epoch`: Epoch interval to apply learning rate decay (default: 100)
* `--load`: Path to load pre-trained weights from a checkpoint
* `--gpu_id`: Specify which GPU to use (default: '0')
* `--save_path`: Directory to save logs and model checkpoints

## Outputs
During training, the script will automatically create the following in your `--save_path` (or `./outputs` by default):
* `logs/`: Contains a `.log` file tracking loss and accuracy for every epoch.
* `checkpoints/`: Contains `.pth` files for the model at every epoch, as well as the dynamically updated `best_model.pth` based on highest validation accuracy.
