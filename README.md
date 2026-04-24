# Detection of Human Emotions using Hybrid CNN-RNN

## Overview
This repository provides a PyTorch implementation of a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) architecture for Facial Expression Recognition (FER). It builds upon the methodologies described in the paper *"Detection of human emotions through facial expressions using hybrid convolutional neural network-recurrent neural network algorithm"*.

The framework extracts spatial features using pre-trained **MobileNetV2** or **InceptionV3** models and processes them through an LSTM-based RNN to classify human emotions.

## Features
* **Hybrid Architecture**: Combines CNNs (MobileNetV2 / InceptionV3) with RNNs (LSTM) for robust feature extraction and classification.
* **Automated Data Management**: Seamlessly downloads and prepares Kaggle datasets (FER2013 and CK+) using `kagglehub` with automatic train/validation splitting.
* **Hardware Acceleration**: Automatically detects and utilizes NVIDIA GPUs (`cuda`), Apple Silicon (`mps`), or defaults to `cpu`.
* **Robust Training Setup**: Features gradient clipping, custom learning rate decay, and comprehensive experiment logging/checkpointing.
* **Configurable Parameters**: Easily manage hyperparameters, models, and datasets through command-line arguments.

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

### 1. Training the Model
You can control the entire training process directly from your terminal using command-line arguments. The script will automatically download the required datasets if they are not already present.

**Example 1: Train MobileNetV2 on CK+**
```bash
python train.py --model mobilenet --dataset ckplus --batchsize 64
```

**Example 2: Train InceptionV3 on FER2013 with custom hyperparameters**
```bash
python train.py --model inception --dataset fer2013 --epoch 200 --lr 0.001 --gpu_id 0
```

### 2. Available Arguments (from `config.py`)

**Architecture and Data:**
* `--model`: Choose the backbone model. Options: `inception`, `mobilenet` (default: `inception`).
* `--dataset`: Choose the dataset. Options: `fer2013`, `ckplus` (default: `fer2013`).

**Training Hyperparameters:**
* `--epoch`: Total number of training epochs (default: `150`).
* `--lr`: Initial learning rate (default: `5e-5`).
* `--batchsize`: Training batch size (default: `32`).
* `--trainsize`: Input image resolution (default: `299`). *Note: The script will automatically adjust this to 224 if MobileNet is selected.*
* `--clip`: Gradient clipping margin (default: `0.5`).
* `--decay_rate`: Decay rate for learning rate (default: `0.1`).
* `--decay_epoch`: Epoch interval to apply learning rate decay (default: `50`).

**Hardware and State:**
* `--gpu_id`: Specify which GPU to use (default: `'0'`).
* `--load`: Path to load pre-trained weights from a checkpoint.
* `--save_path`: Directory to save logs and model checkpoints (default: `./outputs`).

## Outputs
During training, the script will automatically create the following in your `--save_path` (or `./outputs` by default):
* `logs/`: Contains a `.log` file tracking loss and accuracy for every epoch.
* `checkpoints/`: Contains `.pth` files for the model at every epoch, as well as the dynamically updated `best_model.pth` based on the highest validation accuracy.
