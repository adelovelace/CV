import argparse

parser = argparse.ArgumentParser(description="Hybrid CNN-RNN for Emotion Detection")

# Training Hyperparameters
parser.add_argument('--epoch', type=int, default=150, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--trainsize', type=int, default=299, help='image size (224 for mobilenet, 299 for inception, 44 for custom)')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--frozen', type=str, default='112', help='Tag for frozen layers to append to fine-tuning save files (e.g., 112, all, none)')

# Architecture and Data
parser.add_argument('--model', type=str, default='inception', choices=['inception', 'mobilenet', 'custom'], help='Choose the backbone model')
parser.add_argument('--dataset', type=str, default='fer2013', choices=['fer2013', 'ckplus'], help='Choose the dataset')

# Paths and Hardware
parser.add_argument('--load', type=str, default='', help='path to load pre-trained checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu ID')
parser.add_argument('--save_path', type=str, default='./outputs', help='the path to save models and logs')
parser.add_argument('--seed', type=int, default=42, help='Random seed for strict reproducibility')

opt = parser.parse_args()