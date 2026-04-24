import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetV2_RNN(nn.Module):
    def __init__(self, num_classes=7, unfreeze_layers=30):
        super(MobileNetV2_RNN, self).__init__()
        # Load MobileNetV2 with pre-trained PyTorch weights
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        
        # Freeze all layers except the top 'unfreeze_layers'
        total_layers = len(self.features)
        freeze_up_to = total_layers - unfreeze_layers
        for i, child in enumerate(self.features.children()):
            for param in child.parameters():
                param.requires_grad = (i >= freeze_up_to)
                    
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # RNN (LSTM) Layers
        self.lstm1 = nn.LSTM(input_size=1280, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Add a sequence dimension since it's an RNN (Batch, Seq, C, H, W)
        x = x.unsqueeze(1) 
        batch_size, seq_len, C, H, W = x.size()
        
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM sequence processing
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Get output from the last timestep
        x = x[:, -1, :]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x