import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionV3_RNN(nn.Module):
    def __init__(self, num_classes=7, unfreeze_layers=112):
        super(InceptionV3_RNN, self).__init__()
        # Load InceptionV3 with pre-trained PyTorch weights
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
        
        # Safely disable the auxiliary outputs so it doesn't break our RNN during training
        inception.aux_logits = False

        # Replace the final FC layer with Identity to get raw features (2048 dims)
        inception.fc = nn.Identity()
        self.features = inception
        
        # Freeze top layers
        params = list(self.features.parameters())
        total_params = len(params)
        freeze_up_to = total_params - unfreeze_layers
        for i, param in enumerate(params):
            param.requires_grad = (i >= freeze_up_to)
                
        # LSTM Layers
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        
        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        batch_size, seq_len, C, H, W = x.size()
        
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.features(x)
        x = x.view(batch_size, seq_len, -1)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        x = x[:, -1, :]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x