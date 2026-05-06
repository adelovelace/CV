import torch
import torch.nn as nn

class CustomCNN_RNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomCNN_RNN, self).__init__()
        
        # =========================
        # CNN FEATURE EXTRACTOR
        # =========================
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global Average Pooling (Replaces Flatten)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.cnn_dropout = nn.Dropout(0.4)
        
        # =========================
        # TEMPORAL MODELING (GRU)
        # =========================
        # hidden_size=128, num_layers=1 matches the uploaded Keras code
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.gru_dropout = nn.Dropout(0.5)
        
        # =========================
        # CLASSIFIER
        # =========================
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
            # Note: CrossEntropyLoss applies Softmax internally in PyTorch
        )

    def forward(self, x):
        # x is originally (Batch, Channels, Height, Width)
        # Add a dummy sequence dimension: (Batch, Seq_Len=1, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
            
        batch_size, seq_len, C, H, W = x.size()
        
        # Reshape for CNN: (Batch * Seq_Len, C, H, W)
        x = x.view(batch_size * seq_len, C, H, W)
        
        # Pass through CNN
        x = self.cnn(x)
        x = self.cnn_dropout(x)
        
        # Reshape for RNN: (Batch, Seq_Len, Features)
        x = x.view(batch_size, seq_len, -1)
        
        # Pass through GRU
        gru_out, _ = self.gru(x)
        x = self.gru_dropout(gru_out)
        
        # Get the output from the last time step
        last_out = x[:, -1, :]
        
        # Final classification
        out = self.classifier(last_out)
        
        return out