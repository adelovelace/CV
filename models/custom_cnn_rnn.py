import torch
import torch.nn as nn


class CustomCNN_RNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomCNN_RNN, self).__init__()

        # =========================
        # CNN FEATURE EXTRACTOR
        # =========================
        # Input:  (batch, 3, H, W)
        # Output: (batch, 128)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Global Average Pooling avoids a large fully connected layer.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.cnn_dropout = nn.Dropout(0.3)

        # =========================
        # TEMPORAL MODELING
        # =========================
        # The GRU expects input of shape:
        # (batch, sequence_length, feature_dim)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.gru_dropout = nn.Dropout(0.3)

        # =========================
        # CLASSIFIER
        # =========================
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Accepts either:

        1. Static images:
           x shape = (batch, channels, height, width)

        2. Image sequences:
           x shape = (batch, sequence_length, channels, height, width)

        Returns:
           logits shape = (batch, num_classes)

        Note:
           Do not apply softmax here. nn.CrossEntropyLoss expects raw logits.
        """

        # If the input is a batch of single images, create a sequence length of 1.
        if x.dim() == 4:
            x = x.unsqueeze(1)

        batch_size, seq_len, channels, height, width = x.size()

        # Merge batch and sequence dimensions so each frame passes through CNN.
        x = x.reshape(batch_size * seq_len, channels, height, width)

        # Extract spatial features from each frame.
        x = self.cnn(x)
        x = self.cnn_dropout(x)

        # Restore sequence dimension for the GRU.
        x = x.reshape(batch_size, seq_len, -1)

        # Model temporal dependencies between frame-level features.
        gru_out, _ = self.gru(x)

        # Use the final time step for classification.
        last_out = gru_out[:, -1, :]
        last_out = self.gru_dropout(last_out)

        out = self.classifier(last_out)

        return out
