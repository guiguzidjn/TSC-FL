import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

# Basic ResNet block with residual connections and multi-scale convolution
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super(ResNetBlock, self).__init__()

        # Multi-scale convolutions
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Optional dropout
        self.dropout = nn.Dropout(dropout)

        # Identity shortcut (skip connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # Multi-scale convolutions
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = out1 + out2 + out3

        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Add the shortcut connection
        out += self.shortcut(x)
        out = self.relu(out)

        return out

# Enhanced ResNet model with Transformer
class EnhancedResNetModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_blocks, num_heads, dropout):
        super(EnhancedResNetModel, self).__init__()

        # Embedding layer to map input to d_model dimension
        self.embedding = nn.Linear(input_dim, d_model)

        # Create a series of ResNet blocks
        self.res_blocks = nn.Sequential(
            *[ResNetBlock(d_model, d_model, stride=1, dropout=dropout) for _ in range(num_blocks)]
        )

        # Transformer encoder for capturing long-term dependencies
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout),
            num_layers=2
        )

        # Global average pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Fully connected output layer
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)

        # Add a channel dimension for Conv1d
        x = x.unsqueeze(2)  # Now the shape is (batch_size, d_model, 1)

        # Pass through the ResNet blocks
        x = self.res_blocks(x)

        # # Apply global average pooling
        x = self.pooling(x).squeeze(-1)  # Shape: (batch_size, d_model)

        # # Output layer for classification
        x = self.fc(x)

        return x

