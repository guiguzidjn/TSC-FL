# -*- coding: utf-8 -*-
"""
Stage 1 Binary Classification Backbones
ResNet (original), MLP, 1D-CNN, LSTM, GRU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResBTD.binary_Classification import ResNetBlock


class Stage1ResNet(nn.Module):
    """Original ResNet-based binary classifier (from ResBTD)."""
    def __init__(self, input_dim, d_model=128, num_blocks=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.res_blocks = nn.Sequential(
            *[ResNetBlock(d_model, d_model, stride=1, dropout=dropout) for _ in range(num_blocks)]
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(2)
        x = self.res_blocks(x)
        x = self.pooling(x).squeeze(-1)
        return self.fc(x)


class Stage1MLP(nn.Module):
    """Simple MLP binary classifier."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Stage1CNN(nn.Module):
    """1D-CNN binary classifier."""
    def __init__(self, input_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class Stage1LSTM(nn.Module):
    """LSTM-based binary classifier."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        return self.fc(x)


def get_stage1_model(name, input_dim, **kwargs):
    """Factory for Stage 1 models."""
    models = {
        'resnet': Stage1ResNet,
        'mlp': Stage1MLP,
        'cnn': Stage1CNN,
        'lstm': Stage1LSTM,
    }
    if name not in models:
        raise ValueError(f"Unknown Stage 1 model: {name}. Choose from {list(models.keys())}")
    return models[name](input_dim=input_dim, **kwargs)
