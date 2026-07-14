# -*- coding: utf-8 -*-
"""
Stage 2 Multi-class Classifier Variants
Mamba (original), MLP, 1D-CNN, LSTM, GRU, Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Stage2Mamba(nn.Module):
    """Original Mamba-based classifier (from MamMTD)."""
    def __init__(self, input_dim, num_classes, d_model=128, dropout=0.3):
        super().__init__()
        from mamba_ssm import Mamba
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.mamba = nn.Sequential(
            Mamba(d_model=d_model, d_state=128, d_conv=4, expand=2),
            nn.LayerNorm(d_model)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size, num_features = x.shape
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        m = self.mamba(x)
        m_pooled = m.mean(dim=1)
        return self.classifier(m_pooled)


class Stage2MLP(nn.Module):
    """Simple MLP multi-class classifier."""
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout=0.3):
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
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Stage2CNN(nn.Module):
    """1D-CNN multi-class classifier."""
    def __init__(self, input_dim, num_classes, dropout=0.3):
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


class Stage2LSTM(nn.Module):
    """LSTM-based multi-class classifier."""
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class Stage2Transformer(nn.Module):
    """Transformer-based multi-class classifier."""
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def get_stage2_model(name, input_dim, num_classes, **kwargs):
    """Factory for Stage 2 models."""
    models = {
        'mamba': Stage2Mamba,
        'mlp': Stage2MLP,
        'cnn': Stage2CNN,
        'lstm': Stage2LSTM,
        'transformer': Stage2Transformer,
    }
    if name not in models:
        raise ValueError(f"Unknown Stage 2 model: {name}. Choose from {list(models.keys())}")
    return models[name](input_dim=input_dim, num_classes=num_classes, **kwargs)
