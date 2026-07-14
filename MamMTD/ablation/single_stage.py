# -*- coding: utf-8 -*-
"""
Single-Stage Model: Direct multi-class classification (no binary filtering).
Used for single-stage vs. two-stage ablation comparison.
"""

import torch
import torch.nn as nn
from MamMTD.ablation.stage2_classifiers import (
    Stage2Mamba, Stage2MLP, Stage2CNN, Stage2LSTM, Stage2Transformer
)


def get_single_stage_model(backbone_name, input_dim, num_classes, **kwargs):
    """Returns a single-stage model that classifies directly into num_classes
    (including benign as class 0)."""
    model_map = {
        'mamba': Stage2Mamba,
        'mlp': Stage2MLP,
        'cnn': Stage2CNN,
        'lstm': Stage2LSTM,
        'transformer': Stage2Transformer,
    }
    if backbone_name not in model_map:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    # For single-stage, num_classes includes benign (class 0)
    return model_map[backbone_name](input_dim=input_dim, num_classes=num_classes, **kwargs)
