# -*- coding: utf-8 -*-
"""
TSC-FL 消融实验配置
===================
定义所有消融实验组合。
"""

# ============================================================
# 基础训练配置
# ============================================================
BASE_CONFIG = {
    'batch_size': 512,
    'num_clients': 5,
    'communication_rounds': 30,
    'local_epochs': 5,
}

# ============================================================
# 数据集配置
# ============================================================
DATASET_CONFIGS = {
    'CICIDS2017': {
        'input_dim': 78,
        'num_classes': 14,
        'num_attack_classes': 13,
        'data_path': r'D:\Chenhaolei_experiment\dataset\CICIDS2017\MachineLearningCVE',
    },
    'UNSW-NB15': {
        'input_dim': 204,
        'num_classes': 10,
        'num_attack_classes': 9,
        'data_path': r'D:\Chenhaolei_experiment\dataset\UNSW-NB15\CSV Files',
    },
    'Nba-IoT': {
        'input_dim': 115,
        'num_classes': 10,
        'num_attack_classes': 9,
        'data_path': r'D:\Chenhaolei_experiment\dataset\N_BaIOT\archive',
    },
}

# ============================================================
# 消融实验 1: 单阶段 vs 两阶段
# ============================================================
ABLATION_STAGE = {
    'single_stage_mamba': {
        'mode': 'single_stage',
        'stage2_backbone': 'mamba',
        'partition': 'dirichlet',
        'alpha': 0.5,
    },
    'two_stage_resnet_mamba': {
        'mode': 'two_stage',
        'stage1_backbone': 'resnet',
        'stage2_backbone': 'mamba',
        'partition': 'dirichlet',
        'alpha': 0.5,
    },
}

# ============================================================
# 消融实验 2: Stage 1 主干网络变体
# ============================================================
ABLATION_STAGE1 = {
    f's1_{name}': {
        'mode': 'two_stage',
        'stage1_backbone': name,
        'stage2_backbone': 'mamba',
        'partition': 'dirichlet',
        'alpha': 0.5,
    }
    for name in ['resnet', 'mlp', 'cnn', 'lstm']
}

# ============================================================
# 消融实验 3: Stage 2 分类器变体
# ============================================================
ABLATION_STAGE2 = {
    f's2_{name}': {
        'mode': 'two_stage',
        'stage1_backbone': 'resnet',
        'stage2_backbone': name,
        'partition': 'dirichlet',
        'alpha': 0.5,
    }
    for name in ['mamba', 'mlp', 'cnn', 'lstm', 'transformer']
}

# ============================================================
# 消融实验 4: Non-IID 分区策略
# ============================================================
ABLATION_PARTITION = {
    'partition_iid': {
        'mode': 'two_stage',
        'stage1_backbone': 'resnet',
        'stage2_backbone': 'mamba',
        'partition': 'iid',
    },
    'partition_dirichlet_0.1': {
        'mode': 'two_stage',
        'stage1_backbone': 'resnet',
        'stage2_backbone': 'mamba',
        'partition': 'dirichlet',
        'alpha': 0.1,
    },
    'partition_dirichlet_0.5': {
        'mode': 'two_stage',
        'stage1_backbone': 'resnet',
        'stage2_backbone': 'mamba',
        'partition': 'dirichlet',
        'alpha': 0.5,
    },
    'partition_dirichlet_1.0': {
        'mode': 'two_stage',
        'stage1_backbone': 'resnet',
        'stage2_backbone': 'mamba',
        'partition': 'dirichlet',
        'alpha': 1.0,
    },
}

# ============================================================
# 全部消融实验
# ============================================================
ALL_ABLATIONS = {
    **ABLATION_STAGE,
    **ABLATION_STAGE1,
    **ABLATION_STAGE2,
    **ABLATION_PARTITION,
}
