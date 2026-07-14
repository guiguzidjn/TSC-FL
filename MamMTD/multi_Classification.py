import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SimplifiedMambaResNet(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, dropout_rate=0.3):
        super().__init__()

        # === 将每个特征映射到 d_model 维度（特征当序列）===
        # [B, F] -> [B, F, d_model]，其中 F 是序列长度（原始特征数）
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model),  # 每个特征值映射到 d_model 维
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # === Mamba 分支（在特征序列上处理）===
        self.mamba = nn.Sequential(
            Mamba(d_model=d_model, d_state=128, d_conv=4, expand=2),
            nn.LayerNorm(d_model)
        )

        # === 分类头 ===
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # 输入: [B, F] 其中 F 是特征数（38/78/204）
        batch_size, num_features = x.shape

        # === 步骤1: 特征嵌入 [B, F] -> [B, F, d_model] ===
        x = x.unsqueeze(-1)  # [B, F, 1]
        x = self.feature_embedding(x)  # [B, F, d_model]

        # === 步骤2: 并行分支处理 ===
        # Mamba 分支：期望输入 [B, L, d_model]
        m = self.mamba(x)  # [B, F, d_model]

        # === 步骤3: 分类 ===
        m_pooled = m.mean(dim=1)
        return self.classifier(m_pooled)