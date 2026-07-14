# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
"""
CICIDS2017 多分类联邦学习训练脚本 (Mamba + FedAvg)
数据集: CICIDS2017 (78 features, 14 classes)
模型: SimplifiedMambaResNet
"""

import sys
import io
import argparse

# Force UTF-8 output in Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


from MamMTD.train_Multi import *
from MamMTD.data_process.data_processing_CICIDS2017 import preprocess_data_CICIDS


def parse_args():
    parser = argparse.ArgumentParser(description="CICIDS2017 多分类联邦学习训练脚本 (Mamba + FedAvg)")

    # 网络参数
    parser.add_argument('--batch_size', type=int, default=256, help='训练时的批量大小')
    parser.add_argument('--input_dim', type=int, default=78, help='网络输入的维度 (CICIDS2017: 78 features, 14 classes)')
    parser.add_argument('--client_num', type=int, default=5, help='客户端数量')
    parser.add_argument('--communication_round', type=int, default=50, help='通信轮数')
    parser.add_argument('--task', type=str, default='multi', help='流量二分类任务和多分类任务')
    parser.add_argument('--dataset', type=str, default='CICIDS2017', help='CICIDS2017 OR UNSW-NB15 OR Nba-IoT')
    parser.add_argument('--cuda', type=bool, default=True, help='是否使用GPU训练')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    X_train, X_test, y_train, y_test, attack_encode = preprocess_data_CICIDS(
        "D:\\Chenhaolei_experiment\\dataset\\CICIDS2017\\MachineLearningCVE", args
    )

    model = Multi_classification(args, X_train, X_test, y_train, y_test, attack_encode)
