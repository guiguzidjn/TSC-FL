import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data_NbaIoT(file_path, args):
    # 获取目录下所有以 .csv 结尾的文件名
    csv_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]

    # 初始化一个空的列表来存储每个文件的内容
    dataframes = []
    labels = []

    # 遍历所有文件
    for file in csv_files:
        # 提取文件名中的类别，去掉开头的数字和点，正确匹配流量类别
        filename = os.path.basename(file)
        label = filename.split('.', 1)[1].rsplit('.', 1)[0]  # 去掉数字前缀和.csv后缀

        print(f"正在处理文件: {file}, 流量类别: {label}")

        # 读取 CSV 文件
        df1 = pd.read_csv(file)

        # 在最后插入一列 "Label"，值为当前类别
        df1['Label'] = label

        # 保存到列表中
        dataframes.append(df1)
        labels.append(label)

    # 合并所有数据到一个 DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    # 先确保所有数据都转换为字符串
    df['Label'] = df['Label'].astype(str)

    # 创建编码映射
    unique_labels = df['Label'].unique()
    label_mapping = {'BENIGN': 0}
    label_mapping = {'benign': 0}

    counter = 1

    for label in unique_labels:
        if label != 'BENIGN' and label != 'benign':
            label_mapping[label] = counter
            counter += 1

    # 应用映射
    df['attack_cat_encoded'] = df['Label'].map(label_mapping)

    numerical_features = [
        "MI_dir_L5_weight",	"MI_dir_L5_mean", "MI_dir_L5_variance", "MI_dir_L3_weight",	"MI_dir_L3_mean", "MI_dir_L3_variance", "MI_dir_L1_weight",
        "MI_dir_L1_mean", "MI_dir_L1_variance", "MI_dir_L0.1_weight", "MI_dir_L0.1_mean", "MI_dir_L0.1_variance", "MI_dir_L0.01_weight", "MI_dir_L0.01_mean",
        "MI_dir_L0.01_variance", "H_L5_weight", "H_L5_mean", "H_L5_variance", "H_L3_weight", "H_L3_mean",	"H_L3_variance",	"H_L1_weight",
        "H_L1_mean",	"H_L1_variance",	"H_L0.1_weight",	"H_L0.1_mean",	"H_L0.1_variance",	"H_L0.01_weight",	"H_L0.01_mean",
        "H_L0.01_variance",	"HH_L5_weight",	"HH_L5_mean",	"HH_L5_std", 	"HH_L5_magnitude",	"HH_L5_radius",	"HH_L5_covariance",	"HH_L5_pcc",
        "HH_L3_weight",	"HH_L3_mean",	"HH_L3_std",	"HH_L3_magnitude",	"HH_L3_radius",	"HH_L3_covariance",	"HH_L3_pcc",	"HH_L1_weight",
        "HH_L1_mean",	"HH_L1_std",	"HH_L1_magnitude",	"HH_L1_radius",	"HH_L1_covariance",	"HH_L1_pcc",	"HH_L0.1_weight",	"HH_L0.1_mean",
        "HH_L0.1_std",	"HH_L0.1_magnitude",	"HH_L0.1_radius",	"HH_L0.1_covariance",	"HH_L0.1_pcc",	"HH_L0.01_weight",	"HH_L0.01_mean",
        "HH_L0.01_std",	"HH_L0.01_magnitude",	"HH_L0.01_radius",	"HH_L0.01_covariance",	"HH_L0.01_pcc",	"HH_jit_L5_weight",	"HH_jit_L5_mean",
        "HH_jit_L5_variance",	"HH_jit_L3_weight",	"HH_jit_L3_mean",	"HH_jit_L3_variance",	"HH_jit_L1_weight",	"HH_jit_L1_mean",	"HH_jit_L1_variance",
        "HH_jit_L0.1_weight",	"HH_jit_L0.1_mean",	"HH_jit_L0.1_variance",	"HH_jit_L0.01_weight",	"HH_jit_L0.01_mean",	"HH_jit_L0.01_variance",
        "HpHp_L5_weight",	"HpHp_L5_mean",	"HpHp_L5_std",	"HpHp_L5_magnitude",	"HpHp_L5_radius",	"HpHp_L5_covariance",	"HpHp_L5_pcc",	"HpHp_L3_weight",
        "HpHp_L3_mean",	"HpHp_L3_std",	"HpHp_L3_magnitude",	"HpHp_L3_radius",	"HpHp_L3_covariance",	"HpHp_L3_pcc",	"HpHp_L1_weight",	"HpHp_L1_mean",
        "HpHp_L1_std",	"HpHp_L1_magnitude",	"HpHp_L1_radius",	"HpHp_L1_covariance",	"HpHp_L1_pcc",	"HpHp_L0.1_weight",	"HpHp_L0.1_mean",	"HpHp_L0.1_std",
        "HpHp_L0.1_magnitude",	"HpHp_L0.1_radius",	"HpHp_L0.1_covariance",	"HpHp_L0.1_pcc",	"HpHp_L0.01_weight",	"HpHp_L0.01_mean",	"HpHp_L0.01_std",
        "HpHp_L0.01_magnitude",	"HpHp_L0.01_radius",	"HpHp_L0.01_covariance",	"HpHp_L0.01_pcc"

    ]
    # 二值化编码：BENIGN -> 0，其他类别 -> 1
    df['binary_Label'] = df['Label'].map(lambda x: 0 if x == 'benign' else 1)

    print(df['binary_Label'].value_counts())  # 统计各类别数量，检查是否正确

    # 先将所有数值特征转换为数值型（例如，字符串值转化为数字）
    df[numerical_features] = df[numerical_features].apply(pd.to_numeric, errors='coerce')  # 转换为数值，无法转换的变为NaN

    # 填充缺失的数值
    df[numerical_features] = df[numerical_features].fillna(0)

    # 对数值特征进行 Min-Max 归一化
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # 二分类任务
    if args.task == 'binary':
        # 提取攻击数据集
        attack_df = df[df['attack_cat_encoded'] != 0]
        # attack_cat = attack_df['attack_cat_encoded']
        attack_X = attack_df.drop(columns=['Label', 'binary_Label', 'attack_cat_encoded'])  # 特征
        attack_y = attack_df['binary_Label']  # 标签

        # 提取良性数据集
        benign_df = df[df['binary_Label'] == 0]
        benign_cat = benign_df['attack_cat_encoded']
        benign_X = benign_df.drop(columns=['Label', 'binary_Label', 'attack_cat_encoded'])
        benign_y = benign_df['binary_Label']

        # 划分训练集和测试集
        X = df.drop(columns=['Label', 'binary_Label', 'attack_cat_encoded'])  # 特征
        y = df['binary_Label']  # 标签
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 多分类任务
    if args.task == 'multi':
        # 提取攻击数据集
        attack_df = df[df['attack_cat_encoded'] != 0]
        attack_y = attack_df['attack_cat_encoded'] - 1
        attack_X = attack_df.drop(columns=['Label', 'binary_Label', 'attack_cat_encoded'])  # 特征

        # 提取良性数据集
        benign_df = df[df['binary_Label'] == 0]
        benign_cat = benign_df['attack_cat_encoded']
        benign_X = benign_df.drop(columns=['Label', 'binary_Label', 'attack_cat_encoded'])
        benign_y = benign_df['binary_Label']

        # 划分训练集和测试集
        y = df['attack_cat_encoded']
        X = df.drop(columns=['Label', 'binary_Label', 'attack_cat_encoded'])  # 特征
        X_train, X_test, y_train, y_test = train_test_split(attack_X, attack_y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test
