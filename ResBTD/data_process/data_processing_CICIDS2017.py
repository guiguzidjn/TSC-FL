import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess_data_CICIDS(file_path, args):
    """
    数据预处理模块：读取数据，处理缺失值，特征编码，归一化，并划分训练集和测试集后保存。
    :param file_path: 数据集的文件路径
    :param args: 其他相关参数（例如训练集和测试集的比例等）
    :return: 训练集、测试集和对应的标签，以及攻击样本数据集
    """
    # 假设列名为以下内容
    columns = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets',
        ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
        ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
        'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total',
        ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total',' Bwd IAT Mean', ' Bwd IAT Std',
        ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
        ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',' Packet Length Mean',
        ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
        ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', 'ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size',
        ' Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
        ' Bwd Avg Packets/Bulk', ' Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
        'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
        ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min', 'Label'
    ]

    # 获取目录中的所有 CSV 文件
    csv_files = glob.glob(os.path.join(file_path, '*.csv'))
    # csv_files = glob.glob(os.path.join("file_path", '*.csv'))

    df = pd.DataFrame()  # 存储所有数据的列表
    for file_paths in csv_files:
        print(f"Processing file: {file_paths}")
        # 读取数据集
        df1 = pd.read_csv(file_paths, header=None, names=columns, low_memory=False)

        df1 = df1.loc[1:]  # 删除第一列（索引列）

        # 删除第一行（通常是无效数据或重复索引）
        df1 = df1.loc[1:]

        # **1 统一替换 'Infinity' 和 'NaN' 为 np.nan**
        df1.replace(["Infinity", "inf", "-inf", "NaN", "nan"], np.nan, inplace=True)

        # **2 将所有列转换为数值（errors='coerce' 会把无法转换的数据变成 NaN）**
        df1.iloc[:, :-1] = df1.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

        # **3 设置 float64 可接受的最大值**
        max_value = 1.0e+308  # float64 上限

        # **4 找到超出范围的行**
        mask_out_of_range = (df1.iloc[:, :-1] > max_value) | (df1.iloc[:, :-1] < -max_value)

        # **5 删除超出 float64 范围的行**
        df1 = df1[~mask_out_of_range.any(axis=1)]

        # **6 删除包含 NaN 的行**
        df1 = df1.dropna().reset_index(drop=True)

        # **7  合并当前文件的数据**
        df = pd.concat([df, df1], ignore_index=True)

    # 数据清理：删除无用列
    # df = df.drop(columns=["srcip", "dstip", 'sport', "dsport"])
    # df = df.iloc[1:]

    # df = df.fillna('0')  # 填充缺失值为 "0"

    # 先确保所有数据都转换为字符串
    df['Label'] = df['Label'].astype(str)

    # 创建编码映射
    unique_labels = df['Label'].unique()
    label_mapping = {'BENIGN': 0}
    counter = 1

    for label in unique_labels:
        if label != 'BENIGN':
            label_mapping[label] = counter
            counter += 1

    # 应用映射
    df['attack_cat_encoded'] = df['Label'].map(label_mapping)

    # 二值化编码：BENIGN -> 0，其他类别 -> 1
    df['binary_Label'] = df['Label'].map(lambda x: 0 if x == 'BENIGN' else 1)

    print(df['binary_Label'].value_counts())  # 统计各类别数量，检查是否正确

    # # 自定义对攻击类别的编码
    # benign_label = '0'  # 假设良性攻击类别为 "Normal"，根据实际数据修改
    # attack_mapping = {benign_label: 0}  # 良性攻击映射为 0
    #
    # # 获取所有恶性类别，并从 1 开始编码
    # attack_categories = df['Label'].unique()
    # attack_labels = [label for label in attack_categories if label != benign_label]
    # attack_mapping.update({label: idx + 1 for idx, label in enumerate(attack_labels)})
    #
    # # 应用映射到 attack_cat 列
    # df['attack_cat_encoded'] = df['Label'].map(attack_mapping)

    # 打印攻击类别映射
    # print("攻击类别映射:", attack_mapping)

    Binary_features = [
        'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
        ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', 'ECE Flag Count'
    ]

    # 数值特征
    numerical_features = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
        'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
        ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
        ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
        'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
        ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
        ' Min Packet Length', ' Max Packet Length',' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
        ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
        ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', ' Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
        ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
        ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
    ]

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
        attack_df = df[df['binary_Label'] == 1]
        attack_y = attack_df['attack_cat_encoded']  - 1
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

