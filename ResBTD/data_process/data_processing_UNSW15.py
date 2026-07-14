import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import glob
import os
import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess_data_UNSW15(file_path, args):
    """
    数据预处理模块：读取数据，处理缺失值，特征编码，归一化，并划分训练集和测试集后保存。
    :param file_path: 数据集的文件路径
    :param args: 其他相关参数（例如训练集和测试集的比例等）
    :return: 训练集、测试集和对应的标签，以及攻击样本数据集
    """
    # 假设列名为以下内容
    columns = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
        'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
        'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
        'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
    ]

    # 获取目录中的所有 CSV 文件
    csv_files = glob.glob(os.path.join(file_path, '*.csv'))

    df = pd.DataFrame()  # 存储所有数据的列表
    for file_paths in csv_files:
        print(f"Processing file: {file_paths}")
        # 读取数据集
        df1 = pd.read_csv(file_paths, header=None, names=columns, low_memory=False)
        # 将当前文件的数据添加到总数据列表中
        df = pd.concat([df, df1], ignore_index=True)

    # 数据清理：删除无用列
    df = df.drop(columns=["srcip", "dstip", 'sport', "dsport"])

    str_feature = ['proto', 'state', 'service']
    str_feature_space = ['proto', 'state', 'service', "attack_cat"]
    for col in str_feature_space:
        if col in df.columns:
            # 去除首尾空格
            df[col] = df[col].str.strip()
    # 替换 `-` 和空值
    df = df.replace('-', 'unknown')  # 替换 "-" 为 'unknown'
    df = df.fillna('0')  # 填充缺失值为 "0"

    df['attack_cat'] = df['attack_cat'].replace(
        {
            'backdoors': 'Backdoor',
            'Backdoors': 'Backdoor',
        },
        regex=False
    )
    # df = df[df['attack_cat'] != 'Worms']
    # 自定义对攻击类别的编码
    benign_label = '0'  # 假设良性攻击类别为 "Normal"，根据实际数据修改
    attack_mapping = {benign_label: 0}  # 良性攻击映射为 0

    # 获取所有恶性类别，并从 1 开始编码
    attack_categories = df['attack_cat'].unique()
    attack_labels = [label for label in attack_categories if label != benign_label]
    attack_mapping.update({label: idx + 1 for idx, label in enumerate(attack_labels)})

    # 应用映射到 attack_cat 列
    df['attack_cat_encoded'] = df['attack_cat'].map(attack_mapping)

    # 打印攻击类别映射
    print("攻击类别映射:", attack_mapping)

    # 对分类特征进行 One-Hot 编码
    df = pd.get_dummies(df, columns=str_feature)

    # 数值特征
    numerical_features = [
        'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
        'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
        'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt',
        'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
        'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', "Stime", "Ltime"
    ]

    # 先将所有数值特征转换为数值型（例如，字符串值转化为数字）
    df[numerical_features] = df[numerical_features].apply(pd.to_numeric, errors='coerce')  # 转换为数值，无法转换的变为NaN

    # 填充缺失的数值
    df[numerical_features] = df[numerical_features].fillna(0)

    # 对数值特征进行 Min-Max 归一化
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    print(df['attack_cat'].value_counts())

    # 二分类任务
    if args.task == 'binary':
        # 提取攻击数据集
        attack_df = df[df['Label'] == 1]
        attack_cat = attack_df['attack_cat_encoded']
        attack_X = attack_df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])  # 特征
        attack_y = attack_df['Label']  # 标签

        # 提取良性数据集
        benign_df = df[df['Label'] == 0]
        benign_cat = benign_df['attack_cat_encoded']
        benign_X = benign_df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])
        benign_y = benign_df['Label']

        # 划分训练集和测试集
        X = df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])  # 特征
        y = df['Label']  # 标签
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 多分类任务
    if args.task == 'multi':
        # 提取攻击数据集
        attack_df = df[df['Label'] == 1]
        attack_y = attack_df['attack_cat_encoded'] - 1
        attack_X = attack_df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])  # 特征

        X_train, X_test, y_train, y_test = train_test_split(attack_X, attack_y, test_size=0.2, random_state=42)

        # # 提取良性数据集
        # benign_df = df[df['Label'] == 0]
        # benign_cat = benign_df['attack_cat_encoded']
        # benign_X = benign_df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])
        # benign_y = benign_df['Label']
        #
        # # 划分训练集和测试集
        # y = df['attack_cat_encoded']
        # X = df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])  # 特征
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(attack_y.unique())

    return X_train, X_test, y_train, y_test

# def preprocess_data_UNSW15(file_path, args, replacement_ratio=0.4):
#     """
#     数据预处理模块：读取数据，处理缺失值，特征编码，归一化，划分训练集和测试集，
#     然后将一部分训练数据替换到测试数据中。
#
#     :param file_path: 数据集的文件路径
#     :param args: 其他相关参数
#     :param replacement_ratio: 要替换的测试数据比例
#     :return: 训练集、测试集和对应的标签
#     """
#     # 假设列名为以下内容
#     columns = [
#         'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
#         'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
#         'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
#         'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
#         'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
#         'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
#         'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
#     ]
#
#     # 获取目录中的所有 CSV 文件
#     csv_files = glob.glob(os.path.join(file_path, '*.csv'))
#
#     df = pd.DataFrame()  # 存储所有数据的列表
#     for file_paths in csv_files:
#         print(f"Processing file: {file_paths}")
#         # 读取数据集
#         df1 = pd.read_csv(file_paths, header=None, names=columns, low_memory=False)
#         # 将当前文件的数据添加到总数据列表中
#         df = pd.concat([df, df1], ignore_index=True)
#
#     # 数据清理：删除无用列
#     df = df.drop(columns=["srcip", "dstip", 'sport', "dsport"])
#
#     str_feature = ['proto', 'state', 'service']
#     str_feature_space = ['proto', 'state', 'service', "attack_cat"]
#     for col in str_feature_space:
#         if col in df.columns:
#             # 去除首尾空格
#             df[col] = df[col].str.strip()
#     # 替换 `-` 和空值
#     df = df.replace('-', 'unknown')  # 替换 "-" 为 'unknown'
#     df = df.fillna('0')  # 填充缺失值为 "0"
#
#     df['attack_cat'] = df['attack_cat'].replace(
#         {
#             'backdoors': 'Backdoor',
#             'Backdoors': 'Backdoor',
#         },
#         regex=False
#     )
#     # df = df[df['attack_cat'] != 'Worms']
#     # 自定义对攻击类别的编码
#     benign_label = '0'  # 假设良性攻击类别为 "Normal"，根据实际数据修改
#     attack_mapping = {benign_label: 0}  # 良性攻击映射为 0
#
#     # 获取所有恶性类别，并从 1 开始编码
#     attack_categories = df['attack_cat'].unique()
#     attack_labels = [label for label in attack_categories if label != benign_label]
#     attack_mapping.update({label: idx + 1 for idx, label in enumerate(attack_labels)})
#
#     # 应用映射到 attack_cat 列
#     df['attack_cat_encoded'] = df['attack_cat'].map(attack_mapping)
#
#     # 打印攻击类别映射
#     print("攻击类别映射:", attack_mapping)
#
#     # 对分类特征进行 One-Hot 编码
#     df = pd.get_dummies(df, columns=str_feature)
#
#     # 数值特征
#     numerical_features = [
#         'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
#         'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
#         'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt',
#         'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
#         'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
#         'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', "Stime", "Ltime"
#     ]
#
#     # 先将所有数值特征转换为数值型（例如，字符串值转化为数字）
#     df[numerical_features] = df[numerical_features].apply(pd.to_numeric, errors='coerce')  # 转换为数值，无法转换的变为NaN
#
#     # 填充缺失的数值
#     df[numerical_features] = df[numerical_features].fillna(0)
#
#     # 对数值特征进行 Min-Max 归一化
#     scaler = MinMaxScaler()
#     df[numerical_features] = scaler.fit_transform(df[numerical_features])
#
#     print(df['attack_cat'].value_counts())
#
#     # 多分类任务
#     if args.task == 'multi':
#         # 提取攻击数据集
#         attack_df = df[df['Label'] == 1]
#         attack_y = attack_df['attack_cat_encoded'] - 1
#         attack_X = attack_df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])  # 特征
#
#         # 先将数据划分为训练集和测试集
#         X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
#             attack_X, attack_y, test_size=0.2, random_state=42
#         )
#
#         # 计算要替换的测试数据数量
#         num_replace = int(len(X_test_orig) * replacement_ratio)
#
#         # 从训练集中随机选择一部分数据
#         # 使用不同的随机种子，确保选择的是不同的数据
#         replacement_indices = np.random.RandomState(100).choice(
#             len(X_train_orig), num_replace, replace=False
#         )
#
#         # 提取要替换的训练数据
#         X_replacement = X_train_orig.iloc[replacement_indices].copy()
#         y_replacement = y_train_orig.iloc[replacement_indices].copy()
#
#         # 将这部分数据从训练集中移除
#         X_train = X_train_orig.drop(X_train_orig.index[replacement_indices]).reset_index(drop=True)
#         y_train = y_train_orig.drop(y_train_orig.index[replacement_indices]).reset_index(drop=True)
#
#         # 替换一部分测试数据
#         test_indices_to_replace = np.random.RandomState(101).choice(
#             len(X_test_orig), num_replace, replace=False
#         )
#
#         # 创建测试集的副本
#         X_test = X_test_orig.copy()
#         y_test = y_test_orig.copy()
#
#         # 替换选定的测试数据点
#         for i, idx in enumerate(test_indices_to_replace):
#             if i < len(X_replacement):
#                 X_test.iloc[idx] = X_replacement.iloc[i]
#                 y_test.iloc[idx] = y_replacement.iloc[i]
#
#         print(f"已替换 {num_replace} 个测试数据点 (占测试集的 {replacement_ratio * 100:.1f}%)")
#         print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
#
#         print("测试集中攻击类别分布:")
#         print(y_test.value_counts())
#
#     elif args.task == 'binary':
#         # 二分类任务的数据处理 (保持原有逻辑)
#         attack_df = df[df['Label'] == 1]
#         attack_cat = attack_df['attack_cat_encoded']
#         attack_X = attack_df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])  # 特征
#         attack_y = attack_df['Label']  # 标签
#
#         # 提取良性数据集
#         benign_df = df[df['Label'] == 0]
#         benign_cat = benign_df['attack_cat_encoded']
#         benign_X = benign_df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])
#         benign_y = benign_df['Label']
#
#         # 划分训练集和测试集
#         X = df.drop(columns=['Label', 'attack_cat', 'attack_cat_encoded'])  # 特征
#         y = df['Label']  # 标签
#
#         # 先进行正常的训练测试集划分
#         X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
#
#         # 与多分类任务类似，进行测试集替换
#         num_replace = int(len(X_test_orig) * replacement_ratio)
#
#         replacement_indices = np.random.RandomState(100).choice(
#             len(X_train_orig), num_replace, replace=False
#         )
#
#         X_replacement = X_train_orig.iloc[replacement_indices].copy()
#         y_replacement = y_train_orig.iloc[replacement_indices].copy()
#
#         X_train = X_train_orig.drop(X_train_orig.index[replacement_indices]).reset_index(drop=True)
#         y_train = y_train_orig.drop(y_train_orig.index[replacement_indices]).reset_index(drop=True)
#
#         test_indices_to_replace = np.random.RandomState(101).choice(
#             len(X_test_orig), num_replace, replace=False
#         )
#
#         X_test = X_test_orig.copy()
#         y_test = y_test_orig.copy()
#
#         for i, idx in enumerate(test_indices_to_replace):
#             if i < len(X_replacement):
#                 X_test.iloc[idx] = X_replacement.iloc[i]
#                 y_test.iloc[idx] = y_replacement.iloc[i]
#
#         print(f"已替换 {num_replace} 个测试数据点 (占测试集的 {replacement_ratio * 100:.1f}%)")
#         print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
#
#     return X_train, X_test, y_train, y_test
