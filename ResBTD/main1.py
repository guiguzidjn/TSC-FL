import sys
import io
import argparse

# Force UTF-8 output in Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from train_Binary1 import BinaryClassification


from data_process.data_processing_UNSW15 import preprocess_data_UNSW15
from data_process.data_processing_CICIDS2017 import preprocess_data_CICIDS
from data_process.data_processing_NbaIoT import preprocess_data_NbaIoT
def parse_args():
    parser = argparse.ArgumentParser(description="设置训练参数")

    # 网络参数
    # parser.add_argument('--batch_size', type=int, default=256, help='训练时的批量大小')
    parser.add_argument('--batch_size', type=int, default=1024, help='训练时的批量大小')
    parser.add_argument('--input_dim', type=int, default=204, help='网络输入的维度 UNSW-NB15 204 CICIDS2017 78 Nba-IoT 115')
    # parser.add_argument('--input_dim', type=int, default=204, help='网络输入的维度 UNSW-15 204 CICIDS2017 78 Nba-IoT 115')
    parser.add_argument('--client_num', type=int, default=10, help='客户端数量')
    # parser.add_argument('--communication_round', type=int, default=100, help='通讯轮数')
    parser.add_argument('--communication_round', type=int, default=200, help='二分类通讯轮数')

    parser.add_argument('--task', type=str, default='binary', help='流量二分类任务和多分类任务')

    parser.add_argument('--dataset', type=str, default='UNSW-NB15', help='CICIDS2017  OR UNSW-NB15 OR Nba-IoT')
    # parser.add_argument('--dataset', type=str, default='Nba-IoT', help='CICIDS2017  OR UNSW-NB15 OR Nba-IoT')
    # parser.add_argument('--save_interval', type=float, default=5, help='保存模型的间隔（每5轮保存一次）')
    # parser.add_argument('--save_interval', type=float, default=5, help='保存模型的间隔（每5轮保存一次）')

    # 文件路径
    parser.add_argument('--train_data', type=str, default='./train_data.csv', help='训练数据集路径')
    parser.add_argument('--test_data', type=str, default='./test_data.csv', help='测试数据集路径')

    # 其他参数
    parser.add_argument('--cuda', type=bool, default=True, help='是否使用GPU训练')

    # 解析命令行参数
    return parser.parse_args()


# 如果你直接运行args.py，可以直接输出参数
if __name__ == '__main__':
    args = parse_args()

    if args.task == 'binary':
        if args.dataset == 'UNSW-NB15':
            X_train, X_test, y_train, y_test = preprocess_data_UNSW15("D:\\Chenhaolei_experiment\\dataset\\UNSW-NB15\\CSV Files", args)
        elif args.dataset == 'CICIDS2017':
            X_train, X_test, y_train, y_test = preprocess_data_CICIDS("D:\\Chenhaolei_experiment\\dataset\\CICIDS2017\\MachineLearningCVE", args)
        elif args.dataset == 'Nba-IoT':
            X_train, X_test, y_train, y_test = preprocess_data_NbaIoT("D:\\Chenhaolei_experiment\\dataset\\N_BaIOT\\archive", args)
        else:
            print("数据集不存在 ")
        model = BinaryClassification(args, X_train, y_train, X_test, y_test)
    else:
        print("任务不存在 ")

