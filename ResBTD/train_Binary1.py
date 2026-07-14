import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score
from ResBTD.data_process.data_processing_UNSW15 import *
from sklearn.utils import resample
import numpy as np

import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch.nn.functional as F
from sklearn.svm import SVC
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
import wandb
from imblearn.over_sampling import SVMSMOTE
import time
import psutil
from ptflops import get_model_complexity_info
from torchprofile import profile_macs
import os

print(torch.cuda.is_available())


def count_parameters(model):
    """Placeholder."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def measure_inference_time(model, sample_input, device, num_runs=100, warmup_runs=10):
    """Placeholder."""
    model.eval()
    sample_input = sample_input.to(device)

    # Placeholder.
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)

    # Placeholder.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Placeholder.
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    avg_inference_time = (end_time - start_time) / num_runs * 1000  # 杞 崲涓烘 绉?
    return avg_inference_time


def calculate_flops(model, input_shape, device):
    """Placeholder."""
    try:
        # Placeholder.
        flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=False, verbose=False)
        return flops, params
    except:
        try:
            # Placeholder.
            dummy_input = torch.randn(1, *input_shape).to(device)
            macs = profile_macs(model, dummy_input)
            flops = 2 * macs  # MACs to FLOPs approximation
            return f"{flops / 1e9:.2f}G", count_parameters(model)
        except:
            return "N/A", count_parameters(model)


def dirichlet_distribution(y_train, client_num, alpha=0.5):
    """Placeholder."""
    # Placeholder.

    # Args:
# y_train: 璁 粌镙囩
# client_num: 瀹 埛绔 暟阅?
# alpha: 杩 埄鍏嬮浄 嗗竷鍙傛暟锛岃秺灏忔暟鎹 秺涓嶅潎鍖

    # Returns:
# client_indices: 姣忎釜瀹 埛绔 殑鏁版嵁绱 紩 楄
    """Placeholder."""
    num_classes = len(np.unique(y_train))
    client_indices = [[] for _ in range(client_num)]

    # Placeholder.
    for class_id in range(num_classes):
        class_indices = np.where(y_train == class_id)[0]
        np.random.shuffle(class_indices)

        # Placeholder.
        proportions = np.random.dirichlet(np.repeat(alpha, client_num))

        # Placeholder.
        class_distributions = [int(p * len(class_indices)) for p in proportions]

        # Placeholder.
        remainder = len(class_indices) - sum(class_distributions)
        for i in range(remainder):
            class_distributions[i % client_num] += 1

        # Placeholder.
        start_idx = 0
        for client_id in range(client_num):
            end_idx = start_idx + class_distributions[client_id]
            client_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx

    # Placeholder.
    for client_id in range(client_num):
        np.random.shuffle(client_indices[client_id])

    return client_indices


def distribute_data_dirichlet(X_train, y_train, client_num, alpha=0.5, save_data=False):
    """Placeholder."""
    # Placeholder.
    """Placeholder."""
    client_data = []

    # Placeholder.
    X_train_np = X_train.values.astype(np.float32)
    y_train_np = y_train.values.astype(np.int64)

    # Placeholder.
    client_indices = dirichlet_distribution(y_train_np, client_num, alpha)

    # Placeholder.
    for client_id in range(client_num):
        indices = client_indices[client_id]
        client_X = torch.tensor(X_train_np[indices], dtype=torch.float32)
        client_y = torch.tensor(y_train_np[indices], dtype=torch.long)

        client_data.append((client_X, client_y))

        # Placeholder.
        unique, counts = np.unique(y_train_np[indices], return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"Client {client_id}: {len(indices)} samples, distribution: {distribution}")

        # Placeholder.
        if save_data:
            client_dir = f"client_data/client_{client_id}"
            os.makedirs(client_dir, exist_ok=True)
            np.save(os.path.join(client_dir, 'X.npy'), client_X.numpy())
            np.save(os.path.join(client_dir, 'y.npy'), client_y.numpy())

    return client_data


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_and_save_confusion_matrix(conf_matrix, class_names=None, save_path='confusion_matrix', normalize=False):
    """Placeholder."""
    # Placeholder.

# 鍙傛暟:
# conf_matrix: 娣锋穯鐭 樀鏁扮粍
# class_names: 绫诲埆钖岖  楄
# save_path: 淇 瓨璺 缎锛堜笉钖 墿灞曞悕锛?
# normalize: 鏄 惁褰掍竴鍖栨樉绀?
    """Placeholder."""
    plt.figure(figsize=(10, 8))

    # Placeholder.
    if normalize:
        conf_matrix_display = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        conf_matrix_display = conf_matrix
        fmt = 'd'
        title = 'Confusion Matrix'

    # Placeholder.
    sns.heatmap(conf_matrix_display,
                annot=True,
                fmt=fmt,
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Placeholder.
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_path}.svg', bbox_inches='tight')

    plt.close()

    print(f"娣锋穯鐭 樀宸蹭缭瀛树负:")
    print(f"  - {save_path}.png")
    print(f"  - {save_path}.pdf")
    print(f"  - {save_path}.svg")

def evaluate_model_with_metrics(global_model, X_test, y_test, device, batch_size, input_shape, save_confusion_matrix = True, output_dir='./'):
    """Placeholder."""
    # Placeholder.
    """Placeholder."""
    global_model.eval()

    # Placeholder.
    X_test_np = X_test.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.int64)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.long).to(device)

    # Placeholder.
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []

    # Placeholder.
    param_count = count_parameters(global_model)
    model_size = get_model_size(global_model)

    # Placeholder.
    flops, _ = calculate_flops(global_model, input_shape, device)

    # Placeholder.
    sample_input = torch.randn(batch_size, *input_shape)
    inference_time = measure_inference_time(global_model, sample_input, device)

    # Placeholder.
    with torch.no_grad():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = global_model(batch_data)
            probs = torch.softmax(outputs, dim=1)
            pred_labels = torch.argmax(probs, dim=1)

            all_predictions.extend(pred_labels.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Placeholder.
    TP, FP, TN, FN = compute_pos_neg(all_labels, all_predictions)
    TPR, FPR, F1, precision, recall, acc = metrics(TP, FP, TN, FN)

    # Placeholder.
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Placeholder.
    if save_confusion_matrix:
        # Placeholder.
        os.makedirs(output_dir, exist_ok=True)

        # Placeholder.
        unique_classes = sorted(list(set(all_labels)))
        class_names = [str(c) for c in unique_classes]

        # Placeholder.
        plot_and_save_confusion_matrix(
            conf_matrix,
            class_names=class_names,
            save_path=os.path.join(output_dir, 'confusion_matrix'),
            normalize=False
        )

        # Placeholder.
        plot_and_save_confusion_matrix(
            conf_matrix,
            class_names=class_names,
            save_path=os.path.join(output_dir, 'confusion_matrix_normalized'),
            normalize=True
        )

    # Placeholder.
    model_metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': F1,
        'parameters': param_count,
        'model_size_mb': model_size,
        'inference_time_ms': inference_time,
        'flops': flops
    }

    return model_metrics


def compute_pos_neg(y_actual, y_hat):
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1: TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]: FP += 1
        if y_actual[i] == y_hat[i] == 0: TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]: FN += 1
    return TP, FP, TN, FN


def metrics(TP, FP, TN, FN):
    acc = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * TP) / float(2 * TP + FP + FN + 1e-9)
    precision = TP / float(TP + FP + 1e-9)
    recall = TP / float(TP + FN + 1e-9)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR, F1, precision, recall, acc



def get_model_transfer_size(model):
    """Calculate model transfer size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 ** 2)

def BinaryClassification(args, X_train, y_train, X_test, y_test):
    """Placeholder."""
    # Placeholder.
    """Placeholder."""
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_dir = os.path.join("save_model", args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    client_num = args.client_num

    assert isinstance(X_train, pd.DataFrame), "X_train must be a pandas DataFrame"
    assert isinstance(y_train, pd.Series), "y_train must be a pandas Series"

    print("Distributing data using Dirichlet distribution...")
    client_data = distribute_data_dirichlet(X_train, y_train, client_num=args.client_num,
                                            alpha=0.5, save_data=True)

    # Placeholder.
    client_sample_counts = [len(client_X) for client_X, _ in client_data]
    client_mean_samples = np.mean(client_sample_counts)
    client_std_samples = np.std(client_sample_counts)
    print(f"\n{'='*50}")
    print(f"Client Data Distribution Statistics:")
    print(f"  Samples per client: {client_sample_counts}")
    print(f"  Mean: {client_mean_samples:.1f} +/- {client_std_samples:.1f}")
    print(f"  Min: {min(client_sample_counts)}, Max: {max(client_sample_counts)}")
    print(f"{'='*50}\n")

    # ===== Model transfer size for communication cost =====
    dummy_model = binary_Classification.EnhancedResNetModel(input_dim=args.input_dim, output_dim=2, d_model=128, num_blocks=8, num_heads=8, dropout=0.3)
    model_transfer_mb = get_model_transfer_size(dummy_model)
    del dummy_model
    print(f"Model transfer size (per client upload/download): {model_transfer_mb:.2f} MB")
    total_comm_mb_per_round = 2 * client_num * model_transfer_mb
    print(f"Total communication data per round ({client_num} clients x2): {total_comm_mb_per_round:.2f} MB\n")

    global_model = binary_Classification.EnhancedResNetModel(input_dim=args.input_dim, output_dim=2, d_model=128, num_blocks=8, num_heads=8, dropout=0.3)
    global_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    input_shape = (args.input_dim,)

    log_path = "training_metrics_log_unsw_res.csv"
    write_header = not os.path.exists(log_path)

    # Add communication time column if file is new
    if write_header:
        with open(log_path, "w") as f:
            f.write("Round,Accuracy,Precision,Recall,F1_Score,Communication_Time_Sec,Upload_MB,Download_MB,Total_Comm_MB,Parameters,Model_Size_MB,Inference_Time_MS,FLOPs\n")
        write_header = False

    # Placeholder.
    all_round_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'communication_time': [], 'communication_mb': []}

    print("Starting federated training...")
    for round in range(args.communication_round):
        round_start_time = time.time()
        client_models = []

        for client in range(client_num):
            model = binary_Classification.EnhancedResNetModel(input_dim=args.input_dim, output_dim=2, d_model=128, num_blocks=8, num_heads=8, dropout=0.3)
            model.to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            client_X, client_y = client_data[client]
            client_X = torch.tensor(client_X, dtype=torch.float32) if not isinstance(client_X, torch.Tensor) else client_X
            client_y = torch.tensor(client_y, dtype=torch.long) if not isinstance(client_y, torch.Tensor) else client_y
            train_loader = DataLoader(TensorDataset(client_X, client_y),
                                      batch_size=args.batch_size, shuffle=True)

            client_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

            print(f"Training client {client}")
            for epoch in range(5):
                epoch_loss = 0.0
                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    client_optimizer.zero_grad()

                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    client_optimizer.step()

                    epoch_loss += loss.item()

                print(f"Client {client}, Epoch {epoch}: Loss = {epoch_loss / len(train_loader):.4f}")

            client_models.append(model)

        # Placeholder.
        comm_start_time = time.time()
        global_model.load_state_dict(average_weights(client_models))
        optimizer.step()
        communication_time = time.time() - comm_start_time

        if (round + 1) % 10 == 0:
            model_path = os.path.join(save_dir, f"global_model_round_{round + 1}.pth")
            torch.save(global_model.state_dict(), model_path)
            print(f"Model saved at {model_path}")

        print(f"Evaluating global model at round {round + 1}...")
        model_metrics = evaluate_model_with_metrics(
            global_model, X_test, y_test, device, args.batch_size, input_shape
        )

        # Placeholder.
        all_round_metrics['accuracy'].append(model_metrics['accuracy'])
        all_round_metrics['precision'].append(model_metrics['precision'])
        all_round_metrics['recall'].append(model_metrics['recall'])
        all_round_metrics['f1'].append(model_metrics['f1_score'])
        all_round_metrics['communication_time'].append(communication_time)
        all_round_metrics['communication_mb'].append(total_comm_mb)

        with open(log_path, "a") as f:
            upload_mb = model_transfer_mb * client_num
            download_mb = model_transfer_mb * client_num
            total_comm_mb = upload_mb + download_mb
            log_line = (f"{round + 1},{model_metrics['accuracy']:.4f},"
                        f"{model_metrics['precision']:.4f},{model_metrics['recall']:.4f},"
                        f"{model_metrics['f1_score']:.4f},{communication_time:.4f},"
                        f"{upload_mb:.2f},{download_mb:.2f},{total_comm_mb:.2f},"
                        f"{model_metrics['parameters']},"
                        f"{model_metrics['model_size_mb']:.2f},{model_metrics['inference_time_ms']:.2f},"
                        f"{model_metrics['flops']}\n")
            f.write(log_line)

        round_total_time = time.time() - round_start_time
        print(f"Round {round + 1} Evaluation:")
        print(f"Accuracy: {model_metrics['accuracy']:.4f}")
        print(f"Precision: {model_metrics['precision']:.4f}")
        print(f"Recall: {model_metrics['recall']:.4f}")
        print(f"F1 Score: {model_metrics['f1_score']:.4f}")
        print(f"Communication Time: {communication_time:.4f}s")
        print(f"Communication Data: {total_comm_mb:.2f} MB (Upload: {upload_mb:.2f} + Download: {download_mb:.2f})")
        print(f"Round Time: {round_total_time:.2f}s")
        print(f"Parameters: {model_metrics['parameters']:,}")
        print(f"Model Size: {model_metrics['model_size_mb']:.2f} MB")
        print(f"Inference Time: {model_metrics['inference_time_ms']:.2f} ms")
        print(f"FLOPs: {model_metrics['flops']}")
        print("-" * 50)

        torch.cuda.empty_cache()

    # Placeholder.
    acc_mean = np.mean(all_round_metrics['accuracy'])
    acc_std = np.std(all_round_metrics['accuracy'])
    prec_mean = np.mean(all_round_metrics['precision'])
    prec_std = np.std(all_round_metrics['precision'])
    rec_mean = np.mean(all_round_metrics['recall'])
    rec_std = np.std(all_round_metrics['recall'])
    f1_mean = np.mean(all_round_metrics['f1'])
    f1_std = np.std(all_round_metrics['f1'])
    comm_mean = np.mean(all_round_metrics['communication_time'])
    comm_std = np.std(all_round_metrics['communication_time'])
    comm_mb_mean = np.mean(all_round_metrics['communication_mb'])
    comm_mb_std = np.std(all_round_metrics['communication_mb'])

    print("\n" + "=" * 70)
    print("FINAL TRAINING SUMMARY (Mean +/- Std across all rounds)")
    print("=" * 70)
    print(f"  Accuracy:         {acc_mean:.4f} +/- {acc_std:.4f}")
    print(f"  Precision:        {prec_mean:.4f} +/- {prec_std:.4f}")
    print(f"  Recall:           {rec_mean:.4f} +/- {rec_std:.4f}")
    print(f"  F1 Score:         {f1_mean:.4f} +/- {f1_std:.4f}")
    print(f"  Communication Time:   {comm_mean:.4f}s +/- {comm_std:.4f}s per round")
    print(f"  Communication Data:   {comm_mb_mean:.2f}MB +/- {comm_mb_std:.2f}MB per round")
    print(f"  Model Transfer Size:  {model_transfer_mb:.2f} MB (per client direction)")
    print(f"  Client Data:      {client_mean_samples:.1f} +/- {client_std_samples:.1f} samples/client")
    print("=" * 70)

    # Append final summary to CSV
    with open(log_path, "a") as f:
        f.write("\nFINAL SUMMARY (Mean +/- Std)\n")
        f.write("Metric,Mean,Std\n")
        f.write(f"Accuracy,{acc_mean:.6f},{acc_std:.6f}\n")
        f.write(f"Precision,{prec_mean:.6f},{prec_std:.6f}\n")
        f.write(f"Recall,{rec_mean:.6f},{rec_std:.6f}\n")
        f.write(f"F1,{f1_mean:.6f},{f1_std:.6f}\n")
        f.write(f"Communication_Time_Sec,{comm_mean:.6f},{comm_std:.6f}\n")
        f.write(f"Communication_Data_MB,{comm_mb_mean:.2f},{comm_mb_std:.2f}\n")
        f.write(f"Model_Transfer_Size_MB,{model_transfer_mb:.2f},\n")
        f.write(f"Client_Samples,{client_mean_samples:.1f},{client_std_samples:.1f}\n")

    final_model_path = os.path.join(save_dir, "final_global_model.pth")
    torch.save(global_model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    return global_model


def average_weights(client_models):
    """Placeholder."""
    """Aggregate client model weights using weighted averaging."""

    global_dict = client_models[0].state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([model.state_dict()[key].float() for model in client_models], dim=0).mean(dim=0)
    return global_dict