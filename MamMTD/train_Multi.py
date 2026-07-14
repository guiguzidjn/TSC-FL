import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score

from MamMTD import multi_Classification
from MamMTD.data_process.data_processing_UNSW15 import *
from sklearn.utils import resample
import numpy as np


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch.nn.functional as F

import pandas as pd
import torch
from imblearn.over_sampling import SMOTE

from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')
warnings.filterwarnings('ignore', message='.*ill-defined.*')

def client_data_enhance(X, y):
    """Enhanced data balancing strategy with targeted augmentation"""
    class_counts = np.bincount(y)
    target_count = max(class_counts) // 2  # Balance target - half of the majority class

    min_count = 500  # Minimum samples per class
    max_count = 15000  # Maximum samples per class

    # Separate strategies for oversampling and undersampling
    over_strategy = {}  # For classes that need more samples
    under_strategy = {}  # For classes that need fewer samples

    for cls, count in enumerate(class_counts):
        if count < min_count or count < target_count * 1.5:
            # Classes that need oversampling
            target_samples = min(max_count, max(min_count, target_count))
            if target_samples > count:  # Only oversample if target is greater than current
                over_strategy[cls] = target_samples
        elif count > target_count * 1.5:
            # Overrepresented class - needs undersampling
            under_strategy[cls] = min(count, int(target_count * 1.2))

    # Process the data in stages
    try:
        # Stage 1: Undersampling first (if needed)
        if under_strategy:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
            X, y = rus.fit_resample(X, y)

        # Stage 2: PCA-based dimension reduction for efficient processing
        from sklearn.decomposition import PCA
        n_components = min(100, X.shape[1])
        if n_components < 2:
            n_components = 2  # Ensure at least 2 components
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)

        # Stage 3: Oversampling (if needed)
        if over_strategy:
            # Get updated class counts
            updated_class_counts = np.bincount(y)
            # Adjust k_neighbors to be valid (min class size - 1, minimum 1)
            min_class_size = min([updated_class_counts[cls] for cls in over_strategy.keys()])
            k_neighbors = min(5, max(1, min_class_size - 1))

            try:
                # Try SMOTE first
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(
                    sampling_strategy=over_strategy,
                    k_neighbors=k_neighbors,
                    random_state=42
                )
                X_res, y_res = smote.fit_resample(X_pca, y)
            except Exception as e:
                print(f"SMOTE failed: {str(e)}")
                try:
                    # KMeansSMOTE as fallback
                    from imblearn.over_sampling import KMeansSMOTE
                    kmeans_smote = KMeansSMOTE(
                        sampling_strategy=over_strategy,
                        k_neighbors=max(1, k_neighbors - 1),  # Reduce k_neighbors further if needed
                        cluster_balance_threshold=0.05,
                        random_state=42
                    )
                    X_res, y_res = kmeans_smote.fit_resample(X_pca, y)
                except Exception as e:
                    print(f"KMeansSMOTE failed: {str(e)}")
                    # Final fallback to random oversampling
                    from imblearn.over_sampling import RandomOverSampler
                    ros = RandomOverSampler(sampling_strategy=over_strategy, random_state=42)
                    X_res, y_res = ros.fit_resample(X_pca, y)
                    print("Using random oversampling as final fallback")

            # Stage 4: Feature space restoration if PCA was used
            X_res = pca.inverse_transform(X_res)

            # Stage 5: Noise injection for robustness (only for synthetic samples)
            original_indices = np.arange(len(X))
            synthetic_indices = np.setdiff1d(np.arange(len(X_res)), original_indices[:len(X_res)])

            if len(synthetic_indices) > 0:
                noise_scale = 0.02  # 2% noise
                X_res[synthetic_indices] += np.random.normal(0, noise_scale,
                                                             size=(len(synthetic_indices), X_res.shape[1]))
        else:
            # No oversampling needed
            X_res, y_res = X, y
    except Exception as e:
        print(f"Error in main processing pipeline: {str(e)}")
        # Ultimate fallback - just return the original data
        print("Returning original data without enhancement")
        return X, y

    # Verify minimum sample counts and add more if needed
    unique_classes = np.unique(y_res)
    for cls in unique_classes:
        count = (y_res == cls).sum()
        if count < min_count:
            # Add more samples through duplication with perturbation
            indices = np.where(y_res == cls)[0]
            need = min_count - count

            # Handle case where we need more than available (repeat with replacement)
            selected = np.random.choice(indices, size=need, replace=True)

            # Add small random perturbations to duplicated samples
            perturbed_samples = X_res[selected].copy()
            perturbed_samples += np.random.normal(0, 0.05, size=perturbed_samples.shape)

            X_res = np.vstack([X_res, perturbed_samples])
            y_res = np.concatenate([y_res, np.full(need, cls)])

    return X_res, y_res

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score
from MamMTD.data_process.data_processing_UNSW15 import *
from sklearn.utils import resample
import numpy as np


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch.nn.functional as F

import pandas as pd
import torch
from imblearn.over_sampling import SMOTE

from torch.utils.data import DataLoader, TensorDataset
import time
import psutil
import os
from thop import profile, clever_format
import csv
from pathlib import Path
from collections import defaultdict
import copy


pass  # Placeholder

class ModelEfficiencyTracker:
    """Placeholder."""

    def __init__(self, model, input_dim, device):
        self.model = model
        self.input_dim = input_dim
        self.device = device
        self.process = psutil.Process(os.getpid())

    def get_model_size(self):
        """Placeholder."""
        param_size = 0
        buffer_size = 0

        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb

    def get_parameter_count(self):
        """Placeholder."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def get_flops_and_memory(self, batch_size=1):
        """Placeholder."""
        try:
            # Create deep copy of model to avoid modifying original
            import copy
            model_copy = copy.deepcopy(self.model)
            model_copy.eval()

            dummy_input = torch.randn(batch_size, self.input_dim).to(self.device)
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
            flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

            pass  # Placeholder
            memory_access_mb = params * 4 / 1024 ** 2

            pass  # Placeholder
            del model_copy
            torch.cuda.empty_cache()

            return flops, flops_formatted, memory_access_mb
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
            return 0, "N/A", 0

    def get_memory_usage(self):
        """Placeholder."""
        mem_info = self.process.memory_info()
        return mem_info.rss / 1024 ** 2  # RSS: Resident Set Size

    def get_gpu_memory_usage(self):
        """Placeholder."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024 ** 2
        return 0

    def get_all_metrics(self, batch_size=64):
        """Placeholder."""
        total_params, trainable_params = self.get_parameter_count()
        model_size = self.get_model_size()
        flops, flops_str, memory_access = self.get_flops_and_memory(batch_size)
        cpu_memory = self.get_memory_usage()
        gpu_memory = self.get_gpu_memory_usage()

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size,
            'flops': flops,
            'flops_formatted': flops_str,
            'memory_access_mb': memory_access,
            'cpu_memory_mb': cpu_memory,
            'gpu_memory_mb': gpu_memory
        }


pass  # Placeholder

# def distribute_multi_data(X_train, y_train, client_num, save_data=False):
#     client_data = []
#     X_train = X_train.values.astype(np.float32)
#     y_train = y_train.values.astype(np.int64)
#     X_train, y_train = client_data_enhance(X_train, y_train)
#     class_indices = {cls: np.where(y_train == cls)[0] for cls in np.unique(y_train)}
#     client_class_data = {i: [] for i in range(client_num)}
# Placeholder.
#     for cls, indices in class_indices.items():
#         np.random.shuffle(indices)
#         samples_per_client = len(indices) // client_num
#         for i in range(client_num):
#             client_class_data[i].extend(indices[i * samples_per_client: (i + 1) * samples_per_client])
# Placeholder.
#     for client_id, indices in client_class_data.items():
#         client_X = X_train[indices]
#         client_y = y_train[indices]
#         client_data.append((client_X, client_y))
# Placeholder.
#         if save_data:
#             client_dir = f"client_data/client_{client_id}"
#             os.makedirs(client_dir, exist_ok=True)
#             np.save(os.path.join(client_dir, 'X.npy'), client_X)
#             np.save(os.path.join(client_dir, 'y.npy'), client_y)
# Placeholder.
#     return client_data

def distribute_multi_data(X_train, y_train, client_num, alpha=0.5, save_data=False):
    """Placeholder."""
    pass  # Placeholder

# 鍙傛暟:
# - X_train: 璁 粌鏁版嵁鐗瑰緛
# - y_train: 璁 粌鏁版嵁镙囩
# - client_num: 瀹 埛绔 暟阅?
# - alpha: 镫勫埄鍏嬮浄 嗗竷镄勯泦涓 害鍙傛暟
# - alpha < 1: 鏁版嵁 嗗竷旋翠笉鍧囱 锛堟洿Non-IID锛?
# - alpha = 1: 鍧囧寝 嗗竷
# - alpha > 1: 鏁版嵁 嗗竷旋村潎琛?
# - save_data: 鏄 惁淇 瓨鏁版嵁 版枃浠?

# 杩斿洖:
# - client_data:  楄 锛屾疮涓 厓绱犳槸(client_X, client_y)鍏幂粍
    """Placeholder."""
    client_data = []
    X_train = X_train.values.astype(np.float32)
    y_train = y_train.values.astype(np.int64)
    X_train, y_train = client_data_enhance(X_train, y_train)

    pass  # Placeholder
    classes = np.unique(y_train)
    num_classes = len(classes)
    class_indices = {cls: np.where(y_train == cls)[0] for cls in classes}

    pass  # Placeholder
    client_indices = {i: [] for i in range(client_num)}

    pass  # Placeholder
    for cls in classes:
        indices = class_indices[cls]
        np.random.shuffle(indices)

        pass  # Placeholder
        proportions = np.random.dirichlet(np.repeat(alpha, client_num))

        pass  # Placeholder
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

        pass  # Placeholder
        split_indices = np.split(indices, proportions)
        for client_id, split_idx in enumerate(split_indices):
            client_indices[client_id].extend(split_idx)

    pass  # Placeholder
    for client_id, indices in client_indices.items():
        indices = np.array(indices)
# np.random.shuffle(indices)  # 镓扑贡椤哄簭

        client_X = X_train[indices]
        client_y = y_train[indices]
        client_data.append((client_X, client_y))

        if save_data:
            client_dir = f"client_data/client_{client_id}"
            os.makedirs(client_dir, exist_ok=True)
            np.save(os.path.join(client_dir, 'X.npy'), client_X)
            np.save(os.path.join(client_dir, 'y.npy'), client_y)

            pass  # Placeholder
            unique, counts = np.unique(client_y, return_counts=True)
            class_distribution = dict(zip(unique, counts))
# print(f"Client {client_id} - 镙锋湰鏁? {len(client_y)}, 绫诲埆 嗗竷: {class_distribution}")

    return client_data


import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter


pass  # Placeholder

def borderline_smote(X, y, target_class, k_neighbors=5, n_samples=None):
    """Placeholder."""
# Borderline-SMOTE: 鍙  杈圭晫闄勮繎镄勫皯鏁扮被镙锋湰杩涜 杩囬噰镙?

# 鍙傛暟:
# - X: 鐗瑰緛鏁版嵁
# - y: 镙囩
# - target_class:   爣灏戛暟绫?
# - k_neighbors: 杩戦偦鏁伴噺
# - n_samples: 鐢熸垚镄勬牱链 暟阅忥纴None 栾嚜锷  绠?
    """Placeholder."""
    pass  # Placeholder
    minority_indices = np.where(y == target_class)[0]
    majority_indices = np.where(y != target_class)[0]

    if len(minority_indices) == 0:
        return np.array([]), np.array([])

    X_minority = X[minority_indices]
    X_all = X

    pass  # Placeholder
    k_neighbors_adjusted = min(k_neighbors, len(X_all) - 1)
    if k_neighbors_adjusted < 1:
# print(f"璀 憡: 绫诲埆 {target_class} 镙锋湰澶 皯锛屾棤娉叠繘琛孲MOTE")
        return np.array([]), np.array([])

    pass  # Placeholder
    nbrs = NearestNeighbors(n_neighbors=k_neighbors_adjusted + 1).fit(X_all)
    distances, indices = nbrs.kneighbors(X_minority)

    pass  # Placeholder
    borderline_samples = []
    borderline_indices = []

    for i, neighbors in enumerate(indices):
        pass  # Placeholder
        neighbor_classes = y[neighbors[1:]]
        majority_count = np.sum(neighbor_classes != target_class)

        pass  # Placeholder
        if 0.5 <= majority_count / k_neighbors_adjusted < 1.0:
            borderline_samples.append(X_minority[i])
            borderline_indices.append(i)

    if len(borderline_samples) == 0:
        pass  # Placeholder
        borderline_samples = X_minority
        borderline_indices = list(range(len(X_minority)))

    borderline_samples = np.array(borderline_samples)

    pass  # Placeholder
    if len(borderline_samples) < 2:
        print("Placeholder message")
        if n_samples is None:
            n_samples = len(X_minority)
        pass  # Placeholder
        synthetic_samples = []
        for _ in range(n_samples):
            idx = np.random.randint(len(borderline_samples))
            sample = borderline_samples[idx]
            pass  # Placeholder
            noise = np.random.normal(0, 0.01, sample.shape)
            synthetic = sample + noise
            synthetic_samples.append(synthetic)
        return np.array(synthetic_samples), np.full(len(synthetic_samples), target_class)

    pass  # Placeholder
    if n_samples is None:
        n_samples = len(X_minority)  # 榛椫 鐢熸垚涓庡皯鏁扮被 稿悓鏁伴噺镄勬牱链?

    pass  # Placeholder
    k_neighbors_borderline = min(k_neighbors, len(borderline_samples) - 1)

    pass  # Placeholder
    synthetic_samples = []
    nbrs_minority = NearestNeighbors(n_neighbors=k_neighbors_borderline + 1).fit(borderline_samples)

    for _ in range(n_samples):
        pass  # Placeholder
        idx = np.random.randint(len(borderline_samples))
        sample = borderline_samples[idx]

        pass  # Placeholder
        distances, indices = nbrs_minority.kneighbors([sample])
        pass  # Placeholder
        neighbor_idx = np.random.choice(indices[0][1:])
        neighbor = borderline_samples[neighbor_idx]

        pass  # Placeholder
        alpha = np.random.rand()
        synthetic = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic)

    return np.array(synthetic_samples), np.full(len(synthetic_samples), target_class)


def add_gaussian_noise(X, y, target_classes, noise_ratio=0.1):
    """Placeholder."""
    pass  # Placeholder

# 鍙傛暟:
# - X: 鐗瑰緛鏁版嵁
# - y: 镙囩
# - target_classes: 暗 瑕佹坊锷犲櫔澹扮殑绫诲埆 楄
# - noise_ratio: 鍣  寮哄害锛堢浉瀵逛簬鐗瑰緛镙囧嗳宸 级
    """Placeholder."""
    augmented_X = []
    augmented_y = []

    for cls in target_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) == 0:
            continue

        X_cls = X[cls_indices]

        pass  # Placeholder
        std_devs = np.std(X_cls, axis=0)
# std_devs = np.where(std_devs == 0, 1e-6, std_devs)  # 阆垮历闄 浂

        pass  # Placeholder
        noise = np.random.normal(0, noise_ratio, X_cls.shape) * std_devs
        X_noised = X_cls + noise

        augmented_X.append(X_noised)
        augmented_y.append(np.full(len(X_noised), cls))

    if len(augmented_X) > 0:
        return np.vstack(augmented_X), np.concatenate(augmented_y)
    else:
        return np.array([]), np.array([])


def client_data_enhance_method1(X_train, y_train, minority_threshold=0.3,
                                k_neighbors=5, noise_ratio=0.05):
    """Placeholder."""
# 鏂规 1: Borderline-SMOTE + Gaussian Noise

# 鍙傛暟:
# - X_train: 璁 粌鐗瑰緛
# - y_train: 璁 粌镙囩
# - minority_threshold: 灏戛暟绫婚槇炼硷纸浣庝簬姝 瘆渚嬬殑绫昏 璁 负鏄 皯鏁扮被锛?
# - k_neighbors: SMOTE镄勮繎闾绘暟
# - noise_ratio: 楂樻柉鍣  寮哄害
    """Placeholder."""
    pass  # Placeholder
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    minority_classes = [cls for cls, count in class_counts.items()
                        if count / total_samples < minority_threshold]

    if len(minority_classes) == 0:
# print("链  娴嫔埌灏戛暟绫伙纴杩斿洖铡熷 鏁版嵁")
        return X_train, y_train

# print(f"妫 娴嫔埌灏戛暟绫? {minority_classes}")

    pass  # Placeholder
    all_synthetic_X = [X_train]
    all_synthetic_y = [y_train]

    pass  # Placeholder
    for cls in minority_classes:
        cls_count = class_counts[cls]
        pass  # Placeholder
        avg_count = int(total_samples / len(class_counts))
        n_samples = max(0, avg_count - cls_count)

        if n_samples > 0:
# print(f"涓虹被 ?{cls} 鐢熸垚 {n_samples} 涓狟orderline-SMOTE镙锋湰")

            pass  # Placeholder
            actual_k = min(k_neighbors, len(X_train) - 1, cls_count - 1)
# actual_k = max(1, actual_k)  # 镊冲皯涓?

            synthetic_X, synthetic_y = borderline_smote(
                X_train, y_train, cls, k_neighbors=actual_k, n_samples=n_samples
            )

            if len(synthetic_X) > 0:
                all_synthetic_X.append(synthetic_X)
                all_synthetic_y.append(synthetic_y)
                print("Placeholder message")
            else:
                print("Placeholder message")

    pass  # Placeholder
# print(f"涓哄皯鏁扮被娣诲姞楂樻柉鍣  澧炲己")
    noised_X, noised_y = add_gaussian_noise(
        X_train, y_train, minority_classes, noise_ratio=noise_ratio
    )

    if len(noised_X) > 0:
        all_synthetic_X.append(noised_X)
        all_synthetic_y.append(noised_y)

    pass  # Placeholder
    X_enhanced = np.vstack(all_synthetic_X)
    y_enhanced = np.concatenate(all_synthetic_y)

    pass  # Placeholder
    indices = np.arange(len(X_enhanced))
    np.random.shuffle(indices)
    X_enhanced = X_enhanced[indices]
    y_enhanced = y_enhanced[indices]

# print(f"鏁版嵁澧炲己瀹屾垚: {len(y_train)} -> {len(y_enhanced)} 镙锋湰")

    pass  # Placeholder
    enhanced_counts = Counter(y_enhanced)
# print("澧炲己钖庣被  垎宁?")
    for cls in sorted(enhanced_counts.keys()):
        pass  # print(f"  绫诲埆 {cls}: {enhanced_counts[cls]} 镙锋湰")

    return X_enhanced, y_enhanced


pass  # Placeholder

def prototype_based_augmentation(X, y, n_prototypes=5, augment_factor=2.0):
    """Placeholder."""
    pass  # Placeholder

# 鍙傛暟:
# - X: 鐗瑰緛鏁版嵁
# - y: 镙囩
# - n_prototypes: 姣忎釜绫诲埆镄勫师鍨嬫暟阅?
# - augment_factor: 澧炲己炼嶆暟锛堢敚鎴愮殑镙锋湰鏁?= 铡熷 镙锋湰鏁?* augment_factor锛?
    """Placeholder."""
    classes = np.unique(y)
    all_synthetic_X = [X]
    all_synthetic_y = [y]

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        X_cls = X[cls_indices]

        if len(X_cls) < n_prototypes:
            pass  # Placeholder
            prototypes = X_cls
        else:
            pass  # Placeholder
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_prototypes, random_state=42, n_init=10)
            kmeans.fit(X_cls)
            prototypes = kmeans.cluster_centers_

        pass  # Placeholder
        n_samples = int(len(X_cls) * augment_factor)

        synthetic_samples = []

        pass  # Placeholder
        if len(prototypes) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(prototypes)
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
        else:
            pass  # Placeholder
            avg_distance = 1.0
            std_distance = np.std(X_cls, axis=0).mean()

        for _ in range(n_samples):
            pass  # Placeholder
            if len(prototypes) > 1 and np.random.rand() < 0.7:
                idx1, idx2 = np.random.choice(len(prototypes), 2, replace=False)
                proto1, proto2 = prototypes[idx1], prototypes[idx2]
# alpha = np.random.beta(2, 2)  # Beta 嗗竷浣挎 炼兼洿板嗕腑鍦 腑闂?
                synthetic = alpha * proto1 + (1 - alpha) * proto2

            pass  # Placeholder
            pass  # removed orphan else:
            proto_idx = np.random.randint(len(prototypes))
            proto = prototypes[proto_idx]

            pass  # Placeholder
# noise_scale = std_distance * 0.3  # 鍣  灏哄害涓鸿窛绂绘爣鍑嗗樊镄?0%
            noise = np.random.normal(0, noise_scale, proto.shape)
            synthetic = proto + noise

            synthetic_samples.append(synthetic)

        synthetic_samples = np.array(synthetic_samples)
        synthetic_labels = np.full(len(synthetic_samples), cls)

        all_synthetic_X.append(synthetic_samples)
        all_synthetic_y.append(synthetic_labels)

# print(f"绫诲埆 {cls}: 铡熷瀷鏁?{len(prototypes)}, 鐢熸垚镙锋湰鏁?{len(synthetic_samples)}")

    pass  # Placeholder
    X_enhanced = np.vstack(all_synthetic_X)
    y_enhanced = np.concatenate(all_synthetic_y)

    pass  # Placeholder
    indices = np.arange(len(X_enhanced))
    np.random.shuffle(indices)
    X_enhanced = X_enhanced[indices]
    y_enhanced = y_enhanced[indices]

    return X_enhanced, y_enhanced


def client_data_enhance_method2(X_train, y_train, n_prototypes=5, augment_factor=1.5):
    """Placeholder."""
# 鏂规 2: Prototype-based Augmentation

# 鍙傛暟:
# - X_train: 璁 粌鐗瑰緛
# - y_train: 璁 粌镙囩
# - n_prototypes: 姣忎釜绫诲埆镄勫师鍨嬫暟阅?
# - augment_factor: 澧炲己炼嶆暟
    """Placeholder."""
# print(f"浣跨敤锘轰簬铡熷瀷镄勬暟鎹  寮?(铡熷瀷鏁?{n_prototypes}, 澧炲己炼嶆暟={augment_factor})")

    X_enhanced, y_enhanced = prototype_based_augmentation(
        X_train, y_train,
        n_prototypes=n_prototypes,
        augment_factor=augment_factor
    )

# print(f"鏁版嵁澧炲己瀹屾垚: {len(y_train)} -> {len(y_enhanced)} 镙锋湰")
    return X_enhanced, y_enhanced


pass  # Placeholder

def distribute_multi_data_v2(X_train, y_train, client_num, alpha=0.5,
                             enhancement_method=1, save_data=False, **enhance_kwargs):
    """Placeholder."""
    pass  # Placeholder

# 鍙傛暟:
# - X_train: 璁 粌鏁版嵁鐗瑰緛
# - y_train: 璁 粌鏁版嵁镙囩
# - client_num: 瀹 埛绔 暟阅?
# - alpha: 镫勫埄鍏嬮浄 嗗竷镄勯泦涓 害鍙傛暟
# - enhancement_method: 澧炲己鏂规硶 (1: Borderline-SMOTE+Noise, 2: Prototype-based)
# - save_data: 鏄 惁淇 瓨鏁版嵁 版枃浠?
# - enhance_kwargs: 浼犻 掔粰澧炲己鍑芥暟镄勯 澶栧弬鏁?
    """Placeholder."""
    import os

    client_data = []
    X_train = X_train.values.astype(np.float32) if hasattr(X_train, 'values') else X_train.astype(np.float32)
    y_train = y_train.values.astype(np.int64) if hasattr(y_train, 'values') else y_train.astype(np.int64)

    pass  # Placeholder
    classes = np.unique(y_train)
    num_classes = len(classes)
    class_indices = {cls: np.where(y_train == cls)[0] for cls in classes}

    pass  # Placeholder
    client_indices = {i: [] for i in range(client_num)}

    pass  # Placeholder
    for cls in classes:
        indices = class_indices[cls]
        np.random.shuffle(indices)

        pass  # Placeholder
        proportions = np.random.dirichlet(np.repeat(alpha, client_num))

        pass  # Placeholder
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

        pass  # Placeholder
        split_indices = np.split(indices, proportions)
        for client_id, split_idx in enumerate(split_indices):
            client_indices[client_id].extend(split_idx)

    pass  # Placeholder
    for client_id, indices in client_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        client_X = X_train[indices]
        client_y = y_train[indices]

        print(f"\n=== Client {client_id} ===")
# print(f"铡熷 鏁版嵁 - 镙锋湰鏁? {len(client_y)}")

        pass  # Placeholder
        if enhancement_method == 1:
            client_X, client_y = client_data_enhance_method1(
                client_X, client_y, **enhance_kwargs
            )
        elif enhancement_method == 2:
            client_X, client_y = client_data_enhance_method2(
                client_X, client_y, **enhance_kwargs
            )
        else:
            pass  # Unknown enhancement method, skip

        client_data.append((client_X, client_y))

        if save_data:
            client_dir = f"client_data/client_{client_id}"
            os.makedirs(client_dir, exist_ok=True)
            np.save(os.path.join(client_dir, 'X.npy'), client_X)
            np.save(os.path.join(client_dir, 'y.npy'), client_y)

        pass  # Placeholder
        unique, counts = np.unique(client_y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
# print(f"澧炲己钖?- 镙锋湰鏁? {len(client_y)}, 绫诲埆 嗗竷: {class_distribution}")

    return client_data



def get_class_weights(y):
    class_counts = np.bincount(y)
    weights = 1. / (class_counts + 1e-6)
    return torch.tensor(weights / weights.sum(), dtype=torch.float32).to("cuda")


class BalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is None:
            self.alpha = get_class_weights(targets.cpu().numpy()).to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal_weight = (1 - pt) ** self.gamma
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def evaluate_multi_model(global_model, X_test, y_test, device, batch_size):
    global_model.eval()
    X_test = X_test.values.astype(np.float32)
    y_test = y_test.values.astype(np.int64)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []

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

    return {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'f1': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }


def count_labels_in_client_data(client_data):
    for client_id, (client_X, client_y) in enumerate(client_data):
        counts = [f"{i}'s = {(client_y == i).sum().item()}" for i in range(9)]
        print(f"Client {client_id}: {', '.join(counts)}")


def fedavg_aggregation(global_model, client_models, client_data_sizes):
    global_dict = global_model.state_dict()
    total_data_size = sum(client_data_sizes)
    client_weights = [size / total_data_size for size in client_data_sizes]

    for key in global_dict:
        weighted_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for i, (model, weight) in enumerate(zip(client_models, client_weights)):
            client_param = model.state_dict()[key].float()
            weighted_sum += client_param * weight

        if global_dict[key].is_floating_point():
            global_dict[key] = weighted_sum.to(global_dict[key].dtype)
        else:
            global_dict[key] = weighted_sum.round().type_as(global_dict[key])

    return global_dict


def train_client_model(model, data_loader, optimizer, criterion, device, epochs=5):
    """Placeholder."""
    model.train()
    metrics_history = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            outputs = outputs.to(device)
            loss = criterion(outputs, targets)
            loss = loss.to(device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }

        metrics['loss'] = epoch_loss / len(data_loader)
        metrics_history.append(metrics)

        print(f"Epoch {epoch + 1}: Loss = {metrics['loss']:.4f}, Acc = {metrics['accuracy']:.4f}, "
              f"F1 = {metrics['f1']:.4f}")

    training_time = time.time() - start_time
    metrics_history[-1]['training_time'] = training_time

    return metrics_history[-1]


import os
import numpy as np
import matplotlib.pyplot as plt

def save_cm(cm, round_idx, save_dir="confusion_matrix"):
    os.makedirs(save_dir, exist_ok=True)

    pass  # Placeholder
    if hasattr(cm, "cpu"):
        cm = cm.cpu().numpy()

    cm = np.round(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Round {round_idx})")

    base_path = os.path.join(save_dir, f"confusion_matrix_round20_{round_idx}")

    plt.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base_path + ".pdf", bbox_inches="tight")
    plt.savefig(base_path + ".svg", bbox_inches="tight")
    plt.close()

pass  # Placeholder


pass  # Placeholder

def sequence_length_sensitivity_analysis(args, X_test, y_test, device, global_model, num_classes):
    """Analyze model sensitivity to different sequence lengths (feature dimensions).
    
    Uses feature subset masking: selects a subset of features, pads the rest with zeros,
    and runs inference with the original model (no new model creation needed).
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print("\n" + "=" * 70)
    print("Sequence Length Sensitivity Analysis (Mamba Feature Dimension Test)")
    print("=" * 70)

    global_model.eval()
    X_test_np = X_test.values.astype(np.float32) if hasattr(X_test, "values") else np.array(X_test, dtype=np.float32)
    y_test_np = y_test.values.astype(np.int64) if hasattr(y_test, "values") else np.array(y_test, dtype=np.int64)

    original_dim = X_test_np.shape[1]
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    dims = [max(2, int(original_dim * r)) for r in ratios]

    results = []
    for dim in dims:
        np.random.seed(42)
        # Select a subset of feature indices, zero out the rest
        all_indices = np.arange(original_dim)
        selected_indices = np.random.choice(all_indices, size=dim, replace=False)
        mask = np.zeros(original_dim, dtype=np.float32)
        mask[selected_indices] = 1.0
        
        X_subset = X_test_np * mask  # Zero out unselected features
        
        X_tensor = torch.tensor(X_subset, dtype=torch.float32)
        y_tensor = torch.tensor(y_test_np, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                outputs = global_model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

        all_preds = np.array(all_preds)
        acc = accuracy_score(y_test_np, all_preds)
        prec = precision_score(y_test_np, all_preds, average="weighted", zero_division=0)
        rec = recall_score(y_test_np, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test_np, all_preds, average="weighted", zero_division=0)

        results.append({"dim": dim, "ratio": dim/original_dim, "accuracy": acc,
                       "precision": prec, "recall": rec, "f1": f1})

        print(f"  Dim={dim:4d} ({dim/original_dim:.0%}): Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    import csv
    sensitivity_file = "sequence_length_sensitivity.csv"
    with open(sensitivity_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature_Dim", "Ratio", "Accuracy", "Precision", "Recall", "F1"])
        for r in results:
            writer.writerow([r["dim"], f"{r['ratio']:.2f}", f"{r['accuracy']:.6f}",
                           f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}"])

    print(f"\nSensitivity analysis saved to {sensitivity_file}")
    print("=" * 70 + "\n")
    return results


def get_model_transfer_size(model):
    """Calculate model transfer size in MB (for communication cost estimation)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 ** 2)

def Multi_classification(args, X_train, X_test, y_train, y_test, attack_cat_encoded):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    client_num = args.client_num

    # Determine num_classes: use args if available, else derive from data
    if hasattr(args, 'num_classes') and args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")

    # Prepare client data
    client_data = distribute_multi_data(X_train, y_train, client_num=client_num)

    count_labels_in_client_data(client_data)

    pass  # Placeholder
    client_sample_counts = [len(client_X) for client_X, _ in client_data]
    client_mean_samples = np.mean(client_sample_counts)
    client_std_samples = np.std(client_sample_counts)
    print(f"\n{'='*50}")
    print(f"Client Data Distribution Statistics:")
    print(f"  Samples per client: {client_sample_counts}")
# print(f"  Mean: {client_mean_samples:.1f} 鍗?{client_std_samples:.1f}")
    print(f"  Min: {min(client_sample_counts)}, Max: {max(client_sample_counts)}")
    print(f"{'='*50}\n")

    # ===== Calculate model transfer size for communication cost =====
    dummy_model = multi_Classification.SimplifiedMambaResNet(input_dim=args.input_dim, num_classes=num_classes)
    model_transfer_mb = get_model_transfer_size(dummy_model)
    del dummy_model
    print(f"Model transfer size (per client upload/download): {model_transfer_mb:.2f} MB")
    total_comm_mb_per_round = 2 * client_num * model_transfer_mb
    print(f"Total communication data per round ({client_num} clients x2): {total_comm_mb_per_round:.2f} MB\n")

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize global model
    global_model = multi_Classification.SimplifiedMambaResNet(input_dim=args.input_dim, num_classes=num_classes).to(device)

    pass  # Placeholder
    efficiency_tracker = ModelEfficiencyTracker(global_model, args.input_dim, device)

    pass  # Placeholder
    initial_metrics = efficiency_tracker.get_all_metrics(batch_size=args.batch_size)
    print("\n========== Model Efficiency Metrics ==========")
    print(f"Total Parameters: {initial_metrics['total_params']:,}")
    print(f"Trainable Parameters: {initial_metrics['trainable_params']:,}")
    print(f"Model Size: {initial_metrics['model_size_mb']:.2f} MB")
    print(f"FLOPs: {initial_metrics['flops_formatted']}")
    print(f"Memory Access: {initial_metrics['memory_access_mb']:.2f} MB")
    print(f"CPU Memory Usage: {initial_metrics['cpu_memory_mb']:.2f} MB")
    print(f"GPU Memory Usage: {initial_metrics['gpu_memory_mb']:.2f} MB")
    print("=" * 47 + "\n")

    # Define log file path
    log_file = f"training_metrics_mamba_{args.dataset.lower()}.csv"

    # Create CSV file with header (including communication data columns)
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Round",
                "Accuracy", "Precision", "Recall", "F1",
                "Training_Time_Sec", "Avg_Client_Time_Sec",
                "Communication_Time_Sec", "Upload_MB", "Download_MB", "Total_Comm_MB",
                "Total_Params", "Trainable_Params",
                "Model_Size_MB", "FLOPs", "FLOPs_Formatted",
                "Memory_Access_MB", "CPU_Memory_MB", "GPU_Memory_MB"
            ])

    pass  # Placeholder
    all_round_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'communication_time': [],
        'communication_mb': [],
        'round_time': [],
        'client_time': []
    }

    # Main training loop
    for round in range(args.communication_round):
        round_start_time = time.time()

        client_models = []
        client_metrics = []
        class_dists = []
        client_data_sizes = []
        client_training_times = []

        # Train each client
        for client in range(client_num):
            model = multi_Classification.SimplifiedMambaResNet(input_dim=args.input_dim, num_classes=num_classes).to(device)
            model.load_state_dict(global_model.state_dict())

            client_X, client_y = client_data[client]
            client_X = torch.tensor(client_X, dtype=torch.float32)
            client_y = torch.tensor(client_y, dtype=torch.long)

            class_dist = {cls: count for cls, count in zip(*np.unique(client_y, return_counts=True))}
            class_dists.append(class_dist)

            train_loader = DataLoader(
                TensorDataset(client_X, client_y),
                batch_size=min(args.batch_size, len(client_X) // 10),
                shuffle=True,
                num_workers=8,
                pin_memory=True
            )

            client_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=5e-4,
                betas=(0.9, 0.99),
                eps=1e-8,
                weight_decay=0.01
            )

            print(f"Training client {client}")
            metrics = train_client_model(
                model=model,
                data_loader=train_loader,
                optimizer=client_optimizer,
                criterion=criterion,
                device=device,
                epochs=5
            )

            client_models.append(model)
            client_metrics.append(metrics)
            client_data_sizes.append(len(client_X))
            client_training_times.append(metrics.get('training_time', 0))

            torch.cuda.empty_cache()

        pass  # Placeholder
        comm_start_time = time.time()
        # Aggregate models
        global_model.load_state_dict(
            fedavg_aggregation(global_model, client_models, client_data_sizes))
        communication_time = time.time() - comm_start_time

        os.makedirs("multi_model", exist_ok=True)

        # Evaluate global model
        eval_start_time = time.time()
        metrics = evaluate_multi_model(global_model, X_test, y_test, device, args.batch_size)
        eval_time = time.time() - eval_start_time
        cm = metrics['confusion_matrix']
        save_cm(
            cm,
            round_idx=round,
            save_dir="confusion_matrix"
        )

        pass  # Placeholder

        pass  # Placeholder
        # Calculate round total time
        round_total_time = time.time() - round_start_time

        # Get current efficiency metrics
        current_efficiency = efficiency_tracker.get_all_metrics(batch_size=args.batch_size)

        avg_client_time = np.mean(client_training_times) if client_training_times else 0

        # Calculate communication data cost
        upload_mb = model_transfer_mb * client_num
        download_mb = model_transfer_mb * client_num
        total_comm_mb = upload_mb + download_mb

        all_round_metrics['accuracy'].append(metrics['accuracy'])
        all_round_metrics['precision'].append(metrics['precision'])
        all_round_metrics['recall'].append(metrics['recall'])
        all_round_metrics['f1'].append(metrics['f1'])
        all_round_metrics['communication_time'].append(communication_time)
        all_round_metrics['communication_mb'].append(total_comm_mb)
        all_round_metrics['round_time'].append(round_total_time)
        all_round_metrics['client_time'].append(avg_client_time)

        round_num = round + 1
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                f"{metrics['accuracy']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall']:.6f}",
                f"{metrics['f1']:.6f}",
                f"{round_total_time:.2f}",
                f"{avg_client_time:.2f}",
                f"{communication_time:.4f}",
                f"{upload_mb:.2f}",
                f"{download_mb:.2f}",
                f"{total_comm_mb:.2f}",
                current_efficiency['total_params'],
                current_efficiency['trainable_params'],
                f"{current_efficiency['model_size_mb']:.2f}",
                current_efficiency['flops'],
                current_efficiency['flops_formatted'],
                f"{current_efficiency['memory_access_mb']:.2f}",
                f"{current_efficiency['cpu_memory_mb']:.2f}",
                f"{current_efficiency['gpu_memory_mb']:.2f}"
            ])

        print(f"\n{'=' * 60}")
        print(f"Round {round_num} Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Round Training Time: {round_total_time:.2f}s")
        print(f"Average Client Time: {avg_client_time:.2f}s")
        print(f"Communication Time: {communication_time:.4f}s")
        print(f"Communication Data: {total_comm_mb:.2f} MB (Upload: {upload_mb:.2f} + Download: {download_mb:.2f})")
        print(f"Evaluation Time: {eval_time:.2f}s")
        print(f"GPU Memory: {current_efficiency['gpu_memory_mb']:.2f} MB")
        print(f"{'=' * 60}\n")

        torch.cuda.empty_cache()

    pass  # Placeholder
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
# print("FINAL TRAINING SUMMARY (Mean 鍗?Std across all rounds)")
    print("=" * 70)
# print(f"  Accuracy:         {acc_mean:.4f} 鍗?{acc_std:.4f}")
# print(f"  Precision:        {prec_mean:.4f} 鍗?{prec_std:.4f}")
# print(f"  Recall:           {rec_mean:.4f} 鍗?{rec_std:.4f}")
# print(f"  F1 Score:         {f1_mean:.4f} 鍗?{f1_std:.4f}")
# print(f"  Communication Time:   {comm_mean:.4f}s 鍗?{comm_std:.4f}s per round")
# print(f"  Communication Data:   {comm_mb_mean:.2f}MB 鍗?{comm_mb_std:.2f}MB per round")
    print(f"  Model Transfer Size:  {model_transfer_mb:.2f} MB (per client direction)")
# print(f"  Avg Round Time:   {np.mean(all_round_metrics['round_time']):.2f}s 鍗?{np.std(all_round_metrics['round_time']):.2f}s")
# print(f"  Avg Client Time:  {np.mean(all_round_metrics['client_time']):.2f}s 鍗?{np.std(all_round_metrics['client_time']):.2f}s")
# print(f"  Client Data:      {client_mean_samples:.1f} 鍗?{client_std_samples:.1f} samples/client")
    print("=" * 70)

    pass  # Placeholder
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
# writer.writerow([])  # 缁岄缚颟?
        writer.writerow(["FINAL SUMMARY (Mean +/- Std)"])
        writer.writerow(["Metric", "Mean", "Std"])
        writer.writerow(["Accuracy", f"{acc_mean:.6f}", f"{acc_std:.6f}"])
        writer.writerow(["Precision", f"{prec_mean:.6f}", f"{prec_std:.6f}"])
        writer.writerow(["Recall", f"{rec_mean:.6f}", f"{rec_std:.6f}"])
        writer.writerow(["F1", f"{f1_mean:.6f}", f"{f1_std:.6f}"])
        writer.writerow(["Communication_Time_Sec", f"{comm_mean:.6f}", f"{comm_std:.6f}"])
        writer.writerow(["Communication_Data_MB", f"{comm_mb_mean:.2f}", f"{comm_mb_std:.2f}"])
        writer.writerow(["Model_Transfer_Size_MB", f"{model_transfer_mb:.2f}", ""])
        writer.writerow(["Client_Samples", f"{client_mean_samples:.1f}", f"{client_std_samples:.1f}"])

    # Save final model
    final_model_path = "multi_model/global_model_final.pth"
    torch.save(global_model.state_dict(), final_model_path)
    print(f"\nFinal model saved at {final_model_path}")
    print(f"Enhanced metrics logged to {log_file}")

    pass  # Placeholder
    sensitivity_results = sequence_length_sensitivity_analysis(
        args, X_test, y_test, device, global_model, num_classes
    )

    return global_model
