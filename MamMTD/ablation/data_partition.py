# -*- coding: utf-8 -*-
"""
Non-IID Data Partition Strategies for Federated Learning.
Supports Dirichlet distribution with varying alpha and IID baseline.
"""

import numpy as np
from collections import defaultdict


def partition_iid(X, y, num_clients):
    """IID partition: evenly distribute data across clients."""
    X_arr = X.values.astype(np.float32) if hasattr(X, 'values') else np.array(X, dtype=np.float32)
    y_arr = y.values.astype(np.int64) if hasattr(y, 'values') else np.array(y, dtype=np.int64)
    indices = np.random.permutation(len(y_arr))
    clients = np.array_split(indices, num_clients)
    return [(X_arr[c].copy(), y_arr[c].copy()) for c in clients]


def partition_dirichlet(X, y, num_clients, alpha=0.5):
    """
    Non-IID partition using Dirichlet distribution.
    
    Args:
        X: feature DataFrame/array
        y: label Series/array  
        num_clients: number of clients
        alpha: Dirichlet concentration parameter
               - alpha < 1.0: highly skewed (extreme non-IID)
               - alpha = 1.0: moderately skewed
               - alpha > 1.0: approaching IID
    """
    y_np = y.values.astype(np.int64) if hasattr(y, 'values') else np.array(y, dtype=np.int64)
    X_np = X.values.astype(np.float32) if hasattr(X, 'values') else np.array(X, dtype=np.float32)
    
    unique_labels = np.unique(y_np)
    num_classes = len(unique_labels)
    
    # Group indices by class
    class_indices = {c: np.where(y_np == c)[0] for c in unique_labels}
    
    # Dirichlet distribution for each class
    proportions = np.random.dirichlet([alpha] * num_clients, size=num_classes)
    
    client_data = []
    np.random.seed(42)  # Reproducibility
    
    for client_id in range(num_clients):
        client_indices = []
        for class_idx, label in enumerate(unique_labels):
            class_size = len(class_indices[label])
            num_samples = max(1, int(proportions[class_idx, client_id] * class_size))
            selected = np.random.choice(class_indices[label], size=min(num_samples, class_size), replace=False)
            client_indices.extend(selected)
        
        np.random.shuffle(client_indices)
        client_data.append((
            X_np[client_indices].copy(),
            y_np[client_indices].copy()
        ))
    
    return client_data


def get_partition(X, y, num_clients, strategy='dirichlet', alpha=0.5):
    """
    Get data partition based on strategy.
    
    Args:
        strategy: 'iid' or 'dirichlet'
        alpha: Dirichlet alpha (only for dirichlet)
    """
    if strategy == 'iid':
        return partition_iid(X, y, num_clients)
    elif strategy == 'dirichlet':
        return partition_dirichlet(X, y, num_clients, alpha)
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")
