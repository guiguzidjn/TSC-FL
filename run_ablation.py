# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
"""
TSC-FL 消融实验运行器
=====================
一键运行所有消融实验:
  Single-stage vs Two-stage
  Stage 1 主干变体 (ResNet/MLP/CNN/LSTM)
  Stage 2 分类器变体 (Mamba/MLP/CNN/LSTM/Transformer)
  Non-IID 分区策略 (IID, Dirichlet)
"""

import sys, io, csv, argparse
import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from config_ablation import (
    BASE_CONFIG, DATASET_CONFIGS, ALL_ABLATIONS,
    ABLATION_STAGE, ABLATION_STAGE1, ABLATION_STAGE2, ABLATION_PARTITION
)
from MamMTD.ablation.trainer import AblationTrainer


def load_data(dataset_name):
    """加载并预处理数据集。"""
    ds = DATASET_CONFIGS[dataset_name]
    data_path = ds["data_path"]

    if dataset_name == "CICIDS2017":
        from MamMTD.data_process.data_processing_CICIDS2017 import preprocess_data_CICIDS as data_fn
    elif dataset_name == "UNSW-NB15":
        from MamMTD.data_process.data_processing_UNSW15 import preprocess_data_UNSW15 as data_fn
    elif dataset_name == "Nba-IoT":
        from MamMTD.data_process.data_processing_NbaIoT import preprocess_data_NbaIoT as data_fn
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    class Args:
        pass
    args = Args()
    args.task = "multi"
    args.dataset = dataset_name

    X_train, X_test, y_train, y_test, label_mapping = data_fn(data_path, args)

    y_train_np = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    attack_indices = y_train_np != 0
    attack_X_train = X_train.iloc[attack_indices] if hasattr(X_train, "iloc") else X_train[attack_indices]
    attack_y_train = y_train_np[attack_indices] - 1

    y_train_bin = (y_train_np != 0).astype(np.int64)
    y_test_np = y_test.values.astype(np.int64) if hasattr(y_test, "values") else np.array(y_test, dtype=np.int64)
    y_test_bin = (y_test_np != 0).astype(np.int64)
    attack_X_train = attack_X_train.values.astype(np.float32) if hasattr(attack_X_train, 'values') else np.array(attack_X_train, dtype=np.float32)
    attack_y_train = attack_y_train.astype(np.int64)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_train_bin": y_train_bin, "y_test_bin": y_test_bin,
        "attack_X_train": attack_X_train, "attack_y_train": attack_y_train,
        "label_mapping": label_mapping,
        "num_classes": ds["num_classes"],
        "num_attack_classes": ds["num_attack_classes"],
        "input_dim": ds["input_dim"],
    }


def run_experiment(name, config, data, dataset_name):
    """Run a single ablation experiment."""
    cfg = {**BASE_CONFIG, **config, 'input_dim': data['input_dim']}
    trainer = AblationTrainer(cfg)
    prefix = f"{dataset_name}_"

    print(f"\n{'#'*70}")
    print(f"# Ablation: {name}")
    print(f"{'#'*70}")

    try:
        if cfg['mode'] == 'single_stage':
            result = trainer.run_single_stage(
                data['X_train'], data['X_test'],
                data['y_train'], data['y_test'],
                num_classes=data['num_classes'],
                log_prefix=prefix
            )
        elif cfg['mode'] == 'two_stage':
            result = trainer.run_two_stage(
                data['X_train'], data['X_test'],
                data['y_train_bin'], data['y_test_bin'],
                attack_data=(data['attack_X_train'], data['attack_y_train']),
                num_attack_classes=data['num_attack_classes'],
                log_prefix=prefix
            )
        else:
            raise ValueError(f"Unknown mode: {cfg['mode']}")

        result['experiment'] = name
        result['dataset'] = dataset_name
        result['mode'] = cfg['mode']
        return result
    except Exception as e:
        print(f"ERROR in {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_ablations(datasets=None, experiments=None):
    """Run ablation experiments."""
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())
    if experiments is None:
        experiments = ALL_ABLATIONS

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"Loading: {ds_name}")
        print(f"{'='*70}")

        try:
            data = load_data(ds_name)
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        for exp_name, exp_config in experiments.items():
            result = run_experiment(exp_name, exp_config, data, ds_name)
            if result:
                all_results.append(result)
            torch.cuda.empty_cache()

    master_csv = "ablation_comparison.csv"
    write_comparison(all_results, master_csv)
    print(f"\nResults -> {master_csv}")
    return all_results


def write_comparison(results, filename):
    """Write master comparison CSV."""
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'Dataset', 'Experiment', 'Mode',
            'Accuracy_Mean', 'Accuracy_Std',
            'Precision_Mean', 'Precision_Std',
            'Recall_Mean', 'Recall_Std',
            'F1_Mean', 'F1_Std',
            'Comm_MB_Mean', 'Comm_MB_Std',
            'Stage1_Acc', 'Stage1_F1', 'Stage2_Acc', 'Stage2_F1',
            'Log_File'
        ])
        for r in results:
            def v(key, default=0):
                val = r.get(key, default)
                return val if val is not None else default

            w.writerow([
                r.get('dataset', ''),
                r.get('experiment', ''),
                r.get('mode', ''),
                f"{v('accuracy_mean', v('stage2_acc')):.6f}",
                f"{v('accuracy_std'):.6f}",
                f"{v('precision_mean'):.6f}",
                f"{v('precision_std'):.6f}",
                f"{v('recall_mean'):.6f}",
                f"{v('recall_std'):.6f}",
                f"{v('f1_mean', v('stage2_f1')):.6f}",
                f"{v('f1_std'):.6f}",
                f"{v('comm_mb_mean', v('total_comm_mb')):.2f}",
                f"{v('comm_mb_std'):.2f}",
                f"{v('stage1_acc'):.6f}" if r.get('stage1_acc') is not None else '',
                f"{v('stage1_f1'):.6f}" if r.get('stage1_f1') is not None else '',
                f"{v('stage2_acc'):.6f}" if r.get('stage2_acc') is not None else '',
                f"{v('stage2_f1'):.6f}" if r.get('stage2_f1') is not None else '',
                r.get('log_file', r.get('s2_log', '')),
            ])


def parse_args():
    parser = argparse.ArgumentParser(description="TSC-FL Ablation Experiments")
    parser.add_argument('--datasets', nargs='+', default=['CICIDS2017'],
                        choices=['CICIDS2017', 'UNSW-NB15', 'Nba-IoT'])
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'stage', 'stage1', 'stage2', 'partition'])
    parser.add_argument('--rounds', type=int, default=30,
                        help='通信轮数 (越少越快)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    BASE_CONFIG['communication_rounds'] = args.rounds

    exp_map = {
        'stage': ABLATION_STAGE,
        'stage1': ABLATION_STAGE1,
        'stage2': ABLATION_STAGE2,
        'partition': ABLATION_PARTITION,
        'all': ALL_ABLATIONS,
    }

    results = run_ablations(
        datasets=args.datasets,
        experiments=exp_map.get(args.experiment, ALL_ABLATIONS)
    )

    print(f"\n{'='*70}")
    print(f"DONE: {len(results)} experiments")
    print(f"Comparison: ablation_comparison.csv")
    print(f"{'='*70}")
