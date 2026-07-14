# -*- coding: utf-8 -*-
"""
Unified Federated Learning Trainer for Ablation Experiments.
Integrated into TSC-FL project. Supports: single-stage, two-stage,
different backbones, different partitions.
"""

import time, csv, copy, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from MamMTD.train_Multi import fedavg_aggregation
from MamMTD.ablation.data_partition import get_partition

warnings.filterwarnings('ignore')


class AblationTrainer:
    """Handles federated training for all ablation configurations."""

    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_single_stage(self, X_train, X_test, y_train, y_test, num_classes, log_prefix=""):
        """Run single-stage (no binary filtering) federated training."""
        from MamMTD.ablation.stage2_classifiers import get_stage2_model

        cfg = self.config
        backbone = cfg.get('stage2_backbone', 'mamba')

        print(f"\n{'='*60}")
        print(f"[Single-Stage] Backbone: {backbone}, Partition: {cfg['partition']}")
        print(f"{'='*60}")

        client_data = get_partition(
            X_train, y_train,
            num_clients=cfg['num_clients'],
            strategy=cfg['partition'],
            alpha=cfg.get('alpha', 0.5)
        )

        client_sizes = [len(cx) for cx, _ in client_data]
        print(f"Client samples: mean={np.mean(client_sizes):.1f}, std={np.std(client_sizes):.1f}")

        global_model = get_stage2_model(
            backbone, input_dim=cfg['input_dim'], num_classes=num_classes
        ).to(self.device)

        param_size_mb = sum(p.nelement() * p.element_size() for p in global_model.parameters()) / (1024 ** 2)

        log_file = f"ablation_{log_prefix}single_{backbone}_{cfg['partition']}.csv"
        self._init_csv(log_file)

        all_metrics = []
        criterion = nn.CrossEntropyLoss()

        for round_idx in range(cfg['communication_rounds']):
            round_start = time.time()

            client_models = []
            client_times = []

            for cx, cy in client_data:
                model = type(global_model)(input_dim=cfg['input_dim'], num_classes=num_classes).to(self.device)
                model.load_state_dict(global_model.state_dict())

                cx_t = torch.tensor(cx, dtype=torch.float32)
                cy_t = torch.tensor(cy, dtype=torch.long)

                loader = DataLoader(
                    TensorDataset(cx_t, cy_t),
                    batch_size=max(2, min(cfg['batch_size'], len(cx) // 4 + 1)),
                    shuffle=True, drop_last=True,
                )

                opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
                t0 = time.time()
                self._train_client(model, loader, opt, criterion)
                client_times.append(time.time() - t0)
                client_models.append(model)

            global_model.load_state_dict(
                fedavg_aggregation(global_model, client_models, client_sizes)
            )

            metrics = self._evaluate(global_model, X_test, y_test)
            comm_mb = 2 * cfg['num_clients'] * param_size_mb
            round_time = time.time() - round_start

            all_metrics.append({
                'round': round_idx + 1,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'round_time': round_time,
                'comm_mb': comm_mb,
                'avg_client_time': np.mean(client_times)
            })

            print(f"Round {round_idx + 1:3d}: Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1']:.4f}, Comm={comm_mb:.2f}MB")

            self._write_round_csv(log_file, all_metrics[-1])

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        summary = self._compute_summary(all_metrics, param_size_mb, client_sizes)
        self._write_summary_csv(log_file, summary)
        summary['log_file'] = log_file
        return summary

    def run_two_stage(self, X_train, X_test, y_train_bin, y_test_bin,
                      attack_data, num_attack_classes, log_prefix=""):
        """Run two-stage federated training (binary filter + multi-class)."""
        from MamMTD.ablation.stage1_backbones import get_stage1_model
        from MamMTD.ablation.stage2_classifiers import get_stage2_model

        cfg = self.config
        attack_X, attack_y = attack_data

        print(f"\n{'='*60}")
        print(f"[Two-Stage] S1: {cfg['stage1_backbone']}, S2: {cfg['stage2_backbone']}, "
              f"Partition: {cfg['partition']}")
        print(f"{'='*60}")

        # ===== Stage 1: Binary Detection =====
        print("\n--- Stage 1: Binary Detection ---")
        stage1_model = get_stage1_model(cfg['stage1_backbone'], input_dim=cfg['input_dim']).to(self.device)

        s1_client_data = get_partition(
            X_train, y_train_bin,
            num_clients=cfg['num_clients'],
            strategy=cfg['partition'],
            alpha=cfg.get('alpha', 0.5)
        )

        s1_param_mb = sum(p.nelement() * p.element_size() for p in stage1_model.parameters()) / (1024 ** 2)
        log_s1 = f"ablation_{log_prefix}s1_{cfg['stage1_backbone']}_{cfg['partition']}.csv"
        self._init_csv(log_s1)

        s1_metrics = []
        criterion = nn.CrossEntropyLoss()

        for round_idx in range(cfg['communication_rounds']):
            round_start = time.time()
            client_models = []
            s1_client_sizes = [len(cx) for cx, _ in s1_client_data]

            for cx, cy in s1_client_data:
                model = type(stage1_model)(input_dim=cfg['input_dim']).to(self.device)
                model.load_state_dict(stage1_model.state_dict())
                cx_t, cy_t = torch.tensor(cx, dtype=torch.float32), torch.tensor(cy, dtype=torch.long)
                loader = DataLoader(TensorDataset(cx_t, cy_t),
                                    batch_size=max(2, min(cfg['batch_size'], len(cx) // 4 + 1)), shuffle=True, drop_last=True)
                opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
                self._train_client(model, loader, opt, criterion)
                client_models.append(model)

            stage1_model.load_state_dict(fedavg_aggregation(stage1_model, client_models, s1_client_sizes))
            m = self._evaluate(stage1_model, X_test, y_test_bin)
            s1_metrics.append(m)
            comm_mb = 2 * cfg['num_clients'] * s1_param_mb

            print(f"  S1 Round {round_idx + 1:3d}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

            self._write_round_csv(log_s1, {
                'round': round_idx + 1,
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1'],
                'round_time': time.time() - round_start,
                'comm_mb': comm_mb,
                'avg_client_time': 0
            })

        s1_summary = self._compute_summary(
            [{'round': i + 1, 'accuracy': m['accuracy'], 'precision': m['precision'],
              'recall': m['recall'], 'f1': m['f1'],
              'round_time': 0, 'comm_mb': 0, 'avg_client_time': 0}
             for i, m in enumerate(s1_metrics)],
            s1_param_mb, s1_client_sizes
        )
        self._write_summary_csv(log_s1, s1_summary)  # 写入s1文件FINAL SUMMARY

        # ===== Stage 2: Multi-class Attack Detection =====
        print("\n--- Stage 2: Multi-class Attack Detection ---")
        stage2_model = get_stage2_model(
            cfg['stage2_backbone'], input_dim=cfg['input_dim'],
            num_classes=num_attack_classes
        ).to(self.device)

        s2_client_data = get_partition(
            attack_X, attack_y,
            num_clients=cfg['num_clients'],
            strategy=cfg['partition'],
            alpha=cfg.get('alpha', 0.5)
        )

        s2_param_mb = sum(p.nelement() * p.element_size() for p in stage2_model.parameters()) / (1024 ** 2)
        log_s2 = f"ablation_{log_prefix}s2_{cfg['stage2_backbone']}_{cfg['partition']}.csv"
        self._init_csv(log_s2)

        s2_all_metrics = []

        for round_idx in range(cfg['communication_rounds']):
            round_start = time.time()
            client_models = []
            s2_client_sizes = [len(cx) for cx, _ in s2_client_data]
            client_times = []

            for cx, cy in s2_client_data:
                model = type(stage2_model)(input_dim=cfg['input_dim'],
                                           num_classes=num_attack_classes).to(self.device)
                model.load_state_dict(stage2_model.state_dict())
                cx_t, cy_t = torch.tensor(cx, dtype=torch.float32), torch.tensor(cy, dtype=torch.long)
                loader = DataLoader(TensorDataset(cx_t, cy_t),
                                    batch_size=max(2, min(cfg['batch_size'], len(cx) // 4 + 1)), shuffle=True, drop_last=True)
                opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
                t0 = time.time()
                self._train_client(model, loader, opt, criterion)
                client_times.append(time.time() - t0)
                client_models.append(model)

            stage2_model.load_state_dict(fedavg_aggregation(stage2_model, client_models, s2_client_sizes))

            s2_metrics = self._evaluate(stage2_model, attack_X, attack_y)
            comm_mb = 2 * cfg['num_clients'] * s2_param_mb
            round_time = time.time() - round_start

            s2_all_metrics.append({
                'round': round_idx + 1,
                'accuracy': s2_metrics['accuracy'],
                'precision': s2_metrics['precision'],
                'recall': s2_metrics['recall'],
                'f1': s2_metrics['f1'],
                'round_time': round_time,
                'comm_mb': comm_mb,
                'avg_client_time': np.mean(client_times)
            })

            print(f"  S2 Round {round_idx + 1:3d}: Acc={s2_metrics['accuracy']:.4f}, "
                  f"F1={s2_metrics['f1']:.4f}")

            self._write_round_csv(log_s2, s2_all_metrics[-1])
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        s2_summary = self._compute_summary(s2_all_metrics, s2_param_mb, s2_client_sizes)

        total_comm = s1_summary['comm_mb_mean'] + s2_summary['comm_mb_mean']
        combined = {
            'config': f"S1={cfg['stage1_backbone']}_S2={cfg['stage2_backbone']}_{cfg['partition']}",
            'stage1_acc': s1_summary['accuracy_mean'],
            'stage1_f1': s1_summary['f1_mean'],
            'stage2_acc': s2_summary['accuracy_mean'],
            'stage2_f1': s2_summary['f1_mean'],
            'accuracy_mean': s2_summary['accuracy_mean'],
            'accuracy_std': s2_summary['accuracy_std'],
            'precision_mean': s2_summary['precision_mean'],
            'precision_std': s2_summary['precision_std'],
            'recall_mean': s2_summary['recall_mean'],
            'recall_std': s2_summary['recall_std'],
            'f1_mean': s2_summary['f1_mean'],
            'f1_std': s2_summary['f1_std'],
            'comm_mb_mean': total_comm,
            'comm_mb_std': 0.0,
            'total_comm_mb': total_comm,
            's1_log': log_s1,
            's2_log': log_s2,
        }

        self._write_summary_csv(log_s2, s2_summary)
        return combined

    def _train_client(self, model, loader, optimizer, criterion):
        """Train a single client for local_epochs."""
        model.train()
        for _ in range(self.config.get('local_epochs', 5)):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

    def _evaluate(self, model, X_test, y_test):
        """Evaluate model on test data."""
        model.eval()
        X_np = X_test.values.astype(np.float32) if hasattr(X_test, 'values') else np.array(X_test, dtype=np.float32)
        y_np = y_test.values.astype(np.int64) if hasattr(y_test, 'values') else np.array(y_test, dtype=np.int64)

        X_t = torch.tensor(X_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.config['batch_size'], shuffle=False)

        all_preds = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(self.device)
                out = model(bx)
                all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())

        return {
            'accuracy': accuracy_score(y_np, all_preds),
            'precision': precision_score(y_np, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(y_np, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(y_np, all_preds, average='weighted', zero_division=0),
        }

    def _init_csv(self, filename):
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Round', 'Accuracy', 'Precision', 'Recall', 'F1',
                        'Round_Time_Sec', 'Comm_MB', 'Avg_Client_Time_Sec'])

    def _write_round_csv(self, filename, metrics):
        with open(filename, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                metrics['round'],
                f"{metrics['accuracy']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall']:.6f}",
                f"{metrics['f1']:.6f}",
                f"{metrics.get('round_time', 0):.2f}",
                f"{metrics.get('comm_mb', 0):.2f}",
                f"{metrics.get('avg_client_time', 0):.2f}",
            ])

    def _compute_summary(self, all_metrics, param_size_mb, client_sizes):
        """Compute mean +/- std for last 5 rounds."""
        last_n = min(5, len(all_metrics))
        recent = all_metrics[-last_n:]
        accs = [m['accuracy'] for m in recent]
        precs = [m['precision'] for m in recent]
        recs = [m['recall'] for m in recent]
        f1s = [m['f1'] for m in recent]
        comms = [m['comm_mb'] for m in recent]

        return {
            'accuracy_mean': np.mean(accs), 'accuracy_std': np.std(accs),
            'precision_mean': np.mean(precs), 'precision_std': np.std(precs),
            'recall_mean': np.mean(recs), 'recall_std': np.std(recs),
            'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
            'comm_mb_mean': np.mean(comms), 'comm_mb_std': np.std(comms),
            'model_size_mb': param_size_mb,
            'client_samples_mean': np.mean(client_sizes),
            'client_samples_std': np.std(client_sizes),
        }

    def _write_summary_csv(self, filename, summary):
        with open(filename, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([])
            w.writerow(['FINAL SUMMARY (Mean +/- Std, last 5 rounds)'])
            w.writerow(['Metric', 'Mean', 'Std'])
            for k in ['accuracy', 'precision', 'recall', 'f1', 'comm_mb']:
                w.writerow([k, f"{summary[f'{k}_mean']:.6f}", f"{summary[f'{k}_std']:.6f}"])
            w.writerow(['model_size_mb', f"{summary['model_size_mb']:.2f}", ''])
            w.writerow(['client_samples', f"{summary['client_samples_mean']:.1f}",
                       f"{summary['client_samples_std']:.1f}"])