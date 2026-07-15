# TSC-FL: Two-Stage Classifier for Federated Learning

Source code for the paper: **"Efficient Federated Learning Framework for IoT Intrusion Detection: Two-Stage Hierarchical Traffic Classification Method"**

## Overview

TSC-FL is a two-stage federated learning framework for IoT intrusion detection that decouples benign traffic filtering from fine-grained attack classification:

- **Stage 1 - ResBTD**: ResNet-based Binary Traffic Detector for efficient benign filtering
- **Stage 2 - MamMTD**: Mamba-based Multi-class Malicious Traffic Detector for accurate attack-type identification

## Environment

| Package | Version |
|---------|---------|
| CUDA | 11.8 |
| Python | 3.10.13 |
| PyTorch | 2.4.0 |
| Pandas | 2.2.3 |
| Scikit-learn | 1.6.1 |
| Mamba-ssm | 1.2.0.post1 |
| Causal-conv1d | 1.2.0.post2 |

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch==2.4.0 pandas==2.2.3 scikit-learn==1.6.1
pip install mamba-ssm==1.2.0.post1 causal-conv1d==1.2.0.post2
```

**Note for mamba-ssm:** See [this guide](https://blog.csdn.net/yyywxk/article/details/146798627#t1) for detailed Windows installation steps.

## Datasets

Download the following public datasets and update paths in `config_ablation.py`:

| Dataset | Link |
|---------|------|
| CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| N-BaIoT | https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset |

## Usage

### Binary Classification (ResBTD)

```bash
python run_cicids2017_multi.py --task binary
python run_unsw_nb15_multi.py --task binary
python run_nbaiot_multi.py --task binary
```

### Multi-class Classification (MamMTD)

```bash
python run_cicids2017_multi.py --task multi
python run_unsw_nb15_multi.py --task multi
python run_nbaiot_multi.py --task multi
```

### Ablation Experiments

```bash
# All ablation experiments
python run_ablation.py --datasets CICIDS2017 --experiment all --rounds 30

# Specific categories
python run_ablation.py --datasets CICIDS2017 --experiment stage      # Single vs two-stage
python run_ablation.py --datasets CICIDS2017 --experiment stage1     # Stage 1 backbones
python run_ablation.py --datasets CICIDS2017 --experiment stage2     # Stage 2 classifiers
python run_ablation.py --datasets CICIDS2017 --experiment partition  # Non-IID partitions

# Multiple datasets
python run_ablation.py --datasets CICIDS2017 UNSW-NB15 Nba-IoT --experiment all --rounds 30
```

## Project Structure

```
TSC-FL/
├── run_cicids2017_multi.py    # CICIDS2017 training
├── run_unsw_nb15_multi.py     # UNSW-NB15 training
├── run_nbaiot_multi.py        # N-BaIoT training
├── run_ablation.py            # Ablation experiment runner
├── config_ablation.py         # Experiment configurations
├── MamMTD/                    # Multi-class classification
│   ├── multi_Classification.py
│   ├── train_Multi.py
│   ├── data_process/
│   └── ablation/
│       ├── trainer.py
│       ├── stage1_backbones.py
│       ├── stage2_classifiers.py
│       ├── data_partition.py
│       └── single_stage.py
└── ResBTD/                    # Binary classification
    └── binary_Classification.py
```

## Citation

```bibtex
@article{tscfl2025,
  title={Efficient Federated Learning Framework for IoT Intrusion Detection: Two-Stage Hierarchical Traffic Classification Method},
  author={Chen, Haolei and Dong, Jingnan and Shen, Shigen and Xu, Guangxia and Liu, Jun and Liu, Zhiquan},
  year={2025}
}
```
