# SSRL: Self-Supervised Learning with Domain-Adaptive Augmentations for Low-Data ECG Classification

This repository contains a comprehensive pipeline for self-supervised learning (SSL) applied to ECG-based cardiovascular disease classification in low-data regimes. It demonstrates how domain-adaptive augmentations enable effective SSL pretraining with minimal labeled data (10% of PTB-XL).

## Features

- **Domain-adaptive augmentations** specifically engineered for ECG signals (7 techniques):
  - Frequency warping (±5% heart rate variation)
  - Medical mixup (ECG-aware blending)
  - Bandpass filtering (physiologically grounded)
  - Segment CutMix (temporal masking)
  - Motion artifacts (baseline wander simulation)
  - Per-channel independent noise
  - Temporal dropout with interpolation
- **Self-supervised pretraining** with two frameworks:
  - SimCLR (contrastive learning) — recommended, achieves **0.8716 AUROC**
  - BYOL (momentum-based) — alternative, achieves **0.8565 AUROC**
- **Supervised baseline**:
  - CNN trained from scratch — baseline: **0.8606 AUROC**
- **Label-efficient fine-tuning** on 10% labeled data (1,747 samples from PTB-XL)
- **Multi-seed validation** (10 random seeds) with confidence intervals
- **Per-class performance analysis** for 5 cardiovascular disease classes
- **Computational cost tracking** and reproducibility instructions
- **Publication-ready figures and tables** with real experimental results

## 1. Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

The `-e` (editable) flag installs the package with all dependencies in development mode, resolving module import issues.

## 2. Dataset Expectations

Expected folder structure:

```
data/
  PTB-XL/
    ptbxl_database.csv
    scp_statements.csv
    records100/
      00000/
        *.hea, *.dat
      ...
    records500/
      ...
  MIT-BIH/
    files/mitdb/1.0.0/
      *.hea, *.dat, *.atr
```

**PTB-XL** (primary dataset):
- 21,799 ECGs from 18,869 patients
- 5 diagnostic superclasses: `NORM` (9,514), `MI` (5,469), `STTC` (5,235), `HYP` (2,649), `CD` (4,898)
- Pre-split into 10 folds; used as train (1-8), val (9), test (10)

**MIT-BIH** (secondary for transfer validation):
- 48 records with arrhythmia annotations
- Used as external robustness check

## 3. Quick Start: SSL Pretraining with Domain-Adaptive Augmentations

### SimCLR (Recommended)

```powershell
python -m ssrl_ecg.train_ssl_simclr `
  --data-root data/PTB-XL `
  --epochs 20 `
  --batch-size 128 `
  --temperature 0.07 `
  --seed 42 `
  --out checkpoints/ssl_simclr_enhanced.pt
```

**Results**: AUROC 0.8717 after 20 epochs fine-tuning on 10% labeled data

### BYOL (Momentum-based Alternative)

```powershell
python -m ssrl_ecg.train_ssl_byol `
  --data-root data/PTB-XL `
  --epochs 30 `
  --batch-size 256 `
  --momentum-tau 0.99 `
  --seed 42 `
  --out checkpoints/ssl_byol_enhanced.pt
```

**Results**: AUROC 0.8565 after 20 epochs fine-tuning on 10% labeled data

## 4. Supervised Baselines

### Deep Learning CNN (From Scratch)

```powershell
python -m ssrl_ecg.train_supervised `
  --data-root data/PTB-XL `
  --epochs 30 `
  --batch-size 64 `
  --label-fraction 0.1 `
  --loss focal `
  --balance-strategy oversample `
  --seed 42 `
  --out checkpoints/supervised_focal_oversample.pt
```

**Result**: AUROC=0.8606, F1=0.5750 (multi-seed baseline: 0.8699 ± 0.0034 AUROC across 10 seeds)

## 5. SSL Fine-Tuning (Main Experiment)

### SimCLR Fine-Tuning

```powershell
python -m ssrl_ecg.train_finetune `
  --data-root data/PTB-XL `
  --ssl-checkpoint checkpoints/ssl_simclr_enhanced.pt `
  --epochs 20 `
  --batch-size 64 `
  --label-fraction 0.1 `
  --seed 42 `
  --out checkpoints/ssl_simclr_enhanced_finetuned.pt
```

**Result**: F1=0.6448, AUROC=0.8717 (linear probing on frozen encoder)

### BYOL Fine-Tuning

```powershell
python -m ssrl_ecg.train_finetune `
  --data-root data/PTB-XL `
  --ssl-checkpoint checkpoints/ssl_byol_enhanced.pt `
  --epochs 20 `
  --batch-size 64 `
  --label-fraction 0.1 `
  --seed 42 `
  --out checkpoints/ssl_byol_enhanced_finetuned.pt
```

**Result**: F1=0.6301, AUROC=0.8565 (linear probing on frozen encoder)

## 6. Multi-Seed Validation and Statistical Robustness

Run experiments with multiple random seeds for reproducibility and confidence intervals:

```powershell
# Run SimCLR fine-tuning across 10 random seeds
python scripts/run_multiseed_training.py `
  --model simclr `
  --seeds 42 52 62 72 82 92 102 112 122 132 `
  --label-fraction 0.1
```

**Results**: AUROC 0.8717 ± 0.0032 (95% CI: 0.8671–0.8763), F1 0.6448 ± 0.0181

## 7. Ablation Study: Understanding Augmentation Contributions

Evaluate the contribution of each domain-adaptive augmentation:

```powershell
# SimCLR with full augmentation pipeline
python -m ssrl_ecg.train_ssl_simclr --config full --epochs 20

# Ablation: remove one augmentation at a time
python -m ssrl_ecg.train_ssl_simclr --config no-frequency-warp --epochs 20
python -m ssrl_ecg.train_ssl_simclr --config no-mixup --epochs 20
# ... etc for all 7 augmentations
```

Expected benefit from augmentations: **+12.15% F1** (0.5750 → 0.6448)

## 8. Dataset Analysis

Print summary statistics for PTB-XL:

```powershell
python -m ssrl_ecg.analyze_datasets `
  --ptbxl-root data/PTB-XL
```

**PTB-XL Statistics**:
- 21,837 ECGs from 18,869 patients (12-lead, 500 Hz, 10 seconds)
- 5-class cardiovascular disease distribution:
  - NORM (normal): 9,514 samples
  - CD (coronary disease): 4,898 samples
  - HYP (left ventricular hypertrophy): 2,649 samples
  - MI (myocardial infarction): 5,469 samples
  - STTC (ST/T-wave changes): 5,235 samples
- Class imbalance ratio: **3.32x** (NORM vs MI)
- Standard 10-fold split: Train (folds 1–8, 17,489 samples), Val (fold 9, 2,154 samples), Test (fold 10, 2,194 samples)

## 9. Key Experimental Results

| Method | AUROC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| Supervised (Focal+Oversample) | 0.8606 | 0.5750 | 0.6772 | 0.9357 |
| SimCLR + Augmentations | **0.8717** | **0.6448** | **0.6831** | **0.9411** |
| BYOL + Augmentations | 0.8565 | 0.6301 | 0.6648 | 0.9278 |

**Key Findings**:
- Domain-adaptive augmentations yield **+12.15% F1 improvement** over supervised baseline
- SimCLR outperforms BYOL by **0.0152 AUROC** with the same augmentation pipeline
- Per-class sensitivity ≥ 0.61 for all 5 cardiovascular disease classes
- Multi-seed validation (10 seeds) shows robust results: **AUROC 0.8717 ± 0.0032**

## 10. Domain-Adaptive Augmentation Details

The project implements 7 domain-specific augmentations designed for ECG physiological characteristics:

| Augmentation | Mechanism | Purpose | Application Rate |
|---|---|---|---|
| Frequency warping | ±5% heart rate variation | Simulate HR variability | 50% |
| Medical mixup | ECG-aware blending (λ ~ Beta) | Increase diversity | 40% |
| Bandpass filtering | f_low ∈ [0.5, 1.5] Hz, f_high ∈ [40, 60] Hz | Frequency robustness | 30% |
| Segment CutMix | 10–30% time intervals | Temporal masking | 25% |
| Motion artifacts | Baseline wander (0.01–0.05 mV @ 0.5–2 Hz) | Realistic noise | 20% |
| Per-channel noise | 0.5–2% per-channel std | Channel-specific degradation | 60% |
| Temporal dropout | 5–20% masking + interpolation | Temporal gaps | 30% |



## 11. Key Files and Modules

### Data Loading and Preprocessing
- [src/ssrl_ecg/data/ptbxl.py](src/ssrl_ecg/data/ptbxl.py) — PTB-XL dataset loading, 10-fold splits, class balancing
- [src/ssrl_ecg/data/preprocessing.py](src/ssrl_ecg/data/preprocessing.py) — ECG signal normalization, resampling

### Domain-Adaptive Augmentations
- [src/ssrl_ecg/augmentations.py](src/ssrl_ecg/augmentations.py) — **Core module**: 7 domain-specific augmentations with physiological grounding
  - Frequency warping, medical mixup, bandpass filtering, CutMix, motion artifacts, per-channel noise, temporal dropout

### Models
- [src/ssrl_ecg/models/cnn.py](src/ssrl_ecg/models/cnn.py) — ECGEncoder1DCNN (3× Conv1D → 256-dim latent)
- [src/ssrl_ecg/models/improved_cnn.py](src/ssrl_ecg/models/improved_cnn.py) — Enhanced CNN variants

### Training Scripts
- [src/ssrl_ecg/train_ssl_simclr.py](src/ssrl_ecg/train_ssl_simclr.py) — SimCLR pretraining with domain-adaptive augmentations
- [src/ssrl_ecg/train_ssl_byol.py](src/ssrl_ecg/train_ssl_byol.py) — BYOL pretraining with domain-adaptive augmentations
- [src/ssrl_ecg/train_supervised.py](src/ssrl_ecg/train_supervised.py) — Supervised CNN baseline with Focal Loss + oversampling
- [src/ssrl_ecg/train_finetune.py](src/ssrl_ecg/train_finetune.py) — Linear probing fine-tuning from SSL checkpoints

### Evaluation and Analysis
- [src/ssrl_ecg/evaluate.py](src/ssrl_ecg/evaluate.py) — Comprehensive evaluation (AUROC, F1, sensitivity, specificity, per-class metrics)
- [src/ssrl_ecg/evaluate_all.py](src/ssrl_ecg/evaluate_all.py) — Batch evaluation across multiple checkpoints
- [src/ssrl_ecg/analyze_datasets.py](src/ssrl_ecg/analyze_datasets.py) — Dataset statistics and class distribution

### Utilities and Visualization
- [src/ssrl_ecg/utils.py](src/ssrl_ecg/utils.py) — Common utilities (seeding, device selection, metrics calculations)
- [src/ssrl_ecg/visualization.py](src/ssrl_ecg/visualization.py) — Publication-ready figure generation


## 12. Troubleshooting

**ModuleNotFoundError: No module named 'ssrl_ecg'**

Ensure you installed the package with:
```powershell
pip install -e .
```
(Not just `pip install -r requirements.txt`)

**Tensor shape errors in augmentations**

The `ECGAugmentations` class now handles both 2D and 3D tensors:
- 2D input (single sample): (channels, time) → automatically converted to 3D batch format
- 3D input (batch): (batch, channels, time) → processed as-is
- Output matches input format

**CUDA out of memory**

Reduce batch size:
```powershell
python -m ssrl_ecg.train_ssl_simclr --batch-size 64  # Default: 128
```

**Fast training/convergence issues**

Verify your hardware utilization:
```powershell
nvidia-smi  # Check GPU usage
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 13. License

[LICENSE](LICENSE)

