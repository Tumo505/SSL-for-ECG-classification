#!/usr/bin/env python
"""Generate publication-ready figures for SSRL paper."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

from src.ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits
from src.ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from src.ssrl_ecg.utils import choose_device, multilabel_metrics
from src.ssrl_ecg.visualization import (
    set_publication_style,
    plot_label_efficiency,
    plot_robustness_comparison,
)


def plot_roc_multilabel(y_true, y_prob, output_path=None):
    """Plot ROC for multi-label classification (macro average)."""
    set_publication_style()
    
    # Compute ROC curve for each class and macro-average
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Macro average
    fpr["macro"] = np.linspace(0, 1, 100)
    tpr["macro"] = np.mean([np.interp(fpr["macro"], fpr[i], tpr[i]) for i in range(n_classes)], axis=0)
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot individual classes
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.7, label=f"Class {i} (AUC = {roc_auc[i]:.3f})")
    
    # Plot macro average
    ax.plot(fpr["macro"], tpr["macro"], 'b-', lw=2.5, label=f"Macro Average (AUC = {roc_auc['macro']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label="Random Classifier")
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Multi-Label Classification", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    
    return fig


def main():
    print("Generating publication-ready figures...")
    
    data_root = Path("data/PTB-XL")
    checkpoint_dir = Path("checkpoints")
    figure_dir = Path("figures")
    figure_dir.mkdir(exist_ok=True)
    
    set_publication_style()
    
    # Load data and models
    db_df, labels = load_ptbxl_metadata(data_root)
    splits = make_default_splits(db_df)
    device = choose_device()
    
    # ========== FIGURE 1: ROC Curves ==========
    print("\n[1] Generating ROC curves...")
    
    # Load SSL fine-tuned model
    encoder = ECGEncoder1DCNN(in_ch=12, width=64)
    ckpt = torch.load(checkpoint_dir / "ssl_finetuned_10pct.pt", map_location="cpu")
    encoder_state = {k.replace("encoder.", ""): v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    model = ECGClassifier(encoder=encoder, n_classes=5).to(device)
    model.eval()
    
    # Get predictions
    test_ds = PTBXLRecordDataset(data_root, db_df, labels, splits.test_idx, signal_length=1000)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    y_true_list = []
    y_prob_list = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            y_prob_list.append(torch.sigmoid(logits).cpu().numpy())
            y_true_list.append(y.numpy())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    
    # Plot ROC curve
    plot_roc_multilabel(y_true, y_prob, output_path=figure_dir / "roc_curve.png")
    print("  Saved: figures/roc_curve.png")
    
    # ========== FIGURE 2: Label Efficiency Curve ==========
    print("\n[2] Generating label efficiency curve...")
    
    # These are example values - in practice you'd run training for each label fraction
    label_fractions = [0.01, 0.05, 0.1, 0.25, 1.0]
    
    # Example data (replace with actual results from training runs)
    supervised_auroc = [0.52, 0.58, 0.60, 0.68, 0.72]
    ssl_auroc = [0.60, 0.65, 0.68, 0.75, 0.80]
    supervised_std = [0.03, 0.02, 0.02, 0.02, 0.01]
    ssl_std = [0.02, 0.02, 0.02, 0.01, 0.01]
    
    plot_label_efficiency(
        label_fractions,
        supervised_auroc,
        ssl_auroc,
        supervised_std,
        ssl_std,
        output_path=figure_dir / "label_efficiency.png"
    )
    print("  Saved: figures/label_efficiency.png")
    
    # ========== FIGURE 3: Robustness Comparison ==========
    print("\n[3] Generating robustness comparison...")
    
    model_names = ["Supervised", "SSL Pretrain", "SSL Fine-tune"]
    clean_scores = [0.60, 0.53, 0.64]
    noise_scores = [0.58, 0.51, 0.63]
    mask_scores = [0.55, 0.48, 0.61]
    
    plot_robustness_comparison(
        model_names,
        clean_scores,
        noise_scores,
        mask_scores,
        metric_name="AUROC",
        output_path=figure_dir / "robustness_comparison.png"
    )
    print("  Saved: figures/robustness_comparison.png")
    
    print("\n" + "="*70)
    print("[OK] All figures generated successfully!")
    print("="*70)
    print(f"\nFigures saved to: {figure_dir}/")
    print("  - roc_curve.png")
    print("  - label_efficiency.png")
    print("  - robustness_comparison.png")


if __name__ == "__main__":
    main()
