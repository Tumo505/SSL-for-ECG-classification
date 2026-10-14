"""Prediction-level metrics, validation thresholds and paired seed inference."""

import itertools
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from ssrl_ecg.data.ptbxl import DIAGNOSTIC_CLASSES
from ssrl_ecg.utils import multilabel_metrics


def evaluate_predictions(y, p, thresholds=0.5, classes=DIAGNOSTIC_CLASSES):
    if y.shape != p.shape or y.shape[1] != len(classes):
        raise ValueError("Prediction/label shapes must match the declared classes")
    result = multilabel_metrics(y, p, threshold=thresholds)
    pred = p >= thresholds
    result["brier_macro"] = float(np.mean((p - y) ** 2))
    pr_values = [
                average_precision_score(y[:, i], p[:, i])
                for i in range(y.shape[1])
                if y[:, i].sum()
            ]
    result["prauc_macro"] = float(np.mean(pr_values)) if pr_values else None
    per_class = {}
    for i, name in enumerate(classes):
        target, predicted = y[:, i].astype(bool), pred[:, i]
        tp, tn = int((target & predicted).sum()), int((~target & ~predicted).sum())
        fp, fn = int((~target & predicted).sum()), int((target & ~predicted).sum())
        ece = 0.0
        for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (p[:, i] >= low) & (
                (p[:, i] <= high) if high == 1 else (p[:, i] < high)
            )
            if mask.any():
                ece += mask.mean() * abs(float(p[mask, i].mean() - y[mask, i].mean()))
        per_class[name] = dict(
            auroc=(
                float(roc_auc_score(target, p[:, i]))
                if len(np.unique(target)) == 2
                else None
            ),
            prauc=(
                float(average_precision_score(target, p[:, i]))
                if target.any()
                else None
            ),
            f1=float(f1_score(target, predicted, zero_division=0)),
            sensitivity=tp / max(1, tp + fn),
            specificity=tn / max(1, tn + fp),
            precision=tp / max(1, tp + fp),
            support=int(target.sum()),
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            brier=float(np.mean((p[:, i] - y[:, i]) ** 2)),
            ece=float(ece),
        )
    result["ece_macro"] = float(np.mean([v["ece"] for v in per_class.values()]))
    return {"metrics": result, "per_class": per_class, "n_records": len(y)}


def validation_thresholds(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return np.array(
        [
            grid[
                np.argmax(
                    [f1_score(y[:, i], p[:, i] >= t, zero_division=0) for t in grid]
                )
            ]
            for i in range(y.shape[1])
        ]
    )


def seed_summary(values, seed=2026, resamples=10000):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or len(x) == 0:
        raise ValueError("Expected finite, nonempty seed scores")
    result = {
        "n": len(x),
        "mean": float(x.mean()),
        "sample_sd": float(x.std(ddof=1)) if len(x) > 1 else None,
        "ci95_mean": None,
        "min": float(x.min()),
        "max": float(x.max()),
    }
    if len(x) > 1:
        rng = np.random.default_rng(seed)
        means = x[rng.integers(0, len(x), (resamples, len(x)))].mean(1)
        result["ci95_mean"] = np.quantile(means, [0.025, 0.975]).tolist()
    return result


def paired_comparison(a, b):
    """Inputs map seed IDs to scores; never infer pairing from list order."""
    if set(a) != set(b) or not a:
        raise ValueError("Paired methods must have the same nonempty seed IDs")
    seeds = sorted(a)
    differences = np.array([a[s] - b[s] for s in seeds])
    result = seed_summary(differences)
    result["seeds"] = seeds
    result["direction"] = "method_a_minus_method_b"
    result["p_two_sided"] = None
    result["cohens_dz"] = None
    if len(seeds) > 1:
        if len(seeds) <= 16:
            flips = np.array(list(itertools.product([-1, 1], repeat=len(seeds))))
            null = np.abs((flips * differences).mean(1))
            result["p_two_sided"] = float(
                np.mean(null >= abs(differences.mean()) - 1e-12)
            )
        else:
            rng = np.random.default_rng(2026)
            flips = rng.choice([-1, 1], size=(100000, len(seeds)))
            result["p_two_sided"] = float(
                (
                    1
                    + np.sum(
                        np.abs((flips * differences).mean(1))
                        >= abs(differences.mean()) - 1e-12
                    )
                )
                / 100001
            )
        sd = differences.std(ddof=1)
        result["cohens_dz"] = float(differences.mean() / sd) if sd > 0 else None
    return result


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    )
    return adjusted.tolist()
