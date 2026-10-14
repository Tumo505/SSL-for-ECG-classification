"""Generate symmetric tables, paired intervals and diagnostic plots from real runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ssrl_ecg.experiment_metrics import seed_summary, paired_comparison, holm_adjust
from ssrl_ecg.experiment import write_json

METRICS = (
    "f1_macro",
    "auroc_macro",
    "prauc_macro",
    "sensitivity_micro",
    "specificity_micro",
    "brier_macro",
    "ece_macro",
)


def analyse(output_dir, expected_seeds, expected_conditions, plots=True):
    root = Path(output_dir)
    grouped = {name: {} for name in expected_conditions}
    for path in root.glob("*/seed_*/result.json"):
        result = json.loads(path.read_text())
        if result["condition"] in grouped:
            grouped[result["condition"]][result["seed"]] = result
    summaries = {}
    coverage = {}
    for name, results in grouped.items():
        coverage[name] = {
            "completed": sorted(results),
            "missing": sorted(set(expected_seeds) - set(results)),
        }
        if results:
            summaries[name] = {
                m: seed_summary(
                    [r["test_fixed"]["metrics"][m] for r in results.values()]
                )
                for m in METRICS
            }
    comparisons = {}
    pairs = [
        ("simclr_domain", "simclr_generic"),
        ("byol_domain", "byol_generic"),
        ("simclr_domain", "supervised"),
        ("byol_domain", "supervised"),
        ("simclr_domain", "byol_domain"),
        ("simclr_domain", "clocs_cmsc"),
    ]
    pairs += [
        ("simclr_domain", name)
        for name in expected_conditions
        if name.startswith("simclr_without_")
    ]
    pairs += [("byol_domain", name) for name in expected_conditions if name.startswith("byol_without_")]
    for name in expected_conditions:
        if name.endswith("_finetuned") and not name.startswith("clocs_reference"):
            pairs += [(name, "supervised"), (name, name.removesuffix("_finetuned"))]
    pairs += [("clocs_reference_cmsc", "clocs_reference_supervised"),
              ("clocs_reference_finetuned", "clocs_reference_supervised"),
              ("clocs_reference_finetuned", "clocs_reference_cmsc")]
    for a, b in pairs:
        if a not in grouped or b not in grouped:
            continue
        # Partial unpaired coverage is reported, never silently intersected.
        if set(grouped[a]) != set(expected_seeds) or set(grouped[b]) != set(
            expected_seeds
        ):
            continue
        comparisons[a + "__minus__" + b] = {
            m: paired_comparison(
                {s: r["test_fixed"]["metrics"][m] for s, r in grouped[a].items()},
                {s: r["test_fixed"]["metrics"][m] for s, r in grouped[b].items()},
            )
            for m in ("f1_macro", "auroc_macro")
        }
    def family_for(pair):
        if "without_" in pair:
            return "byol_ablation" if pair.startswith("byol") else "ablation"
        if "clocs_reference" in pair:
            return "reference_architecture"
        if "finetuned" in pair:
            return "finetuning"
        return "headline"

    for family in ("headline", "ablation", "byol_ablation", "finetuning", "reference_architecture"):
        keys = [k for k in comparisons if family_for(k) == family]
        for metric in ("f1_macro", "auroc_macro"):
            valid = [
                k for k in keys if comparisons[k][metric]["p_two_sided"] is not None
            ]
            if valid:
                adjusted = holm_adjust(
                    [comparisons[k][metric]["p_two_sided"] for k in valid]
                )
                for k, p in zip(valid, adjusted):
                    comparisons[k][metric].update(
                        p_holm=p,
                        family=family,
                        metric_role="primary" if metric == "f1_macro" else "secondary",
                    )
    report = {
        "status": (
            "complete"
            if all(not c["missing"] for c in coverage.values())
            else "incomplete"
        ),
        "expected_seeds": expected_seeds,
        "coverage": coverage,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "uncertainty": "bootstrap CI of seed mean; finite test-patient uncertainty is separate",
        "primary_metric": "macro-F1 at threshold 0.5",
        "secondary_metric": "macro-AUROC",
        "smoke": any(r["smoke"] for rs in grouped.values() for r in rs.values()),
    }
    write_json(root / "summary.json", report)
    with (root / "table5.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "metric",
                "n_seeds",
                "mean",
                "sample_sd",
                "ci95_mean_lower",
                "ci95_mean_upper",
            ]
        )
        for name, metrics in summaries.items():
            for metric, s in metrics.items():
                writer.writerow(
                    [
                        name,
                        metric,
                        s["n"],
                        s["mean"],
                        s["sample_sd"],
                        *(s["ci95_mean"] or [None, None]),
                    ]
                )
    with (root / "seed_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "seed",
                "best_epoch",
                "pretrain_seconds",
                "downstream_seconds",
                *METRICS,
            ]
        )
        for name, results in grouped.items():
            for seed, r in sorted(results.items()):
                writer.writerow(
                    [
                        name,
                        seed,
                        r["best_epoch"],
                        r["pretrain_seconds"],
                        r["downstream_seconds"],
                        *[r["test_fixed"]["metrics"][m] for m in METRICS],
                    ]
                )
    with (root / "per_class.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "seed",
                "class",
                "support",
                "auroc",
                "prauc",
                "f1",
                "sensitivity",
                "specificity",
                "precision",
                "brier",
                "ece",
            ]
        )
        for name, results in grouped.items():
            for seed, r in sorted(results.items()):
                for cls, v in r["test_fixed"]["per_class"].items():
                    writer.writerow(
                        [
                            name,
                            seed,
                            cls,
                            *[
                                v[m]
                                for m in (
                                    "support",
                                    "auroc",
                                    "prauc",
                                    "f1",
                                    "sensitivity",
                                    "specificity",
                                    "precision",
                                    "brier",
                                    "ece",
                                )
                            ],
                        ]
                    )
    if plots and summaries:
        make_plots(root, grouped)
    return report


def make_plots(root, grouped):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, rs in grouped.items():
        if not rs:
            continue
        histories = []
        for seed in rs:
            histories.append(
                json.loads((root / name / f"seed_{seed}" / "history.json").read_text())
            )
        for ax, key in zip(axes, ("f1_macro", "auroc_macro")):
            runs = [
                [e["validation"][key] for e in h if e["stage"] == "downstream"]
                for h in histories
            ]
            if len({len(v) for v in runs}) != 1:
                continue
            values = np.asarray(runs)
            mean = values.mean(0)
            x = np.arange(1, len(mean) + 1)
            line = ax.plot(x, mean, label=name)[0]
            if len(values) > 1:
                ax.fill_between(
                    x,
                    mean - values.std(0, ddof=1),
                    mean + values.std(0, ddof=1),
                    color=line.get_color(),
                    alpha=0.1,
                )
            ax.set(xlabel="Downstream epoch", ylabel="Validation " + key)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(root / "convergence.png", dpi=160)
    plt.close(fig)
    for name in ("supervised", "byol_domain", "simclr_domain"):
        for seed in grouped.get(name, {}):
            path = root / name / f"seed_{seed}"
            arrays = np.load(path / "test_predictions.npz")
            y, p = arrays["labels"], arrays["probabilities"]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
            for i, cls in enumerate(arrays["classes"]):
                conf, acc = [], []
                for low in np.arange(0, 1, 0.1):
                    mask = (p[:, i] >= low) & (p[:, i] < low + 0.1)
                    if mask.any():
                        conf.append(p[mask, i].mean())
                        acc.append(y[mask, i].mean())
                ax.plot(conf, acc, marker=".", label=str(cls))
            ax.set(
                xlabel="Mean predicted probability",
                ylabel="Observed positive fraction",
                title=f"{name}, seed {seed}",
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(path / "reliability.png", dpi=140)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.output_dir / "study.json").read_text())
    analyse(args.output_dir, config["seeds"], [c["name"] for c in config["conditions"]])


if __name__ == "__main__":
    main()
