#!/usr/bin/env python3
"""Refresh study analyses without touching the running training protocol."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from filelock import FileLock, Timeout
from ssrl_ecg.experiment_metrics import evaluate_predictions, seed_summary


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def write_table(path, header, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def summary_row(method, metric, values, expected):
    stats = seed_summary(values)
    return [method, metric, stats["n"], expected, stats["mean"],
            stats["sample_sd"], *(stats["ci95_mean"] or [None, None])]


def extension_report(root, output, config):
    """Report mixed PTB-XL/transfer runs without inventing absent diagnostics."""
    import hashlib
    names=[c['name'] for c in config['conditions']]
    coverage={};rows=[]
    for name in names:
        completed=[];values={m:[] for m in ('f1_macro','auroc_macro','brier_macro','ece_macro')}
        for seed in config['seeds']:
            path=root/name/f'seed_{seed}'
            if not (path/'result.json').exists():continue
            result=json.loads((path/'result.json').read_text());provenance=json.loads((path/'provenance.json').read_text())
            keys=('config','condition','seed','code_sha256','data') if 'data' in provenance else ('config','condition','seed','parent')
            fingerprint=hashlib.sha256(json.dumps({k:provenance[k] for k in keys},sort_keys=True).encode()).hexdigest()
            if result['status']!='complete' or result['smoke'] or result['fingerprint']!=fingerprint or provenance['fingerprint']!=fingerprint:
                raise ValueError('Invalid completed run '+str(path))
            a=np.load(path/'test_predictions.npz')
            computed=evaluate_predictions(a['labels'],a['probabilities'],classes=a['classes'])['metrics']
            for metric in values:
                if not np.isclose(computed[metric],result['test_fixed']['metrics'][metric],atol=1e-8):raise ValueError('Prediction metric mismatch')
                values[metric].append(computed[metric])
            completed.append(seed)
        coverage[name]={'completed':completed,'missing':sorted(set(config['seeds'])-set(completed))}
        for metric,vs in values.items():
            if vs:rows.append(summary_row(name,metric,vs,len(config['seeds'])))
    try:
        with FileLock(str(root/'study.lock'),timeout=0):running=False
    except Timeout:running=True
    log=root/'training.log'
    health=dict(completed_runs=sum(len(v['completed']) for v in coverage.values()),expected_runs=len(names)*len(config['seeds']),
                complete=all(not v['missing'] for v in coverage.values()),coverage=coverage,study_lock_held=running,
                log_age_seconds=time.time()-log.stat().st_mtime if log.exists() else None)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'progress.json',health)
    write_table(output/'seed_metrics.csv',['condition','metric','n_seeds','expected_seeds','mean','sample_sd','ci95_seed_mean_lower','ci95_seed_mean_upper'],rows)
    (output/'study_report.txt').write_text(
        'ECG EXPERIMENT RESULTS\n'
        +f"Completed runs: {health['completed_runs']}/{health['expected_runs']}.\n"
        +'PTB-XL augmentation matching and four labelled fractions use 20 downstream epochs. MIT-BIH binary annotation-window transfer uses ten epochs on a shared previously evaluated subject partition.\n'
        +'Saved fixed-threshold metrics and provenance fingerprints are recomputed. Transfer does not provide validation-tuned thresholds or representation diagnostics; these are not inferred from other tasks.\n'
        +'Paired panel tests, pointwise patient/joint intervals and scientific interpretations: results/extensions/analysis/study_report.txt and panel_comparisons.csv.\n')
    return health


def report(root, output):
    config = json.loads((root / "study.json").read_text())
    if config.get("suite")=="label_efficiency_and_transfer":
        return extension_report(root,output,config)
    seeds = set(config["seeds"])
    names = [c["name"] for c in config["conditions"]]
    grouped = {name: {} for name in names}
    source_hashes = set()
    for path in sorted(root.glob("*/seed_*/result.json")):
        result = json.loads(path.read_text())
        if result["condition"] not in grouped or result["seed"] not in seeds:
            continue
        if result["status"] != "complete" or result["smoke"]:
            raise ValueError(f"Not a completed publication run: {path}")
        provenance = json.loads((path.parent / "provenance.json").read_text())
        if provenance["fingerprint"] != result["fingerprint"]:
            raise ValueError(f"Result provenance mismatch: {path}")
        source_hashes.add(provenance["code_sha256"])
        grouped[result["condition"]][result["seed"]] = result
    if len(source_hashes) > 1:
        manifest = root/'analysis/run_manifest.json'
        if not config.get('sources') or not manifest.exists():
            raise ValueError("Multiple training source hashes require an explicit combined run manifest")
        recorded = {(r['condition'],r['seed']):(r['fingerprint'],r['code_sha256'])
                    for r in json.loads(manifest.read_text())}
        for name,rs in grouped.items():
            for seed,result in rs.items():
                provenance=json.loads((root/name/f'seed_{seed}'/'provenance.json').read_text())
                if recorded.get((name,seed))!=(result['fingerprint'],provenance['code_sha256']):
                    raise ValueError('Combined manifest does not match actual run provenance')
    output.mkdir(parents=True, exist_ok=True)
    coverage = {name: {"completed": sorted(rs), "missing": sorted(seeds-set(rs))}
                for name, rs in grouped.items()}
    complete = all(not item["missing"] for item in coverage.values())
    try:
        with FileLock(str(root / "study.lock"), timeout=0):
            running = False
    except Timeout:
        running = True
    log = root / "training.log"
    health = {
        "study_lock_held": running,
        "log_age_seconds": time.time()-log.stat().st_mtime if log.exists() else None,
        "training_status": json.loads((root / "status.json").read_text()) if (root/"status.json").exists() else {"status": "analysis"},
        "completed_runs": sum(len(rs) for rs in grouped.values()),
        "expected_runs": len(names)*len(seeds),
        "complete": complete,
        "coverage": coverage,
    }
    save_json(output / "progress.json", health)
    header = ["method", "metric", "n_seeds", "expected_seeds", "mean",
              "sample_sd", "ci95_seed_mean_lower", "ci95_seed_mean_upper"]
    baseline, thresholds, per_class, representations = [], [], [], []
    diagnostics = {}
    for name, rs in grouped.items():
        if not rs:
            continue
        if name == "supervised":
            for metric in next(iter(rs.values()))["test_fixed"]["metrics"]:
                baseline.append(summary_row(name, metric,
                    [r["test_fixed"]["metrics"][metric] for r in rs.values()], len(seeds)))
        for setting in ("test_fixed", "test_validation_tuned"):
            for metric in ("f1_macro", "auroc_macro", "brier_macro", "ece_macro"):
                thresholds.append(summary_row(name, setting+"_"+metric,
                    [r[setting]["metrics"][metric] for r in rs.values()], len(seeds)))
        for cls, metrics in next(iter(rs.values()))["test_fixed"]["per_class"].items():
            supports = {r["test_fixed"]["per_class"][cls]["support"] for r in rs.values()}
            if len(supports) != 1:
                raise ValueError(f"Test support changed across seeds: {name} {cls}")
            for metric in ("f1", "auroc", "prauc", "sensitivity", "specificity", "precision"):
                values = [r["test_fixed"]["per_class"][cls][metric] for r in rs.values()]
                if all(v is not None for v in values):
                    per_class.append([cls, next(iter(supports)),
                        *summary_row(name, metric, values, len(seeds))])
        for seed, r in sorted(rs.items()):
            d = r["representation_validation"]
            representations.append([name, seed, d["mean_feature_std"],
                                    d["effective_rank"], d["n_features"]])
            arrays = np.load(root/name/f"seed_{seed}"/"test_predictions.npz")
            p = arrays["probabilities"]
            for setting, threshold in (("test_fixed", 0.5),
                                       ("test_validation_tuned", np.asarray(r["validation_thresholds"]))):
                recomputed = evaluate_predictions(arrays["labels"], p, threshold)
                for metric in ("f1_macro", "auroc_macro", "brier_macro", "ece_macro"):
                    if not np.isclose(recomputed["metrics"][metric],
                                      r[setting]["metrics"][metric], atol=1e-8, rtol=1e-7):
                        raise ValueError(f"Prediction/metric mismatch: {name} {seed} {setting} {metric}")
            diagnostics[f"{name}/seed_{seed}"] = {
                "fixed_threshold_f1": r["test_fixed"]["metrics"]["f1_macro"],
                "validation_tuned_threshold_f1": r["test_validation_tuned"]["metrics"]["f1_macro"],
                "validation_thresholds": r["validation_thresholds"],
                "probability_quantiles_by_class": np.quantile(p, [0.1, 0.5, 0.9], axis=0).tolist(),
                "representation_validation": d,
            }
    write_table(output/"table4_baseline.csv", header, baseline)
    write_table(output/"threshold_calibration.csv", header, thresholds)
    write_table(output/"table6_per_class.csv", ["class", "positive_support", *header], per_class)
    write_table(output/"representation_diagnostics.csv",
                ["method", "seed", "validation_mean_feature_std", "validation_effective_rank", "feature_count"], representations)
    save_json(output/"diagnostics.json", diagnostics)
    text = ["ECG STUDY RESULTS", "",
            f"Completed runs: {health['completed_runs']}/{health['expected_runs']}.",
            "Status: "+("Complete." if complete else "Training in progress; final inference pending."), "",
            "PTB-XL v1.0.3 contains 21,799 recordings from 18,869 patients. Folds 1-8 train, 9 validate and 10 test.",
            "SSL uses 17,418 training records; diagnostic eligibility leaves 17,084, with exactly 1,708 labelled records per seed. Validation/test counts are 2,146/2,158.",
            "The shared backbone uses six convolution blocks, per-record/per-lead normalization and 100 Hz waveforms. Fixed 0.5 F1 is primary; validation-only thresholds are secondary.",
            "Confidence intervals in the seed tables describe independently trained models. Patient-cluster and joint patient/seed intervals are stored in patient_uncertainty.json.",
            "The shared-backbone CLOCS CMSC condition and the native 500 Hz single-lead reference-architecture conditions are distinct benchmarks."]
    if complete:
        summary = json.loads((root/"summary.json").read_text())
        if summary["status"] != "complete":
            raise ValueError("Run final paired analysis before producing the report")
        text += ["", "MEASURED RESULTS",
                 "Every declared condition has ten completed seeds."]
        for name in grouped:
            values = summary["summaries"][name]
            text.append(f"{name}: macro-F1 {values['f1_macro']['mean']:.4f}; macro-AUROC {values['auroc_macro']['mean']:.4f}.")
        rows = []
        for pair, metrics in summary["paired_comparisons"].items():
            for metric, stats in metrics.items():
                rows.append([pair, metric, stats["n"], stats["mean"],
                             *stats["ci95_mean"], stats["p_two_sided"],
                             stats["p_holm"], stats["cohens_dz"], stats["family"]])
                text.append(f"{pair}, {metric}: paired difference {stats['mean']:+.4f}, "
                            f"95% CI [{stats['ci95_mean'][0]:.4f}, {stats['ci95_mean'][1]:.4f}], "
                            f"Holm-adjusted p {stats['p_holm']:.4f}.")
        write_table(output/"paired_comparisons.csv",
                    ["comparison", "metric", "n_seeds", "paired_mean_difference",
                     "ci95_lower", "ci95_upper", "p_exact", "p_holm", "cohens_dz", "family"], rows)
        text += ["", "INTERPRETATION",
                 "The signed paired differences and intervals report gains and deficits under the declared protocol. Holm-adjusted tests and pointwise bootstrap intervals answer different uncertainty questions.",
                 "Omission effects are conditional on the remaining transformation library; they do not identify interactions or establish physiological fidelity.",
                 "Validation-only thresholds are secondary operating-point analyses. They do not change AUROC or establish calibration. Brier/ECE and reliability curves report calibration separately.",
                 "Matched epochs do not equalize FLOPs or elapsed time. Reference-architecture CLOCS uses single-lead 500 Hz windows and has its own architecture-matched supervised control.",
                 "Manuscript: artifacts/Deep Learning Indaba.docx."]
        uncertainty=output/'patient_uncertainty.json'
        if uncertainty.exists():
            patient=json.loads(uncertainty.read_text())
            text += ["", "PATIENT AND SEED UNCERTAINTY"]
            for pair,metrics in patient['paired_intervals'].items():
                for metric,values in metrics.items():
                    low,high=values['patients_and_seeds_ci95']
                    text.append(f"{pair}, {metric}: joint patient and seed 95% CI [{low:+.4f}, {high:+.4f}].")
    temporary = output/"study_report.txt.tmp"
    temporary.write_text("\n".join(text)+"\n")
    temporary.replace(output/"study_report.txt")
    return health


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=Path("results/study"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    state = report(args.study_dir, args.output_dir or args.study_dir/"analysis")
    print(json.dumps({k: state[k] for k in ("completed_runs", "expected_runs", "complete", "study_lock_held", "log_age_seconds")}, indent=2))
