#!/usr/bin/env python3
"""Paired patient-cluster and joint patient/seed uncertainty from predictions.

Each sampled patient contributes every one of their test records. Identical
draws are used across methods. Joint intervals additionally sample training
seeds; they concern a mean over independently trained models, not an ensemble.
They are conditional on training/validation patients and fixed thresholds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ssrl_ecg.experiment import write_json


def weighted_metrics(labels, probabilities, weights, threshold=0.5, allow_undefined_auc=False):
    """Vectorised weighted F1/AUROC, with exact treatment of tied scores."""
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities)
    w = np.atleast_2d(weights).astype(np.float64)
    predicted = p >= threshold
    tp = w @ (y & predicted).astype(float)
    fp = w @ (~y & predicted).astype(float)
    fn = w @ (y & ~predicted).astype(float)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp), where=(2*tp+fp+fn)>0).mean(1)
    aucs = []
    for cls in range(y.shape[1]):
        order = np.argsort(p[:, cls], kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(p[order, cls]))+1]
        positive = np.add.reduceat(w[:, order]*y[order, cls], starts, axis=1)
        negative = np.add.reduceat(w[:, order]*(~y[order, cls]), starts, axis=1)
        denom = positive.sum(1)*negative.sum(1)
        if (denom == 0).any() and not allow_undefined_auc:
            raise ValueError("A bootstrap draw lacks a class; increase patient sample or report undefined AUROC")
        numerator=(positive*(np.cumsum(negative, axis=1)-0.5*negative)).sum(1)
        aucs.append(np.divide(numerator,denom,out=np.full_like(numerator,np.nan),where=denom>0))
    return np.column_stack((f1, np.mean(aucs, axis=0)))


def interval(values):
    if not np.isfinite(values).any():raise ValueError("No defined bootstrap estimates")
    return np.nanquantile(values, [.025, .975]).tolist()


def analyse(root, repetitions=2000, random_seed=20261012, batch_size=64, allow_undefined_auc=False):
    root = Path(root)
    summary = json.loads((root/"summary.json").read_text())
    seeds = summary["expected_seeds"]
    names = [n for n,c in summary["coverage"].items() if not c["missing"]]
    if summary["smoke"]:
        raise ValueError("Smoke predictions cannot establish scientific uncertainty")
    first = np.load(root/names[0]/f"seed_{seeds[0]}"/"test_predictions.npz")
    patients, inverse = np.unique(first["patient_ids"], return_inverse=True)
    n_patients = len(patients)
    rng = np.random.default_rng(random_seed)
    counts = rng.multinomial(n_patients, np.full(n_patients, 1/n_patients), size=repetitions)
    seed_draws = rng.integers(0, len(seeds), size=(repetitions, len(seeds)))
    draws, point = {}, {}
    for name in names:
        values, estimates = [], []
        for seed in seeds:
            arrays = np.load(root/name/f"seed_{seed}"/"test_predictions.npz")
            for key in ("ecg_ids", "patient_ids", "labels", "classes"):
                if not np.array_equal(arrays[key], first[key]):
                    raise ValueError(f"Unaligned paired test predictions: {name}/{seed}/{key}")
            probabilities = arrays["probabilities"]
            estimates.append(weighted_metrics(first["labels"], probabilities, np.ones(len(inverse)))[0])
            values.append(np.concatenate([
                weighted_metrics(first["labels"], probabilities, counts[start:start+batch_size, inverse], allow_undefined_auc=allow_undefined_auc)
                for start in range(0, repetitions, batch_size)
            ]))
        samples = np.stack(values, axis=1)  # bootstrap, trained seed, metric
        draws[name] = {"patients": samples.mean(1),
                       "patients_and_seeds": samples[np.arange(repetitions)[:,None], seed_draws].mean(1)}
        point[name] = np.mean(estimates, axis=0)
        for i, metric in enumerate(("f1_macro", "auroc_macro")):
            if not np.isclose(point[name][i], summary["summaries"][name][metric]["mean"], atol=1e-7):
                raise ValueError("Point estimate disagrees with saved seed summary")
        print(f"Patient uncertainty: {name} ({len(seeds)} seeds)", flush=True)
    output = dict(repetitions=repetitions, random_seed=random_seed, n_test_patients=n_patients,
                  n_test_records=len(inverse), seeds=seeds, status=summary["status"],
                  estimand="mean single-model metric over training seeds; fixed 0.5 threshold",
                  conditioning="fixed training and validation patients; population patient-cluster resampling",
                  method_intervals={}, paired_intervals={},
                  undefined_auc_policy="exclude undefined class-absent draws from AUROC intervals" if allow_undefined_auc else "raise",
                  valid_bootstrap_draws={name:{kind:[int(np.isfinite(samples[:,i]).sum()) for i in range(2)] for kind,samples in draws[name].items()} for name in names})
    for name in names:
        output["method_intervals"][name] = {
            metric: dict(mean=float(point[name][i]), **{kind+"_ci95": interval(samples[:,i])
                                                      for kind,samples in draws[name].items()})
            for i,metric in enumerate(("f1_macro", "auroc_macro"))
        }
    for pair in summary["paired_comparisons"]:
        a,b = pair.split("__minus__")
        output["paired_intervals"][pair] = {
            metric: dict(mean=float(point[a][i]-point[b][i]),
                         **{kind+"_ci95": interval(draws[a][kind][:,i]-draws[b][kind][:,i]) for kind in draws[a]})
            for i,metric in enumerate(("f1_macro", "auroc_macro"))
        }
    write_json(root/"analysis/patient_uncertainty.json", output)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, default=Path("results/study"))
    p.add_argument("--repetitions", type=int, default=2000)
    args = p.parse_args()
    if args.repetitions < 100:
        p.error("Use at least 100 bootstrap repetitions")
    analyse(args.study_dir, args.repetitions)
