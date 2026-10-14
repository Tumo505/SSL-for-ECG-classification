from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ssrl_ecg.data.ptbxl import PTBXLRecordDataset, load_ptbxl_metadata, make_default_splits, sample_labelled_indices


def extract_ecg_features(signal: np.ndarray, sampling_rate: int = 100) -> dict:
    """Extract handcrafted ECG features from a single 12-lead record."""
    features = {}

    # Per-lead statistics (simplified: use first lead for now)
    lead = signal[0] if signal.ndim > 1 else signal
    features["mean"] = float(np.mean(lead))
    features["std"] = float(np.std(lead))
    features["max"] = float(np.max(lead))
    features["min"] = float(np.min(lead))
    features["range"] = float(np.ptp(lead))

    # Derivative-based features (rough QRS detection proxy)
    diff = np.diff(lead)
    features["mean_gradient"] = float(np.mean(np.abs(diff)))
    features["max_gradient"] = float(np.max(np.abs(diff)))

    # Energy and entropy proxies
    features["energy"] = float(np.sum(lead**2))
    features["entropy"] = float(-np.sum((lead**2 / np.sum(lead**2 + 1e-10)) * np.log(lead**2 / np.sum(lead**2 + 1e-10) + 1e-10)))

    # Heart rate estimate (zero crossings as simple proxy)
    zero_crossings = np.sum(np.diff(np.sign(lead - np.mean(lead))) != 0)
    features["zero_crossings"] = float(zero_crossings)

    # RMS energy per lead (if multi-lead, compute for each)
    if signal.ndim > 1:
        for i in range(min(12, signal.shape[0])):
            l = signal[i]
            features[f"rms_lead_{i}"] = float(np.sqrt(np.mean(l**2)))
    else:
        features["rms"] = float(np.sqrt(np.mean(lead**2)))

    return features


def main():
    import json
    import os
    import sys
    import importlib.util
    from sklearn.multioutput import MultiOutputClassifier
    from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT
    from ssrl_ecg.experiment_metrics import evaluate_predictions
    from ssrl_ecg.experiment import write_json
    parser = argparse.ArgumentParser(description="Handcrafted features with RF, XGBoost or LightGBM.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=("rf", "xgb", "lgb"), default="rf")
    parser.add_argument("--output-dir", type=Path, default=Path("results/traditional_ml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    # Reuse the OpenMP library already shipped with Torch on this Mac.
    if sys.platform == "darwin" and args.model != "rf" and not os.environ.get("DYLD_LIBRARY_PATH"):
        location = Path(importlib.util.find_spec("torch").origin).parent/"lib"
        if (location/"libomp.dylib").exists():
            os.execve(sys.executable, [sys.executable, "-m", "ssrl_ecg.train_traditional_ml", *sys.argv[1:]],
                      {**os.environ, "DYLD_LIBRARY_PATH": str(location)})
    df, labels = load_ptbxl_metadata(args.data_root)
    splits = make_default_splits(df)
    eligible = lambda ids: ids[labels[ids].sum(1)>0]
    train = sample_labelled_indices(eligible(splits.train_idx), labels, args.label_fraction, args.seed)
    ids = dict(train=train, validation=eligible(splits.val_idx), test=eligible(splits.test_idx))
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        ids = {k:rng.choice(v, min(128,len(v)), replace=False) for k,v in ids.items()}
    features = {}
    for name, indices in ids.items():
        ds = PTBXLRecordDataset(args.data_root, df, labels, indices, return_labels=False)
        features[name] = pd.DataFrame([extract_ecg_features(ds[i].numpy()) for i in range(len(ds))]).fillna(0).to_numpy()
    scaler = StandardScaler().fit(features["train"])
    features = {k:scaler.transform(v) for k,v in features.items()}
    trees = 10 if args.smoke else 100
    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=trees,max_depth=15,random_state=args.seed,n_jobs=4)
    elif args.model == "xgb":
        from xgboost import XGBClassifier
        model = MultiOutputClassifier(XGBClassifier(n_estimators=trees,max_depth=5,learning_rate=.1,random_state=args.seed,n_jobs=4))
    else:
        from lightgbm import LGBMClassifier
        model = MultiOutputClassifier(LGBMClassifier(n_estimators=trees,max_depth=5,random_state=args.seed,n_jobs=4,verbosity=-1))
    model.fit(features["train"], labels[ids["train"]])
    output = args.output_dir/args.model
    output.mkdir(parents=True, exist_ok=True)
    result = dict(model=args.model,seed=args.seed,smoke=args.smoke,counts={k:len(v) for k,v in ids.items()})
    for name in ("validation", "test"):
        raw = model.predict_proba(features[name])
        classes = model.classes_ if args.model == "rf" else [m.classes_ for m in model.estimators_]
        if isinstance(raw, list):
            probabilities = np.column_stack([prob[:,np.flatnonzero(cls==1)[0]] if 1 in cls else np.zeros(len(prob))
                                             for prob,cls in zip(raw,classes)])
        else:
            probabilities = raw
        result[name] = evaluate_predictions(labels[ids[name]], probabilities)
        np.savez_compressed(output/(name+"_predictions.npz"), ecg_ids=df.iloc[ids[name]].ecg_id.to_numpy(),
                            patient_ids=df.iloc[ids[name]].patient_id.to_numpy(),labels=labels[ids[name]],probabilities=probabilities)
    write_json(output/"result.json", result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
