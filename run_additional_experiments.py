#!/usr/bin/env python3
"""Run BYOL omissions, full fine-tuning and reference-architecture CLOCS.

The completed linear-probe study is immutable. Reused encoder files are
checksum-locked, and each downstream run records its parent fingerprint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

from filelock import FileLock
import numpy as np
import torch

from ssrl_ecg.augmentations import DOMAIN_TRANSFORMS
from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT
from ssrl_ecg.experiment import prepare_data, run_condition, write_json
from ssrl_ecg.experiment_analysis import analyse
from ssrl_ecg.utils import choose_device


def conditions():
    result = [dict(name="byol_without_"+t, method="byol", policy="domain", exclude=t)
              for t in DOMAIN_TRANSFORMS]
    result += [dict(name=m+"_finetuned", method="finetune", parent=m, freeze_encoder=False)
               for m in ("simclr_generic", "simclr_domain", "byol_generic", "byol_domain", "clocs_cmsc")]
    result += [dict(name="clocs_reference_supervised", method="supervised", backbone="clocs_reference"),
               dict(name="clocs_reference_cmsc", method="clocs", backbone="clocs_reference"),
               dict(name="clocs_reference_finetuned", method="finetune", backbone="clocs_reference",
                    parent="clocs_reference_cmsc", freeze_encoder=False)]
    return result


def checkpoint_identity(path):
    result = json.loads((path.parent/"result.json").read_text())
    provenance = json.loads((path.parent/"provenance.json").read_text())
    if result["status"] != "complete" or result["fingerprint"] != provenance["fingerprint"]:
        raise ValueError("Parent checkpoint is not from a completed verified run")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["fingerprint"] != result["fingerprint"]:
        raise ValueError("Encoder fingerprint does not match its parent result")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                fingerprint=result["fingerprint"], code_sha256=provenance["code_sha256"],
                pretrain_seconds=result["pretrain_seconds"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/experiments"))
    p.add_argument("--base-study", type=Path, default=Path("results/study"))
    p.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda", "auto"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--conditions", nargs="+")
    args = p.parse_args()
    config = json.loads(Path("configs/experiment_study.json").read_text())
    config.update(data_root=str(DEFAULT_DATA_ROOT.resolve()), smoke=args.smoke,
                  conditions=conditions(), suite="augmentation_and_finetuning")
    if args.smoke:
        config.update(seeds=[42], pretrain_epochs=1, downstream_epochs=2, pretrain_batch_size=64)
    if args.conditions:
        known = {c["name"] for c in config["conditions"]}
        if set(args.conditions)-known:
            p.error("Unknown conditions")
        config["conditions"] = [c for c in config["conditions"] if c["name"] in args.conditions]
    config["parent_encoders"] = {
        f"{c['parent']}/seed_{s}": checkpoint_identity(args.base_study/c["parent"]/f"seed_{s}"/"pretrained.pt")
        for c in config["conditions"] if c.get("parent") and c.get("backbone") != "clocs_reference"
        for s in config["seeds"]
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    device = choose_device(args.device)
    with FileLock(str(output/"study.lock"), timeout=0):
        manifest = output/"study.json"
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError("Study configuration changed; choose a new directory")
        write_json(manifest, config)
        prepared = {}
        names = [c["name"] for c in config["conditions"]]
        print(f"Running {len(names)*len(config['seeds'])} runs on {device}", flush=True)
        for seed in config["seeds"]:
            for c in config["conditions"]:
                rate = 500 if c.get("backbone") == "clocs_reference" else 100
                current = {**config, "sampling_rate": rate}
                parent = None
                if c.get("parent"):
                    base = output if c.get("backbone") == "clocs_reference" else args.base_study
                    parent = checkpoint_identity(base/c["parent"]/f"seed_{seed}"/"pretrained.pt")
                    if base == args.base_study and parent != config["parent_encoders"][f"{c['parent']}/seed_{seed}"]:
                        raise ValueError("A checksum-locked parent checkpoint changed")
                    current.update(initial_encoder=parent["path"], initial_encoder_sha256=parent["sha256"],
                                   parent_fingerprint=parent["fingerprint"])
                directory = output/c["name"]/f"seed_{seed}"
                write_json(output/"status.json", dict(status="running", condition=c["name"], seed=seed))
                try:
                    if rate not in prepared:
                        prepared[rate] = prepare_data(current)
                    result = run_condition(current, c, seed, directory, prepared[rate], device)
                    if parent:
                        write_json(directory/"encoder_lineage.json", parent)
                        if not args.smoke:
                            original = np.load(Path(parent["path"]).parent/"splits.npz")
                            new = np.load(directory/"splits.npz")
                            for key in ("labelled_ecg_ids", "val_ecg_ids", "test_ecg_ids"):
                                if not np.array_equal(original[key], new[key]):
                                    raise ValueError(f"Parent/child split mismatch: {key}")
                except Exception as exc:
                    write_json(output/"status.json", dict(status="failed", condition=c["name"], seed=seed,
                               error=str(exc), traceback=traceback.format_exc()))
                    raise
                analyse(output, config["seeds"], names, plots=False)
        report = analyse(output, config["seeds"], names)
        write_json(output/"status.json", dict(status=report["status"], completed_runs=len(names)*len(config["seeds"])))


if __name__ == "__main__":
    main()
