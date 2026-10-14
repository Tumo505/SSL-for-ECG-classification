#!/usr/bin/env python3
"""Run the patient-separated ECG study on CUDA, Apple MPS or CPU.

Usage: .venv/bin/python run_experiments.py --smoke --device mps
       .venv/bin/python run_experiments.py --device mps
Interrupted runs resume at the last completed epoch using saved RNG state.
"""

import argparse
import json
import os
from pathlib import Path
import traceback

import torch
from filelock import FileLock

from ssrl_ecg.augmentations import DOMAIN_TRANSFORMS
from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT
from ssrl_ecg.experiment import prepare_data, run_condition, write_json
from ssrl_ecg.experiment_analysis import analyse
from ssrl_ecg.utils import choose_device


def conditions():
    result = [
        {"name": "supervised", "method": "supervised"},
        {"name": "simclr_generic", "method": "simclr", "policy": "generic"},
        {"name": "simclr_domain", "method": "simclr", "policy": "domain"},
        {"name": "byol_generic", "method": "byol", "policy": "generic"},
        {"name": "byol_domain", "method": "byol", "policy": "domain"},
        {"name": "clocs_cmsc", "method": "clocs"},
    ]
    result += [
        {
            "name": "simclr_without_" + t,
            "method": "simclr",
            "policy": "domain",
            "exclude": t,
        }
        for t in DOMAIN_TRANSFORMS
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment_study.json")
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--downstream-epochs", type=int)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(config["cache_dir"]).resolve() / "matplotlib")
    )
    config["conditions"] = conditions()
    if args.data_root:
        config["data_root"] = str(args.data_root.resolve())
    else:
        config["data_root"] = str(DEFAULT_DATA_ROOT.resolve())
    config["smoke"] = args.smoke
    if args.smoke:
        config.update(
            pretrain_epochs=1, downstream_epochs=2, pretrain_batch_size=64, seeds=[42]
        )
    if args.seeds:
        config["seeds"] = args.seeds
    if args.conditions:
        known = {c["name"] for c in config["conditions"]}
        if set(args.conditions) - known:
            parser.error("Unknown condition names")
        config["conditions"] = [
            c for c in config["conditions"] if c["name"] in args.conditions
        ]
    if args.pretrain_epochs:
        config["pretrain_epochs"] = args.pretrain_epochs
    if args.downstream_epochs:
        config["downstream_epochs"] = args.downstream_epochs
    if len(set(config["seeds"])) != len(config["seeds"]):
        parser.error("Seeds must be unique")
    if any(
        config[k] < 1
        for k in (
            "pretrain_epochs",
            "downstream_epochs",
            "pretrain_batch_size",
            "downstream_batch_size",
        )
    ):
        parser.error("Budgets must be positive")
    output = args.output_dir or Path(
        "results/smoke" if args.smoke else "results/study"
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.analyse_only:
        study = json.loads((output / "study.json").read_text())
        analyse(output, study["seeds"], [c["name"] for c in study["conditions"]])
        return
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = choose_device(args.device)
    with FileLock(str(output / "study.lock"), timeout=0):
        study = output / "study.json"
        if study.exists() and json.loads(study.read_text()) != config:
            raise ValueError(
                "Existing study configuration differs; use another output directory"
            )
        write_json(study, config)
        prepared = prepare_data(config)
        names = [c["name"] for c in config["conditions"]]
        print(
            f"Running {len(names)*len(config['seeds'])} conditions/seeds on {device}. Output: {output.resolve()}",
            flush=True,
        )
        # Rotate method order by seed to reduce systematic wall-time ordering effects.
        for i, seed in enumerate(config["seeds"]):
            ordered = (
                config["conditions"][i % len(names) :]
                + config["conditions"][: i % len(names)]
            )
            for condition in ordered:
                run_dir = output / condition["name"] / f"seed_{seed}"
                write_json(
                    output / "status.json",
                    {"status": "running", "condition": condition["name"], "seed": seed},
                )
                try:
                    run_condition(config, condition, seed, run_dir, prepared, device)
                except Exception as exc:
                    write_json(
                        output / "status.json",
                        {
                            "status": "failed",
                            "condition": condition["name"],
                            "seed": seed,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise
                analyse(output, config["seeds"], names, plots=False)
        report = analyse(output, config["seeds"], names)
        write_json(
            output / "status.json",
            {
                "status": report["status"],
                "completed_runs": sum(
                    len(c["completed"]) for c in report["coverage"].values()
                ),
            },
        )
        print(f"Study {report['status']}. Read {output/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
