#!/usr/bin/env python3
"""Paired, seed-keyed ECG study inference and result tables."""

import argparse
import json
from pathlib import Path
from ssrl_ecg.experiment_analysis import analyse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.results_dir / "study.json").read_text())
    report = analyse(
        args.results_dir, config["seeds"], [c["name"] for c in config["conditions"]]
    )
    print(f"Coverage: {report['status']}. Paired tests require all declared seeds.")


if __name__ == "__main__":
    main()
