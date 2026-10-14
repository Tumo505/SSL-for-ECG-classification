#!/usr/bin/env python3
"""Supervised-only multi-seed study using held-out test evaluation."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/supervised_multiseed")
    )
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "run_experiments.py"),
        "--conditions",
        "supervised",
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.quick:
        cmd += ["--seeds", "42", "52", "62", "--downstream-epochs", "5"]
    if args.data_root:
        cmd += ["--data-root", str(args.data_root)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
