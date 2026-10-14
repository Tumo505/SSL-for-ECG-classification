#!/usr/bin/env python3
"""Start/resume the local MPS study and preserve its process/log details."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/experiments")
    )
    parser.add_argument("--suite", choices=("main", "additional", "extensions"), default="additional")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    process_file = output / "process.json"
    if process_file.exists():
        pid = json.loads(process_file.read_text())["pid"]
        # A locked study is the definitive guard; the process ID is informative.
        from filelock import FileLock, Timeout

        try:
            with FileLock(str(output / "study.lock"), timeout=0):
                pass
        except Timeout:
            raise SystemExit(f"Study is already running (recorded PID {pid})")
    command = [
        "/usr/bin/caffeinate",
        "-i",
        sys.executable,
        "-u",
        str(root / {"main": "run_experiments.py", "additional": "run_additional_experiments.py", "extensions": "run_extension_experiments.py"}[args.suite]),
        "--device",
        "mps",
        "--output-dir",
        str(output),
    ]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(root / "data_cache" / "matplotlib")
    env["PYTHONUNBUFFERED"] = "1"
    with (output / "training.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    process_file.write_text(
        json.dumps(
            {
                "pid": process.pid,
                "command": command,
                "log": str(output / "training.log"),
            },
            indent=2,
        )
    )
    print(
        json.dumps({"pid": process.pid, "log": str(output / "training.log")}, indent=2)
    )


if __name__ == "__main__":
    main()
