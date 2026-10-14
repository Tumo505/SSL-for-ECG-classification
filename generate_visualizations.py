#!/usr/bin/env python3
"""Generate study figures from measured predictions and epoch histories."""
import argparse
import json
import os
from pathlib import Path
from ssrl_ecg.experiment_analysis import analyse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,default=Path('results/study'))
    args=p.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(Path('data_cache/matplotlib').resolve()))
    config=json.loads((args.study_dir/'study.json').read_text())
    report=analyse(args.study_dir,config['seeds'],[c['name'] for c in config['conditions']],plots=True)
    print(f"Measured figures generated; study status: {report['status']}")

if __name__=='__main__':
    main()
