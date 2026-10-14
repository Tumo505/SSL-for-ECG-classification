"""Read and verify saved predictions for reproducible model comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from ssrl_ecg.experiment_metrics import evaluate_predictions


def main(per_class=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir',type=Path,default=Path('results/study'))
    parser.add_argument('--conditions',nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    summary=json.loads((args.study_dir/'summary.json').read_text())
    if summary['status']!='complete' or summary['smoke']:
        raise ValueError('Scientific comparisons require a complete non-smoke study')
    names=args.conditions or list(summary['summaries'])
    result={}
    for name in names:
        if name not in summary['summaries']:
            raise ValueError('Unknown condition '+name)
        verified=[]
        for seed in summary['expected_seeds']:
            directory=args.study_dir/name/f'seed_{seed}'
            run=json.loads((directory/'result.json').read_text())
            arrays=np.load(directory/'test_predictions.npz')
            recomputed=evaluate_predictions(arrays['labels'],arrays['probabilities'])
            for metric,value in recomputed['metrics'].items():
                if not np.isclose(value,run['test_fixed']['metrics'][metric],atol=1e-7):
                    raise ValueError('Saved result disagrees with its predictions')
            verified.append(dict(seed=seed,**recomputed['per_class']) if per_class else dict(seed=seed,**recomputed['metrics']))
        result[name]=dict(seed_summary=summary['summaries'][name],verified_predictions=verified)
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
