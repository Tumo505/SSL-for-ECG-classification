"""Audit all completed scientific runs against provenance and saved predictions."""
import hashlib
import json
from pathlib import Path
import math
import numpy as np
from ssrl_ecg.experiment import write_json
from ssrl_ecg.experiment_metrics import evaluate_predictions
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_extension_experiments import implementation_hash

ROOT=Path(__file__).resolve().parents[1]


def main():
    verified=[];parents={};labelled={};transfer_splits=None
    for suite in ('study','experiments','extensions'):
        root=ROOT/'results'/suite;config=json.loads((root/'study.json').read_text())
        if config['smoke']:raise ValueError('Smoke suite cannot establish scientific completion')
        if suite=='extensions' and config['implementation_sha256']!=implementation_hash():raise ValueError('Training implementation changed')
        for c in config['conditions']:
            for seed in config['seeds']:
                path=root/c['name']/f'seed_{seed}'
                r=json.loads((path/'result.json').read_text());p=json.loads((path/'provenance.json').read_text())
                keys=('config','condition','seed','code_sha256','data') if 'data' in p else ('config','condition','seed','parent')
                fingerprint=hashlib.sha256(json.dumps({k:p[k] for k in keys},sort_keys=True).encode()).hexdigest()
                if r['status']!='complete' or r['smoke'] or r['fingerprint']!=fingerprint or p['fingerprint']!=fingerprint:raise ValueError('Invalid result or fingerprint '+str(path))
                history=json.loads((path/'history.json').read_text())
                if any(not math.isfinite(e['loss']) for e in history):raise ValueError('Nonfinite loss')
                epochs=r.get('record_splits') and 10 or config['downstream_epochs']
                if len([e for e in history if e['stage']=='downstream'])!=epochs:raise ValueError('Downstream budget mismatch')
                lineage=path/'encoder_lineage.json'
                if lineage.exists():
                    parent=json.loads(lineage.read_text());source=Path(parent['path'])
                    if str(source) not in parents:parents[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
                    if parents[str(source)]!=parent['sha256']:raise ValueError('Parent checksum mismatch')
                for pred_file,metric_key in [('test_predictions.npz','test_fixed'),('validation_predictions.npz',None)]:
                    if not (path/pred_file).exists():
                        if pred_file=='test_predictions.npz':raise ValueError('Missing test predictions')
                        continue
                    a=np.load(path/pred_file)
                    if not np.isfinite(a['probabilities']).all():raise ValueError('Nonfinite predictions')
                    fixed=evaluate_predictions(a['labels'],a['probabilities'],classes=a['classes'])['metrics']
                    if metric_key:
                        for m,value in fixed.items():
                            saved=r[metric_key]['metrics'][m]
                            if value is None or saved is None:
                                if value!=saved:raise ValueError('Undefined metric mismatch')
                            elif not np.isclose(value,saved,atol=1e-8):raise ValueError('Prediction metric mismatch '+m)
                if 'record_splits' in r:
                    if transfer_splits is None:transfer_splits=r['record_splits']
                    if transfer_splits!=r['record_splits']:raise ValueError('Transfer partitions differ')
                elif suite=='extensions':
                    split=np.load(path/'splits.npz')
                    key=(seed,c.get('label_fraction',.1))
                    if key in labelled and not np.array_equal(split['labelled_ecg_ids'],labelled[key]):raise ValueError('Unpaired labelled records')
                    labelled[key]=split['labelled_ecg_ids']
                verified.append(dict(suite=suite,condition=c['name'],seed=seed,fingerprint=fingerprint))
        print('Verified suite: '+suite,flush=True)
    assert len(verified)==570
    write_json(ROOT/'results/extensions/analysis/verification.json',dict(status='verified',runs=570,
        unique_parent_checkpoints=len(parents),all_fingerprints_recomputed=True,all_test_metrics_recomputed=True,
        all_losses_finite=True,downstream_budgets_verified=True,labelled_subsets_paired=True,
        transfer_subject_split_shared=True,extension_source_hash_verified=True,runs_manifest=verified))


if __name__=='__main__':main()
