#!/usr/bin/env python3
"""Combine completed immutable suites and produce paired scientific outputs."""
import json
from pathlib import Path
from ssrl_ecg.experiment import write_json
from ssrl_ecg.experiment_analysis import analyse
from patient_uncertainty import analyse as patient_analyse

root=Path(__file__).resolve().parents[1]
sources=[root/'results/study',root/'results/experiments']
output=root/'results/combined'
output.mkdir(parents=True,exist_ok=True)
conditions=[]
seeds=None
lineage=[]
for source in sources:
    config=json.loads((source/'study.json').read_text())
    if config['smoke']:
        raise ValueError('Cannot combine smoke studies')
    if seeds is not None and seeds!=config['seeds']:
        raise ValueError('Seed sets differ across studies')
    seeds=config['seeds']
    for c in config['conditions']:
        condition=c['name']
        target=output/condition
        if target.exists() and target.resolve()!=(source/condition).resolve():
            raise ValueError('Condition name collision')
        if not target.exists():
            target.symlink_to(source/condition,target_is_directory=True)
        for seed in seeds:
            directory=source/condition/f'seed_{seed}'
            result=json.loads((directory/'result.json').read_text())
            identity=json.loads((directory/'provenance.json').read_text())
            if result['status']!='complete' or result['smoke'] or result['fingerprint']!=identity['fingerprint']:
                raise ValueError('Incomplete or mismatched study run')
            lineage.append(dict(condition=condition,seed=seed,fingerprint=result['fingerprint'],
                                code_sha256=identity['code_sha256'],data=identity['data']))
        conditions.append(c)
write_json(output/'study.json',dict(seeds=seeds,conditions=conditions,smoke=False,sources=[str(s) for s in sources]))
write_json(output/'analysis/run_manifest.json',lineage)
report=analyse(output,seeds,[c['name'] for c in conditions])
assert report['status']=='complete'
patient_analyse(output)
from study_report import report as study_report
study_report(output,output/"analysis")
print('Completed combined analysis: '+str(output))
