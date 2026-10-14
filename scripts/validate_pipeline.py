#!/usr/bin/env python3
"""Execute real-data smoke paths and record their exact scope and outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ssrl_ecg.experiment import source_hash, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('mps','cpu','auto'),default='mps')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/'results/pipeline_validation'
    output.mkdir(parents=True,exist_ok=True)
    python=sys.executable
    common=['--device',args.device,'--epochs','1','--smoke']
    commands={
        'additional_suite':[python,'run_additional_experiments.py','--device',args.device,'--smoke','--output-dir',str(output/'additional_smoke')],
        'reconstruction':[python,'-m','ssrl_ecg.train_ssl',*common,'--out',str(output/'reconstruction.pt')],
        'resnet':[python,'-m','ssrl_ecg.train_resnet1d',*common,'--out',str(output/'resnet.pt')],
        'supervised_cli':[python,'-m','ssrl_ecg.train_supervised',*common,'--out',str(output/'supervised.pt')],
        'simclr_cli':[python,'-m','ssrl_ecg.train_ssl_simclr',*common,'--out',str(output/'simclr.pt')],
        'byol_cli':[python,'-m','ssrl_ecg.train_ssl_byol',*common,'--out',str(output/'byol.pt')],
        'contrastive_alias':[python,'-m','ssrl_ecg.train_ssl_contrastive',*common,'--out',str(output/'contrastive.pt')],
        'finetune_cli':[python,'-m','ssrl_ecg.train_finetune',*common,'--ssl-checkpoint','results/study/simclr_domain/seed_42/pretrained.pt','--out',str(output/'finetuned.pt')],
        'evaluate':[python,'-m','ssrl_ecg.evaluate','--device',args.device,'--smoke','--checkpoint','results/study/supervised/seed_42/best.pt'],
        'label_scarcity':[python,'-m','ssrl_ecg.label_scarcity_benchmark','--device',args.device,'--smoke','--epochs','1','--fractions','.05','--seeds','42','--output-dir',str(output/'label_scarcity')],
    }
    for method in ('rf','xgb','lgb'):
        commands[method]=[python,'-m','ssrl_ecg.train_traditional_ml','--smoke','--model',method,'--output-dir',str(output/'traditional')]
    dataset=root/'Data/mit-bih-arrhythmia-database-1.0.0'
    if dataset.exists():
        commands['mitbih_transfer']=[python,'-m','ssrl_ecg.transfer_mitbih','--mitbih-root',str(dataset),'--checkpoint','results/study/simclr_domain/seed_42/pretrained.pt','--use-ssl','--device',args.device,'--smoke','--finetune-epochs','2','--output-dir',str(output/'mitbih')]
    report=dict(source_sha256=source_hash(),device=args.device,scope='real-data smoke tests; not publication effect estimates',
                checks={},mitbih_data_available=dataset.exists())
    for name,command in commands.items():
        print('Validating '+name,flush=True)
        started=time.monotonic()
        with (output/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,cwd=root,env={**os.environ,'MPLCONFIGDIR':str(root/'data_cache/matplotlib')},stdout=stream,stderr=subprocess.STDOUT)
        report['checks'][name]=dict(command=command,exit_code=result.returncode,seconds=time.monotonic()-started)
        write_json(output/'validation.json',report)
        if result.returncode:
            raise SystemExit(f'{name} failed; read {output/(name+".log")}')
    if report['source_sha256']!=source_hash():
        raise RuntimeError('Source changed during validation')
    report['status']='passed'
    write_json(output/'validation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
