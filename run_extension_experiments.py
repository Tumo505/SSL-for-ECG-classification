#!/usr/bin/env python3
"""Run locked generic matching, label efficiency and multi-seed transfer studies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import torch
from filelock import FileLock
sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
from matched_augmentations import MatchedGeneric
from transfer_study import prepare as prepare_transfer,run as run_transfer
from run_additional_experiments import checkpoint_identity
from ssrl_ecg import experiment
from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT
from ssrl_ecg.experiment_analysis import analyse
from ssrl_ecg.utils import choose_device

ROOT=Path(__file__).resolve().parent
PARENTS=('simclr_generic','simclr_domain','byol_generic','byol_domain')


def conditions():
    rows=[]
    for framework in ('simclr','byol'):
        name=framework+'_generic_matched'
        rows += [dict(name=name,method=framework,policy='generic_matched',family='matched_generic'),
                 dict(name=name+'_finetuned',method='finetune',parent=name,own_parent=True,freeze_encoder=False,family='matched_generic')]
    for fraction in (.01,.05,.20,.50):
        for parent in ('supervised',)+PARENTS:
            rows.append(dict(name=f'{parent}_labels_{int(fraction*100):02d}',method='supervised' if parent=='supervised' else 'finetune',
                             **({} if parent=='supervised' else dict(parent=parent,freeze_encoder=False)),
                             label_fraction=fraction,family='label_efficiency'))
    for parent in ('supervised',)+PARENTS:
        rows.append(dict(name='mitbih_'+parent,method='transfer',family='external_transfer',
                         **({} if parent=='supervised' else dict(parent=parent))))
    return rows


def implementation_hash():
    digest=hashlib.sha256()
    for path in [ROOT/'run_extension_experiments.py',ROOT/'run_additional_experiments.py',ROOT/'scripts/matched_augmentations.py',ROOT/'scripts/transfer_study.py']:
        digest.update(path.name.encode());digest.update(path.read_bytes())
    digest.update(experiment.source_hash().encode())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/extensions')
    p.add_argument('--device',default='mps',choices=('mps','cpu','auto'))
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--conditions',nargs='+')
    args=p.parse_args()
    config=json.loads((ROOT/'configs/experiment_study.json').read_text())
    config.update(data_root=str(DEFAULT_DATA_ROOT.resolve()),smoke=args.smoke,conditions=conditions(),
                  suite='label_efficiency_and_transfer',implementation_sha256=implementation_hash(),
                  original_results_already_observed=True,analysis_plan=dict(matched_generic_contrasts=8,label_efficiency_contrasts=30,external_transfer_contrasts=6,holm='separate per metric within each new panel',joint_patient_seed_bootstrap=2000,original_10percent_stratum='already observed'),analysis_timing='settings fixed before these new runs; follow-up study',
                  transfer_split_seed=42,transfer_epochs=10,
                  transfer_interpretation='existing seed-42 partition previously evaluated; corroborative transfer, not pristine external validation',
                  matching='same gate events and exact per-view RMS distance; not clinical equivalence')
    if args.smoke:config.update(seeds=[42],pretrain_epochs=1,downstream_epochs=2)
    if args.conditions:
        unknown=set(args.conditions)-{c['name'] for c in config['conditions']}
        if unknown:p.error('Unknown conditions: '+str(unknown))
        config['conditions']=[c for c in config['conditions'] if c['name'] in args.conditions]
    config['parent_encoders']={f'{parent}/seed_{seed}':checkpoint_identity(ROOT/'results/study'/parent/f'seed_{seed}'/'pretrained.pt') for parent in PARENTS for seed in config['seeds']}
    data_root=ROOT/'Data/mit-bih-arrhythmia-database-1.0.0'
    digest=hashlib.sha256()
    for path in sorted(data_root.iterdir()):
        if path.suffix in ('.hea','.dat','.atr'):digest.update(path.name.encode());digest.update(path.read_bytes())
    config['mitbih_data_sha256']=digest.hexdigest()
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    original_factory=experiment.make_augmentations
    experiment.make_augmentations=lambda policy,exclude,rate,seed: MatchedGeneric(rate,seed) if policy=='generic_matched' else original_factory(policy,exclude,rate,seed)
    torch.set_num_threads(config['cpu_threads']);device=choose_device(args.device)
    with FileLock(str(output/'study.lock'),timeout=0):
        manifest=output/'study.json'
        if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Changed protocol or source; choose a new directory')
        experiment.write_json(manifest,config)
        prepared=None;transfer=None
        names=[c['name'] for c in config['conditions']]
        for seed in config['seeds']:
            for c in config['conditions']:
                directory=output/c['name']/f'seed_{seed}'
                experiment.write_json(output/'status.json',dict(status='running',condition=c['name'],seed=seed,family=c['family']))
                try:
                    current={**config,'label_fraction':c.get('label_fraction',.1)}
                    parent=None
                    if c.get('parent'):
                        base=output if c.get('own_parent') else ROOT/'results/study'
                        parent=checkpoint_identity(base/c['parent']/f'seed_{seed}'/'pretrained.pt')
                        if not c.get('own_parent') and parent!=config['parent_encoders'][f"{c['parent']}/seed_{seed}"]:raise ValueError('Changed locked parent')
                        current.update(initial_encoder=parent['path'],initial_encoder_sha256=parent['sha256'],parent_fingerprint=parent['fingerprint'])
                    if c['method']=='transfer':
                        if transfer is None:transfer=prepare_transfer(data_root,args.smoke)
                        run_transfer(current,c,seed,directory,transfer,device,parent)
                    else:
                        if prepared is None:prepared=experiment.prepare_data(current)
                        experiment.run_condition(current,c,seed,directory,prepared,device)
                        if parent and not args.smoke:
                            old=np.load(Path(parent['path']).parent/'splits.npz');new=np.load(directory/'splits.npz')
                            for key in ('val_ecg_ids','test_ecg_ids'):
                                if not np.array_equal(old[key],new[key]):raise ValueError('Changed held-out partitions')
                    if parent:experiment.write_json(directory/'encoder_lineage.json',parent)
                except Exception as exc:
                    experiment.write_json(output/'status.json',dict(status='failed',condition=c['name'],seed=seed,error=str(exc),traceback=traceback.format_exc()))
                    raise
        report=analyse(output,config['seeds'],names,plots=False)
        experiment.write_json(output/'status.json',dict(status=report['status'],completed_runs=len(names)*len(config['seeds'])))


if __name__=='__main__':main()
