"""Resumable ten-seed MIT-BIH transfer with a fixed shared subject partition."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ssrl_ecg.data.mitbih import MITBIHDataset,record_splits,patient_id
from ssrl_ecg.models.cnn import ECGClassifier,ECGEncoder1DCNN
from ssrl_ecg.experiment import cpu_state,write_json,save_checkpoint,checkpoint_rng,restore_rng
from ssrl_ecg.experiment_metrics import evaluate_predictions
from ssrl_ecg.transfer_mitbih import predictions
from ssrl_ecg.utils import set_seed


def prepare(root,smoke=False):
    records=sorted(p.stem for p in root.glob('*.hea'))
    if len(records)!=48:raise ValueError('Expected all 48 MIT-BIH records')
    splits=record_splits(records,42)
    groups={k:{patient_id(r) for r in v} for k,v in splits.items()}
    if any(groups[a]&groups[b] for a,b in [('train','test'),('train','validation'),('validation','test')]):
        raise ValueError('Subject leakage')
    ds={k:MITBIHDataset(root,v,max_windows_per_record=4 if smoke else None) for k,v in splits.items()}
    return splits,ds


def run(config,condition,seed,output,prepared,device,parent=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    splits,ds=prepared
    identity=dict(config=config,condition=condition,seed=seed,parent=parent)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    identity['fingerprint']=fingerprint
    provenance=output/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())!=identity:raise ValueError('Changed transfer provenance')
    write_json(provenance,identity)
    if (output/'result.json').exists():return json.loads((output/'result.json').read_text())
    set_seed(seed)
    encoder=ECGEncoder1DCNN()
    if parent:
        if hashlib.sha256(Path(parent['path']).read_bytes()).hexdigest()!=parent['sha256']:raise ValueError('Changed parent encoder')
        encoder.load_state_dict(torch.load(parent['path'],map_location='cpu',weights_only=False)['encoder'])
    model=ECGClassifier(encoder,n_classes=1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    criterion=torch.nn.BCEWithLogitsLoss()
    generator=torch.Generator().manual_seed(seed+40000)
    loaders={k:DataLoader(d,batch_size=32,shuffle=k=='train',generator=generator if k=='train' else None) for k,d in ds.items()}
    checkpoint=output/'resume.pt';history=[];best=None;best_f1=-1;start=1;seconds_total=0.
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['fingerprint']!=fingerprint:raise ValueError('Changed resume fingerprint')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'],None,generator,device)
        start=state['epoch']+1;history=state['history'];best=state['best'];best_f1=state['best_f1'];seconds_total=state['seconds']
    budget=2 if config['smoke'] else 10
    for epoch in range(start,budget+1):
        model.train();started=time.monotonic();total=0.;n=0
        for x,y in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x.to(device)),y.to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite transfer loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(x);n+=len(x)
        y,p=predictions(model,loaders['validation'],device)
        metrics=evaluate_predictions(y,p,classes=('ectopic_window',))['metrics']
        if metrics['f1_macro']>best_f1:best_f1=metrics['f1_macro'];best=dict(model=cpu_state(model),epoch=epoch)
        seconds=time.monotonic()-started;seconds_total+=seconds
        history.append(dict(stage='downstream',epoch=epoch,loss=total/n,validation=metrics,seconds=seconds))
        save_checkpoint(checkpoint,dict(fingerprint=fingerprint,epoch=epoch,model=cpu_state(model),optimizer=optimizer.state_dict(),
                        rng=checkpoint_rng(None,generator,device),history=history,best=best,best_f1=best_f1,seconds=seconds_total))
        write_json(output/'history.json',history)
        print(f"{condition['name']} seed={seed} downstream {epoch}/{budget} val F1={metrics['f1_macro']:.4f} {seconds:.1f}s",flush=True)
    model.load_state_dict(best['model'])
    y,p=predictions(model,loaders['test'],device)
    # Subjects and window order are shared across every condition and training seed.
    np.savez_compressed(output/'test_predictions.npz',labels=y,probabilities=p,
                        patient_ids=np.array([patient_id(r) for r in ds['test'].record_ids]),record_ids=ds['test'].record_ids,
                        ecg_ids=np.arange(len(y)),classes=np.array(['ectopic_window']))
    save_checkpoint(output/'best.pt',best)
    result=dict(status='complete',condition=condition['name'],seed=seed,smoke=config['smoke'],fingerprint=fingerprint,
                best_epoch=best['epoch'],pretrain_seconds=0.,downstream_seconds=seconds_total,
                record_splits=splits,counts={k:len(d) for k,d in ds.items()},
                task='binary nonoverlapping ten-second annotation windows; not AAMI beat classification',
                split_seed=42,training_seed=seed,parent=parent,test_fixed=evaluate_predictions(y,p,classes=('ectopic_window',)))
    write_json(output/'result.json',result);write_json(output/'status.json',dict(status='complete'))
    return result
