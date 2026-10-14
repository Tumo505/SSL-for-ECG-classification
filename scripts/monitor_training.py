"""Report measured training progress without modifying training state."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from filelock import FileLock,Timeout


def snapshot(root):
    config=json.loads((root/'study.json').read_text())
    expected=len(config['conditions'])*len(config['seeds'])
    results=[json.loads(p.read_text()) for p in root.glob('*/seed_*/result.json')]
    completed=sum(r['status']=='complete' for r in results)
    status=json.loads((root/'status.json').read_text())
    try:
        with FileLock(str(root/'study.lock'),timeout=0):held=False
    except Timeout:
        held=True
    previous=root/'analysis/monitor_snapshot.json'
    old=json.loads(previous.read_text()) if previous.exists() else None
    report=dict(checked_at_unix=time.time(),completed_runs=completed,expected_runs=expected,
                percent=100*completed/expected,completed_since_previous_check=completed-old['completed_runs'] if old else None,
                study_lock_held=held,study_status=status,
                log_age_seconds=time.time()-(root/'training.log').stat().st_mtime if (root/'training.log').exists() else None,
                errors=[],current_epoch=None)
    for path in root.glob('*/seed_*/history.json'):
        for item in json.loads(path.read_text()):
            if not math.isfinite(item['loss']):
                report['errors'].append(f'Nonfinite loss: {path} {item["stage"]} epoch {item["epoch"]}')
    for path in root.glob('*/seed_*/status.json'):
        item=json.loads(path.read_text())
        if item.get('status')=='failed':report['errors'].append(dict(path=str(path),error=item.get('error')))
    if status.get('status')=='failed':report['errors'].append(status.get('error'))
    if status.get('condition') and status.get('seed') is not None:
        directory=root/status['condition']/f"seed_{status['seed']}"
        history=json.loads((directory/'history.json').read_text()) if (directory/'history.json').exists() else []
        if history:
            last=history[-1]
            recent=[item for item in history if item['stage']==last['stage']][-3:]
            report['current_epoch']=dict(stage=last['stage'],epoch=last['epoch'],
                total_epochs=(config.get('transfer_epochs',10) if status.get('family')=='external_transfer' else config['pretrain_epochs' if last['stage']=='pretrain' else 'downstream_epochs']),
                recent_losses=[item['loss'] for item in recent],
                recent_validation_macro_f1=[item['validation']['f1_macro'] for item in recent if 'validation' in item],
                approximate_epoch_seconds=statistics.median(item['seconds'] for item in recent))
        else:
            report['current_run_note']='No epoch is logged yet for the current condition and seed.'
    previous.parent.mkdir(parents=True,exist_ok=True)
    temporary=previous.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(previous)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,default=Path('results/experiments'))
    args=p.parse_args()
    print(json.dumps(snapshot(args.study_dir),indent=2))
