"""Show the declared experiment coverage and measured training durations."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
for name in ('study','experiments'):
    folder=root/'results'/name
    if not (folder/'study.json').exists():
        continue
    config=json.loads((folder/'study.json').read_text())
    runs=[json.loads(p.read_text()) for p in folder.glob('*/seed_*/result.json')]
    expected=len(config['conditions'])*len(config['seeds'])
    print(f'{name}: {len(runs)}/{expected} completed; {config["pretrain_epochs"]} pretraining epochs; {config["downstream_epochs"]} downstream epochs')
    for condition in config['conditions']:
        records=[r for r in runs if r['condition']==condition['name']]
        if records:
            elapsed=[r['pretrain_seconds']+r['downstream_seconds'] for r in records]
            print(f"  {condition['name']}: {len(records)}/{len(config['seeds'])} seeds, mean recorded training time {sum(elapsed)/len(elapsed):.1f} s")
