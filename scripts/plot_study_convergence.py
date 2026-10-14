#!/usr/bin/env python3
"""Plot objective-specific pretraining and validation curves across all seeds."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

root = Path(__file__).resolve().parents[1]/'results/study'
config = json.loads((root/'study.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(10,7), sharex='col')
for column, method in enumerate(['simclr','byol']):
    for policy in ['generic','domain']:
        name = method+'_'+policy
        histories = [json.loads((root/name/f'seed_{seed}'/'history.json').read_text())
                     for seed in config['seeds']]
        for row, stage, key in [(0,'pretrain','loss'),(1,'downstream','f1_macro')]:
            values=[]
            for history in histories:
                epochs = [v for v in history if v['stage'] == stage]
                values.append([v[key] if row == 0 else v['validation'][key] for v in epochs])
            values=np.asarray(values)
            if values.shape != (10,20):
                raise ValueError(f'Incomplete history for {name} {stage}: {values.shape}')
            mean=values.mean(0); sd=values.std(0,ddof=1); x=np.arange(1,21)
            ax=axes[row,column]
            line=ax.plot(x,mean,label=policy)[0]
            ax.fill_between(x,mean-sd,mean+sd,color=line.get_color(),alpha=.15)
            ax.set(xlabel=stage.capitalize()+' epoch', ylabel='SSL objective loss' if row == 0 else 'Validation macro-F1 at 0.5')
            ax.grid(alpha=.2);ax.legend()
    axes[0,column].set_title(method.upper()+' across ten seeds')
fig.suptitle('Mean and sample SD within each objective and protocol')
fig.tight_layout()
fig.savefig(root/'analysis/ssl_convergence.png',dpi=200)
plt.close(fig)
