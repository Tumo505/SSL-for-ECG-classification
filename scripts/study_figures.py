#!/usr/bin/env python3
"""Export complete study data and scientific figures from verified predictions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from ssrl_ecg.experiment_metrics import seed_summary

ROOT=Path(__file__).resolve().parents[1]/'results/combined'
s=json.loads((ROOT/'summary.json').read_text())
assert s['status']=='complete' and not s['smoke']
config=json.loads((ROOT/'study.json').read_text())
methods={}
for name,coverage in s['coverage'].items():
    assert len(coverage['completed'])==10 and not coverage['missing']
    runs=[json.loads((ROOT/name/f'seed_{seed}'/'result.json').read_text()) for seed in config['seeds']]
    first=runs[0]['test_fixed']['per_class']
    methods[name]={
        'tuned':{k:seed_summary([r['test_validation_tuned']['metrics'][k] for r in runs]) for k in runs[0]['test_validation_tuned']['metrics']},
        'per_class':{c:{k:seed_summary([r['test_fixed']['per_class'][c][k] for r in runs]) for k,v in metrics.items() if k!='support' and all(r['test_fixed']['per_class'][c][k] is not None for r in runs)} for c,metrics in first.items()},
        'support':{c:m['support'] for c,m in first.items()},
        'times':{k:seed_summary([r[k] for r in runs]) for k in ['pretrain_seconds','downstream_seconds']},
        'representation':{k:seed_summary([r['representation_validation'][k] for r in runs]) for k in ['mean_feature_std','effective_rank','n_features']}}
(ROOT/'analysis/manuscript_data.json').write_text(json.dumps(dict(summary=s,config=config,methods=methods),indent=2,allow_nan=False))
# Refresh the manuscript data consumed at the stable study location.
(ROOT.parent/'study/analysis/manuscript_data.json').write_text((ROOT/'analysis/manuscript_data.json').read_text())
plt.rcParams.update({'font.size':10,'savefig.bbox':'tight'})
def save(fig,name):
    for ext in ['png','pdf']:
        fig.savefig(ROOT/'analysis'/f'{name}.{ext}',dpi=200)
    plt.close(fig)
names=['supervised','simclr_generic','simclr_domain','byol_generic','byol_domain','clocs_cmsc','simclr_generic_finetuned','simclr_domain_finetuned','byol_generic_finetuned','byol_domain_finetuned','clocs_cmsc_finetuned']
labels=['Supervised','SimCLR generic frozen','SimCLR domain frozen','BYOL generic frozen','BYOL domain frozen','CLOCS shared frozen','SimCLR generic fine-tuned','SimCLR domain fine-tuned','BYOL generic fine-tuned','BYOL domain fine-tuned','CLOCS shared fine-tuned']
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,metric,title in zip(axes,['f1_macro','auroc_macro'],['Macro-F1 at threshold 0.5','Macro-AUROC']):
    for i,name in enumerate(names):
        v=s['summaries'][name][metric]; lo,hi=v['ci95_mean']
        ax.errorbar(v['mean'],i,xerr=[[v['mean']-lo],[hi-v['mean']]],fmt='o',color='black' if name=='supervised' else ('#0072B2' if 'finetuned' in name else '#D55E00'),capsize=3)
    ax.set(title=title,yticks=range(len(names)),yticklabels=labels)
    ax.grid(axis='x',alpha=.2)
axes[0].invert_yaxis()
fig.suptitle('Ten-seed means and pointwise 95% seed confidence intervals')
fig.tight_layout(); save(fig,'performance')
transforms=['frequency_warp','mixup','bandpass','cutmix','motion','channel_noise','temporal_dropout']
fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
for row,method in enumerate(['simclr','byol']):
    for col,metric in enumerate(['f1_macro','auroc_macro']):
        ax=axes[row,col]
        for i,t in enumerate(transforms):
            v=s['paired_comparisons'][method+'_domain__minus__'+method+'_without_'+t][metric];lo,hi=v['ci95_mean']
            ax.errorbar(v['mean'],i,xerr=[[v['mean']-lo],[hi-v['mean']]],fmt='o',capsize=3,color='#0072B2' if v['p_holm']<.05 else '#777777')
        ax.axvline(0,color='black',linewidth=.8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set(title=method.upper()+' '+('macro-F1' if col==0 else 'macro-AUROC'),yticks=range(7),yticklabels=[t.replace('_',' ') for t in transforms],xlabel='Full policy minus omitted transformation')
        ax.grid(axis='x',alpha=.2)
axes[0,0].invert_yaxis()
fig.suptitle('Conditional omission effects; blue denotes Holm p < 0.05\nPointwise 95% seed confidence intervals')
fig.tight_layout();save(fig,'augmentation_effects')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric in zip(axes,['f1_macro','auroc_macro']):
    for name in ['supervised','simclr_domain','byol_domain','simclr_domain_finetuned','byol_domain_finetuned']:
        values=np.array([[e['validation'][metric] for e in json.loads((ROOT/name/f'seed_{seed}'/'history.json').read_text()) if e['stage']=='downstream'] for seed in config['seeds']])
        x=np.arange(1,21); mean=values.mean(0);line=ax.plot(x,mean,label=name.replace('_',' '))[0]
        ax.fill_between(x,mean-values.std(0,ddof=1),mean+values.std(0,ddof=1),alpha=.12,color=line.get_color())
    ax.set(xlabel='Downstream epoch',ylabel='Validation '+metric.replace('_',' ')); ax.grid(alpha=.2)
axes[0].legend(fontsize=7)
fig.suptitle('Validation convergence; bands show sample SD across ten seeds')
fig.tight_layout();save(fig,'validation_convergence')
print('Exported manuscript data and three scientific figure sets')
