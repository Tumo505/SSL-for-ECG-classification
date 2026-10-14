"""Scientific plots and paired tables for the complete experiment panels."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'results/extensions/analysis'
PANELS=('matched_generic','label_efficiency','external_transfer')
reports={p:json.loads((ROOT/p/'summary.json').read_text()) for p in PANELS}
rows=[]
lines=['ECG experiment results','570 completed runs across 57 conditions; ten seeds per condition.','All run fingerprints, parent checkpoints, finite losses and saved test metrics verified.','Intervals are pointwise; decisions use panel- and metric-specific Holm adjustments.']
for panel,report in reports.items():
    uncertainty=json.loads((ROOT/panel/'analysis/patient_uncertainty.json').read_text())
    lines += ['',panel.replace('_',' ').title()]
    for name,values in report['summaries'].items():
        lines.append(f"{name}: F1 {values['f1_macro']['mean']:.4f} ± {values['f1_macro']['sample_sd']:.4f}; AUROC {values['auroc_macro']['mean']:.4f} ± {values['auroc_macro']['sample_sd']:.4f}")
    for pair,stats in report['paired_comparisons'].items():
        for metric,s in stats.items():
            ci=uncertainty['paired_intervals'][pair][metric]
            row=dict(panel=panel,contrast=pair,metric=metric,mean_difference=s['mean'],p_two_sided=s['p_two_sided'],p_holm=s['p_holm'],joint_ci_low=ci['patients_and_seeds_ci95'][0],joint_ci_high=ci['patients_and_seeds_ci95'][1],patient_ci_low=ci['patients_ci95'][0],patient_ci_high=ci['patients_ci95'][1])
            rows.append(row)
            lines.append(f"{pair} {metric}: difference {s['mean']:+.4f}; Holm p {s['p_holm']:.4f}; joint 95% CI [{row['joint_ci_low']:+.4f}, {row['joint_ci_high']:+.4f}]")
with (ROOT/'panel_comparisons.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines += ['','Interpretation','Matching generic gate events and RMS perturbation improves frozen generic representations. Domain SimCLR has lower frozen AUROC than matched generic SimCLR; BYOL retains frozen F1 and AUROC advantages. Full fine-tuning does not establish domain advantages over matched generic controls.','No label-efficiency or MIT-BIH contrast survives its panel Holm adjustment. The narrower 10% fine-tuning family retains its original results; this does not imply significance under the broader 30-contrast label-efficiency family.','MIT-BIH is corroborative binary annotation-window transfer on a previously evaluated fixed subject partition, not independent clinical confirmation or an AAMI beat benchmark.','Equal epochs are not equal compute. Matching RMS does not match clinical, morphological or spectral fidelity. Interaction and prospective clinical studies remain outside this experiment.']
(ROOT/'study_report.txt').write_text('\n'.join(lines)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#333333','#2878B5','#D95319','#56A38A','#9254A2']
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for i,parent in enumerate(('supervised','simclr_generic','simclr_domain','byol_generic','byol_domain')):
    fractions=[1,5,10,20,50]
    names=[f'{parent}_labels_{f:02d}' if f!=10 else parent+('_finetuned' if parent!='supervised' else '') for f in fractions]
    for ax,metric,title in zip(axes,('f1_macro','auroc_macro'),('Macro-F1','Macro-AUROC')):
        v=[reports['label_efficiency']['summaries'][n][metric] for n in names]
        ax.plot(fractions,[a['mean'] for a in v],'-o',label=parent.replace('_',' ').replace('simclr','SimCLR').replace('byol','BYOL').capitalize() if parent=='supervised' else parent.replace('_',' ').replace('simclr','SimCLR').replace('byol','BYOL'),color=colors[i],markersize=4)
        ax.fill_between(fractions,[a['ci95_mean'][0] for a in v],[a['ci95_mean'][1] for a in v],alpha=.09,color=colors[i])
        ax.set_xscale('log');ax.set_xticks(fractions,labels=fractions);ax.set(xlabel='Labelled training records (%)',ylabel=title);ax.grid(alpha=.15)
axes[0].legend(fontsize=8);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(ROOT/f'label_efficiency.{ext}',dpi=200)
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,6.3))
for ri,framework in enumerate(('simclr','byol')):
 for ci,metric in enumerate(('f1_macro','auroc_macro')):
    ax=axes[ri,ci]
    for j,suffix in enumerate(('','_finetuned')):
        names=[framework+'_generic'+suffix,framework+'_generic_matched'+suffix,framework+'_domain'+suffix]
        v=[reports['matched_generic']['summaries'][n][metric] for n in names];means=np.array([a['mean'] for a in v]);lo=np.array([a['ci95_mean'][0] for a in v]);hi=np.array([a['ci95_mean'][1] for a in v])
        ax.errorbar(np.arange(3)+(j-.5)*.1,means,yerr=[means-lo,hi-means],marker='o',capsize=3,label=['Frozen','Fine-tuned'][j])
    ax.set_xticks(range(3),labels=['Generic','Gate/RMS matched','Domain']);ax.set_title(framework.upper()+' '+('macro-F1' if ci==0 else 'macro-AUROC'));ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(ROOT/f'matched_generic.{ext}',dpi=200)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
names=list(reports['external_transfer']['summaries'])
for ax,metric in zip(axes,('f1_macro','auroc_macro')):
    v=[reports['external_transfer']['summaries'][n][metric] for n in names];means=np.array([a['mean'] for a in v]);lo=np.array([a['ci95_mean'][0] for a in v]);hi=np.array([a['ci95_mean'][1] for a in v]);ax.errorbar(range(5),means,yerr=[means-lo,hi-means],fmt='o',capsize=4)
    ax.set_xticks(range(5),labels=['Supervised','SimCLR\ngeneric','SimCLR\ndomain','BYOL\ngeneric','BYOL\ndomain']);ax.set_ylabel('Macro-F1' if metric=='f1_macro' else 'AUROC');ax.grid(alpha=.15)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(ROOT/f'external_transfer.{ext}',dpi=200)
plt.close(fig)
print('Exported scientific figures, paired comparison CSV and study report')
