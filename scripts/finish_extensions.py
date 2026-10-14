"""Final paired inference for complete label-efficiency and transfer panels."""
import json
import hashlib
from pathlib import Path
import numpy as np
from ssrl_ecg.experiment import write_json
from ssrl_ecg.experiment_analysis import analyse
from ssrl_ecg.experiment_metrics import paired_comparison,holm_adjust,evaluate_predictions
from patient_uncertainty import analyse as patient_analyse

ROOT=Path(__file__).resolve().parents[1]


def panel(name,entries,pairs,seeds):
    root=ROOT/'results/extensions/analysis'/name;root.mkdir(parents=True,exist_ok=True)
    for condition,source in entries.items():
        target=root/condition
        if target.exists() and target.resolve()!=source.resolve():raise ValueError('Panel name collision')
        if not target.exists():target.symlink_to(source,target_is_directory=True)
        for seed in seeds:
            path=source/f'seed_{seed}'
            result=json.loads((path/'result.json').read_text())
            provenance=json.loads((path/'provenance.json').read_text())
            if result['smoke'] or result['status']!='complete' or result['fingerprint']!=provenance['fingerprint']:raise ValueError('Incomplete or unverified run')
            keys=('config','condition','seed','code_sha256','data') if 'data' in provenance else ('config','condition','seed','parent')
            actual=hashlib.sha256(json.dumps({k:provenance[k] for k in keys},sort_keys=True).encode()).hexdigest()
            if actual!=result['fingerprint']:raise ValueError('Fingerprint recomputation failed')
            lineage=path/'encoder_lineage.json'
            if lineage.exists():
                parent=json.loads(lineage.read_text())
                if hashlib.sha256(Path(parent['path']).read_bytes()).hexdigest()!=parent['sha256']:raise ValueError('Changed parent checkpoint')
            arrays=np.load(path/'test_predictions.npz')
            metrics=evaluate_predictions(arrays['labels'],arrays['probabilities'],classes=arrays['classes'])['metrics']
            for m in ('f1_macro','auroc_macro','brier_macro','ece_macro'):
                if not np.isclose(metrics[m],result['test_fixed']['metrics'][m],atol=1e-8):raise ValueError('Saved prediction mismatch')
    write_json(root/'study.json',dict(seeds=seeds,conditions=[dict(name=n) for n in entries],smoke=False))
    report=analyse(root,seeds,list(entries),plots=False)
    # These families were declared before the new runs, after seeing the original study.
    comparisons={}
    for a,b in pairs:
        results={n:{s:json.loads((entries[n]/f'seed_{s}'/'result.json').read_text()) for s in seeds} for n in (a,b)}
        comparisons[a+'__minus__'+b]={m:paired_comparison({s:r['test_fixed']['metrics'][m] for s,r in results[a].items()},
                                                       {s:r['test_fixed']['metrics'][m] for s,r in results[b].items()}) for m in ('f1_macro','auroc_macro')}
    for m in ('f1_macro','auroc_macro'):
        for pair,p in zip(comparisons,holm_adjust([r[m]['p_two_sided'] for r in comparisons.values()])):
            comparisons[pair][m].update(p_holm=p,family=name,analysis_timing='declared before new runs; original outcomes already observed')
    report['paired_comparisons']=comparisons
    write_json(root/'summary.json',report)
    patient_analyse(root,allow_undefined_auc=name=='external_transfer')
    return report


def main():
    base=ROOT/'results/extensions';config=json.loads((base/'study.json').read_text())
    if config['smoke']:raise ValueError('Cannot analyse smoke as scientific evidence')
    seeds=config['seeds']
    expected=[base/c['name']/f'seed_{s}'/'result.json' for c in config['conditions'] for s in seeds]
    if not all(p.exists() for p in expected):raise ValueError('All 290 declared runs must complete first')
    entries={};pairs=[]
    for framework in ('simclr','byol'):
        for suffix in ('','_finetuned'):
            matched=framework+'_generic_matched'+suffix;domain=framework+'_domain'+suffix;generic=framework+'_generic'+suffix
            for name in (matched,domain,generic):entries[name]=(base if 'matched' in name else ROOT/'results/combined')/name
            pairs.extend([(domain,matched),(matched,generic)])
    matching=panel('matched_generic',entries,pairs,seeds)
    entries={};pairs=[]
    for fraction in (1,5,10,20,50):
        methods={}
        for parent in ('supervised','simclr_generic','simclr_domain','byol_generic','byol_domain'):
            name=f'{parent}_labels_{fraction:02d}' if fraction!=10 else parent+('_finetuned' if parent!='supervised' else '')
            entries[name]=(ROOT/'results/combined' if fraction==10 else base)/name;methods[parent]=name
        pairs.extend([(methods[parent],methods['supervised']) for parent in methods if parent!='supervised'])
        pairs.extend([(methods[f+'_domain'],methods[f+'_generic']) for f in ('simclr','byol')])
    efficiency=panel('label_efficiency',entries,pairs,seeds)
    entries={c['name']:base/c['name'] for c in config['conditions'] if c['method']=='transfer'}
    pairs=[(n,'mitbih_supervised') for n in entries if n!='mitbih_supervised']
    pairs += [(f'mitbih_{f}_domain',f'mitbih_{f}_generic') for f in ('simclr','byol')]
    transfer=panel('external_transfer',entries,pairs,seeds)
    # Export plots from measured means and seed intervals; never select models by test outcomes.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for parent in ('supervised','simclr_generic','simclr_domain','byol_generic','byol_domain'):
        for ax,metric in zip(axes,('f1_macro','auroc_macro')):
            names=[f'{parent}_labels_{f:02d}' if f!=10 else parent+('_finetuned' if parent!='supervised' else '') for f in (1,5,10,20,50)]
            values=[efficiency['summaries'][n][metric] for n in names]
            ax.plot([1,5,10,20,50],[v['mean'] for v in values],marker='o',label=parent)
            ax.set(xlabel='Labelled training fraction (%)',ylabel=metric)
    axes[0].legend(fontsize=7);fig.tight_layout()
    fig.savefig(base/'analysis/label_efficiency.pdf');fig.savefig(base/'analysis/label_efficiency.png',dpi=180);plt.close(fig)
    write_json(base/'analysis/completion.json',dict(status='complete',completed_runs=len(expected),original_runs=280,
               total_runs=280+len(expected),panels=['matched_generic','label_efficiency','external_transfer'],
               scope='MIT-BIH binary annotation-window transfer; same previously evaluated seed-42 subject partition'))
    import runpy
    runpy.run_path(str(ROOT/'scripts/extension_figures.py'),run_name='__main__')
    print('Extension analyses complete',flush=True)


if __name__=='__main__':main()
