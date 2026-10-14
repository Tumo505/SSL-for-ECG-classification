"""Post-hoc paired inference and joint patient/seed calibration uncertainty."""
import csv
import json
from pathlib import Path
import numpy as np
from ssrl_ecg.experiment import write_json
from ssrl_ecg.experiment_metrics import paired_comparison, holm_adjust, evaluate_predictions
from patient_uncertainty import weighted_metrics, interval


def calibration(y,p,w):
    w=np.atleast_2d(w).astype(float)
    total=w.sum(1)
    brier=(w@((p-y)**2).mean(1))/total
    ece=np.zeros(len(w))
    for c in range(y.shape[1]):
        for j in range(10):
            mask=(p[:,c]>=j/10)&((p[:,c]<(j+1)/10) if j<9 else (p[:,c]<=1))
            ece+=np.abs(w@((p[:,c]-y[:,c])*mask))/total/y.shape[1]
    return np.column_stack((brier,ece))


def main():
    root=Path('results/combined'); out=Path('results/extensions/analysis'); out.mkdir(parents=True,exist_ok=True)
    seeds=json.loads((root/'study.json').read_text())['seeds']
    pairs=[('simclr_domain_finetuned','simclr_generic_finetuned','f1_macro','downstream_policy'),
           ('byol_domain_finetuned','byol_generic_finetuned','f1_macro','downstream_policy'),
           ('simclr_domain_finetuned','simclr_generic_finetuned','auroc_macro','downstream_policy'),
           ('byol_domain_finetuned','byol_generic_finetuned','auroc_macro','downstream_policy')]
    for a,b in [('byol_domain','supervised'),('byol_domain_finetuned','supervised'),
                ('byol_domain_finetuned','byol_domain'),('simclr_domain_finetuned','simclr_generic_finetuned'),
                ('byol_domain_finetuned','byol_generic_finetuned')]:
        pairs += [(a,b,m,'calibration') for m in ('brier_macro','ece_macro')]
    names=sorted({n for a,b,_,_ in pairs for n in (a,b)})
    first=np.load(root/names[0]/f'seed_{seeds[0]}'/'test_predictions.npz')
    patients,inverse=np.unique(first['patient_ids'],return_inverse=True)
    rng=np.random.default_rng(20261012)
    reps=2000
    counts=rng.multinomial(len(patients),np.ones(len(patients))/len(patients),size=reps)
    seed_draws=rng.integers(0,len(seeds),(reps,len(seeds)))
    draws,points={},{}
    for name in names:
        values=[]; scores=[]
        for seed in seeds:
            a=np.load(root/name/f'seed_{seed}'/'test_predictions.npz')
            for key in ('labels','patient_ids','ecg_ids','classes'):
                if not np.array_equal(a[key],first[key]):raise ValueError('Unaligned predictions')
            y,p=a['labels'],a['probabilities']
            metrics=evaluate_predictions(y,p)['metrics']
            saved=json.loads((root/name/f'seed_{seed}'/'result.json').read_text())['test_fixed']['metrics']
            for m in ('f1_macro','auroc_macro','brier_macro','ece_macro'):
                if not np.isclose(metrics[m],saved[m],atol=1e-8):raise ValueError('Prediction metric mismatch')
            scores.append([metrics[m] for m in ('f1_macro','auroc_macro','brier_macro','ece_macro')])
            values.append(np.concatenate([np.column_stack((weighted_metrics(y,p,counts[i:i+64,inverse]),calibration(y,p,counts[i:i+64,inverse]))) for i in range(0,reps,64)]))
        samples=np.stack(values,axis=1)
        draws[name]=samples[np.arange(reps)[:,None],seed_draws].mean(1)
        points[name]=np.array(scores)
        print('Post-hoc uncertainty: '+name,flush=True)
    metric_names=('f1_macro','auroc_macro','brier_macro','ece_macro')
    rows=[]
    for a,b,m,family in pairs:
        i=metric_names.index(m)
        stat=paired_comparison(dict(zip(seeds,points[a][:,i])),dict(zip(seeds,points[b][:,i])))
        rows.append(dict(method_a=a,method_b=b,metric=m,family=family,analysis_timing='post_hoc',
                         mean_difference=stat['mean'],p_two_sided=stat['p_two_sided'],
                         seed_ci95=stat['ci95_mean'],joint_ci95=interval(draws[a][:,i]-draws[b][:,i])))
    # All metrics within each post-hoc family are corrected together.
    for family in ('downstream_policy','calibration'):
        selected=[r for r in rows if r['family']==family]
        for r,p in zip(selected,holm_adjust([r['p_two_sided'] for r in selected])):r['p_holm']=p
    write_json(out/'paired_inference.json',dict(status='complete',analysis_timing='post_hoc; original families unchanged',
                 repetitions=reps,n_patients=len(patients),comparisons=rows,
                 uncertainty='pointwise joint patient/seed intervals; conditional on observed training/validation subjects'))
    with (out/'paired_inference.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    timings=[]
    for name in json.loads((root/'study.json').read_text())['conditions']:
        name=name['name']
        rs=[json.loads((root/name/f'seed_{s}'/'result.json').read_text()) for s in seeds]
        timings.append(dict(condition=name,mean_pretrain_seconds=float(np.mean([r['pretrain_seconds'] for r in rs])),
                            mean_downstream_seconds=float(np.mean([r['downstream_seconds'] for r in rs])),
                            timing_scope='measured on this Mac; reused encoder pretraining excluded from downstream runs'))
    write_json(out/'compute_profile.json',timings)


if __name__=='__main__':main()
