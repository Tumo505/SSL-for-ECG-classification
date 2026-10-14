"""Checks for matching, weighted calibration and interrupted transfer training."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from matched_augmentations import MatchedGeneric
from analyse_extensions import calibration
from ssrl_ecg.experiment import checkpoint_rng,restore_rng
from ssrl_ecg.experiment_metrics import evaluate_predictions
import transfer_study


def test_matched_magnitude_and_rng_replay():
    torch.set_num_threads(1)
    x=torch.randn(8,12,1000,generator=torch.Generator().manual_seed(9))
    augmentor=MatchedGeneric(seed=19)
    views=augmentor(x)
    assert len(augmentor.recorded)==18
    assert augmentor.match_diagnostics['max_absolute_rms_error']<1e-5
    assert not torch.equal(views[0],views[1])
    generator=torch.Generator().manual_seed(8)
    state=checkpoint_rng(augmentor,generator,torch.device('cpu'))
    expected=augmentor(x)
    restore_rng(state,augmentor,generator,torch.device('cpu'))
    actual=augmentor(x)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))


def test_weighted_calibration_equals_explicit_patient_repetition():
    y=np.array([[0,1],[1,0],[1,1],[0,0]],dtype=float)
    p=np.array([[.1,1.],[.5,.9],[.8,.6],[.2,0.]])
    weights=np.array([2,0,3,1])
    repeated=np.repeat(np.arange(4),weights)
    reference=evaluate_predictions(y[repeated],p[repeated],classes=('a','b'))['metrics']
    actual=calibration(y,p,weights)[0]
    assert np.allclose(actual,[reference['brier_macro'],reference['ece_macro']])


def test_transfer_resume_preserves_predictions(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,n):
            self.x=torch.randn(n,12,1000,generator=torch.Generator().manual_seed(n))
            self.y=torch.tensor([[float(i%2)] for i in range(n)])
            self.record_ids=['101']*n
        def __len__(self):return len(self.x)
        def __getitem__(self,i):return self.x[i],self.y[i]
    prepared=({'train':['101'],'validation':['102'],'test':['103']},
              {'train':Dataset(16),'validation':Dataset(8),'test':Dataset(10)})
    config={'smoke':True};condition={'name':'test_transfer'};device=torch.device('cpu')
    a=transfer_study.run(config,condition,42,tmp_path/'full',prepared,device)
    original=transfer_study.save_checkpoint
    interrupted=[False]
    def stop(path,value):
        original(path,value)
        if path.name=='resume.pt' and not interrupted[0]:
            interrupted[0]=True
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(transfer_study,'save_checkpoint',stop)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        transfer_study.run(config,condition,42,tmp_path/'resumed',prepared,device)
    monkeypatch.setattr(transfer_study,'save_checkpoint',original)
    b=transfer_study.run(config,condition,42,tmp_path/'resumed',prepared,device)
    assert a['best_epoch']==b['best_epoch']
    assert np.array_equal(np.load(tmp_path/'full/test_predictions.npz')['probabilities'],
                          np.load(tmp_path/'resumed/test_predictions.npz')['probabilities'])


def test_subject_bootstrap_can_report_undefined_auc_without_inventing_a_score():
    from patient_uncertainty import weighted_metrics
    y=np.array([[0],[0],[1]],dtype=float)
    p=np.array([[.1],[.3],[.8]])
    weights=np.array([[1,1,0],[1,0,1]])
    values=weighted_metrics(y,p,weights,allow_undefined_auc=True)
    assert np.isnan(values[0,1])
    assert values[1,1]==1.
    assert np.isfinite(values[:,0]).all()
