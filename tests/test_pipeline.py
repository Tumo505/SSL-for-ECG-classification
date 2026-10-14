from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ssrl_ecg.augmentations import DOMAIN_TRANSFORMS, make_augmentations
from ssrl_ecg.data.ptbxl import (
    PTBXLSplits,
    make_default_splits,
    sample_labelled_indices,
)
from ssrl_ecg.experiment import cpu_state, run_condition
from ssrl_ecg.experiment_metrics import paired_comparison, seed_summary, holm_adjust
from ssrl_ecg.models.cnn import ECGEncoder1DCNN
from ssrl_ecg.models.simclr import NTXentLoss
from ssrl_ecg.models.clocs import PatientNCELoss, temporal_views
from ssrl_ecg.train_ssl_byol import BYOLModel, byol_loss
from ssrl_ecg.models.losses import FocalLoss


@pytest.fixture(autouse=True)
def small_cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("n", [2, 3, 8])
def test_ntxent_matches_explicit_positive_pair_reference(n):
    torch.manual_seed(9)
    a, b = torch.randn(n, 16, requires_grad=True), torch.randn(
        n, 16, requires_grad=True
    )
    z = F.normalize(torch.cat((a, b)), dim=1)
    scores = z @ z.T / 0.2
    reference = torch.stack(
        [
            torch.logsumexp(torch.cat((scores[i, :i], scores[i, i + 1 :])), dim=0)
            - scores[i, (i + n) % (2 * n)]
            for i in range(2 * n)
        ]
    ).mean()
    actual = NTXentLoss(0.2)(a, b)
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_byol_cross_view_objective_and_target_gradients():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = a.flip(1)
    assert byol_loss(a, a, b, b).item() == pytest.approx(4.0)
    model = BYOLModel(ECGEncoder1DCNN(width=64), hidden_dim=32, projection_dim=16)
    for online, target in zip(
        model.online_projector.parameters(), model.target_projector.parameters()
    ):
        torch.testing.assert_close(online, target)
    x, y = torch.randn(2, 12, 1000), torch.randn(2, 12, 1000)
    loss = byol_loss(*model(x, y))
    loss.backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert all(p.grad is None for p in model.target_projector.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters())
    before = cpu_state(model.target_projector)
    with torch.no_grad():
        for p in model.online_projector.parameters():
            p.add_(1)
    model.update_target_network(0.5)
    for key, value in model.target_projector.named_parameters():
        torch.testing.assert_close(
            value,
            (before[key] + dict(model.online_projector.named_parameters())[key]) / 2,
        )


@pytest.mark.parametrize("excluded", [None, *DOMAIN_TRANSFORMS])
def test_domain_policy_removes_only_selected_transform(excluded):
    x = torch.randn(32, 12, 1000)
    saved = x.clone()
    aug = make_augmentations("domain", excluded, seed=42)
    a, b = aug(x)
    assert (
        a.shape == b.shape == x.shape
        and torch.isfinite(a).all()
        and torch.isfinite(b).all()
    )
    torch.testing.assert_close(x, saved)
    assert not torch.equal(a, b)
    if excluded:
        assert excluded not in aug.last_applied
    for transform in set(DOMAIN_TRANSFORMS) - {excluded}:
        assert aug.last_applied[transform] > 0
    a2, b2 = make_augmentations("domain", excluded, seed=42)(x)
    torch.testing.assert_close(a, a2)
    torch.testing.assert_close(b, b2)


def test_generic_and_single_record_policies():
    x = torch.randn(12, 1000)
    aug = make_augmentations("generic")
    a, b = aug(x)
    assert a.shape == b.shape == x.shape
    assert set(aug.last_applied) == {"jitter", "scaling", "temporal_mask"}
    with pytest.raises(ValueError):
        make_augmentations("generic", "motion")


def test_label_sampling_and_patient_split_guard():
    labels = np.tile(np.eye(5, dtype=np.float32), (20, 1))
    a = sample_labelled_indices(np.arange(100), labels, 0.1, 42)
    b = sample_labelled_indices(np.arange(100), labels, 0.1, 42)
    assert len(a) == 10 and np.array_equal(a, b) and (labels[a].sum(0) > 0).all()
    df = pd.DataFrame({"strat_fold": [1, 9, 10], "patient_id": [1, 2, 1]})
    with pytest.raises(ValueError, match="leakage"):
        make_default_splits(df)


def test_seed_statistics_preserve_pairing():
    assert seed_summary([0.5])["ci95_mean"] is None
    with pytest.raises(ValueError):
        paired_comparison({42: 0.5}, {52: 0.6})
    result = paired_comparison({42: 0.7, 52: 0.8}, {52: 0.6, 42: 0.5})
    assert result["mean"] == pytest.approx(0.2)
    assert holm_adjust([0.01, 0.03, 0.2]) == pytest.approx([0.03, 0.06, 0.2])


def test_clocs_patient_loss_and_temporal_views():
    x = torch.randn(3, 12, 1000)
    a, b = temporal_views(x)
    torch.testing.assert_close(a, x[:, :, :500])
    torch.testing.assert_close(b, x[:, :, 500:])
    z = torch.eye(3, requires_grad=True)
    ids = torch.arange(3)
    criterion = PatientNCELoss(0.1)
    reference = F.cross_entropy(z @ z.T / 0.1, ids)
    loss = criterion(z, z, ids)
    torch.testing.assert_close(loss, reference)
    repeated = criterion(z, z, torch.tensor([1, 1, 2]))
    repeated.backward()
    assert torch.isfinite(repeated) and torch.isfinite(z.grad).all()


def test_focal_alpha_distinguishes_positive_negative():
    logits = torch.zeros((1, 2))
    y = torch.tensor([[1.0, 0.0]])
    loss = FocalLoss(alpha=0.25, gamma=0, reduction="none")(logits, y)
    assert loss[0, 1] / loss[0, 0] == pytest.approx(3)


@pytest.mark.parametrize("method", ["supervised", "simclr", "byol", "clocs"])
def test_training_checkpoint_predictions_and_freeze(tmp_path, method):
    # Synthetic waveforms verify the full optimiser/selection/export path, not
    # scientific performance. Real PTB-XL smoke runs are a separate gate.
    n = 120
    labels = np.tile(np.eye(5, dtype=np.float32), (24, 1))
    df = pd.DataFrame({"ecg_id": np.arange(n), "patient_id": np.arange(n)})
    splits = PTBXLSplits(np.arange(80), np.arange(80, 100), np.arange(100, 120))
    signals = np.random.default_rng(2).normal(size=(n, 12, 1000)).astype(np.float32)
    prepared = (df, labels, splits, signals, {"synthetic": True})
    config = dict(
        data_root=str(tmp_path),
        sampling_rate=100,
        smoke=True,
        label_fraction=0.5,
        pretrain_epochs=1,
        pretrain_batch_size=8,
        pretrain_lr=0.001,
        temperature=0.07,
        clocs_temperature=0.1,
        momentum_tau=0.999,
        downstream_epochs=2,
        downstream_batch_size=8,
        downstream_lr=0.001,
    )
    condition = {"name": method, "method": method, "policy": "domain"}
    result = run_condition(
        config, condition, 42, tmp_path / method, prepared, torch.device("cpu")
    )
    assert result["status"] == "complete" and result["test_fixed"]["n_records"] == 20
    checkpoint = torch.load(tmp_path / method / "best.pt", weights_only=False)
    assert checkpoint["epoch"] == result["best_epoch"]
    predictions = np.load(tmp_path / method / "test_predictions.npz")
    assert (
        predictions["probabilities"].shape == (20, 5)
        and np.isfinite(predictions["probabilities"]).all()
    )
    if method != "supervised":
        pretrained = torch.load(
            tmp_path / method / "pretrained.pt", weights_only=False
        )["encoder"]
        for k, v in pretrained.items():
            torch.testing.assert_close(checkpoint["model"]["encoder." + k], v)
    again = run_condition(
        config, condition, 42, tmp_path / method, prepared, torch.device("cpu")
    )
    assert again == result
    changed = {**config, "downstream_epochs": 3}
    with pytest.raises(ValueError, match="provenance"):
        run_condition(
            changed, condition, 42, tmp_path / method, prepared, torch.device("cpu")
        )


def test_interrupted_downstream_resumes_identical_cpu_run(tmp_path, monkeypatch):
    import ssrl_ecg.experiment as experiment

    n = 120
    labels = np.tile(np.eye(5, dtype=np.float32), (24, 1))
    df = pd.DataFrame({"ecg_id": np.arange(n), "patient_id": np.arange(n)})
    prepared = (
        df,
        labels,
        PTBXLSplits(np.arange(80), np.arange(80, 100), np.arange(100, 120)),
        np.random.default_rng(2).normal(size=(n, 12, 1000)).astype(np.float32),
        {"synthetic": True},
    )
    config = dict(
        data_root=str(tmp_path),
        sampling_rate=100,
        smoke=True,
        label_fraction=0.5,
        pretrain_epochs=1,
        pretrain_batch_size=8,
        pretrain_lr=0.001,
        temperature=0.07,
        clocs_temperature=0.1,
        momentum_tau=0.999,
        downstream_epochs=2,
        downstream_batch_size=8,
        downstream_lr=0.001,
    )
    condition = {"name": "supervised", "method": "supervised"}
    uninterrupted = run_condition(
        config, condition, 42, tmp_path / "uninterrupted", prepared, torch.device("cpu")
    )
    original_save = experiment.save_checkpoint

    def interrupted_save(path, data):
        original_save(path, data)
        if path.name == "downstream_last.pt" and data["epoch"] == 1:
            raise RuntimeError("Simulated interruption after epoch checkpoint")

    monkeypatch.setattr(experiment, "save_checkpoint", interrupted_save)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        run_condition(
            config, condition, 42, tmp_path / "resumed", prepared, torch.device("cpu")
        )
    monkeypatch.setattr(experiment, "save_checkpoint", original_save)
    resumed = run_condition(
        config, condition, 42, tmp_path / "resumed", prepared, torch.device("cpu")
    )
    a = np.load(tmp_path / "uninterrupted" / "test_predictions.npz")["probabilities"]
    b = np.load(tmp_path / "resumed" / "test_predictions.npz")["probabilities"]
    np.testing.assert_array_equal(a, b)
    assert uninterrupted["best_epoch"] == resumed["best_epoch"]


def test_interrupted_ssl_restores_augmentation_rng(tmp_path, monkeypatch):
    import ssrl_ecg.experiment as experiment

    n = 120
    labels = np.tile(np.eye(5, dtype=np.float32), (24, 1))
    df = pd.DataFrame({"ecg_id": np.arange(n), "patient_id": np.arange(n)})
    prepared = (
        df,
        labels,
        PTBXLSplits(np.arange(80), np.arange(80, 100), np.arange(100, 120)),
        np.random.default_rng(2).normal(size=(n, 12, 1000)).astype(np.float32),
        {"synthetic": True},
    )
    config = dict(
        data_root=str(tmp_path),
        sampling_rate=100,
        smoke=True,
        label_fraction=0.5,
        pretrain_epochs=2,
        pretrain_batch_size=8,
        pretrain_lr=0.001,
        temperature=0.07,
        clocs_temperature=0.1,
        momentum_tau=0.999,
        downstream_epochs=1,
        downstream_batch_size=8,
        downstream_lr=0.001,
    )
    condition = {"name": "simclr", "method": "simclr", "policy": "domain"}
    run_condition(
        config, condition, 42, tmp_path / "uninterrupted", prepared, torch.device("cpu")
    )
    original_save = experiment.save_checkpoint

    def interrupted_save(path, data):
        original_save(path, data)
        if path.name == "pretrain_last.pt" and data["epoch"] == 1:
            raise RuntimeError("Simulated SSL interruption")

    monkeypatch.setattr(experiment, "save_checkpoint", interrupted_save)
    with pytest.raises(RuntimeError, match="Simulated SSL"):
        run_condition(
            config, condition, 42, tmp_path / "resumed", prepared, torch.device("cpu")
        )
    monkeypatch.setattr(experiment, "save_checkpoint", original_save)
    run_condition(
        config, condition, 42, tmp_path / "resumed", prepared, torch.device("cpu")
    )
    a = np.load(tmp_path / "uninterrupted" / "test_predictions.npz")["probabilities"]
    b = np.load(tmp_path / "resumed" / "test_predictions.npz")["probabilities"]
    np.testing.assert_array_equal(a, b)


def test_multilabel_sampling_enforces_exact_record_budget():
    rng = np.random.default_rng(8)
    labels = (rng.random((1000, 5)) < 0.25).astype(np.float32)
    labels[labels.sum(1) == 0, 0] = 1
    for seed in (42, 52, 62):
        idx = sample_labelled_indices(np.arange(1000), labels, 0.1, seed)
        assert len(idx) == len(np.unique(idx)) == 100
        assert (labels[idx].sum(0) > 0).all()
        np.testing.assert_array_equal(
            idx, sample_labelled_indices(np.arange(1000), labels, 0.1, seed)
        )


def test_patient_bootstrap_metrics_match_sklearn_with_ties():
    from scripts.patient_uncertainty import weighted_metrics
    from sklearn.metrics import f1_score, roc_auc_score
    labels = np.array([[0,1],[1,0],[1,1],[0,0],[1,0],[0,1]])
    probabilities = np.array([[.2,.8],[.2,.5],[.8,.8],[.5,.5],[.8,.2],[.5,.2]])
    weights = np.array([[1,1,1,1,1,1],[2,0,3,1,2,1]])
    actual = weighted_metrics(labels,probabilities,weights)
    for row,w in zip(actual,weights):
        expected = [f1_score(labels,probabilities>=.5,average='macro',sample_weight=w),
                    roc_auc_score(labels,probabilities,average='macro',sample_weight=w)]
        np.testing.assert_allclose(row,expected,atol=1e-12)


def test_reference_clocs_uses_only_lead_ii_and_first_downstream_segment():
    from ssrl_ecg.models.clocs import ReferenceCLOCSEncoder, temporal_views
    encoder=ReferenceCLOCSEncoder().eval()
    x=torch.randn(3,12,5000)
    changed=x.clone()
    changed[:,0,:]+=100
    changed[:,:,2500:]+=100
    torch.testing.assert_close(encoder(x),encoder(changed))
    first,second=temporal_views(x)
    assert encoder(first).shape==(3,256,1)
    assert not torch.allclose(encoder(first),encoder(second))
    with pytest.raises(ValueError,match='500 Hz'):
        encoder(torch.randn(3,12,1000))


def test_transfer_partitions_keep_two_records_of_one_patient_together():
    from ssrl_ecg.data.mitbih import record_splits,patient_id
    split=record_splits(['100','101','103','105','106','108','109','201','202'])
    subjects={k:{patient_id(r) for r in v} for k,v in split.items()}
    assert not subjects['train'] & subjects['validation']
    assert not subjects['train'] & subjects['test']
    assert not subjects['test'] & subjects['validation']
    assert any({'201','202'} <= set(records) for records in split.values())


def test_transfer_annotation_windows_exclude_unknown_beats(tmp_path):
    import wfdb
    from ssrl_ecg.data.mitbih import MITBIHDataset
    signal=np.column_stack((np.sin(np.arange(10800)/20),np.cos(np.arange(10800)/20)))
    wfdb.wrsamp('100',fs=360,units=['mV','mV'],sig_name=['MLII','V5'],p_signal=signal,write_dir=str(tmp_path))
    wfdb.wrann('100','atr',sample=np.array([100,3700,7300]),symbol=['N','V','Q'],write_dir=str(tmp_path))
    data=MITBIHDataset(tmp_path,['100'])
    assert len(data)==2 and data.labels==[0.,1.]
    x,y=data[1]
    assert x.shape==(12,1000) and y.tolist()==[1.]
    torch.testing.assert_close(x[0],x[-1])
    assert torch.isfinite(x).all()


def test_finetuning_reuses_encoder_then_updates_it(tmp_path):
    from ssrl_ecg.experiment import run_condition
    from ssrl_ecg.models.cnn import ECGEncoder1DCNN
    n=120
    labels=np.tile(np.eye(5,dtype=np.float32),(24,1))
    df=pd.DataFrame({'ecg_id':np.arange(n),'patient_id':np.arange(n)})
    prepared=(df,labels,PTBXLSplits(np.arange(80),np.arange(80,100),np.arange(100,120)),
              np.random.default_rng(2).normal(size=(n,12,1000)).astype(np.float32),{'synthetic':True})
    encoder=ECGEncoder1DCNN()
    original={k:v.clone() for k,v in encoder.state_dict().items()}
    path=tmp_path/'encoder.pt'
    torch.save({'encoder':original},path)
    config=dict(data_root=str(tmp_path),sampling_rate=100,smoke=True,label_fraction=.5,
                downstream_epochs=1,downstream_batch_size=8,downstream_lr=.001,initial_encoder=str(path))
    condition=dict(name='finetuned',method='finetune',freeze_encoder=False)
    result=run_condition(config,condition,42,tmp_path/'run',prepared,torch.device('cpu'))
    final=torch.load(tmp_path/'run/best.pt',weights_only=False)['model']
    assert not result['frozen_encoder']
    assert not torch.equal(original['features.0.block.0.weight'],final['encoder.features.0.block.0.weight'])
