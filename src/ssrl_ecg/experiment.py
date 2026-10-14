"""Shared training protocol for ECG experiments, with resumable checkpoints."""

from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from filelock import FileLock

from ssrl_ecg.augmentations import make_augmentations
from ssrl_ecg.data.ptbxl import (
    DIAGNOSTIC_CLASSES,
    PTBXLRecordDataset,
    load_ptbxl_metadata,
    make_default_splits,
    sample_labelled_indices,
)
from ssrl_ecg.data.balancing import create_balanced_dataloader
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.models.resnet1d import ResNetEncoder
from ssrl_ecg.models.simclr import SimCLRModel, NTXentLoss
from ssrl_ecg.models.clocs import (
    PatientNCELoss, temporal_views, ReferenceCLOCSEncoder, ReferenceCLOCSModel,
)
from ssrl_ecg.train_ssl_byol import BYOLModel, byol_loss
from ssrl_ecg.models.losses import (
    FocalLoss,
    WeightedBCELoss,
    ClassBalancedLoss,
    compute_class_weights,
)
from ssrl_ecg.experiment_metrics import evaluate_predictions, validation_thresholds
from ssrl_ecg.utils import choose_device, set_seed


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def save_checkpoint(path, data):
    tmp = Path(str(path) + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def source_hash():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(p.relative_to(Path(__file__).parent)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def cache_signals(root, df, labels, sampling_rate, cache_dir):
    """Validate each waveform once, then reuse a read-only normalised memmap."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_hash = hashlib.sha256(
        (Path(root) / "ptbxl_database.csv").read_bytes()
    ).hexdigest()
    statement_hash = hashlib.sha256(
        (Path(root) / "scp_statements.csv").read_bytes()
    ).hexdigest()
    waveform_files = hashlib.sha256()
    column = "filename_hr" if sampling_rate == 500 else "filename_lr"
    for record in df[column]:
        for extension in (".hea", ".dat"):
            file = Path(root) / (record + extension)
            stat = file.stat()
            waveform_files.update(
                f"{record}{extension}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            )
    identity = {
        "root": str(Path(root).resolve()),
        "metadata_sha256": metadata_hash,
        "statements_sha256": statement_hash,
        "waveform_size_mtime_digest": waveform_files.hexdigest(),
        "sampling_rate": sampling_rate,
        "normalization": "per_record_per_lead_population_std_v1",
        "n_records": len(df),
        "classes": list(DIAGNOSTIC_CLASSES),
    }
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    path = cache_dir / f"ptbxl_{token}.npy"
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            tmp = Path(str(path) + ".tmp")
            array = np.lib.format.open_memmap(
                tmp,
                mode="w+",
                dtype=np.float32,
                shape=(len(df), 12, sampling_rate * 10),
            )
            ds = PTBXLRecordDataset(
                root,
                df,
                labels,
                np.arange(len(df)),
                use_high_resolution=sampling_rate == 500,
                signal_length=sampling_rate * 10,
                return_labels=False,
            )
            for i in tqdm(range(len(ds)), desc="Validate and cache PTB-XL"):
                array[i] = ds[i].numpy()
            array.flush()
            del array
            tmp.replace(path)
            write_json(path.with_suffix(".json"), identity)
        array = np.load(path, mmap_mode="r")
        if array.shape != (len(df), 12, sampling_rate * 10):
            raise ValueError("Cache shape mismatch")
    return array, identity


def prepare_data(config):
    root = Path(config["data_root"])
    df, labels = load_ptbxl_metadata(root)
    splits = make_default_splits(df)
    signals, cache_identity = cache_signals(
        root, df, labels, config["sampling_rate"], config["cache_dir"]
    )
    return df, labels, splits, signals, cache_identity


def checkpoint_rng(augmentor, generator, device):
    result = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader": generator.get_state(),
    }
    if augmentor is not None:
        result["augmentation_torch"] = augmentor.generator.get_state()
        result["augmentation_numpy"] = augmentor.rng.bit_generator.state
    if device.type == "mps":
        result["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state, augmentor, generator, device):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    generator.set_state(state["loader"])
    if augmentor is not None:
        augmentor.generator.set_state(state["augmentation_torch"])
        augmentor.rng.bit_generator.state = state["augmentation_numpy"]
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, features = [], [], []
    for x, y in loader:
        x = x.to(device)
        z = model.pool(model.encoder(x)).squeeze(-1)
        p = torch.sigmoid(model.head(z))
        ys.append(y.numpy())
        ps.append(p.cpu().numpy())
        features.append(z.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(features)


def representation_diagnostics(features):
    centered = features.astype(np.float64) - features.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    prob = singular / max(singular.sum(), 1e-12)
    entropy = -np.sum(prob[prob > 0] * np.log(prob[prob > 0]))
    return {
        "mean_feature_std": float(centered.std(0).mean()),
        "effective_rank": float(np.exp(entropy)),
        "n_features": features.shape[1],
        "n_validation_records": len(features),
    }


def run_condition(config, condition, seed, output, prepared, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df, labels, splits, signals, cache_identity = prepared
    method, policy, exclude = (
        condition["method"],
        condition.get("policy", "domain"),
        condition.get("exclude"),
    )
    smoke = config.get("smoke", False)
    train_idx = splits.train_idx
    eligible_train = train_idx[labels[train_idx].sum(1) > 0]
    sampled = sample_labelled_indices(
        eligible_train, labels, config["label_fraction"], seed
    )
    val_idx = splits.val_idx[labels[splits.val_idx].sum(1) > 0]
    test_idx = splits.test_idx[labels[splits.test_idx].sum(1) > 0]
    if smoke:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(train_idx, min(256, len(train_idx)), replace=False)
        sampled = rng.choice(sampled, min(128, len(sampled)), replace=False)
        val_idx = rng.choice(val_idx, min(256, len(val_idx)), replace=False)
        test_idx = rng.choice(test_idx, min(256, len(test_idx)), replace=False)
    identity = {
        "config": config,
        "condition": condition,
        "seed": seed,
        "code_sha256": source_hash(),
        "data": cache_identity,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    if (output / "result.json").exists():
        existing = json.loads((output / "result.json").read_text())
        if existing["fingerprint"] != fingerprint:
            raise ValueError(f"Existing run has different provenance: {output}")
        return existing
    if (output / "provenance.json").exists():
        if (
            json.loads((output / "provenance.json").read_text())["fingerprint"]
            != fingerprint
        ):
            raise ValueError(f"Cannot resume changed protocol/code: {output}")
    identity["fingerprint"] = fingerprint
    identity["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "wfdb",
                "iterative-stratification",
            )
        },
        "determinism": "seeded_rng; MPS bitwise reproducibility not guaranteed",
    }
    write_json(output / "provenance.json", identity)
    np.savez_compressed(
        output / "splits.npz",
        ssl_ecg_ids=df.iloc[train_idx].ecg_id.to_numpy(),
        labelled_ecg_ids=df.iloc[sampled].ecg_id.to_numpy(),
        val_ecg_ids=df.iloc[val_idx].ecg_id.to_numpy(),
        test_ecg_ids=df.iloc[test_idx].ecg_id.to_numpy(),
        labelled_indices=sampled,
        train_patient_ids=df.iloc[train_idx].patient_id.to_numpy(),
        val_patient_ids=df.iloc[val_idx].patient_id.to_numpy(),
        test_patient_ids=df.iloc[test_idx].patient_id.to_numpy(),
    )
    set_seed(seed)
    reference_clocs = condition.get("backbone") == "clocs_reference"
    if reference_clocs and config["sampling_rate"] != 500:
        raise ValueError("Reference CLOCS must use native 500 Hz waveforms")
    encoder = (ReferenceCLOCSEncoder() if reference_clocs else ResNetEncoder()
               if condition.get("backbone") == "resnet" else ECGEncoder1DCNN(in_ch=12, width=64))
    length = config["sampling_rate"] * 10

    def dataset(idx, return_labels=True):
        return PTBXLRecordDataset(
            config["data_root"],
            df,
            labels,
            idx,
            use_high_resolution=config["sampling_rate"] == 500,
            signal_length=length,
            return_labels=return_labels,
            signals=signals,
        )

    status = {
        "condition": condition["name"],
        "seed": seed,
        "smoke": smoke,
        "status": "running",
    }
    write_json(output / "status.json", status)
    history = []
    if (output / "history.json").exists():
        history = json.loads((output / "history.json").read_text())
    pretrain_seconds = 0.0
    if method == "finetune":
        ckpt = torch.load(
            config["initial_encoder"], map_location="cpu", weights_only=False
        )
        encoder.load_state_dict(ckpt["encoder"])
    elif method != "supervised":
        pretrained = output / "pretrained.pt"
        if pretrained.exists():
            ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
            encoder.load_state_dict(ckpt["encoder"])
            pretrain_seconds = ckpt["seconds"]
        else:
            model = (
                BYOLModel(encoder)
                if method == "byol"
                else ReferenceCLOCSModel(encoder) if reference_clocs
                else SimCLRModel(encoder, projection_dim=128)
            )
            model.to(device)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=config["pretrain_lr"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["pretrain_epochs"]
            )
            augmentor = (
                None
                if method == "clocs"
                else make_augmentations(
                    policy, exclude, config["sampling_rate"], seed + 10000
                )
            )
            generator = torch.Generator().manual_seed(seed + 20000)
            loader = DataLoader(
                dataset(train_idx, False),
                batch_size=config["pretrain_batch_size"],
                shuffle=True,
                drop_last=True,
                generator=generator,
                num_workers=0,
            )
            patient_ids = torch.tensor(
                df.iloc[train_idx].patient_id.to_numpy(), dtype=torch.int64
            )
            # CLOCS must carry IDs through the exact shuffled batch.
            if method == "clocs":
                from torch.utils.data import TensorDataset

                indices = TensorDataset(torch.arange(len(train_idx)), patient_ids)
                loader = DataLoader(
                    indices,
                    batch_size=config["pretrain_batch_size"],
                    shuffle=True,
                    drop_last=True,
                    generator=generator,
                    num_workers=0,
                )
            criterion = (
                PatientNCELoss(config["clocs_temperature"])
                if method == "clocs"
                else NTXentLoss(config["temperature"])
            )
            write_json(
                output / "augmentation.json",
                (
                    augmentor.specification()
                    if augmentor
                    else {
                        "method": "CLOCS CMSC reference architecture" if reference_clocs else "CLOCS CMSC shared backbone",
                        "views": "adjacent nonoverlapping 5-second lead-II segments at 500 Hz" if reference_clocs else "adjacent nonoverlapping 5-second 12-lead segments",
                        "objective": "patient-specific cross-view NCE equations 2-4",
                        "backbone": "author three-block CNN with 256-dimensional linear embedding" if reference_clocs else "shared study backbone and projection head",
                        "reference": "https://proceedings.mlr.press/v139/kiyasseh21a.html",
                    }
                ),
            )
            last = output / "pretrain_last.pt"
            start_epoch = 1
            steps = 0
            if last.exists():
                ckpt = torch.load(last, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt["epoch"] + 1
                steps, pretrain_seconds = ckpt["steps"], ckpt["seconds"]
                restore_rng(ckpt["rng"], augmentor, generator, device)
                history = [
                    e
                    for e in history
                    if e["stage"] != "pretrain" or e["epoch"] < start_epoch
                ]
            for epoch in range(start_epoch, config["pretrain_epochs"] + 1):
                model.train()
                total = n = 0
                started = time.monotonic()
                for batch in loader:
                    if method == "clocs":
                        local_idx, pids = batch
                        x = torch.from_numpy(
                            np.array(signals[train_idx[local_idx.numpy()]], copy=True)
                        )
                        x1, x2 = temporal_views(x)
                    else:
                        x1, x2 = augmentor(batch)
                    x1, x2 = x1.to(device), x2.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if method == "byol":
                        loss = byol_loss(*model(x1, x2))
                    else:
                        _, z1 = model(x1)
                        _, z2 = model(x2)
                        loss = (
                            criterion(z1, z2, pids.to(device))
                            if method == "clocs"
                            else criterion(z1, z2)
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite SSL loss in {output}")
                    loss.backward()
                    optimizer.step()
                    if method == "byol":
                        model.update_target_network(config["momentum_tau"])
                    total += loss.item() * len(x1)
                    n += len(x1)
                    steps += 1
                if n == 0:
                    raise ValueError("No complete SSL batches; reduce batch size")
                scheduler.step()
                seconds = time.monotonic() - started
                pretrain_seconds += seconds
                entry = {
                    "stage": "pretrain",
                    "epoch": epoch,
                    "loss": total / n,
                    "seconds": seconds,
                    "steps": steps,
                    "examples_seen": steps * config["pretrain_batch_size"],
                }
                history.append(entry)
                write_json(output / "history.json", history)
                print(
                    f"{condition['name']} seed={seed} SSL {epoch}/{config['pretrain_epochs']} loss={total/n:.5f} {seconds:.1f}s",
                    flush=True,
                )
                write_json(output / "status.json", {**status, **entry})
                save_checkpoint(
                    last,
                    {
                        "model": cpu_state(model),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "steps": steps,
                        "seconds": pretrain_seconds,
                        "rng": checkpoint_rng(augmentor, generator, device),
                    },
                )
            save_checkpoint(
                pretrained,
                {
                    "encoder": cpu_state(model.encoder),
                    "seconds": pretrain_seconds,
                    "fingerprint": fingerprint,
                },
            )
            encoder = model.encoder.cpu()
            del model, optimizer, scheduler
            if device.type == "mps":
                torch.mps.empty_cache()
    if config.get("pretrain_only"):
        result = {
            "fingerprint": fingerprint,
            "condition": condition["name"],
            "seed": seed,
            "status": "pretraining_complete",
            "pretrain_seconds": pretrain_seconds,
        }
        write_json(output / "status.json", result)
        return result
    # All downstream heads share the same initialisation RNG, loss and sampler.
    set_seed(seed + 30000)
    model = ECGClassifier(encoder, n_classes=len(DIAGNOSTIC_CLASSES)).to(device)
    frozen = method != "supervised" and condition.get("freeze_encoder", config.get("freeze_encoder", True))
    if frozen:
        for p in model.encoder.parameters():
            p.requires_grad = False
    generator = torch.Generator().manual_seed(seed + 40000)
    loader = create_balanced_dataloader(
        dataset(sampled),
        labels[sampled],
        config["downstream_batch_size"],
        strategy=config.get("balance_strategy", "oversample"),
        generator=generator,
    )
    val_loader = DataLoader(
        dataset(val_idx), batch_size=config["downstream_batch_size"], shuffle=False
    )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config["downstream_lr"]
    )
    loss_name = config.get("loss", "focal")
    if loss_name == "focal":
        criterion = FocalLoss(alpha=config.get("focal_alpha", 0.25), gamma=2)
    elif loss_name == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss_name == "weighted":
        criterion = WeightedBCELoss(compute_class_weights(labels[sampled]).to(device))
    elif loss_name == "class_balanced":
        criterion = ClassBalancedLoss(torch.tensor(labels[sampled].sum(0))).to(device)
    else:
        raise ValueError(f"Unknown loss {loss_name}")
    start_epoch = 1
    best_f1 = -1
    best_epoch = 0
    downstream_seconds = 0
    best_path = output / "best.pt"
    last = output / "downstream_last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]
        best_epoch = ckpt["best_epoch"]
        downstream_seconds = ckpt["seconds"]
        restore_rng(ckpt["rng"], None, generator, device)
        history = [
            e for e in history if e["stage"] != "downstream" or e["epoch"] < start_epoch
        ]
    for epoch in range(start_epoch, config["downstream_epochs"] + 1):
        started = time.monotonic()
        model.train()
        if frozen:
            model.encoder.eval()
        total = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite downstream loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        y, p, _ = predict(model, val_loader, device)
        metrics = evaluate_predictions(y, p)["metrics"]
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_epoch = epoch
            save_checkpoint(
                best_path,
                {
                    "model": cpu_state(model),
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "fingerprint": fingerprint,
                },
            )
        seconds = time.monotonic() - started
        downstream_seconds += seconds
        entry = {
            "stage": "downstream",
            "epoch": epoch,
            "loss": total / n,
            "validation": metrics,
            "seconds": seconds,
        }
        history.append(entry)
        write_json(output / "history.json", history)
        write_json(output / "status.json", {**status, **entry})
        print(
            f"{condition['name']} seed={seed} downstream {epoch}/{config['downstream_epochs']} val F1={metrics['f1_macro']:.4f} {seconds:.1f}s",
            flush=True,
        )
        save_checkpoint(
            last,
            {
                "model": cpu_state(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "seconds": downstream_seconds,
                "rng": checkpoint_rng(None, generator, device),
            },
        )
    model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=False)["model"]
    )
    vy, vp, features = predict(model, val_loader, device)
    thresholds = validation_thresholds(vy, vp)
    test_loader = DataLoader(
        dataset(test_idx), batch_size=config["downstream_batch_size"], shuffle=False
    )
    ty, tp, _ = predict(model, test_loader, device)
    for name, idx, y, p in [
        ("validation", val_idx, vy, vp),
        ("test", test_idx, ty, tp),
    ]:
        np.savez_compressed(
            output / f"{name}_predictions.npz",
            ecg_ids=df.iloc[idx].ecg_id.to_numpy(),
            patient_ids=df.iloc[idx].patient_id.to_numpy(),
            labels=y,
            probabilities=p,
            thresholds=thresholds,
            classes=np.array(DIAGNOSTIC_CLASSES),
        )
    result = {
        "fingerprint": fingerprint,
        "condition": condition["name"],
        "seed": seed,
        "status": "complete",
        "smoke": smoke,
        "best_epoch": best_epoch,
        "frozen_encoder": frozen,
        "primary_threshold": 0.5,
        "validation_thresholds": thresholds.tolist(),
        "test_fixed": evaluate_predictions(ty, tp),
        "test_validation_tuned": evaluate_predictions(ty, tp, thresholds),
        "representation_validation": representation_diagnostics(features),
        "pretrain_seconds": pretrain_seconds,
        "downstream_seconds": downstream_seconds,
        "counts": {
            "pretrain": len(train_idx),
            "labelled": len(sampled),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "checkpoint": str(best_path.resolve()),
    }
    write_json(output / "result.json", result)
    write_json(output / "status.json", result)
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return result
