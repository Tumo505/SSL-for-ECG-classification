"""Legacy module entry points use the same repaired protocol as the study runner."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch

from ssrl_ecg.augmentations import DOMAIN_TRANSFORMS
from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT
from ssrl_ecg.utils import choose_device


def run_training_cli(method, backbone=None):
    # Deferred import keeps BYOL model definitions independent of this adapter.
    from ssrl_ecg.experiment import prepare_data, run_condition

    parser = argparse.ArgumentParser(
        description=f"Run repaired {method} ECG training on CUDA/MPS/CPU"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128 if method in ("simclr", "byol", "clocs") else 64,
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500), default=100)
    parser.add_argument("--signal-length", type=int)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument(
        "--out", type=Path, default=Path("checkpoints") / (method + ".pt")
    )
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument(
        "--loss",
        choices=("bce", "focal", "weighted", "class_balanced"),
        default="focal",
    )
    parser.add_argument(
        "--balance-strategy",
        choices=("standard", "stratified", "oversample"),
        default="oversample",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--momentum-tau", type=float, default=0.999)
    parser.add_argument(
        "--config",
        choices=(
            "full",
            "generic",
            *["no-" + t.replace("_", "-") for t in DOMAIN_TRANSFORMS],
        ),
        default="full",
    )
    parser.add_argument("--exclude", choices=DOMAIN_TRANSFORMS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--ssl-checkpoint", type=Path, required=method == "finetune")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or not 0 < args.label_fraction <= 1:
        parser.error("Require positive epochs, batch size >= 2 and label fraction in (0, 1]")
    if args.signal_length and args.signal_length != args.sampling_rate * 10:
        parser.error("signal length must match the ten-second sampling rate")
    if args.projection_dim != 128:
        parser.error(
            "The shared SimCLR/CLOCS protocol uses projection dimension 128; BYOL uses 256"
        )
    config = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs" / "experiment_study.json"
        ).read_text()
    )
    config.update(
        data_root=str(args.data_root.resolve()),
        cache_dir=str(args.cache_dir.resolve()),
        sampling_rate=args.sampling_rate,
        label_fraction=args.label_fraction,
        loss=args.loss,
        balance_strategy=args.balance_strategy,
        temperature=args.temperature,
        momentum_tau=args.momentum_tau,
        smoke=args.smoke,
        freeze_encoder=args.freeze_encoder if method == "finetune" else True,
    )
    ssl = method in ("simclr", "byol", "clocs")
    config["pretrain_only"] = ssl
    if ssl:
        config.update(
            pretrain_epochs=args.epochs,
            pretrain_batch_size=args.batch_size,
            pretrain_lr=args.lr,
        )
    else:
        config.update(
            downstream_epochs=args.epochs,
            downstream_batch_size=args.batch_size,
            downstream_lr=args.lr,
        )
    if method == "finetune":
        config["initial_encoder"] = str(args.ssl_checkpoint.resolve())
        config["initial_encoder_sha256"] = hashlib.sha256(
            args.ssl_checkpoint.read_bytes()
        ).hexdigest()
    condition = {
        "name": method,
        "method": method,
        "policy": "generic" if args.config == "generic" else "domain",
    }
    if backbone:
        condition["backbone"] = backbone
    if args.exclude:
        condition["exclude"] = args.exclude
    if args.config.startswith("no-"):
        condition["exclude"] = args.config[3:].replace("-", "_")
    torch.set_num_threads(config.get("cpu_threads", 4))
    prepared = prepare_data(config)
    output = args.out.parent / (args.out.stem + "_run")
    run_condition(
        config, condition, args.seed, output, prepared, choose_device(args.device)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / ("pretrained.pt" if ssl else "best.pt"), args.out)
    print(
        f"Saved {args.out.resolve()}; predictions/provenance in {output.resolve()}",
        flush=True,
    )
