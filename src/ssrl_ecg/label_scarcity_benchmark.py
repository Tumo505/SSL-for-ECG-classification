"""Evaluate supervised, frozen and trainable SSL encoders at declared label budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from filelock import FileLock
from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT
from ssrl_ecg.experiment import prepare_data, run_condition, write_json
from ssrl_ecg.experiment_analysis import analyse
from ssrl_ecg.utils import choose_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument('--encoder-study', type=Path, default=Path('results/study/simclr_domain'))
    p.add_argument('--output-dir', type=Path, default=Path('results/label_scarcity'))
    p.add_argument('--fractions', type=float, nargs='+', default=[.01,.05,.1,.25,1.0])
    p.add_argument('--seeds', type=int, nargs='+', default=[42,52,62])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--device', choices=['auto','mps','cpu','cuda'], default='auto')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if any(not 0 < f <= 1 for f in args.fractions) or args.epochs < 1:
        p.error('Fractions must be in (0,1] and epochs positive')
    base = json.loads((Path(__file__).resolve().parents[2]/'configs/experiment_study.json').read_text())
    base.update(data_root=str(args.data_root.resolve()), seeds=args.seeds, smoke=args.smoke,
                downstream_epochs=args.epochs)
    torch.set_num_threads(4)
    prepared = prepare_data(base)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_dir/'study.lock'), timeout=0):
        for fraction in args.fractions:
            root = args.output_dir/f'fraction_{fraction:g}'
            cases = [dict(name='supervised',method='supervised'),
                     dict(name='simclr_domain_frozen',method='finetune',freeze_encoder=True),
                     dict(name='simclr_domain_finetuned',method='finetune',freeze_encoder=False)]
            config = dict(base, label_fraction=fraction, conditions=cases)
            root.mkdir(parents=True, exist_ok=True)
            plan = root/'study.json'
            if plan.exists() and json.loads(plan.read_text()) != config:
                raise ValueError('Existing label-budget protocol differs')
            write_json(plan,config)
            for seed in args.seeds:
                encoder = args.encoder_study/f'seed_{seed}'/'pretrained.pt'
                if not encoder.exists():
                    raise FileNotFoundError(f'A separately trained encoder is required for seed {seed}: {encoder}')
                for condition in cases:
                    current = dict(config)
                    if condition['method']=='finetune':
                        current.update(initial_encoder=str(encoder.resolve()),
                                       initial_encoder_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest())
                    run_condition(current,condition,seed,root/condition['name']/f'seed_{seed}',prepared,device)
            analyse(root,args.seeds,[c['name'] for c in cases])

if __name__ == '__main__':
    main()
