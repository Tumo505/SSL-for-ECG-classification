"""Train and evaluate subject-separated binary-window MIT-BIH transfer."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ssrl_ecg.data.mitbih import MITBIHDataset, record_splits, patient_id
from ssrl_ecg.models.cnn import ECGClassifier, ECGEncoder1DCNN
from ssrl_ecg.experiment import cpu_state, write_json, save_checkpoint
from ssrl_ecg.experiment_metrics import evaluate_predictions
from ssrl_ecg.utils import choose_device, set_seed


def predictions(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x,y in loader:
            ys.append(y.numpy())
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mitbih-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--freeze-encoder', action='store_true')
    p.add_argument('--use-ssl', action='store_true', help='Require an SSL encoder checkpoint')
    p.add_argument('--finetune-epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto','mps','cpu','cuda'), default='auto')
    p.add_argument('--output-dir', type=Path, default=Path('results/mitbih'))
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.finetune_epochs < 1 or args.batch_size < 1 or args.use_ssl and not args.checkpoint:
        p.error('Positive budgets and a checkpoint for --use-ssl are required')
    set_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    records = sorted(path.stem for path in args.mitbih_root.glob('*.hea'))
    splits = record_splits(records,args.seed)
    ds = {k:MITBIHDataset(args.mitbih_root,v,max_windows_per_record=4 if args.smoke else None) for k,v in splits.items()}
    loaders = {k:DataLoader(d,batch_size=args.batch_size,shuffle=k=='train',
                           generator=torch.Generator().manual_seed(args.seed)) for k,d in ds.items()}
    encoder = ECGEncoder1DCNN()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        state = checkpoint.get('encoder')
        if state is None:
            if args.use_ssl:
                raise ValueError('--use-ssl requires an encoder checkpoint')
            state = {k.removeprefix('encoder.'):v for k,v in checkpoint['model'].items() if k.startswith('encoder.')}
        encoder.load_state_dict(state)
    model = ECGClassifier(encoder,n_classes=1).to(device)
    if args.freeze_encoder:
        if not args.checkpoint:
            p.error('Freezing requires a pretrained checkpoint')
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_f1, history, best = -1, [], None
    for epoch in range(1,args.finetune_epochs+1):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        for x,y in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)),y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite transfer loss')
            loss.backward()
            optimizer.step()
        y,probabilities = predictions(model,loaders['validation'],device)
        metrics = evaluate_predictions(y,probabilities,classes=('ectopic_window',))['metrics']
        history.append(dict(epoch=epoch,validation=metrics))
        if metrics['f1_macro'] > best_f1:
            best_f1, best = metrics['f1_macro'], dict(model=cpu_state(model),epoch=epoch)
        print(f"MIT-BIH epoch {epoch}/{args.finetune_epochs} validation F1={metrics['f1_macro']:.4f}",flush=True)
    model.load_state_dict(best['model'])
    y,probabilities = predictions(model,loaders['test'],device)
    root=args.output_dir
    root.mkdir(parents=True,exist_ok=True)
    save_checkpoint(root/'best.pt',best)
    np.savez_compressed(root/'test_predictions.npz',labels=y,probabilities=probabilities,
                        record_ids=ds['test'].record_ids,patient_ids=[patient_id(r) for r in ds['test'].record_ids])
    write_json(root/'result.json',dict(task='binary nonoverlapping ten-second annotation windows',
               smoke=args.smoke,seed=args.seed,record_splits=splits,counts={k:len(d) for k,d in ds.items()},
               checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
               frozen_encoder=args.freeze_encoder,best_epoch=best['epoch'],history=history,
               test=evaluate_predictions(y,probabilities,classes=('ectopic_window',))))


if __name__ == '__main__':
    main()
