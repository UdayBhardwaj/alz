"""
Subject-level cross-validated training of the 3D ResNet on a benchmark manifest.

Every fold is split by subject, checked with ``check_split_integrity()`` before
a single gradient step, and the inner validation split used for early stopping
is drawn from the training subjects only, so the test fold is never touched
until the final evaluation. Early stopping is on validation AUC rather than
accuracy: with balanced classes the two usually agree, but AUC does not depend
on the 0.5 threshold, which matters when a model is still uncalibrated.

Outputs per fold, in ``--out``:
    fold{k}.pt                 state_dict, the format run_evaluation.py loads
    fold{k}_test_manifest.csv  the exact held-out subjects, for evaluation
    fold{k}_history.json       per-epoch loss / val AUC
    fold{k}_demographics.json  demographic_summary() of train / val / test
and ``summary.json`` with test AUC and accuracy per fold, the seed and the
full configuration.

Usage
-----
    python train.py --manifest bench/atrophy/manifest.csv --out runs/atrophy
    python train.py ... --folds 0           # pilot: first fold only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.adni_dataset import (ADNIDataset, ScanRecord, check_split_integrity,
                               demographic_summary, read_manifest,
                               subject_level_splits)
from models.resnet3d import resnet3d_10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inner_split(records: Sequence[ScanRecord], val_fraction: float,
                seed: int) -> Tuple[List[ScanRecord], List[ScanRecord]]:
    """Subject-level train / validation split inside a training fold."""
    n_folds = max(2, round(1.0 / val_fraction))
    train, val = next(iter(subject_level_splits(records, n_folds=n_folds, seed=seed)))
    check_split_integrity(train, val)
    return train, val


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def class_weights(records: Sequence[ScanRecord], n_classes: int) -> torch.Tensor:
    counts = np.bincount([r.label for r in records], minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def write_manifest(records: Sequence[ScanRecord], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "subject_id", "label", "age", "sex"])
        for r in records:
            w.writerow([r.path, r.subject_id, r.label,
                        "" if r.age is None else r.age, r.sex or ""])


def train_fold(k: int, train_recs, test_recs, args, device, out: Path) -> Dict[str, float]:
    seed_everything(args.seed + k)
    check_split_integrity(train_recs, test_recs)
    tr, va = inner_split(train_recs, args.val_fraction, args.seed + k)
    check_split_integrity(tr, test_recs)
    check_split_integrity(va, test_recs)

    write_manifest(test_recs, out / f"fold{k}_test_manifest.csv")
    with open(out / f"fold{k}_demographics.json", "w") as fh:
        json.dump({"train": demographic_summary(tr), "val": demographic_summary(va),
                   "test": demographic_summary(test_recs)}, fh, indent=2)

    shape = tuple(np.load(tr[0].path, mmap_mode="r").shape)
    mk = lambda recs, aug, shuf: DataLoader(
        ADNIDataset(recs, augment=aug, target_shape=shape),
        batch_size=args.batch_size, shuffle=shuf, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    dl_tr, dl_va, dl_te = mk(tr, args.augment, True), mk(va, False, False), mk(test_recs, False, False)

    model = resnet3d_10(n_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(tr, 2).to(device))
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_auc, best_epoch, bad, history = -1.0, -1, 0, []
    ckpt = out / f"fold{k}.pt"
    for epoch in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        yv, pv = predict(model, dl_va, device, amp)
        v_auc = auc(yv, pv)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1),
                        "val_auc": v_auc, "val_acc": float(((pv > 0.5) == yv).mean()),
                        "seconds": round(time.time() - t0, 1)})
        print(f"  fold {k} epoch {epoch:3d}  loss {total / max(n, 1):.4f}  "
              f"val AUC {v_auc:.4f}  ({history[-1]['seconds']}s)", flush=True)
        if v_auc > best_auc + 1e-4:
            best_auc, best_epoch, bad = v_auc, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                break

    with open(out / f"fold{k}_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    yt, pt = predict(model, dl_te, device, amp)
    res = {"fold": k, "best_epoch": best_epoch, "val_auc": best_auc,
           "test_auc": auc(yt, pt), "test_acc": float(((pt > 0.5) == yt).mean()),
           "n_train": len(tr), "n_val": len(va), "n_test": len(test_recs)}
    print(f"  fold {k} TEST  AUC {res['test_auc']:.4f}  acc {res['test_acc']:.4f}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--folds", type=int, nargs="*", default=None,
                    help="subset of folds to run (pilot: --folds 0)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--augment", action="store_true",
                    help="shift / intensity augmentation (ADNIDataset._augment)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    records = read_manifest(args.manifest)
    print(f"{len(records)} scans, {len({r.subject_id for r in records})} subjects, "
          f"device {device}")

    results = []
    for k, (train_recs, test_recs) in enumerate(
            subject_level_splits(records, n_folds=args.n_folds, seed=args.seed)):
        if args.folds is not None and k not in args.folds:
            continue
        results.append(train_fold(k, train_recs, test_recs, args, device, out))

    aucs = [r["test_auc"] for r in results]
    summary = {"manifest": os.path.abspath(args.manifest), "config": vars(args),
               "folds": results, "test_auc_mean": float(np.nanmean(aucs)),
               "test_auc_std": float(np.nanstd(aucs))}
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"test AUC {summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f} "
          f"over {len(results)} fold(s)")


if __name__ == "__main__":
    main()
