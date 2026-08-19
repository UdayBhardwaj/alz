"""
Main evaluation driver.

Mirrors ``run_xai_eval`` from the 2D FER study: iterate a held-out set, compute
every explanation method on every volume, and accumulate faithfulness,
plausibility and sanity statistics into the results tables.

Two design decisions carried over deliberately:
  - explanations target the model's own prediction, not the ground-truth label;
  - the random-saliency baseline is computed alongside, not as an afterthought.

Usage
-----
    python run_evaluation.py --checkpoint runs/fold0.pt \\
                             --manifest data/test_manifest.csv \\
                             --out results/fold0
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data.adni_dataset import ADNIDataset, ScanRecord
from eval.faithfulness import insertion_deletion_auc, random_baseline_auc
from eval.sanity import cascading_randomization
from models.resnet3d import resnet3d_10
from xai.baselines import GradientInput3D, IntegratedGradients3D
from xai.gradcam3d import GradCAM3D, GradCAMPlusPlus3D
from xai.lrp3d import LRP3D

CLASS_NAMES = ["CN", "AD"]


def build_explainers(model: torch.nn.Module) -> Dict[str, object]:
    """Instantiate every attribution method under comparison."""
    return {
        "LRP":        LRP3D(model, input_bounds=(-4.0, 4.0)),
        "GradCAM":    GradCAM3D(model),
        "GradCAM++":  GradCAMPlusPlus3D(model),
        "IG":         IntegratedGradients3D(model, n_steps=32),
        "GradInput":  GradientInput3D(model),
    }


def evaluate(model: torch.nn.Module,
             dataset: ADNIDataset,
             patch: int = 8,
             n_steps: int = 50,
             max_volumes: int = None,
             seed: int = 0) -> Dict[str, object]:
    device = next(model.parameters()).device
    explainers = build_explainers(model)

    faith: Dict[str, List] = defaultdict(list)
    maps_cache: Dict[str, np.ndarray] = {}
    n = len(dataset) if max_volumes is None else min(max_volumes, len(dataset))

    print(f"Evaluating {n} volumes x {len(explainers)} methods")

    for i in range(n):
        x, label = dataset[i]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = int(model(x).argmax(1).item())

        for name, fn in explainers.items():
            sal, _ = fn(x, pred)
            res = insertion_deletion_auc(model, x, sal, pred,
                                         patch=patch, n_steps=n_steps)
            faith[name].append(res)
            if i == 0:
                maps_cache[name] = sal

        faith["Random"].append(
            random_baseline_auc(model, x, pred, patch=patch,
                                n_steps=n_steps, seed=seed + i)
        )

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}")

    for fn in explainers.values():
        if hasattr(fn, "remove"):
            fn.remove()

    summary = {}
    for name, results in faith.items():
        ins = [r.insertion_auc for r in results]
        dele = [r.deletion_auc for r in results]
        summary[name] = {
            "insertion_mean": float(np.mean(ins)),
            "insertion_std": float(np.std(ins)),
            "deletion_mean": float(np.mean(dele)),
            "deletion_std": float(np.std(dele)),
            "score": float(np.mean(ins) - np.mean(dele)),
            "n": len(results),
        }
    return {"faithfulness": summary, "example_maps": maps_cache}


def print_table(summary: Dict[str, Dict[str, float]]) -> None:
    """Console version of the paper's Table 2."""
    print(f"\n{'Method':<12}{'Insertion':>18}{'Deletion':>18}{'Score':>10}")
    print("-" * 58)
    order = sorted(summary, key=lambda k: -summary[k]["score"])
    for name in order:
        s = summary[name]
        print(f"{name:<12}"
              f"{s['insertion_mean']*100:>11.2f} ±{s['insertion_std']*100:>5.2f}"
              f"{s['deletion_mean']*100:>11.2f} ±{s['deletion_std']*100:>5.2f}"
              f"{s['score']*100:>+10.2f}")
    print("-" * 58)
    print("Insertion higher is better; Deletion lower is better.")
    print("Any method not clearly above Random is not explaining anything.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True,
                    help="CSV with columns: path, subject_id, label")
    ap.add_argument("--out", default="results")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max-volumes", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import csv
    with open(args.manifest) as fh:
        records = [ScanRecord(path=r["path"], subject_id=r["subject_id"],
                              label=int(r["label"]))
                   for r in csv.DictReader(fh)]

    device = torch.device(args.device)
    model = resnet3d_10(n_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()

    ds = ADNIDataset(records, augment=False)
    out = evaluate(model, ds, patch=args.patch, n_steps=args.steps,
                   max_volumes=args.max_volumes)

    print_table(out["faithfulness"])

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out, "faithfulness.json"), "w") as fh:
        json.dump(out["faithfulness"], fh, indent=2)
    np.savez_compressed(os.path.join(args.out, "example_maps.npz"),
                        **out["example_maps"])
    print(f"Written to {args.out}/")


if __name__ == "__main__":
    main()
