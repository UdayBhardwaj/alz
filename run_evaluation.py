"""
Main evaluation driver.

Mirrors ``run_xai_eval`` from the 2D FER study: iterate a held-out set, compute
every explanation method on every volume, and score each map on the paper's
three axes:

    faithfulness   insertion / deletion AUC, against a random-map baseline
    plausibility   relevance mass and enrichment in the atlas hippocampus
    correctness    agreement with the benchmark's ground-truth maps
                   (atrophy and shortcut), when a benchmark manifest is given

Two design decisions carried over deliberately:
  - explanations target the model's own prediction, not the ground-truth label;
  - the random-saliency baseline is computed alongside, not as an afterthought.

Every (volume, method) pair is written as one row of ``per_volume.csv`` so
statistics can be computed over subjects, not over pooled maps.

Usage
-----
    python run_evaluation.py --checkpoint runs/atrophy/fold0.pt \\
        --manifest runs/atrophy/fold0_test_manifest.csv \\
        --bench bench/atrophy/manifest.csv --out results/atrophy_fold0

``--manifest`` fixes *which* subjects are evaluated (the fold's held-out
set); ``--bench`` supplies their volumes and ground truth. Pointing ``--bench``
at a ``--flip-shortcut`` build of the same cohort evaluates the same subjects
with the shortcut moved to the other class: the reliance test.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from data.adni_dataset import ADNIDataset, ScanRecord, read_manifest
from eval.anatomical import ground_truth_scores
from eval.faithfulness import insertion_deletion_auc, random_baseline_auc
from models.resnet3d import resnet3d_10
from xai.baselines import GradientInput3D, IntegratedGradients3D, Occlusion3D
from xai.gradcam3d import GradCAM3D, GradCAMPlusPlus3D
from xai.lrp3d import LRP3D

CLASS_NAMES = ["CN", "AD"]


def build_explainers(model: torch.nn.Module, occlusion: bool = True) -> Dict[str, object]:
    """Instantiate every attribution method under comparison."""
    ex = {
        "LRP":        LRP3D(model, input_bounds=(-4.0, 4.0)),
        "GradCAM":    GradCAM3D(model),
        "GradCAM++":  GradCAMPlusPlus3D(model),
        "IG":         IntegratedGradients3D(model, n_steps=32),
        "GradInput":  GradientInput3D(model),
    }
    if occlusion:
        ex["Occlusion"] = Occlusion3D(model)
    return ex


def load_bench(path: str) -> Dict[str, Dict[str, str]]:
    """subject_id -> benchmark row, with paths made absolute."""
    base = os.path.dirname(os.path.abspath(path))
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for key in ("path", "atrophy_gt", "shortcut_gt"):
                if r.get(key) and not os.path.isabs(r[key]):
                    r[key] = os.path.join(base, r[key])
            rows[r["subject_id"]] = r
    return rows


def _gt(saliency: np.ndarray, gt_path: Optional[str], brain: np.ndarray,
        prefix: str) -> Dict[str, float]:
    if not gt_path:
        return {}
    gt = np.load(gt_path)
    if not (gt[brain] > 0).any():
        return {}
    s = ground_truth_scores(saliency, gt, brain)
    return {f"{prefix}_{k}": v for k, v in s.items()}


def evaluate(model: torch.nn.Module,
             dataset: ADNIDataset,
             bench_rows: Optional[Dict[str, Dict[str, str]]] = None,
             region_mask: Optional[np.ndarray] = None,
             patch: int = 8,
             n_steps: int = 50,
             max_volumes: Optional[int] = None,
             occlusion: bool = True,
             seed: int = 0) -> Dict[str, object]:
    device = next(model.parameters()).device
    explainers = build_explainers(model, occlusion)

    rows: List[Dict[str, object]] = []
    faith: Dict[str, List] = defaultdict(list)
    maps_cache: Dict[str, np.ndarray] = {}
    n = len(dataset) if max_volumes is None else min(max_volumes, len(dataset))
    print(f"Evaluating {n} volumes x {len(explainers)} methods")

    for i in range(n):
        rec = dataset.records[i]
        x, label = dataset[i]
        x = x.unsqueeze(0).to(device)
        brain = x[0, 0].cpu().numpy() != 0

        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0].cpu().numpy()
        pred = int(prob.argmax())
        bench = bench_rows.get(rec.subject_id, {}) if bench_rows else {}

        sal_by_method = {name: fn(x, pred)[0] for name, fn in explainers.items()}
        sal_by_method["Random"] = (np.random.default_rng(seed + i)
                                   .random(x.shape[2:]).astype(np.float32))

        for name, sal in sal_by_method.items():
            if name == "Random":
                res = random_baseline_auc(model, x, pred, patch=patch,
                                          n_steps=n_steps, seed=seed + i)
            else:
                res = insertion_deletion_auc(model, x, sal, pred,
                                             patch=patch, n_steps=n_steps)
            faith[name].append(res)
            row = {"subject_id": rec.subject_id, "label": label, "pred": pred,
                   "prob_ad": float(prob[1]), "method": name,
                   "shortcut": bench.get("shortcut", ""),
                   "insertion": res.insertion_auc, "deletion": res.deletion_auc,
                   "faithfulness": res.score}
            if region_mask is not None:
                p = ground_truth_scores(sal, region_mask.astype(np.float32), brain)
                row.update({"plaus_mass": p["mass_fraction"],
                            "plaus_enrichment": p["enrichment"]})
            row.update(_gt(sal, bench.get("atrophy_gt"), brain, "atr"))
            row.update(_gt(sal, bench.get("shortcut_gt"), brain, "sc"))
            rows.append(row)
            if i == 0:
                maps_cache[name] = sal

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{n}", flush=True)

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
    labels = np.array([r["label"] for r in rows if r["method"] == "Random"])
    preds = np.array([r["pred"] for r in rows if r["method"] == "Random"])
    return {"faithfulness": summary, "rows": rows, "example_maps": maps_cache,
            "accuracy": float((labels == preds).mean()) if len(labels) else float("nan")}


def print_table(summary: Dict[str, Dict[str, float]]) -> None:
    """Console version of the paper's faithfulness table."""
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
                    help="CSV with columns: path, subject_id, label (the subjects to evaluate)")
    ap.add_argument("--bench", default=None,
                    help="benchmark manifest supplying volumes and ground truth by subject_id")
    ap.add_argument("--region-mask", default=None,
                    help="hippocampus mask .npy for plausibility; defaults to the "
                         "benchmark's gt/region_mask.npy")
    ap.add_argument("--out", default="results")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max-volumes", type=int, default=None)
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    records = read_manifest(args.manifest)
    bench_rows = None
    region = None
    if args.bench:
        bench_rows = load_bench(args.bench)
        missing = [r.subject_id for r in records if r.subject_id not in bench_rows]
        if missing:
            raise SystemExit(f"{len(missing)} subjects missing from --bench, e.g. {missing[:3]}")
        records = [ScanRecord(path=bench_rows[r.subject_id]["path"],
                              subject_id=r.subject_id,
                              label=int(bench_rows[r.subject_id]["label"]),
                              age=r.age, sex=r.sex) for r in records]
        default_region = Path(args.bench).parent / "gt" / "region_mask.npy"
        if args.region_mask is None and default_region.exists():
            args.region_mask = str(default_region)
    if args.region_mask:
        region = np.load(args.region_mask).astype(bool)

    device = torch.device(args.device)
    model = resnet3d_10(n_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()

    shape = tuple(np.load(records[0].path, mmap_mode="r").shape)
    ds = ADNIDataset(records, augment=False, target_shape=shape)
    out = evaluate(model, ds, bench_rows, region, patch=args.patch,
                   n_steps=args.steps, max_volumes=args.max_volumes,
                   occlusion=not args.no_occlusion)

    print_table(out["faithfulness"])
    print(f"accuracy on evaluated volumes: {out['accuracy']:.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out, "faithfulness.json"), "w") as fh:
        json.dump({"faithfulness": out["faithfulness"], "accuracy": out["accuracy"]},
                  fh, indent=2)
    keys = sorted({k for r in out["rows"] for k in r},
                  key=lambda k: (k not in ("subject_id", "label", "pred", "prob_ad",
                                           "method", "shortcut"), k))
    with open(os.path.join(args.out, "per_volume.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out["rows"])
    np.savez_compressed(os.path.join(args.out, "example_maps.npz"),
                        **out["example_maps"])
    print(f"Written to {args.out}/")


if __name__ == "__main__":
    main()
