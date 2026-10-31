"""
Sanity checks (Adebayo et al. 2018) on the benchmark's trained models.

Two tests, both scored by Spearman correlation with the original map:

  cascading model randomisation
      Re-initialise the atrophy model from the output layer backwards, one
      stage at a time, and recompute each method's map. A method that depends
      on what the model learned decorrelates quickly; one that stays near 1
      reflects the input, not the model.

  data randomisation
      Explain the same atrophy scans with the resampling-control model of the
      same fold. That model was trained on labels unrelated to the images, so
      it is the permuted-label model of the original test without extra
      training. High similarity means the map reflects image structure rather
      than anything learned about the label.

    python run_sanity.py --work <work> --runs runs_full --out <work>/analysis_full \\
        --per-fold 20
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from analyze import boot_ci
from data.adni_dataset import ADNIDataset, ScanRecord, read_manifest
from eval.sanity import cascading_randomization, rank_correlation, ssim_3d
from models.resnet3d import resnet3d_10
from run_evaluation import load_bench
from xai.baselines import GradientInput3D, IntegratedGradients3D, Occlusion3D
from xai.gradcam3d import GradCAM3D, GradCAMPlusPlus3D
from xai.lrp3d import LRP3D

STAGES = ["fc", "layer4", "layer3", "layer2", "layer1", "stem"]


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    m = resnet3d_10(n_classes=2).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.eval()


CONSTRUCTORS = {
    "LRP":       lambda m: LRP3D(m, input_bounds=(-4.0, 4.0)),
    "GradCAM":   GradCAM3D,
    "GradCAM++": GradCAMPlusPlus3D,
    "IG":        lambda m: IntegratedGradients3D(m, n_steps=32),
    "GradInput": GradientInput3D,
    "Occlusion": Occlusion3D,
}   # same settings as run_evaluation.build_explainers, built one at a time


def factory(name: str):
    """Explainer factory for one method: builds only that method on the given
    model, runs it once and removes its hooks, so no hook outlives the call
    (a leftover Grad-CAM hook would also make the model impossible to copy)."""
    def make(model):
        def run(x, cls):
            ex = CONSTRUCTORS[name](model)
            try:
                return ex(x, cls)
            finally:
                if hasattr(ex, "remove"):
                    ex.remove()
        return run
    return make


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--runs", default="runs_full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--per-fold", type=int, default=20, help="test subjects explained per fold")
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    bench = load_bench(str(work / "bench" / "atrophy" / "manifest.csv"))
    methods = ["LRP", "GradCAM", "GradCAM++", "IG", "GradInput"] + ([] if args.no_occlusion else ["Occlusion"])

    rows = []
    for k in args.folds:
        runs = work / args.runs
        model = load_model(runs / "atrophy" / f"fold{k}.pt", device)
        control = load_model(runs / "null" / f"fold{k}.pt", device)
        recs = read_manifest(str(runs / "atrophy" / f"fold{k}_test_manifest.csv"))[: args.per_fold]
        recs = [ScanRecord(path=bench[r.subject_id]["path"], subject_id=r.subject_id,
                           label=int(bench[r.subject_id]["label"])) for r in recs]
        ds = ADNIDataset(recs, augment=False, target_shape=(96, 96, 96))
        for i, rec in enumerate(ds.records):
            x = ds[i][0].unsqueeze(0).to(device)
            with torch.no_grad():
                cls = int(model(x).argmax(1))
            for m in methods:
                f = factory(m)
                casc = cascading_randomization(model, x, f, STAGES, class_idx=cls, seed=k)
                for stage, rho in casc.items():
                    rows.append({"fold": k, "subject_id": rec.subject_id, "method": m,
                                 "test": "cascading", "stage": stage, "spearman": rho})
                a, _ = f(model)(x, cls)
                b, _ = f(control)(x, cls)
                rows.append({"fold": k, "subject_id": rec.subject_id, "method": m,
                             "test": "data", "stage": "control_model",
                             "spearman": rank_correlation(a, b), "ssim": ssim_3d(a, b)})
            print(f"fold {k}: {i + 1}/{len(ds)}", flush=True)

    with open(out / "sanity_per_volume.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fold", "subject_id", "method", "test", "stage", "spearman", "ssim"])
        w.writeheader()
        w.writerows(rows)

    summary = {"n_subjects": len({r["subject_id"] for r in rows}), "cascading": {}, "data": {}}
    # A fully randomised model can give a constant map, whose rank correlation
    # is undefined (NaN). boot_ci drops those; their share is reported so a
    # "decorrelated" stage cannot hide a degenerate map.
    summary["constant_map_fraction"] = {}
    for m in methods:
        summary["cascading"][m] = {s: boot_ci([r["spearman"] for r in rows if r["method"] == m
                                               and r["test"] == "cascading" and r["stage"] == s])
                                   for s in STAGES}
        summary["constant_map_fraction"][m] = {
            s: float(np.mean([r["spearman"] != r["spearman"] for r in rows if r["method"] == m
                              and r["test"] == "cascading" and r["stage"] == s])) for s in STAGES}
        d = [r for r in rows if r["method"] == m and r["test"] == "data"]
        summary["data"][m] = {"spearman": boot_ci([r["spearman"] for r in d]),
                              "ssim": boot_ci([r["ssim"] for r in d])}
    (out / "sanity_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'Method':<11}" + "".join(f"{s:>9}" for s in STAGES) + f"{'data':>9}")
    for m in methods:
        print(f"{m:<11}" + "".join(f"{summary['cascading'][m][s][0]:>9.2f}" for s in STAGES)
              + f"{summary['data'][m]['spearman'][0]:>9.2f}")


if __name__ == "__main__":
    main()
