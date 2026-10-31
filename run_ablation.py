"""
Occlusion window-size ablation (E7).

Occlusion is the strongest method on most axes, and its maps are coarse: a
16^3 window slid at stride 8. This re-runs it at windows 8, 16 and 32 (stride
half the window) to see how much of its plausibility, correctness and
faithfulness comes from the window size.

  atrophy model, atrophy scans   plausibility, atrophy correctness,
                                 faithfulness, and lift at the shortcut
                                 location (the no-bump baseline of E9)
  B model (p = 0.9), B scans     shortcut correctness

    python run_ablation.py --work <work> --out <work>/analysis_full --per-fold 20
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
from eval.anatomical import ground_truth_scores
from eval.faithfulness import insertion_deletion_auc
from run_evaluation import load_bench
from run_sanity import load_model
from xai.baselines import Occlusion3D

WINDOWS = [8, 16, 32]


def lift(sal, gt, brain):
    s = ground_truth_scores(sal, gt.astype(np.float32), brain)
    return s["auprc"] / s["auprc_chance"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--runs", default="runs_full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--per-fold", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    work, out = Path(args.work), Path(args.out)
    device = torch.device(args.device)
    region = np.load(work / "bench" / "atrophy" / "gt" / "region_mask.npy").astype(bool)
    location = np.load(work / "bench" / "atrophy" / "gt" / "shortcut_location.npy").astype(bool)

    rows = []
    for k in args.folds:
        for cond in ("atrophy", "both_p90"):
            bench = load_bench(str(work / "bench" / cond / "manifest.csv"))
            model = load_model(work / args.runs / cond / f"fold{k}.pt", device)
            recs = read_manifest(str(work / args.runs / cond / f"fold{k}_test_manifest.csv"))[: args.per_fold]
            recs = [ScanRecord(path=bench[r.subject_id]["path"], subject_id=r.subject_id,
                               label=int(bench[r.subject_id]["label"])) for r in recs]
            ds = ADNIDataset(recs, augment=False, target_shape=region.shape)
            for i, rec in enumerate(ds.records):
                x = ds[i][0].unsqueeze(0).to(device)
                brain = x[0, 0].cpu().numpy() != 0
                with torch.no_grad():
                    cls = int(model(x).argmax(1))
                b = bench[rec.subject_id]
                for w in WINDOWS:
                    sal, _ = Occlusion3D(model, window=w, stride=w // 2)(x, cls)
                    row = {"fold": k, "condition": cond, "subject_id": rec.subject_id, "window": w}
                    if cond == "atrophy":
                        p = ground_truth_scores(sal, region.astype(np.float32), brain)
                        row["plaus_enrichment"] = p["enrichment"]
                        row["faithfulness"] = insertion_deletion_auc(model, x, sal, cls, patch=8, n_steps=50).score
                        row["atr_lift"] = lift(sal, np.load(b["atrophy_gt"]), brain) \
                            if b["label"] == "1" else float("nan")
                        row["location_lift"] = lift(sal, location, brain)
                    elif b["shortcut"] == "1":
                        row["sc_lift"] = lift(sal, np.load(b["shortcut_gt"]), brain)
                    rows.append(row)
            print(f"fold {k} {cond}: {len(ds)} done", flush=True)

    keys = ["fold", "condition", "subject_id", "window", "plaus_enrichment", "faithfulness",
            "atr_lift", "location_lift", "sc_lift"]
    with open(out / "ablation_per_volume.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    summary = {}
    for w in WINDOWS:
        sel = lambda c, key: [r[key] for r in rows if r["window"] == w and r["condition"] == c
                              and key in r and r[key] == r[key]]
        summary[w] = {"plaus_enrichment": boot_ci(sel("atrophy", "plaus_enrichment")),
                      "faithfulness": boot_ci(sel("atrophy", "faithfulness")),
                      "atr_lift": boot_ci(sel("atrophy", "atr_lift")),
                      "location_lift": boot_ci(sel("atrophy", "location_lift")),
                      "sc_lift_B": boot_ci(sel("both_p90", "sc_lift")),
                      "n_atrophy": len(sel("atrophy", "faithfulness"))}
    (out / "ablation_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for w, s in summary.items():
        print(w, {k: (round(v[0], 3) if isinstance(v, list) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
