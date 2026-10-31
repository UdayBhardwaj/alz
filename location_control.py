"""
Two controls for the shortcut analysis, written to ``extra_numbers.json``.

1. Location control. The shortcut sits at the same place in every scan, so a
   method that simply favours that place would score well against the
   shortcut ground truth without detecting the bump. Explaining the
   atrophy-only model (which never saw a bump) against the same location gives
   each method's baseline lift there. Produce the evaluation first with

       python run_evaluation.py ... --bench bench/atrophy/manifest_loc.csv \\
           --out results_full/atrophy_loc/fold<k>

   where manifest_loc.csv points every shortcut_gt at the shortcut location
   mask.

2. Reliance at the logit. Flip accuracy only shows whether decisions change.
   The mean shift of the logit towards the bump's class, between each subject
   with and without the bump, measures how much the model's score depends on
   it even when decisions do not change.

    python location_control.py --results <work>/results_full --out <work>/analysis_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import METHODS, boot_ci


def _load(root: Path, tag: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in sorted((root / tag).glob("fold*/per_volume.csv"))],
                     ignore_index=True)


def _logit(p: pd.Series) -> np.ndarray:
    p = np.clip(p.to_numpy(float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def logit_shift(root: Path, cond: str) -> dict:
    """Shift of the logit towards the class the bump indicates, over subjects
    whose bump status differs between the set and its flipped copy."""
    a, b = (_load(root, t) for t in (cond, f"{cond}_flip"))
    a = a[a.method == "Random"].set_index("subject_id")      # one row per subject
    b = b[b.method == "Random"].set_index("subject_id")
    j = a[["prob_ad", "shortcut", "label", "pred"]].join(
        b[["prob_ad", "shortcut", "pred"]].add_suffix("_f"))
    ch = j[j.shortcut != j.shortcut_f]
    d = _logit(ch.prob_ad_f) - _logit(ch.prob_ad)
    s = np.where(ch.shortcut_f == 1, d, -d)
    return {"n_changed": int(len(ch)), "logit_shift": boot_ci(s),
            "median_abs_logit": float(np.median(np.abs(_logit(a.prob_ad)))),
            "decisions_changed": float((ch.pred != ch.pred_f).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reliance", nargs="+", default=["shortcut", "both_p90"],
                    help="conditions with a flipped test set")
    args = ap.parse_args()
    root, out = Path(args.results), Path(args.out)

    loc = _load(root, "atrophy_loc")
    loc["lift"] = loc.sc_auprc / loc.sc_auprc_chance
    numbers = {"location_lift_atrophy_model": {m: boot_ci(loc[loc.method == m].lift.dropna())
                                               for m in METHODS},
               "n_subjects_location": int(loc.subject_id.nunique()),
               "reliance": {c: logit_shift(root, c) for c in args.reliance}}
    (out / "extra_numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
