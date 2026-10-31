"""
Turn evaluation outputs into the paper's tables, statistics and figures.

Expects a results root laid out by ``run_pilot.py``:

    <root>/<condition>/fold<k>/per_volume.csv        evaluation on the condition itself
    <root>/<condition>_flip/fold<k>/per_volume.csv   same subjects, shortcut moved
    <runs>/<condition>/summary.json                  training summary

Statistics are computed over subjects: each subject contributes one value per
method (there is one scan per IXI subject), and confidence intervals are
percentile bootstraps over subjects. Correctness is reported as AUPRC lift,
AUPRC divided by its chance level (the ground-truth support fraction), so 1.0
means no better than a random map whatever the size of the target.

Every number written to the tables also goes to ``key_numbers.json``; the
manuscript quotes that file, never a retyped value.

    python analyze.py --results results --runs runs --out analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METHODS = ["LRP", "GradCAM", "GradCAM++", "IG", "GradInput", "Occlusion", "Random"]
TITLES = {"null": "Resampling control (null)", "atrophy": "A: atrophy only",
          "shortcut": "C: shortcut only", "both_p90": "B: atrophy + shortcut, p = 0.9",
          "both_p70": "B: atrophy + shortcut, p = 0.7", "both_p100": "B: atrophy + shortcut, p = 1.0",
          "both_p90_amp10": "B: p = 0.9, bump amplitude 1.0"}


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> List[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def load(root: Path, tag: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((root / tag).glob("fold*/per_volume.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for k in ("atr", "sc"):
        if f"{k}_auprc" in df:
            df[f"{k}_lift"] = df[f"{k}_auprc"] / df[f"{k}_auprc_chance"]
    return df


def fmt(m: List[float], digits: int = 3) -> str:
    return "–" if not np.isfinite(m[0]) else f"{m[0]:.{digits}f} [{m[1]:.{digits}f}, {m[2]:.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="*", default=["null", "atrophy", "shortcut", "both_p90"])
    args = ap.parse_args()

    root, runs, out = Path(args.results), Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key: Dict[str, object] = {}
    md: List[str] = []

    # --- classification and reliance --------------------------------------
    md += ["## Classification and shortcut reliance", "",
           "| Condition | Test AUC (mean ± sd over folds) | Acc on evaluated set | Acc with shortcut flipped |",
           "|---|---|---|---|"]
    for c in args.conditions:
        summ = runs / c / "summary.json"
        auc = json.loads(summ.read_text()) if summ.exists() else {}
        df, dff = load(root, c), load(root, f"{c}_flip")
        acc = lambda d: float((d[d.method == "Random"].label == d[d.method == "Random"].pred).mean()) if len(d) else float("nan")
        a, af = acc(df), acc(dff)
        key[c] = {"test_auc_mean": auc.get("test_auc_mean"), "test_auc_std": auc.get("test_auc_std"),
                  "n_folds": len(auc.get("folds", [])), "acc_eval": a, "acc_flipped": af,
                  "n_subjects_eval": int(df.subject_id.nunique()) if len(df) else 0}
        md.append(f"| {c} | {auc.get('test_auc_mean', float('nan')):.3f} ± {auc.get('test_auc_std', float('nan')):.3f} "
                  f"| {a:.3f} | {'–' if not np.isfinite(af) else f'{af:.3f}'} |")
    md.append("")

    # --- three axes per method ----------------------------------------------
    for c in args.conditions:
        df = load(root, c)
        if df.empty:
            continue
        md += [f"## Condition `{c}`: per-method scores (mean [95% CI over subjects])", "",
               "| Method | Faithfulness (ins − del) | Plausibility (hippocampus mass) | Plausibility enrichment | Atrophy AUPRC lift | Shortcut AUPRC lift |",
               "|---|---|---|---|---|---|"]
        key[c]["methods"] = {}
        for m in METHODS:
            d = df[df.method == m]
            if d.empty:
                continue
            row = {"faithfulness": boot_ci(d.faithfulness),
                   "plaus_mass": boot_ci(d.get("plaus_mass", pd.Series(dtype=float))),
                   "plaus_enrichment": boot_ci(d.get("plaus_enrichment", pd.Series(dtype=float))),
                   "atr_lift": boot_ci(d.get("atr_lift", pd.Series(dtype=float)).dropna()),
                   "sc_lift": boot_ci(d.get("sc_lift", pd.Series(dtype=float)).dropna())}
            # H1: per-volume plausibility vs faithfulness
            pm = d[["plaus_mass", "faithfulness"]].dropna() if "plaus_mass" in d else pd.DataFrame()
            if len(pm) > 3:
                rho, p = spearmanr(pm.plaus_mass, pm.faithfulness)
                row["rho_plaus_faith"], row["p_plaus_faith"] = float(rho), float(p)
            key[c]["methods"][m] = row
            md.append(f"| {m} | {fmt(row['faithfulness'], 4)} | {fmt(row['plaus_mass'])} | "
                      f"{fmt(row['plaus_enrichment'], 2)} | {fmt(row['atr_lift'], 2)} | {fmt(row['sc_lift'], 2)} |")
        md.append("")
        md += ["Spearman ρ (plausibility vs faithfulness, per volume): "
               + ", ".join(f"{m} {v['rho_plaus_faith']:+.2f}" for m, v in key[c]["methods"].items()
                           if "rho_plaus_faith" in v), ""]

    (out / "key_numbers.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # --- figure: plausibility vs correctness per method -----------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in args.conditions if c in key and key[c].get("methods")]
    if conds:
        ncol = 2 if len(conds) > 2 else len(conds)
        nrow = -(-len(conds) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), squeeze=False)
        series = [("plaus_enrichment", "Plausibility (hippocampus enrichment)", "#4B6CB7"),
                  ("atr_lift", "Correctness: atrophy (AUPRC lift)", "#2E7D5B"),
                  ("sc_lift", "Correctness: shortcut (AUPRC lift)", "#E8743B")]
        for ax, c in zip(axes.flat, conds):
            ms = [m for m in METHODS if m in key[c]["methods"]]
            x = np.arange(len(ms))
            for j, (k, lab, col) in enumerate(series):
                v = np.array([key[c]["methods"][m][k] for m in ms], dtype=float)   # mean, lo, hi
                if not np.isfinite(v[:, 0]).any():
                    continue
                err = np.nan_to_num(np.abs(v[:, 1:].T - v[:, 0]))
                ax.bar(x + (j - 1) * 0.27, np.nan_to_num(v[:, 0]), 0.27, yerr=err,
                       error_kw=dict(lw=0.7, capsize=1.5, ecolor="#333333"), label=lab, color=col)
            ax.axhline(1.0, color="#5F6378", lw=0.8, ls="--")
            ax.set_xticks(x, ms, rotation=35, ha="right", fontsize=9)
            ax.set_title(TITLES.get(c, c), fontsize=11)
            ax.set_ylabel("× chance (1.0 = random map)", fontsize=9)
        for ax in list(axes.flat)[len(conds):]:
            ax.axis("off")
        handles, labels = [], []
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out / "fig_axes_by_method.png", dpi=150)
    print(f"written {out}/tables.md, key_numbers.json, fig_axes_by_method.png")


if __name__ == "__main__":
    main()
