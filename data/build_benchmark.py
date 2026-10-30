"""
Build one condition of the ground-truth benchmark from preprocessed IXI scans.

Conditions (see the experiment table in the paper plan):

    atrophy   label 1 = simulated hippocampal atrophy, no shortcut anywhere
    shortcut  label 1 = planted intensity bump, no atrophy anywhere
    both      label 1 = atrophy; the bump is present in a fraction
              ``--p-shortcut`` of label-1 and ``1 - p`` of label-0 volumes
    null      resampling control: no atrophy, no shortcut, labels still
              assigned; every scan is warped exactly as in the other
              conditions, so a model must stay at chance unless the warp
              itself leaks the label

Every volume, of either class, is resampled exactly once through the same
jitter-plus-atrophy warp, so interpolation is not a label cue (see
``simulate.py``).

For the shortcut-reliance test, build the test set a second time with
``--flip-shortcut``: the bump is moved to the opposite class. A model that
relies on atrophy keeps its accuracy; one that relies on the bump collapses.

Usage
-----
    python -m data.build_benchmark --npy-dir data/ixi_npy \\
        --demographics data/IXI.csv --region-mask data/hippocampus_96.npy \\
        --condition both --p-shortcut 0.9 --out bench/both_p90
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from data.ixi_dataset import IXISubject, ixi_subjects
from data.simulate import (AtrophyConfig, ShortcutConfig, assign_labels,
                           assign_shortcuts, default_shortcut_center,
                           simulate_volume)

__all__ = ["CONDITIONS", "build_condition", "hippocampus_mask_on_grid"]

CONDITIONS = ("atrophy", "shortcut", "both", "null")

MANIFEST_FIELDS = ["path", "subject_id", "label", "site", "age", "sex",
                   "volume_loss_target", "volume_loss_achieved",
                   "min_jacobian", "shortcut", "atrophy_gt", "shortcut_gt"]


def hippocampus_mask_on_grid(target_shape: Tuple[int, int, int] = (96, 96, 96),
                             mni_template: Optional[str] = None) -> np.ndarray:
    """Harvard-Oxford left + right hippocampus on the analysis grid.

    Reproduces the geometry of ``data/preprocess.py``: atlas resampled into
    the MNI template that registration targets, then zoomed to
    ``target_shape`` exactly as the volumes are. Requires nilearn.

    Check the result on one preprocessed volume before building a cohort.
    If the mask does not sit on the hippocampus, every ground-truth number
    downstream is wrong, and nothing else in the pipeline would notice.
    """
    from nilearn import datasets, image
    from scipy.ndimage import zoom

    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    labels = list(ho.labels)
    tmpl = (image.load_img(mni_template) if mni_template
            else datasets.load_mni152_template(resolution=2))   # same grid as preprocess
    atlas = image.resample_to_img(ho.maps, tmpl, interpolation="nearest")
    arr = np.asarray(atlas.dataobj)

    ids = [labels.index(n) for n in ("Left Hippocampus", "Right Hippocampus")]
    mask = np.isin(arr, ids).astype(np.float32)
    factors = [s / a for s, a in zip(target_shape, mask.shape)]
    return zoom(mask, factors, order=0) > 0.5


def build_condition(subjects: Sequence[IXISubject],
                    region_mask: np.ndarray,
                    condition: str,
                    out_dir: str,
                    p_shortcut: float = 1.0,
                    flip_shortcut: bool = False,
                    seed: int = 0,
                    atrophy_cfg: Optional[AtrophyConfig] = None,
                    shortcut_cfg: Optional[ShortcutConfig] = None) -> Path:
    """Generate volumes, ground-truth maps and a manifest for one condition.

    Each IXI subject appears once, as either a control or a positive, so the
    raw data's one-scan-per-subject property is preserved and splits by
    subject_id remain leakage-free. Returns the manifest path; its
    ``path, subject_id, label`` columns are what ``run_evaluation.py`` reads.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if flip_shortcut and condition in ("atrophy", "null"):
        raise ValueError("flip_shortcut needs a condition with a shortcut")
    acfg = atrophy_cfg or AtrophyConfig()
    scfg = shortcut_cfg or ShortcutConfig()
    rng = np.random.default_rng(seed)

    out = Path(out_dir)
    (out / "volumes").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(exist_ok=True)

    subjects = list(subjects)
    labels = assign_labels([s.site for s in subjects],
                           [s.age for s in subjects], seed=seed)

    if condition == "atrophy":
        has_atrophy, has_shortcut = labels == 1, np.zeros_like(labels)
    elif condition == "shortcut":
        has_atrophy, has_shortcut = np.zeros_like(labels, bool), labels.copy()
    elif condition == "null":
        has_atrophy, has_shortcut = np.zeros_like(labels, bool), np.zeros_like(labels)
    else:
        has_atrophy = labels == 1
        has_shortcut = assign_shortcuts(labels, p_shortcut, seed=seed + 1)
    if flip_shortcut:
        has_shortcut = assign_shortcuts(1 - labels,
                                        p_shortcut if condition == "both" else 1.0,
                                        seed=seed + 2)

    # One shortcut location for the whole cohort, from a template-like brain
    # mask (the union across a few subjects is robust to any one bad strip).
    center = scfg.center
    if center is None and has_shortcut.any():
        brain = np.zeros(region_mask.shape, bool)
        for s in subjects[:10]:
            brain |= np.load(s.path) != 0
        center = default_shortcut_center(region_mask, brain)

    np.save(out / "gt" / "region_mask.npy", region_mask.astype(np.uint8))
    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for i, s in enumerate(subjects):
            vol = np.load(s.path).astype(np.float32)
            loss = (float(rng.uniform(*acfg.volume_loss_range))
                    if has_atrophy[i] else 0.0)
            sim = simulate_volume(vol, region_mask, rng,
                                  volume_loss=loss,
                                  shortcut=bool(has_shortcut[i]),
                                  atrophy_cfg=acfg, shortcut_cfg=scfg,
                                  shortcut_center=center)

            vpath = out / "volumes" / f"{s.subject_id}.npy"
            apath = out / "gt" / f"{s.subject_id}_atrophy.npy"
            spath = out / "gt" / f"{s.subject_id}_shortcut.npy"
            np.save(vpath, sim.volume)
            np.save(apath, sim.atrophy_gt)
            np.save(spath, sim.shortcut_gt)

            w.writerow({
                "path": str(vpath), "subject_id": s.subject_id,
                "label": int(labels[i]), "site": s.site,
                "age": "" if s.age is None else s.age,
                "sex": s.sex or "",
                **sim.meta,
                "atrophy_gt": str(apath), "shortcut_gt": str(spath),
            })
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(subjects)}")

    with open(out / "config.json", "w") as fh:
        json.dump({
            "condition": condition, "p_shortcut": p_shortcut,
            "flip_shortcut": flip_shortcut, "seed": seed,
            "shortcut_center": list(center) if center is not None else None,
            "atrophy": vars(acfg), "shortcut": vars(scfg),
            "n_subjects": len(subjects),
            "n_positive": int(labels.sum()),
            "n_shortcut": int(np.asarray(has_shortcut).sum()),
        }, fh, indent=2, default=list)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npy-dir", required=True,
                    help="preprocessed IXI volumes from data/preprocess.py")
    ap.add_argument("--demographics", default=None,
                    help="IXI.xls or a CSV export of it")
    ap.add_argument("--region-mask", default=None,
                    help=".npy hippocampus mask on the volume grid; computed "
                         "from Harvard-Oxford via nilearn if omitted")
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--p-shortcut", type=float, default=1.0)
    ap.add_argument("--flip-shortcut", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    subjects = ixi_subjects(args.npy_dir, args.demographics, pattern="*.npy")
    if not subjects:
        raise SystemExit(f"no IXI .npy volumes in {args.npy_dir}")
    shape = np.load(subjects[0].path, mmap_mode="r").shape

    if args.region_mask:
        region = np.load(args.region_mask).astype(bool)
    else:
        region = hippocampus_mask_on_grid(shape)
    if region.shape != shape:
        raise SystemExit(f"region mask {region.shape} != volume grid {shape}")

    print(f"{len(subjects)} subjects, condition={args.condition}")
    m = build_condition(subjects, region, args.condition, args.out,
                        p_shortcut=args.p_shortcut,
                        flip_shortcut=args.flip_shortcut, seed=args.seed)
    print(f"manifest: {m}")


if __name__ == "__main__":
    main()
