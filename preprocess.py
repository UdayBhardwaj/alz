"""
T1-weighted MRI preprocessing for ADNI / OASIS-3 / AIBL.

Pipeline:
    1. N4 bias field correction      — removes scanner intensity inhomogeneity
    2. Skull stripping               — removes non-brain tissue
    3. Affine registration to MNI152 — puts every subject on a common grid
    4. Resample to the analysis grid
    5. Intensity normalisation       — z-score within the brain mask

Steps 3 and 5 are not optional for this study specifically. Without common-space
registration, atlas-based relevance analysis is meaningless because voxel
(40, 55, 30) is a different structure in every subject. Without brain-masked
normalisation, background voxels distort the z-scores and the network can pick
up on scanner-specific intensity signatures — which then show up faithfully in
the explanations as relevance outside the brain.

Requires: pip install nibabel SimpleITK antspyx nilearn
For skull stripping, HD-BET or FreeSurfer SynthStrip give better results than
the ANTs default and are worth the extra install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = ["PreprocessConfig", "preprocess_volume", "normalise_intensity"]


@dataclass
class PreprocessConfig:
    target_shape: Tuple[int, int, int] = (96, 96, 96)
    mni_template: Optional[str] = None      # None -> nilearn's MNI152
    do_n4: bool = True
    do_skullstrip: bool = True
    do_register: bool = True
    skullstrip_method: str = "hdbet"        # "hdbet" | "synthstrip" | "ants"
    clip_percentiles: Tuple[float, float] = (0.5, 99.5)


def normalise_intensity(vol: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        clip: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Z-score normalisation computed inside the brain mask only.

    Computing statistics over the whole volume lets the large background region
    dominate, and the resulting scale varies with head size and field of view.
    """
    if mask is None:
        mask = vol > 0
    brain = vol[mask]
    if brain.size == 0:
        raise ValueError("empty brain mask — skull stripping failed")

    lo, hi = np.percentile(brain, clip)
    vol = np.clip(vol, lo, hi)
    brain = vol[mask]

    mu, sd = brain.mean(), brain.std()
    if sd < 1e-8:
        raise ValueError("zero-variance volume")

    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = ((vol[mask] - mu) / sd).astype(np.float32)
    return out


def preprocess_volume(in_path: str,
                      out_path: str,
                      cfg: Optional[PreprocessConfig] = None) -> np.ndarray:
    """Run the full pipeline on one scan and save as .npy.

    Returns the preprocessed array. Save as .npy rather than NIfTI: training
    reads these thousands of times and .npy memory-maps without header parsing.
    """
    cfg = cfg or PreprocessConfig()

    try:
        import ants
    except ImportError as exc:                        # pragma: no cover
        raise ImportError(
            "antspyx required: pip install antspyx nibabel nilearn"
        ) from exc

    img = ants.image_read(in_path)

    if cfg.do_n4:
        img = ants.n4_bias_field_correction(img)

    mask = None
    if cfg.do_skullstrip:
        img, mask = _skullstrip(img, cfg.skullstrip_method)

    if cfg.do_register:
        template = _load_template(cfg.mni_template)
        reg = ants.registration(fixed=template, moving=img,
                                type_of_transform="Affine")
        img = reg["warpedmovout"]
        if mask is not None:
            mask = ants.apply_transforms(
                fixed=template, moving=mask,
                transformlist=reg["fwdtransforms"],
                interpolator="nearestNeighbor",
            )

    arr = img.numpy()
    mask_arr = mask.numpy().astype(bool) if mask is not None else None

    arr, mask_arr = _resample(arr, mask_arr, cfg.target_shape)
    arr = normalise_intensity(arr, mask_arr, cfg.clip_percentiles)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr.astype(np.float32))
    return arr


def _skullstrip(img, method: str):
    import ants
    if method == "ants":
        mask = ants.get_mask(img)
        return ants.mask_image(img, mask), mask
    raise NotImplementedError(
        f"skullstrip method '{method}' — install HD-BET "
        "(github.com/MIC-DKFZ/HD-BET) or SynthStrip and call it here. "
        "The ANTs fallback ('ants') is usable but noticeably worse at the "
        "inferior temporal boundary, which is exactly where AD atrophy lives."
    )


def _load_template(path: Optional[str]):
    import ants
    if path:
        return ants.image_read(path)
    from nilearn import datasets
    return ants.image_read(datasets.load_mni152_template().get_filename())


def _resample(arr: np.ndarray, mask: Optional[np.ndarray],
              shape: Tuple[int, int, int]):
    from scipy.ndimage import zoom
    factors = [s / a for s, a in zip(shape, arr.shape)]
    arr_r = zoom(arr, factors, order=1)
    mask_r = zoom(mask.astype(np.float32), factors, order=0) > 0.5 if mask is not None else None
    return arr_r, mask_r


def preprocess_cohort(manifest_csv: str, out_dir: str,
                      cfg: Optional[PreprocessConfig] = None) -> None:
    """Batch-process a cohort from a CSV with columns: path, subject_id, label.

    Roughly 30-90 s per scan single-threaded. For ADNI-scale cohorts run this
    once, overnight, and cache the output.
    """
    import csv
    cfg = cfg or PreprocessConfig()
    with open(manifest_csv) as fh:
        rows = list(csv.DictReader(fh))

    for i, row in enumerate(rows, 1):
        stem = Path(row["path"]).stem.replace(".nii", "")
        out = os.path.join(out_dir, f"{row['subject_id']}_{stem}.npy")
        if os.path.exists(out):
            continue
        try:
            preprocess_volume(row["path"], out, cfg)
            print(f"[{i}/{len(rows)}] {stem}")
        except Exception as exc:
            print(f"[{i}/{len(rows)}] FAILED {stem}: {exc}")
