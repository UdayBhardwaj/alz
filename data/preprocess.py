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
    skullstrip_method: str = "hdbet"        # "hdbet" | "synthstrip" | "ants" | "template"
    mask_dilation: int = 2                  # voxels; "template" method only
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

    if cfg.do_skullstrip and cfg.skullstrip_method == "template":
        arr, mask_arr = _register_template_masked(img, cfg)
        arr, mask_arr = _resample(arr, mask_arr, cfg.target_shape)
        arr = normalise_intensity(arr, mask_arr, cfg.clip_percentiles)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr.astype(np.float32))
        return arr

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


def _head_template_2mm():
    """ICBM152 2009 head T1 and brain mask, resampled onto the 2 mm MNI152
    grid that the atlas and every benchmark volume use. Cached on disk."""
    import tempfile
    from nilearn import datasets, image

    cache = Path(tempfile.gettempdir())
    t_fn, m_fn = cache / "icbm2009_head_2mm.nii.gz", cache / "icbm2009_mask_2mm.nii.gz"
    if not (t_fn.exists() and m_fn.exists()):
        grid = datasets.load_mni152_template(resolution=2)
        icbm = datasets.fetch_icbm152_2009()
        image.resample_to_img(icbm["t1"], grid, interpolation="continuous").to_filename(t_fn)
        image.resample_to_img(icbm["mask"], grid, interpolation="nearest").to_filename(m_fn)
    return str(t_fn), str(m_fn)


def _register_template_masked(img, cfg: PreprocessConfig):
    """Registration-based brain extraction: head-to-head, then template mask.

    Raw IXI scans include skull and neck. Registering them to a brain-only
    template lets the affine squeeze the whole head into the brain outline,
    and the brain mask then keeps skull (seen in the first atlas QC). So the
    head is registered to the ICBM152 2009 *head* template, and the
    template's brain mask, slightly dilated, cuts the brain out of the
    registered head. Everything downstream lives in MNI space, so a
    template-space mask is all we need. Requires only ANTs and nilearn.
    """
    import ants
    from scipy.ndimage import binary_dilation

    t_fn, m_fn = _head_template_2mm()
    template = ants.image_read(t_fn)
    mask_np = ants.image_read(m_fn).numpy() > 0.5
    mask_np = binary_dilation(mask_np, iterations=cfg.mask_dilation)

    reg = ants.registration(fixed=template, moving=img, type_of_transform="Affine")
    arr = reg["warpedmovout"].numpy()
    return arr * mask_np, mask_np


def _load_template(path: Optional[str]):
    import ants
    if path:
        return ants.image_read(path)
    import tempfile
    from nilearn import datasets
    # Pinned to 2 mm: hippocampus_mask_on_grid() and the MNI brain mask must
    # use the same grid. Newer nilearn returns an in-memory image with no
    # filename, so write it out for ANTs.
    tmpl = datasets.load_mni152_template(resolution=2)
    fn = tmpl.get_filename()
    if not fn:
        fn = os.path.join(tempfile.gettempdir(), "mni152_2mm_template.nii.gz")
        if not os.path.exists(fn):
            tmpl.to_filename(fn)
    return ants.image_read(fn)


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
