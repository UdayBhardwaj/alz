"""
Visual QC: does the atlas hippocampus land on the hippocampus?

Preprocesses one scan (if needed), builds the hippocampus mask on the same
grid, and writes three orthogonal slices through the mask centroid with the
mask outlined. Every ground-truth number in the benchmark depends on this
overlay being right, and nothing else in the pipeline would notice if it
were not.

    python qc_atlas.py --nifti path/to/IXI002-Guys-0828-T1.nii.gz --out qc
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from data.build_benchmark import hippocampus_mask_on_grid
from data.preprocess import PreprocessConfig, preprocess_volume


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti", required=True)
    ap.add_argument("--out", default="qc")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npy = out / (Path(args.nifti).name.replace(".nii.gz", "") + ".npy")
    cfg = PreprocessConfig(skullstrip_method="template")
    t0 = time.time()
    vol = preprocess_volume(args.nifti, str(npy), cfg)
    print(f"preprocessed in {time.time() - t0:.1f}s -> {vol.shape}")

    mask = hippocampus_mask_on_grid(vol.shape)
    np.save(out / "hippocampus_96.npy", mask.astype(np.uint8))
    c = [int(round(v)) for v in np.argwhere(mask).mean(0)]
    inside = vol[mask]
    print(f"mask voxels {int(mask.sum())}, centroid {c}; "
          f"mean intensity inside mask {inside.mean():+.2f} (brain z-scored), "
          f"fraction of mask on zero background {(inside == 0).mean():.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = [("sagittal (x)", vol[c[0]], mask[c[0]]),
             ("coronal (y)", vol[:, c[1]], mask[:, c[1]]),
             ("axial (z)", vol[:, :, c[2]], mask[:, :, c[2]])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for j, (name, img, m) in enumerate(views):
        for i in range(2):
            ax = axes[i, j]
            ax.imshow(img.T, cmap="gray", origin="lower")
            if i == 1:
                ax.contour(m.T.astype(float), levels=[0.5], colors="red", linewidths=1.2)
            ax.set_title(f"{name}" + (" + atlas hippocampus" if i else ""), fontsize=11)
            ax.axis("off")
    fig.suptitle(f"Atlas QC: {Path(args.nifti).name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "qc_hippocampus.png", dpi=110)
    print(f"written {out / 'qc_hippocampus.png'}")


if __name__ == "__main__":
    main()
