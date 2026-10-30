"""
Benchmark demonstration on a synthetic brain phantom — no downloads needed.

Shows, in one figure, what the ground-truth benchmark does to a scan:

    control          resampled through the shared warp, no disease
    atrophy          both "hippocampi" shrunk by the target volume loss
    ground truth     |det J - 1| of the atrophy deformation (where it changed)
    shortcut         faint planted bump and its ground-truth support

and prints the checks that make the ground truth trustworthy.

    python demo.py            # writes demo_output.png
"""

from __future__ import annotations

import numpy as np

from data.simulate import simulate_volume
from eval.anatomical import ground_truth_scores


def phantom(D: int = 64, seed: int = 0):
    """Ellipsoidal brain, darker ventricle-like core, two bright 'hippocampi'."""
    z, y, x = np.mgrid[:D, :D, :D].astype(np.float64)
    c = (D - 1) / 2
    brain = ((z - c) / 26) ** 2 + ((y - c) / 24) ** 2 + ((x - c) / 29) ** 2 <= 1
    vent = ((z - c) / 8) ** 2 + ((y - c - 2) / 5) ** 2 + ((x - c) / 10) ** 2 <= 1
    region = np.zeros((D, D, D), bool)
    for xc in (c - 13, c + 13):
        region |= ((z - c) / 7) ** 2 + ((y - c + 4) / 4) ** 2 + ((x - xc) / 4) ** 2 <= 1
    rng = np.random.default_rng(seed)
    vol = np.where(brain, 0.25 * rng.standard_normal((D, D, D)), 0.0)
    vol[vent] -= 1.5
    vol[region] += 1.5
    return vol.astype(np.float32), region, brain


def main() -> None:
    vol, region, brain = phantom()
    ctrl = simulate_volume(vol, region, np.random.default_rng(1), volume_loss=0.0)
    atr = simulate_volume(vol, region, np.random.default_rng(1), volume_loss=0.25)
    sc = simulate_volume(vol, region, np.random.default_rng(1), shortcut=True)

    gt = atr.atrophy_gt
    oracle = ground_truth_scores(gt, gt, brain)
    rand = ground_truth_scores(np.random.default_rng(2).random(vol.shape), gt, brain)
    changed_outside = int(((sc.volume != ctrl.volume) & (sc.shortcut_gt == 0)).sum())

    print("Ground-truth benchmark demo (64^3 phantom)")
    print("-" * 52)
    print(f"target hippocampal volume loss      25.0%")
    print(f"achieved (partial-volume measured)  {atr.meta['volume_loss_achieved']:.1%}")
    print(f"min Jacobian determinant (>0 = no folding)  {atr.meta['min_jacobian']:.3f}")
    print(f"control carries atrophy ground truth?       {bool(ctrl.atrophy_gt.any())}")
    print(f"shortcut voxels changed outside its mask    {changed_outside}")
    print(f"correctness AUPRC: oracle map {oracle['auprc']:.3f}, random map "
          f"{rand['auprc']:.3f} (chance {rand['auprc_chance']:.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zc = int(np.argwhere(region)[:, 0].mean())
    zs = int(np.argwhere(sc.shortcut_gt > 0)[:, 0].mean())
    panels = [
        (ctrl.volume[zc], "Control (resampled)", "gray", None),
        (atr.volume[zc], "Simulated atrophy (25%)", "gray", None),
        (np.abs(atr.volume[zc] - ctrl.volume[zc]), "|Atrophy − control|", "magma", None),
        (gt[zc], "Atrophy ground truth |det J − 1|", "magma", None),
        (sc.volume[zs], "Planted shortcut (faint)", "gray", None),
        (sc.shortcut_gt[zs], "Shortcut ground truth", "magma", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (img, title, cmap, _) in zip(axes.ravel(), panels):
        ax.imshow(img.T, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle("Faithful or Plausible? — ground-truth benchmark on a phantom", fontsize=14)
    fig.tight_layout()
    fig.savefig("demo_output.png", dpi=120)
    print("figure written to demo_output.png")


if __name__ == "__main__":
    main()
