"""
Live demonstration on real benchmark scans with the trained models.

    python demo_live.py --bench work/bench --runs work/runs_full

1. Loads the fold-1 atrophy model and classifies two held-out test scans
   (IXI002 with simulated atrophy, IXI013 without).
2. Explains IXI002 with five attribution methods and scores every map against
   the exact ground truth (correctness) and the atlas hippocampus (plausibility).
3. Reliance test: the shortcut-only model sees IXI002 with the planted shortcut
   in place and then moved; its prediction should follow the shortcut.
4. Shows the maps in a window (close it to continue).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from eval.anatomical import ground_truth_scores
from models.resnet3d import resnet3d_10
from xai.baselines import GradientInput3D, IntegratedGradients3D, Occlusion3D
from xai.gradcam3d import GradCAM3D, GradCAMPlusPlus3D
from xai.lrp3d import LRP3D

BENCH = "work/bench"                   # set from --bench / --runs in main()
RUNS = "work/runs_full"
FOLD = 1
POS, NEG = "IXI002", "IXI013"          # first positive / negative in the fold-1 test split
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(condition: str) -> torch.nn.Module:
    m = resnet3d_10(n_classes=2).to(DEV)
    m.load_state_dict(torch.load(f"{RUNS}/{condition}/fold{FOLD}.pt", map_location=DEV, weights_only=True))
    return m.eval()


def volume(condition: str, subject: str) -> torch.Tensor:
    v = np.load(f"{BENCH}/{condition}/volumes/{subject}.npy").astype(np.float32)
    return torch.from_numpy(v)[None, None].to(DEV)


def p_ad(model: torch.nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.softmax(model(x), 1)[0, 1])


def banner(text: str) -> None:
    print("\n" + "=" * 72 + f"\n  {text}\n" + "=" * 72)


def main() -> None:
    global BENCH, RUNS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", default=BENCH, help="benchmark folder written by run_pilot.py")
    ap.add_argument("--runs", default=RUNS, help="folder with the trained fold checkpoints")
    args = ap.parse_args()
    BENCH, RUNS = args.bench, args.runs

    banner(f"1. Classification with the trained 3D ResNet-10 (fold {FOLD}, device: {DEV.type.upper()})")
    model = load_model("atrophy")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    for subj, truth in ((POS, "simulated atrophy"), (NEG, "no atrophy")):
        p = p_ad(model, volume("atrophy", subj))
        print(f"  {subj} ({truth:<17}) -> P(atrophy) = {p:.4f}  ->  predicted: {'ATROPHY' if p > 0.5 else 'normal'}")

    banner(f"2. Explaining {POS} with five methods and scoring against ground truth")
    x = volume("atrophy", POS)
    brain = x[0, 0].cpu().numpy() != 0
    gt = np.load(f"{BENCH}/atrophy/gt/{POS}_atrophy.npy")
    hippo = np.load(f"{BENCH}/atrophy/gt/region_mask.npy").astype(np.float32)
    methods = {"Occlusion": Occlusion3D(model), "LRP": LRP3D(model, input_bounds=(-4.0, 4.0)),
               "Integrated Gradients": IntegratedGradients3D(model, n_steps=32),
               "Gradient x Input": GradientInput3D(model), "Grad-CAM": GradCAM3D(model)}
    maps = {}
    print(f"  {'Method':<22}{'time':>8}{'correctness (x chance)':>26}{'plausibility':>15}")
    print("  " + "-" * 70)
    for name, fn in methods.items():
        t0 = time.time()
        sal = fn(x, 1)[0]
        dt = time.time() - t0
        corr = ground_truth_scores(sal, gt, brain)
        plaus = ground_truth_scores(sal, hippo, brain)
        maps[name] = sal
        print(f"  {name:<22}{dt:>7.1f}s{corr['auprc'] / corr['auprc_chance']:>20.2f}x{plaus['enrichment']:>17.2f}")
        if hasattr(fn, "remove"):
            fn.remove()
    rnd = np.random.default_rng(0).random(gt.shape).astype(np.float32)
    rc = ground_truth_scores(rnd, gt, brain)
    print(f"  {'Random map':<22}{'':>8}{rc['auprc'] / rc['auprc_chance']:>20.2f}x{ground_truth_scores(rnd, hippo, brain)['enrichment']:>17.2f}")
    print("  (1.00x = no better than a random map)")

    banner("3. Reliance test: the shortcut-only model with the shortcut in place vs moved")
    sc_model = load_model("shortcut")
    import csv
    with open(f"{RUNS}/shortcut/fold{FOLD}_test_manifest.csv") as fh:
        subjects = [r["subject_id"] for r in csv.DictReader(fh) if r["label"] == "1"]
    print(f"  Moving the shortcut to the other class for the first positive test scans of fold {FOLD}:")
    print(f"  {'Subject':<10}{'shortcut in place':>20}{'shortcut moved':>18}   decision")
    flipped = dropped = 0
    for subj in subjects[:6]:
        p_in, p_out = p_ad(sc_model, volume("shortcut", subj)), p_ad(sc_model, volume("shortcut_flip", subj))
        changed = (p_in > 0.5) != (p_out > 0.5)
        flipped += changed; dropped += p_out < p_in
        print(f"  {subj:<10}{p_in:>20.4f}{p_out:>18.4f}   {'CHANGED' if changed else 'unchanged'}")
    print(f"  Moving only the faint blob lowers P(positive) in {dropped} of 6 scans and changes {flipped} decisions:")
    print("  this model relies on the shortcut, so a faithful explanation must point at the blob.")

    banner("4. Showing the maps (close the window to continue)")
    import matplotlib.pyplot as plt
    cy = int(round(np.argwhere(hippo > 0).mean(0)[1]))          # coronal slice through the hippocampus
    vol = x[0, 0].detach().cpu().numpy()
    lim = dict(vmin=np.percentile(vol[brain], 1), vmax=np.percentile(vol[brain], 99))
    fig, ax = plt.subplots(1, 2 + len(maps), figsize=(17, 3.6))
    fig.canvas.manager.set_window_title(f"Live explanation of {POS} (coronal slice {cy})")
    ax[0].imshow(vol[:, cy, :].T, origin="lower", cmap="gray", **lim); ax[0].set_title("Input scan")
    ax[1].imshow(vol[:, cy, :].T, origin="lower", cmap="gray", **lim)
    ax[1].imshow(np.ma.masked_equal(gt[:, cy, :], 0).T, origin="lower", cmap="autumn", alpha=0.8)
    ax[1].set_title("Ground truth")
    for a, (name, sal) in zip(ax[2:], maps.items()):
        s = np.maximum(sal, 0) * brain; s = s / (np.percentile(s[s > 0], 99.5) if (s > 0).any() else 1)
        a.imshow(vol[:, cy, :].T, origin="lower", cmap="gray", **lim)
        a.imshow(np.ma.masked_less(np.clip(s[:, cy, :], 0, 1), 0.08).T, origin="lower", cmap="inferno", vmin=0, vmax=1, alpha=0.85)
        a.set_title(name)
    for a in ax:
        a.contour((hippo[:, cy, :] > 0).T.astype(float), levels=[0.5], colors="cyan", linewidths=0.8, origin="lower")
        a.axis("off")
    fig.suptitle(f"{POS}: P(atrophy) = {p_ad(model, x):.3f}  —  cyan: atlas hippocampus", fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
