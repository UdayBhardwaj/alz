"""Qualitative figures for the paper, drawn from files already on disk.

fig_benchmark.png  the benchmark on one real IXI scan (control, atrophy, truth
                   map, faint and strong shortcut)
fig_maps.png       the six methods' maps for the same test subject under the
                   atrophy, shortcut and strong-shortcut models

The maps are the ones run_evaluation.py saved for the first test volume of
fold 1 (IXI002, a positive in every condition); nothing is recomputed.
    python make_figures.py <out_dir>
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
BENCH = "C:/Users/bhard/ixi/pilot/bench"
RES = "C:/Users/bhard/ixi/pilot/results_full"
RUNS = "C:/Users/bhard/ixi/pilot/runs_full"
SUBJ, FOLD = "IXI002", 1
METHODS = ["Occlusion", "LRP", "IG", "GradInput", "GradCAM", "GradCAM++"]
NAMES = {"IG": "Integrated\nGradients", "GradInput": "Gradient\n× Input", "GradCAM": "Grad-CAM",
         "GradCAM++": "Grad-CAM++"}
ROWS = [("atrophy", "A: atrophy"), ("shortcut", "C: shortcut"), ("both_p90_amp10", "B: amplitude 1.0")]
plt.rcParams.update({"font.family": "serif", "font.size": 8})

vol = lambda c: np.load(f"{BENCH}/{c}/volumes/{SUBJ}.npy").squeeze()
gt = lambda c, k: np.load(f"{BENCH}/{c}/gt/{SUBJ}_{k}.npy").squeeze()
HIPPO = np.load(f"{BENCH}/atrophy/gt/region_mask.npy").astype(bool)
BRAIN = vol("atrophy") != 0
hy, hz = (int(round(v)) for v in np.argwhere(HIPPO).mean(0)[1:])
sx = int(round(np.argwhere(gt("shortcut", "shortcut") > 0).mean(0)[0]))


def first_test_subject(c):
    with open(f"{RUNS}/{c}/fold{FOLD}_test_manifest.csv") as fh:
        return next(csv.DictReader(fh))["subject_id"]


def show(ax, img, title=None, **kw):
    ax.imshow(img.T, origin="lower", cmap=kw.pop("cmap", "gray"), interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)


def contour(ax, mask, color, ls="-"):
    ax.contour(mask.T.astype(float), levels=[0.5], colors=color, linewidths=0.8, linestyles=ls, origin="lower")


def benchmark_figure():
    ctrl, atr = vol("null"), vol("atrophy")
    weak, strong = vol("shortcut"), vol("both_p90_amp10")
    g = gt("atrophy", "atrophy")
    lim = dict(vmin=np.percentile(atr[BRAIN], 1), vmax=np.percentile(atr[BRAIN], 99))
    cor = lambda v: v[:, hy, :]
    sag = lambda v: v[sx, :, :]
    fig, ax = plt.subplots(1, 6, figsize=(7.16, 1.55))
    show(ax[0], cor(ctrl), "(a) control", **lim)
    show(ax[1], cor(atr), "(b) atrophy", **lim)
    show(ax[2], np.abs(cor(atr) - cor(ctrl)), "(c) |b $-$ a|", cmap="magma")
    show(ax[3], cor(atr), "(d) truth map", **lim)
    ax[3].imshow(np.ma.masked_equal(cor(g), 0).T, origin="lower", cmap="autumn", alpha=0.8, interpolation="nearest")
    contour(ax[3], cor(HIPPO), "cyan")
    show(ax[4], sag(weak), "(e) shortcut, 0.3", **lim)
    show(ax[5], sag(strong), "(f) shortcut, 1.0", **lim)
    for a in ax[4:]:
        contour(a, sag(gt("shortcut", "shortcut")) > 0, "lime", "--")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.04)
    fig.savefig(os.path.join(OUT, "fig_benchmark.png"), dpi=300)
    plt.close(fig)


def maps_figure():
    back = vol("atrophy")
    proj = back.mean(axis=2)                      # axial mean projection as backdrop
    hip = HIPPO.any(axis=2)
    sc = gt("shortcut", "shortcut").max(axis=2) > 0
    fig, ax = plt.subplots(len(ROWS), len(METHODS) + 1, figsize=(7.16, 3.3))
    for r, (c, label) in enumerate(ROWS):
        assert first_test_subject(c) == SUBJ, c
        maps = np.load(f"{RES}/{c}/fold{FOLD}/example_maps.npz")
        truth = gt(c, "atrophy" if c == "atrophy" else "shortcut").max(axis=2)
        show(ax[r, 0], proj, "Ground truth" if r == 0 else None)
        ax[r, 0].imshow(np.ma.masked_equal(truth, 0).T, origin="lower", cmap="autumn", alpha=0.8,
                        interpolation="nearest")
        ax[r, 0].set_ylabel(label, fontsize=8)
        contour(ax[r, 0], hip, "cyan")
        contour(ax[r, 0], sc, "lime", "--")
        for j, m in enumerate(METHODS):
            a = ax[r, j + 1]
            s = np.maximum(maps[m], 0) * BRAIN
            p = s.max(axis=2)
            p = p / (np.percentile(p[p > 0], 99.5) if (p > 0).any() else 1)
            show(a, proj, NAMES.get(m, m) if r == 0 else None)
            a.imshow(np.ma.masked_less(np.clip(p, 0, 1), 0.08).T, origin="lower", cmap="inferno",
                     vmin=0, vmax=1, alpha=0.9, interpolation="nearest")
            contour(a, hip, "cyan")
            contour(a, sc, "lime", "--")
    for r in range(len(ROWS)):
        ax[r, 0].set_yticks([])
        ax[r, 0].yaxis.set_visible(True)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.9, bottom=0.01, wspace=0.03, hspace=0.04)
    fig.savefig(os.path.join(OUT, "fig_maps.png"), dpi=300)
    plt.close(fig)


def axes_figure():
    """Plausibility and correctness per method and condition (the 581-subject numbers)."""
    import json
    an = "C:/Users/bhard/ixi/pilot/analysis_full"
    kn = json.load(open(f"{an}/key_numbers.json", encoding="utf-8"))
    loc = json.load(open(f"{an}/extra_numbers.json", encoding="utf-8"))["location_lift_atrophy_model"]
    methods = METHODS + ["Random"]
    short = {"Occlusion": "Occ.", "LRP": "LRP", "IG": "IG", "GradInput": "G×I", "GradCAM": "GC",
             "GradCAM++": "GC++", "Random": "Rand."}
    axes = [("plaus_enrichment", "Plausibility (hippocampus)", "#4B6CB7"),
            ("atr_lift", "Correctness: atrophy", "#2E7D5B"),
            ("sc_lift", "Correctness: shortcut", "#E8743B")]
    panels = [("null", "Resampling control"), ("atrophy", "A: atrophy only"),
              ("shortcut", "C: shortcut only"), ("both_p90", "B: atrophy + shortcut")]
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.9))
    x = np.arange(len(methods))
    for a, (c, title) in zip(ax, panels):
        present = [k for k in axes if not np.isnan(kn[c]["methods"]["LRP"][k[0]][0])]
        w = 0.8 / len(present)
        for i, (k, label, col) in enumerate(present):
            v = np.array([kn[c]["methods"][m][k] for m in methods])
            a.bar(x + (i - (len(present) - 1) / 2) * w, v[:, 0], w, color=col, label=label,
                  yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], error_kw=dict(lw=0.6, capsize=1))
            if k == "sc_lift" and c != "shortcut":
                a.scatter(x[:-1] + (i - (len(present) - 1) / 2) * w, [loc[m][0] for m in METHODS],
                          marker="_", s=40, color="black", zorder=3, linewidths=1.2,
                          label="Location baseline")
        a.axhline(1.0, color="grey", ls="--", lw=0.6)
        a.set_title(title, fontsize=8)
        a.set_xticks(x)
        a.set_xticklabels([short[m] for m in methods], rotation=90, fontsize=6.5)
        a.tick_params(axis="y", labelsize=6.5)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("× chance (random map = 1)", fontsize=7)
    handles = {}
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.36, wspace=0.28)
    fig.savefig(os.path.join(OUT, "fig_axes.png"), dpi=300)
    plt.close(fig)


os.makedirs(OUT, exist_ok=True)
benchmark_figure()
maps_figure()
axes_figure()
print("figures in", OUT)
