"""
Smoke tests on synthetic volumes — no ADNI access required.

Run these before touching real data. They verify that the pipeline is wired
correctly and, more usefully, that the metrics behave the way the paper will
claim they do.

    python -m tests.test_synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.resnet3d import resnet3d_10, fold_batchnorm  # noqa: E402
from xai.lrp3d import LRP3D                       # noqa: E402
from xai.gradcam3d import GradCAM3D               # noqa: E402
from eval.faithfulness import (                   # noqa: E402
    insertion_deletion_auc, random_baseline_auc, blur_baseline,
)
from eval.sanity import cascading_randomization, rank_correlation  # noqa: E402
from eval.anatomical import ground_truth_scores  # noqa: E402
from data.ixi_dataset import parse_ixi_filename  # noqa: E402
from data.simulate import (                       # noqa: E402
    AtrophyConfig, assign_labels, assign_shortcuts, simulate_volume,
)

SHAPE = (48, 48, 48)
PASS, FAIL = "  PASS", "  FAIL"


def _model():
    torch.manual_seed(0)
    m = resnet3d_10(n_classes=2, widen=0.25).eval()
    return m


def _volume(seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, SHAPE).astype(np.float32)
    # A blob standing in for a structure of interest.
    v[18:30, 18:30, 18:30] += 2.0
    return torch.from_numpy(v).unsqueeze(0).unsqueeze(0)


def test_forward():
    m, x = _model(), _volume()
    with torch.no_grad():
        out = m(x)
    ok = out.shape == (1, 2)
    print(f"{PASS if ok else FAIL}  forward -> {tuple(out.shape)}")
    return ok


def test_lrp_runs():
    m, x = _model(), _volume()
    lrp = LRP3D(m, input_bounds=(-4.0, 4.0))
    R, cls = lrp(x)
    ok = R.shape == SHAPE and np.isfinite(R).all() and R.max() > 0
    print(f"{PASS if ok else FAIL}  LRP -> shape {R.shape}, class {cls}, "
          f"max {R.max():.3f}, nonzero {(R > 0).mean():.1%}")
    return ok


def test_lrp_class_sensitivity():
    """Relevance for class 0 and class 1 must differ.

    If they match, the propagation has collapsed to something
    class-independent and the whole method is broken.
    """
    m, x = _model(), _volume()
    lrp = LRP3D(m, input_bounds=(-4.0, 4.0))
    r0, _ = lrp(x, class_idx=0)
    r1, _ = lrp(x, class_idx=1)
    rho = rank_correlation(r0, r1)
    ok = not np.isclose(rho, 1.0, atol=1e-4)
    print(f"{PASS if ok else FAIL}  LRP class sensitivity -> rho(c0,c1) = {rho:.4f}")
    return ok


def test_gradcam_runs():
    m, x = _model(), _volume()
    cam_fn = GradCAM3D(m)
    cam, cls = cam_fn(x)
    cam_fn.remove()
    ok = cam.shape == SHAPE and np.isfinite(cam).all()
    print(f"{PASS if ok else FAIL}  Grad-CAM -> shape {cam.shape}, class {cls}")
    return ok


def test_ig_after_gradcam():
    """Integrated Gradients must work on an input another explainer used.

    run_evaluation calls every method on the same tensor; Grad-CAM leaving
    requires_grad set on it once made IG crash on a missing .grad.
    """
    from xai.baselines import IntegratedGradients3D
    m, x = _model(), _volume()
    cam = GradCAM3D(m)
    cam(x)
    cam.remove()
    ig, _ = IntegratedGradients3D(m, n_steps=4, batch_size=2)(x, 0)
    ok = ig.shape == SHAPE and np.isfinite(ig).all() and np.abs(ig).max() > 0
    print(f"{PASS if ok else FAIL}  IG after Grad-CAM -> shape {ig.shape}, "
          f"max |IG| {np.abs(ig).max():.3f}")
    return ok


def test_blur_baseline():
    x = _volume()
    b = blur_baseline(x, sigma=3.0)
    ok = b.shape == x.shape and b.std() < x.std()
    print(f"{PASS if ok else FAIL}  blur baseline -> std {x.std():.3f} "
          f"-> {b.std():.3f}")
    return ok


def test_faithfulness():
    m, x = _model(), _volume()
    with torch.no_grad():
        cls = int(m(x).argmax(1).item())

    lrp = LRP3D(m, input_bounds=(-4.0, 4.0))
    R, _ = lrp(x, cls)

    res = insertion_deletion_auc(m, x, R, cls, patch=8, n_steps=20)
    rnd = random_baseline_auc(m, x, cls, patch=8, n_steps=20)

    ok = (0 <= res.insertion_auc <= 1 and 0 <= res.deletion_auc <= 1
          and res.n_patches == 216)
    print(f"{PASS if ok else FAIL}  faithfulness -> LRP {res}")
    print(f"         random baseline -> {rnd}")
    return ok


def test_faithfulness_ordering():
    """A perfect oracle map must beat a random one.

    Construct saliency equal to the true gradient magnitude — the most
    faithful map obtainable by construction — and confirm the metric ranks it
    above noise. If this fails, the metric implementation is wrong, and no
    result computed with it means anything.
    """
    m, x = _model(), _volume()
    x_g = x.clone().requires_grad_(True)
    logits = m(x_g)
    cls = int(logits.argmax(1).item())
    logits[0, cls].backward()
    oracle = x_g.grad.abs().squeeze().detach().numpy()

    good = insertion_deletion_auc(m, x, oracle, cls, patch=8, n_steps=20)
    rand = random_baseline_auc(m, x, cls, patch=8, n_steps=20, seed=1)

    ok = good.score > rand.score
    print(f"{PASS if ok else FAIL}  metric ordering -> oracle {good.score:+.4f} "
          f"vs random {rand.score:+.4f}")
    return ok


def test_sanity_check():
    m, x = _model(), _volume()
    res = cascading_randomization(
        m, x,
        explainer_factory=lambda mm: LRP3D(mm, input_bounds=(-4.0, 4.0)),
        seed=0,
    )
    ok = len(res) > 0 and all(np.isfinite(v) or np.isnan(v) for v in res.values())
    print(f"{PASS if ok else FAIL}  cascading randomization ->")
    for k, v in res.items():
        print(f"         {k:<8s} rho = {v:+.4f}")
    return ok


# ---------------------------------------------------------------------------
# BatchNorm folding and conservation
# ---------------------------------------------------------------------------

def _model_with_bn_stats(bias_free: bool = False):
    """Synthetic model whose BN layers are far from identity.

    Default-initialised BN (mean 0, var 1, gamma 1, beta 0) is a near no-op,
    so a fold test on it passes even when the fold is wrong. Random running
    statistics make every term of the fold formula matter. ``bias_free``
    zeroes the mean shift and every bias, so the network is positively
    homogeneous and LRP conservation must hold exactly up to the epsilons.
    """
    m = _model()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm3d):
                n = mod.num_features
                mod.weight.copy_(0.5 + torch.rand(n, generator=g))
                mod.running_var.copy_(0.2 + torch.rand(n, generator=g))
                if bias_free:
                    mod.running_mean.zero_()
                    mod.bias.zero_()
                else:
                    mod.running_mean.copy_(torch.randn(n, generator=g) * 0.3)
                    mod.bias.copy_(torch.randn(n, generator=g) * 0.3)
        if bias_free:
            m.fc.bias.zero_()
    return m.eval()


def test_fold_batchnorm_equivalence():
    """Folded and original forward passes must agree to float tolerance.

    The safety net for the fold: a wrong fold yields plausible-looking LRP
    maps that are silently explaining a different function.
    """
    m, x = _model_with_bn_stats(), _volume()
    folded = fold_batchnorm(m)
    with torch.no_grad():
        a, b = m(x), folded(x)
    n_bn = sum(isinstance(mod, torch.nn.BatchNorm3d) for mod in folded.modules())
    err = (a - b).abs().max().item()
    ok = n_bn == 0 and torch.allclose(a, b, rtol=1e-4, atol=1e-5)
    print(f"{PASS if ok else FAIL}  BN fold -> max |diff| {err:.2e}, "
          f"BN left {n_bn}")
    return ok


def test_lrp_layers_compose():
    """Composing lrp_layers() must reproduce forward(), BN folded or not."""
    m, x = _model_with_bn_stats(), _volume()
    with torch.no_grad():
        ref = m(x)
        errs = []
        for fold in (True, False):
            a = x
            for layer in m.lrp_layers(fold_bn=fold):
                a = layer(a)
            errs.append((a - ref).abs().max().item())
    ok = all(e < 1e-4 for e in errs)
    print(f"{PASS if ok else FAIL}  lrp_layers composition -> max |diff| "
          f"folded {errs[0]:.2e}, unfolded {errs[1]:.2e}")
    return ok


def test_lrp_conservation():
    """Sum of input relevance must equal the initialising logit on a
    bias-free network, for every class.

    The threshold is loose relative to float error and tight relative to
    any real leak: a rule that drops or double-counts relevance anywhere in
    the chain is off by tens of percent, not fractions of one.
    """
    m, x = _model_with_bn_stats(bias_free=True), _volume()
    lrp = LRP3D(m, input_bounds=(-8.0, 8.0))
    errs = [lrp.conservation_error(x, class_idx=c) for c in (0, 1)]
    ok = all(np.isfinite(e) and e < 1e-2 for e in errs)
    print(f"{PASS if ok else FAIL}  LRP conservation (bias-free) -> rel err "
          f"c0 {errs[0]:.2e}, c1 {errs[1]:.2e}")
    return ok


# ---------------------------------------------------------------------------
# Benchmark simulation
# ---------------------------------------------------------------------------

def _phantom():
    """Ellipsoidal 'brain' with two separate bright 'hippocampi'.

    Background is exactly zero, as after preprocessing. Two components so
    the per-component shrink is exercised; a shared centroid would pull them
    together instead.
    """
    D = 48
    z, y, x = np.mgrid[:D, :D, :D].astype(np.float64)
    c = (D - 1) / 2
    brain = ((z - c) / 20) ** 2 + ((y - c) / 18) ** 2 + ((x - c) / 22) ** 2 <= 1
    region = np.zeros((D, D, D), bool)
    for xc in (c - 9, c + 9):
        region |= ((z - c) / 5) ** 2 + ((y - c) / 4) ** 2 + ((x - xc) / 4) ** 2 <= 1
    rng = np.random.default_rng(0)
    vol = np.where(brain, 0.3 * rng.standard_normal((D, D, D)), 0.0)
    vol[region] += 2.0
    return vol.astype(np.float32), region, brain


def test_atrophy_simulation():
    """Atrophy must shrink the region by about the requested amount, never
    fold, and keep its ground truth local: >90% of GT mass within 2 falloff
    sigmas (Euclidean) of the region, and none at all beyond 3."""
    vol, region, _ = _phantom()
    cfg = AtrophyConfig()
    sim = simulate_volume(vol, region, np.random.default_rng(0),
                          volume_loss=0.25, atrophy_cfg=cfg)
    achieved = sim.meta["volume_loss_achieved"]
    gt = sim.atrophy_gt > 0
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~region)
    sigma = cfg.falloff_sigma
    on_target = sim.atrophy_gt[dist <= 2 * sigma].sum() / max(sim.atrophy_gt.sum(), 1e-12)
    far = int((gt & (dist > 3 * sigma)).sum())
    ok = (0.15 <= achieved <= 0.35 and sim.meta["min_jacobian"] > 0
          and gt.any() and on_target > 0.9 and far == 0
          and not sim.shortcut_gt.any())
    print(f"{PASS if ok else FAIL}  atrophy -> target 25%, achieved "
          f"{achieved:.1%}, min detJ {sim.meta['min_jacobian']:.3f}, "
          f"GT mass within 2 sigma {on_target:.1%}, beyond 3 sigma {far} vox")
    return ok


def test_controls_resampled():
    """Controls must be resampled too, and carry no atrophy ground truth.

    If a control passed through untouched while positives were
    interpolated, the classifier could detect interpolation alone. With
    jitter switched off, a zero-loss warp must be the identity.
    """
    vol, region, brain = _phantom()
    ctrl = simulate_volume(vol, region, np.random.default_rng(1), volume_loss=0.0)
    ident = simulate_volume(vol, region, np.random.default_rng(1), volume_loss=0.0,
                            atrophy_cfg=AtrophyConfig(jitter_amplitude=0.0))
    changed = np.abs(ctrl.volume - vol)[brain].mean()
    ok = (changed > 1e-3 and not ctrl.atrophy_gt.any()
          and np.allclose(ident.volume, vol, atol=1e-5))
    print(f"{PASS if ok else FAIL}  controls resampled -> mean |change| "
          f"{changed:.4f}, identity warp exact {np.allclose(ident.volume, vol, atol=1e-5)}")
    return ok


def test_shortcut_injection():
    """The bump must sit inside the brain, away from the region, and no
    voxel outside its ground truth may change.

    Same seed for both calls, so the warps are identical and the only
    difference is the bump. Any change outside the GT support would be
    signal the GT does not account for.
    """
    vol, region, brain = _phantom()
    rng_a, rng_b = np.random.default_rng(2), np.random.default_rng(2)
    plain = simulate_volume(vol, region, rng_a, shortcut=False)
    marked = simulate_volume(vol, region, rng_b, shortcut=True)
    diff = marked.volume != plain.volume
    gt = marked.shortcut_gt > 0
    outside = int((diff & ~gt).sum())
    ok = (gt.any() and diff.any() and not (gt & ~brain).any()
          and not (gt & region).any() and outside == 0)
    print(f"{PASS if ok else FAIL}  shortcut -> {gt.sum()} GT voxels, "
          f"{diff.sum()} changed, {outside} changed outside GT")
    return ok


def test_label_and_shortcut_assignment():
    """Labels balanced within site; shortcut counts exactly as requested."""
    rng = np.random.default_rng(0)
    sites = list(rng.choice(["Guys", "HH", "IOP"], size=300))
    ages = list(rng.uniform(20, 86, size=300))
    y = assign_labels(sites, ages, seed=0)
    site_balance = max(abs(int(sum(y[i] for i in range(300) if sites[i] == s))
                           - sites.count(s) / 2) for s in set(sites))
    age_gap = abs(np.mean([a for a, l in zip(ages, y) if l == 1])
                  - np.mean([a for a, l in zip(ages, y) if l == 0]))
    sc = assign_shortcuts(y, 0.9, seed=0)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    ok = (site_balance <= 1 and age_gap < 2.0
          and sc[y == 1].sum() == round(0.9 * n1)
          and sc[y == 0].sum() == round(0.1 * n0))
    print(f"{PASS if ok else FAIL}  assignment -> site imbalance {site_balance}, "
          f"age gap {age_gap:.2f}y, shortcut {sc[y == 1].sum()}/{n1} pos, "
          f"{sc[y == 0].sum()}/{n0} neg")
    return ok


def test_ground_truth_metric_ordering():
    """A map equal to the GT must beat a random map on every GT score."""
    vol, region, brain = _phantom()
    sim = simulate_volume(vol, region, np.random.default_rng(0), volume_loss=0.25)
    rnd = np.random.default_rng(3).random(vol.shape)
    good = ground_truth_scores(sim.atrophy_gt, sim.atrophy_gt, brain)
    bad = ground_truth_scores(rnd, sim.atrophy_gt, brain)
    ok = (good["auprc"] > 0.99 and good["precision_at_k"] > 0.99
          and bad["auprc"] < 2 * bad["auprc_chance"]
          and good["enrichment"] > bad["enrichment"])
    print(f"{PASS if ok else FAIL}  GT metric ordering -> AUPRC oracle "
          f"{good['auprc']:.3f} vs random {bad['auprc']:.3f} "
          f"(chance {bad['auprc_chance']:.3f})")
    return ok


def test_ixi_filename_parsing():
    cases = {
        "IXI002-Guys-0828-T1.nii.gz": ("IXI002", "Guys"),
        "IXI012-HH-1211-T1.nii.gz": ("IXI012", "HH"),
        "IXI002_IXI002-Guys-0828-T1.npy": ("IXI002", "Guys"),
    }
    got = {k: parse_ixi_filename(k) for k in cases}
    try:
        parse_ixi_filename("sub-01_T1w.nii.gz")
        rejects = False
    except ValueError:
        rejects = True
    ok = got == cases and rejects
    print(f"{PASS if ok else FAIL}  IXI filename parsing")
    return ok


def main() -> int:
    tests = [
        test_forward,
        test_lrp_runs,
        test_lrp_class_sensitivity,
        test_gradcam_runs,
        test_ig_after_gradcam,
        test_blur_baseline,
        test_faithfulness,
        test_faithfulness_ordering,
        test_sanity_check,
        test_fold_batchnorm_equivalence,
        test_lrp_layers_compose,
        test_lrp_conservation,
        test_atrophy_simulation,
        test_controls_resampled,
        test_shortcut_injection,
        test_label_and_shortcut_assignment,
        test_ground_truth_metric_ordering,
        test_ixi_filename_parsing,
    ]
    print(f"\nSynthetic pipeline tests  (volume {SHAPE})\n" + "-" * 60)
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as exc:
            print(f"{FAIL}  {t.__name__} raised {type(exc).__name__}: {exc}")
            results.append(False)
    print("-" * 60)
    print(f"{sum(results)}/{len(results)} passed\n")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
