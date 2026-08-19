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

from models.resnet3d import resnet3d_10          # noqa: E402
from xai.lrp3d import LRP3D                       # noqa: E402
from xai.gradcam3d import GradCAM3D               # noqa: E402
from eval.faithfulness import (                   # noqa: E402
    insertion_deletion_auc, random_baseline_auc, blur_baseline,
)
from eval.sanity import cascading_randomization, rank_correlation  # noqa: E402

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


def main() -> int:
    tests = [
        test_forward,
        test_lrp_runs,
        test_lrp_class_sensitivity,
        test_gradcam_runs,
        test_blur_baseline,
        test_faithfulness,
        test_faithfulness_ordering,
        test_sanity_check,
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
