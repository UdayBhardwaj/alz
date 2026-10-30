# TASKS

Ordered. Work top-down. Re-run `python -m tests.test_synthetic` after each item.

---

## ~~1. BUG — conservation check is broken~~ — DONE

`LRP3D._propagate()` returns unnormalised relevance; `conservation_error()`
now reports `|sum(R) - logit| / |logit|`. `test_lrp_conservation` asserts
< 1e-2 on a bias-free model (measured ~1e-5).

**It found a real bug:** `Rule.zbeta` subtracted the bound terms
(`a·∇a − l·∇l − h·∇h`) although their gradients already carry the minus sign,
so input relevance summed to roughly −logit (relative error 1.97). Fixed to
`a·∇a + l·∇l + h·∇h`. On the synthetic model LRP's faithfulness score moved
from −0.014 (below random) to +0.035. Any LRP map from before this fix is
invalid. For the methods section: conservation holds to ~1e-5 up to bias
absorption.

---

## ~~2. METHOD — fold BatchNorm into preceding convolutions~~ — DONE

`fold_batchnorm()` in `models/resnet3d.py`; `lrp_layers(fold_bn=True)` uses
it. `test_fold_batchnorm_equivalence` runs with randomised BN statistics
(default BN is a near no-op and would pass a wrong fold): max |diff| ~1e-6.
`test_lrp_layers_compose` checks the layer list reproduces `forward()`.

---

## 2b. Run the benchmark pipeline on real IXI data

- `download_ixi()`, then `preprocess_cohort()` (CPU, slowest step; run once).
- **Verify `hippocampus_mask_on_grid()` lands on the hippocampus** in one
  preprocessed volume (overlay and look). Axis order between ANTs and
  nibabel/nilearn has not been checked on real data.
- Build `atrophy`, `shortcut`, and `both` at p ∈ {0.5, 0.7, 0.9, 1.0}, plus
  `--flip-shortcut` test sets.
- Resampling control for the paper: train on controls vs controls (both
  through the jitter warp, random labels) and show chance accuracy.

---

## 3. Write `train.py`

Must use `subject_level_splits` and call `check_split_integrity()` on every fold.
Reads the benchmark `manifest.csv` (`path, subject_id, label`).

- Mixed precision (volumes are large; batch size will be 2–8)
- Class-weighted loss (labels are balanced by construction now, but keep it
  for robustness)
- Needs a CUDA build of PyTorch for RTX 50-series (CUDA 12.8+)
- Early stopping on validation AUC, not accuracy
- One checkpoint per fold, in the format `run_evaluation.py` expects
- Log the `demographic_summary()` of each split to the run directory
- Seed everything and record the seed

---

## 4. Ablations — blocked until a trained checkpoint exists

- `lower_upper_split` in `LRP3D` defaults to 0.5, inherited from the VGG16
  block-3 boundary. Arbitrary for a 3D ResNet. Sweep it and report.
- `patch=8` in `insertion_deletion_auc` sets the resolution ceiling on every
  faithfulness number in the paper. Sweep {4, 8, 16} and show the ranking is
  stable. An obvious reviewer question; cheap to pre-empt.

---

## 5. Cross-validation against a reference implementation

Validate `xai/lrp3d.py` against Zennit on a small 2D model where both apply.
Turns "we implemented LRP" into "our implementation agrees with the reference to
within X". Add `zennit` as a dev-only dependency.
