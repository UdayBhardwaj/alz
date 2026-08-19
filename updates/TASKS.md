# TASKS

Ordered. Work top-down. Re-run `python -m tests.test_synthetic` after each item.

---

## 1. BUG — conservation check is broken

`LRP3D.conservation_error()` in `xai/lrp3d.py` calls `self(x)`, which
max-normalises `R` before returning, then compares `R.sum()` against the raw
logit. The ratio is meaningless.

Add an unnormalised path through the propagator (an internal method, or a flag)
so total input relevance can be compared against the initialising logit. Add a
test asserting the relative error is below a sensible threshold on the synthetic
model.

Worth doing properly: almost no applied XAI paper reports conservation, so this
becomes an implementation-correctness claim in the methods section.

---

## 2. METHOD — fold BatchNorm into preceding convolutions

BatchNorm is currently propagated as its own epsilon-rule layer. Canonical LRP
(Montavon et al. 2019) folds BN into the preceding conv first; treating it
separately distorts the relevance distribution.

The 2D paper used VGG16, which has no BN, so this never arose. It matters here.

- Implement `fold_batchnorm()` in `models/resnet3d.py`, returning an LRP-ready
  copy with BN absorbed into conv weights and biases.
- Use it in `lrp_layers()`.
- Add a test asserting the folded model's forward pass matches the original to
  float tolerance. **This test is the whole safety net** — a silently wrong fold
  produces plausible-looking maps and corrupts every downstream number.

---

## 3. Write `train.py`

Must use `subject_level_splits` and call `check_split_integrity()` on every fold.

- Mixed precision (volumes are large; batch size will be 2–8)
- Class-weighted loss — ADNI is imbalanced
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
