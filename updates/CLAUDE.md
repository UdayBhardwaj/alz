# CLAUDE.md

## What this is

Evaluation framework for a research paper on XAI in 3D MRI-based Alzheimer's
classification. Follow-up to *Explainable Facial Expression Recognition: A
Comparative Study of LRP and Grad-CAM* (ICDAM 2026, Springer LNNS).

**The central claim:** explanation *plausibility* (the heatmap lands on the
hippocampus, as a clinician expects) and *faithfulness* (those voxels actually
drive the prediction) are independent properties that can diverge. Most AD-XAI
papers only demonstrate the former and assert the latter.

This means the evaluation code is not supporting infrastructure — it *is* the
result. A silent bug in `eval/faithfulness.py` invalidates the paper. Treat
correctness there as higher priority than features anywhere else.

## Data: ground-truth benchmark on IXI (ADNI and OASIS access were refused)

Healthy IXI T1 scans (CC BY-SA 3.0, no application) with disease simulated on
top, so the true signal location is known voxel by voxel:

- `data/ixi_dataset.py` — download, filename parsing, demographics.
- `data/simulate.py` — hippocampal atrophy warp + ground-truth |detJ − 1| map;
  a shared sub-voxel jitter so controls are resampled exactly like positives;
  planted shortcut bump; site/age-balanced labels; exact shortcut correlation.
- `data/build_benchmark.py` — one condition (`atrophy` | `shortcut` | `both`)
  → volumes, GT maps, `manifest.csv`. `--flip-shortcut` builds the reliance
  test set.
- `eval/anatomical.py:ground_truth_scores` — the correctness axis (mass in GT,
  AUPRC vs GT, precision@|GT|), next to plausibility and faithfulness.

Pipeline: `download_ixi` → `preprocess_cohort` → `build_benchmark` → train →
`run_evaluation`. Before building a cohort, check `hippocampus_mask_on_grid()`
visually on one preprocessed volume: a misplaced mask corrupts every GT number
and nothing downstream would catch it.

## Working agreement

- Run `python -m tests.test_synthetic` before and after every change. All 17 pass.
- **Never weaken or delete a test to make it pass.** If a test fails, either the
  code is wrong or the test encodes a wrong assumption — say which, don't
  quietly adjust the threshold.
- Ask before adding a dependency.
- Prefer a correct slow implementation over a fast wrong one. This runs offline
  on a few hundred volumes, not in production.
- Keep the numpy/torch style already in the files: type hints, docstrings that
  explain *why* rather than restate the signature.

## Two tests that matter more than they look

`test_lrp_class_sensitivity` — catches the failure where relevance propagation
collapses to something class-independent. The maps still look plausible; they're
just meaningless. This is the exact failure the paper is about.

`test_faithfulness_ordering` — builds a by-construction-optimal saliency map and
asserts the metric ranks it above noise. If this fails, no number the harness
produces means anything.

## Three failure modes that sink papers in this field

1. **Subject-level leakage.** ADNI has multiple visits per participant. Split by
   scan and the network memorises individuals, not disease. This is the
   mechanism behind the 99% accuracies that never replicate. Any code path
   producing a train/test split must call `check_split_integrity()`.

2. **The Kaggle "Alzheimer's MRI" dataset.** Pre-split 2D slices with subjects
   spanning the split. Several 2025 papers use it. Never use it here.

3. **Demographic imbalance.** If the AD group is meaningfully older than CN, the
   model classifies on age-related atrophy and the explanations faithfully
   highlight it. `demographic_summary()` output belongs in the paper.

## Architecture notes

`models/resnet3d.py` exposes `lrp_layers()` — the ordered list of callables whose
composition equals `forward()`. Residual blocks appear as **single entries**;
`xai/lrp3d.py` passes each whole block to the epsilon rule and lets autograd
split relevance across the main and skip branches in proportion to their
contribution to z. That is the canonical residual LRP split, and it is why this
is real LRP rather than the Gradient×Input proxy the 2D paper had to use.

If you change `lrp_layers()`, that property is what you must preserve.

`lrp_layers()` returns layers from a BatchNorm-folded copy (`fold_batchnorm`),
made fresh on each call so a later-trained or randomised model is never
explained with stale weights. `test_lrp_conservation` checks that input
relevance sums to the logit on a bias-free network (currently ~1e-5); it is
what caught the z^B sign error, so keep it passing.

`xai/baselines.py` retains Gradient×Input deliberately, as a labelled baseline —
it lets the paper quantify what the 2D substitution cost.

## Current state

Working, tested, no trained checkpoint yet. Open work is in `TASKS.md`.
