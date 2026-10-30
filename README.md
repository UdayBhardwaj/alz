# Faithful or Plausible?

A ground-truth benchmark for explainable AI on 3D brain MRI. Alzheimer-like
hippocampal atrophy is simulated on healthy T1 scans from the open IXI
dataset, and a faint intensity shortcut is optionally planted far from the
hippocampus. Because the disease signal is inserted by the code, its location
is known voxel by voxel. Explanation maps can therefore be scored on three
axes:

- **Plausibility:** relevance concentrated on the hippocampus (atlas).
- **Faithfulness:** insertion minus deletion AUC.
- **Correctness:** AUPRC against the ground-truth atrophy and shortcut maps.

Methods compared: LRP (composite rules, BatchNorm folded, conservation
checked), Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion and
Gradient × Input, for a 3D ResNet-10.

B.E. CSE (AI & ML) project, Chandigarh University. Team: Uday Bhardwaj,
Aadarsh Ahlawat, Arun Kumar Bhagat, Karamjodh Singh. Supervisor: Ms Richa Tiwari.

## Results (5-fold cross-validation, 581 IXI subjects)

| Condition | Test AUC | Accuracy with shortcut moved |
|---|---|---|
| Resampling control (no signal) | 0.515 ± 0.036 | – |
| A: atrophy only | 0.992 ± 0.006 | – |
| C: shortcut only | 0.826 ± 0.070 | 0.253 (relies on it) |
| B: atrophy + shortcut, p = 0.9 | 0.980 ± 0.025 | 0.921 (barely uses it) |

Main findings:

- On atrophy, faithfulness and correctness rank the six methods identically.
- Plausibility exaggerates differences between methods, and its per-volume
  correlation with faithfulness is at most moderate and changes sign.
- Faithfulness stays high on a model that learned nothing.
- Grad-CAM and Grad-CAM++ give 2.7–3.0× chance relevance to the shortcut's
  position even in a model that never saw a shortcut.

Full tables are in `results/5fold/` (`tables.md`, `key_numbers.json`,
`extra_numbers.json`). The draft paper, report and slides are in
`docs/phase-iii/`.

## Layout

```
data/        IXI loader, preprocessing, atrophy/shortcut simulation, benchmark builder
models/      3D ResNet with BatchNorm folding for LRP
xai/         LRP and baseline attribution methods
eval/        faithfulness, anatomical plausibility, ground-truth correctness, sanity checks
tests/       synthetic test suite (python -m tests.test_synthetic)
train.py             subject-level cross-validated training
run_evaluation.py    explain a test set and score every map
analyze.py           tables, bootstrap CIs and the figure
location_control.py  shortcut-location control and logit-level reliance
run_pilot.py         build -> train -> eval -> analyze in one command
preprocess_ixi.py    parallel, resumable preprocessing of all IXI scans
qc_atlas.py          visual check that the hippocampus mask sits on the hippocampus
```

## Reproducing

Requires Python 3.12 (antspyx has no wheel for newer versions yet) and a CUDA GPU
for training.

```bash
pip install -r requirements.txt
python -m tests.test_synthetic                       # 18 tests, CPU, ~1 min
python preprocess_ixi.py --raw ixi/T1 --out ixi/npy --workers 5
python qc_atlas.py --nifti ixi/T1/IXI002-Guys-0828-T1.nii.gz --out ixi/qc
python run_pilot.py --npy ixi/npy --demographics ixi/IXI.csv \
    --region-mask ixi/qc/hippocampus_96.npy --work work \
    --folds 0 1 2 3 4 --eval-volumes 0 --tag full
```

`qc_atlas.py` writes `hippocampus_96.npy` and an overlay image; check that the
mask sits on the hippocampus before building any benchmark. Every stage skips
work whose output exists, so an interrupted run can be restarted.
Preprocessing is CPU-heavy; lower `--workers` if the machine runs hot.

## Data and licence

This work uses the IXI dataset (brain-development.org), released under
CC BY-SA 3.0. The raw scans are not redistributed here; download them from
brain-development.org. Benchmark data derived from IXI must be shared under
the same licence.
