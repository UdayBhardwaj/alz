## Classification and shortcut reliance

| Condition | Test AUC (mean ± sd over folds) | Acc on evaluated set | Acc with shortcut flipped |
|---|---|---|---|
| null | 0.515 ± 0.036 | 0.484 | – |
| atrophy | 0.992 ± 0.006 | 0.941 | – |
| shortcut | 0.826 ± 0.070 | 0.747 | 0.253 |
| both_p90 | 0.980 ± 0.025 | 0.924 | 0.921 |

## Condition `null`: per-method scores (mean [95% CI over subjects])

| Method | Faithfulness (ins − del) | Plausibility (hippocampus mass) | Plausibility enrichment | Atrophy AUPRC lift | Shortcut AUPRC lift |
|---|---|---|---|---|---|
| LRP | 0.4651 [0.4432, 0.4864] | 0.002 [0.002, 0.002] | 0.46 [0.43, 0.48] | – | – |
| GradCAM | 0.0534 [0.0366, 0.0718] | 0.004 [0.004, 0.004] | 0.73 [0.70, 0.76] | – | – |
| GradCAM++ | 0.0724 [0.0544, 0.0913] | 0.004 [0.004, 0.004] | 0.85 [0.84, 0.86] | – | – |
| IG | 0.4077 [0.3836, 0.4303] | 0.002 [0.002, 0.002] | 0.30 [0.29, 0.32] | – | – |
| GradInput | 0.3998 [0.3762, 0.4223] | 0.001 [0.001, 0.002] | 0.29 [0.28, 0.30] | – | – |
| Occlusion | 0.6955 [0.6702, 0.7198] | 0.004 [0.004, 0.004] | 0.74 [0.68, 0.80] | – | – |
| Random | -0.0060 [-0.0247, 0.0140] | 0.005 [0.005, 0.005] | 1.00 [1.00, 1.00] | – | – |

Spearman ρ (plausibility vs faithfulness, per volume): LRP -0.36, GradCAM -0.05, GradCAM++ -0.39, IG -0.25, GradInput -0.25, Occlusion -0.49, Random +0.01

## Condition `atrophy`: per-method scores (mean [95% CI over subjects])

| Method | Faithfulness (ins − del) | Plausibility (hippocampus mass) | Plausibility enrichment | Atrophy AUPRC lift | Shortcut AUPRC lift |
|---|---|---|---|---|---|
| LRP | 0.6166 [0.5952, 0.6381] | 0.011 [0.011, 0.011] | 2.14 [2.06, 2.23] | 1.82 [1.77, 1.86] | – |
| GradCAM | 0.3014 [0.2759, 0.3268] | 0.006 [0.006, 0.006] | 1.14 [1.12, 1.16] | 1.32 [1.26, 1.39] | – |
| GradCAM++ | 0.2919 [0.2653, 0.3179] | 0.006 [0.006, 0.006] | 1.14 [1.13, 1.15] | 1.25 [1.20, 1.30] | – |
| IG | 0.5575 [0.5330, 0.5828] | 0.007 [0.007, 0.007] | 1.40 [1.36, 1.43] | 1.58 [1.55, 1.60] | – |
| GradInput | 0.5456 [0.5211, 0.5714] | 0.007 [0.006, 0.007] | 1.27 [1.23, 1.30] | 1.57 [1.55, 1.60] | – |
| Occlusion | 0.8593 [0.8481, 0.8706] | 0.031 [0.030, 0.033] | 6.09 [5.80, 6.38] | 3.17 [3.03, 3.31] | – |
| Random | 0.0133 [-0.0126, 0.0394] | 0.005 [0.005, 0.005] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | – |

Spearman ρ (plausibility vs faithfulness, per volume): LRP +0.28, GradCAM +0.32, GradCAM++ +0.34, IG +0.16, GradInput -0.02, Occlusion +0.37, Random +0.04

## Condition `shortcut`: per-method scores (mean [95% CI over subjects])

| Method | Faithfulness (ins − del) | Plausibility (hippocampus mass) | Plausibility enrichment | Atrophy AUPRC lift | Shortcut AUPRC lift |
|---|---|---|---|---|---|
| LRP | 0.6359 [0.6156, 0.6559] | 0.001 [0.001, 0.001] | 0.22 [0.22, 0.23] | – | 17.62 [16.35, 18.99] |
| GradCAM | 0.3465 [0.3214, 0.3707] | 0.004 [0.004, 0.004] | 0.79 [0.78, 0.81] | – | 7.36 [6.58, 8.23] |
| GradCAM++ | 0.3219 [0.2970, 0.3451] | 0.004 [0.004, 0.004] | 0.81 [0.80, 0.82] | – | 8.19 [7.36, 9.04] |
| IG | 0.5844 [0.5609, 0.6072] | 0.001 [0.001, 0.001] | 0.19 [0.18, 0.20] | – | 8.17 [7.53, 8.85] |
| GradInput | 0.5723 [0.5484, 0.5953] | 0.001 [0.001, 0.001] | 0.16 [0.15, 0.17] | – | 8.27 [7.63, 8.97] |
| Occlusion | 0.8828 [0.8740, 0.8913] | 0.001 [0.001, 0.001] | 0.18 [0.16, 0.20] | – | 29.62 [27.62, 31.75] |
| Random | -0.0302 [-0.0607, 0.0010] | 0.005 [0.005, 0.005] | 1.00 [1.00, 1.00] | – | 1.01 [1.00, 1.01] |

Spearman ρ (plausibility vs faithfulness, per volume): LRP +0.25, GradCAM -0.41, GradCAM++ -0.50, IG -0.24, GradInput +0.19, Occlusion +0.18, Random -0.04

## Condition `both_p90`: per-method scores (mean [95% CI over subjects])

| Method | Faithfulness (ins − del) | Plausibility (hippocampus mass) | Plausibility enrichment | Atrophy AUPRC lift | Shortcut AUPRC lift |
|---|---|---|---|---|---|
| LRP | 0.5581 [0.5292, 0.5888] | 0.011 [0.011, 0.012] | 2.21 [2.11, 2.31] | 1.67 [1.63, 1.73] | 1.33 [1.27, 1.39] |
| GradCAM | 0.3002 [0.2705, 0.3315] | 0.006 [0.006, 0.006] | 1.15 [1.14, 1.17] | 1.34 [1.29, 1.40] | 2.53 [2.25, 2.84] |
| GradCAM++ | 0.3009 [0.2707, 0.3319] | 0.006 [0.006, 0.006] | 1.17 [1.16, 1.18] | 1.25 [1.21, 1.30] | 2.67 [2.45, 2.90] |
| IG | 0.5047 [0.4729, 0.5379] | 0.007 [0.007, 0.007] | 1.40 [1.36, 1.44] | 1.45 [1.43, 1.48] | 1.07 [1.05, 1.08] |
| GradInput | 0.4981 [0.4657, 0.5316] | 0.007 [0.006, 0.007] | 1.28 [1.25, 1.31] | 1.47 [1.44, 1.50] | 1.01 [1.00, 1.03] |
| Occlusion | 0.8023 [0.7862, 0.8196] | 0.025 [0.024, 0.027] | 4.90 [4.62, 5.18] | 2.57 [2.45, 2.69] | 3.41 [3.03, 3.81] |
| Random | 0.0068 [-0.0163, 0.0300] | 0.005 [0.005, 0.005] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.01 [1.00, 1.01] |

Spearman ρ (plausibility vs faithfulness, per volume): LRP +0.44, GradCAM +0.33, GradCAM++ +0.39, IG +0.15, GradInput -0.12, Occlusion +0.54, Random +0.02
