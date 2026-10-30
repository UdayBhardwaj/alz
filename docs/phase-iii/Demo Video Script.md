# Phase III demo video: script (about 3 minutes)

Record the screen with **Win + Alt + R** (Xbox Game Bar) or OBS. One person narrates.
Open a terminal in the project folder (`alz-main`) before you start.

---

## 1. The problem (20 s)
**Show:** slide 2 of the Phase III presentation.

> "Explanation heatmaps for Alzheimer's MRI models usually highlight the hippocampus, and papers take that as proof the model learned the disease. Our project tests whether a plausible heatmap is also faithful. ADNI and OASIS refused access, so we built a benchmark on open IXI brain scans where we insert the disease signal ourselves and know exactly where it is."

## 2. The code (30 s)
**Show:** the project folder in VS Code or File Explorer.

> "The code is organised by pipeline stage: `data/` loads IXI, preprocesses scans and simulates atrophy; `models/` is the 3D ResNet; `xai/` has six explanation methods including LRP; `eval/` scores every heatmap on plausibility, faithfulness and correctness."

Open `data/simulate.py` and scroll briefly through the docstring at the top.

## 3. Live demo (60 s)
**Run:**

```
C:\Users\bhard\alzenv\Scripts\python.exe demo.py
```

**Narrate the console output:**
> "We ask for a 25% hippocampal volume loss and measure 23%. The deformation never folds. The control scan carries no atrophy ground truth, and the planted shortcut changes zero voxels outside its mask, so the ground truth is exact."

**Open `demo_output.png`** and walk through the panels:
> "Top row: a control scan, the same scan with simulated atrophy, and the difference. Bottom row: the ground-truth map of where tissue changed, and the faint planted shortcut with its ground truth. A correct explanation must match these maps."

## 4. Verification (40 s)
**Run** (if Windows allows PyTorch to load, see note below):

```
C:\Users\bhard\alzenv\Scripts\python.exe -m tests.test_synthetic
```

Otherwise **show** `Test Log 2026-10-22.txt`.

> "Seventeen synthetic tests pass. The important one is LRP conservation: it caught a sign error in the LRP input rule that made relevance sum to minus the model output, while the heatmaps still looked plausible. After the fix, LRP went from worse-than-random to better-than-random faithfulness. That is our thesis in miniature."

## 5. Plan (20 s)
**Show:** slide 10 (benchmark conditions) and slide 13 (execution plan).

> "Next we preprocess all IXI scans, build the three benchmark conditions, and train on our RTX 5080. Results come in December and January, and the paper follows in February."

---

### Note: PyTorch blocked by Windows
On 29 October, Windows Smart App Control blocked PyTorch's DLLs (`WinError 4551`). `demo.py` does not need PyTorch and runs normally. To run the tests on camera, PyTorch must be allowed to load. That is a Windows security setting (Settings → Privacy & security → Windows Security → App & browser control → Smart App Control); decide it yourself, or use the saved test log instead.
