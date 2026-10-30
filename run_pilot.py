"""
Pilot experiment: build the benchmark, train, evaluate, analyse.

Conditions (one shared label assignment, seed 0, so every condition sees the
same subjects in the same folds):

    null       resampling control: warp only, must stay at chance
    atrophy    condition A
    shortcut   condition C, plus a flipped test set
    both_p90   condition B with p = 0.9, plus a flipped test set

    python run_pilot.py --npy ixi/npy --demographics ixi/IXI.csv \\
        --region-mask ixi/qc/hippocampus_96.npy --work work --folds 0

Stages can be run separately with --stages build train eval analyze; every
stage skips work whose output already exists, so the script can be re-run
after an interruption.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BUILDS = {
    "null":          dict(condition="null"),
    "atrophy":       dict(condition="atrophy"),
    "shortcut":      dict(condition="shortcut"),
    "shortcut_flip": dict(condition="shortcut", flip_shortcut=True),
    "both_p90":      dict(condition="both", p_shortcut=0.9),
    "both_p90_flip": dict(condition="both", p_shortcut=0.9, flip_shortcut=True),
}
TRAIN = ["null", "atrophy", "shortcut", "both_p90"]
EVAL = {  # evaluation set -> condition whose checkpoint explains it
    "null": "null", "atrophy": "atrophy", "shortcut": "shortcut",
    "shortcut_flip": "shortcut", "both_p90": "both_p90", "both_p90_flip": "both_p90",
}


def _build(tag: str, npy: str, demo: str, region: str, work: str) -> str:
    import numpy as np
    from data.build_benchmark import build_condition
    from data.ixi_dataset import ixi_subjects
    out = Path(work) / "bench" / tag
    if (out / "manifest.csv").exists():
        return f"{tag}: exists"
    subs = ixi_subjects(npy, demo, pattern="*.npy")
    build_condition(subs, np.load(region).astype(bool), out_dir=str(out), seed=0, **BUILDS[tag])
    return f"{tag}: built {len(subs)}"


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--demographics", required=True)
    ap.add_argument("--region-mask", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval-volumes", type=int, default=40, help="0 = every test subject")
    ap.add_argument("--tag", default="", help="suffix for runs/results/analysis dirs, so a full run "
                    "does not overwrite the pilot; the benchmark is shared")
    ap.add_argument("--stages", nargs="+", default=["build", "train", "eval", "analyze"])
    ap.add_argument("--build-procs", type=int, default=3, help="parallel benchmark builds (CPU heat)")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers for training")
    args = ap.parse_args()
    work, py = Path(args.work), sys.executable
    sfx = f"_{args.tag}" if args.tag else ""
    runs, results = work / f"runs{sfx}", work / f"results{sfx}"

    if "build" in args.stages:
        with ProcessPoolExecutor(args.build_procs) as ex:
            for msg in ex.map(_build, BUILDS, [args.npy] * 6, [args.demographics] * 6,
                              [args.region_mask] * 6, [str(work)] * 6):
                print(msg, flush=True)

    if "train" in args.stages:
        for c in TRAIN:
            out = runs / c
            if all((out / f"fold{k}.pt").exists() for k in args.folds) and (out / "summary.json").exists():
                continue
            run([py, "train.py", "--manifest", str(work / "bench" / c / "manifest.csv"),
                 "--out", str(out), "--folds", *map(str, args.folds),
                 "--epochs", str(args.epochs), "--workers", str(args.workers)])

    if "eval" in args.stages:
        for tag, c in EVAL.items():
            for k in args.folds:
                out = results / tag / f"fold{k}"
                if (out / "per_volume.csv").exists():
                    continue
                run([py, "run_evaluation.py",
                     "--checkpoint", str(runs / c / f"fold{k}.pt"),
                     "--manifest", str(runs / c / f"fold{k}_test_manifest.csv"),
                     "--bench", str(work / "bench" / tag / "manifest.csv"),
                     "--out", str(out)]
                    + (["--max-volumes", str(args.eval_volumes)] if args.eval_volumes > 0 else []))

    if "analyze" in args.stages:
        run([py, "analyze.py", "--results", str(results), "--runs", str(runs),
             "--out", str(work / f"analysis{sfx}"),
             "--conditions", "null", "atrophy", "shortcut", "both_p90"])


if __name__ == "__main__":
    main()
