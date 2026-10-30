"""
Preprocess every IXI T1 scan in parallel with the "template" brain extraction.

Resumable: scans whose output .npy already exists are skipped, so an
interrupted run can simply be restarted. Failures are listed at the end and
in ``failed.txt`` rather than stopping the batch.

    python preprocess_ixi.py --raw ixi/T1 --out ixi/npy --workers 8
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _one(nifti: str, out: str, threads: int) -> str:
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    from data.preprocess import PreprocessConfig, preprocess_volume
    preprocess_volume(nifti, out, PreprocessConfig(skullstrip_method="template"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    args = ap.parse_args()

    from data.ixi_dataset import ixi_subjects
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in ixi_subjects(args.raw):
        stem = Path(s.path).name.replace(".nii.gz", "")
        target = out / f"{s.subject_id}_{stem}.npy"
        if not target.exists():
            jobs.append((s.path, str(target)))
    print(f"{len(jobs)} scans to process with {args.workers} workers", flush=True)

    t0, failed = time.time(), []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_one, n, o, args.threads_per_worker): n for n, o in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as exc:
                failed.append(f"{futs[f]}\t{type(exc).__name__}: {exc}")
            if i % 20 == 0 or i == len(jobs):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(jobs)}  {rate:.1f}s/scan  eta {rate * (len(jobs) - i) / 60:.0f} min"
                      f"  failed {len(failed)}", flush=True)
    if failed:
        (out / "failed.txt").write_text("\n".join(failed))
        print(f"{len(failed)} failed; see {out / 'failed.txt'}")


if __name__ == "__main__":
    main()
