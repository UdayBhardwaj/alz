"""
ADNI dataset loading with subject-level splitting.

The single most important thing in this file is ``subject_level_splits``.

ADNI contains multiple visits per participant, often years apart but
structurally near-identical. Splitting at the scan level puts a subject's
baseline scan in train and their month-12 scan in test, and the network
recognises the individual rather than the disease. This is the mechanism behind
the 99%+ accuracies that appear regularly in the AD deep-learning literature
and fail to replicate. Reviewers who know the field check for it first.

The same applies to the popular Kaggle "Alzheimer's MRI" dataset, which is
distributed as pre-split 2D slices with subjects spanning the split. It should
not be used for any result intended for publication.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ScanRecord", "ADNIDataset", "subject_level_splits",
           "check_split_integrity"]

CLASS_MAP = {"CN": 0, "AD": 1, "MCI": 2}


@dataclass(frozen=True)
class ScanRecord:
    """One MRI volume and the metadata needed for correct splitting."""
    path: str
    subject_id: str
    label: int
    visit: str = "bl"
    age: Optional[float] = None
    sex: Optional[str] = None
    apoe4: Optional[int] = None


class ADNIDataset(Dataset):
    """Preprocessed T1-weighted volumes.

    Expects volumes already run through ``data/preprocess.py``: N4-corrected,
    skull-stripped, MNI-registered, intensity-normalised, saved as .npy at a
    fixed grid. Doing preprocessing on the fly wastes hours per epoch.
    """

    def __init__(self,
                 records: Sequence[ScanRecord],
                 augment: bool = False,
                 target_shape: Tuple[int, int, int] = (96, 96, 96)):
        self.records = list(records)
        self.augment = augment
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        vol = np.load(rec.path).astype(np.float32)

        if vol.shape != self.target_shape:
            raise ValueError(
                f"{rec.path} has shape {vol.shape}, expected {self.target_shape}. "
                "Re-run preprocessing rather than resizing here."
            )

        if self.augment:
            vol = self._augment(vol)

        return torch.from_numpy(vol).unsqueeze(0), rec.label

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        """Conservative augmentation.

        Deliberately excludes left-right flipping. AD atrophy is asymmetric,
        and flipping teaches the model that laterality is noise — which also
        corrupts any subsequent lateralised interpretability analysis.
        """
        rng = np.random.default_rng()
        if rng.random() < 0.5:
            shift = rng.integers(-4, 5, size=3)
            vol = np.roll(vol, shift, axis=(0, 1, 2))
        if rng.random() < 0.3:
            vol = vol * rng.uniform(0.95, 1.05)
        if rng.random() < 0.3:
            vol = vol + rng.normal(0, 0.02, vol.shape).astype(np.float32)
        return vol

    def label_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for r in self.records:
            counts[r.label] = counts.get(r.label, 0) + 1
        return counts


def subject_level_splits(records: Sequence[ScanRecord],
                         n_folds: int = 5,
                         seed: int = 42
                         ) -> Iterator[Tuple[List[ScanRecord], List[ScanRecord]]]:
    """Stratified group k-fold: every scan from a subject stays on one side.

    Stratification is on the subject's label so class balance holds across
    folds; grouping is on subject_id so no participant spans the split.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:                        # pragma: no cover
        raise ImportError("scikit-learn required") from exc

    records = list(records)
    groups = [r.subject_id for r in records]
    y = [r.label for r in records]

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=seed)
    for train_idx, test_idx in splitter.split(np.zeros(len(records)), y, groups):
        yield ([records[i] for i in train_idx],
               [records[i] for i in test_idx])


def check_split_integrity(train: Sequence[ScanRecord],
                          test: Sequence[ScanRecord]) -> None:
    """Raise if any subject appears on both sides.

    Call this on every fold. It costs microseconds and catches the error that
    invalidates the entire study.
    """
    overlap = {r.subject_id for r in train} & {r.subject_id for r in test}
    if overlap:
        raise AssertionError(
            f"subject leakage: {len(overlap)} subject(s) in both train and test, "
            f"e.g. {sorted(overlap)[:5]}"
        )


def demographic_summary(records: Sequence[ScanRecord]) -> Dict[str, object]:
    """Age / sex / APOE4 breakdown per class.

    Include this table in the paper. If the AD group is six years older than
    the CN group, the network can classify on age-related atrophy alone and
    the explanations will faithfully highlight it.
    """
    out: Dict[str, object] = {}
    for cls_name, cls_idx in CLASS_MAP.items():
        subset = [r for r in records if r.label == cls_idx]
        if not subset:
            continue
        ages = [r.age for r in subset if r.age is not None]
        out[cls_name] = {
            "n_scans": len(subset),
            "n_subjects": len({r.subject_id for r in subset}),
            "age_mean": float(np.mean(ages)) if ages else None,
            "age_std": float(np.std(ages)) if ages else None,
            "pct_female": (100.0 * sum(r.sex == "F" for r in subset) / len(subset)),
            "pct_apoe4": (100.0 * sum(bool(r.apoe4) for r in subset
                                      if r.apoe4 is not None) / len(subset)),
        }
    return out
