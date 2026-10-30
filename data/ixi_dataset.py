"""
IXI loading: download, filename parsing, demographics.

IXI (Imperial College London, brain-development.org) is ~580 T1-weighted scans
of healthy adults aged 20-86, released under CC BY-SA 3.0 with no application
step. It replaces ADNI as the source of real anatomy for the ground-truth
benchmark; disease is simulated on top of it by ``data/simulate.py``.

Two properties of IXI matter for the study design:

  - **One scan per subject.** The multi-visit leakage that dominates ADNI work
    does not arise from the raw data. It *can* be reintroduced if both a
    control and an atrophied copy of the same brain are generated; those share
    a subject_id so ``check_split_integrity()`` keeps them on one side.

  - **Three acquisition sites** (Guys: Philips 1.5T, HH: Philips 3T, IOP:
    GE 1.5T). Site is a strong, unplanted confound. ``assign_labels`` in
    ``simulate.py`` balances labels within site for that reason, and site is
    recorded on every record so it can be reported next to
    ``demographic_summary()``.

Attribution required by the licence: acknowledge the IXI dataset and
brain-development.org in the paper.
"""

from __future__ import annotations

import csv
import os
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = ["IXI_T1_URL", "IXI_DEMOGRAPHICS_URL", "IXISubject",
           "parse_ixi_filename", "download_ixi", "load_ixi_demographics",
           "ixi_subjects", "write_preprocess_manifest"]

IXI_BASE = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"
IXI_T1_URL = f"{IXI_BASE}/IXI-T1.tar"          # ~4.5 GB
IXI_DEMOGRAPHICS_URL = f"{IXI_BASE}/IXI.xls"

# e.g. IXI002-Guys-0828-T1.nii.gz
_NAME_RE = re.compile(r"^IXI(?P<num>\d+)-(?P<site>[A-Za-z]+)-(?P<scan>\d+)-T1")


@dataclass(frozen=True)
class IXISubject:
    """One IXI T1 scan. No label: labels are assigned by the benchmark."""
    path: str
    subject_id: str          # "IXI002"
    site: str                # "Guys" | "HH" | "IOP"
    age: Optional[float] = None
    sex: Optional[str] = None


def parse_ixi_filename(name: str) -> Tuple[str, str]:
    """(subject_id, site) from an IXI file name or a preprocessed .npy name.

    Preprocessing writes ``{subject_id}_{stem}.npy``, so both raw and
    preprocessed names are accepted.
    """
    base = os.path.basename(name)
    if "_" in base and base.startswith("IXI") and not _NAME_RE.match(base):
        base = base.split("_", 1)[1]
    m = _NAME_RE.match(base)
    if not m:
        raise ValueError(f"not an IXI T1 file name: {name}")
    return f"IXI{int(m.group('num')):03d}", m.group("site")


def download_ixi(dest: str, include_images: bool = True) -> Path:
    """Fetch the T1 archive and demographics into ``dest`` and extract.

    Skips anything already present, so it is safe to re-run after an
    interrupted download. Returns the directory holding the .nii.gz files.
    """
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)

    xls = dest_p / "IXI.xls"
    if not xls.exists():
        urllib.request.urlretrieve(IXI_DEMOGRAPHICS_URL, xls)

    img_dir = dest_p / "T1"
    if include_images and not any(img_dir.glob("*.nii.gz")):
        tar_path = dest_p / "IXI-T1.tar"
        if not tar_path.exists():
            print(f"downloading {IXI_T1_URL} (~4.5 GB)")
            urllib.request.urlretrieve(IXI_T1_URL, tar_path)
        img_dir.mkdir(exist_ok=True)
        with tarfile.open(tar_path) as tf:
            # "data" filter refuses absolute paths and links leaving dest.
            tf.extractall(img_dir, filter="data")
    return img_dir


def load_ixi_demographics(path: str) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
    """Map subject_id -> (age, sex) from IXI.xls or a CSV export of it.

    The spreadsheet has duplicate rows for some subjects and missing ages
    for others; the first row with a usable age wins. SEX_ID is 1 = male,
    2 = female. Reading .xls needs pandas + xlrd; exporting to CSV avoids
    both.
    """
    if path.lower().endswith((".xls", ".xlsx")):
        try:
            import pandas as pd
        except ImportError as exc:                    # pragma: no cover
            raise ImportError("reading IXI.xls needs pandas and xlrd; "
                              "or export it to CSV") from exc
        rows = pd.read_excel(path).to_dict("records")
    else:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))

    out: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    # The real header is "SEX_ID (1=m, 2=f)"; match by prefix so an exact
    # "SEX_ID" (or a re-export) works too.
    sex_col = next((k for k in (rows[0] if rows else {}) if str(k).startswith("SEX_ID")), "SEX_ID")
    for row in rows:
        try:
            sid = f"IXI{int(float(row['IXI_ID'])):03d}"
        except (KeyError, TypeError, ValueError):
            continue
        age = _float_or_none(row.get("AGE"))
        sex = {1: "M", 2: "F"}.get(_int_or_none(row.get(sex_col)))
        if sid not in out or (out[sid][0] is None and age is not None):
            out[sid] = (age, sex)
    return out


def ixi_subjects(directory: str,
                 demographics: Optional[str] = None,
                 pattern: str = "*") -> List[IXISubject]:
    """Every IXI T1 file under ``directory``, raw (.nii.gz) or preprocessed
    (.npy), joined with demographics when given. Sorted by subject_id."""
    demo = load_ixi_demographics(demographics) if demographics else {}
    out: List[IXISubject] = []
    for p in sorted(Path(directory).glob(pattern)):
        if not p.name.endswith((".nii.gz", ".nii", ".npy")):
            continue
        try:
            sid, site = parse_ixi_filename(p.name)
        except ValueError:
            continue
        age, sex = demo.get(sid, (None, None))
        out.append(IXISubject(str(p), sid, site, age, sex))

    ids = [s.subject_id for s in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IXI subject ids in directory; IXI has one "
                         "T1 per subject, so this is a stale or mixed folder")
    return out


def write_preprocess_manifest(subjects: List[IXISubject], out_csv: str) -> None:
    """CSV in the format ``data/preprocess.py:preprocess_cohort`` reads."""
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "subject_id", "label"])
        for s in subjects:
            w.writerow([s.path, s.subject_id, -1])


def _float_or_none(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f                      # NaN from pandas


def _int_or_none(v) -> Optional[int]:
    f = _float_or_none(v)
    return None if f is None else int(f)
