"""
Anatomical grounding of relevance maps.

This is the 3D counterpart of the FACS action-unit analysis in the 2D FER
paper: instead of asking whether saliency lands on the muscles that define an
expression, we ask whether it lands on the structures that define Alzheimer's
pathology.

The central quantity is **relevance mass per region**, normalised by region
volume. Raw mass is misleading — cerebral white matter is enormous and will
dominate any unnormalised ranking regardless of what the model actually used.

The regions that matter for AD, in rough order of expected atrophy:
    hippocampus, entorhinal cortex, amygdala, parahippocampal gyrus,
    posterior cingulate, precuneus, inferior lateral ventricles (enlargement).

Design note for the paper's argument: plausibility and faithfulness are
separate axes. A map can score highly here — hippocampus lit up, exactly what
a radiologist expects — while scoring poorly on Insertion AUC, meaning those
voxels are not what the network actually used. Reporting both, and showing
where they disagree, is the contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["AD_REGIONS", "RegionRelevance", "region_relevance",
           "plausibility_score", "load_atlas"]


# Region names as they appear in the Harvard-Oxford subcortical and
# cortical atlases. Adjust if using AAL or Desikan-Killiany.
AD_REGIONS: Dict[str, List[str]] = {
    "medial_temporal": [
        "Left Hippocampus", "Right Hippocampus",
        "Left Amygdala", "Right Amygdala",
        "Parahippocampal Gyrus, anterior division",
        "Parahippocampal Gyrus, posterior division",
    ],
    "posterior_cortical": [
        "Cingulate Gyrus, posterior division",
        "Precuneous Cortex",
    ],
    "ventricular": [
        "Left Lateral Ventricle", "Right Lateral Ventricle",
    ],
}


@dataclass
class RegionRelevance:
    """Relevance summary for one anatomical region."""
    name: str
    mass: float                 # total relevance in region
    fraction: float             # share of whole-volume relevance
    density: float              # mass per voxel
    enrichment: float           # density relative to whole-brain mean
    n_voxels: int

    def __repr__(self) -> str:
        return (f"{self.name:<44s} frac={self.fraction:6.2%} "
                f"enrich={self.enrichment:5.2f}x")


def load_atlas(atlas_name: str = "harvard_oxford"):
    """Fetch a labelled atlas resampled to the analysis grid.

    Requires nilearn. Kept as a thin wrapper so the atlas choice is a single
    edit, and so the rest of the module is testable without the download.

    Returns (label_volume, index_to_name).
    """
    try:
        from nilearn import datasets, image
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "nilearn is required for atlas loading: pip install nilearn"
        ) from exc

    if atlas_name == "harvard_oxford":
        sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        return np.asarray(image.load_img(sub.maps).dataobj), list(sub.labels)
    if atlas_name == "aal":
        aal = datasets.fetch_atlas_aal()
        return np.asarray(image.load_img(aal.maps).dataobj), list(aal.labels)
    raise ValueError(f"unknown atlas: {atlas_name}")


def region_relevance(saliency: np.ndarray,
                     atlas: np.ndarray,
                     labels: Sequence[str],
                     brain_mask: Optional[np.ndarray] = None,
                     top_k: Optional[int] = None) -> List[RegionRelevance]:
    """Decompose a relevance map across atlas regions.

    Parameters
    ----------
    saliency
        Relevance map, shape (D, H, W). Non-negative.
    atlas
        Integer label volume on the same grid. Index 0 is background.
    labels
        Region names, indexed by atlas value.
    brain_mask
        Optional boolean mask. Relevance outside the brain — skull, neck,
        background — is a preprocessing failure and should be excluded from
        the normalisation, not silently averaged in.
    top_k
        Return only the k most enriched regions.

    Returns
    -------
    List of RegionRelevance, sorted by enrichment.
    """
    if saliency.shape != atlas.shape:
        raise ValueError(
            f"saliency {saliency.shape} and atlas {atlas.shape} must share a grid; "
            "resample the atlas to the analysis resolution first"
        )

    sal = np.maximum(saliency, 0.0).astype(np.float64)
    if brain_mask is not None:
        sal = sal * brain_mask

    total = sal.sum()
    if total < 1e-12:
        return []

    n_brain = int(brain_mask.sum()) if brain_mask is not None else sal.size
    global_density = total / max(n_brain, 1)

    out: List[RegionRelevance] = []
    for idx in np.unique(atlas):
        if idx == 0:
            continue
        region_mask = atlas == idx
        n = int(region_mask.sum())
        if n == 0:
            continue
        mass = float(sal[region_mask].sum())
        density = mass / n
        out.append(RegionRelevance(
            name=labels[int(idx)] if int(idx) < len(labels) else f"region_{idx}",
            mass=mass,
            fraction=mass / total,
            density=density,
            enrichment=density / global_density if global_density > 0 else 0.0,
            n_voxels=n,
        ))

    out.sort(key=lambda r: r.enrichment, reverse=True)
    return out[:top_k] if top_k else out


def plausibility_score(regions: List[RegionRelevance],
                       target_groups: Optional[Dict[str, List[str]]] = None
                       ) -> Dict[str, float]:
    """Fraction of total relevance falling in AD-associated structures.

    This is the "does it look right to a clinician" axis. Pair it with
    Insertion AUC to test whether plausible maps are actually faithful ones.
    """
    groups = target_groups or AD_REGIONS
    by_name = {r.name: r for r in regions}

    scores: Dict[str, float] = {}
    for group, names in groups.items():
        scores[group] = float(sum(by_name[n].fraction
                                  for n in names if n in by_name))
    scores["ad_total"] = float(sum(scores.values()))
    return scores
