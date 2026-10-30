"""
Ground-truth benchmark construction: simulated atrophy and planted shortcuts.

Real AD cohorts cannot tell us which voxels a classifier *should* use, so
explanation correctness can only be argued indirectly. Here the disease signal
is put in by hand, on healthy IXI brains, and its location is recorded voxel by
voxel. That turns three properties the paper separates into things that can be
measured on the same map:

    plausibility  - relevance lands in the hippocampus        (anatomical.py)
    faithfulness  - removing those voxels changes the output  (faithfulness.py)
    correctness   - relevance lands where the signal actually is (this file's GT)

Atrophy model
-------------
Phenomenological, not biophysical. Each connected component of the region mask
(left and right hippocampus are separate components) is shrunk toward its own
centroid by a smooth backward displacement

    u(x) = s * sum_k w_k(x) * (q_k(x) - c_k),     out(x) = in(x + u(x))

where ``q_k(x)`` is the nearest point of component k (x itself inside it) and
``w_k`` is 1 on the component and decays as a Gaussian of distance from it.
The structure shrinks uniformly; outside, the boundary's displacement fades
into surrounding tissue and CSF, which stretches to fill the space. This is what makes the
inferior horn of the lateral ventricle enlarge alongside the hippocampus, as it
does in AD. Simulators such as Simul@trophy (Khanal et al. 2017) are more
realistic; this one is dependency-free, exactly invertible-checked, and its
ground truth is the deformation itself rather than an approximation of it.

Resampling control
------------------
Warping a volume interpolates it, and interpolation smooths. If only atrophied
scans were resampled, a classifier could detect the smoothing instead of the
atrophy, and every explanation would "correctly" highlight an artefact. Every
volume, control or not, is therefore resampled once through a random smooth
sub-voxel jitter field; atrophy, where present, is added to that same field.
Both classes then carry the same interpolation signature everywhere.

Shortcut model
--------------
A faint Gaussian intensity bump at a fixed location far from the region of
interest. Its correlation with the label is set exactly (see
``assign_shortcuts``), so the degree to which a model *can* rely on it is a
controlled variable, and whether it *does* is measured by flipping it at test
time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

__all__ = ["AtrophyConfig", "ShortcutConfig", "SimulatedVolume",
           "volume_loss_to_scale", "atrophy_field", "jitter_field",
           "jacobian_determinant", "apply_warp", "shortcut_bump",
           "default_shortcut_center", "simulate_volume",
           "assign_labels", "assign_shortcuts"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AtrophyConfig:
    """Parameters of the atrophy simulation.

    ``volume_loss_range`` is the target fractional loss of region volume.
    Hippocampal volume in mild-to-moderate AD is typically 15-30% below
    age-matched controls, which is where the default sits. The loss actually
    achieved is measured and recorded per subject, because the smooth weight
    does not shrink the structure perfectly uniformly.
    """
    volume_loss_range: Tuple[float, float] = (0.15, 0.30)
    falloff_sigma: float = 3.0          # voxels; how far the deformation spreads
    jitter_amplitude: float = 0.5       # voxels; max shared sub-voxel jitter
    jitter_sigma: float = 8.0           # voxels; smoothness of the jitter
    gt_threshold: float = 0.02          # |J - 1| above this counts as changed
    interp_order: int = 1


@dataclass
class ShortcutConfig:
    """Parameters of the planted shortcut.

    The amplitude is in z-score units of the intensity-normalised volume.
    0.3 is well below tissue contrast and hard to see by eye, which is the
    point: a radiologist would not notice it, a CNN can.
    """
    amplitude: float = 0.3
    radius: float = 4.0                 # Gaussian sigma, voxels
    center: Optional[Tuple[int, int, int]] = None   # None -> default_shortcut_center
    gt_threshold: float = 0.05          # Gaussian truncated below this fraction of peak


@dataclass
class SimulatedVolume:
    """One benchmark volume and its ground truth, all on the output grid."""
    volume: np.ndarray
    atrophy_gt: np.ndarray              # float, |J - 1|; zero where untouched
    shortcut_gt: np.ndarray             # float, bump profile inside the brain
    meta: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deformation fields
# ---------------------------------------------------------------------------

def volume_loss_to_scale(volume_loss: float) -> float:
    """Displacement scale ``s`` giving the requested fractional volume loss
    for a uniform isotropic shrink: (1 + s)^-3 = 1 - loss."""
    if not 0.0 <= volume_loss < 1.0:
        raise ValueError(f"volume_loss must be in [0, 1), got {volume_loss}")
    return (1.0 - volume_loss) ** (-1.0 / 3.0) - 1.0


def _grid(shape: Tuple[int, ...]) -> np.ndarray:
    return np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape],
                                indexing="ij"))


def atrophy_field(region_mask: np.ndarray,
                  volume_loss: float,
                  falloff_sigma: float = 3.0) -> np.ndarray:
    """Backward displacement field (3, D, H, W) shrinking each component of
    ``region_mask`` toward its own centroid.

    Components are handled separately because a single centroid for a
    bilateral structure would sit on the midline and pull both hippocampi
    toward it instead of shrinking them.
    """
    s = volume_loss_to_scale(volume_loss)
    u = np.zeros((3,) + region_mask.shape, dtype=np.float64)
    if s == 0.0:
        return u

    labels, n = ndimage.label(region_mask.astype(bool))
    if n == 0:
        raise ValueError("region_mask is empty")

    grid = _grid(region_mask.shape)
    for k in range(1, n + 1):
        comp = labels == k
        c = np.array(ndimage.center_of_mass(comp)).reshape(3, 1, 1, 1)
        # Inside the structure: u = s (x - c), a uniform shrink, so a
        # star-shaped structure loses exactly the target volume. (A blurred
        # mask as weight puts w < 1 inside and under-shrinks: 13.9% for a 25%
        # target in the phantom test.)
        # Outside: the displacement of the nearest structure point q(x),
        # faded by a Gaussian of the distance to it. Using (x - c) outside
        # instead lets the displacement grow with distance from the centroid,
        # spreading "atrophy" across the brain (61% of GT mass near the
        # region, GT covering 43% of the brain, in the same test).
        d, idx = ndimage.distance_transform_edt(~comp, return_indices=True)
        w = np.exp(-d ** 2 / (2.0 * falloff_sigma ** 2))
        u += s * w[None] * (idx.astype(np.float64) - c)
    return u


def jitter_field(shape: Tuple[int, int, int],
                 rng: np.random.Generator,
                 amplitude: float = 0.5,
                 sigma: float = 8.0) -> np.ndarray:
    """Random smooth sub-voxel displacement applied to *every* volume.

    Exists only to give controls the same interpolation signature as
    atrophied volumes. Amplitude is the maximum displacement in voxels.
    """
    u = np.zeros((3,) + tuple(shape), dtype=np.float64)
    if amplitude <= 0:
        return u
    for d in range(3):
        f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
        peak = np.abs(f).max()
        u[d] = amplitude * f / peak if peak > 0 else 0.0
    return u


def jacobian_determinant(u: np.ndarray) -> np.ndarray:
    """det(I + grad u) of the map x -> x + u(x).

    For a backward field, values above 1 mean the output samples a larger
    input region (content shrinks), below 1 the reverse. Non-positive values
    mean the map folds and the warp is invalid.
    """
    J = np.empty((3, 3) + u.shape[1:], dtype=np.float64)
    for i in range(3):
        grads = np.gradient(u[i])
        for j in range(3):
            J[i, j] = grads[j] + (1.0 if i == j else 0.0)
    return (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
            - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
            + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))


def apply_warp(vol: np.ndarray, u: np.ndarray, order: int = 1) -> np.ndarray:
    """Backward warp: ``out(x) = vol(x + u(x))``."""
    coords = _grid(vol.shape) + u
    return ndimage.map_coordinates(vol, coords, order=order, mode="nearest")


# ---------------------------------------------------------------------------
# Shortcut
# ---------------------------------------------------------------------------

def shortcut_bump(shape: Tuple[int, int, int],
                  center: Sequence[float],
                  radius: float) -> np.ndarray:
    """Unit-peak isotropic Gaussian centred at ``center``."""
    grid = _grid(shape)
    c = np.asarray(center, dtype=np.float64).reshape(3, 1, 1, 1)
    d2 = ((grid - c) ** 2).sum(0)
    return np.exp(-d2 / (2.0 * radius ** 2))


def default_shortcut_center(region_mask: np.ndarray,
                            brain_mask: np.ndarray,
                            margin: float = 8.0) -> Tuple[int, int, int]:
    """Brain voxel farthest from the region of interest, at least ``margin``
    voxels inside the brain boundary.

    Computed once per cohort on a template brain mask and then held fixed:
    in registered space the shortcut must sit at the same coordinates in every
    subject, or it is not a shortcut the model can learn.
    """
    inside = ndimage.distance_transform_edt(brain_mask.astype(bool))
    candidates = inside >= margin
    if not candidates.any():
        raise ValueError("no brain voxel is `margin` voxels from the boundary")
    from_region = ndimage.distance_transform_edt(~region_mask.astype(bool))
    score = np.where(candidates, from_region, -np.inf)
    return tuple(int(i) for i in np.unravel_index(np.argmax(score), score.shape))


# ---------------------------------------------------------------------------
# One volume
# ---------------------------------------------------------------------------

def simulate_volume(vol: np.ndarray,
                    region_mask: np.ndarray,
                    rng: np.random.Generator,
                    volume_loss: float = 0.0,
                    shortcut: bool = False,
                    atrophy_cfg: Optional[AtrophyConfig] = None,
                    shortcut_cfg: Optional[ShortcutConfig] = None,
                    shortcut_center: Optional[Tuple[int, int, int]] = None,
                    ) -> SimulatedVolume:
    """Produce one benchmark volume from a preprocessed healthy scan.

    ``vol`` is expected in the output of ``data/preprocess.py``: registered,
    brain-masked, z-scored, background exactly zero. Every call resamples the
    volume once (jitter + atrophy), regardless of ``volume_loss``.
    """
    acfg = atrophy_cfg or AtrophyConfig()
    scfg = shortcut_cfg or ShortcutConfig()
    shape = vol.shape
    brain = vol != 0

    u_atr = atrophy_field(region_mask, volume_loss, acfg.falloff_sigma)
    u = u_atr + jitter_field(shape, rng, acfg.jitter_amplitude, acfg.jitter_sigma)

    det = jacobian_determinant(u)
    if det.min() <= 0:
        raise ValueError(
            f"deformation folds (min det J = {det.min():.3f}); lower the volume "
            "loss or increase falloff_sigma"
        )

    out = apply_warp(vol.astype(np.float64), u, acfg.interp_order)
    brain_out = apply_warp(brain.astype(np.float64), u, order=0) > 0.5
    out[~brain_out] = 0.0

    # Ground truth is the atrophy deformation alone. The jitter is
    # label-independent noise, so it is deliberately excluded.
    atr_mag = np.abs(jacobian_determinant(u_atr) - 1.0)
    atr_mag[atr_mag < acfg.gt_threshold] = 0.0
    atr_mag[~brain_out] = 0.0

    # Achieved shrink of the region, measured as partial volume of the
    # warped mask. Thresholding it instead would report zero: at ~2 mm
    # voxels, a 25% hippocampal loss moves the boundary by well under a
    # voxel, which no binary mask can resolve.
    region_f = region_mask.astype(np.float64)
    region_out = apply_warp(region_f, u_atr, order=1)
    achieved = 1.0 - region_out.sum() / max(region_f.sum(), 1.0)

    sc_gt = np.zeros(shape, dtype=np.float64)
    if shortcut:
        center = scfg.center or shortcut_center
        if center is None:
            center = default_shortcut_center(region_mask, brain_out)
        # Truncate the Gaussian at the GT threshold (shifted so it stays
        # continuous). An untruncated tail would change every brain voxel by
        # a tiny amount, so signal would exist outside the GT support and a
        # map that found it would be scored as wrong.
        g = shortcut_bump(shape, center, scfg.radius)
        t = scfg.gt_threshold
        bump = np.clip((g - t) / (1.0 - t), 0.0, None) * brain_out
        out = out + scfg.amplitude * bump
        sc_gt = bump

    return SimulatedVolume(
        volume=out.astype(np.float32),
        atrophy_gt=atr_mag.astype(np.float32),
        shortcut_gt=sc_gt.astype(np.float32),
        meta={
            "volume_loss_target": float(volume_loss),
            "volume_loss_achieved": float(achieved),
            "min_jacobian": float(det.min()),
            "shortcut": int(bool(shortcut)),
        },
    )


# ---------------------------------------------------------------------------
# Cohort-level assignment
# ---------------------------------------------------------------------------

def assign_labels(sites: Sequence[str],
                  ages: Sequence[Optional[float]],
                  seed: int = 0) -> np.ndarray:
    """Binary labels balanced within each acquisition site and across age.

    IXI was acquired at three sites on different scanners. If labels were
    drawn at random, an unlucky draw could correlate the label with site and
    hand the model a scanner shortcut nobody planted. Within each site,
    subjects are ordered by age and labels alternate from a random start, so
    each class gets a near-identical age and site distribution. Subjects with
    unknown age are shuffled in at random.
    """
    rng = np.random.default_rng(seed)
    sites = list(sites)
    y = np.zeros(len(sites), dtype=np.int64)
    for site in sorted(set(sites)):
        idx = np.array([i for i, s in enumerate(sites) if s == site])
        idx = idx[rng.permutation(len(idx))]
        key = np.array([ages[i] if ages[i] is not None else rng.uniform(0, 100)
                        for i in idx], dtype=np.float64)
        idx = idx[np.argsort(key, kind="stable")]
        start = int(rng.integers(2))
        y[idx] = (np.arange(len(idx)) + start) % 2
    return y


def assign_shortcuts(labels: Sequence[int],
                     p_given_positive: float,
                     seed: int = 0) -> np.ndarray:
    """Shortcut presence with an exactly controlled label correlation.

    Exactly ``round(p * n1)`` positives and ``round((1 - p) * n0)`` negatives
    carry the shortcut. p = 0.5 makes it uninformative, p = 1.0 makes it a
    perfect predictor. Sampling the counts instead would add noise to the very
    variable the experiment is sweeping.
    """
    if not 0.0 <= p_given_positive <= 1.0:
        raise ValueError("p_given_positive must be in [0, 1]")
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    out = np.zeros(len(labels), dtype=np.int64)
    for cls, p in ((1, p_given_positive), (0, 1.0 - p_given_positive)):
        idx = np.flatnonzero(labels == cls)
        k = int(round(p * len(idx)))
        out[rng.permutation(idx)[:k]] = 1
    return out
