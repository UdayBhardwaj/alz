"""
Sanity checks for saliency methods (Adebayo et al., NeurIPS 2018).

Faithfulness metrics answer "does the map track the model's decision?".
Sanity checks answer a prior question: "does the map depend on the model at
all?" Some widely used methods — Guided Backprop most notoriously — produce
near-identical, anatomically plausible-looking maps after the network's weights
have been randomised. Such a method is an edge detector wearing a lab coat.

No AD-XAI paper found in the 2019-2026 literature reports these tests. Running
them is a low-cost, high-credibility contribution: it is the difference between
asserting that explanations are trustworthy and demonstrating it.

Two tests are implemented:

  cascading_randomization
      Randomise layers from the output backwards, one stage at a time,
      recomputing the map after each. A method that passes shows monotonically
      decaying similarity to the original map.

  data_randomization
      Compare maps from a model trained on true labels against one trained on
      permuted labels. Requires two trained checkpoints, so it is run once at
      the end rather than per-volume.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

__all__ = ["ssim_3d", "rank_correlation", "cascading_randomization",
           "data_randomization"]


def rank_correlation(a: np.ndarray, b: np.ndarray,
                     abs_value: bool = True) -> float:
    """Spearman rank correlation between two attribution maps.

    Rank-based rather than Pearson because attribution scales differ wildly
    across methods and only the ordering feeds the faithfulness metrics.
    """
    x, y = a.ravel(), b.ravel()
    if abs_value:
        x, y = np.abs(x), np.abs(y)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def ssim_3d(a: np.ndarray, b: np.ndarray,
            data_range: Optional[float] = None) -> float:
    """Global SSIM between two volumes. Complements rank correlation by
    being sensitive to spatial structure rather than ordering alone."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    if data_range is None:
        data_range = max(a.max() - a.min(), b.max() - b.min(), 1e-12)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / den) if den > 1e-12 else float("nan")


def _randomize_(module: nn.Module, generator: torch.Generator) -> None:
    """Reinitialise every parameter in a module, in place."""
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, mean=1.0, std=0.1, generator=generator)
            nn.init.zeros_(m.bias)


def cascading_randomization(model: nn.Module,
                            x: torch.Tensor,
                            explainer_factory: Callable[[nn.Module], Callable],
                            stage_names: Optional[List[str]] = None,
                            class_idx: Optional[int] = None,
                            seed: int = 0) -> Dict[str, float]:
    """Similarity of the explanation to its original after progressive
    randomisation of the network, top stage first.

    Parameters
    ----------
    model
        Trained network.
    x
        A single input volume, shape (1, C, D, H, W).
    explainer_factory
        Callable taking a model and returning an explainer with signature
        ``(x, class_idx) -> (map, class_idx)``. A factory is required rather
        than an explainer instance because Grad-CAM registers hooks on a
        specific module and must be rebuilt for each randomised copy.
    stage_names
        Attribute names to randomise, in output-to-input order. Defaults to
        the ResNet3D stage layout.
    seed
        Controls reinitialisation, so the test is reproducible.

    Returns
    -------
    Mapping from stage name to Spearman correlation with the original map.
    Values that stay near 1.0 indicate the method is insensitive to the
    model's learned parameters — a failed check.
    """
    if stage_names is None:
        stage_names = ["fc", "layer4", "layer3", "layer2", "layer1", "stem"]

    generator = torch.Generator().manual_seed(seed)

    base_map, cls = explainer_factory(model)(x, class_idx)
    results: Dict[str, float] = {}

    work = copy.deepcopy(model).eval()
    for name in stage_names:
        stage = getattr(work, name, None)
        if stage is None:
            continue
        _randomize_(stage, generator)
        m, _ = explainer_factory(work)(x, cls)
        results[name] = rank_correlation(base_map, m)
    return results


def data_randomization(trained_model: nn.Module,
                       permuted_label_model: nn.Module,
                       x: torch.Tensor,
                       explainer_factory: Callable[[nn.Module], Callable],
                       class_idx: Optional[int] = None) -> Tuple[float, float]:
    """Compare explanations from a properly trained model against one trained
    on randomly permuted labels.

    Returns (spearman, ssim). Both near zero is the pass condition. High
    similarity means the map reflects input statistics — brain edges, skull
    boundary, ventricle contrast — rather than anything the model learned
    about disease.
    """
    a, cls = explainer_factory(trained_model)(x, class_idx)
    b, _ = explainer_factory(permuted_label_model)(x, cls)
    return rank_correlation(a, b), ssim_3d(a, b)
