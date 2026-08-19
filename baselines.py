"""
Baseline attribution methods, for comparison against LRP and Grad-CAM.

Included so the comparison is genuinely competitive rather than a straw man:

  IntegratedGradients3D
      Satisfies completeness by construction. Standard, strong, and the
      obvious thing a reviewer will ask why you omitted.

  Occlusion3D
      Model-agnostic and directly causal — it perturbs and measures. Slow, but
      the closest thing to a ground truth for "what did the network use", which
      makes it a useful reference point for the faithfulness metrics.

  GradientInput3D
      What the 2D paper used as an LRP proxy for ResNet50. Kept as a labelled
      baseline so the new paper can quantify what that substitution cost —
      turning a limitation of the previous work into a measured result.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = ["IntegratedGradients3D", "Occlusion3D", "GradientInput3D"]


def _normalise(r: np.ndarray, signed: bool = False) -> np.ndarray:
    if not signed:
        r = np.maximum(r, 0.0)
    m = np.abs(r).max()
    return (r / m).astype(np.float32) if m > 1e-12 else r.astype(np.float32)


class GradientInput3D:
    """Gradient x Input. The 2D paper's ResNet fallback, kept as a baseline."""

    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def __call__(self, x: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        device = next(self.model.parameters()).device
        x = x.to(device).detach().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        r = (x.grad * x.detach()).squeeze(0).sum(0).cpu().numpy()
        return _normalise(r), class_idx


class IntegratedGradients3D:
    """Integrated Gradients (Sundararajan et al. 2017).

    The baseline choice matters and should be stated in the paper. A zero
    volume is not neutral in z-scored MRI space; a blurred volume or the
    cohort mean template is the defensible option.
    """

    def __init__(self, model: nn.Module, n_steps: int = 32,
                 batch_size: int = 8):
        self.model = model.eval()
        self.n_steps = n_steps
        self.batch_size = batch_size

    def __call__(self, x: torch.Tensor,
                 class_idx: Optional[int] = None,
                 baseline: Optional[torch.Tensor] = None
                 ) -> Tuple[np.ndarray, int]:
        device = next(self.model.parameters()).device
        x = x.to(device)
        if baseline is None:
            from eval.faithfulness import blur_baseline
            baseline = blur_baseline(x, sigma=5.0)
        baseline = baseline.to(device)

        with torch.no_grad():
            if class_idx is None:
                class_idx = int(self.model(x).argmax(1).item())

        alphas = torch.linspace(1.0 / self.n_steps, 1.0, self.n_steps,
                                device=device)
        total = torch.zeros_like(x)

        for i in range(0, self.n_steps, self.batch_size):
            chunk = alphas[i:i + self.batch_size]
            pts = torch.cat([baseline + a * (x - baseline) for a in chunk], 0)
            pts.requires_grad_(True)
            out = self.model(pts)
            self.model.zero_grad(set_to_none=True)
            out[:, class_idx].sum().backward()
            total = total + pts.grad.sum(0, keepdim=True)

        ig = (x - baseline) * total / self.n_steps
        return _normalise(ig.squeeze(0).sum(0).detach().cpu().numpy()), class_idx


class Occlusion3D:
    """Sliding-window occlusion sensitivity.

    Expensive: a 96^3 volume with a 16^3 window at stride 8 is ~1,300 forward
    passes. Run it on a subsample of the test set rather than all of it.
    """

    def __init__(self, model: nn.Module, window: int = 16, stride: int = 8,
                 batch_size: int = 16):
        self.model = model.eval()
        self.window = window
        self.stride = stride
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, x: torch.Tensor,
                 class_idx: Optional[int] = None,
                 fill: Optional[float] = 0.0) -> Tuple[np.ndarray, int]:
        device = next(self.model.parameters()).device
        x = x.to(device)
        D, H, W = x.shape[2:]

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        base_p = torch.softmax(logits, 1)[0, class_idx].item()

        heat = np.zeros((D, H, W), dtype=np.float64)
        count = np.zeros((D, H, W), dtype=np.float64)

        coords, batch = [], []
        starts = [(d, h, w)
                  for d in range(0, D - self.window + 1, self.stride)
                  for h in range(0, H - self.window + 1, self.stride)
                  for w in range(0, W - self.window + 1, self.stride)]

        for idx, (d, h, w) in enumerate(starts):
            occ = x.clone()
            occ[:, :, d:d + self.window, h:h + self.window,
                w:w + self.window] = fill
            batch.append(occ)
            coords.append((d, h, w))

            if len(batch) == self.batch_size or idx == len(starts) - 1:
                p = torch.softmax(self.model(torch.cat(batch, 0)), 1)[:, class_idx]
                for (dd, hh, ww), prob in zip(coords, p.cpu().numpy()):
                    sl = (slice(dd, dd + self.window),
                          slice(hh, hh + self.window),
                          slice(ww, ww + self.window))
                    heat[sl] += base_p - float(prob)
                    count[sl] += 1
                batch, coords = [], []

        heat = np.divide(heat, np.maximum(count, 1))
        return _normalise(heat), class_idx
