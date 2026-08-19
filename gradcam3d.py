"""
Grad-CAM and Grad-CAM++ for 3D convolutional networks.

Ported from the 2D implementation. Two changes:
  - channel weights pool over (D, H, W) instead of (H, W);
  - upsampling uses trilinear interpolation rather than ``cv2.resize``.

A note that belongs in the paper's discussion: 3D Grad-CAM is coarser than it
looks. After a stride-2 stem, a stride-2 pool and three stride-2 stages, the
layer4 feature map of a 128^3 volume is 4^3. Every relevance value is therefore
smeared across a 32^3-voxel cube on upsampling. Structures the size of the
hippocampus occupy only a few such cubes, so apparent anatomical precision in a
Grad-CAM figure is largely an artefact of interpolation. This is a substantive
reason to expect LRP to win on faithfulness in 3D even where Grad-CAM produces
the more convincing-looking picture.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GradCAM3D", "GradCAMPlusPlus3D"]


class GradCAM3D:
    """Gradient-weighted class activation mapping for volumetric inputs."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model.eval()
        if target_layer is None:
            if not hasattr(model, "target_layer"):
                raise TypeError(
                    "pass target_layer explicitly, or implement "
                    "target_layer() on the model"
                )
            target_layer = model.target_layer()
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_act),
            target_layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, _m, _i, out):
        self._act = out.clone()

    def _save_grad(self, _m, _gi, gout):
        self._grad = gout[0].clone().detach()

    def remove(self) -> None:
        """Detach hooks. Call when done, or the model leaks memory."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def _weights(self) -> torch.Tensor:
        return self._grad.mean(dim=(2, 3, 4), keepdim=True)

    def __call__(self, x: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        device = next(self.model.parameters()).device
        x = x.to(device).requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        cam = F.relu((self._weights() * self._act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tuple(x.shape[2:]),
                            mode="trilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 1e-12:
            cam = cam / cam.max()
        return cam.astype(np.float32), class_idx


class GradCAMPlusPlus3D(GradCAM3D):
    """Grad-CAM++ weighting, which handles multiple disjoint evidence regions.

    Relevant for AD, where atrophy is bilateral: plain Grad-CAM tends to
    collapse onto one hemisphere when both contribute.
    """

    def _weights(self) -> torch.Tensor:
        g = self._grad
        g2, g3 = g.pow(2), g.pow(3)
        act_sum = self._act.sum(dim=(2, 3, 4), keepdim=True)
        denom = 2.0 * g2 + act_sum * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        return (alpha * F.relu(g)).sum(dim=(2, 3, 4), keepdim=True)
