"""
Layer-wise Relevance Propagation for 3D convolutional networks.

Ported from the 2D FER implementation (Bhardwaj & Bhagat, ICDAM 2026) with two
substantive changes:

  1. All rules are dimension-agnostic. The gradient-trick formulation
     ``R_in = a * d/da [ z * (R_out / z).detach() ]`` never references the
     spatial rank of the tensor, so the same code path serves Conv2d and Conv3d.

  2. Residual blocks are handled natively rather than falling back to
     Gradient x Input. A whole ``nn.Module`` is passed to the rule as an opaque
     function; autograd then distributes relevance across the main and skip
     branches in proportion to their contribution to z, which is exactly the
     canonical residual LRP split

         r_main = r_out * z_main / (z_main + z_skip)
         r_skip = r_out * z_skip / (z_main + z_skip)

     This removes the central limitation of the 2D paper, where ResNet50
     "LRP" was actually Gradient x Input.

Composite strategy (Montavon et al. 2019), preserved from the 2D work:

    input conv        -> z^B   (bounded-domain rule, pixel/voxel bounds)
    lower conv blocks -> alpha1beta0  (positive contributions only)
    upper conv blocks -> epsilon      (small stabiliser)
    classifier head   -> epsilon

Relevance is initialised from the raw logit, not softmax and not one-hot, so
the map keeps the model's own output scale.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LRP3D", "Rule"]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

class Rule:
    """Namespace of relevance propagation rules.

    Every rule takes a callable ``fn`` (any nn.Module or lambda), the input
    activation ``a`` that was fed to it during the forward pass, and the
    relevance ``r`` arriving at its output. It returns the relevance
    redistributed onto ``a``.
    """

    EPS_DEFAULT = 1e-6

    @staticmethod
    def epsilon(fn: Callable, a: torch.Tensor, r: torch.Tensor,
                eps: float = EPS_DEFAULT) -> torch.Tensor:
        """LRP-eps. Safe default; handles residual blocks correctly."""
        a = a.detach().requires_grad_(True)
        z = fn(a)
        z = z + eps * torch.sign(z) + eps          # signed stabiliser
        s = (r / z).detach()
        (z * s).sum().backward()
        return (a * a.grad).detach()

    @staticmethod
    def alpha1beta0(fn: Callable, a: torch.Tensor, r: torch.Tensor,
                    eps: float = 1e-9) -> torch.Tensor:
        """LRP-alpha1beta0. Positive contributions only; sharpens the map.

        Implemented via the positive part of the pre-activation, matching the
        2D implementation so results stay comparable across the two papers.
        """
        a = a.detach().requires_grad_(True)
        z = fn(a)
        z_pos = z.clamp(min=0) + eps
        s = (r / z_pos).detach()
        (z_pos * s).sum().backward()
        return (a * a.grad).detach()

    @staticmethod
    def zbeta(conv: nn.Module, a: torch.Tensor, r: torch.Tensor,
              lo: float, hi: float, eps: float = EPS_DEFAULT) -> torch.Tensor:
        """z^B rule for the input layer, with box constraints [lo, hi].

        For MRI the bounds are the min/max of the intensity-normalised volume,
        not the ImageNet-derived per-channel bounds used in the 2D version.
        Supports Conv2d and Conv3d.
        """
        conv_fn = F.conv3d if isinstance(conv, nn.Conv3d) else F.conv2d
        kw = dict(stride=conv.stride, padding=conv.padding,
                  dilation=conv.dilation, groups=conv.groups)

        a = a.detach().requires_grad_(True)
        l = torch.full_like(a, lo).requires_grad_(True)
        h = torch.full_like(a, hi).requires_grad_(True)

        w = conv.weight
        wp, wn = w.clamp(min=0), w.clamp(max=0)

        z = (conv_fn(a, w, bias=conv.bias, **kw)
             - conv_fn(l, wp, bias=None, **kw)
             - conv_fn(h, wn, bias=None, **kw))
        z = z + eps
        s = (r / z).detach()
        (z * s).sum().backward()
        return (a * a.grad - l * l.grad - h * h.grad).detach()

    @staticmethod
    def passthrough(fn: Callable, a: torch.Tensor, r: torch.Tensor,
                    **_) -> torch.Tensor:
        """Identity propagation, for ReLU and Dropout."""
        return r


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

class LRP3D:
    """Composite LRP for a sequentially-decomposable 3D network.

    The model must expose ``lrp_layers() -> List[nn.Module]``: the ordered list
    of callables whose composition equals ``forward``. Residual blocks appear in
    that list as single entries and are propagated with the epsilon rule, which
    performs the correct branch split.

    Parameters
    ----------
    model
        Network in eval mode.
    lower_upper_split
        Fraction of the layer list treated as "lower" (alpha1beta0). Layers
        beyond it use epsilon. Default 0.5 mirrors the 2D configuration, where
        the boundary sat at the end of VGG16 block 3.
    input_bounds
        (lo, hi) for the z^B rule, in the same normalised space as the input.
    device
        Torch device.
    """

    def __init__(self,
                 model: nn.Module,
                 lower_upper_split: float = 0.5,
                 input_bounds: Tuple[float, float] = (-3.0, 3.0),
                 eps: float = Rule.EPS_DEFAULT,
                 device: Optional[torch.device] = None):
        if not hasattr(model, "lrp_layers"):
            raise TypeError(
                "model must implement lrp_layers() returning the ordered list "
                "of modules whose composition equals forward()."
            )
        self.model = model.eval()
        self.layers: List[nn.Module] = list(model.lrp_layers())
        self.eps = eps
        self.lo, self.hi = input_bounds
        self.device = device or next(model.parameters()).device
        self.boundary = int(len(self.layers) * lower_upper_split)

    # -- rule assignment ---------------------------------------------------

    def _rule_for(self, idx: int, layer: nn.Module) -> str:
        if isinstance(layer, (nn.ReLU, nn.Dropout, nn.Dropout3d, nn.Identity)):
            return "passthrough"
        if idx == 0 and isinstance(layer, (nn.Conv2d, nn.Conv3d)):
            return "zbeta"
        if idx <= self.boundary:
            return "alpha1beta0"
        return "epsilon"

    def _apply(self, rule: str, layer: nn.Module,
               a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if rule == "passthrough":
            return Rule.passthrough(layer, a, r)
        if rule == "zbeta":
            return Rule.zbeta(layer, a, r, self.lo, self.hi, self.eps)
        if rule == "alpha1beta0":
            return Rule.alpha1beta0(layer, a, r)
        return Rule.epsilon(layer, a, r, self.eps)

    # -- main call ---------------------------------------------------------

    def __call__(self,
                 x: torch.Tensor,
                 class_idx: Optional[int] = None,
                 return_signed: bool = False) -> Tuple[np.ndarray, int]:
        """Attribute a single volume.

        Parameters
        ----------
        x
            Input of shape (1, C, D, H, W) — or (1, C, H, W) in 2D.
        class_idx
            Target class. Defaults to the model's own prediction, which is the
            right choice for faithfulness evaluation: we are explaining what
            the model did, not what it should have done.
        return_signed
            If True, keep negative relevance instead of clamping at zero.

        Returns
        -------
        (relevance, class_idx)
            relevance: numpy array of shape (D, H, W), channel-summed and
            max-normalised to [0, 1] (or [-1, 1] when ``return_signed``).
        """
        x = x.to(self.device)

        # Forward pass, caching the input to each layer.
        acts: List[torch.Tensor] = []
        a = x
        with torch.no_grad():
            for layer in self.layers:
                acts.append(a.detach())
                a = layer(a)
        logits = a

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                f"expected logits of shape (1, n_classes), got {tuple(logits.shape)}"
            )

        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        # Relevance init: the raw logit, preserving output scale.
        r = torch.zeros_like(logits)
        r[0, class_idx] = logits[0, class_idx]

        # Backward pass through the layer list.
        for idx in reversed(range(len(self.layers))):
            layer, a_in = self.layers[idx], acts[idx]
            r = self._apply(self._rule_for(idx, layer), layer, a_in, r)

        R = r.squeeze(0).sum(0).detach().cpu().numpy()
        if not return_signed:
            R = np.maximum(R, 0.0)
        denom = np.abs(R).max()
        if denom > 1e-12:
            R = R / denom
        return R.astype(np.float32), class_idx

    # -- diagnostics -------------------------------------------------------

    def conservation_error(self, x: torch.Tensor,
                           class_idx: Optional[int] = None) -> float:
        """Relative deviation from the LRP conservation property.

        Sum of input relevance should approximately equal the initialising
        logit. Large values mean a rule in the chain is leaking relevance —
        worth reporting in the paper as an implementation-correctness check,
        since almost no applied XAI paper does.
        """
        R, cls = self(x, class_idx, return_signed=True)
        with torch.no_grad():
            logit = self.model(x.to(self.device))[0, cls].item()
        if abs(logit) < 1e-12:
            return float("nan")
        # R is max-normalised, so recover scale from the unnormalised pass.
        return float(abs(R.sum()) / abs(logit))
