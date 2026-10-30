"""
3D ResNet backbones for volumetric MRI classification.

The distinguishing requirement here is ``lrp_layers()``: the ordered list of
callables whose composition equals ``forward``. The LRP propagator walks that
list backwards, and because each residual block is a single entry, the epsilon
rule performs the branch split internally instead of requiring hand-written
skip-connection bookkeeping.

Pretrained weights: for real experiments, initialise from MedicalNet
(Chen et al. 2019) or Models Genesis rather than training from scratch. ADNI's
usable sample size is in the low thousands of subjects, which is not enough to
train a 3D ResNet from random init without severe overfitting.
"""

from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn

__all__ = ["ResNet3D", "resnet3d_10", "resnet3d_18", "BasicBlock3D",
           "fold_batchnorm"]


def conv3x3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock3D(nn.Module):
    """Standard two-conv residual block.

    ``inplace=False`` throughout: in-place ReLU corrupts the cached activations
    that LRP replays, and the resulting maps are silently wrong rather than
    erroring. The 2D code had the same guard for the same reason.
    """
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv3x3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, 1)


class ResNet3D(nn.Module):
    """3D ResNet for binary or multi-class diagnostic classification.

    Parameters
    ----------
    block, layers
        Residual block type and per-stage depth.
    n_classes
        2 for CN vs AD; 3 if MCI is a separate class.
    in_channels
        1 for a single T1-weighted volume.
    widen
        Channel multiplier. The default halves standard ResNet widths, which
        is appropriate for the sample sizes available in ADNI.
    """

    def __init__(self,
                 block: Type[nn.Module] = BasicBlock3D,
                 layers: List[int] = (1, 1, 1, 1),
                 n_classes: int = 2,
                 in_channels: int = 1,
                 widen: float = 0.5,
                 dropout: float = 0.4):
        super().__init__()
        w = [max(8, int(c * widen)) for c in (64, 128, 256, 512)]
        self.in_ch = w[0]

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, w[0], kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm3d(w[0]),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(block, w[0], layers[0], 1)
        self.layer2 = self._make_stage(block, w[1], layers[1], 2)
        self.layer3 = self._make_stage(block, w[2], layers[2], 2)
        self.layer4 = self._make_stage(block, w[3], layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.flatten = _Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(w[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, block, out_ch: int, blocks: int,
                    stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_ch, out_ch * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch * block.expansion),
            )
        stage = [block(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch * block.expansion
        stage += [block(self.in_ch, out_ch) for _ in range(1, blocks)]
        return nn.Sequential(*stage)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.flatten(self.avgpool(x))
        return self.fc(self.dropout(x))

    # -- LRP support -------------------------------------------------------

    def lrp_layers(self, fold_bn: bool = True) -> List[nn.Module]:
        """Ordered decomposition of forward() for relevance propagation.

        Stem convolution is listed separately so the propagator can apply the
        z^B bounded-domain rule to it. Residual blocks appear whole.

        With ``fold_bn`` (the default) the layers come from a BN-folded copy
        made at call time, so every conv carries its BatchNorm as weight and
        bias and the BN slot in the stem becomes an Identity (passthrough).
        Folding at call time rather than caching means a model that is
        trained or randomised afterwards is never explained with stale
        weights; the cascading-randomisation check relies on that.
        """
        src = fold_batchnorm(self) if fold_bn else self
        layers: List[nn.Module] = list(src.stem.children())
        for stage in (src.layer1, src.layer2, src.layer3, src.layer4):
            layers.extend(stage.children())
        layers += [src.avgpool, src.flatten, src.fc]
        return layers

    def target_layer(self) -> nn.Module:
        """Feature layer Grad-CAM hooks into."""
        return self.layer4


def _fold_conv_bn(conv: nn.Conv3d, bn: nn.BatchNorm3d) -> nn.Conv3d:
    """Conv followed by eval-mode BN, as a single conv with bias.

        BN(conv(x)) = gamma * (W*x + b - mu) / sqrt(var + eps) + beta
                    = (s * W) * x + (beta + s * (b - mu)),   s = gamma / sqrt(var + eps)
    """
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv3d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      stride=conv.stride, padding=conv.padding,
                      dilation=conv.dilation, groups=conv.groups, bias=True)
    fused = fused.to(device=conv.weight.device, dtype=conv.weight.dtype)
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * s.view(-1, *([1] * (conv.weight.ndim - 1))))
        fused.bias.copy_(bn.bias + s * (b - bn.running_mean))
    return fused


def fold_batchnorm(model: ResNet3D) -> ResNet3D:
    """LRP-ready copy of ``model`` with every BatchNorm absorbed into the
    convolution before it (Montavon et al. 2019).

    Propagating relevance through BN as a separate layer treats its
    mean-shift as an input-dependent contribution and distorts the map; after
    folding, the shift lives in a bias, which LRP handles as an ordinary
    relevance sink. The original model is not modified.

    Only valid in eval mode: the fold uses running statistics, so a copy made
    from a model in train mode would not reproduce the forward pass the
    model actually computes there. The copy is always returned in eval mode.
    """
    import copy

    folded = copy.deepcopy(model).eval()

    stem = list(folded.stem.children())
    folded.stem = nn.Sequential(_fold_conv_bn(stem[0], stem[1]),
                                nn.Identity(), *stem[2:])

    for stage in (folded.layer1, folded.layer2, folded.layer3, folded.layer4):
        for block in stage:
            block.conv1 = _fold_conv_bn(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = _fold_conv_bn(block.conv2, block.bn2)
            block.bn2 = nn.Identity()
            if block.downsample is not None:
                conv, bn = block.downsample[0], block.downsample[1]
                block.downsample = nn.Sequential(_fold_conv_bn(conv, bn))

    leftover = [m for m in folded.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if leftover:
        raise RuntimeError(
            f"fold_batchnorm left {len(leftover)} BatchNorm layer(s) unfolded; "
            "the block layout changed and this function needs updating"
        )
    return folded.eval()


def resnet3d_10(**kw) -> ResNet3D:
    return ResNet3D(BasicBlock3D, [1, 1, 1, 1], **kw)


def resnet3d_18(**kw) -> ResNet3D:
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], **kw)
