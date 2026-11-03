"""Network V (t-direct variant) — vision-only relative pose estimator.

Step H experiment: instead of predicting (R, t/d, n) and reconstructing
``t = (t/d) * d``, predict (R, t) directly. Same backbone (VGG8), 9-dim head
(6 for R as continuous-6D rotation + 3 for t in metres).

Drops ``head_td`` and ``head_n`` entirely. ``n`` is unused; the supervision
uses metric translation directly so the plane normal is no longer part of the
output decomposition.

Cross-reference: ``models/network_v.py`` is the original (t/d) variant. This
file is a sibling — the trainer / loss / eval pick the variant by config flag.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from .backbones import VGG8Backbone
from utils.geometry import rot6d_to_matrix


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NetworkV_tdirect(nn.Module):
    """Vision-only direct-translation regressor (R, t)."""

    def __init__(self, feat_dim: int = 512,
                 cnn_channels: Sequence[int] | None = None) -> None:
        super().__init__()
        bb_kwargs = {"in_channels": 6, "feat_dim": feat_dim}
        if cnn_channels is not None:
            bb_kwargs["channels"] = tuple(cnn_channels)
        self.backbone = VGG8Backbone(**bb_kwargs)
        self.head_rot = _MLPHead(feat_dim, 6)
        self.head_t = _MLPHead(feat_dim, 3)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (R_3x3, t_3, feat). ``t`` is in metres, cam-a frame."""
        x = torch.cat([img_a, img_b], dim=1)   # (B, 6, H, W)
        feat = self.backbone(x)
        rot6 = self.head_rot(feat)
        t = self.head_t(feat)
        R = rot6d_to_matrix(rot6)
        return R, t, feat
