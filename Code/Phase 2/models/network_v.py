"""Network V — vision-only relative pose estimator.

Input:  stacked RGB frame pair as (B, 6, H, W).
Output: decomposed-homography parameters
            R   ∈ SO(3)   (via 6D continuous rep → 3x3)
            td  ∈ R^3     (t / d, scaled translation)
            n   ∈ S^2     (plane normal in camera 1)

The final pose predicted by the network is (R, t, d) with t = d * td. At
training time we supervise R, td, and n directly; at eval time callers can
combine td with a learned scale (Network VI) or with the known altitude (ground
truth, for debugging).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
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


class NetworkV(nn.Module):
    """Vision-only decomposed-homography regressor."""

    def __init__(self, feat_dim: int = 512,
                 normal_init_bias: Sequence[float] = (0.0, 0.0, 1.0),
                 cnn_channels: Sequence[int] | None = None) -> None:
        super().__init__()
        bb_kwargs = {"in_channels": 6, "feat_dim": feat_dim}
        if cnn_channels is not None:
            bb_kwargs["channels"] = tuple(cnn_channels)
        self.backbone = VGG8Backbone(**bb_kwargs)
        self.head_rot = _MLPHead(feat_dim, 6)
        self.head_td = _MLPHead(feat_dim, 3)
        self.head_n = _MLPHead(feat_dim, 3)
        with torch.no_grad():
            # Bias the plane-normal head so the *pre-normalisation* output
            # starts pointed in the prior direction. The outermost linear is
            # inside `_MLPHead.net[-1]`.
            self.head_n.net[-1].bias.copy_(
                torch.tensor(normal_init_bias, dtype=torch.float32))

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (R_3x3, td_3, n_3, feat) where feat is the 512-d CNN feature."""
        x = torch.cat([img_a, img_b], dim=1)   # (B, 6, H, W)
        feat = self.backbone(x)
        rot6 = self.head_rot(feat)
        td = self.head_td(feat)
        n_raw = self.head_n(feat)
        n = F.normalize(n_raw, dim=-1, eps=1e-6)
        R = rot6d_to_matrix(rot6)
        return R, td, n, feat
