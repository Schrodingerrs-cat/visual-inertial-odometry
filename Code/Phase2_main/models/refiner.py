"""Residual refiner used by NetworkVI_tdirect.

Tiny MLP producing a small (δR, δt) correction to the vision branch's
prediction. Tanh-bounded to ±0.1 rad / ±0.2 m so the refiner cannot dominate
or override V; it is a *correction* mechanism, not a replacement for V.

Extracted from the legacy ``models/network_vi.py`` so the t-direct VI does
not pull in the older (t/d) NetworkV / scale-mixer / NoDecomp / Concat
variants.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from utils.geometry import rodrigues_to_matrix


class _ResidualRefiner(nn.Module):
    """Tiny MLP producing a small (δR, δt) correction.

    Axis-angle δR is tanh-squashed to ±0.1 rad; δt is tanh-squashed to ±0.2 m.
    """

    def __init__(self, vision_dim: int, imu_dim: int, hidden: int = 64,
                 max_d_rot: float = 0.1, max_d_trans: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim + imu_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 6),
        )
        self.max_d_rot = float(max_d_rot)
        self.max_d_trans = float(max_d_trans)

    def forward(self, vfeat: torch.Tensor, ifeat: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([vfeat, ifeat], dim=-1))
        d_aa = torch.tanh(raw[..., :3]) * self.max_d_rot
        d_t = torch.tanh(raw[..., 3:]) * self.max_d_trans
        dR = rodrigues_to_matrix(d_aa)
        return dR, d_t
