"""Network I — IMU-only relative pose estimator (Step S2 architecture).

Input:  IMU window (B, T, 6) where channels are [ax, ay, az, gx, gy, gz] in
        the body frame with gravity preserved (no pre-subtraction).
Output: rotation matrix R (B, 3, 3), metric translation t (B, 3), and the
        encoder feature ifeat (B, encoder.out_dim).

Step S2 (2026-05-03) swaps the prior unidirectional / single-MLP-head design
for the architecture that actually trains well — sourced from
`models/imu_model.py:IMUModel`:

    * Bidirectional LSTM (2 layers, hidden=128 per direction)
    * All layers from both directions concatenated (512-d) and projected to
      a 256-d feature with ReLU + dropout 0.1
    * Two separate heads — rotation (6-D continuous, Zhou et al. 2019) and
      translation (3-D)

Public API (kept stable so train/eval scripts and VI variants don't break):

    NetworkI(hidden_dim=128)         # constructor signature unchanged
    forward(imu) -> (R, t, ifeat)    # tuple shape unchanged

`ifeat` is now `encoder.out_dim`-d (default 256) instead of `hidden_dim`-d
(was 128). `NetworkI.out_dim` is exposed so VI variants can size their
fusion modules correctly without hardcoding.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .backbones import IMUEncoder
from utils.geometry import rot6d_to_matrix


class NetworkI(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2,
                 feat_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = IMUEncoder(
            input_size=6,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            feat_dim=feat_dim,
            dropout=dropout,
        )
        self.out_dim = self.encoder.out_dim
        self.head_rot = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
        self.head_t = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, imu: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (R_3x3, t_3, feat) where feat is the encoder projection."""
        feat = self.encoder(imu)              # (B, out_dim)
        rot6 = self.head_rot(feat)            # (B, 6)
        t    = self.head_t(feat)              # (B, 3)
        R    = rot6d_to_matrix(rot6)          # (B, 3, 3)
        return R, t, feat
