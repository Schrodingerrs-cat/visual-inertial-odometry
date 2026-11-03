"""IMU-only relative pose estimator.

Input:  IMU sequence  (B, T, 6)  — [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
Output: predicted relative pose R ∈ SO(3) and t ∈ ℝ³.

Architecture
------------
Bidirectional LSTM (2 layers, hidden_size=128) processes the T-step IMU window.
The last hidden states of both directions are concatenated → 512-d vector.
Two parallel MLP heads predict the 6D rotation and the 3D translation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from utils.geometry import rot6d_to_matrix


class IMUEncoder(nn.Module):
    """Bidirectional LSTM encoder for an IMU sequence.

    Parameters
    ----------
    input_size  : 6 (acc + gyro)
    hidden_size : LSTM hidden units per direction
    num_layers  : number of stacked LSTM layers
    feat_dim    : output projection dimension
    dropout     : dropout between LSTM layers (0 = disabled)
    """

    def __init__(self,
                 input_size: int = 6,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 feat_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2 * num_layers  # all layers, both directions
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        imu : (B, T, 6)
        Returns
        -------
        feat : (B, feat_dim)
        """
        _, (h_n, _) = self.lstm(imu)
        # h_n : (num_layers * 2, B, hidden_size) — fwd and bwd layers interleaved
        B = imu.shape[0]
        h = h_n.permute(1, 0, 2).reshape(B, -1)   # (B, num_layers*2*hidden)
        return self.proj(h)                          # (B, feat_dim)


class IMUModel(nn.Module):
    """IMU-only relative pose estimator.

    forward(imu) → (R, t, feat)
        R    : (B, 3, 3) rotation matrices
        t    : (B, 3)    translation vectors
        feat : (B, feat_dim) encoder features (used by FusionModel)
    """

    def __init__(self,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 feat_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = IMUEncoder(input_size=6,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   feat_dim=feat_dim,
                                   dropout=dropout)
        self.head_rot = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
        self.head_t = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(imu)            # (B, feat_dim)
        rot6 = self.head_rot(feat)          # (B, 6)
        t    = self.head_t(feat)            # (B, 3)
        R    = rot6d_to_matrix(rot6)        # (B, 3, 3)
        return R, t, feat
