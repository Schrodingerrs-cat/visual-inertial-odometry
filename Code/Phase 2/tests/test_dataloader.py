"""Unit tests for FramePairDataset.

The dataset depends on rendered data on disk. To keep the test self-contained
and runnable without having generated a trajectory, we synthesise a minimal
`traj_999` directory under a temp path and point FramePairDataset at it.

Sampler behavior is exercised by tests/test_sampling.py.

Run with:
    pytest tests/test_dataloader.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataloader import FramePairDataset  # noqa: E402


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def _synth_trajectory(tmp_path: Path, traj_id: int = 999,
                      n_kept: int = 12) -> Path:
    """Write a dummy traj with `n_kept` rendered frames (step = 10)."""
    traj_dir = tmp_path / f"traj_{traj_id:04d}"
    (traj_dir / "frames").mkdir(parents=True, exist_ok=True)
    n_samples = (n_kept - 1) * 10 + 1

    poses = np.zeros((n_samples, 7))
    poses[:, 0] = np.linspace(0.0, 0.05 * (n_samples - 1), n_samples)
    poses[:, 2] = 2.0                  # altitude (matches sampler mid-range)
    poses[:, 3:] = _identity_quat()
    np.save(traj_dir / "poses_full.npy", poses)

    imu = np.zeros((n_samples, 6))
    imu[:, 2] = 9.81
    np.save(traj_dir / "imu.npy", imu)

    metadata = {
        "trajectory_id": traj_id, "seed": 0,
        "n_samples": int(n_samples), "n_frames": int(n_kept),
        "fps_imu": 1000, "fps_image": 100,
        "K": [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]],
        "image_size": [480, 640],
        "plane_dims_meters": [18.0, 12.0],
        "plane_aspect_ratio": 18.0 / 12.0,
        "plane_sizing_policy": "stretched_per_image_v1",
        "altitude_initial": 2.0,
        "altitude_m": 2.0,
        "texture_pool": "train",
        "texture_path": "Data/textures_train/tex_00.webp",
        "texture_name": "tex_00.webp",
        "texture_original_filename": "synthetic.jpg",
        "confinement_scale": 1.0,
        "trajectory_seed": 1,
        "trajectory_params": {},
        "noise_params": {"sigma_acc": 0.0, "sigma_gyro": 0.0},
        "gravity_world": [0.0, 0.0, -9.81],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))

    for k in range(n_kept):
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        arr[:, :, 0] = (k * 20) % 255
        Image.fromarray(arr).save(traj_dir / "frames" / f"{k:05d}.png")
    return tmp_path


def test_dataset_shapes(tmp_path: Path) -> None:
    _synth_trajectory(tmp_path, traj_id=999, n_kept=10)
    ds = FramePairDataset(raw_dir=tmp_path, traj_ids=[999],
                          img_size=(480, 640), imu_window=10)
    assert len(ds) == 9  # 10 frames → 9 pairs
    sample = ds[0]
    assert sample["img_a"].shape == (3, 480, 640)
    assert sample["img_b"].shape == (3, 480, 640)
    assert sample["imu"].shape == (10, 6)
    assert sample["R_rel"].shape == (3, 3)
    assert sample["t_rel"].shape == (3,)
    assert sample["normal"].shape == (3,)
    assert sample["depth"].dtype == torch.float32
    # S5 fix: depth equals frame_a's per-frame world-frame z, not the scalar
    # metadata altitude_initial. For this synth traj both are 2.0.
    assert float(sample["depth"].item()) == pytest.approx(2.0, abs=1e-6)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
