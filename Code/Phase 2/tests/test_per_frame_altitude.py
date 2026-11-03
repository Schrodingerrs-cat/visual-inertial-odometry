"""S5 fix test — per-frame altitude vs. altitude_initial in GT (t/d).

Background: the dataloader's ``depth`` field is used by
``DecomposedHomographyLoss`` to compute ``td_gt = t_gt / depth``. Before the
2026-04-24 fix the dataloader returned ``metadata["altitude_initial"]`` — a
single scalar per trajectory. When the drone's altitude varies, this
produces a systematic bias in the supervision target.

This test constructs a synthetic two-frame trajectory where frame_a sits
at altitude 1.6 m while ``altitude_initial`` is set to 1.7 m. |t_world| is
0.03 m. Correct GT (t/d) = 0.03 / 1.6 ≈ 0.01875; the pre-fix code
would have produced 0.03 / 1.7 ≈ 0.01765.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from utils.dataloader import FramePairDataset  # noqa: E402


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def _synth_traj_with_altitude_drift(root: Path, traj_id: int = 777,
                                    z_a: float = 1.6, z_initial: float = 1.7,
                                    dx: float = 0.03) -> Path:
    """Build a 2-kept-frame trajectory:
       frame_a is at world-frame altitude ``z_a`` and x=0;
       frame_b is at world-frame altitude ``z_a`` (level) and x=dx.
       metadata.altitude_initial is set to ``z_initial``, which is DIFFERENT
       from z_a — to surface the bug the S5 fix removes.
    """
    traj_dir = root / f"traj_{traj_id:04d}"
    (traj_dir / "frames").mkdir(parents=True, exist_ok=True)
    n_kept = 2
    n_samples = (n_kept - 1) * 10 + 1

    poses = np.zeros((n_samples, 7))
    # Sample 0 (frame_a pose index = 0) at altitude z_a, x=0.
    # Sample 10 (frame_b pose index = 10) at altitude z_a, x=dx.
    poses[:, 2] = z_a
    poses[:, 3:] = _identity_quat()
    for i in range(n_samples):
        poses[i, 0] = dx * (i / 10.0)
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
        "plane_sizing_policy": "stretched_per_image_v1",
        "altitude_initial": z_initial,       # DELIBERATELY != z_a (bug surface).
        "altitude_m": z_initial,
        "texture_pool": "train",
        "texture_path": "Data/textures_train/tex_00.webp",
        "texture_name": "tex_00.webp",
        "texture_original_filename": "synthetic.jpg",
        "confinement_scale": 1.0,
        "trajectory_params": {},
        "noise_params": {"sigma_acc": 0.0, "sigma_gyro": 0.0},
        "gravity_world": [0.0, 0.0, -9.81],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))

    for k in range(n_kept):
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        arr[:, :, 0] = (k * 20) % 255
        Image.fromarray(arr).save(traj_dir / "frames" / f"{k:05d}.png")
    return root


def test_depth_is_per_frame_not_scalar(tmp_path: Path) -> None:
    """The S5 fix: sample['depth'] equals frame_a's world-frame z, NOT
    metadata['altitude_initial']."""
    _synth_traj_with_altitude_drift(tmp_path, z_a=1.6, z_initial=1.7, dx=0.03)
    ds = FramePairDataset(raw_dir=tmp_path, traj_ids=[777],
                          img_size=(480, 640), imu_window=10)
    sample = ds[0]

    # Per-frame altitude = z_a = 1.6.
    assert float(sample["depth"].item()) == pytest.approx(1.6, abs=1e-6)

    # Compute GT (t/d) and verify it matches 0.03 / 1.6.
    td_gt_correct = sample["t_rel"] / sample["depth"]
    assert float(td_gt_correct[0].item()) == pytest.approx(0.03 / 1.6, abs=1e-4)

    # The pre-fix code would have used z_initial = 1.7, producing a larger
    # denominator and a smaller td_gt. Confirm the two are distinguishable at
    # the precision the loss operates at.
    td_gt_old = sample["t_rel"] / torch.tensor(1.7, dtype=torch.float32)
    assert float(td_gt_old[0].item()) == pytest.approx(0.03 / 1.7, abs=1e-4)
    bias_frac = abs((0.03 / 1.6) - (0.03 / 1.7)) / (0.03 / 1.6)
    assert bias_frac > 0.05, (
        "synth case designed to show > 5% systematic bias between per-frame "
        f"and scalar altitude — measured {bias_frac*100:.2f}%.")


def test_depth_matches_pose_z_frame_index(tmp_path: Path) -> None:
    """For a multi-frame synth trajectory where altitude varies linearly,
    each sample's depth equals frame_a's z, not a global constant."""
    # Build a 4-kept-frame trajectory with altitude drifting from 1.5 → 3.0 m.
    traj_id = 778
    root = tmp_path
    traj_dir = root / f"traj_{traj_id:04d}"
    (traj_dir / "frames").mkdir(parents=True, exist_ok=True)
    n_kept = 4
    n_samples = (n_kept - 1) * 10 + 1

    poses = np.zeros((n_samples, 7))
    poses[:, 2] = np.linspace(1.5, 3.0, n_samples)
    poses[:, 3:] = _identity_quat()
    np.save(traj_dir / "poses_full.npy", poses)
    np.save(traj_dir / "imu.npy", np.zeros((n_samples, 6)))

    metadata = {
        "trajectory_id": traj_id,
        "n_samples": int(n_samples), "n_frames": int(n_kept),
        "fps_imu": 1000, "fps_image": 100,
        "K": [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]],
        "image_size": [480, 640],
        "altitude_initial": 1.5, "altitude_m": 1.5,
        "plane_dims_meters": [18.0, 12.0],
        "plane_sizing_policy": "stretched_per_image_v1",
        "texture_pool": "train",
        "texture_path": "Data/textures_train/tex_00.webp",
        "texture_name": "tex_00.webp",
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    for k in range(n_kept):
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        Image.fromarray(arr).save(traj_dir / "frames" / f"{k:05d}.png")

    ds = FramePairDataset(raw_dir=root, traj_ids=[traj_id],
                          img_size=(480, 640), imu_window=10)
    for idx in range(len(ds)):
        sample = ds[idx]
        # frame_a pose row index = idx * 10 (since imu_window = 10)
        expected = float(poses[idx * 10, 2])
        assert float(sample["depth"].item()) == pytest.approx(expected, abs=1e-6), (
            f"sample {idx}: depth {sample['depth'].item()} != expected "
            f"per-frame z {expected}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
