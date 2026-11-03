"""Unit tests for trajectory_library and attitude coupling.

Run:
    pytest tests/test_trajectories.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "blender"))
sys.path.insert(0, str(HERE))

from trajectory_library import (  # noqa: E402
    DT, GRAVITY, MINITEST_N_SAMPLES, N_SAMPLES,
    attitude_from_acceleration,
    clover_trajectory,
    figure_8_trajectory,
    flat_spiral_trajectory,
    minitest_trajectory,
    sinusoid_trajectory,
    validate_trajectory,
)


# Use a generously sized plane so confinement is a no-op for these tests.
PLANE_X = 75.0
PLANE_Y = 75.0


# ---------------------------------------------------------------------------
# Family generators pass validity (with confinement applied)
# ---------------------------------------------------------------------------

def _assert_valid(traj, n_samples: int = N_SAMPLES):
    assert "pos" in traj and "yaw" in traj
    assert traj["pos"].shape == (n_samples, 3)
    assert traj["yaw"].shape == (n_samples,)
    ok, reason = validate_trajectory(traj, DT)
    assert ok, f"validity failed: {reason}"


def test_sinusoid_validity() -> None:
    _assert_valid(sinusoid_trajectory(42, PLANE_X, PLANE_Y))


def test_flat_spiral_validity() -> None:
    _assert_valid(flat_spiral_trajectory(2000, PLANE_X, PLANE_Y))


def test_flat_spiral_bounded_over_full_duration() -> None:
    """Damped spiral must stay bounded — ``r = r0 + r_gain*(1 - exp(-t/tau))``
    has asymptote r0 + r_gain ≤ 7 m. Test several seeds."""
    for seed in (2000, 2100, 2200):
        traj = flat_spiral_trajectory(seed, PLANE_X, PLANE_Y)
        xy = traj["pos"][:, :2]
        max_r = float(np.linalg.norm(xy, axis=-1).max())
        assert max_r < 10.0, (
            f"flat_spiral seed={seed} max radius {max_r:.2f} m ≥ 10 m "
            f"(expected bounded by asymptote r0 + r_gain ≤ 7 m "
            f"+ confinement headroom).")


def test_figure_8_validity() -> None:
    _assert_valid(figure_8_trajectory(2001, PLANE_X, PLANE_Y))


def test_clover_validity() -> None:
    _assert_valid(clover_trajectory(2002, PLANE_X, PLANE_Y))


def test_minitest_validity() -> None:
    traj = minitest_trajectory(3000, PLANE_X, PLANE_Y)
    _assert_valid(traj, n_samples=MINITEST_N_SAMPLES)
    assert traj["meta"]["family"] == "clover"
    assert abs(traj["meta"]["duration_s"] - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# attitude_from_acceleration — physical correctness
# ---------------------------------------------------------------------------

def test_zero_acceleration_gives_zero_rp() -> None:
    T = N_SAMPLES
    pos = np.tile(np.array([0.0, 0.0, 5.0]), (T, 1))
    yaw = np.zeros(T)
    att = attitude_from_acceleration(pos, yaw, DT)
    assert np.allclose(att["roll"], 0.0, atol=1e-12)
    assert np.allclose(att["pitch"], 0.0, atol=1e-12)


def test_pure_x_acceleration_gives_signed_pitch() -> None:
    T = 1000
    t = np.arange(T) * DT
    pos = np.stack([0.5 * t ** 2, np.zeros(T), 5.0 * np.ones(T)], axis=-1)
    yaw = np.zeros(T)
    att = attitude_from_acceleration(pos, yaw, DT)
    pitch_mid = att["pitch"][T // 4 : 3 * T // 4]
    expected = np.arctan(1.0 / GRAVITY)
    assert pitch_mid.mean() > 0
    assert 0.5 * expected <= pitch_mid.mean() <= 1.5 * expected


def test_small_x_oscillation_pitch_amplitude() -> None:
    T = 4000
    t = np.arange(T) * DT
    pos = np.stack([0.1 * np.sin(t), np.zeros(T), 5.0 * np.ones(T)], axis=-1)
    yaw = np.zeros(T)
    att = attitude_from_acceleration(pos, yaw, DT)
    pitch = att["pitch"][200:-200]
    expected = np.arctan(0.1 / GRAVITY)
    observed = float(np.max(np.abs(pitch)))
    assert 0.75 * expected <= observed <= 1.25 * expected


def test_pure_yaw_rate_gives_axial_gyro() -> None:
    T = 2000
    t = np.arange(T) * DT
    pos = np.tile(np.array([0.0, 0.0, 5.0]), (T, 1))
    yaw = 0.1 * t
    att = attitude_from_acceleration(pos, yaw, DT)
    omega = att["ang_vel_body"][200:-200]
    assert np.max(np.abs(omega[:, 0])) < 1e-6
    assert np.max(np.abs(omega[:, 1])) < 1e-6
    assert np.allclose(omega[:, 2], 0.1, atol=1e-6)


# ---------------------------------------------------------------------------
# ω_body independent cross-check via R_dot · R.T = skew(ω_world_meas)
# ---------------------------------------------------------------------------

def _R_wb_from_rpy(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    T = roll.shape[0]
    R = np.zeros((T, 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def test_omega_body_matches_R_dot_skew() -> None:
    traj = flat_spiral_trajectory(2000, PLANE_X, PLANE_Y)
    att = attitude_from_acceleration(traj["pos"], traj["yaw"], DT)
    R = _R_wb_from_rpy(att["roll"], att["pitch"], att["yaw"])
    R_dot = np.zeros_like(R)
    R_dot[1:-1] = (R[2:] - R[:-2]) / (2.0 * DT)
    R_dot[0] = (R[1] - R[0]) / DT
    R_dot[-1] = (R[-1] - R[-2]) / DT
    Omega = np.einsum("nij,njk->nik", np.transpose(R, (0, 2, 1)), R_dot)
    w_x =  Omega[:, 2, 1]
    w_y = -Omega[:, 2, 0]
    w_z =  Omega[:, 1, 0]
    omega_ref = np.stack([w_x, w_y, w_z], axis=-1)
    omega_lib = att["ang_vel_body"]
    diff = omega_lib[200:-200] - omega_ref[200:-200]
    assert np.max(np.abs(diff)) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
