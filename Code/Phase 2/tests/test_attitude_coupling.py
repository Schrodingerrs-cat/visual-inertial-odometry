"""Physics invariant: R_wb @ [0,0,1] == normalized(acc_world + [0,0,g]).

This is the thrust-alignment equation for a quadrotor with body frame
+X forward, +Y left, +Z up (FLU / ENU). The thrust always points along the
body's +Z axis; Newton's law says thrust/mass = acc − (−g ẑ) = acc + g ẑ.

Regression test for the sign bug on the roll computation in
``attitude_from_acceleration`` (fixed 2026-04-23). The bug caused the
drone_dynamics sanity check to report max error ≈ 2·|sin(roll)|.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "blender"))

from trajectory_library import attitude_from_acceleration, GRAVITY, DT


def _synthesize_pos_for_accel(acc_world: np.ndarray, dt: float, n: int = 50,
                               z0: float = 2.0) -> np.ndarray:
    """Return a (n, 3) position trajectory whose 2nd central-difference ≈ acc_world.

    ``pos[k] = 0.5 · acc · (k·dt)² + z_offset``  so the second derivative is acc.
    """
    t = np.arange(n, dtype=np.float64) * dt
    pos = 0.5 * acc_world[None, :] * (t[:, None] ** 2)
    pos[:, 2] += z0
    return pos


def _R_wb_from_rpy_scalar(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) · Ry(pitch) · Rx(roll). Same matrix _R_wb_from_rpy produces."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                 cp * cr]
    ])


def thrust_alignment_error(acc_world: np.ndarray, yaw: float = 0.0) -> float:
    """Compute |R_wb @ [0,0,1] − normalized(acc_world + g·ẑ)| at mid-trajectory."""
    pos = _synthesize_pos_for_accel(acc_world, dt=DT, n=200)
    yaw_seq = np.full(200, yaw, dtype=np.float64)
    att = attitude_from_acceleration(pos, yaw_seq, DT, g=GRAVITY)
    mid = 100
    R = _R_wb_from_rpy_scalar(float(att["roll"][mid]),
                              float(att["pitch"][mid]),
                              float(att["yaw"][mid]))
    b_z = R @ np.array([0.0, 0.0, 1.0])
    thrust_ref = acc_world + np.array([0.0, 0.0, GRAVITY])
    thrust_ref /= np.linalg.norm(thrust_ref)
    return float(np.linalg.norm(b_z - thrust_ref))


# ---------------------------------------------------------------------------
# Regression tests (one per axis + diagonals)
# ---------------------------------------------------------------------------

def test_plus_y_acceleration_gives_negative_roll() -> None:
    """acc=(0,+1,0): body +Z tilts toward +Y, which (about body +X) requires
    NEGATIVE roll in FLU convention.
    """
    pos = _synthesize_pos_for_accel(np.array([0.0, 1.0, 0.0]), DT, n=200)
    yaw = np.zeros(200)
    att = attitude_from_acceleration(pos, yaw, DT)
    assert att["roll"][100] < 0.0, (
        f"expected negative roll for +Y accel, got {att['roll'][100]:+.5f}")
    assert abs(att["pitch"][100]) < 1e-6


def test_plus_x_acceleration_gives_positive_pitch() -> None:
    """acc=(+1,0,0): body +Z tilts toward +X, which about body +Y is +pitch."""
    pos = _synthesize_pos_for_accel(np.array([1.0, 0.0, 0.0]), DT, n=200)
    yaw = np.zeros(200)
    att = attitude_from_acceleration(pos, yaw, DT)
    assert att["pitch"][100] > 0.0
    assert abs(att["roll"][100]) < 1e-6


def test_five_sample_thrust_alignment() -> None:
    """Spot-check 5 accelerations: thrust alignment error must be < 1e-6."""
    cases = [
        np.array([0.0,  1.0, 0.0]),
        np.array([1.0,  0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0,  1.0, 0.0]),
        np.array([-0.5, 0.7, 0.0]),
    ]
    errs = [thrust_alignment_error(a) for a in cases]
    for a, e in zip(cases, errs):
        assert e < 1e-6, f"acc={a}: alignment error {e:.6e} exceeds 1e-6"


if __name__ == "__main__":
    # Self-running (no pytest needed).
    print("Running attitude-coupling regression tests...\n")
    try:
        test_plus_y_acceleration_gives_negative_roll()
        print("PASS  test_plus_y_acceleration_gives_negative_roll")
    except AssertionError as exc:
        print(f"FAIL  test_plus_y_acceleration_gives_negative_roll: {exc}")
    try:
        test_plus_x_acceleration_gives_positive_pitch()
        print("PASS  test_plus_x_acceleration_gives_positive_pitch")
    except AssertionError as exc:
        print(f"FAIL  test_plus_x_acceleration_gives_positive_pitch: {exc}")

    print("\n5-sample spot check (error must be < 1e-6):")
    cases = [
        np.array([0.0,  1.0, 0.0]),
        np.array([1.0,  0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0,  1.0, 0.0]),
        np.array([-0.5, 0.7, 0.0]),
    ]
    max_err = 0.0
    for a in cases:
        e = thrust_alignment_error(a)
        flag = "PASS" if e < 1e-6 else "FAIL"
        print(f"  {flag}  acc={tuple(a)}:  err = {e:.3e}")
        max_err = max(max_err, e)
    print(f"\nMax error across 5 samples: {max_err:.3e}")
    print("OK" if max_err < 1e-6 else "BUG STILL PRESENT")
