"""Synthetic IMU simulation for Phase 2.

Pipeline (replaces the prior triple-finite-diff pipeline that produced
~2.34 rad/s gyro spikes at trajectory boundaries for cosine-start families):

  1. Velocity / acceleration in world frame from the position trajectory using
     2nd-order central differences with 2nd-order one-sided stencils at the
     endpoints. The one-sided stencil is exact for quadratics and for the
     small-(omega*dt) regime of our trajectories it matches the analytic
     derivative to numerical precision — no boundary spike.
  2. Body attitude via coordinated-turn:
        body-z (FLU) along the thrust direction (acc - g_world),
        body-x along the velocity vector,
        body-y = body-z x body-x (right-hand FLU).
     This makes the body yaw with the orbital motion so the gyro carries the
     orbital rotation rate rather than nothing.
  3. Body angular velocity from R-chain finite difference: R_dot via 2nd-order
     scheme, then omega_skew = R^T @ R_dot, omega = vee(omega_skew). One stage
     of differentiation on a smooth rotation matrix; no compounding boundary
     amplification.
  4. Specific force (accelerometer reading) = R^T @ (a_world - g_world).
     Stationary reading ~ +g in body z (FLU convention).

Convention (FLU body, ENU world):
  - Body: x-forward (along velocity), y-left, z-up (along thrust).
  - World gravity: g_world = [0, 0, -g]. Stationary IMU reads (0, 0, +g).
  - JPL/Hamilton quaternion not used here — this module returns rotation
    matrices and Euler angles only.

The function `simulate_imu` retains its old call signature (pos, roll, pitch,
yaw, dt, ...). The roll/pitch/yaw arguments are kept for backwards
compatibility with existing callers in `generate_data.py` but are IGNORED:
attitude is recomputed via coordinated-turn from `pos` so the IMU is internally
consistent and informative regardless of what attitude the caller derived.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Differentiation helpers
# ---------------------------------------------------------------------------

def _central_diff_o2(x: np.ndarray, dt: float) -> np.ndarray:
    """Central differences along axis 0 with 2nd-order one-sided endpoints.

    Interior:   x'[i]   = (x[i+1] - x[i-1]) / (2*dt)             (O(dt^2))
    Left end:   x'[0]   = (-3*x[0] + 4*x[1] - x[2]) / (2*dt)     (O(dt^2))
    Right end:  x'[-1]  = (3*x[-1] - 4*x[-2] + x[-3]) / (2*dt)   (O(dt^2))

    The one-sided stencil is exact on quadratic polynomials. For our smooth
    parametric trajectories sampled at 1000 Hz with omega <= 1 rad/s, the
    Taylor remainder past the quadratic is at the 1e-7 level — i.e.
    indistinguishable from the analytic derivative.
    """
    if x.shape[0] < 3:
        raise ValueError(f"_central_diff_o2 needs at least 3 samples, got {x.shape[0]}")
    d = np.empty_like(x, dtype=np.float64)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0]    = (-3.0 * x[0]  + 4.0 * x[1]  - x[2])  / (2.0 * dt)
    d[-1]   = ( 3.0 * x[-1] - 4.0 * x[-2] + x[-3]) / (2.0 * dt)
    return d


# ---------------------------------------------------------------------------
# Coordinated-turn attitude
# ---------------------------------------------------------------------------

def _coordinated_turn_R_wb(vel: np.ndarray, acc_world: np.ndarray,
                           g: float = GRAVITY,
                           eps: float = 1e-3) -> np.ndarray:
    """Build R_wb (T, 3, 3) such that v_world = R_wb @ v_body, FLU body frame.

    body-z along thrust direction = (acc_world - g_world), so stationary
    body-z = world-z (up). body-x along velocity. body-y = z x x.

    When ||vel||[i] <= eps (degenerate body-x direction) the fallback is to
    HOLD the previous valid body-x rather than snap to [1, 0, 0]. The snap
    fallback caused a ~1200 rad/s discontinuity in R_wb at the trajectory
    boundaries of Gaussian-smoothed paths (diamond / zigzag / triangle),
    where reflect-padding zeros the smoothed velocity exactly at sample 0
    and sample T-1. Holding the previous direction keeps R_wb continuous
    through the boundary; the resulting body-x is geometrically meaningless
    at a true v=0 sample but the continuity prevents the R-chain ω finite-
    diff from manufacturing a spurious spike.

    eps = 1e-3 m/s (1 mm/s): below this, body-x = vel/||vel|| is numerically
    noise-driven (float64 ULP fluctuations dominate the direction). The
    earlier 1e-9 threshold caught the strictly-zero case but left a
    "barely-above-eps" gray zone (e.g. 1e-9..1e-3 m/s) where vel/||vel||
    pointed in random directions per sample, producing a residual ~3 rad/s
    boundary discontinuity on diamond/zigzag/triangle when the smoothed
    boundary velocity happened to land in that gray zone. 1 mm/s is well
    below any physically meaningful traversal speed in this corpus
    (figure8/circle/ellipse min |vel| > 0.1 m/s by construction at any
    target_peak_speed >= 0.3 m/s) so the fallback branch only fires on the
    smoothed-piecewise families' boundary samples.
    """
    g_world = np.array([0.0, 0.0, -g], dtype=np.float64)
    thrust = acc_world - g_world                         # (T, 3)
    tn = np.linalg.norm(thrust, axis=1, keepdims=True)
    z_body = np.where(tn > eps, thrust / np.maximum(tn, eps),
                      np.broadcast_to(np.array([[0.0, 0.0, 1.0]]), thrust.shape))

    vn = np.linalg.norm(vel, axis=1, keepdims=True).reshape(-1)
    T = vel.shape[0]
    x_raw = np.empty_like(vel, dtype=np.float64)
    valid_mask = vn > eps
    if not valid_mask.any():
        # Pathological input: the whole trajectory is degenerate. Keep the
        # legacy [1, 0, 0] fallback as a final safety net (no NaN downstream).
        print("[imu_simulation] WARNING: _coordinated_turn_R_wb received "
              "an all-degenerate velocity (||vel|| <= eps for every sample). "
              "Falling back to body-x = [1, 0, 0] for the entire trajectory.")
        x_raw[:] = np.array([1.0, 0.0, 0.0])
    else:
        # Step 1: fill all valid samples with vel / ||vel||.
        x_raw[valid_mask] = (vel[valid_mask].T
                              / np.maximum(vn[valid_mask], eps)).T
        # Step 2: backfill any leading degenerate samples (sample 0 may be
        # below eps due to reflect-padded smoother) from the first valid one.
        first_valid = int(np.argmax(valid_mask))   # first True index
        if first_valid > 0:
            x_raw[:first_valid] = x_raw[first_valid]
        # Step 3: forward-fill any later degenerate samples by carrying the
        # previous valid direction. Vectorised via cumulative max-of-index.
        idx = np.arange(T)
        last_valid_idx = np.where(valid_mask, idx, -1)
        last_valid_idx = np.maximum.accumulate(last_valid_idx)
        # last_valid_idx[i] >= 0 for all i once we've passed first_valid.
        degenerate_after = (~valid_mask) & (idx > first_valid)
        if degenerate_after.any():
            x_raw[degenerate_after] = x_raw[last_valid_idx[degenerate_after]]

    y_body = np.cross(z_body, x_raw)
    yn = np.linalg.norm(y_body, axis=1, keepdims=True)
    # If x_raw is parallel to z_body (vertical motion), pick an arbitrary y.
    y_body = np.where(yn > eps, y_body / np.maximum(yn, eps),
                      np.broadcast_to(np.array([[0.0, 1.0, 0.0]]), y_body.shape))

    x_body = np.cross(y_body, z_body)                    # reorthogonalize
    xn = np.linalg.norm(x_body, axis=1, keepdims=True)
    x_body = x_body / np.maximum(xn, eps)

    R_wb = np.stack([x_body, y_body, z_body], axis=-1)   # columns are body axes
    return R_wb


def _euler_zyx_from_R(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (roll, pitch, yaw) from R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    pitch = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    roll  = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    yaw   = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return roll, pitch, yaw


def _gyro_from_rotation_chain(R: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity (T, 3) from R_wb via R^T @ R_dot."""
    R_dot = _central_diff_o2(R, dt)                                # (T, 3, 3)
    omega_skew = np.einsum("tji,tjk->tik", R, R_dot)               # R^T @ R_dot
    return np.column_stack([omega_skew[:, 2, 1],                   # x
                             omega_skew[:, 0, 2],                  # y
                             omega_skew[:, 1, 0]])                 # z


# ---------------------------------------------------------------------------
# Public: attitude bundle (coordinated turn from position only)
# ---------------------------------------------------------------------------

def coordinated_turn_attitude(pos: np.ndarray, dt: float,
                              g: float = GRAVITY,
                              vel: Optional[np.ndarray] = None,
                              acc: Optional[np.ndarray] = None,
                              ) -> Dict[str, np.ndarray]:
    """Compute coordinated-turn attitude + kinematics from position trajectory.

    `vel` and `acc` (Step S — Option B) are optional analytic derivatives
    supplied by the simple_* samplers; when present they are used directly.
    Otherwise both are computed from `pos` via the 2nd-order central diff.

    Returns a dict with: roll, pitch, yaw, acc_world, ang_vel_body, R_wb.
    All arrays are length-T along axis 0 and time-aligned with `pos`.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if vel is None:
        vel = _central_diff_o2(pos, dt)
    else:
        vel = np.asarray(vel, dtype=np.float64)
    if acc is None:
        acc = _central_diff_o2(vel, dt)
    else:
        acc = np.asarray(acc, dtype=np.float64)
    R_wb = _coordinated_turn_R_wb(vel, acc, g=g)
    roll, pitch, yaw = _euler_zyx_from_R(R_wb)
    omega_body = _gyro_from_rotation_chain(R_wb, dt)
    return {
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "acc_world": acc,
        "ang_vel_body": omega_body,
        "R_wb": R_wb,
    }


# ---------------------------------------------------------------------------
# Public: noisy IMU (accel + gyro)
# ---------------------------------------------------------------------------

def simulate_imu(pos: np.ndarray, roll: np.ndarray, pitch: np.ndarray,
                 yaw: np.ndarray, dt: float,
                 sigma_acc: float = 0.02, sigma_gyro: float = 0.002,
                 g: float = GRAVITY, seed: int = 0,
                 vel: Optional[np.ndarray] = None,
                 acc: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
    """Return (T, 6) body-frame IMU readings [ax, ay, az, gx, gy, gz].

    The roll/pitch/yaw arguments are accepted for backwards compatibility but
    IGNORED. Attitude and angular velocity are recomputed from `pos` via
    coordinated-turn so the IMU is informative (gyro carries orbital rate)
    and free of the boundary spikes that came from triple-stacked finite
    differences in the prior implementation.

    `vel` and `acc` are optional analytic derivatives (Step S — Option B);
    when supplied they are forwarded to `coordinated_turn_attitude`.
    """
    pos = np.asarray(pos, dtype=np.float64)
    T = pos.shape[0]
    del roll, pitch, yaw  # intentionally unused (see docstring)

    att = coordinated_turn_attitude(pos, dt, g=g, vel=vel, acc=acc)
    R_wb = att["R_wb"]                                            # (T, 3, 3)
    a_world = att["acc_world"]                                    # (T, 3)
    omega_body = att["ang_vel_body"]                              # (T, 3)

    g_world = np.array([0.0, 0.0, -g], dtype=np.float64)
    sf_world = a_world - g_world                                  # (T, 3)
    acc_body = np.einsum("tji,tj->ti", R_wb, sf_world)            # R^T @ sf

    rng = np.random.default_rng(int(seed))
    acc_noisy = acc_body + rng.normal(0.0, sigma_acc, acc_body.shape)
    gyr_noisy = omega_body + rng.normal(0.0, sigma_gyro, omega_body.shape)

    out = np.concatenate([acc_noisy, gyr_noisy], axis=-1).astype(np.float64)
    assert out.shape == (T, 6), f"IMU shape mismatch {out.shape}"
    return out
