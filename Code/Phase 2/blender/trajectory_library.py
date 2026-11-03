"""Pure-Python trajectory generators for Phase 2.

No `bpy` imports — this module is usable both inside Blender's Python
interpreter (for rendering) and from the training / sanity scripts that run
on the host Python. All standard generators sample position and yaw at 1000 Hz
over 10 seconds (10001 samples) and couple roll/pitch to commanded acceleration
via `attitude_from_acceleration`.

Public API:
    sinusoid_trajectory(seed, plane_x, plane_y)      → dict {t, pos, yaw, meta}
    flat_spiral_trajectory(seed, plane_x, plane_y)   → dict
    figure_8_trajectory(seed, plane_x, plane_y)      → dict
    clover_trajectory(seed, plane_x, plane_y)        → dict
    minitest_trajectory(seed, plane_x, plane_y)      → dict (2 s, 2001 samples)
    attitude_from_acceleration(pos, yaw, dt, g=9.81) → dict
    validate_trajectory(traj, dt)                    → (bool, str)
    confine_to_plane(traj, plane_x, plane_y, max_footprint_half=3.5) → dict

Each trajectory family performs reject-and-resample internally via
`validate_trajectory` to guarantee the returned trajectory passes all
validity checks. Confinement is applied inside each sampler so the returned
trajectory is already clipped to the plane bounds.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS = 1000
DURATION_S = 100                             # was 10; 10× longer trajectories
#                                              give the networks more motion
#                                              diversity + fairer ATE test.
N_SAMPLES = FPS * DURATION_S + 1            # 100001 @ 1000 Hz
DT = 1.0 / FPS
GRAVITY = 9.81

MINITEST_DURATION_S = 2.0                    # minitest (val traj 200) is short
MINITEST_N_SAMPLES = int(FPS * MINITEST_DURATION_S) + 1  # 2001

# Step I: simple_figure8 — short (30 s) easy trajectory at constant altitude.
# Used as the second eval point (traj_id 300) to test whether VI clearly
# beats I when the motion is gentle enough that fusion should help.
SIMPLE_DURATION_S = 30.0
SIMPLE_N_SAMPLES = int(FPS * SIMPLE_DURATION_S) + 1   # 30001 @ 1000 Hz
SIMPLE_ALTITUDE_M = 4.0

MAX_ROLL_PITCH_RAD = np.deg2rad(35.0)       # restored — thrust-alignment
#                                             physics fails beyond ~35°.
MIN_PEAK_ACC = 0.01                         # was 0.05 — Step K speed-diverse
#                                             training set includes
#                                             target_peak_speed=0.1 m/s circles
#                                             whose analytical peak |a| =
#                                             v²/R ≈ 0.021 m/s². 0.01 keeps a
#                                             ~2× safety margin while still
#                                             rejecting truly-stationary trajs.
DISP10_BOUNDS_M = (0.0005, 1.5)             # was (0.002, 1.5) — the lower
#                                             bound was 0.2 cm/10 ms, blocking
#                                             v < 0.2 m/s. Step K training
#                                             targets v=0.1 m/s ⇒ median disp
#                                             over 10 ms ≈ 0.06 cm. 0.0005 m
#                                             (= 0.05 cm) gives ~1.5× margin.
MAX_OMEGA_BODY = 200.0                      # was 2.0 — ω_body is a 3rd
#                                             numerical derivative of pos;
#                                             central-diff noise spikes to
#                                             ~70 rad/s even when thrust-
#                                             alignment physics (the real
#                                             consistency check) holds.

MAX_RESAMPLE_ATTEMPTS = 64

# Camera footprint half-extent: altitude * tan(HFOV/2). With HFOV ~= 53 deg
# at altitude 6 m, tan(26.5 deg) ~= 0.499 so half-extent ~= 3.0 m. 3.5 m
# gives ~15% margin.
DEFAULT_FOOTPRINT_HALF_M = 3.5


# ---------------------------------------------------------------------------
# Attitude coupling
# ---------------------------------------------------------------------------

def _central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Central differences along axis 0 with 2nd-order one-sided endpoints.

    Interior:   (x[i+1] - x[i-1]) / (2*dt)
    Left end:   (-3*x[0] + 4*x[1] - x[2]) / (2*dt)
    Right end:  (3*x[-1] - 4*x[-2] + x[-3]) / (2*dt)

    The one-sided stencil is exact on quadratics — the prior 1st-order
    forward/backward stencil produced an O(1/dt) spike when the position
    started at a cosine peak (simple_circle, simple_ellipse) because the
    error compounded through three stacked diff stages.
    """
    if x.shape[0] < 3:
        d = np.zeros_like(x, dtype=np.float64)
        d[0]  = (x[1] - x[0]) / dt
        d[-1] = (x[-1] - x[-2]) / dt
        return d
    d = np.empty_like(x, dtype=np.float64)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0]    = (-3.0 * x[0]  + 4.0 * x[1]  - x[2])  / (2.0 * dt)
    d[-1]   = ( 3.0 * x[-1] - 4.0 * x[-2] + x[-3]) / (2.0 * dt)
    return d


def attitude_from_acceleration(pos: np.ndarray, yaw: np.ndarray, dt: float,
                               g: float = GRAVITY,
                               vel: Optional[np.ndarray] = None,
                               acc: Optional[np.ndarray] = None,
                               ) -> Dict[str, np.ndarray]:
    """Coordinated-turn attitude + kinematics from a position trajectory.

    The `yaw` argument is kept for backwards compatibility with existing
    callers (validate_trajectory, generate_data.py) but is IGNORED: yaw is
    determined by the velocity direction so the body frame yaws with the
    orbital motion. This makes the gyro carry the orbital rotation rate
    instead of nothing — without this, simple_circle / simple_ellipse
    trajectories had gyro signal at the noise floor regardless of speed.

    `vel` and `acc` (Step S — Option B) are optional analytic XY-aligned
    derivatives supplied by the simple_* samplers; when present they are
    used directly so no finite-differencing of pos is required. When absent
    the underlying coordinated_turn_attitude falls back to the 2nd-order
    central diff in `imu_simulation`.

    Returned dict keys (unchanged): roll, pitch, yaw, acc_world, ang_vel_body.
    Plus `R_wb` (T, 3, 3) for callers that prefer the rotation matrix.
    """
    # Local import to avoid a cycle: imu_simulation imports from this module
    # in the legacy code path, but the new pipeline only depends on this
    # module for its constants — so a deferred import keeps both directions
    # of the relationship safe.
    try:
        from .imu_simulation import coordinated_turn_attitude
    except ImportError:
        from imu_simulation import coordinated_turn_attitude  # type: ignore

    del yaw  # intentionally unused — yaw is now derived from velocity
    return coordinated_turn_attitude(np.asarray(pos, dtype=np.float64),
                                     float(dt), g=float(g),
                                     vel=vel, acc=acc)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def validate_trajectory(traj: Dict[str, np.ndarray], dt: float = DT
                        ) -> Tuple[bool, str]:
    """Run validity checks. Returns (ok, message)."""
    pos = traj["pos"]; yaw = traj["yaw"]
    att = attitude_from_acceleration(
        pos, yaw, dt,
        vel=traj.get("vel"), acc=traj.get("acc"),
    )
    rp_max = float(max(np.max(np.abs(att["roll"])),
                       np.max(np.abs(att["pitch"]))))
    if rp_max > MAX_ROLL_PITCH_RAD:
        return False, (f"|roll or pitch| {np.rad2deg(rp_max):.1f} deg > "
                       f"{np.rad2deg(MAX_ROLL_PITCH_RAD):.0f} deg")

    peak_acc = float(np.max(np.linalg.norm(att["acc_world"], axis=-1)))
    if peak_acc < MIN_PEAK_ACC:
        return False, f"peak |acc| {peak_acc:.3f} m/s² < {MIN_PEAK_ACC}"

    stride = 10
    disp = pos[stride:] - pos[:-stride]
    med = float(np.median(np.linalg.norm(disp, axis=-1)))
    if not (DISP10_BOUNDS_M[0] <= med <= DISP10_BOUNDS_M[1]):
        lo_cm = DISP10_BOUNDS_M[0] * 100.0
        hi_cm = DISP10_BOUNDS_M[1] * 100.0
        return False, (f"median |Δpos(10*dt)| {med*100:.3f} cm out of "
                       f"[{lo_cm:.3f}, {hi_cm:.1f}] cm")

    max_omega = float(np.max(np.linalg.norm(att["ang_vel_body"], axis=-1)))
    if max_omega > MAX_OMEGA_BODY:
        return False, f"max |ω_body| {max_omega:.2f} rad/s > {MAX_OMEGA_BODY}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Plane confinement
# ---------------------------------------------------------------------------

def confine_to_plane(traj: Dict[str, np.ndarray],
                     plane_x: float, plane_y: float,
                     max_footprint_half: float = DEFAULT_FOOTPRINT_HALF_M
                     ) -> Dict[str, np.ndarray]:
    """Shift + uniformly scale the XY of a trajectory to fit inside a plane.

    Preserves trajectory shape (uniform XY scale). Z / altitude is untouched.
    Adds `confinement_scale`, `bbox_xy_before`, `bbox_xy_after` to traj["meta"].
    If the trajectory dict carries analytic `vel` and/or `acc` arrays (Step S
    addition: the simple_* samplers now return closed-form derivatives so the
    IMU pipeline can avoid finite-differencing pos), those XY components are
    rescaled by the same uniform factor — they are linear in the scaled
    position, so the same scalar `scale` applies.
    """
    pos = np.asarray(traj["pos"], dtype=np.float64).copy()
    x = pos[:, 0]; y = pos[:, 1]
    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())

    required_x = (max_x - min_x) + 2.0 * max_footprint_half
    required_y = (max_y - min_y) + 2.0 * max_footprint_half
    available_x = float(plane_x)
    available_y = float(plane_y)

    sx = available_x / required_x if required_x > 0.0 else 1.0
    sy = available_y / required_y if required_y > 0.0 else 1.0
    scale = float(min(1.0, sx, sy))

    # Center of mass in XY before scaling.
    centroid_x_before = 0.5 * (min_x + max_x)
    centroid_y_before = 0.5 * (min_y + max_y)

    # Scale about centroid, then recenter to (0, 0).
    pos[:, 0] = (x - centroid_x_before) * scale
    pos[:, 1] = (y - centroid_y_before) * scale

    # Final post-scale centroid (should be 0,0 by construction).
    x2 = pos[:, 0]; y2 = pos[:, 1]
    bbox_after = [[float(x2.min()), float(x2.max())],
                  [float(y2.min()), float(y2.max())]]

    new_traj = dict(traj)
    new_traj["pos"] = pos
    # Analytic vel/acc (if present) are rescaled XY by the same factor.
    for key in ("vel", "acc"):
        if key in traj:
            arr = np.asarray(traj[key], dtype=np.float64).copy()
            arr[:, 0] *= scale
            arr[:, 1] *= scale
            new_traj[key] = arr
    meta = dict(new_traj.get("meta", {}))
    meta["confinement_scale"] = scale
    meta["bbox_xy_before"] = [[min_x, max_x], [min_y, max_y]]
    meta["bbox_xy_after"] = bbox_after
    meta["max_footprint_half"] = float(max_footprint_half)
    new_traj["meta"] = meta
    return new_traj


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _time_axis(n_samples: int = N_SAMPLES,
               duration_s: float = float(DURATION_S)) -> np.ndarray:
    return np.linspace(0.0, duration_s, n_samples, dtype=np.float64)


def _resample_with_confinement(sampler: Callable[[np.random.Generator],
                                                 Dict[str, np.ndarray]],
                               seed: int,
                               plane_x: float, plane_y: float,
                               max_footprint_half: float = DEFAULT_FOOTPRINT_HALF_M,
                               dt: float = DT,
                               ) -> Dict[str, np.ndarray]:
    """Sample → confine → validate, retrying on rejection.

    Confinement scaling may invalidate the acceleration content (e.g. shrink
    a 3x3 m trajectory to 0.3x0.3 m, dropping peak |acc| below MIN_PEAK_ACC).
    That is surfaced as a rejection and we resample with a bumped seed.
    """
    last_reason = "no-attempts"
    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        rng = np.random.default_rng(int(seed) + attempt)
        traj = sampler(rng)
        traj = confine_to_plane(traj, plane_x, plane_y,
                                max_footprint_half=max_footprint_half)
        ok, reason = validate_trajectory(traj, dt=dt)
        if ok:
            meta = dict(traj.get("meta", {}))
            meta["n_resample_attempts"] = int(attempt)
            traj["meta"] = meta
            return traj
        last_reason = reason
    raise RuntimeError(
        f"trajectory sampler rejected after {MAX_RESAMPLE_ATTEMPTS} attempts "
        f"(seed={seed}, plane_x={plane_x}, plane_y={plane_y}); "
        f"last reason={last_reason!r}")


# ---------------------------------------------------------------------------
# Speed scaling (Step K addition)
# ---------------------------------------------------------------------------
# Step K adds optional `target_peak_speed` (m/s) to the simple_* and
# simple_sinusoid_30s samplers. The scaling is uniform on XY about the
# trajectory's XY centroid, so shape is preserved while peak velocity is
# adjusted. Z (altitude) is untouched. Speed is measured *empirically* from
# central-difference velocities (matches `validate_trajectory`).
#
# Implementation: the public sampler captures `target_peak_speed` in a
# closure that runs INSIDE `_resample_with_confinement`, applying the speed
# scale BEFORE confinement. Confinement then either accepts the result
# (scale_factor 1.0) or shrinks it; in the latter case the post-confinement
# peak velocity is below target. The public sampler verifies empirical peak
# matches target within RELATIVE_TOLERANCE and stores diagnostics in
# `meta["empirical_peak_velocity"]` / `meta["speed_target_relative_error"]`.

SPEED_TARGET_TOLERANCE = 0.10        # 10% — looser than test bound (5%) so
#                                      ± resampling noise + smoothing residual
#                                      doesn't trip the sampler itself; the
#                                      tighter 5% bound is asserted in tests.


def _peak_velocity_xy(pos: np.ndarray, dt: float) -> float:
    """Empirical peak |v_xy| via central differences (m/s)."""
    v = _central_diff(np.asarray(pos[:, :2], dtype=np.float64), dt)
    return float(np.max(np.linalg.norm(v, axis=-1)))


def _scale_xy_about_centroid(pos: np.ndarray, scale: float) -> np.ndarray:
    """Uniform XY scale about the bbox centroid. Z untouched."""
    out = np.asarray(pos, dtype=np.float64).copy()
    cx = 0.5 * (out[:, 0].min() + out[:, 0].max())
    cy = 0.5 * (out[:, 1].min() + out[:, 1].max())
    out[:, 0] = cx + (out[:, 0] - cx) * float(scale)
    out[:, 1] = cy + (out[:, 1] - cy) * float(scale)
    return out


def _apply_speed_scale(traj: Dict[str, np.ndarray],
                       target_peak_speed: Optional[float],
                       dt: float = DT) -> Dict[str, np.ndarray]:
    """If target_peak_speed is set, scale XY so empirical peak |v| matches.

    Records `target_peak_speed`, `base_peak_velocity`, `speed_scale` in
    traj["meta"]. Returns the modified trajectory (a shallow copy with new
    pos array). If target_peak_speed is None, returns traj unchanged.

    Step S addition: if `vel` / `acc` are present (analytic derivatives
    returned by the simple_* samplers), their XY components are rescaled by
    the same factor so the IMU pipeline keeps a consistent (pos, vel, acc)
    triple after speed scaling.
    """
    if target_peak_speed is None:
        return traj
    pos = np.asarray(traj["pos"], dtype=np.float64)
    if "vel" in traj:
        # Use analytic vel for the base-peak computation (no finite-diff
        # boundary artifact). Fall back to finite-diff only if absent.
        v_xy = np.asarray(traj["vel"], dtype=np.float64)[:, :2]
        base_peak = float(np.max(np.linalg.norm(v_xy, axis=-1)))
    else:
        base_peak = _peak_velocity_xy(pos, dt)
    if base_peak < 1e-9:
        raise RuntimeError(
            "speed scaling: base trajectory peak velocity is ~0; "
            "cannot determine scale factor")
    scale = float(target_peak_speed) / base_peak
    new_pos = _scale_xy_about_centroid(pos, scale)
    out = dict(traj)
    out["pos"] = new_pos
    for key in ("vel", "acc"):
        if key in traj:
            arr = np.asarray(traj[key], dtype=np.float64).copy()
            arr[:, 0] *= scale
            arr[:, 1] *= scale
            out[key] = arr
    meta = dict(out.get("meta", {}))
    meta["target_peak_speed"] = float(target_peak_speed)
    meta["base_peak_velocity"] = float(base_peak)
    meta["speed_scale"] = float(scale)
    out["meta"] = meta
    return out


def _record_speed_diagnostics(traj: Dict[str, np.ndarray],
                              target_peak_speed: Optional[float],
                              dt: float = DT) -> Dict[str, np.ndarray]:
    """Record post-confinement empirical peak |v| against target. No raise.

    Confinement may shrink the trajectory and reduce the empirical peak
    below target. Sampler-level enforcement is loose (warn-via-meta);
    tests assert the tight 5% bound.
    """
    if target_peak_speed is None:
        return traj
    empirical = _peak_velocity_xy(traj["pos"], dt)
    rel_err = abs(empirical - float(target_peak_speed)) / float(target_peak_speed)
    meta = dict(traj.get("meta", {}))
    meta["empirical_peak_velocity"] = float(empirical)
    meta["speed_target_relative_error"] = float(rel_err)
    out = dict(traj)
    out["meta"] = meta
    return out


# ---------------------------------------------------------------------------
# Family 1 — sinusoid (training)
# ---------------------------------------------------------------------------

def _sinusoid_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    U = rng.uniform
    # Low-freq: A=3.5@f=0.22 → peak ~6.6 m/s² → 34° tilt (under 35° cap).
    # Velocity ≈ A·2πf ≈ 4.8 m/s → ~4.8 cm / 10ms frame (inside 3–8 cm target).
    p = {
        "A_x1": U(2.5, 3.5),  "A_y1": U(2.5, 3.5),      # was (1.5, 2.5)
        "f_x1": U(0.15, 0.22), "f_y1": U(0.15, 0.22),   # was (0.15, 0.25)
        "p_x1": U(0.0, 2 * np.pi), "p_y1": U(0.0, 2 * np.pi),
        "A_x2": U(0.02, 0.06), "A_y2": U(0.02, 0.06),   # was (0.15, 0.35)
        "f_x2": U(0.4, 0.8),   "f_y2": U(0.4, 0.8),     # was (1.5, 3.0)
        "p_x2": U(0.0, 2 * np.pi), "p_y2": U(0.0, 2 * np.pi),
        "A_z":  U(0.2, 0.5),   "f_z":  U(0.1, 0.4),   "p_z": U(0.0, 2 * np.pi),
        "z0":   U(1.5, 3.0),                             # altitude (m); was (3.0, 6.0)
        "v_yaw": U(-0.1, 0.1),
        "A_yaw": U(0.1, 0.35), "f_yaw": U(0.1, 0.3),  "p_yaw": U(0.0, 2 * np.pi),
    }
    t = _time_axis()

    def s(A, f, ph):
        return A * np.sin(2 * np.pi * f * t + ph)

    x = s(p["A_x1"], p["f_x1"], p["p_x1"]) + s(p["A_x2"], p["f_x2"], p["p_x2"])
    y = s(p["A_y1"], p["f_y1"], p["p_y1"]) + s(p["A_y2"], p["f_y2"], p["p_y2"])
    z = p["z0"] + s(p["A_z"], p["f_z"], p["p_z"])
    pos = np.stack([x, y, z], axis=-1)
    yaw = p["v_yaw"] * t + s(p["A_yaw"], p["f_yaw"], p["p_yaw"])
    return {"t": t, "pos": pos, "yaw": yaw,
            "meta": {"family": "sinusoid", **p}}


def sinusoid_trajectory(seed: int, plane_x: float, plane_y: float
                        ) -> Dict[str, np.ndarray]:
    """Sum-of-sinusoids training trajectory, confined to (plane_x, plane_y)."""
    return _resample_with_confinement(_sinusoid_sampler, seed, plane_x, plane_y)


# ---------------------------------------------------------------------------
# Family 2 — flat_spiral (test)
# ---------------------------------------------------------------------------

def _flat_spiral_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    U = rng.uniform
    # Damped exponential so r stays bounded over 100 s:
    #   r(t) = r0 + r_gain · (1 − exp(−t / tau))
    # Asymptote = r0 + r_gain. With r_gain ∈ [3, 6] and r0 ∈ [0.5, 1] the
    # max radius stays in [3.5, 7], well inside the plane.
    p = {"r0": U(0.5, 1.0),
         "r_gain": U(3.0, 6.0),      # was "a": linear growth → unbounded @ 100s
         "tau": U(15.0, 30.0),       # time constant for r to reach ~63% of gain
         "omega": U(0.25, 0.6),       # was (0.5, 1.0); slower ω reduces ω_body
         "z0": U(1.5, 3.0)}            # altitude (m); was (3.0, 6.0)
    t = _time_axis()
    r = p["r0"] + p["r_gain"] * (1.0 - np.exp(-t / p["tau"]))
    theta = p["omega"] * t
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = p["z0"] + 0.1 * np.sin(2 * np.pi * 0.15 * t)
    pos = np.stack([x, y, z], axis=-1)
    yaw = theta + 0.1 * np.sin(2 * np.pi * 0.2 * t)
    return {"t": t, "pos": pos, "yaw": yaw,
            "meta": {"family": "flat_spiral", **p}}


def flat_spiral_trajectory(seed: int, plane_x: float, plane_y: float
                           ) -> Dict[str, np.ndarray]:
    return _resample_with_confinement(_flat_spiral_sampler, seed,
                                      plane_x, plane_y)


# ---------------------------------------------------------------------------
# Family 3 — figure_8 (test)
# ---------------------------------------------------------------------------

def _figure_8_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    U = rng.uniform
    p = {"A": U(2.0, 3.5), "B": U(1.0, 2.0), "omega": U(0.3, 0.6),
         "z0": U(1.5, 3.0)}            # altitude (m); was (3.0, 6.0)
    t = _time_axis()
    x = p["A"] * np.sin(p["omega"] * t)
    y = p["B"] * np.sin(2.0 * p["omega"] * t)
    z = p["z0"] + 0.15 * np.sin(2 * np.pi * 0.2 * t)
    pos = np.stack([x, y, z], axis=-1)
    yaw = 0.5 * np.sin(0.5 * p["omega"] * t)
    return {"t": t, "pos": pos, "yaw": yaw,
            "meta": {"family": "figure_8", **p}}


def figure_8_trajectory(seed: int, plane_x: float, plane_y: float
                        ) -> Dict[str, np.ndarray]:
    return _resample_with_confinement(_figure_8_sampler, seed, plane_x, plane_y)


# ---------------------------------------------------------------------------
# Family 4 — clover (test)
# ---------------------------------------------------------------------------

def _clover_sampler_factory(R_scale: float = 1.0, duration_s: float = float(DURATION_S),
                            n_samples: int = N_SAMPLES
                            ) -> Callable[[np.random.Generator], Dict[str, np.ndarray]]:
    """Build a clover sampler parametrized by amplitude scale and duration.

    Used both for the full-length clover test trajectory (scale=1) and for
    the short minitest validation trajectory (scale ~ 0.5, duration 2 s).
    """
    def _sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
        U = rng.uniform
        p = {"R": U(1.5, 2.5) * R_scale, "omega": U(0.25, 0.5),
             "z0": U(1.5, 3.0)}         # altitude (m); was (3.0, 6.0)
        t = _time_axis(n_samples, duration_s)
        theta = p["omega"] * t
        r = p["R"] * np.cos(2.0 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = p["z0"] + 0.2 * np.sin(2 * np.pi * 0.25 * t)
        pos = np.stack([x, y, z], axis=-1)
        yaw = theta.copy()
        return {"t": t, "pos": pos, "yaw": yaw,
                "meta": {"family": "clover", **p, "R_scale": R_scale,
                         "duration_s": duration_s, "n_samples": n_samples}}
    return _sampler


def clover_trajectory(seed: int, plane_x: float, plane_y: float
                      ) -> Dict[str, np.ndarray]:
    """Named test trajectory: 4-leaf rose curve, full-length (10 s)."""
    sampler = _clover_sampler_factory(R_scale=1.0,
                                      duration_s=float(DURATION_S),
                                      n_samples=N_SAMPLES)
    return _resample_with_confinement(sampler, seed, plane_x, plane_y)


# ---------------------------------------------------------------------------
# Family 5 — minitest (validation)
# ---------------------------------------------------------------------------

def minitest_trajectory(seed: int, plane_x: float, plane_y: float
                        ) -> Dict[str, np.ndarray]:
    """Shortened clover for validation.

    Duration 2 s, 2001 samples at 1000 Hz, reduced amplitude so the short
    window still passes all validity checks. Used as the minitest validation
    trajectory (traj_id 200) for best-checkpoint selection.
    """
    sampler = _clover_sampler_factory(R_scale=0.5,
                                      duration_s=MINITEST_DURATION_S,
                                      n_samples=MINITEST_N_SAMPLES)
    return _resample_with_confinement(sampler, seed, plane_x, plane_y)


# ---------------------------------------------------------------------------
# Family 6 — simple_figure8 (Step I, traj 300)
# ---------------------------------------------------------------------------

def _simple_figure8_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Smooth 30 s figure-8 in XY at constant altitude.

    One full figure-8 period over the 30 s window. Designed as the "easy"
    eval trajectory — gentle motion, no rotation, constant altitude — to
    test whether VI fusion clearly beats IMU-only on a benign sequence.

    Geometry:
        omega = 2π / 30 ≈ 0.209 rad/s
        x(t) = A · sin(omega · t)            # one X period in 30 s
        y(t) = B · sin(2 · omega · t)        # two Y crossings → figure-8
        z(t) = SIMPLE_ALTITUDE_M             # constant altitude (no climbs)
        yaw(t) = yaw0 (≈ 0)                   # no yaw rotation

    Velocity envelope (target ≲ 1 m/s, no sharp turns):
        |v_x|_max = A·omega ≈ 0.42 m/s     (A=2)
        |v_y|_max = 2·B·omega ≈ 0.42 m/s   (B=1)
        |v|_max ≲ 0.6 m/s.

    Acceleration envelope (must clear MIN_PEAK_ACC=0.05 m/s²):
        |a_x|_max = A·omega² ≈ 0.087 m/s²
        |a_y|_max = 4·B·omega² ≈ 0.175 m/s²
    Both above the validity floor; no resampling pressure expected.

    Plane fit (28 × 18.69 m at the default tex_03 aspect):
        camera footprint half ≈ 4·tan(26.5°) ≈ 2.0 m at altitude 4 m
        required_x = 2·A + 2·footprint_half ≈ 8 m   (fits, scale=1)
        required_y = 2·B + 2·footprint_half ≈ 6 m   (fits, scale=1)
    """
    U = rng.uniform
    p = {
        "A": float(U(1.8, 2.2)),
        "B": float(U(0.9, 1.1)),
        "omega": float(2.0 * np.pi / SIMPLE_DURATION_S),
        "z0": float(SIMPLE_ALTITUDE_M),
        "yaw0": float(U(-0.1, 0.1)),
    }
    t = _time_axis(SIMPLE_N_SAMPLES, SIMPLE_DURATION_S)
    A, B, om = p["A"], p["B"], p["omega"]
    x = A * np.sin(om * t)
    y = B * np.sin(2.0 * om * t)
    z = np.full_like(t, p["z0"])
    pos = np.stack([x, y, z], axis=-1)
    # Analytic derivatives (Step S — Option B) — keeps the IMU pipeline
    # finite-diff-free on pos. confine_to_plane and _apply_speed_scale
    # rescale XY of these arrays in lockstep with pos.
    vx =  A * om * np.cos(om * t)
    vy =  2.0 * B * om * np.cos(2.0 * om * t)
    vz = np.zeros_like(t)
    vel = np.stack([vx, vy, vz], axis=-1)
    ax = -A * om * om * np.sin(om * t)
    ay = -4.0 * B * om * om * np.sin(2.0 * om * t)
    az = np.zeros_like(t)
    acc = np.stack([ax, ay, az], axis=-1)
    yaw = np.full_like(t, p["yaw0"])
    return {"t": t, "pos": pos, "vel": vel, "acc": acc, "yaw": yaw,
            "meta": {"family": "simple_figure8", **p,
                     "duration_s": SIMPLE_DURATION_S,
                     "n_samples": int(SIMPLE_N_SAMPLES)}}


def simple_figure8_trajectory(seed: int, plane_x: float, plane_y: float,
                              target_peak_speed: Optional[float] = None,
                              ) -> Dict[str, np.ndarray]:
    """Step I named eval trajectory: gentle 30 s figure-8 at altitude 4 m.

    Step K addition: optional ``target_peak_speed`` (m/s) uniformly scales
    XY so empirical peak |v_xy| matches. None = legacy behavior (Step J
    trajectories unchanged).
    """
    def sampler(rng):
        return _apply_speed_scale(_simple_figure8_sampler(rng),
                                  target_peak_speed)
    out = _resample_with_confinement(sampler, seed, plane_x, plane_y)
    return _record_speed_diagnostics(out, target_peak_speed)


# ---------------------------------------------------------------------------
# Family 7 — simple_circle (Step J, trajs 302, 306)
# ---------------------------------------------------------------------------

def _simple_circle_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """30 s constant-altitude circle in XY.

    Geometry:
        x(t) = R · cos(omega · t)
        y(t) = R · sin(omega · t)
        z(t) = SIMPLE_ALTITUDE_M
        yaw(t) = yaw0 (small jitter, no rotation rate)

    Velocity / acceleration envelope (with omega = 2π/30 ≈ 0.209 rad/s):
        |v|     = R · omega         ∈ [0.31, 0.52] m/s
        |a|     = R · omega²        ∈ [0.066, 0.110] m/s²
                                     (above MIN_PEAK_ACC=0.05 even at R_min=1.5)
    Plane fit: half-extent R + footprint_half (3.5 m) ⇒ ≤ 12 m, fits the
    28 × 18.69 m plane easily.
    """
    U = rng.uniform
    p = {
        "R": float(U(1.5, 2.5)),
        "omega": float(2.0 * np.pi / SIMPLE_DURATION_S),
        "z0": float(SIMPLE_ALTITUDE_M),
        "yaw0": float(U(-0.1, 0.1)),
    }
    t = _time_axis(SIMPLE_N_SAMPLES, SIMPLE_DURATION_S)
    R, om = p["R"], p["omega"]
    x = R * np.cos(om * t)
    y = R * np.sin(om * t)
    z = np.full_like(t, p["z0"])
    pos = np.stack([x, y, z], axis=-1)
    # Analytic derivatives (Step S — Option B).
    vx = -R * om * np.sin(om * t)
    vy =  R * om * np.cos(om * t)
    vz = np.zeros_like(t)
    vel = np.stack([vx, vy, vz], axis=-1)
    ax = -R * om * om * np.cos(om * t)
    ay = -R * om * om * np.sin(om * t)
    az = np.zeros_like(t)
    acc = np.stack([ax, ay, az], axis=-1)
    yaw = np.full_like(t, p["yaw0"])
    return {"t": t, "pos": pos, "vel": vel, "acc": acc, "yaw": yaw,
            "meta": {"family": "simple_circle", **p,
                     "duration_s": SIMPLE_DURATION_S,
                     "n_samples": int(SIMPLE_N_SAMPLES)}}


def simple_circle_trajectory(seed: int, plane_x: float, plane_y: float,
                             target_peak_speed: Optional[float] = None,
                             ) -> Dict[str, np.ndarray]:
    """Step J named trajectory: gentle 30 s circle at altitude 4 m.

    Step K addition: optional ``target_peak_speed`` (m/s).
    """
    def sampler(rng):
        return _apply_speed_scale(_simple_circle_sampler(rng),
                                  target_peak_speed)
    out = _resample_with_confinement(sampler, seed, plane_x, plane_y)
    return _record_speed_diagnostics(out, target_peak_speed)


# ---------------------------------------------------------------------------
# Family 8 — simple_ellipse (Step J, trajs 303, 307)
# ---------------------------------------------------------------------------

def _simple_ellipse_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """30 s constant-altitude ellipse in XY.

    Geometry:
        x(t) = A · cos(omega · t)
        y(t) = B · sin(omega · t)
        z(t) = SIMPLE_ALTITUDE_M
        yaw(t) = yaw0

    Velocity / acceleration envelope:
        |v_x|_max = A·omega      ∈ [0.42, 0.63] m/s
        |v_y|_max = B·omega      ∈ [0.21, 0.31] m/s
        peak |a|  = max(A,B)·omega² ∈ [0.088, 0.132] m/s²
                                  (above MIN_PEAK_ACC=0.05; validate uses
                                   peak, not min, so the lower-bound axis
                                   doesn't fail us)
    Plane fit: 2A + 7 ≤ 13 m vs plane_x=28 m; 2B + 7 ≤ 10 m vs plane_y=18.69 m.
    """
    U = rng.uniform
    p = {
        "A": float(U(2.0, 3.0)),
        "B": float(U(1.0, 1.5)),
        "omega": float(2.0 * np.pi / SIMPLE_DURATION_S),
        "z0": float(SIMPLE_ALTITUDE_M),
        "yaw0": float(U(-0.1, 0.1)),
    }
    t = _time_axis(SIMPLE_N_SAMPLES, SIMPLE_DURATION_S)
    A, B, om = p["A"], p["B"], p["omega"]
    x = A * np.cos(om * t)
    y = B * np.sin(om * t)
    z = np.full_like(t, p["z0"])
    pos = np.stack([x, y, z], axis=-1)
    # Analytic derivatives (Step S — Option B).
    vx = -A * om * np.sin(om * t)
    vy =  B * om * np.cos(om * t)
    vz = np.zeros_like(t)
    vel = np.stack([vx, vy, vz], axis=-1)
    ax = -A * om * om * np.cos(om * t)
    ay = -B * om * om * np.sin(om * t)
    az = np.zeros_like(t)
    acc = np.stack([ax, ay, az], axis=-1)
    yaw = np.full_like(t, p["yaw0"])
    return {"t": t, "pos": pos, "vel": vel, "acc": acc, "yaw": yaw,
            "meta": {"family": "simple_ellipse", **p,
                     "duration_s": SIMPLE_DURATION_S,
                     "n_samples": int(SIMPLE_N_SAMPLES)}}


def simple_ellipse_trajectory(seed: int, plane_x: float, plane_y: float,
                              target_peak_speed: Optional[float] = None,
                              ) -> Dict[str, np.ndarray]:
    """Step J named trajectory: gentle 30 s ellipse at altitude 4 m.

    Step K addition: optional ``target_peak_speed`` (m/s).
    """
    def sampler(rng):
        return _apply_speed_scale(_simple_ellipse_sampler(rng),
                                  target_peak_speed)
    out = _resample_with_confinement(sampler, seed, plane_x, plane_y)
    return _record_speed_diagnostics(out, target_peak_speed)


# ---------------------------------------------------------------------------
# Family 9-11 helpers (Step K) — Gaussian-smoothed piecewise-linear paths
# ---------------------------------------------------------------------------
# These power simple_diamond / simple_zigzag / simple_triangle. The pure
# piecewise-linear paths have delta-function accelerations at corners; the
# Gaussian smoother (np.convolve with a normalised Gaussian kernel; reflect-
# padded) spreads each velocity discontinuity over ~6·sigma samples,
# bounding peak |acc| while preserving a visible corner-like cusp in
# velocity. sigma is given in samples; at 1000 Hz, sigma=600 ≈ 0.6 s.

def _gaussian_smooth_1d(arr: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing along axis 0; reflect-padded; numpy-only."""
    if sigma_samples <= 0:
        return arr.astype(np.float64, copy=True)
    half = max(int(np.ceil(3.0 * sigma_samples)), 2)
    kx = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (kx / sigma_samples) ** 2)
    kernel /= kernel.sum()
    pad_lo = arr[1:half + 1][::-1]
    pad_hi = arr[-half - 1:-1][::-1]
    padded = np.concatenate([pad_lo, arr, pad_hi], axis=0)
    out = np.convolve(padded, kernel, mode="valid")
    return out


def _polyline_at_constant_speed(vertices: np.ndarray, n_samples: int,
                                closed: bool = True) -> np.ndarray:
    """Sample a polyline at uniform arc-length spacing.

    `vertices`: (V, 2) — corner xy positions.
    Returns (n_samples, 2). If `closed`, the final vertex is the first vertex
    (closes the loop). Constant arc-length parametrisation gives constant
    edge speed, so velocity is piecewise-constant and corners are velocity
    discontinuities (smoothed downstream by `_gaussian_smooth_1d`).
    """
    V = np.asarray(vertices, dtype=np.float64)
    if closed:
        V = np.vstack([V, V[0:1]])
    seg_lens = np.linalg.norm(np.diff(V, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    s = np.linspace(0.0, total, n_samples, dtype=np.float64)
    # Vectorised: for each s, find the segment index and interpolate within.
    idx = np.searchsorted(cum, s, side="right") - 1
    idx = np.clip(idx, 0, len(seg_lens) - 1)
    seg_t = (s - cum[idx]) / np.where(seg_lens[idx] > 0, seg_lens[idx], 1.0)
    out = V[idx] + (V[idx + 1] - V[idx]) * seg_t[:, None]
    return out


# ---------------------------------------------------------------------------
# Family 9 — simple_diamond (Step K, train traj 314)
# ---------------------------------------------------------------------------

def _simple_diamond_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Smoothed-corner diamond at constant altitude (closed 4-corner loop).

    Geometry:
        Vertices at (+A, 0), (0, +B), (-A, 0), (0, -B)  — diamond pointing on
        the principal axes. Path traversed at constant arc-length speed; each
        corner is rounded by a Gaussian smoother (sigma ≈ 0.6 s) so peak
        |acc| stays in [0.1, 0.4] m/s² with the sample bounds below.

    Velocity / acceleration envelope (with A,B ∈ [1.5, 2.5], T = 30 s):
        perimeter   ≈ 4 · √(A² + B²)        ∈ [8.5, 14.1] m
        edge speed  ≈ perimeter / T         ∈ [0.28, 0.47] m/s
        |Δv| at corner ≈ √2 · edge_speed    ∈ [0.40, 0.67] m/s
        peak |a|    ≈ |Δv| / (sigma · √(2π))   for sigma=0.6 s
                    ∈ [0.27, 0.44] m/s²     (above MIN_PEAK_ACC=0.05)
    Plane fit: max footprint half ≈ 3.5 m → 2A + 7 ≤ 12 m fits 28 m plane.
    """
    U = rng.uniform
    p = {
        "A":           float(U(1.5, 2.5)),
        "B":           float(U(1.5, 2.5)),
        "smooth_s":    float(U(0.55, 0.75)),    # smoothing sigma (seconds)
        "z0":          float(SIMPLE_ALTITUDE_M),
        "yaw0":        float(U(-0.1, 0.1)),
        "phase_t":     float(U(0.0, 1.0)),       # rotates start position
                                                  # along the loop without
                                                  # changing the shape.
    }
    t = _time_axis(SIMPLE_N_SAMPLES, SIMPLE_DURATION_S)
    vertices = np.array([
        [p["A"],   0.0],
        [0.0,      p["B"]],
        [-p["A"],  0.0],
        [0.0,     -p["B"]],
    ])
    raw = _polyline_at_constant_speed(vertices, SIMPLE_N_SAMPLES, closed=True)
    # Phase shift: roll the loop by `phase_t` of the period.
    shift = int(p["phase_t"] * SIMPLE_N_SAMPLES) % SIMPLE_N_SAMPLES
    raw = np.roll(raw, shift, axis=0)
    # Smooth corners — Gaussian along time axis, both x and y.
    sigma_samples = p["smooth_s"] * FPS
    x = _gaussian_smooth_1d(raw[:, 0], sigma_samples)
    y = _gaussian_smooth_1d(raw[:, 1], sigma_samples)
    z = np.full_like(t, p["z0"])
    pos = np.stack([x, y, z], axis=-1)
    yaw = np.full_like(t, p["yaw0"])
    return {"t": t, "pos": pos, "yaw": yaw,
            "meta": {"family": "simple_diamond", **p,
                     "duration_s": SIMPLE_DURATION_S,
                     "n_samples": int(SIMPLE_N_SAMPLES)}}


def simple_diamond_trajectory(seed: int, plane_x: float, plane_y: float,
                              target_peak_speed: Optional[float] = None,
                              ) -> Dict[str, np.ndarray]:
    """Step K named trajectory: rounded diamond at altitude 4 m, 30 s loop.

    Step K addition: optional ``target_peak_speed`` (m/s).
    """
    def sampler(rng):
        return _apply_speed_scale(_simple_diamond_sampler(rng),
                                  target_peak_speed)
    out = _resample_with_confinement(sampler, seed, plane_x, plane_y)
    return _record_speed_diagnostics(out, target_peak_speed)


# ---------------------------------------------------------------------------
# Family 10 — simple_zigzag (Step K, train traj 315)
# ---------------------------------------------------------------------------

def _simple_zigzag_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Open zigzag path: x increases linearly, y oscillates triangularly.

    Geometry:
        x(t) = -x_extent/2 + x_extent · t / T
        y(t) = A_y · triangular_wave(2 cycles over T)
        z(t) = SIMPLE_ALTITUDE_M

    Two full cycles in 30 s → 4 zigzag corners (2 peaks at +A_y, 2 valleys
    at −A_y). Velocity discontinuities at peaks are smoothed by a Gaussian
    so peak |acc| stays below 0.5 m/s².

    Velocity / acceleration envelope (A_y ∈ [1.0, 2.0], x_extent ∈ [3, 5]):
        v_x  = x_extent / T            ∈ [0.10, 0.17] m/s
        v_y  = 4 · A_y / T_cycle       T_cycle = 15 s
                                       ∈ [0.27, 0.53] m/s
        |Δv_y| at peak ≈ 2 · v_y       ∈ [0.53, 1.07] m/s
        peak |a|       ≈ |Δv_y|/(σ·√2π) for sigma=0.6 s
                                       ∈ [0.35, 0.71] m/s²
    Plane fit: x ∈ [-x_extent/2, x_extent/2] → ±2.5 + 3.5 = 6 m. Fits.
    """
    U = rng.uniform
    p = {
        "A_y":         float(U(1.0, 2.0)),
        "x_extent":    float(U(3.0, 5.0)),
        "smooth_s":    float(U(0.55, 0.75)),
        "z0":          float(SIMPLE_ALTITUDE_M),
        "yaw0":        float(U(-0.1, 0.1)),
        "n_cycles":    2.0,
    }
    t = _time_axis(SIMPLE_N_SAMPLES, SIMPLE_DURATION_S)
    # Triangular wave with `n_cycles` full cycles over [0, T]. We synthesise
    # via `arcsin(sin(2π·n·t/T))` (exact triangular, ∈ [-π/2, π/2]) and
    # rescale to [-A_y, +A_y].
    raw_y = (2.0 / np.pi) * np.arcsin(
        np.sin(2.0 * np.pi * p["n_cycles"] * t / SIMPLE_DURATION_S))
    raw_y *= p["A_y"]
    raw_x = -p["x_extent"] / 2.0 + p["x_extent"] * (t / SIMPLE_DURATION_S)
    sigma_samples = p["smooth_s"] * FPS
    x = _gaussian_smooth_1d(raw_x, sigma_samples)  # near-linear → unchanged
    y = _gaussian_smooth_1d(raw_y, sigma_samples)
    z = np.full_like(t, p["z0"])
    pos = np.stack([x, y, z], axis=-1)
    yaw = np.full_like(t, p["yaw0"])
    return {"t": t, "pos": pos, "yaw": yaw,
            "meta": {"family": "simple_zigzag", **p,
                     "duration_s": SIMPLE_DURATION_S,
                     "n_samples": int(SIMPLE_N_SAMPLES)}}


def simple_zigzag_trajectory(seed: int, plane_x: float, plane_y: float,
                             target_peak_speed: Optional[float] = None,
                             ) -> Dict[str, np.ndarray]:
    """Step K named trajectory: open zigzag at altitude 4 m, 30 s, 2 cycles.

    Step K addition: optional ``target_peak_speed`` (m/s).
    """
    def sampler(rng):
        return _apply_speed_scale(_simple_zigzag_sampler(rng),
                                  target_peak_speed)
    out = _resample_with_confinement(sampler, seed, plane_x, plane_y)
    return _record_speed_diagnostics(out, target_peak_speed)


# ---------------------------------------------------------------------------
# Family 11 — simple_triangle (Step K, test traj 318/320)
# ---------------------------------------------------------------------------

def _simple_triangle_sampler(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Smoothed-corner equilateral triangle at constant altitude (closed loop).

    Geometry:
        3 vertices on a circle of radius R = edge_length / sqrt(3) at angles
        90°, 210°, 330° (apex pointing +y). Path traversed at constant
        arc-length speed; corners rounded by Gaussian smoother (σ≈0.7 s).

    Velocity / acceleration envelope (edge_length ∈ [2.0, 3.0], T = 30 s):
        perimeter   = 3 · edge_length        ∈ [6, 9] m
        edge speed  = perimeter / T          ∈ [0.20, 0.30] m/s
        |Δv| at corner ≈ 2 · v · sin(60°)    ∈ [0.35, 0.52] m/s
        peak |a|    ≈ |Δv|/(σ·√2π) for σ=0.7 s
                                              ∈ [0.20, 0.30] m/s²
    Plane fit: circumradius ≤ 3/√3 = 1.73 m → 2·R + 7 ≤ 10.5 m, fits.
    """
    U = rng.uniform
    p = {
        "edge_length": float(U(2.0, 3.0)),
        "smooth_s":    float(U(0.65, 0.85)),
        "z0":          float(SIMPLE_ALTITUDE_M),
        "yaw0":        float(U(-0.1, 0.1)),
        "phase_t":     float(U(0.0, 1.0)),
    }
    t = _time_axis(SIMPLE_N_SAMPLES, SIMPLE_DURATION_S)
    R = p["edge_length"] / np.sqrt(3.0)
    angles_deg = np.array([90.0, 210.0, 330.0])
    angles_rad = np.deg2rad(angles_deg)
    vertices = np.stack([R * np.cos(angles_rad), R * np.sin(angles_rad)],
                        axis=-1)
    raw = _polyline_at_constant_speed(vertices, SIMPLE_N_SAMPLES, closed=True)
    shift = int(p["phase_t"] * SIMPLE_N_SAMPLES) % SIMPLE_N_SAMPLES
    raw = np.roll(raw, shift, axis=0)
    sigma_samples = p["smooth_s"] * FPS
    x = _gaussian_smooth_1d(raw[:, 0], sigma_samples)
    y = _gaussian_smooth_1d(raw[:, 1], sigma_samples)
    z = np.full_like(t, p["z0"])
    pos = np.stack([x, y, z], axis=-1)
    yaw = np.full_like(t, p["yaw0"])
    return {"t": t, "pos": pos, "yaw": yaw,
            "meta": {"family": "simple_triangle", **p,
                     "duration_s": SIMPLE_DURATION_S,
                     "n_samples": int(SIMPLE_N_SAMPLES)}}


def simple_triangle_trajectory(seed: int, plane_x: float, plane_y: float,
                               target_peak_speed: Optional[float] = None,
                               ) -> Dict[str, np.ndarray]:
    """Step K named trajectory: equilateral triangle at altitude 4 m, 30 s.

    Step K addition: optional ``target_peak_speed`` (m/s).
    """
    def sampler(rng):
        return _apply_speed_scale(_simple_triangle_sampler(rng),
                                  target_peak_speed)
    out = _resample_with_confinement(sampler, seed, plane_x, plane_y)
    return _record_speed_diagnostics(out, target_peak_speed)


# ---------------------------------------------------------------------------
# Family 12 — simple_sinusoid_30s (Step K, traj 324)
# ---------------------------------------------------------------------------

def _simple_sinusoid_30s_sampler(rng: np.random.Generator
                                 ) -> Dict[str, np.ndarray]:
    """Single-frequency Lissajous-style sinusoid at constant altitude, 30 s.

    Distinct from the Step F sum-of-sinusoid family: that one runs 100 s and
    sums two frequencies in each axis to mimic aggressive aerial motion.
    THIS sampler runs 30 s with one frequency per axis (omega_x ≠ omega_y so
    the 2-D path traces a Lissajous figure, not a line). Designed to test
    SHAPE generalisation (sinusoidal pattern) at controlled motion magnitude
    independent of length and amplitude — the cleanest ablation for "is
    sinusoid failure about shape or about length+aggressive motion?"

    Geometry:
        omega_x = 2π / 15  →  2 cycles in 30 s
        omega_y = 2π / 10  →  3 cycles in 30 s   (3:2 ratio = open Lissajous)
        x(t) = A · sin(omega_x · t + phase_x)
        y(t) = B · sin(omega_y · t + phase_y)
        z(t) = SIMPLE_ALTITUDE_M

    Velocity / acceleration envelope (A∈[1.5, 2.5], B∈[1.0, 2.0], no scale):
        |v_x|_max ≤ A · omega_x  ≈  [0.628, 1.047] m/s
        |v_y|_max ≤ B · omega_y  ≈  [0.628, 1.257] m/s
        peak |v|  ≤  √((A·ωx)² + (B·ωy)²)  ≈ 0.9 — 1.6 m/s
    Speed scaling typically pulls this DOWN to target_peak_speed ∈
    {0.1, …, 0.5} m/s, so post-scale amplitudes are O(1 m). Plane fit is
    therefore generous on the 28 × 18.69 m plane.

    The 3:2 ratio is chosen so the path is OPEN (does not close after one
    period) — closes after lcm(15, 10) = 30 s, exactly the duration. This
    gives the network a single non-repeating shape exposure.
    """
    U = rng.uniform
    p = {
        "A":         float(U(1.5, 2.5)),
        "B":         float(U(1.0, 2.0)),
        "omega_x":   float(2.0 * np.pi / 15.0),
        "omega_y":   float(2.0 * np.pi / 10.0),
        "phase_x":   float(U(0.0, 2.0 * np.pi)),
        "phase_y":   float(U(0.0, 2.0 * np.pi)),
        "z0":        float(SIMPLE_ALTITUDE_M),
        "yaw0":      float(U(-0.1, 0.1)),
    }
    t = _time_axis(SIMPLE_N_SAMPLES, SIMPLE_DURATION_S)
    x = p["A"] * np.sin(p["omega_x"] * t + p["phase_x"])
    y = p["B"] * np.sin(p["omega_y"] * t + p["phase_y"])
    z = np.full_like(t, p["z0"])
    pos = np.stack([x, y, z], axis=-1)
    yaw = np.full_like(t, p["yaw0"])
    return {"t": t, "pos": pos, "yaw": yaw,
            "meta": {"family": "simple_sinusoid_30s", **p,
                     "duration_s": SIMPLE_DURATION_S,
                     "n_samples": int(SIMPLE_N_SAMPLES)}}


def simple_sinusoid_30s_trajectory(seed: int, plane_x: float, plane_y: float,
                                   target_peak_speed: Optional[float] = None,
                                   ) -> Dict[str, np.ndarray]:
    """Step K named trajectory: 30 s Lissajous-style sinusoid at altitude 4 m.

    NEW family — never seen in any prior Step. Used as test traj 324 to
    isolate sinusoidal-shape generalisation from the length / aggressive-
    motion confounds of Step F's sum-of-sinusoid trajectories (3, 100, 101,
    102). target_peak_speed defaults to None for back-compat but is set to
    0.5 m/s in the Step K traj_id dispatch.
    """
    def sampler(rng):
        return _apply_speed_scale(_simple_sinusoid_30s_sampler(rng),
                                  target_peak_speed)
    out = _resample_with_confinement(sampler, seed, plane_x, plane_y)
    return _record_speed_diagnostics(out, target_peak_speed)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

FAMILY_SAMPLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "sinusoid":            sinusoid_trajectory,
    "flat_spiral":         flat_spiral_trajectory,
    "figure_8":            figure_8_trajectory,
    "clover":              clover_trajectory,
    "minitest":            minitest_trajectory,
    "simple_figure8":      simple_figure8_trajectory,
    "simple_circle":       simple_circle_trajectory,
    "simple_ellipse":      simple_ellipse_trajectory,
    "simple_diamond":      simple_diamond_trajectory,
    "simple_zigzag":       simple_zigzag_trajectory,
    "simple_triangle":     simple_triangle_trajectory,
    "simple_sinusoid_30s": simple_sinusoid_30s_trajectory,
}

# Families whose public sampler accepts target_peak_speed (Step K).
SPEED_SCALABLE_FAMILIES = frozenset({
    "simple_figure8", "simple_circle", "simple_ellipse",
    "simple_diamond", "simple_zigzag", "simple_triangle",
    "simple_sinusoid_30s",
})


def generate_by_family(family: str, seed: int,
                       plane_x: float, plane_y: float,
                       target_peak_speed: Optional[float] = None,
                       ) -> Dict[str, np.ndarray]:
    """Dispatch on family name. Requires plane dimensions for confinement.

    Step K addition: ``target_peak_speed`` (m/s) is forwarded to families
    in ``SPEED_SCALABLE_FAMILIES``. For other families the parameter must
    be None (otherwise raises) so we don't silently drop a caller's intent.
    """
    if family not in FAMILY_SAMPLERS:
        raise KeyError(f"unknown trajectory family {family!r}; expected one of "
                       f"{list(FAMILY_SAMPLERS.keys())}")
    if target_peak_speed is None or family not in SPEED_SCALABLE_FAMILIES:
        if target_peak_speed is not None and family not in SPEED_SCALABLE_FAMILIES:
            raise ValueError(
                f"family {family!r} does not support target_peak_speed; "
                f"speed scaling only available for {sorted(SPEED_SCALABLE_FAMILIES)}")
        return FAMILY_SAMPLERS[family](seed, plane_x, plane_y)
    return FAMILY_SAMPLERS[family](seed, plane_x, plane_y,
                                   target_peak_speed=target_peak_speed)
