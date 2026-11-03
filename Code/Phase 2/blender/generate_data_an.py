#!/usr/bin/env python3
"""Generate synthetic VIO dataset with circular trajectories.

Circular path: x(t)=R·cos(ωt), y(t)=R·sin(ωt), z(t)=h+A·sin(2ωt)
The sinusoidal height term ensures both roll AND pitch vary over the trajectory.
Attitude follows a coordinated-turn (quadrotor banking) model so that the
body-frame z-axis aligns with the thrust direction — giving natural roll from
centripetal acceleration and pitch from the vertical velocity component.

IMU noise is added via gnss-ins-sim's acc_gen / gyro_gen.

Usage
-----
python generate_data.py                        # defaults: 5 train + 2 test seqs
python generate_data.py --n-train 8 --n-test 3 --out-dir data
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ── gnss-ins-sim is one level up from V2/ ─────────────────────────────────────
_V2_ROOT = Path(__file__).resolve().parent
_GNSS_SIM = _V2_ROOT.parent / "gnss-ins-sim"
if str(_GNSS_SIM) not in sys.path:
    sys.path.insert(0, str(_GNSS_SIM))

from gnss_ins_sim.sim import imu_model      # noqa: E402
from gnss_ins_sim.pathgen import pathgen    # noqa: E402

G = 9.80665          # m/s²
MAX_BANK_DEG = 45.0  # hard limit per assignment


# ── pure rotation helpers ──────────────────────────────────────────────────────

def _rot_body_to_world_coordinated(vel: np.ndarray, acc_world: np.ndarray,
                                   eps: float = 1e-9) -> np.ndarray:
    """Quadrotor coordinated-turn attitude: body-x along velocity, body-z
    opposes net thrust (gravity + linear accel), body-y completes the frame.

    Parameters
    ----------
    vel, acc_world : (N, 3) world-frame arrays
    Returns
    -------
    rot : (N, 3, 3) rotation matrices R such that v_world = R @ v_body
    """
    g_world = np.array([0., 0., -G])
    thrust = acc_world - g_world                           # net non-gravity accel
    tn = np.linalg.norm(thrust, axis=1, keepdims=True)
    z_ax = -np.where(tn > eps, thrust / np.maximum(tn, eps),
                     np.broadcast_to(np.array([[0., 0., 1.]]), thrust.shape))

    vn = np.linalg.norm(vel, axis=1, keepdims=True)
    x_ax = np.where(vn > eps, vel / np.maximum(vn, eps),
                    np.broadcast_to(np.array([[1., 0., 0.]]), vel.shape))

    y_ax = np.cross(z_ax, x_ax)
    yn = np.linalg.norm(y_ax, axis=1, keepdims=True)
    y_ax = np.where(yn > eps, y_ax / np.maximum(yn, eps),
                    np.broadcast_to(np.array([[0., 1., 0.]]), y_ax.shape))

    x_ax = np.cross(y_ax, z_ax)   # reorthogonalise
    xn = np.linalg.norm(x_ax, axis=1, keepdims=True)
    x_ax = np.where(xn > eps, x_ax / np.maximum(xn, eps), x_ax)

    # columns are body axes expressed in world: [x_body | y_body | z_body]
    return np.stack([x_ax, y_ax, z_ax], axis=-1)  # (N, 3, 3)


def _gyro_from_rotation_chain(rot: np.ndarray, hz: float) -> np.ndarray:
    """Body-frame angular velocity from finite-difference of rotation matrices."""
    dt = 1.0 / hz
    rot_dot = np.empty_like(rot)
    rot_dot[1:-1] = (rot[2:] - rot[:-2]) / (2.0 * dt)
    rot_dot[0] = (rot[1] - rot[0]) / dt
    rot_dot[-1] = (rot[-1] - rot[-2]) / dt
    # ω_skew = R^T · Ṙ  →  extract [ω_x, ω_y, ω_z] from skew-symmetric matrix
    omega_skew = np.einsum("nij,njk->nik", np.transpose(rot, (0, 2, 1)), rot_dot)
    return np.column_stack([omega_skew[:, 2, 1],
                             omega_skew[:, 0, 2],
                             omega_skew[:, 1, 0]])


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Single 3×3 rotation matrix → quaternion (w, x, y, z)."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                          0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _rotmats_to_quats(rots: np.ndarray) -> np.ndarray:
    return np.array([_rotmat_to_quat_wxyz(R) for R in rots])


def _rotmat_to_ypr(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles (yaw, pitch, roll) from rotation matrix."""
    pitch = math.atan2(-float(R[2, 0]),
                       math.hypot(float(R[2, 1]), float(R[2, 2])))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return yaw, pitch, roll


# ── trajectory generation ──────────────────────────────────────────────────────

def generate_sequence(
    radius_m: float = 8.0,
    frequency_hz: float = 0.10,
    altitude_m: float = 8.0,
    height_amp_m: float = 0.50,
    duration_s: float = 60.0,
    imu_hz: float = 1000.0,
    camera_stride: int = 10,
    imu_accuracy: str = "mid-accuracy",
    seed: int = 0,
) -> dict:
    """Generate one circular VIO sequence.

    The height oscillation z(t)=h+A·sin(2ωt) ensures the velocity vector has
    a non-zero z-component, which through the coordinated-turn model produces
    pitch variation in addition to the natural roll from centripetal banking.

    Returns
    -------
    dict with keys: time, position, velocity, acceleration_world,
        specific_force, gyro_ideal, accel_meas, gyro_meas, yaw_pitch_roll,
        quat_wxyz, rot_body_to_world, camera_indices, relative_poses,
        max_bank_deg.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * imu_hz)) + 1
    t = np.arange(n, dtype=float) / imu_hz

    omega = 2.0 * math.pi * frequency_hz
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    theta = omega * t + phase

    # ── kinematics ────────────────────────────────────────────────────────────
    pos = np.column_stack([
        radius_m * np.cos(theta),
        radius_m * np.sin(theta),
        altitude_m + height_amp_m * np.sin(2.0 * omega * t),
    ])
    vel = np.column_stack([
        -radius_m * omega * np.sin(theta),
         radius_m * omega * np.cos(theta),
         2.0 * height_amp_m * omega * np.cos(2.0 * omega * t),
    ])
    acc_world = np.column_stack([
        -radius_m * omega ** 2 * np.cos(theta),
        -radius_m * omega ** 2 * np.sin(theta),
        -4.0 * height_amp_m * omega ** 2 * np.sin(2.0 * omega * t),
    ])

    # Check bank angle limit (centripetal vs gravity)
    a_centripetal = np.linalg.norm(acc_world[:, :2], axis=1)
    max_bank_rad = float(np.max(np.arctan2(a_centripetal, G)))
    if max_bank_rad > math.radians(MAX_BANK_DEG):
        raise ValueError(
            f"Bank angle {math.degrees(max_bank_rad):.1f}° exceeds {MAX_BANK_DEG}°. "
            "Reduce radius or frequency."
        )

    # ── attitude (coordinated turn → natural roll + pitch) ───────────────────
    rot = _rot_body_to_world_coordinated(vel, acc_world)  # (N, 3, 3)

    # ── ideal IMU ─────────────────────────────────────────────────────────────
    g_world = np.array([0.0, 0.0, -G])
    specific_force = np.einsum(
        "nij,nj->ni", np.transpose(rot, (0, 2, 1)), acc_world - g_world
    )
    gyro_ideal = _gyro_from_rotation_chain(rot, imu_hz)

    # ── gnss-ins-sim noise ────────────────────────────────────────────────────
    np.random.seed(seed)
    imu_err = imu_model.IMU(accuracy=imu_accuracy, axis=6, gps=False)
    accel_meas = pathgen.acc_gen(imu_hz, specific_force, imu_err.accel_err)
    gyro_meas = pathgen.gyro_gen(imu_hz, gyro_ideal, imu_err.gyro_err)

    # ── derived quantities ────────────────────────────────────────────────────
    ypr_arr = np.array([_rotmat_to_ypr(rot[i]) for i in range(n)])  # (N,3)
    quat = _rotmats_to_quats(rot)  # (N, 4) as (w,x,y,z)

    # ── camera frames & relative poses ───────────────────────────────────────
    cam_idx = np.arange(0, n, camera_stride, dtype=int)
    rel_poses = []
    for k, (i, j) in enumerate(zip(cam_idx[:-1], cam_idx[1:])):
        Ri, Rj = rot[i], rot[j]
        R_rel = Ri.T @ Rj
        t_rel = Ri.T @ (pos[j] - pos[i])
        q_rel = _rotmat_to_quat_wxyz(R_rel)
        rel_poses.append([k, t[i], t[j], int(i), int(j),
                          t_rel[0], t_rel[1], t_rel[2],
                          q_rel[0], q_rel[1], q_rel[2], q_rel[3]])

    return {
        "time": t,
        "position": pos,
        "velocity": vel,
        "acceleration_world": acc_world,
        "specific_force": specific_force,
        "gyro_ideal": gyro_ideal,
        "accel_meas": accel_meas,
        "gyro_meas": gyro_meas,
        "yaw_pitch_roll": ypr_arr,
        "quat_wxyz": quat,
        "rot_body_to_world": rot,
        "camera_indices": cam_idx,
        "relative_poses": np.array(rel_poses, dtype=float),
        "max_bank_deg": math.degrees(max_bank_rad),
    }


def save_sequence(out_dir: Path, seq: dict, params: dict) -> None:
    """Write all CSVs, the NPZ archive, and metadata to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)  # placeholder; filled by Blender

    t = seq["time"]
    pos = seq["position"]
    vel = seq["velocity"]
    acc = seq["acceleration_world"]
    sf = seq["specific_force"]
    gyro_ref = seq["gyro_ideal"]
    accel = seq["accel_meas"]
    gyro = seq["gyro_meas"]
    ypr = seq["yaw_pitch_roll"]
    quat = seq["quat_wxyz"]
    cam_idx = seq["camera_indices"]

    # trajectory.csv — full ground-truth trajectory at IMU rate
    np.savetxt(
        out_dir / "trajectory.csv",
        np.column_stack([t, pos, vel, acc, ypr, quat]),
        delimiter=",",
        header="time_s,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,"
               "acc_x,acc_y,acc_z,yaw_rad,pitch_rad,roll_rad,"
               "quat_w,quat_x,quat_y,quat_z",
        comments="", fmt="%.10f",
    )

    # imu.csv — noisy measurements at IMU rate
    np.savetxt(
        out_dir / "imu.csv",
        np.column_stack([t, accel, gyro, sf, gyro_ref]),
        delimiter=",",
        header="time_s,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,"
               "ref_acc_x,ref_acc_y,ref_acc_z,ref_gyro_x,ref_gyro_y,ref_gyro_z",
        comments="", fmt="%.10f",
    )

    # camera_poses.csv — pose at each camera frame
    np.savetxt(
        out_dir / "camera_poses.csv",
        np.column_stack([
            np.arange(len(cam_idx)), t[cam_idx],
            pos[cam_idx], ypr[cam_idx], quat[cam_idx],
        ]),
        delimiter=",",
        header="frame_idx,time_s,pos_x,pos_y,pos_z,"
               "yaw_rad,pitch_rad,roll_rad,quat_w,quat_x,quat_y,quat_z",
        comments="", fmt="%.10f",
    )

    # relative_poses.csv — relative pose between consecutive camera frames
    np.savetxt(
        out_dir / "relative_poses.csv",
        seq["relative_poses"],
        delimiter=",",
        header="pair_idx,t0_s,t1_s,imu_idx_a,imu_idx_b,"
               "tx,ty,tz,qw,qx,qy,qz",
        comments="", fmt="%.10f",
    )

    # dataset.npz — everything in one file for fast loading
    np.savez_compressed(
        out_dir / "dataset.npz",
        time=t, position=pos, velocity=vel,
        acceleration_world=acc, specific_force=sf,
        accel_meas=accel, gyro_meas=gyro,
        gyro_ideal=gyro_ref, yaw_pitch_roll=ypr,
        quat_wxyz=quat, rot_body_to_world=seq["rot_body_to_world"],
        camera_indices=cam_idx,
        relative_poses=seq["relative_poses"],
    )

    meta = {
        **params,
        "max_bank_deg": float(seq["max_bank_deg"]),
        "n_imu_samples": int(len(t)),
        "n_camera_frames": int(len(cam_idx)),
        "n_pairs": int(len(seq["relative_poses"])),
        "coordinate_frame": (
            "world: x/y horizontal, z-up; "
            "body: x-forward, y-right, z-down (coordinated-turn quadrotor)"
        ),
        "relative_pose_convention": "R_rel = R_a^T @ R_b,  t_rel = R_a^T (p_b - p_a)",
        "quaternion_convention": "(w, x, y, z)",
        "imu_frame": "body frame — specific force (not raw accel)",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  → {out_dir}  "
          f"(bank={seq['max_bank_deg']:.1f}°, "
          f"{len(cam_idx)} frames, {len(t)} IMU samples)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _random_params(seed: int) -> dict:
    r = np.random.default_rng(seed)
    return {
        "radius_m":     float(r.uniform(4.0, 14.0)),
        "frequency_hz": float(r.uniform(0.05, 0.15)),
        "altitude_m":   float(r.uniform(6.0, 12.0)),
        "height_amp_m": float(r.uniform(0.3, 1.2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",      type=Path, default=Path("data"))
    parser.add_argument("--n-train",      type=int,  default=5)
    parser.add_argument("--n-test",       type=int,  default=2)
    parser.add_argument("--duration",     type=float, default=60.0)
    parser.add_argument("--imu-hz",       type=float, default=1000.0)
    parser.add_argument("--camera-stride", type=int,  default=10)
    parser.add_argument("--imu-accuracy",
                        choices=["low-accuracy", "mid-accuracy", "high-accuracy"],
                        default="mid-accuracy")
    parser.add_argument("--seed",         type=int,  default=42)
    args = parser.parse_args()

    print(f"Generating {args.n_train} train + {args.n_test} test sequences "
          f"→ {args.out_dir.resolve()}")

    for i in range(args.n_train):
        seed = args.seed + i
        params = _random_params(seed)
        params.update({"split": "train", "seq_id": i,
                       "imu_accuracy": args.imu_accuracy,
                       "duration_s": args.duration,
                       "imu_hz": args.imu_hz,
                       "camera_stride": args.camera_stride})
        print(f"[train {i:04d}] R={params['radius_m']:.1f}m  "
              f"f={params['frequency_hz']:.3f}Hz  "
              f"h={params['altitude_m']:.1f}m  "
              f"A_z={params['height_amp_m']:.2f}m")
        seq = generate_sequence(
            radius_m=params["radius_m"],
            frequency_hz=params["frequency_hz"],
            altitude_m=params["altitude_m"],
            height_amp_m=params["height_amp_m"],
            duration_s=args.duration,
            imu_hz=args.imu_hz,
            camera_stride=args.camera_stride,
            imu_accuracy=args.imu_accuracy,
            seed=seed,
        )
        save_sequence(args.out_dir / f"seq_{i:04d}", seq, params)

    for j in range(args.n_test):
        seq_id = 100 + j
        seed = args.seed + 1000 + j
        params = _random_params(seed)
        params.update({"split": "test", "seq_id": seq_id,
                       "imu_accuracy": args.imu_accuracy,
                       "duration_s": args.duration,
                       "imu_hz": args.imu_hz,
                       "camera_stride": args.camera_stride})
        print(f"[test  {seq_id:04d}] R={params['radius_m']:.1f}m  "
              f"f={params['frequency_hz']:.3f}Hz  "
              f"h={params['altitude_m']:.1f}m  "
              f"A_z={params['height_amp_m']:.2f}m")
        seq = generate_sequence(
            radius_m=params["radius_m"],
            frequency_hz=params["frequency_hz"],
            altitude_m=params["altitude_m"],
            height_amp_m=params["height_amp_m"],
            duration_s=args.duration,
            imu_hz=args.imu_hz,
            camera_stride=args.camera_stride,
            imu_accuracy=args.imu_accuracy,
            seed=seed,
        )
        save_sequence(args.out_dir / f"seq_{seq_id:04d}", seq, params)

    print("Done.")


if __name__ == "__main__":
    main()
