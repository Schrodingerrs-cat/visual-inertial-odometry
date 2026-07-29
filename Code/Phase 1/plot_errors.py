"""
Trajectory-error plots in the style of rpg_trajectory_evaluation.

For a given est CSV + EuRoC GT CSV, produces six figures:

  fig1_trans_error.png       translation drift (x/y/z) vs distance traveled
  fig2_rot_error.png         attitude error (yaw/pitch/roll) vs distance
  fig3_rel_trans_error.png   relative translation error boxplot over sub-traj lengths
  fig4_rel_yaw_error.png     relative yaw error boxplot over sub-traj lengths
  fig5_traj_side.png         trajectory side view (x, z)
  fig6_traj_top.png          trajectory top view (x, y)

Alignment: Umeyama SE(3) (no scale), matching evaluate.py / make_video.py.

Usage:
    python plot_errors.py --est ../../Results/Phase1/MH_01_easy_est.csv \
        --gt  ../../Data/machine_hall/MH_01_easy/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
        --out-dir ../../Results/Phase1/MH_01_easy_plots \
        [--title MH_01_easy]
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate import load_est, associate, umeyama


# ---------- loaders ----------

def load_est_full(path):
    """est CSV columns: timestamp, tx, ty, tz, qx, qy, qz, qw (JPL)."""
    ts, P, Q = [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ts.append(float(row[0]))
            P.append([float(row[1]), float(row[2]), float(row[3])])
            Q.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    return np.array(ts), np.array(P), np.array(Q)


def load_gt_full(path):
    """EuRoC GT CSV: [ns, px, py, pz, qw, qx, qy, qz, ...]."""
    ts, P, Q = [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ts.append(float(row[0]) * 1e-9)
            P.append([float(row[1]), float(row[2]), float(row[3])])
            # Hamilton body-to-world [w, x, y, z]
            Q.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    return np.array(ts), np.array(P), np.array(Q)


# ---------- rotations ----------

def jpl_to_R_bw(q_jpl):
    """Body-to-world rotation from the JPL quaternion saved by run_eval.py.

    run_eval.py writes `q = to_quaternion(R_bw)`, so `to_rotation(q) = R_bw`
    directly — no transpose. Formula matches `utils.to_rotation`.
    """
    x, y, z, w = q_jpl
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
        [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
    ])


def hamilton_to_R_bw(q_ham):
    """Hamilton [qw,qx,qy,qz] body-to-world -> rotation matrix."""
    w, x, y, z = q_ham
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_ypr(R):
    """ZYX intrinsic (yaw about Z, pitch about Y, roll about X), radians."""
    sy = np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
    if sy < 1e-9:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


# ---------- geometry helpers ----------

def cumulative_distance(P):
    """Piecewise arclength along an (N,3) position array."""
    d = np.zeros(len(P))
    d[1:] = np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    return d


def relative_errors_at_distance(P_gt, R_gt, P_est, R_est, cum, target_d,
                                stride=5, tol=0.1):
    """For every start i, find first j with cum[j]-cum[i] >= target_d.
    Return per-pair (||t_err||, |yaw_err|_deg) using RPE:
        T_err = inv(T_gt_ij) @ T_est_ij,  T_*_ij = inv(T_i) @ T_j.
    `tol` is the max allowed excess over target_d (fraction of d).
    """
    n = len(P_gt)
    trans_errs, yaw_errs = [], []
    max_d = target_d * (1.0 + tol)
    j_start = 0
    for i in range(0, n - 1, stride):
        d0 = cum[i]
        # advance j monotonically
        j = max(j_start, i + 1)
        while j < n and cum[j] - d0 < target_d:
            j += 1
        if j >= n:
            break
        if cum[j] - d0 > max_d:
            continue
        j_start = max(j_start, i + 1)  # j always ratchets forward per-i block

        # relative poses
        dR_gt = R_gt[i].T @ R_gt[j]
        dt_gt = R_gt[i].T @ (P_gt[j] - P_gt[i])
        dR_est = R_est[i].T @ R_est[j]
        dt_est = R_est[i].T @ (P_est[j] - P_est[i])

        # error pose = inv(gt) @ est
        R_err = dR_gt.T @ dR_est
        t_err = dR_gt.T @ (dt_est - dt_gt)

        yaw, _, _ = R_to_ypr(R_err)
        trans_errs.append(np.linalg.norm(t_err))
        yaw_errs.append(abs(np.degrees(yaw)))
    return np.array(trans_errs), np.array(yaw_errs)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--title', default='')
    ap.add_argument('--max-dt', type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    est_ts, est_P, est_Q = load_est_full(args.est)
    gt_ts, gt_P, gt_Q = load_gt_full(args.gt)

    pairs = associate(est_ts, gt_ts, max_dt=args.max_dt)
    if len(pairs) < 10:
        raise RuntimeError(f'too few matched pairs: {len(pairs)}')

    est_idx = np.array([i for i, _ in pairs])
    gt_idx = np.array([j for _, j in pairs])
    est_P_m = est_P[est_idx]
    gt_P_m = gt_P[gt_idx]
    est_Q_m = est_Q[est_idx]
    gt_Q_m = gt_Q[gt_idx]

    # Umeyama SE(3): gt ≈ R @ est + t  (no scale).
    s, R_align, t_align = umeyama(est_P_m, gt_P_m, with_scale=False)
    est_P_aligned = (R_align @ est_P_m.T).T + t_align

    # Aligned body-to-world rotations for est.
    est_R_bw = np.array([jpl_to_R_bw(q) for q in est_Q_m])
    est_R_bw_aligned = np.array([R_align @ R for R in est_R_bw])

    gt_R_bw = np.array([hamilton_to_R_bw(q) for q in gt_Q_m])

    # Cumulative distance along GT trajectory (in metres).
    cum = cumulative_distance(gt_P_m)
    total_len = float(cum[-1])

    # ---- Fig 1: translation drift in mm vs distance ----
    trans_err_mm = (est_P_aligned - gt_P_m) * 1000.0
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(cum, trans_err_mm[:, 0], color='red', linewidth=1.0, label='x')
    ax.plot(cum, trans_err_mm[:, 1], color='green', linewidth=1.0, label='y')
    ax.plot(cum, trans_err_mm[:, 2], color='blue', linewidth=1.0, label='z')
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Position Drift [mm]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    if args.title:
        ax.set_title(f'{args.title} — Translation Error')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig1_trans_error.png'), dpi=130)
    plt.close(fig)

    # ---- Fig 2: rotation error (yaw/pitch/roll) in deg vs distance ----
    ypr = np.zeros((len(cum), 3))
    for k in range(len(cum)):
        R_err = gt_R_bw[k].T @ est_R_bw_aligned[k]
        y, p, r = R_to_ypr(R_err)
        ypr[k] = [np.degrees(y), np.degrees(p), np.degrees(r)]

    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(cum, ypr[:, 0], color='red', linewidth=1.0, label='yaw')
    ax.plot(cum, ypr[:, 1], color='green', linewidth=1.0, label='pitch')
    ax.plot(cum, ypr[:, 2], color='blue', linewidth=1.0, label='roll')
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Orient. err. [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    if args.title:
        ax.set_title(f'{args.title} — Rotation Error')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig2_rot_error.png'), dpi=130)
    plt.close(fig)

    # ---- Figs 3 & 4: relative error boxplots over 5 sub-trajectory lengths ----
    fracs = [0.10, 0.20, 0.30, 0.40, 0.50]
    dists = [round(total_len * f, 2) for f in fracs]

    trans_box, yaw_box = [], []
    est_R_bw_aligned_arr = est_R_bw_aligned
    for d in dists:
        tr, yr = relative_errors_at_distance(
            gt_P_m, gt_R_bw, est_P_aligned, est_R_bw_aligned_arr,
            cum, target_d=d, stride=5, tol=0.1)
        trans_box.append(tr if len(tr) else np.array([0.0]))
        yaw_box.append(yr if len(yr) else np.array([0.0]))

    labels = [f'{d:.2f}' for d in dists]
    positions = np.arange(1, len(dists) + 1)

    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.boxplot(trans_box, positions=positions, widths=0.6,
               showfliers=False, patch_artist=False)
    ax.set_xticks(positions); ax.set_xticklabels(labels)
    ax.set_xlabel('Distance traveled [m]')
    ax.set_ylabel('Translation error [m]')
    ax.grid(True, alpha=0.3)
    ax.plot([], [], color='tab:blue', label='Estimate')
    ax.legend(loc='upper left', fontsize=9)
    if args.title:
        ax.set_title(f'{args.title} — Relative Translation Error')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig3_rel_trans_error.png'), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.boxplot(yaw_box, positions=positions, widths=0.6,
               showfliers=False, patch_artist=False)
    ax.set_xticks(positions); ax.set_xticklabels(labels)
    ax.set_xlabel('Distance traveled [m]')
    ax.set_ylabel('Yaw error [deg]')
    ax.grid(True, alpha=0.3)
    ax.plot([], [], color='tab:blue', label='Estimate')
    ax.legend(loc='upper left', fontsize=9)
    if args.title:
        ax.set_title(f'{args.title} — Relative Yaw Error')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig4_rel_yaw_error.png'), dpi=130)
    plt.close(fig)

    # ---- Fig 5: trajectory side view (x, z) ----
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(est_P_aligned[:, 0], est_P_aligned[:, 2],
            color='tab:blue', linewidth=1.3, label='Estimate')
    ax.plot(gt_P_m[:, 0], gt_P_m[:, 2],
            color='red', linewidth=1.3, label='Groundtruth')
    ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    if args.title:
        ax.set_title(f'{args.title} — Trajectory (side view)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig5_traj_side.png'), dpi=130)
    plt.close(fig)

    # ---- Fig 6: trajectory top view (x, y) ----
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(est_P_aligned[:, 0], est_P_aligned[:, 1],
            color='tab:blue', linewidth=1.3, label='Estimate')
    ax.plot(gt_P_m[:, 0], gt_P_m[:, 1],
            color='red', linewidth=1.3, label='Groundtruth')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    if args.title:
        ax.set_title(f'{args.title} — Trajectory (top view)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig6_traj_top.png'), dpi=130)
    plt.close(fig)

    # ---- Summary ----
    err_norm = np.linalg.norm(est_P_aligned - gt_P_m, axis=1)
    rmse = float(np.sqrt((err_norm ** 2).mean()))
    median = float(np.median(err_norm))

    print(f'wrote 6 figures to {args.out_dir}')
    print(f'  total trajectory length : {total_len:.2f} m')
    print(f'  ATE RMSE (SE3-aligned)  : {rmse:.4f} m')
    print(f'  ATE median              : {median:.4f} m')
    for d, tb in zip(dists, trans_box):
        print(f'  rel trans err @ {d:6.2f} m : median {np.median(tb):.3f} m, '
              f'n={len(tb)}')


if __name__ == '__main__':
    main()
