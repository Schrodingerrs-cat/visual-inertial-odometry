"""
Trajectory evaluation: ATE RMSE/mean/median/max with Umeyama SE(3) alignment.

Usage:
    python evaluate.py --est <est.csv> --gt <gt.csv> [--plot <out.png>] [--no-scale]

est CSV:  timestamp,tx,ty,tz,qx,qy,qz,qw      (body-in-world, JPL quaternion)
gt CSV:   EuRoC state_groundtruth_estimate0/data.csv
          (timestamp[ns], px,py,pz, qw,qx,qy,qz, vx,vy,vz, bwx,bwy,bwz, bax,bay,baz)
"""
import argparse
import csv
import os
import numpy as np


def load_est(path):
    ts, P = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ts.append(float(row[0]))
            P.append([float(row[1]), float(row[2]), float(row[3])])
    return np.array(ts), np.array(P)


def load_gt(path):
    ts, P = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ts.append(float(row[0]) * 1e-9)
            P.append([float(row[1]), float(row[2]), float(row[3])])
    return np.array(ts), np.array(P)


def associate(est_ts, gt_ts, max_dt=0.02):
    """Nearest-timestamp association. Returns matching index pairs."""
    gt_sorted = np.argsort(gt_ts)
    gt_ts_sorted = gt_ts[gt_sorted]
    pairs = []
    for i, t in enumerate(est_ts):
        j = np.searchsorted(gt_ts_sorted, t)
        candidates = []
        if j < len(gt_ts_sorted):
            candidates.append(j)
        if j > 0:
            candidates.append(j - 1)
        best = min(candidates, key=lambda k: abs(gt_ts_sorted[k] - t))
        if abs(gt_ts_sorted[best] - t) <= max_dt:
            pairs.append((i, gt_sorted[best]))
    return pairs


def umeyama(src, dst, with_scale=True):
    """Umeyama SE(3) (or Sim(3) if with_scale). src,dst: (N,3). Returns s,R,t."""
    assert src.shape == dst.shape
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = (D * np.diag(S)).sum() / var_src
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


def ate(est, gt, align_scale=False):
    s, R, t = umeyama(est, gt, with_scale=align_scale)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    return {
        'rmse': float(np.sqrt((err ** 2).mean())),
        'mean': float(err.mean()),
        'median': float(np.median(err)),
        'max': float(err.max()),
        'min': float(err.min()),
        'n': int(len(err)),
        'scale': float(s),
    }, aligned


def plot(aligned_est, gt, out_png, title=''):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(gt[:, 0], gt[:, 1], 'r-', label='GT', linewidth=1)
    ax[0].plot(aligned_est[:, 0], aligned_est[:, 1], 'b-', label='MSCKF', linewidth=1)
    ax[0].set_xlabel('x [m]'); ax[0].set_ylabel('y [m]')
    ax[0].set_title(f'{title} (top-down)')
    ax[0].axis('equal'); ax[0].legend(); ax[0].grid(True)

    t = np.arange(len(gt))
    ax[1].plot(t, gt[:, 2], 'r-', label='GT z', linewidth=1)
    ax[1].plot(t, aligned_est[:, 2], 'b-', label='MSCKF z', linewidth=1)
    ax[1].set_xlabel('sample'); ax[1].set_ylabel('z [m]')
    ax[1].set_title('altitude'); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def evaluate(est_csv, gt_csv, plot_png=None, align_scale=False, title=''):
    est_ts, est_P = load_est(est_csv)
    gt_ts, gt_P = load_gt(gt_csv)
    pairs = associate(est_ts, gt_ts, max_dt=0.02)
    if len(pairs) < 10:
        raise RuntimeError(f'too few pairs: {len(pairs)}')
    est_m = np.array([est_P[i] for i, _ in pairs])
    gt_m = np.array([gt_P[j] for _, j in pairs])
    stats, aligned = ate(est_m, gt_m, align_scale=align_scale)
    if plot_png:
        plot(aligned, gt_m, plot_png, title=title)
    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--plot', default=None)
    ap.add_argument('--scale', action='store_true',
                    help='Allow scale in alignment (Sim3 instead of SE3)')
    ap.add_argument('--title', default='')
    args = ap.parse_args()
    s = evaluate(args.est, args.gt, args.plot, align_scale=args.scale, title=args.title)
    print(f'ATE RMSE  : {s["rmse"]:.4f} m')
    print(f'    mean  : {s["mean"]:.4f} m')
    print(f'    median: {s["median"]:.4f} m')
    print(f'    max   : {s["max"]:.4f} m')
    print(f'    n     : {s["n"]}  scale={s["scale"]:.4f}')
