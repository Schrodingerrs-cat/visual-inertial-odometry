"""
Offline renderer: produce Output.mp4 from an est CSV + EuRoC GT CSV.

Computes Umeyama SE(3) alignment, transforms GT into the estimate frame,
then renders an animation with three panels:
  * top-down XY trajectory (GT red, estimate black)
  * altitude (Z) vs sample index
  * running ATE error per-sample

Uses cv2.VideoWriter so no external ffmpeg binary is required.

Usage:
    python make_video.py --est ../../Results/Phase1/MH_01_easy_est.csv \
        --gt /path/to/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
        --out ../Output.mp4 [--fps 30] [--frames 240] [--title MH_01_easy]
"""
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate import load_est, load_gt, associate, umeyama


def _fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', default='Output.mp4')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--title', default='')
    args = ap.parse_args()

    est_ts, est_P = load_est(args.est)
    gt_ts, gt_P = load_gt(args.gt)
    pairs = associate(est_ts, gt_ts, max_dt=0.02)
    if len(pairs) < 10:
        raise RuntimeError(f'too few pairs: {len(pairs)}')
    est_m = np.array([est_P[i] for i, _ in pairs])
    gt_m = np.array([gt_P[j] for _, j in pairs])

    s, R, t = umeyama(est_m, gt_m, with_scale=False)
    gt_in_filter = ((gt_P - t) @ R) / s
    gt_m_in_filter = ((gt_m - t) @ R) / s
    err = np.linalg.norm(est_m - gt_m_in_filter, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    mean_e = float(err.mean())
    max_e = float(err.max())

    n_est = len(est_P)
    idx = np.linspace(0, n_est - 1, args.frames).astype(int)

    fig, (axXY, axZ, axE) = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
    header = f'ATE RMSE = {rmse:.3f} m   mean = {mean_e:.3f} m   max = {max_e:.3f} m'
    if args.title:
        fig.suptitle(f'{args.title} — S-MSCKF\n{header}', fontsize=11)
    else:
        fig.suptitle(f'S-MSCKF\n{header}', fontsize=11)

    # full-track context
    axXY.plot(gt_in_filter[:, 0], gt_in_filter[:, 1],
              color='red', linewidth=1.0, alpha=0.35)
    axXY.plot(est_P[:, 0], est_P[:, 1], color=(0.4, 0.4, 0.4),
              linewidth=0.8, alpha=0.25)

    (gt_line,) = axXY.plot([], [], color='red',
                           linewidth=2.0, label='Ground truth')
    (est_line,) = axXY.plot([], [], color='black',
                            linewidth=1.4, label='MSCKF estimate')
    (head,) = axXY.plot([], [], 'bo', markersize=7)
    axXY.set_xlabel('X [m]'); axXY.set_ylabel('Y [m]')
    axXY.set_title('top-down trajectory')
    axXY.set_aspect('equal', adjustable='datalim')
    axXY.grid(True, alpha=0.3)
    axXY.legend(loc='upper right', fontsize=9)

    tsample = np.arange(len(est_P))
    axZ.plot(tsample, gt_in_filter[:, 2], color='red',
             linewidth=1.0, alpha=0.35)
    axZ.plot(tsample, est_P[:, 2], color=(0.4, 0.4, 0.4),
             linewidth=0.8, alpha=0.25)
    (gt_z,) = axZ.plot([], [], color='red', linewidth=1.8, label='GT Z')
    (est_z,) = axZ.plot([], [], color='black', linewidth=1.3, label='Est Z')
    axZ.set_xlabel('sample'); axZ.set_ylabel('Z [m]')
    axZ.set_title('altitude')
    axZ.grid(True, alpha=0.3); axZ.legend(loc='upper right', fontsize=9)

    # per-sample error, sampled at matched pairs
    err_ts = np.arange(len(err))
    axE.plot(err_ts, err, color=(0.85, 0.85, 0.85), linewidth=0.8, alpha=0.5)
    (err_line,) = axE.plot([], [], color='crimson', linewidth=1.4)
    axE.axhline(rmse, color='gray', linestyle='--', linewidth=1.0,
                label=f'RMSE {rmse:.3f}')
    axE.set_xlabel('matched sample'); axE.set_ylabel('||error|| [m]')
    axE.set_title('translation error over time')
    axE.grid(True, alpha=0.3); axE.legend(loc='upper right', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.canvas.draw()
    first = _fig_to_bgr(fig)
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open VideoWriter for {args.out}')

    # map estimate-index k to matched-error index via np.searchsorted on pair indices
    pair_est_i = np.array([i for i, _ in pairs])

    for k_i, k in enumerate(idx):
        est_k = est_P[:k + 1]
        gt_k = gt_in_filter[:k + 1]
        est_line.set_data(est_k[:, 0], est_k[:, 1])
        gt_line.set_data(gt_k[:, 0], gt_k[:, 1])
        head.set_data([est_P[k, 0]], [est_P[k, 1]])

        est_z.set_data(tsample[:k + 1], est_P[:k + 1, 2])
        gt_z.set_data(tsample[:k + 1], gt_in_filter[:k + 1, 2])

        ei = int(np.searchsorted(pair_est_i, k, side='right'))
        err_line.set_data(err_ts[:ei], err[:ei])

        writer.write(_fig_to_bgr(fig))

    writer.release()
    plt.close(fig)
    print(f'wrote {args.out}  ({len(idx)} frames @ {args.fps} fps)')
    print(f'  ATE RMSE   = {rmse:.4f} m')
    print(f'      mean   = {mean_e:.4f} m')
    print(f'      median = {float(np.median(err)):.4f} m')
    print(f'      max    = {max_e:.4f} m')


if __name__ == '__main__':
    main()
