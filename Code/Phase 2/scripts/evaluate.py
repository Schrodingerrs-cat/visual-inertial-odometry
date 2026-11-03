"""Evaluate V / I / VI (+ ablation variants) on the test split.

Workflow per method and per test trajectory:

    1. Load the trained checkpoint.
    2. Iterate consecutive frame-pairs in the trajectory.
    3. Predict relative pose (R, t) for each pair.
    4. Dead-reckon starting from the GT pose of frame 0.
    5. Write TUM trajectories for GT and prediction; compute aligned ATE via
       evo_ape, per-2-second window drift median/p95, and endpoint drift.
    6. Render: 3D overlay, per-axis XYZ vs time, per-axis RPY vs time,
       2D top-down (XY) projection.
    7. Append long-form rows to `outputs/results.csv` as
       (method, traj_id, metric_name, value).

Usage:
    python scripts/evaluate.py --config configs/default.yaml \\
        --checkpoints_dir outputs/ --split test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from models.network_i import NetworkI  # noqa: E402
from models.network_v import NetworkV  # noqa: E402
from models.network_vi import (  # noqa: E402
    NetworkVI, NetworkVIConcat, NetworkVINoDecomp,
)
from utils.dataloader import build_eval_loader  # noqa: E402
from utils.eval import (  # noqa: E402
    dead_reckon_from_gt_start, gt_trajectory_frames, matrix_to_rpy,
    plot_3d_overlay, plot_per_axis, run_evo_ape, segment_drift, write_tum,
)
from utils.geometry import quat_to_matrix  # noqa: E402
from utils.training import get_device, load_checkpoint, load_config, set_seed  # noqa: E402

# Optional: VI-concat-decompose variant is only present if models expose it.
try:
    from models.network_vi import NetworkVIConcatDecompose  # noqa: E402
    _HAVE_CONCAT_DECOMPOSE = True
except ImportError:
    NetworkVIConcatDecompose = None  # type: ignore
    _HAVE_CONCAT_DECOMPOSE = False


METHODS = ("v", "i", "vi", "vi_concat", "vi_no_decomp", "vi_concat_decompose")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoints_dir", type=str, default="outputs")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "test"])
    p.add_argument("--methods", type=str, nargs="+", default=list(METHODS),
                   help="Subset of methods to evaluate.")
    p.add_argument("--skip_missing", action="store_true",
                   help="Skip a method if its checkpoint does not exist.")
    # Step S2: per-trajectory override so the orchestrator can evaluate the
    # val trajectory (e.g. 305) without adding a `val_ids` config key. When
    # supplied, overrides the split-based ID lookup below.
    p.add_argument("--traj_ids", type=int, nargs="+", default=None,
                   help="Override split IDs with an explicit list "
                        "(e.g. --traj_ids 305).")
    return p.parse_args()


def _build_model(method: str, cfg: dict, device: torch.device):
    if method == "v":
        return NetworkV(feat_dim=cfg["model"]["cnn_feat_dim"]).to(device)
    if method == "i":
        return NetworkI(hidden_dim=cfg["model"]["lstm_hidden"]).to(device)
    fb = float(cfg.get("eval", {}).get("fallback_threshold", 0.05))
    if method == "vi":
        return NetworkVI(feat_dim=cfg["model"]["cnn_feat_dim"],
                         lstm_hidden=cfg["model"]["lstm_hidden"],
                         fallback_threshold=fb).to(device)
    if method == "vi_concat":
        return NetworkVIConcat(feat_dim=cfg["model"]["cnn_feat_dim"],
                               lstm_hidden=cfg["model"]["lstm_hidden"]).to(device)
    if method == "vi_no_decomp":
        return NetworkVINoDecomp(feat_dim=cfg["model"]["cnn_feat_dim"],
                                 lstm_hidden=cfg["model"]["lstm_hidden"]).to(device)
    if method == "vi_concat_decompose":
        if not _HAVE_CONCAT_DECOMPOSE:
            raise ImportError("NetworkVIConcatDecompose not available; "
                              "add it to models/network_vi.py")
        return NetworkVIConcatDecompose(
            feat_dim=cfg["model"]["cnn_feat_dim"],
            lstm_hidden=cfg["model"]["lstm_hidden"]).to(device)
    raise ValueError(f"unknown method {method}")


def _predict(method: str, model: torch.nn.Module, batch: dict
            ) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return (R, t, fell_back_to_imu).

    For method='v', translation uses the dataloader's per-frame
    batch["depth"] (= world-frame z at frame_a, i.e. poses_full[a, 2]),
    not metadata's scalar altitude_initial. Matches the S5 training
    supervision and removes the 9–17% altitude-variance bias at eval.
    """
    fell_back = False
    if method == "v":
        R, td, n, _f = model(batch["img_a"], batch["img_b"])
        # batch["depth"] is (B,); reshape to (B, 1) to broadcast over td.
        t = td * batch["depth"].view(-1, 1)
    elif method == "i":
        R, t, _f = model(batch["imu"])
    elif method in ("vi", "vi_no_decomp", "vi_concat", "vi_concat_decompose"):
        preds = model(batch["img_a"], batch["img_b"], batch["imu"])
        R, t = preds["R"], preds["t"]
        # Tier-1 B.3: only the full VI network attaches this flag.
        if "use_imu_fallback" in preds:
            flag = preds["use_imu_fallback"]
            if bool(flag[0].item()):
                R = preds["R_i"]
                t = preds["t_i"]
                fell_back = True
    else:
        raise ValueError(method)
    return R[0], t[0], fell_back


def _plot_2d_topdown(gt_ts: np.ndarray, pred_ts: np.ndarray, path: Path,
                     title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(gt_ts[:, 0], gt_ts[:, 1], color="red", lw=1.5, label="GT")
    ax.plot(pred_ts[:, 0], pred_ts[:, 1], color="blue", lw=1.0, label="pred")
    ax.scatter([gt_ts[0, 0]], [gt_ts[0, 1]], color="black", marker="o", s=30,
               zorder=5, label="start")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def _endpoint_drift(gt_ts: np.ndarray, pred_ts: np.ndarray) -> float:
    return float(np.linalg.norm(pred_ts[-1] - gt_ts[-1]))


def _write_row(writer: csv.writer, method: str, tid: int,
               metrics: dict) -> None:
    for name, value in metrics.items():
        writer.writerow([method, tid, name, f"{float(value):.6f}"])


def _evaluate_method(method: str, cfg: dict, device: torch.device,
                     ckpt_path: Path, test_ids, writer: csv.writer) -> None:
    model = _build_model(method, cfg, device)
    blob = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()

    traj_dir = Path(cfg["paths"]["trajectories_dir"])
    plot_dir = Path(cfg["paths"]["plots_dir"])
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for tid in test_ids:
        loader, ds = build_eval_loader(
            raw_dir=Path(cfg["data"]["raw_dir"]),
            traj_ids=[tid], img_size=tuple(cfg["data"]["img_size"]),
            num_workers=0)
        traj = list(ds._trajectories.values())[0]
        rels: list[tuple[np.ndarray, np.ndarray]] = []
        fallback_count = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                R, t, fell_back = _predict(method, model, batch)
                if fell_back:
                    fallback_count += 1
                rels.append((R.cpu().numpy(), t.cpu().numpy()))
        fallback_fraction = fallback_count / max(1, len(rels))

        times, pred_ts, pred_q = dead_reckon_from_gt_start(traj, rels)
        gt_times, gt_ts, gt_q = gt_trajectory_frames(traj)

        N = min(len(times), len(gt_times))
        times = times[:N]; pred_ts = pred_ts[:N]; pred_q = pred_q[:N]
        gt_ts = gt_ts[:N]; gt_q = gt_q[:N]; gt_times = gt_times[:N]

        pred_tum = traj_dir / f"{method}_{tid}.tum"
        gt_tum = traj_dir / f"gt_{tid}.tum"
        write_tum(gt_tum, gt_times, gt_ts, gt_q)
        write_tum(pred_tum, times, pred_ts, pred_q)

        ate = run_evo_ape(gt_tum, pred_tum, evo_bin=cfg["eval"]["evo_bin"])
        med, p95, _arr = segment_drift(gt_ts, pred_ts, times,
                                       window_s=cfg["eval"]["segment_seconds"])
        endpoint = _endpoint_drift(gt_ts, pred_ts)

        plot_3d_overlay(gt_ts, pred_ts,
                        plot_dir / f"{method}_{tid}_3d.png",
                        title=f"{method} — traj {tid} (3D overlay)")
        plot_per_axis(times, gt_ts, pred_ts,
                      plot_dir / f"{method}_{tid}_xyz.png",
                      ylabels=("x (m)", "y (m)", "z (m)"),
                      title=f"{method} — traj {tid} (position vs time)")
        gt_R = quat_to_matrix(torch.from_numpy(gt_q)).numpy()
        pr_R = quat_to_matrix(torch.from_numpy(pred_q)).numpy()
        gt_rpy = np.degrees(matrix_to_rpy(gt_R))
        pr_rpy = np.degrees(matrix_to_rpy(pr_R))
        plot_per_axis(times, gt_rpy, pr_rpy,
                      plot_dir / f"{method}_{tid}_rpy.png",
                      ylabels=("roll (deg)", "pitch (deg)", "yaw (deg)"),
                      title=f"{method} — traj {tid} (rpy vs time)")
        _plot_2d_topdown(gt_ts, pred_ts,
                         plot_dir / f"{method}_{tid}_topdown.png",
                         title=f"{method} — traj {tid} (top-down XY)")

        _write_row(writer, method, tid, {
            "ate_aligned": ate.ate_aligned,
            "ate_endpoint": endpoint,
            "ate_endpoint_evo": ate.ate_endpoint,
            "seg2s_median": med,
            "seg2s_p95": p95,
            "fallback_fraction": fallback_fraction,
        })
        print(f"[eval] {method:<22} traj {tid}: "
              f"ATE-aligned={ate.ate_aligned:.3f} m, "
              f"endpoint={endpoint:.3f} m, "
              f"seg2s(med/p95)={med:.3f}/{p95:.3f} m, "
              f"fallback={fallback_fraction:.1%}")


def _ckpt_path(method: str, checkpoints_dir: Path) -> Path:
    sub = {
        "v": "v", "i": "i", "vi": "vi",
        "vi_concat": "vi_concat",
        "vi_no_decomp": "vi_no_decomp",
        "vi_concat_decompose": "vi_concat_decompose",
    }[method]
    return checkpoints_dir / sub / "best.pt"


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    # Step S2: --traj_ids takes precedence over the split-based lookup.
    split_ids = (list(args.traj_ids) if args.traj_ids
                 else cfg["data"][f"{args.split}_ids"])
    results_csv = Path(cfg["paths"]["outputs_dir"]) / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["method", "traj_id", "metric_name", "value"])
        ckpt_root = Path(args.checkpoints_dir)
        for method in args.methods:
            ckpt = _ckpt_path(method, ckpt_root)
            if not ckpt.exists():
                msg = f"[eval] checkpoint missing for {method}: {ckpt}"
                if args.skip_missing:
                    print(msg + " — skipping."); continue
                else:
                    raise FileNotFoundError(msg)
            _evaluate_method(method, cfg, device, ckpt, split_ids, writer)
    print(f"[eval] wrote results to {results_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
