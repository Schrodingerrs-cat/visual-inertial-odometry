"""Evaluate V (t-direct) / I / VI (anchored) on the test split — Step I.

Mirrors ``scripts/evaluate_tdirect.py`` but binds ``vi`` to
``NetworkVI_anchored``. The other two methods are unchanged:
  * ``v``  → ``NetworkV_tdirect`` (loaded from outputs/step_i/v, symlinked
              to outputs/step_h/v).
  * ``i``  → ``NetworkI``         (loaded from outputs/step_i/i, symlinked
              to outputs/step_g/i).

For ``vi`` the model returns a dict with the gate signals (``rho_v``,
``alpha_t``, ``alpha_R``) in addition to the fused ``(R, t)``. Per-frame
gate values are accumulated across the whole trajectory and summarised in
``outputs/step_i/eval/gate_stats.json`` so we can interpret what the gate is
actually doing — see ``outputs/step_i/CHANGES.md`` for the table that
consumes those numbers.

Asserts ``cfg.model.variant == 'anchored'`` for safety.

Usage:
    python scripts/evaluate_anchored.py --config configs/step_i.yaml \\
        --checkpoints_dir outputs/step_i --split test \\
        --methods v i vi --skip_missing
"""
from __future__ import annotations

import argparse
import csv
import json
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
from models.network_v_tdirect import NetworkV_tdirect  # noqa: E402
from models.network_vi_anchored import NetworkVI_anchored  # noqa: E402
from utils.dataloader import build_eval_loader  # noqa: E402
from utils.eval import (  # noqa: E402
    dead_reckon_from_gt_start, gt_trajectory_frames, matrix_to_rpy,
    plot_3d_overlay, plot_per_axis, run_evo_ape, segment_drift, write_tum,
)
from utils.geometry import quat_to_matrix  # noqa: E402
from utils.training import get_device, load_checkpoint, load_config, set_seed  # noqa: E402


METHODS = ("v", "i", "vi")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/step_i.yaml")
    p.add_argument("--checkpoints_dir", type=str, default="outputs/step_i")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "test"])
    p.add_argument("--methods", type=str, nargs="+", default=list(METHODS))
    p.add_argument("--skip_missing", action="store_true")
    return p.parse_args()


def _build_model(method: str, cfg: dict, device: torch.device):
    cam = cfg.get("camera", {})
    model_cfg = cfg.get("model", {})
    if method == "v":
        return NetworkV_tdirect(feat_dim=model_cfg["cnn_feat_dim"]).to(device)
    if method == "i":
        return NetworkI(hidden_dim=model_cfg["lstm_hidden"]).to(device)
    if method == "vi":
        return NetworkVI_anchored(
            feat_dim=model_cfg["cnn_feat_dim"],
            lstm_hidden=model_cfg["lstm_hidden"],
            fx=float(cam.get("fx", 640.0)),
            fy=float(cam.get("fy", 640.0)),
            cx=float(cam.get("cx", 320.0)),
            cy=float(cam.get("cy", 240.0)),
            default_altitude_m=float(model_cfg.get("default_altitude_m", 4.0)),
            rho_decay=float(model_cfg.get("rho_decay", 8.0)),
        ).to(device)
    raise ValueError(f"unknown method {method}")


def _predict(method: str, model: torch.nn.Module, batch: dict
            ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    """Return (R, t, vi_extras). t-direct never falls back to IMU.

    For ``vi`` we additionally return per-sample gate signals so the caller
    can aggregate ``rho_v``, ``alpha_t``, ``alpha_R`` and the per-frame V
    error for the gate-stats summary.
    """
    if method == "v":
        R, t, _feat = model(batch["img_a"], batch["img_b"])
        return R[0], t[0], None
    if method == "i":
        R, t, _feat = model(batch["imu"])
        return R[0], t[0], None
    if method == "vi":
        # Eval-time: pass GT per-frame altitude through to the photometric
        # warp. The dataloader always provides batch["depth"] (per-frame
        # camera world-z); using it keeps rho_v's calibration consistent
        # with training-time conditioning. Falling back to None →
        # default_altitude_m would systematically bias rho_v low on traj 3
        # (sinusoid altitudes 1.5–3.0 m vs default 4.0 m).
        preds = model(batch["img_a"], batch["img_b"], batch["imu"],
                      depth_gt=batch.get("depth"))
        R, t = preds["R"], preds["t"]
        extras = {
            "rho_v": float(preds["rho_v"][0].item()),
            "alpha_t": float(preds["alpha_t"][0].item()),
            "alpha_R": float(preds["alpha_R"][0].item()),
            "t_v": preds["t_v"][0].detach().cpu().numpy(),
            "t_i": preds["t_i"][0].detach().cpu().numpy(),
        }
        return R[0], t[0], extras
    raise ValueError(method)


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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, or NaN if either input has zero variance."""
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = x.std(); sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _summarise_gate(extras_per_frame: list[dict],
                    pred_rels_t: list[np.ndarray],
                    gt_rels_t: list[np.ndarray]) -> dict:
    """Build the per-trajectory gate-statistics summary.

    Per the Step I deliverable, we want:
      * mean, std of alpha_t       — gate's translation preference
      * mean of alpha_R            — gate's rotation preference
      * mean of rho_v              — average photometric self-confidence
      * corr(rho_v, ||t_v - t_gt||)
            → tests whether rho_v is an actual signal: if rho_v drops when
              t_v gets worse, the correlation is NEGATIVE (rho high = error
              low). A value near 0 means the photometric signal is noise.
      * frac_alpha_t_v_dominant    — fraction of frames where alpha_t > 0.5
      * frac_alpha_t_i_dominant    — fraction of frames where alpha_t < 0.5

    Returns a JSON-serialisable dict.
    """
    if not extras_per_frame:
        return {"n_frames": 0}

    rho = np.array([e["rho_v"] for e in extras_per_frame])
    a_t = np.array([e["alpha_t"] for e in extras_per_frame])
    a_R = np.array([e["alpha_R"] for e in extras_per_frame])
    t_v = np.stack([e["t_v"] for e in extras_per_frame])              # (N, 3)

    # Per-frame V translation error vs GT relative translation. Lengths may
    # differ by 1 (we have N pairs of relative poses); align by taking the
    # min length so the correlation is computed over the common prefix.
    gt_t_arr = np.stack(gt_rels_t)                                    # (M, 3)
    M = min(t_v.shape[0], gt_t_arr.shape[0])
    err_t_v = np.linalg.norm(t_v[:M] - gt_t_arr[:M], axis=-1)         # (M,)
    rho_aligned = rho[:M]

    n_frames = int(t_v.shape[0])
    return {
        "n_frames": n_frames,
        "mean_alpha_t":  float(a_t.mean()),
        "std_alpha_t":   float(a_t.std()),
        "mean_alpha_R":  float(a_R.mean()),
        "std_alpha_R":   float(a_R.std()),
        "mean_rho_v":    float(rho.mean()),
        "std_rho_v":     float(rho.std()),
        "corr_rho_v_err_t_v": _safe_corr(rho_aligned, err_t_v),
        "frac_alpha_t_v_dominant": float((a_t > 0.5).mean()),
        "frac_alpha_t_i_dominant": float((a_t < 0.5).mean()),
        "frac_alpha_R_v_dominant": float((a_R > 0.5).mean()),
        "frac_alpha_R_i_dominant": float((a_R < 0.5).mean()),
    }


def _evaluate_method(method: str, cfg: dict, device: torch.device,
                     ckpt_path: Path, test_ids, writer: csv.writer,
                     gate_stats_out: dict) -> None:
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
        per_frame_gate: list[dict] = []
        per_frame_gt_t: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                R, t, extras = _predict(method, model, batch)
                rels.append((R.cpu().numpy(), t.cpu().numpy()))
                if extras is not None:
                    per_frame_gate.append(extras)
                    per_frame_gt_t.append(
                        batch["t_rel"][0].detach().cpu().numpy())

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
            "fallback_fraction": 0.0,
        })
        print(f"[eval] {method:<5} traj {tid}: "
              f"ATE-aligned={ate.ate_aligned:.3f} m, "
              f"endpoint={endpoint:.3f} m, "
              f"seg2s(med/p95)={med:.3f}/{p95:.3f} m")

        if method == "vi":
            stats = _summarise_gate(per_frame_gate, [r[1] for r in rels],
                                    per_frame_gt_t)
            gate_stats_out.setdefault("vi", {})[str(tid)] = stats
            print(f"[eval] vi    traj {tid}: gate stats — "
                  f"mean_alpha_t={stats['mean_alpha_t']:.3f}±"
                  f"{stats['std_alpha_t']:.3f}, "
                  f"mean_alpha_R={stats['mean_alpha_R']:.3f}, "
                  f"mean_rho_v={stats['mean_rho_v']:.3f}, "
                  f"corr(rho_v, |t_v−t_gt|)={stats['corr_rho_v_err_t_v']:.3f}, "
                  f"frac(alpha_t>0.5)={stats['frac_alpha_t_v_dominant']:.2f}")


def _ckpt_path(method: str, checkpoints_dir: Path) -> Path:
    sub = {"v": "v", "i": "i", "vi": "vi"}[method]
    return checkpoints_dir / sub / "best.pt"


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    variant = cfg.get("model", {}).get("variant", None)
    assert variant == "anchored", (
        f"evaluate_anchored.py requires model.variant=='anchored' "
        f"in config {args.config}, got {variant!r}")
    set_seed(cfg["seed"])
    device = get_device()

    split_ids = cfg["data"][f"{args.split}_ids"]
    out_dir = Path(cfg["paths"]["outputs_dir"])
    results_csv = out_dir / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    gate_stats_out: dict = {}
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
            _evaluate_method(method, cfg, device, ckpt, split_ids, writer,
                             gate_stats_out)

    if gate_stats_out:
        eval_dir = out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        gate_path = eval_dir / "gate_stats.json"
        gate_path.write_text(json.dumps(gate_stats_out, indent=2))
        print(f"[eval] wrote gate stats to {gate_path}")
    print(f"[eval] wrote results to {results_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
