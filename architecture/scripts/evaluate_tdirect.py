"""Evaluate V (t-direct) / I / VI (t-direct) on the test split.

Mirror of ``scripts/evaluate.py`` but binds ``v`` to ``NetworkV_tdirect`` and
``vi`` to ``NetworkVI_tdirect``. ``i`` is unchanged (the symlinked Step G I
checkpoint loads directly into ``NetworkI``).

For each test trajectory we now produce:

  Per (method, traj_id) — single-method plots:
    * ``{plots_dir}/{method}_{tid}_3d.png``         — 3D overlay (GT vs pred)
    * ``{plots_dir}/{method}_{tid}_topdown.png``    — top-down XY
    * ``{plots_dir}/{method}_{tid}_xyz.png``        — position vs time
    * ``{plots_dir}/{method}_{tid}_rpy.png``        — roll/pitch/yaw vs time
    * ``{plots_dir}/{method}_{tid}_xyz_error.png``  — pred − GT per axis (NEW)

  Per traj_id — comparison overlay across V/I/VI:
    * ``{plots_dir}/comparison_{tid}_xy.png``       — top-down XY (NEW)
    * ``{plots_dir}/comparison_{tid}_3d.png``       — 3D overlay (NEW)
    * ``{plots_dir}/comparison_{tid}_xyz_error.png`` — per-axis error (NEW)

ATE is now embedded in single-method plot titles. Start point is green / end
point is red across all plots. See ``utils/eval.py`` for the plotting impls.

Asserts ``cfg.model.variant == 'tdirect'`` for safety.

Usage:
    python scripts/evaluate_tdirect.py --config configs/step_h.yaml \\
        --checkpoints_dir outputs/step_h --split test \\
        --methods v i vi --skip_missing
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from models.network_i import NetworkI  # noqa: E402
from models.network_v_tdirect import NetworkV_tdirect  # noqa: E402
from models.network_vi_tdirect import NetworkVI_tdirect  # noqa: E402
from utils.dataloader import build_eval_loader  # noqa: E402
from utils.eval import (  # noqa: E402
    MethodTrace, dead_reckon_from_gt_start, gt_trajectory_frames,
    matrix_to_rpy, plot_2d_topdown, plot_3d_overlay,
    plot_3d_overlay_methods, plot_per_axis, plot_per_axis_error,
    plot_per_axis_error_methods, plot_xy_overlay_methods,
    run_evo_ape, segment_drift, write_tum,
)
from utils.geometry import quat_to_matrix  # noqa: E402
from utils.training import get_device, load_checkpoint, load_config, set_seed  # noqa: E402


METHODS = ("v", "i", "vi")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/step_h.yaml")
    p.add_argument("--checkpoints_dir", type=str, default="outputs/step_h")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "test"])
    p.add_argument("--methods", type=str, nargs="+", default=list(METHODS))
    p.add_argument("--skip_missing", action="store_true")
    return p.parse_args()


def _build_model(method: str, cfg: dict, device: torch.device):
    if method == "v":
        return NetworkV_tdirect(feat_dim=cfg["model"]["cnn_feat_dim"]).to(device)
    if method == "i":
        ctx_len = cfg.get("data", {}).get("imu_context_length", None)
        if ctx_len is not None and int(ctx_len) != 10:
            print(f"[eval] method=i — consuming batch['imu_context'] of "
                  f"length {ctx_len} (Phase 2 causal context window).")
        else:
            print(f"[eval] method=i — consuming batch['imu_context'] "
                  f"(aliased to legacy length-10 batch['imu']).")
        return NetworkI(hidden_dim=cfg["model"]["lstm_hidden"]).to(device)
    if method == "vi":
        # VI verification: the IMU branch of NetworkVI_tdirect must consume
        # the legacy length-10 batch['imu'] tensor, NOT the (possibly
        # longer) batch['imu_context']. This is enforced in _predict above.
        # Print the contract at build time so a regression is loud.
        print("[eval] method=vi — IMU branch consumes batch['imu'] "
              "(legacy length 10); NOT batch['imu_context']. "
              "VI checkpoint compatibility preserved across phases.")
        return NetworkVI_tdirect(
            feat_dim=cfg["model"]["cnn_feat_dim"],
            lstm_hidden=cfg["model"]["lstm_hidden"]).to(device)
    raise ValueError(f"unknown method {method}")


def _predict(method: str, model: torch.nn.Module, batch: dict
            ) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return (R, t, fell_back). t-direct never falls back to IMU.

    Phase 2 routing: the I branch consumes ``batch["imu_context"]`` —
    the optional longer causal context window emitted by FramePairDataset
    when ``imu_context_length`` is set. Falls back to ``batch["imu"]``
    when the dataloader didn't emit a context tensor (Phase 0 / 1
    behavior). VI continues to consume ``batch["imu"]`` (legacy length
    10) — its checkpoint was trained on length-10 windows and must not
    be fed the longer context.
    """
    if method == "v":
        R, t, _feat = model(batch["img_a"], batch["img_b"])
    elif method == "i":
        imu_for_i = batch.get("imu_context", batch["imu"])
        R, t, _feat = model(imu_for_i)
    elif method == "vi":
        preds = model(batch["img_a"], batch["img_b"], batch["imu"])
        R, t = preds["R"], preds["t"]
    else:
        raise ValueError(method)
    return R[0], t[0], False


def _endpoint_drift(gt_ts: np.ndarray, pred_ts: np.ndarray) -> float:
    return float(np.linalg.norm(pred_ts[-1] - gt_ts[-1]))


def _write_row(writer: csv.writer, method: str, tid: int,
               metrics: dict) -> None:
    for name, value in metrics.items():
        writer.writerow([method, tid, name, f"{float(value):.6f}"])


def _ckpt_path(method: str, checkpoints_dir: Path) -> Path:
    sub = {"v": "v", "i": "i", "vi": "vi"}[method]
    return checkpoints_dir / sub / "best.pt"


def _load_ready_models(methods, cfg, device, ckpt_root: Path,
                       skip_missing: bool) -> dict:
    """Build + load every requested method's checkpoint up front.

    Returns a dict keyed by method name; methods whose checkpoint is missing
    (and ``skip_missing`` was set) are silently dropped.
    """
    models: dict = {}
    for method in methods:
        ckpt = _ckpt_path(method, ckpt_root)
        if not ckpt.exists():
            msg = f"[eval] checkpoint missing for {method}: {ckpt}"
            if skip_missing:
                print(msg + " — skipping."); continue
            raise FileNotFoundError(msg)
        m = _build_model(method, cfg, device)
        blob = load_checkpoint(ckpt, map_location=device)
        m.load_state_dict(blob["model"], strict=False)
        m.eval()
        models[method] = m
    return models


def _run_method_on_traj(method: str, model: torch.nn.Module,
                        device: torch.device, loader, traj):
    """Forward-pass over the loader, dead-reckon, return (times, pred_ts,
    pred_q, gt_times, gt_ts, gt_q) all truncated to the common length N."""
    rels: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            R, t, _fb = _predict(method, model, batch)
            rels.append((R.cpu().numpy(), t.cpu().numpy()))
    times, pred_ts, pred_q = dead_reckon_from_gt_start(traj, rels)
    gt_times, gt_ts, gt_q = gt_trajectory_frames(traj)
    N = min(len(times), len(gt_times))
    return (times[:N], pred_ts[:N], pred_q[:N],
            gt_times[:N], gt_ts[:N], gt_q[:N])


def _evaluate_per_traj(cfg: dict, device: torch.device,
                       models: dict, test_ids,
                       writer: csv.writer) -> None:
    """For each tid: run every loaded method, write rows + per-method plots,
    then render the V/I/VI comparison plots."""
    traj_dir = Path(cfg["paths"]["trajectories_dir"])
    plot_dir = Path(cfg["paths"]["plots_dir"])
    traj_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    imu_ctx_len = cfg.get("data", {}).get("imu_context_length", None)
    for tid in test_ids:
        loader, ds = build_eval_loader(
            raw_dir=Path(cfg["data"]["raw_dir"]),
            traj_ids=[tid], img_size=tuple(cfg["data"]["img_size"]),
            num_workers=0,
            imu_context_length=imu_ctx_len)
        traj = list(ds._trajectories.values())[0]

        traces: list[MethodTrace] = []
        gt_anchor_ts: np.ndarray | None = None  # set on first method

        for method, model in models.items():
            (times, pred_ts, pred_q,
             gt_times, gt_ts, gt_q) = _run_method_on_traj(
                method, model, device, loader, traj)

            if gt_anchor_ts is None:
                gt_anchor_ts = gt_ts.copy()

            pred_tum = traj_dir / f"{method}_{tid}.tum"
            gt_tum = traj_dir / f"gt_{tid}.tum"
            write_tum(gt_tum, gt_times, gt_ts, gt_q)
            write_tum(pred_tum, times, pred_ts, pred_q)

            ate = run_evo_ape(gt_tum, pred_tum, evo_bin=cfg["eval"]["evo_bin"])
            med, p95, _arr = segment_drift(gt_ts, pred_ts, times,
                                           window_s=cfg["eval"]["segment_seconds"])
            endpoint = _endpoint_drift(gt_ts, pred_ts)
            ate_value = float(ate.ate_aligned)

            # --- per-method single-trajectory plots ----------------------
            plot_3d_overlay(gt_ts, pred_ts,
                            plot_dir / f"{method}_{tid}_3d.png",
                            title=f"{method} — traj {tid} (3D overlay)",
                            ate=ate_value)
            plot_per_axis(times, gt_ts, pred_ts,
                          plot_dir / f"{method}_{tid}_xyz.png",
                          ylabels=("x (m)", "y (m)", "z (m)"),
                          title=f"{method} — traj {tid} (position vs time)",
                          ate=ate_value)
            gt_R = quat_to_matrix(torch.from_numpy(gt_q)).numpy()
            pr_R = quat_to_matrix(torch.from_numpy(pred_q)).numpy()
            gt_rpy = np.degrees(matrix_to_rpy(gt_R))
            pr_rpy = np.degrees(matrix_to_rpy(pr_R))
            plot_per_axis(times, gt_rpy, pr_rpy,
                          plot_dir / f"{method}_{tid}_rpy.png",
                          ylabels=("roll (deg)", "pitch (deg)", "yaw (deg)"),
                          title=f"{method} — traj {tid} (rpy vs time)",
                          ate=ate_value)
            plot_2d_topdown(gt_ts, pred_ts,
                            plot_dir / f"{method}_{tid}_topdown.png",
                            title=f"{method} — traj {tid} (top-down XY)",
                            ate=ate_value)
            plot_per_axis_error(times, gt_ts, pred_ts,
                                plot_dir / f"{method}_{tid}_xyz_error.png",
                                title=f"{method} — traj {tid} (position error)",
                                ate=ate_value)

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

            traces.append(MethodTrace(
                method=method, pred_ts=pred_ts, times=times,
                ate=ate_value,
            ))

        # --- per-traj comparison overlays (V vs I vs VI) -----------------
        if traces and gt_anchor_ts is not None and len(traces) >= 2:
            plot_xy_overlay_methods(
                gt_anchor_ts, traces,
                plot_dir / f"comparison_{tid}_xy.png",
                title=f"comparison — traj {tid} (top-down XY)")
            plot_3d_overlay_methods(
                gt_anchor_ts, traces,
                plot_dir / f"comparison_{tid}_3d.png",
                title=f"comparison — traj {tid} (3D overlay)")
            plot_per_axis_error_methods(
                gt_anchor_ts, traces,
                plot_dir / f"comparison_{tid}_xyz_error.png",
                title=f"comparison — traj {tid} (per-axis error)")
            print(f"[eval] traj {tid}: comparison plots → "
                  f"{plot_dir}/comparison_{tid}_*.png")


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    variant = cfg.get("model", {}).get("variant", None)
    assert variant == "tdirect", (
        f"evaluate_tdirect.py requires model.variant=='tdirect' "
        f"in config {args.config}, got {variant!r}")
    set_seed(cfg["seed"])
    device = get_device()

    split_ids = cfg["data"][f"{args.split}_ids"]
    results_csv = Path(cfg["paths"]["outputs_dir"]) / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    ckpt_root = Path(args.checkpoints_dir)

    models = _load_ready_models(args.methods, cfg, device,
                                ckpt_root, args.skip_missing)
    if not models:
        print("[eval] no methods to evaluate (all checkpoints missing).")
        return 0

    with results_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["method", "traj_id", "metric_name", "value"])
        _evaluate_per_traj(cfg, device, models, split_ids, writer)
    print(f"[eval] wrote results to {results_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
