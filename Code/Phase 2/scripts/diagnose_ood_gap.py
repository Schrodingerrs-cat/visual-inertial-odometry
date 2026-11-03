"""S4 diagnostic — train- vs test-split ATE gap for Network V only.

Hypothesis (S4): V overfit to training trajectories and fails on test shapes.

This is a standalone V-only dead-reckoning evaluator. It avoids the full
scripts/evaluate.py pipeline (which instantiates all 6 methods and was
reported to get stuck). All it does is:

  1. Load V's best.pt from ``{cfg.paths.checkpoints_dir}/v/best.pt``
     (default ``outputs/v/best.pt``; overridable via --checkpoints_dir).
     These are the canonical paths written by scripts/train_v.py.
  2. For each trajectory in `trajs`, iterate consecutive frame-pairs, predict
     (R, td), multiply td by the trajectory's altitude_initial (exactly what
     evaluate.py does for method='v'), and dead-reckon onto GT start.
  3. Compute SE(3)-aligned ATE via Umeyama (Horn).

Defaults mirror configs/default.yaml: train_ids = [0, 1, 2, 3],
test_ids = [100, 101, 102].

Read-only: writes only JSON to outputs/audit/ood_gap.json (and prints a
verdict). Does not touch results.csv or trajectories_dir.

Verdicts:
  - train ATE < 3 and test ATE ~13 → CONFIRMED (classic overfitting).
  - train ATE ≈ test ATE (both ~13) → UNLIKELY (representation is bad
    everywhere, not OOD-specific).
  - train ATE > 8 and test ATE ~13 → both-bad (capacity/data problem).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from models.network_v import NetworkV  # noqa: E402
from utils.dataloader import build_eval_loader, _quat_to_R  # noqa: E402
from utils.geometry import compose_se3  # noqa: E402
from utils.training import get_device, load_checkpoint, load_config  # noqa: E402


DEFAULT_TRAIN = (0, 1, 2, 3)
DEFAULT_TEST = (100, 101, 102)


def _ate_se3_aligned(ts_pred: np.ndarray, ts_gt: np.ndarray) -> float:
    """SE(3)-aligned ATE (no scale). Umeyama / Horn on translations."""
    N = min(ts_pred.shape[0], ts_gt.shape[0])
    X = ts_pred[:N].T
    Y = ts_gt[:N].T
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    Xc = X - mu_x
    Yc = Y - mu_y
    S = (Yc @ Xc.T) / N
    U, _sv, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    t = mu_y.flatten() - R @ mu_x.flatten()
    aligned = R @ X + t[:, None]
    return float(np.sqrt(((aligned - Y) ** 2).sum(axis=0).mean()))


def _evaluate_v_on_trajectory(model: NetworkV, device: torch.device,
                              raw_dir: Path, traj_id: int,
                              img_size: tuple[int, int]) -> dict:
    loader, ds = build_eval_loader(
        raw_dir=raw_dir, traj_ids=[traj_id], img_size=img_size, num_workers=0)
    traj = list(ds._trajectories.values())[0]
    meta = traj.metadata
    altitude = float(meta.get("altitude_initial",
                              meta.get("altitude_m", 4.0)))

    rels: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            img_a = batch["img_a"].to(device, non_blocking=True)
            img_b = batch["img_b"].to(device, non_blocking=True)
            R, td, _n, _f = model(img_a, img_b)
            t = td * altitude
            rels.append((R[0].cpu().numpy(), t[0].cpu().numpy()))

    # Dead-reckon onto GT frame-0 pose.
    row0 = traj.poses_full[0]
    R_cur = _quat_to_R(row0[3:])
    t_cur = row0[:3].astype(np.float64).copy()
    N = len(rels) + 1
    ts = np.zeros((N, 3))
    Rs = np.zeros((N, 3, 3))
    ts[0] = t_cur; Rs[0] = R_cur
    for i, (dR, dt) in enumerate(rels):
        R_cur, t_cur = compose_se3(Rs[i], ts[i],
                                   dR.astype(np.float64),
                                   dt.astype(np.float64))
        Rs[i + 1] = R_cur; ts[i + 1] = t_cur

    # GT translations sampled one-per-kept-frame (stride 10 into poses_full).
    stride = 10
    gt_ts = traj.poses_full[::stride, :3]
    M = min(ts.shape[0], gt_ts.shape[0])
    ate_aligned = _ate_se3_aligned(ts[:M], gt_ts[:M])
    endpoint = float(np.linalg.norm(ts[M - 1] - gt_ts[M - 1]))
    return {
        "traj_id": traj_id,
        "altitude_initial": altitude,
        "n_pairs": len(rels),
        "ate_aligned_m": ate_aligned,
        "endpoint_drift_m": endpoint,
    }


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoints_dir", type=str, default=None,
                   help="Dir containing v/best.pt. Defaults to "
                        "cfg.paths.checkpoints_dir (i.e. the value used by "
                        "scripts/train_v.py; usually 'outputs').")
    p.add_argument("--train_ids", type=int, nargs="+", default=list(DEFAULT_TRAIN))
    p.add_argument("--test_ids",  type=int, nargs="+", default=list(DEFAULT_TEST))
    p.add_argument("--out_dir", type=str, default="outputs/audit")
    return p.parse_args()


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    device = get_device()

    img_size = tuple(cfg["data"]["img_size"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    ckpts_dir = Path(args.checkpoints_dir) if args.checkpoints_dir \
        else Path(cfg["paths"]["checkpoints_dir"])
    ckpt = ckpts_dir / "v" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"V checkpoint not found at {ckpt}. scripts/train_v.py writes "
            f"this path — has training completed?")

    model = NetworkV(feat_dim=cfg["model"]["cnn_feat_dim"]).to(device)
    blob = load_checkpoint(ckpt, map_location=device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[S4] loaded V from {ckpt}  "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    print(f"[S4] device={device}, img_size={img_size}, raw_dir={raw_dir}")

    train_rows = [_evaluate_v_on_trajectory(model, device, raw_dir, tid, img_size)
                  for tid in args.train_ids]
    test_rows = [_evaluate_v_on_trajectory(model, device, raw_dir, tid, img_size)
                 for tid in args.test_ids]

    print("\n=== S4: OOD generalization gap (Network V) ===")
    for r in train_rows:
        print(f"  TRAIN traj {r['traj_id']:>4d} (alt {r['altitude_initial']:.3f} m, "
              f"{r['n_pairs']} pairs): "
              f"ATE-aligned = {r['ate_aligned_m']:.3f} m, "
              f"endpoint = {r['endpoint_drift_m']:.3f} m")
    for r in test_rows:
        print(f"  TEST  traj {r['traj_id']:>4d} (alt {r['altitude_initial']:.3f} m, "
              f"{r['n_pairs']} pairs): "
              f"ATE-aligned = {r['ate_aligned_m']:.3f} m, "
              f"endpoint = {r['endpoint_drift_m']:.3f} m")

    mean_train = float(np.mean([r["ate_aligned_m"] for r in train_rows]))
    mean_test = float(np.mean([r["ate_aligned_m"] for r in test_rows]))
    ratio = mean_test / max(1e-6, mean_train)
    print(f"\n  mean TRAIN ATE = {mean_train:.3f} m")
    print(f"  mean TEST  ATE = {mean_test:.3f} m")
    print(f"  ratio test/train = {ratio:.2f}")

    if mean_train < 3.0 and mean_test >= 8.0:
        verdict = "CONFIRMED"
        reason = "train ATE < 3 m while test ATE ≥ 8 m — V overfits motion patterns"
    elif mean_train >= 8.0 and mean_test >= 8.0:
        verdict = "UNLIKELY (both-bad)"
        reason = "train and test ATE both > 8 m — representation problem, not OOD"
    elif abs(mean_train - mean_test) / max(1e-6, max(mean_train, mean_test)) < 0.25:
        verdict = "UNLIKELY"
        reason = "train ≈ test ATE — no generalization gap"
    else:
        verdict = "PARTIAL"
        reason = "gap exists but not in the textbook CONFIRMED range"

    print(f"\n[S4] VERDICT: {verdict}")
    print(f"  - {reason}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "ood_gap.json"
    out_json.write_text(json.dumps({
        "checkpoint": str(ckpt),
        "mean_train_ate_m": mean_train,
        "mean_test_ate_m": mean_test,
        "ratio_test_over_train": ratio,
        "train_per_traj": train_rows,
        "test_per_traj": test_rows,
        "verdict": verdict,
        "verdict_reason": reason,
    }, indent=2))
    print(f"[S4] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
