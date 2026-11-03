"""Step K — precompute RAFT teacher predictions for the simple-shape trajs.

Iterates trajectories listed in ``configs/step_k.yaml::data.train_ids`` plus
``data.val_traj_id`` (default: 301..305), runs ``RAFTTeacher.predict_full``
in batches over every consecutive frame pair, and saves the result to
``{raw_dir}/traj_{N:04d}/raft_predictions.npz``.

The cache is consumed by ``utils/dataloader.py`` when constructed with
``raft_cache=True`` (Step K's ``train_v_distill.py``).

R3 mitigation — stale cache detection: we hash every input PNG and store the
combined SHA-1 in the cache metadata. The dataloader re-computes the hash at
training startup and aborts on mismatch.

Test trajectories (305-as-test, 306, 307, 300) do NOT need a cache; the
teacher is training-time only. We still cache 305 because it's the val/
minitest traj for best.pt selection — the val loop in ``train_v_distill.py``
calls the same loss function (with distillation), so the cache is consulted.

Usage:
    python scripts/precompute_raft.py --config configs/step_k.yaml
    python scripts/precompute_raft.py --config configs/step_k.yaml \\
        --traj_ids 301 302 303 304 305
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from models.raft_teacher import (  # noqa: E402
    DEFAULT_CORRESPONDENCE_STRIDE, DEFAULT_NUM_FLOW_UPDATES,
    DEFAULT_RANSAC_THRESHOLD_PX, RAFTTeacher,
)
from utils.dataloader import (  # noqa: E402
    FramePairDataset, _compute_frames_sha1, _load_image, traj_dir_name,
)
from utils.training import get_device, load_config  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/step_k.yaml")
    p.add_argument("--traj_ids", type=int, nargs="*", default=None,
                   help="Override config; default = train_ids + val_traj_id.")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Pairs per RAFT forward call. 2 keeps GPU mem low.")
    p.add_argument("--weights", type=str,
                   default="Raft_Large_Weights.C_T_SKHT_V2",
                   help="Override RAFT weights enum. Fallback path "
                        "(carpet uniformity): C_T_V2.")
    p.add_argument("--num_flow_updates", type=int,
                   default=DEFAULT_NUM_FLOW_UPDATES)
    p.add_argument("--correspondence_stride", type=int,
                   default=DEFAULT_CORRESPONDENCE_STRIDE)
    p.add_argument("--ransac_threshold_px", type=float,
                   default=DEFAULT_RANSAC_THRESHOLD_PX)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing raft_predictions.npz.")
    return p.parse_args()


def _resolve_traj_ids(cfg: dict, override: List[int] | None) -> List[int]:
    if override:
        return [int(x) for x in override]
    train_ids = list(cfg["data"]["train_ids"])
    val = int(cfg["data"]["val_traj_id"])
    out = list(dict.fromkeys(train_ids + [val]))
    return out


def _gather_pair_paths(traj_dir: Path, n_pairs: int):
    """Return [(frame_a_path, frame_b_path)] for k = 0..n_pairs-1."""
    out = []
    frames_dir = traj_dir / "frames"
    for k in range(n_pairs):
        a = frames_dir / f"{k:05d}.png"
        b = frames_dir / f"{k + 1:05d}.png"
        out.append((a, b))
    return out


def _build_intrinsics(cfg: dict) -> np.ndarray:
    cam = cfg.get("camera", {})
    K = np.array([
        [float(cam.get("fx", 640.0)), 0.0, float(cam.get("cx", 320.0))],
        [0.0, float(cam.get("fy", 640.0)), float(cam.get("cy", 240.0))],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def _process_traj(teacher: RAFTTeacher, traj_id: int, traj_dir: Path,
                  img_size, K_np: np.ndarray, batch_size: int,
                  device: torch.device) -> dict:
    """Run RAFT over every consecutive pair in one trajectory.

    Returns a dict with batched arrays + per-traj metadata.
    """
    # Load the trajectory metadata using the dataloader's helper. We don't
    # build a full Dataset because we want to control batching tightly.
    ds = FramePairDataset(raw_dir=traj_dir.parent, traj_ids=[traj_id],
                          img_size=img_size, mode="eval", preload_metadata=True,
                          raft_cache=False)
    traj = ds._trajectories[traj_id]
    n_pairs = traj.n_frames - 1
    pair_paths = _gather_pair_paths(traj_dir, n_pairs)

    # Per-pair altitude (depth) for translation scale recovery.
    depth_per_pair = np.array(
        [float(traj.poses_full[k * ds.imu_window, 2]) for k in range(n_pairs)],
        dtype=np.float32)

    R_b_all = np.zeros((n_pairs, 3, 3), dtype=np.float32)
    t_b_all = np.zeros((n_pairs, 3), dtype=np.float32)
    sigma_b_all = np.full((n_pairs,), float("inf"), dtype=np.float32)
    valid_all = np.zeros((n_pairs,), dtype=bool)
    flow_med_all = np.zeros((n_pairs,), dtype=np.float32)

    K_torch = torch.from_numpy(K_np.astype(np.float32))

    t0 = time.time()
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        imgs_a = torch.stack([_load_image(p[0], img_size)
                              for p in pair_paths[start:end]], dim=0)
        imgs_b = torch.stack([_load_image(p[1], img_size)
                              for p in pair_paths[start:end]], dim=0)
        imgs_a = imgs_a.to(device, non_blocking=True)
        imgs_b = imgs_b.to(device, non_blocking=True)
        depth = torch.from_numpy(depth_per_pair[start:end]).to(device)

        pred = teacher.predict_full(imgs_a, imgs_b,
                                    K=K_torch.to(device),
                                    depth_m=depth)

        R_b_all[start:end] = pred.R_b
        t_b_all[start:end] = pred.t_b
        sigma_b_all[start:end] = pred.sigma_b_px
        valid_all[start:end] = pred.valid_mask
        flow_med_all[start:end] = pred.flow_med_px

        if (start // batch_size) % 50 == 0:
            print(f"  traj_{traj_id}: pair {start}/{n_pairs} "
                  f"valid_so_far={int(valid_all[:end].sum())}/{end}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  traj_{traj_id}: {n_pairs} pairs in {elapsed:.1f}s "
          f"({elapsed*1000/max(1, n_pairs):.1f} ms/pair, "
          f"valid_frac={valid_all.mean():.3f}, "
          f"mean_sigma_px={float(np.nanmean(sigma_b_all[np.isfinite(sigma_b_all)])):.2f}, "
          f"mean_flow_px={flow_med_all.mean():.2f})")

    frames_sha1 = _compute_frames_sha1(traj)

    meta = {
        "traj_id": int(traj_id),
        "raft_arch": "raft_large",
        "raft_weights": teacher.weights_name,
        "torchvision_version": __import__("torchvision").__version__,
        "torch_version": torch.__version__,
        "n_flow_updates": teacher.num_flow_updates,
        "correspondence_stride": teacher.correspondence_stride,
        "ransac_threshold_px": teacher.ransac_threshold_px,
        "img_size": list(img_size),
        "K": K_np.astype(np.float32).tolist(),
        "n_pairs": int(n_pairs),
        "valid_fraction": float(valid_all.mean()),
        "mean_sigma_px_valid": float(
            np.nanmean(sigma_b_all[valid_all]) if valid_all.any()
            else float("nan")),
        "mean_flow_px": float(flow_med_all.mean()),
        "frames_sha1": frames_sha1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(elapsed),
    }
    return {
        "R_b": R_b_all,
        "t_b": t_b_all,
        "sigma_b": sigma_b_all,
        "valid_mask": valid_all,
        "flow_med_px": flow_med_all,
        "meta": meta,
    }


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    device = get_device()
    img_size = tuple(cfg["data"]["img_size"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    traj_ids = _resolve_traj_ids(cfg, args.traj_ids)
    K_np = _build_intrinsics(cfg)

    print(f"[precompute_raft] device={device} weights={args.weights} "
          f"trajs={traj_ids}")
    teacher = RAFTTeacher(
        weights=args.weights,
        num_flow_updates=args.num_flow_updates,
        correspondence_stride=args.correspondence_stride,
        ransac_threshold_px=args.ransac_threshold_px,
    )
    teacher.setup(device)

    summary = {}
    for tid in traj_ids:
        traj_dir = raw_dir / traj_dir_name(tid)
        cache_path = traj_dir / "raft_predictions.npz"
        if cache_path.exists() and not args.force:
            print(f"[precompute_raft] traj_{tid}: cache exists at "
                  f"{cache_path} — skipping (use --force to overwrite).")
            summary[str(tid)] = {"status": "skipped"}
            continue
        if not traj_dir.exists():
            print(f"[precompute_raft] traj_{tid}: directory missing "
                  f"({traj_dir}) — skipping.")
            summary[str(tid)] = {"status": "missing"}
            continue

        out = _process_traj(teacher, tid, traj_dir, img_size, K_np,
                            args.batch_size, device)
        # Save .npz with both arrays and a 0-d object for the meta dict.
        np.savez(
            str(cache_path),
            R_b=out["R_b"],
            t_b=out["t_b"],
            sigma_b=out["sigma_b"],
            valid_mask=out["valid_mask"],
            flow_med_px=out["flow_med_px"],
            meta=np.array(out["meta"], dtype=object),
        )
        print(f"[precompute_raft] traj_{tid}: wrote {cache_path}")
        summary[str(tid)] = {
            "status": "ok",
            "valid_fraction": out["meta"]["valid_fraction"],
            "mean_sigma_px_valid": out["meta"]["mean_sigma_px_valid"],
            "mean_flow_px": out["meta"]["mean_flow_px"],
            "elapsed_seconds": out["meta"]["elapsed_seconds"],
        }

    summary_path = Path(cfg["paths"]["outputs_dir"]) / "raft_precompute_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[precompute_raft] summary → {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
