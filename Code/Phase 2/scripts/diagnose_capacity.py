"""S6 diagnostic — network capacity vs. data complexity.

Hypothesis (S6): an 8-layer VGG8 CNN with the default channel ladder is
undersized for the task.

Checks:
  1. Print Network V's total parameter count + per-submodule breakdown.
  2. Parse V's train log written by scripts/train_v.py at
     ``{cfg.paths.outputs_dir}/v/train.log`` (default ``outputs/v/train.log``):
     - Final train vs. val loss.
     - Is train_loss still decreasing over the last 5 epochs?
     - Train/val gap = val_loss - train_loss.
  3. Load V's best.pt from ``{cfg.paths.checkpoints_dir}/v/best.pt`` and
     evaluate L_rot, L_td, L_n, L_warp on 20 random samples from the training
     dataset. Report per-sample component values, mean/variance per
     component, and the dominant/variance-heavy one.

Verdicts:
  - Train plateaued AND train ≈ val   → CONFIRMED (underfitting).
  - Train still descending             → more training, not capacity.
  - Val >> train                       → overfitting (more regularization).

Read-only. Writes JSON to outputs/audit/capacity.json.
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

from models.losses import DecomposedHomographyLoss  # noqa: E402
from models.network_v import NetworkV  # noqa: E402
from utils.dataloader import FramePairDataset  # noqa: E402
from utils.training import get_device, load_checkpoint, load_config  # noqa: E402


PLATEAU_WINDOW = 5
PLATEAU_ABS_TOL = 1e-3   # final 5-epoch slope in loss-units / epoch


def _param_breakdown(model: NetworkV) -> dict:
    def count(module: torch.nn.Module) -> int:
        return int(sum(p.numel() for p in module.parameters()))
    bb = model.backbone
    breakdown = {
        "total":       count(model),
        "backbone":    count(bb),
        "bb.stage1":   count(bb.stage1),
        "bb.stage2":   count(bb.stage2),
        "bb.stage3":   count(bb.stage3),
        "bb.stage4":   count(bb.stage4),
        "bb.proj":     count(bb.proj),
        "head_rot":    count(model.head_rot),
        "head_td":     count(model.head_td),
        "head_n":      count(model.head_n),
    }
    return breakdown


def _parse_train_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {"present": False}
    lines = log_path.read_text().splitlines()
    if len(lines) < 2:
        return {"present": True, "rows": 0}
    header = lines[0].split("\t")
    idx_epoch = header.index("epoch")
    idx_trn = header.index("train_loss")
    idx_val = header.index("val_loss")
    rows: list[tuple[int, float, float]] = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 4:
            continue
        try:
            rows.append((int(parts[idx_epoch]),
                         float(parts[idx_trn]), float(parts[idx_val])))
        except ValueError:
            continue

    if not rows:
        return {"present": True, "rows": 0}
    epochs = np.array([r[0] for r in rows])
    trn = np.array([r[1] for r in rows])
    val = np.array([r[2] for r in rows])
    w = min(PLATEAU_WINDOW, len(trn))
    tail_trn = trn[-w:]
    # Slope of a simple least-squares line over the last `w` epochs.
    if w >= 2:
        xs = np.arange(w, dtype=np.float64)
        slope = float(np.polyfit(xs, tail_trn, 1)[0])  # per-epoch change
    else:
        slope = 0.0
    plateaued = abs(slope) < PLATEAU_ABS_TOL
    gap = float(val[-1] - trn[-1])
    return {
        "present":           True,
        "rows":              int(len(rows)),
        "final_epoch":       int(epochs[-1]),
        "final_train_loss":  float(trn[-1]),
        "final_val_loss":    float(val[-1]),
        "min_train_loss":    float(trn.min()),
        "min_val_loss":      float(val.min()),
        "train_val_gap":     gap,
        "tail_slope_per_epoch": slope,
        "plateau_window":    int(w),
        "plateaued":         bool(plateaued),
    }




def _per_sample_components(model: NetworkV, loss_fn: DecomposedHomographyLoss,
                           ds: FramePairDataset, K: torch.Tensor,
                           device: torch.device, n_samples: int,
                           rng: np.random.Generator) -> list[dict]:
    """Compute L_rot, L_td, L_n, L_warp for `n_samples` random single samples."""
    rows: list[dict] = []
    indices = rng.integers(low=0, high=len(ds), size=n_samples).tolist()
    # Use a temporary loss_fn with hovering disabled-equivalent (width tiny)
    # so we observe per-sample signal rather than the hover-weighted mean.
    # Instead we batch-of-1: the hover weight simply scales the single sample,
    # so we also report the pre-weight L_td / L_n via a direct recompute.
    model.eval()
    for idx in indices:
        sample = ds[int(idx)]
        img_a = sample["img_a"].unsqueeze(0).to(device)
        img_b = sample["img_b"].unsqueeze(0).to(device)
        R_gt = sample["R_rel"].unsqueeze(0).to(device)
        t_gt = sample["t_rel"].unsqueeze(0).to(device)
        n_gt = sample["normal"].unsqueeze(0).to(device)
        d_gt = sample["depth"].unsqueeze(0).to(device)
        traj_id = int(sample["traj_id"].item())
        frame_idx = int(sample["frame_idx"].item())
        t_norm = float(t_gt.norm(dim=-1).item())

        with torch.no_grad():
            R_pred, td_pred, n_pred, _ = model(img_a, img_b)
            out = loss_fn(R_pred, td_pred, n_pred,
                          R_gt, t_gt, n_gt, d_gt,
                          img_a=img_a, img_b=img_b, K=K)

            # Unweighted per-sample l_td: recompute without hover mean.
            td_gt = t_gt / d_gt.clamp_min(1e-3).unsqueeze(-1)
            l_td_raw = float((td_pred - td_gt).abs().sum(-1).item())

        rows.append({
            "sample_idx": int(idx),
            "traj_id": traj_id,
            "frame_idx": frame_idx,
            "t_norm_m": t_norm,
            "l_rot":   float(out["l_rot"].item()),
            "l_td":    float(out["l_td"].item()),
            "l_td_raw": l_td_raw,
            "l_n":     float(out["l_n"].item()),
            "l_warp":  float(out["l_warp"].item()),
            "hover_w": float(out["hover_weight_mean"].item()),
        })
    return rows


def _summarize_components(rows: list[dict]) -> dict:
    arr = {k: np.array([r[k] for r in rows], dtype=np.float64)
           for k in ("l_rot", "l_td_raw", "l_n", "l_warp")}
    summary = {k: {"mean": float(v.mean()),
                   "std":  float(v.std()),
                   "p95":  float(np.quantile(v, 0.95)),
                   "min":  float(v.min()),
                   "max":  float(v.max())} for k, v in arr.items()}
    # Dominant = highest *weighted* contribution using config weights.
    return summary


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoints_dir", type=str, default=None,
                   help="Dir containing v/best.pt and v/train.log. Defaults "
                        "to cfg.paths.checkpoints_dir (i.e. the value used "
                        "by scripts/train_v.py; usually 'outputs').")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--out_dir", type=str, default="outputs/audit")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    device = get_device()
    ckpts_dir = Path(args.checkpoints_dir) if args.checkpoints_dir \
        else Path(cfg["paths"]["checkpoints_dir"])
    ckpt = ckpts_dir / "v" / "best.pt"
    log_path = ckpts_dir / "v" / "train.log"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"V checkpoint not found at {ckpt}. scripts/train_v.py writes "
            f"this path — has training completed?")

    model = NetworkV(feat_dim=cfg["model"]["cnn_feat_dim"]).to(device)
    blob = load_checkpoint(ckpt, map_location=device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=False)
    model.eval()

    params = _param_breakdown(model)
    log_stats = _parse_train_log(log_path)

    # Build a tiny training dataset to draw random samples from.
    raw_dir = Path(cfg["data"]["raw_dir"])
    img_size = tuple(cfg["data"]["img_size"])
    ds = FramePairDataset(raw_dir=raw_dir,
                          traj_ids=cfg["data"]["train_ids"],
                          img_size=img_size, mode="eval")

    # K from the first trajectory's metadata.
    first_tid = next(iter(ds._trajectories.keys()))
    K_np = ds._trajectories[first_tid].metadata["K"]
    K = torch.tensor(K_np, dtype=torch.float32, device=device)

    lw = cfg["loss"]
    loss_fn = DecomposedHomographyLoss(
        w_rot=lw["w_rot"], w_td=lw["w_td"], w_n=lw["w_n"], w_warp=lw["w_warp"],
        hover_center=lw.get("hover_mask_center", 0.01),
        hover_width=lw.get("hover_mask_width", 0.005))

    rng = np.random.default_rng(args.seed)
    rows = _per_sample_components(model, loss_fn, ds, K, device,
                                  args.n_samples, rng)
    comp_summary = _summarize_components(rows)
    # Weighted contribution (mean value × loss-config weight).
    weighted = {
        "L_rot":  lw["w_rot"]  * comp_summary["l_rot"]["mean"],
        "L_td":   lw["w_td"]   * comp_summary["l_td_raw"]["mean"],
        "L_n":    lw["w_n"]    * comp_summary["l_n"]["mean"],
        "L_warp": lw["w_warp"] * comp_summary["l_warp"]["mean"],
    }
    dominant = max(weighted.items(), key=lambda kv: kv[1])
    # Coefficient-of-variation across samples — high means "network struggles
    # on some inputs" per the hypothesis.
    cv = {
        "L_rot":   comp_summary["l_rot"]["std"]   / max(1e-9, comp_summary["l_rot"]["mean"]),
        "L_td":    comp_summary["l_td_raw"]["std"] / max(1e-9, comp_summary["l_td_raw"]["mean"]),
        "L_n":     comp_summary["l_n"]["std"]     / max(1e-9, comp_summary["l_n"]["mean"]),
        "L_warp":  comp_summary["l_warp"]["std"]  / max(1e-9, comp_summary["l_warp"]["mean"]),
    }

    print("\n=== S6: Network capacity ===")
    print("Parameter counts (NetworkV):")
    for k, v in params.items():
        print(f"  {k:<12s} {v:>10,d}")
    print("\nTrain log summary:")
    if not log_stats.get("present"):
        print(f"  (missing: {log_path})")
    else:
        print(f"  log path:          {log_path}")
        print(f"  rows logged:       {log_stats['rows']}")
        print(f"  final epoch:       {log_stats['final_epoch']}")
        print(f"  final train_loss:  {log_stats['final_train_loss']:.6f}")
        print(f"  final val_loss:    {log_stats['final_val_loss']:.6f}")
        print(f"  min train_loss:    {log_stats['min_train_loss']:.6f}")
        print(f"  min val_loss:      {log_stats['min_val_loss']:.6f}")
        print(f"  train/val gap:     {log_stats['train_val_gap']:+.6f}  "
              f"(val − train)")
        print(f"  tail slope (per epoch, last {log_stats['plateau_window']}):  "
              f"{log_stats['tail_slope_per_epoch']:+.6f}  "
              f"→ plateaued={log_stats['plateaued']}")

    print("\nPer-sample loss components (unweighted means over "
          f"{len(rows)} random train pairs):")
    for name, k in (("L_rot", "l_rot"), ("L_td", "l_td_raw"),
                    ("L_n", "l_n"), ("L_warp", "l_warp")):
        s = comp_summary[k]
        print(f"  {name:<6s} mean={s['mean']:.5f}  std={s['std']:.5f}  "
              f"p95={s['p95']:.5f}  min={s['min']:.5f}  max={s['max']:.5f}  "
              f"CV={s['std'] / max(1e-9, s['mean']):.3f}")
    print(f"\n  Weighted contributions (w_* × mean(component)):")
    for k, v in weighted.items():
        print(f"    {k:<6s} {v:.5f}")
    print(f"  → dominant: {dominant[0]} (weighted {dominant[1]:.5f})")
    print(f"  Coefficients of variation (per-component): {cv}")

    # Verdict logic.
    if not log_stats.get("present"):
        verdict = "INCONCLUSIVE"
        reason = "train.log missing; cannot determine plateau or gap."
    else:
        gap = log_stats["train_val_gap"]
        plateaued = log_stats["plateaued"]
        # "small" gap: under 15% of final val loss — matches our V run
        # (train=0.132, val=0.134, gap≈1.5%).
        small_gap = abs(gap) < 0.15 * abs(log_stats["final_val_loss"])
        if plateaued and small_gap:
            verdict = "CONFIRMED (underfitting)"
            reason = ("train_loss plateaued AND train ≈ val — "
                      "more capacity would likely help.")
        elif not plateaued:
            verdict = "UNLIKELY (undertrained)"
            reason = ("train_loss still descending — more training, not "
                      "more capacity, is the first lever.")
        elif gap > 0.15 * log_stats["final_val_loss"]:
            verdict = "UNLIKELY (overfitting)"
            reason = ("val >> train — add regularization or data, don't "
                      "add capacity.")
        else:
            verdict = "INCONCLUSIVE"
            reason = "Signals mixed."

    print(f"\n[S6] VERDICT: {verdict}")
    print(f"  - {reason}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "capacity.json"
    out_json.write_text(json.dumps({
        "checkpoint": str(ckpt),
        "train_log_path": str(log_path),
        "param_breakdown": params,
        "train_log": log_stats,
        "per_sample_rows": rows,
        "component_summary": comp_summary,
        "weighted_contributions": weighted,
        "coefficient_of_variation": cv,
        "dominant_component": dominant[0],
        "verdict": verdict,
        "verdict_reason": reason,
    }, indent=2))
    print(f"[S6] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
