"""VI architecture ablation — 4 variants × 3 test trajectories.

Trains four VI variants and evaluates each on the test split:
    - vi_concat           : group10-style concat-MLP 6-DoF (no decomposition)
    - vi_no_decomp        : decomposed vision head, learned scale, no IMU fusion
    - vi_concat_decompose : decomposed head + concat-MLP fusion (learned, not
                            analytic) — isolates the analytic-fusion contribution
    - vi_full             : decomposed head + analytic IMU scale fusion

Writes:
    outputs/studies/ablation_long.csv   per-(method, traj_id, metric_name, value)
    outputs/studies/ablation_ate.csv    4x3 ATE-aligned pivot table
    outputs/studies/ablation.png        bar chart

Same training protocol as train_vi.py: training traj list excludes the first
entry in config.data.train_ids; that first entry is the validation traj used
to pick the best checkpoint by dead-reckoned val-ATE.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from _train_common import (  # noqa: E402
    TextLogger, build_val_loader_single_traj, compute_val_ate_traj0,
    maybe_save_periodic,
)
from models.losses import GeodesicRotationLoss, VILoss  # noqa: E402
from models.network_vi import (  # noqa: E402
    NetworkVI, NetworkVIConcat, NetworkVIConcatDecompose, NetworkVINoDecomp,
)
from utils.dataloader import build_train_loader  # noqa: E402
from utils.training import (  # noqa: E402
    cosine_warmup_lr, get_device, load_config, save_checkpoint, set_seed,
)


VARIANTS = ("vi_concat", "vi_no_decomp", "vi_concat_decompose", "vi_full")
# evaluate.py subdir names. vi_full → "vi".
_VARIANT_TO_METHOD = {
    "vi_concat": "vi_concat",
    "vi_no_decomp": "vi_no_decomp",
    "vi_concat_decompose": "vi_concat_decompose",
    "vi_full": "vi",
}


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--skip_train", action="store_true")
    return p.parse_args()


def _build(variant: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    fd = cfg["model"]["cnn_feat_dim"]; lh = cfg["model"]["lstm_hidden"]
    if variant == "vi_concat":
        return NetworkVIConcat(feat_dim=fd, lstm_hidden=lh).to(device)
    if variant == "vi_no_decomp":
        return NetworkVINoDecomp(feat_dim=fd, lstm_hidden=lh).to(device)
    if variant == "vi_concat_decompose":
        return NetworkVIConcatDecompose(feat_dim=fd, lstm_hidden=lh).to(device)
    if variant == "vi_full":
        return NetworkVI(feat_dim=fd, lstm_hidden=lh).to(device)
    raise ValueError(variant)


def _variant_loss(variant: str, cfg: dict) -> Callable[..., Dict[str, torch.Tensor]]:
    if variant == "vi_concat":
        geo = GeodesicRotationLoss()
        def _loss(preds, R_gt, t_gt, *_a, **_kw):
            l_rot = geo(preds["R"], R_gt)
            l_t = (preds["t"] - t_gt).abs().sum(-1).mean()
            return {"total": cfg["loss"]["w_rot"] * l_rot
                             + cfg["loss"]["w_trans"] * l_t,
                    "l_rot": l_rot, "l_trans": l_t}
        return _loss
    base = VILoss(w_rot=cfg["loss"]["w_rot"], w_td=cfg["loss"]["w_td"],
                  w_n=cfg["loss"]["w_n"], w_warp=cfg["loss"]["w_warp"],
                  w_trans=cfg["loss"]["w_trans"],
                  w_imu_trans=cfg["loss"]["w_imu_trans"],
                  w_rot_imu=cfg["loss"]["w_rot_imu"],
                  w_d=cfg["loss"]["w_d"])
    def _loss(preds, R_gt, t_gt, n_gt, d_gt, img_a=None, img_b=None, K=None):
        return base(preds, R_gt, t_gt, n_gt, d_gt, img_a, img_b, K)
    return _loss


def _forward_rel(model, batch):
    preds = model(batch["img_a"], batch["img_b"], batch["imu"])
    return preds["R"], preds["t"]


def _train_variant(variant: str, cfg: dict, device: torch.device,
                   out_root: Path) -> Path:
    out_dir = out_root / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    logger = TextLogger(out_dir / "train.log")

    all_train = list(cfg["data"]["train_ids"])
    val_tid = all_train[0]
    train_tids = all_train[1:] if len(all_train) > 1 else all_train

    loader, ds = build_train_loader(
        raw_dir=Path(cfg["data"]["raw_dir"]),
        traj_ids=train_tids,
        img_size=tuple(cfg["data"]["img_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        sampling_mode=cfg["sampling"]["mode"],
        num_bins=cfg["sampling"]["num_bins"],
        bin_edges=cfg["sampling"]["bin_edges_m"],
        alpha=cfg["sampling"].get("alpha", 0.5),
        seed=cfg["seed"],
    )
    val_loader, val_ds = build_val_loader_single_traj(
        raw_dir=Path(cfg["data"]["raw_dir"]),
        traj_id=val_tid, img_size=tuple(cfg["data"]["img_size"]),
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
    )
    model = _build(variant, cfg, device)
    loss_fn = _variant_loss(variant, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    n_epochs = (cfg["train"]["epochs_vi_stage1"]
                + cfg["train"]["epochs_vi_stage2"])
    K = torch.tensor(next(iter(ds._trajectories.values())).metadata["K"],
                     dtype=torch.float32, device=device)
    best_ate = float("inf")
    for epoch in range(n_epochs):
        cosine_warmup_lr(opt, epoch, n_epochs, cfg["train"]["warmup_epochs"],
                         cfg["train"]["lr"])
        model.train()
        pbar = tqdm(loader, desc=f"{variant} epoch {epoch+1}/{n_epochs}")
        total, m = 0.0, 0
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            preds = model(batch["img_a"], batch["img_b"], batch["imu"])
            l = loss_fn(preds, batch["R_rel"], batch["t_rel"],
                        batch["normal"], batch["depth"],
                        batch["img_a"], batch["img_b"], K)
            opt.zero_grad(set_to_none=True)
            l["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["train"]["grad_clip"])
            opt.step()
            total += float(l["total"]); m += 1
        mean_total = total / max(1, m)

        val_ate = compute_val_ate_traj0(model, val_loader, val_ds,
                                        _forward_rel, device)
        lr_now = opt.param_groups[0]["lr"]
        writer.add_scalar("epoch/train_total", mean_total, epoch)
        writer.add_scalar("epoch/val_ATE", val_ate, epoch)
        logger.log_epoch(epoch=epoch, stage="single", train_loss=mean_total,
                         val_loss="", val_ATE_on_traj_0=val_ate, lr=lr_now)
        save_checkpoint(out_dir / "last.pt", model, opt, epoch,
                        {"train_loss": mean_total, "val_ATE": val_ate})
        maybe_save_periodic(lambda p, meta: save_checkpoint(p, model, opt, epoch, meta),
                            out_dir, epoch, every=5,
                            meta={"train_loss": mean_total, "val_ATE": val_ate})
        if val_ate < best_ate:
            best_ate = val_ate
            save_checkpoint(out_dir / "best.pt", model, opt, epoch,
                            {"val_ATE": val_ate, "variant": variant,
                             "note": "best-by-val-ATE"})
    writer.close()
    print(f"[ablation] {variant} done. best val_ATE = {best_ate:.4f}")
    return out_dir / "best.pt"


def _run_evaluate(cfg_path: str, out_root: Path, methods: list[str]) -> None:
    # Ensure the "vi" checkpoint exists (vi_full aliased to vi).
    vi_full_src = out_root / "vi_full" / "best.pt"
    vi_dst = out_root / "vi" / "best.pt"
    vi_dst.parent.mkdir(parents=True, exist_ok=True)
    if vi_full_src.exists():
        try:
            if vi_dst.exists() or vi_dst.is_symlink(): vi_dst.unlink()
            vi_dst.symlink_to(Path("..") / "vi_full" / "best.pt")
        except OSError:
            import shutil; shutil.copy(vi_full_src, vi_dst)
    subprocess.check_call([sys.executable, "scripts/evaluate.py",
                           "--config", cfg_path,
                           "--checkpoints_dir", str(out_root),
                           "--methods"] + methods + ["--skip_missing"])


def _build_reports(cfg: dict, studies: Path, methods: list[str]) -> None:
    master = Path(cfg["paths"]["outputs_dir"]) / "results.csv"
    if not master.exists():
        print("[ablation] no master results.csv; cannot build ablation csv/plot.")
        return
    rows = list(csv.DictReader(master.open()))
    keep = [r for r in rows if r["method"] in methods]
    # Long-form copy for the study.
    with (studies / "ablation_long.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "traj_id", "metric_name", "value"])
        w.writeheader()
        for r in keep:
            w.writerow({k: r[k] for k in w.fieldnames})
    # 4x3 ATE-aligned pivot.
    ate_rows = [r for r in keep if r["metric_name"] == "ate_aligned"]
    traj_ids = sorted({r["traj_id"] for r in ate_rows}, key=int)
    ate_pivot: dict[str, dict[str, float]] = {m: {} for m in methods}
    for r in ate_rows:
        ate_pivot[r["method"]][r["traj_id"]] = float(r["value"])
    with (studies / "ablation_ate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"traj_{t}" for t in traj_ids] + ["mean"])
        for m in methods:
            vals = [ate_pivot[m].get(t, float("nan")) for t in traj_ids]
            valid = [v for v in vals if not np.isnan(v)]
            mean = float(np.mean(valid)) if valid else float("nan")
            w.writerow([m] + [f"{v:.6f}" for v in vals] + [f"{mean:.6f}"])
    # Bar chart: per-method mean-ATE with std across trajs.
    means = [float(np.mean(list(ate_pivot[m].values()) or [np.nan])) for m in methods]
    errs  = [float(np.std(list(ate_pivot[m].values()) or [np.nan])) for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(methods)), means, yerr=errs, tick_label=methods,
           color=["#CC5555", "#EAA94C", "#7D9CC9", "#2E6E31"])
    ax.set_ylabel("ATE-aligned (m), mean over test trajectories")
    ax.set_title("VI ablation (4 variants × 3 test trajectories)")
    fig.tight_layout(); fig.savefig(studies / "ablation.png", dpi=140); plt.close(fig)
    print(f"[ablation] wrote {studies/'ablation_long.csv'}, "
          f"{studies/'ablation_ate.csv'}, {studies/'ablation.png'}")


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    out_root = Path(cfg["paths"]["outputs_dir"])
    studies = Path(cfg["paths"]["studies_dir"]); studies.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        for v in VARIANTS:
            _train_variant(v, cfg, device, out_root)

    methods = [_VARIANT_TO_METHOD[v] for v in VARIANTS]
    _run_evaluate(args.config, out_root, methods)
    _build_reports(cfg, studies, methods)
    return 0


if __name__ == "__main__":
    sys.exit(main())
