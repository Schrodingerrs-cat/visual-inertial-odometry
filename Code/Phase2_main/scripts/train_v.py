"""Train Network V — t-direct variant (Step H).

Mirrors ``scripts/train_v.py`` but uses ``NetworkV_tdirect`` and
``TDirectPoseLoss``. The model predicts (R, t) directly, so:
  * No ``td_pred * d_gt`` reconstruction step in ``_forward_rel``.
  * No ``K`` / no photometric warp loss.
  * No ``n_pred`` supervision.

Best-checkpoint selection is unchanged: ``val_ATE_on_minitest`` (which is
traj 3 under step_h.yaml).

Asserts ``cfg.model.variant == 'tdirect'`` to fail loudly if pointed at the
wrong config.

Usage:
    python scripts/train_v_tdirect.py --config configs/step_h.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from _train_common import (  # noqa: E402
    TextLogger, build_val_loader_single_traj, compute_val_ate_on_trajectory,
    compute_val_loss, generate_training_plots, maybe_save_periodic,
)
from models.losses_tdirect import TDirectPoseLoss  # noqa: E402
from models.network_v_tdirect import NetworkV_tdirect  # noqa: E402
from utils.dataloader import PhotometricAugment, build_train_loader  # noqa: E402
from utils.training import (  # noqa: E402
    cosine_warmup_lr, get_device, load_config, save_checkpoint, set_seed,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/step_h.yaml")
    p.add_argument("--out_subdir", type=str, default="v")
    p.add_argument("--sampling_mode", type=str, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _loss_compute(loss_fn):
    def fn(model, batch):
        R_pred, t_pred, _feat = model(batch["img_a"], batch["img_b"])
        return loss_fn(R_pred, t_pred,
                       R_gt=batch["R_rel"], t_gt=batch["t_rel"])
    return fn


def _forward_rel(model, batch):
    R_pred, t_pred, _feat = model(batch["img_a"], batch["img_b"])
    return R_pred, t_pred


def _maybe_build_augment(cfg: dict) -> PhotometricAugment | None:
    aug_cfg = cfg.get("augmentation", {}).get("photometric", {})
    if not aug_cfg.get("enabled", False):
        return None
    return PhotometricAugment(
        brightness=aug_cfg.get("brightness", 0.3),
        contrast=aug_cfg.get("contrast", 0.3),
        saturation=aug_cfg.get("saturation", 0.2),
        hue=aug_cfg.get("hue", 0.05),
        gaussian_noise_std=aug_cfg.get("gaussian_noise_std", 0.01),
        grayscale_prob=aug_cfg.get("grayscale_prob", 0.1))


def main() -> int:
    args = _parse()
    cfg = load_config(args.config)
    variant = cfg.get("model", {}).get("variant", None)
    assert variant == "tdirect", (
        f"train_v_tdirect.py requires model.variant=='tdirect' "
        f"in config {args.config}, got {variant!r}")
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = get_device()

    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["paths"]["outputs_dir"]) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    logger = TextLogger(out_dir / "train.log")

    all_train = list(cfg["data"]["train_ids"])
    val_tid = cfg["data"].get("val_traj_id", all_train[0])
    minitest_tid = cfg["data"].get("minitest_traj_id", 200)
    train_tids = list(all_train)
    print(f"[train_v_tdirect] train={train_tids} val={val_tid} "
          f"minitest={minitest_tid} seed={seed}")

    sampling_mode = args.sampling_mode or cfg["sampling"]["mode"]
    alpha = args.alpha if args.alpha is not None else cfg["sampling"].get("alpha", 0.5)
    augment = _maybe_build_augment(cfg)
    loader, _ds = build_train_loader(
        raw_dir=raw_dir, traj_ids=train_tids,
        img_size=tuple(cfg["data"]["img_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        sampling_mode=sampling_mode,
        num_bins=cfg["sampling"].get("num_bins", 5),
        bin_edges=cfg["sampling"].get("bin_edges_m"),
        alpha=alpha,
        seed=seed,
        photometric_augment=augment,
    )
    val_loader, val_ds = build_val_loader_single_traj(
        raw_dir=raw_dir, traj_id=val_tid,
        img_size=tuple(cfg["data"]["img_size"]),
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
    )
    minitest_loader, minitest_ds = build_val_loader_single_traj(
        raw_dir=raw_dir, traj_id=minitest_tid,
        img_size=tuple(cfg["data"]["img_size"]),
        num_workers=max(1, cfg["data"]["num_workers"] // 2),
    )

    model = NetworkV_tdirect(feat_dim=cfg["model"]["cnn_feat_dim"]).to(device)
    loss_fn = TDirectPoseLoss(
        w_rot=cfg["loss"]["w_rot"], w_t=cfg["loss"]["w_t"],
        hover_center=cfg["loss"].get("hover_mask_center", 0.01),
        hover_width=cfg["loss"].get("hover_mask_width", 0.005)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    loss_train = _loss_compute(loss_fn)
    loss_val = _loss_compute(loss_fn)

    best_ate = float("inf")
    total_epochs = cfg["train"]["epochs_v"]
    for epoch in range(total_epochs):
        cosine_warmup_lr(opt, epoch, total_epochs,
                         cfg["train"]["warmup_epochs"], cfg["train"]["lr"])
        model.train()
        pbar = tqdm(loader, desc=f"V[tdirect] epoch {epoch+1}/{total_epochs}")
        epoch_total, n = 0.0, 0
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            losses = loss_train(model, batch)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["train"]["grad_clip"])
            opt.step()
            epoch_total += losses["total"].item(); n += 1
            if i % cfg["train"]["log_every"] == 0:
                step = epoch * len(loader) + i
                for k, v in losses.items():
                    if torch.is_tensor(v):
                        writer.add_scalar(f"train/{k}", float(v.item()), step)
                pbar.set_postfix(total=float(losses['total']))
        mean_train = epoch_total / max(1, n)

        val_loss = compute_val_loss(model, val_loader, loss_val, device)
        val_ate_traj0 = compute_val_ate_on_trajectory(
            model, val_loader, val_ds, _forward_rel, device)
        val_ate_mini = compute_val_ate_on_trajectory(
            model, minitest_loader, minitest_ds, _forward_rel, device)
        lr_now = opt.param_groups[0]["lr"]
        writer.add_scalar("epoch/train_total", mean_train, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_ATE_traj0", val_ate_traj0, epoch)
        writer.add_scalar("epoch/val_ATE_minitest", val_ate_mini, epoch)
        logger.log_epoch(epoch=epoch, stage="single", train_loss=mean_train,
                         val_loss=val_loss,
                         val_ATE_on_traj_0=val_ate_traj0,
                         val_ATE_on_minitest=val_ate_mini, lr=lr_now)

        save_checkpoint(out_dir / "last.pt", model, opt, epoch,
                        {"train_loss": mean_train, "val_loss": val_loss,
                         "val_ATE_traj0": val_ate_traj0,
                         "val_ATE_minitest": val_ate_mini})
        maybe_save_periodic(lambda p, meta: save_checkpoint(p, model, opt, epoch, meta),
                            out_dir, epoch, every=5,
                            meta={"train_loss": mean_train,
                                  "val_ATE_minitest": val_ate_mini})
        if val_ate_mini < best_ate:
            best_ate = val_ate_mini
            save_checkpoint(out_dir / "best.pt", model, opt, epoch,
                            {"val_ATE_minitest": val_ate_mini,
                             "val_ATE_traj0": val_ate_traj0,
                             "note": "best-by-val-ATE-minitest"})
            print(f"[train_v_tdirect] epoch {epoch}: "
                  f"new best val_ATE_minitest = {val_ate_mini:.4f}")
    writer.close()
    print(f"[train_v_tdirect] done. best val_ATE_on_minitest = {best_ate:.4f}")

    generate_training_plots(
        model=model,
        loaders={"val": (val_loader, val_ds),
                 "minitest": (minitest_loader, minitest_ds)},
        forward_rel=_forward_rel,
        device=device,
        out_plots_dir=out_dir / "plots",
        method="v",
        tag="final",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
