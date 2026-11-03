"""Full evaluation orchestrator (Step S2).

One command produces every plot the user needs, neatly labeled, into a
single timestamped folder under ``Results/Phase2/eval_<YYYYMMDD_HHMMSS>/``.

Pipeline (each step in its own try/except — a failure does not abort later
steps):

  0. Run ``scripts/plot_imu_diag.py`` directly to
     ``Results/Phase2/imu_diag_postfix/``. Always runs first; needs only
     ``data/raw_g/`` (no checkpoints).
  1. Create ``Results/Phase2/eval_<ts>/{trajectories,losses,imu_diag}/`` and
     copy the imu_diag_postfix output into ``imu_diag/``.
  2. Run ``scripts/evaluate.py`` once per split: train, val (via
     ``--traj_ids 305``), test. Each call may fail loudly if checkpoints are
     missing; the orchestrator logs the failure and continues.
  3. Run ``plot_results.plot_all`` to produce the standard per-traj +
     overlay + training-curve plots into the canonical
     ``Results/Phase2/`` locations.
  4. Relabel-and-copy pass: pull the produced PNGs into the timestamped
     bundle with the spec-mandated naming
     (``train_figure8_0.5mps_gt_vs_pred_overlay.png`` etc.). The
     V/I/VI three-up subplot row is generated here from the same TUM
     bundles ``plot_all`` already loaded — without modifying any
     ``plot_results/*.py`` file.
  5. Print the final eval-bundle path.

Usage:
    python3 scripts/full_eval.py
    python3 scripts/full_eval.py --config configs/step_j.yaml \\
        --checkpoints_dir outputs/run_1 --val_traj_id 305
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import yaml                       # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

# Reuse the existing loaders without touching plot_results/* code.
from plot_results import common as _PC                         # noqa: E402
from plot_results.plot_all import main as _plot_all_main       # noqa: E402
from scripts.plot_imu_diag import (                            # noqa: E402
    plot_imu_diag, _label_for_traj, DEFAULT_OUT_DIR as _DEFAULT_IMU_OUT,
)


REPO_ROOT = HERE.parents[1]
RESULTS_ROOT = REPO_ROOT / "Results" / "Phase2"

# Method → user-facing network name (for filenames + titles in the bundle).
_NET_NAME = {
    "v":  "NetworkV",
    "i":  "NetworkI",
    "vi": "NetworkVI",
}
_METHODS_FOR_BUNDLE = ("v", "i", "vi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_run(cmd: List[str], stage: str, log: list) -> bool:
    """Run a subprocess, log result, return True on success."""
    print(f"[full_eval] {stage}: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=HERE, check=False)
    except Exception as e:                                       # noqa: BLE001
        log.append({"stage": stage, "ok": False, "error": str(e)})
        print(f"[full_eval] {stage}: EXCEPTION {e}")
        return False
    ok = proc.returncode == 0
    log.append({"stage": stage, "ok": ok, "returncode": proc.returncode})
    print(f"[full_eval] {stage}: {'OK' if ok else f'FAIL rc={proc.returncode}'}")
    return ok


def _copy_tree(src: Path, dst: Path) -> int:
    """Mirror src into dst (dst created/extended). Returns file count copied."""
    n = 0
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    return n


def _build_label_map(raw_dir: Path,
                     ids: Iterable[int]) -> Dict[int, Tuple[str, str, str, str]]:
    """traj_id -> (split, family_short, speed_str, label_no_split).

    label_no_split is `<family>_<speed>mps`. The orchestrator prepends the
    split discovered at the same time.
    """
    out: Dict[int, Tuple[str, str, str, str]] = {}
    for tid in ids:
        td = raw_dir / f"traj_{tid:04d}"
        if not (td / "metadata.json").exists():
            continue
        label_full, split, fam, speed, is_target = _label_for_traj(td)
        # label_full = f"{split}_{fam}_{speed}mps" — strip the leading split_.
        speed_str = label_full.split("_")[-1]   # "1.0mps" or "auto1.0mps"
        out[tid] = (split, fam, speed_str, f"{fam}_{speed_str}")
    return out


# ---------------------------------------------------------------------------
# 1x3 subplot row (V/I/VI side-by-side per trajectory) — new figure variant.
# Uses plot_results.common loaders only; does NOT modify plot_results/*.py.
# ---------------------------------------------------------------------------

def _three_up_subplot(traj_id: int, traj_dir: Path,
                      out_path: Path, title: str) -> bool:
    bundles = []
    for m in _METHODS_FOR_BUNDLE:
        b = _PC.load_bundle(m, traj_id, traj_dir)
        if b is not None:
            bundles.append((m, b))
    if not bundles:
        return False
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    fig.suptitle(title, fontsize=12)
    for col, m in enumerate(_METHODS_FOR_BUNDLE):
        ax = axs[0, col]
        match = next((b for (mm, b) in bundles if mm == m), None)
        if match is None:
            ax.set_title(f"{_NET_NAME[m]}: (no checkpoint)")
            ax.axis('off'); continue
        ax.plot(match.gt_P[:, 0], match.gt_P[:, 1],
                color=_PC.GT_COLOR, lw=_PC.GT_LW, label='GT')
        ax.plot(match.pred_aligned_P[:, 0], match.pred_aligned_P[:, 1],
                color=_PC.EST_COLOR, lw=_PC.EST_LW, label=_NET_NAME[m])
        ax.set_title(_NET_NAME[m])
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/step_j.yaml")
    p.add_argument("--checkpoints_dir", type=str, default="outputs",
                   help="Passed to scripts/evaluate.py.")
    p.add_argument("--outputs_dir", type=str, default=None,
                   help="Where plot_all looks for {net}/train.log + "
                        "trajectories/. Defaults to --checkpoints_dir so "
                        "the standard `outputs/<step>/{v,i,vi}/train.log` "
                        "layout works without further config.")
    p.add_argument("--data_dir", type=Path,
                   default=HERE / "data" / "raw_g")
    p.add_argument("--results_dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--val_traj_id", type=int, default=None,
                   help="Defaults to cfg.data.val_traj_id.")
    p.add_argument("--methods", nargs="+", default=list(_METHODS_FOR_BUNDLE),
                   help="Methods passed to scripts/evaluate.py.")
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip the eval subprocesses; reuse whatever TUMs "
                        "already exist under outputs/trajectories/.")
    return p.parse_args()


def main() -> int:
    args = _parse()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train_ids = list(cfg["data"]["train_ids"])
    test_ids  = list(cfg["data"].get("test_ids", []))
    val_id    = (args.val_traj_id if args.val_traj_id is not None
                 else int(cfg["data"]["val_traj_id"]))

    ts = _ts()
    bundle = args.results_dir / f"eval_{ts}"
    bundle_traj = bundle / "trajectories"
    bundle_loss = bundle / "losses"
    bundle_imu  = bundle / "imu_diag"
    for d in (bundle_traj, bundle_loss, bundle_imu):
        d.mkdir(parents=True, exist_ok=True)

    log: list = []
    print(f"[full_eval] bundle = {bundle}")
    print(f"[full_eval] config = {args.config}  ckpts = {args.checkpoints_dir}")
    print(f"[full_eval] train_ids={train_ids}  val_id={val_id}  "
          f"test_ids={test_ids}")

    # 0. IMU diag — independent of checkpoints, runs first.
    imu_summary: Dict[str, dict] = {}
    try:
        imu_summary = plot_imu_diag(args.data_dir, _DEFAULT_IMU_OUT)
        log.append({"stage": "imu_diag", "ok": True,
                    "n_trajs": len(imu_summary)})
    except Exception as e:                                       # noqa: BLE001
        log.append({"stage": "imu_diag", "ok": False, "error": str(e)})
        print(f"[full_eval] imu_diag FAILED: {e}")

    # 1. Copy IMU diag into the bundle.
    n = _copy_tree(_DEFAULT_IMU_OUT, bundle_imu)
    print(f"[full_eval] copied {n} IMU diag files -> {bundle_imu}")

    # 2. Run evaluator per split (each in a try/except).
    if not args.skip_eval:
        eval_methods = list(args.methods)
        for split_name, override_ids in (
            ("train", None),
            ("val",   [val_id]),
            ("test",  None),
        ):
            cmd = [sys.executable, "scripts/evaluate.py",
                   "--config", args.config,
                   "--checkpoints_dir", args.checkpoints_dir,
                   "--methods", *eval_methods,
                   "--skip_missing"]
            if override_ids is None:
                cmd += ["--split", split_name]
            else:
                cmd += ["--split", "test",         # arg is required even with override
                        "--traj_ids", *map(str, override_ids)]
            _safe_run(cmd, f"evaluate[{split_name}]", log)
    else:
        print("[full_eval] --skip_eval set; not invoking scripts/evaluate.py")

    # 3. Run the standard plotting pipeline (training + trajectories +
    # distributions + comparison + ablations) into Results/Phase2/.
    # plot_results.common hardcodes TRAIN_IDS=(0,1,2,3), TEST_IDS=(100,101,102)
    # and OUTPUTS_ROOT = phase2/outputs. Override at runtime so plot_all picks
    # up the actual config IDs and the actual checkpoint subdir without
    # modifying any plot_results/*.py file.
    outputs_dir = (Path(args.outputs_dir) if args.outputs_dir
                   else Path(args.checkpoints_dir))
    if not outputs_dir.is_absolute():
        outputs_dir = (HERE / outputs_dir).resolve()
    saved_train = _PC.TRAIN_IDS
    saved_test = _PC.TEST_IDS
    saved_test_names = dict(_PC.TEST_NAMES)
    _PC.TRAIN_IDS = tuple(train_ids)
    # Plot every traj that has GT — easier than juggling val/test sets.
    _PC.TEST_IDS = tuple(sorted({val_id, *test_ids}))
    for tid, (split, fam, speed_str, _) in (
            _build_label_map(args.data_dir,
                             {*train_ids, val_id, *test_ids}).items()):
        _PC.TEST_NAMES[tid] = f"{fam}_{speed_str}"
    plot_all_argv_backup = sys.argv[:]
    try:
        sys.argv = ["plot_all",
                    "--outputs", str(outputs_dir),
                    "--results", str(args.results_dir),
                    "--split", "both",
                    "--methods", ",".join(args.methods)]
        _plot_all_main()
        log.append({"stage": "plot_all", "ok": True})
    except SystemExit as e:
        log.append({"stage": "plot_all", "ok": (e.code in (0, None)),
                    "exit_code": int(e.code or 0)})
    except Exception as e:                                       # noqa: BLE001
        log.append({"stage": "plot_all", "ok": False, "error": str(e)})
        print(f"[full_eval] plot_all FAILED: {e}")
    finally:
        sys.argv = plot_all_argv_backup
        _PC.TRAIN_IDS = saved_train
        _PC.TEST_IDS = saved_test
        _PC.TEST_NAMES.clear(); _PC.TEST_NAMES.update(saved_test_names)

    # 4. Relabel-and-copy pass into the timestamped bundle.
    all_ids = sorted(set(train_ids) | {val_id} | set(test_ids))
    label_map = _build_label_map(args.data_dir, all_ids)

    # 4a. Loss curves (training_plots already wrote to Results/Phase2/training/)
    training_dir = args.results_dir / "training"
    for m in args.methods:
        src = training_dir / f"{m}_loss.png"
        if src.exists():
            shutil.copy2(src, bundle_loss / f"{_NET_NAME[m]}_loss_curves.png")
    for combo in ("combined_loss.png", "combined_val_ate.png"):
        src = training_dir / combo
        if src.exists():
            new_name = ("combined_loss_curves.png"
                        if combo == "combined_loss.png" else combo)
            shutil.copy2(src, bundle_loss / new_name)

    # 4b. Per-trajectory plots: copy overlays + write the new 1x3 row.
    traj_tum_dir = HERE / "outputs" / "trajectories"
    for tid in all_ids:
        if tid not in label_map:
            continue
        split, fam, speed_str, _ = label_map[tid]
        base = f"{split}_{fam}_{speed_str}_gt_vs_pred"
        # Copy the existing V+I+VI shared-axes overlay (top view).
        # plot_results.trajectory_plots._overlay_top writes "top.png".
        overlay_src = args.results_dir / f"overlay_{tid}" / "top.png"
        if overlay_src.exists():
            shutil.copy2(overlay_src, bundle_traj / f"{base}_overlay.png")
        # Generate the new 1x3 subplot row from TUM bundles.
        title = (f"{split} / {fam} @ {speed_str} (traj_{tid:04d}) — "
                 "GT vs prediction per network")
        ok = _three_up_subplot(tid, traj_tum_dir,
                               bundle_traj / f"{base}_subplots.png", title)
        if not ok:
            print(f"[full_eval] traj {tid}: no TUMs found; "
                  "skipped subplots variant")

    # 5. Write a manifest + print final path.
    manifest = {
        "timestamp": ts,
        "config": args.config,
        "checkpoints_dir": args.checkpoints_dir,
        "data_dir": str(args.data_dir),
        "train_ids": train_ids,
        "val_id": val_id,
        "test_ids": test_ids,
        "label_map": {str(k): {"split": v[0], "family": v[1],
                                "speed": v[2]}
                       for k, v in label_map.items()},
        "imu_diag": imu_summary,
        "stages": log,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                      default=str) + "\n")
    print()
    print(f"[full_eval] DONE.  bundle: {bundle}")
    print(f"[full_eval] manifest: {bundle / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
