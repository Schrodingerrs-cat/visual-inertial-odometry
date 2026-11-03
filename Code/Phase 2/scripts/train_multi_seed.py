"""Multi-seed training wrapper (Tier-1 B.4).

Trains one of ``{v, i, vi}`` sequentially across several integer seeds, each
into its own output subdirectory. After all seeds finish, parses each run's
final ``val_ATE_on_minitest`` from the tabular ``train.log`` and writes a
summary CSV plus mean±std to stdout.

Runs sequentially in a single process (not three parallel SBATCH jobs) so
the cluster's 4-concurrent-job cap isn't hit by a single multi-seed batch.

Usage:
    python scripts/train_multi_seed.py --target vi --seeds 1 2 3 \\
                                       --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional


TARGET_SCRIPT = {
    "v":  "scripts/train_v.py",
    "i":  "scripts/train_i.py",
    "vi": "scripts/train_vi.py",
}


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=str, required=True, choices=sorted(TARGET_SCRIPT))
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--output_root", type=str, default="outputs",
                   help="Parent dir for per-seed subdirs (default: outputs/).")
    p.add_argument("--pretrained_v", type=str, default="",
                   help="For --target vi: pretrained V checkpoint path.")
    p.add_argument("--pretrained_i", type=str, default="",
                   help="For --target vi: pretrained I checkpoint path.")
    return p.parse_args()


def _parse_final_metric(train_log: Path, col: str) -> Optional[float]:
    if not train_log.exists():
        return None
    with train_log.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        if col not in header:
            return None
        j = header.index(col)
        last_val: Optional[float] = None
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= j: continue
            try:
                last_val = float(parts[j])
            except ValueError:
                continue
        return last_val


def _parse_best_from_stdout(stdout_text: str) -> Optional[float]:
    m = re.search(r"best val_ATE_on_minitest\s*=\s*([0-9eE.+\-]+)", stdout_text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _parse_final_train_loss(train_log: Path) -> Optional[float]:
    return _parse_final_metric(train_log, "train_loss")


def _run_one_seed(target: str, seed: int, cfg_path: str,
                  output_root: Path, args: argparse.Namespace
                  ) -> Dict[str, float]:
    sub = f"{target}_seed{seed}"
    out_dir = output_root / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, TARGET_SCRIPT[target],
           "--config", cfg_path,
           "--out_subdir", sub,
           "--seed", str(seed)]
    if target == "vi":
        if args.pretrained_v:
            cmd += ["--pretrained_v", args.pretrained_v]
        if args.pretrained_i:
            cmd += ["--pretrained_i", args.pretrained_i]
    print(f"[multi_seed] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # Print the subprocess output so it is visible in SLURM logs.
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)

    train_log = out_dir / "train.log"
    return {
        "seed": float(seed),
        "val_ATE_minitest": (
            _parse_best_from_stdout(result.stdout)
            if _parse_best_from_stdout(result.stdout) is not None
            else (_parse_final_metric(train_log, "val_ATE_on_minitest") or float("nan"))
        ),
        "val_ATE_traj0":  _parse_final_metric(train_log, "val_ATE_on_traj_0") or float("nan"),
        "final_train_loss": _parse_final_train_loss(train_log) or float("nan"),
    }


def main() -> int:
    args = _parse()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for s in args.seeds:
        rows.append(_run_one_seed(args.target, s, args.config, output_root, args))

    summary_path = output_root / f"{args.target}_multiseed_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "val_ATE_minitest", "val_ATE_traj0", "final_train_loss"])
        for r in rows:
            w.writerow([int(r["seed"]),
                        f"{r['val_ATE_minitest']:.6f}",
                        f"{r['val_ATE_traj0']:.6f}",
                        f"{r['final_train_loss']:.6f}"])

    minis = [r["val_ATE_minitest"] for r in rows
             if r["val_ATE_minitest"] == r["val_ATE_minitest"]]  # drop NaN
    if minis:
        mu = mean(minis)
        sd = stdev(minis) if len(minis) > 1 else 0.0
        print(f"[multi_seed] target={args.target}  seeds={args.seeds}")
        print(f"[multi_seed] val_ATE_on_minitest (mean ± std) = "
              f"{mu:.4f} ± {sd:.4f} over {len(minis)} seeds")
    print(f"[multi_seed] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
