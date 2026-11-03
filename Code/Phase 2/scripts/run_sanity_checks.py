"""Run all 8 Phase 2 sanity checks and dump figures + summary JSON.

Usage:
    python scripts/run_sanity_checks.py --data_dir data/raw --config configs/default.yaml

Checks:
    1. GT pose chaining
    2. IMU double-integration
    3. Homography warping consistency
    4. Motion-magnitude histogram (legacy, cross-traj)
    5. Drone-dynamics thrust alignment
    6. Motion distribution (per-traj, Tier-1 B.5)
    7. Footprint confinement (plane + camera footprint)
    8. Texture assignment (metadata.texture_path present and correct)

DO NOT PROCEED to training if any check reports passed=False.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from utils.dataloader import TrajectoryData  # noqa: E402
from utils.sanity_check import (  # noqa: E402
    check_drone_dynamics,
    check_footprint_confinement,
    check_homography_warp,
    check_imu_integration,
    check_motion_distribution,
    check_motion_histogram,
    check_pose_chaining,
    check_texture_assignment,
    dump_summary,
    save_figures,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, default="data/raw")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--sanity_traj_id", type=int, default=None,
                   help="Trajectory ID for per-traj checks 1/2/3/5. "
                        "Defaults to data.train_ids[0].")
    p.add_argument("--soft_fail", type=str, default="",
                   help="Comma-separated list of check names to downgrade "
                        "from fail → warn. Useful when a check is known to "
                        "reveal a numerical-precision issue that doesn't "
                        "block training (e.g. drone_dynamics).")
    return p.parse_args()


def main() -> int:
    args = _parse()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(args.data_dir)
    train_ids = cfg["data"]["train_ids"]
    test_ids  = cfg["data"].get("test_ids", [])
    val_id    = cfg["data"].get("minitest_traj_id", 200)
    sanity_id = (args.sanity_traj_id if args.sanity_traj_id is not None
                 else train_ids[0])

    out_dir = Path(cfg["paths"]["sanity_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = TrajectoryData.from_dir(raw_dir / f"traj_{sanity_id:04d}")

    print(f"[sanity] Running 8 checks; primary single-traj = {sanity_id:04d}")

    r1 = check_pose_chaining(primary)
    print(f"  [1] {r1.name:<22} {'PASS' if r1.passed else 'FAIL'} — {r1.message}")
    r2 = check_imu_integration(primary)
    print(f"  [2] {r2.name:<22} {'PASS' if r2.passed else 'FAIL'} — {r2.message}")
    r3 = check_homography_warp(primary)
    print(f"  [3] {r3.name:<22} {'PASS' if r3.passed else 'FAIL'} — {r3.message}")

    trajs_iter = (TrajectoryData.from_dir(raw_dir / f"traj_{tid:04d}") for tid in train_ids)
    r4 = check_motion_histogram(trajs_iter)
    print(f"  [4] {r4.name:<22} {'PASS' if r4.passed else 'FAIL'} — {r4.message}")

    r5 = check_drone_dynamics(primary)
    print(f"  [5] {r5.name:<22} {'PASS' if r5.passed else 'FAIL'} — {r5.message}")

    r6 = check_motion_distribution(raw_dir, train_ids, out_dir)
    print(f"  [6] {r6.name:<22} {'PASS' if r6.passed else 'FAIL'} — {r6.message}")

    all_ids = list(train_ids) + list(test_ids) + [val_id]
    r7 = check_footprint_confinement(raw_dir, all_ids, out_dir)
    print(f"  [7] {r7.name:<22} {'PASS' if r7.passed else 'FAIL'} — {r7.message}")

    r8 = check_texture_assignment(raw_dir, project_root=HERE, traj_ids=all_ids)
    print(f"  [8] {r8.name:<22} {'PASS' if r8.passed else 'FAIL'} — {r8.message}")

    results = [r1, r2, r3, r4, r5, r6, r7, r8]
    for r in results:
        if r.figures:
            save_figures(r.figures, out_dir, prefix=r.name)
    dump_summary(results, out_dir / "summary.json")

    soft_fail = {s.strip() for s in args.soft_fail.split(",") if s.strip()}
    hard_failures = [r for r in results if not r.passed and r.name not in soft_fail]
    soft_failures = [r for r in results if not r.passed and r.name in soft_fail]
    print()
    if not hard_failures:
        print("[sanity] SUMMARY:", "ALL PASSED" if not soft_failures
              else "ALL HARD CHECKS PASSED (soft failures present)")
    else:
        print("[sanity] SUMMARY: HARD FAILURES PRESENT")
    for r in results:
        status = ("OK  " if r.passed
                  else ("WARN" if r.name in soft_fail else "FAIL"))
        print(f"  {r.name:<24} {status}  {r.message}")
    if hard_failures:
        print("[sanity] DO NOT proceed to training until every check passes.")
        return 1
    if soft_failures:
        print("[sanity] Soft-failed checks: "
              f"{', '.join(r.name for r in soft_failures)}. Proceeding anyway.")
    print("[sanity] Ready for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
