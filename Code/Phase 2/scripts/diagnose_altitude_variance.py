"""S5 diagnostic — altitude variance vs. GT (t/d) supervision.

Hypothesis (S5): GT (t/d) is supervised using `altitude_initial` (a scalar
taken from frame 0 via metadata), but the drone's altitude varies frame to
frame. Every frame where true altitude differs from initial is being
supervised with a systematically-wrong target by
``(altitude_true / altitude_initial - 1)``.

For each trajectory in [0, 1, 2, 3, 100, 101, 102] this reports:
  - min/max/median/std of z across all poses_full rows.
  - std/mean ratio (the S5 threshold uses 0.1).
  - max/min ratio.
  - altitude_initial from metadata.json (for audit).
  - mean and p95 of the systematic-error fraction
    |altitude_true - altitude_initial| / altitude_initial.

Also statically audits utils/dataloader.py to confirm which altitude source
the GT (t/d) computation uses (the hypothesis presumes option (a) — the
scalar initial altitude).

Read-only. Writes summary JSON to outputs/audit/altitude_variance.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


DEFAULT_TRAJS = (0, 1, 2, 3, 100, 101, 102)


def _traj_altitude_stats(traj_dir: Path) -> dict:
    poses = np.load(traj_dir / "poses_full.npy")
    z = poses[:, 2].astype(np.float64)
    meta = json.loads((traj_dir / "metadata.json").read_text())
    alt_init_meta = float(meta.get("altitude_initial",
                                   meta.get("altitude_m", float("nan"))))
    alt_init_from_pose = float(z[0])

    systematic_err_frac = np.abs(z - alt_init_meta) / max(1e-6, alt_init_meta)
    return {
        "traj_id": int(meta.get("trajectory_id", -1)),
        "n_samples": int(z.shape[0]),
        "altitude_initial_meta": alt_init_meta,
        "altitude_initial_from_pose_z0": alt_init_from_pose,
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_median": float(np.median(z)),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "z_std_over_mean": float(z.std() / max(1e-6, z.mean())),
        "z_max_over_min": float(z.max() / max(1e-6, z.min())),
        "systematic_err_frac_mean": float(systematic_err_frac.mean()),
        "systematic_err_frac_p95":  float(np.quantile(systematic_err_frac, 0.95)),
        "systematic_err_frac_max":  float(systematic_err_frac.max()),
    }


def _audit_dataloader_altitude_source() -> dict:
    """Read utils/dataloader.py to classify the altitude source in GT (t/d).

    Returns {"option": "a|b|c", "evidence": [lines]}.
    """
    path = HERE / "utils" / "dataloader.py"
    txt = path.read_text()

    # Look for the function that returns depth for a sample.
    # Known spot: _altitude(meta) returns altitude_initial if present
    # (option a), and dataloader then uses `d = altitude` which flows into
    # the sample's "depth" field.
    evidence = []
    if re.search(r'_altitude\s*\(.*?\)[\s\S]*?altitude_initial', txt):
        evidence.append("_altitude() returns metadata['altitude_initial']")
    if re.search(r'd\s*=\s*altitude', txt):
        evidence.append("d = altitude (scalar per-trajectory)")
    if "poses_full[a, 2]" in txt or "poses_full[sample_idx, 2]" in txt:
        evidence.append("per-frame z-coordinate used for depth")

    if any("per-frame" in e for e in evidence):
        option = "b"
    elif any("altitude_initial" in e for e in evidence):
        option = "a"
    else:
        option = "c"
    return {"option": option, "evidence": evidence,
            "source_file": str(path.relative_to(HERE))}


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", type=str, default="data/raw")
    p.add_argument("--trajs", type=int, nargs="+", default=list(DEFAULT_TRAJS))
    p.add_argument("--out_dir", type=str, default="outputs/audit")
    return p.parse_args()


def main() -> int:
    args = _parse()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for tid in args.trajs:
        traj_dir = raw_dir / f"traj_{int(tid):04d}"
        if not traj_dir.is_dir():
            print(f"[S5] skip missing {traj_dir}")
            continue
        rows.append(_traj_altitude_stats(traj_dir))

    audit = _audit_dataloader_altitude_source()

    print("\n=== S5: Altitude variance ===")
    hdr = (f"{'tid':>4} | {'n':>6} | "
           f"{'z_min':>6} | {'z_max':>6} | "
           f"{'z_med':>6} | {'z_std':>6} | "
           f"{'std/mean':>8} | {'max/min':>7} | "
           f"{'|err|_p95':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['traj_id']:>4d} | "
              f"{r['n_samples']:>6d} | "
              f"{r['z_min']:>6.3f} | "
              f"{r['z_max']:>6.3f} | "
              f"{r['z_median']:>6.3f} | "
              f"{r['z_std']:>6.3f} | "
              f"{r['z_std_over_mean']:>8.4f} | "
              f"{r['z_max_over_min']:>7.3f} | "
              f"{r['systematic_err_frac_p95']:>9.4f}")

    print("\nDataloader altitude-source audit:")
    print(f"  option: ({audit['option']})  "
          f"(a=altitude_initial scalar, b=per-frame z, c=unknown)")
    for e in audit["evidence"]:
        print(f"  - {e}")

    if not rows:
        print("[S5] no trajectories processed — cannot verdict.")
        return 1

    med_stdmean = float(np.median([r["z_std_over_mean"] for r in rows]))
    uses_initial = (audit["option"] == "a")

    print(f"\n  median std/mean across trajs: {med_stdmean:.4f}  "
          f"(threshold 0.10)")
    print(f"  uses altitude_initial (option a)? {uses_initial}")

    if med_stdmean > 0.10 and uses_initial:
        verdict = "CONFIRMED"
        reason = ("altitude varies > 10% and dataloader uses "
                  "altitude_initial scalar — GT (t/d) has systematic noise")
    elif med_stdmean < 0.05 or not uses_initial:
        verdict = "UNLIKELY"
        reason = ("altitude nearly constant OR dataloader uses per-frame "
                  "altitude — no systematic bias")
    else:
        verdict = "PARTIAL"
        reason = ("5–10% altitude variation OR ambiguous source — "
                  "moderate systematic noise")

    print(f"\n[S5] VERDICT: {verdict}")
    print(f"  - {reason}")

    out_json = out_dir / "altitude_variance.json"
    out_json.write_text(json.dumps({
        "per_traj": rows,
        "dataloader_audit": audit,
        "median_std_over_mean": med_stdmean,
        "verdict": verdict,
        "verdict_reason": reason,
    }, indent=2))
    print(f"[S5] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
