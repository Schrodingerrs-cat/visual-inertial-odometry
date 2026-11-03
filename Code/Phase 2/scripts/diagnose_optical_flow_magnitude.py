"""S3 diagnostic — optical flow magnitude between frame pairs.

Hypothesis (S3): the dataloader's consecutive PNG pair (which spans 10 IMU
samples) produces only 2–3 pixels of optical flow on average. A CNN typically
needs ~5–50 pixels of flow to learn reliable motion.

For each trajectory in [0, 1, 2, 3]:
  * Sample 100 random pairs at the natural dataloader stride (adjacent PNGs;
    "stride-10" = 10 IMU samples apart in time).
  * Also sample 100 pairs at PNG-stride 5 ("stride-50" = 50 IMU samples apart).
  * For every pair, compute dense Farneback flow and report:
      - median |flow| (px)
      - 95th-percentile |flow| (px)
      - fraction of sub-pixel pixels (|flow| < 1)

Saves a histogram to outputs/audit/flow_magnitude_analysis.png.

Verdict:
  - median stride-10 < 3 px AND median stride-50 in [10, 30] px → S3 CONFIRMED
  - median stride-10 > 5 px                                    → S3 UNLIKELY
  - median stride-50 < 5 px                                    → S3 "drone barely moves"

Note: the PNG index step of 1 corresponds to 10 IMU samples in the pipeline
(_IMU_WINDOW = 10, RENDER_STRIDE = 10). So "PNG stride 5" = "50 IMU samples".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_TRAJS = (0, 1, 2, 3)
N_PAIRS_PER_TRAJ = 100
PNG_STRIDE_SHORT = 1     # adjacent PNGs = 10 IMU samples ("stride-10")
PNG_STRIDE_LONG = 5      # 5 PNGs apart   = 50 IMU samples ("stride-50")


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cv2.imread returned None for {path}")
    return img


def _farneback_flow_mag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    return np.sqrt((flow ** 2).sum(axis=-1))


def _sample_pair_stats(frames_dir: Path, stride: int, n_pairs: int,
                       rng: np.random.Generator) -> dict:
    png_files = sorted(frames_dir.glob("*.png"))
    n_frames = len(png_files)
    if n_frames < stride + 2:
        return {"median": float("nan"), "p95": float("nan"),
                "sub_px_fraction": float("nan"),
                "n_pairs_used": 0, "per_pair_medians": np.zeros(0)}
    max_a = n_frames - 1 - stride
    ks = rng.integers(low=0, high=max_a + 1, size=n_pairs)
    per_pair_medians = np.zeros(n_pairs, dtype=np.float64)
    per_pair_p95 = np.zeros(n_pairs, dtype=np.float64)
    sub_px_sum = 0.0
    tot_px = 0
    for i, k in enumerate(ks):
        a = _load_gray(png_files[k])
        b = _load_gray(png_files[k + stride])
        mag = _farneback_flow_mag(a, b)
        per_pair_medians[i] = float(np.median(mag))
        per_pair_p95[i] = float(np.quantile(mag, 0.95))
        sub_px_sum += float((mag < 1.0).sum())
        tot_px += mag.size
    return {
        "median":          float(np.median(per_pair_medians)),
        "p95":             float(np.median(per_pair_p95)),
        "sub_px_fraction": sub_px_sum / max(1, tot_px),
        "n_pairs_used":    int(n_pairs),
        "per_pair_medians": per_pair_medians,
    }


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", type=str, default="data/raw")
    p.add_argument("--trajs", type=int, nargs="+", default=list(DEFAULT_TRAJS))
    p.add_argument("--n_pairs", type=int, default=N_PAIRS_PER_TRAJ)
    p.add_argument("--out_dir", type=str, default="outputs/audit")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = _parse()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    short_rows: list[dict] = []
    long_rows: list[dict] = []
    all_short_medians: list[float] = []
    all_long_medians: list[float] = []

    for tid in args.trajs:
        traj_dir = raw_dir / f"traj_{int(tid):04d}"
        frames_dir = traj_dir / "frames"
        if not frames_dir.is_dir():
            print(f"[S3] skip missing {frames_dir}")
            continue
        s = _sample_pair_stats(frames_dir, PNG_STRIDE_SHORT, args.n_pairs, rng)
        l = _sample_pair_stats(frames_dir, PNG_STRIDE_LONG,  args.n_pairs, rng)
        s_row = {"traj_id": tid, **{k: v for k, v in s.items()
                                    if k != "per_pair_medians"}}
        l_row = {"traj_id": tid, **{k: v for k, v in l.items()
                                    if k != "per_pair_medians"}}
        short_rows.append(s_row)
        long_rows.append(l_row)
        all_short_medians.extend(s["per_pair_medians"].tolist())
        all_long_medians.extend(l["per_pair_medians"].tolist())

    # Histogram plot.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(1.0,
                              max(all_short_medians + all_long_medians + [1.0])),
                       50)
    if all_short_medians:
        ax.hist(all_short_medians, bins=bins, alpha=0.6,
                label=f"stride-10 (adjacent PNG)", color="tab:blue")
    if all_long_medians:
        ax.hist(all_long_medians, bins=bins, alpha=0.6,
                label=f"stride-50 (5 PNG apart)", color="tab:orange")
    ax.axvline(3.0, linestyle="--", color="red", alpha=0.7,
               label="3 px (S3 threshold)")
    ax.axvline(5.0, linestyle=":", color="green", alpha=0.7,
               label="5 px (CNN floor)")
    ax.set_xlabel("per-pair median |flow| (pixels)")
    ax.set_ylabel("pair count")
    ax.set_title("Farneback optical-flow magnitude (trajs {})".format(args.trajs))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = out_dir / "flow_magnitude_analysis.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    # Print tables.
    print("\n=== S3: Optical flow magnitude ===")
    hdr = f"{'traj':>5} | {'stride':>7} | {'med_px':>8} | {'p95_px':>8} | {'subpx%':>7} | n"
    print(hdr); print("-" * len(hdr))
    for r in short_rows:
        print(f"{r['traj_id']:>5d} | {'10':>7s} | "
              f"{r['median']:>8.3f} | {r['p95']:>8.3f} | "
              f"{r['sub_px_fraction']*100:>6.2f}% | {r['n_pairs_used']}")
    for r in long_rows:
        print(f"{r['traj_id']:>5d} | {'50':>7s} | "
              f"{r['median']:>8.3f} | {r['p95']:>8.3f} | "
              f"{r['sub_px_fraction']*100:>6.2f}% | {r['n_pairs_used']}")

    if not short_rows:
        print("[S3] no trajectories processed — cannot verdict.")
        return 1

    med_short = float(np.median([r["median"] for r in short_rows]))
    med_long = float(np.median([r["median"] for r in long_rows])) \
        if long_rows else float("nan")

    print("\nSummary (median across trajectories):")
    print(f"  stride-10 median |flow|: {med_short:.3f} px")
    print(f"  stride-50 median |flow|: {med_long:.3f} px")

    verdict = "UNLIKELY"
    reasons: list[str] = []
    if med_short < 3.0 and 10.0 <= med_long <= 30.0:
        verdict = "CONFIRMED"
        reasons.append("stride-10 < 3 px and stride-50 in [10, 30] px")
    if med_short > 5.0:
        verdict = "UNLIKELY"
        reasons.append("stride-10 > 5 px — CNN already has usable signal")
    if not np.isnan(med_long) and med_long < 5.0:
        verdict = "DRONE BARELY MOVES"
        reasons.append("even stride-50 < 5 px — upstream sampler issue")

    print(f"\n[S3] VERDICT: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    print(f"[S3] wrote plot to {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
