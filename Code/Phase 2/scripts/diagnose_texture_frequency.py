"""S1 diagnostic — texture frequency content.

Hypothesis (S1): rendered frames lack high-frequency local features, so the
CNN has no distinctive patches to match between frame_a and frame_b.

On 5 sample frames from traj_0000 this script reports, per frame:
  - Sharpness score (Laplacian variance).
  - Harris corner density (corners per 1000 px).
  - Edge density % (strong Canny pixels).
  - Fraction of FFT energy above 0.25 × Nyquist (radial).

A radial-FFT-power plot is saved to outputs/audit/texture_frequency_analysis.png.

Verdict rules (printed at end):
  - Harris density < 5 corners/1000px on most frames → S1 CONFIRMED.
  - FFT energy > 0.25 Nyquist fraction < 10% → S1 CONFIRMED.
  - Otherwise → S1 UNLIKELY.

The script is read-only: it does not write anywhere outside outputs/audit/.
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

SAMPLE_FRAMES = (0, 1000, 3000, 5000, 8000)
HARRIS_K = 0.04
HARRIS_THRESHOLD_FRAC = 0.01       # corners where response > 1% of peak
CANNY_LOW, CANNY_HIGH = 100, 200
NYQUIST_CUTOFF_FRAC = 0.25         # report fraction of energy > 0.25*Nyquist


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cv2.imread returned None for {path}")
    return img


def _radial_power_spectrum(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (radii_px, mean_power) where radii goes from 0 .. min(H,W)/2."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    power = np.abs(f) ** 2
    H, W = gray.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    r_max = min(cy, cx)
    radial = np.zeros(r_max + 1, dtype=np.float64)
    counts = np.zeros(r_max + 1, dtype=np.int64)
    mask = r <= r_max
    flat_r = r[mask]
    flat_p = power[mask]
    np.add.at(radial, flat_r, flat_p)
    np.add.at(counts, flat_r, 1)
    radial = radial / np.maximum(counts, 1)
    return np.arange(r_max + 1), radial


def _harris_density_per_1000px(gray: np.ndarray) -> float:
    g32 = np.float32(gray)
    resp = cv2.cornerHarris(g32, blockSize=2, ksize=3, k=HARRIS_K)
    # resp can be +/-; corners are local maxima with response > frac * peak
    peak = resp.max() if resp.size else 0.0
    if peak <= 0.0:
        return 0.0
    mask = resp > (HARRIS_THRESHOLD_FRAC * peak)
    n_corners = int(mask.sum())
    n_px = gray.size
    return 1000.0 * n_corners / max(1, n_px)


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _canny_edge_density_pct(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    return 100.0 * float((edges > 0).sum()) / max(1, gray.size)


def _fraction_hf_energy(radii: np.ndarray, radial_power: np.ndarray,
                       cutoff_frac: float = NYQUIST_CUTOFF_FRAC) -> float:
    if radii.size == 0:
        return float("nan")
    nyquist = radii[-1]
    cutoff = cutoff_frac * nyquist
    total = (radial_power * radii).sum()  # weight by annulus circumference ~r
    hi = (radial_power[radii >= cutoff] * radii[radii >= cutoff]).sum()
    return float(hi / total) if total > 0 else 0.0


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", type=str, default="data/raw",
                   help="Root containing traj_XXXX/ directories.")
    p.add_argument("--traj_id", type=int, default=0,
                   help="Trajectory id to sample from (default: 0).")
    p.add_argument("--out_dir", type=str, default="outputs/audit",
                   help="Where to write the plot.")
    return p.parse_args()


def main() -> int:
    args = _parse()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_dir = raw_dir / f"traj_{int(args.traj_id):04d}"
    frames_dir = traj_dir / "frames"
    if not frames_dir.is_dir():
        print(f"[S1] frames dir missing: {frames_dir}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    fig, ax = plt.subplots(figsize=(7, 5))

    for k in SAMPLE_FRAMES:
        fpath = frames_dir / f"{k:05d}.png"
        if not fpath.exists():
            print(f"[S1] skip missing frame {fpath}")
            continue
        gray = _load_gray(fpath)
        lap_var = _laplacian_variance(gray)
        harris = _harris_density_per_1000px(gray)
        edge_pct = _canny_edge_density_pct(gray)
        radii, power = _radial_power_spectrum(gray)
        hf_frac = _fraction_hf_energy(radii, power)
        rows.append({
            "frame": k,
            "sharpness": lap_var,
            "harris_per_1000px": harris,
            "edge_density_pct": edge_pct,
            "hf_fraction": hf_frac,
        })
        # Log-scale radial power, normalized so plots are comparable.
        if power.sum() > 0:
            ax.plot(radii / max(1, radii[-1]), np.log10(power + 1e-12),
                    label=f"frame {k:05d}")

    ax.set_xlabel("normalized radial frequency (0 = DC, 1 = Nyquist)")
    ax.set_ylabel("log10 mean radial power")
    ax.set_title(f"traj_{args.traj_id:04d} — radial FFT power spectrum")
    ax.axvline(NYQUIST_CUTOFF_FRAC, linestyle="--", color="black",
               alpha=0.5, linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = out_dir / "texture_frequency_analysis.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    print("\n=== S1: Texture frequency content ===")
    hdr = f"{'frame':>7} | {'sharpness':>10} | {'harris/1k':>10} | {'edge%':>7} | {'HF_frac':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['frame']:>7d} | "
              f"{r['sharpness']:>10.3f} | "
              f"{r['harris_per_1000px']:>10.3f} | "
              f"{r['edge_density_pct']:>7.3f} | "
              f"{r['hf_fraction']:>8.4f}")

    if not rows:
        print("[S1] no frames processed — cannot verdict.")
        return 1

    harris_vals = [r["harris_per_1000px"] for r in rows]
    hf_vals = [r["hf_fraction"] for r in rows]
    med_harris = float(np.median(harris_vals))
    med_hf = float(np.median(hf_vals))
    low_harris = sum(1 for v in harris_vals if v < 5.0)
    low_hf = sum(1 for v in hf_vals if v < 0.10)

    print("\nSummary:")
    print(f"  median Harris density: {med_harris:.3f} corners/1000px "
          f"(threshold 5.0; {low_harris}/{len(rows)} below)")
    print(f"  median HF energy fraction (>0.25 Nyquist): {med_hf:.4f} "
          f"(threshold 0.10; {low_hf}/{len(rows)} below)")

    verdict = "UNLIKELY"
    reasons: list[str] = []
    if low_harris >= (len(rows) + 1) // 2:
        verdict = "CONFIRMED"
        reasons.append("most frames have <5 Harris corners/1000px")
    if med_hf < 0.10:
        verdict = "CONFIRMED"
        reasons.append("median HF-fraction < 10%")

    print(f"\n[S1] VERDICT: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    print(f"[S1] wrote plot to {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
