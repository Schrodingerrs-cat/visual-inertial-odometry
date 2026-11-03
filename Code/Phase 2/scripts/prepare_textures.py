"""Normalize user-provided carpet texture images and split into train / test.

Input: arbitrary-format images in ``Data/textures_raw/``. Formats accepted:
``.jpg``, ``.jpeg``, ``.png``, ``.webp`` (case-insensitive). Exactly 5 images
are required for the 4-train / 1-test split.

Output:
    Data/textures_processed/
        tex_00.webp … tex_04.webp            lossless .webp, native aspect
        manifest.json                        original filenames + (W, H)
    Data/textures_train/
        tex_00.webp … tex_03.webp            one per training trajectory
    Data/textures_test/
        tex_04.webp                          shared by held-out-texture test trajs
    Data/textures_split.json                 which originals → train / test

Sizing policy ``stretched_per_image_v1``:
  * Textures keep their native aspect ratio (no center-crop).
  * Plane dimensions are derived per-image in blender/generate_data.py
    (longer edge = 28 m, shorter edge = 28 / aspect).
  * Images with longer edge < 1500 px are rejected — at 28 m plane that's
    ~54 px/m on the longer edge (below the 100 px/m soft target but
    usable). Carpets with longer edge ≥ 2800 px meet the soft target.

Invocation:
    python scripts/prepare_textures.py
        [--raw_dir Data/textures_raw]
        [--processed_dir Data/textures_processed]
        [--train_dir Data/textures_train]
        [--test_dir Data/textures_test]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List

from PIL import Image


ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# stretched_per_image_v1 constants — must match blender/generate_data.py.
PLANE_LONGER_M = 28.0
DENSITY_TARGET_PX_PER_M = 100.0                  # soft target (warnings only)
MIN_LONGER_EDGE_PX_HARD_REJECT = 1500            # ~54 px/m on a 28 m plane

NUM_TEXTURES_REQUIRED = 5
NUM_TRAIN = 4   # tex_00 … tex_03  (one per training trajectory)
NUM_TEST = 1    # tex_04           (held-out; shared by test trajs)

# Altitude we cite in the per-image report. Matches the trajectory-library
# sampler midpoint of U(1.5, 3.0).
REPORT_ALTITUDE_M = 2.0


def _scan_raw(raw_dir: Path) -> List[Path]:
    """Return files in raw_dir with accepted extensions, sorted alphabetically."""
    files = [p for p in raw_dir.iterdir()
             if p.is_file() and p.suffix.lower() in ACCEPTED_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert any mode (RGBA, L, P, CMYK, ...) to RGB."""
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def _link_or_copy(src: Path, dst: Path) -> None:
    """Prefer symlink; fall back to copy on systems that disallow symlinks."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel = os.path.relpath(src.resolve(), dst.parent.resolve())
        os.symlink(rel, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _plane_dims(pix_w: int, pix_h: int,
                plane_longer_m: float) -> tuple[float, float]:
    """Mirror of blender.generate_data.compute_plane_dims (local dup so this
    script can run without pulling bpy into its import graph)."""
    aspect = float(pix_w) / float(pix_h)
    if aspect >= 1.0:
        return plane_longer_m, plane_longer_m / aspect
    return plane_longer_m * aspect, plane_longer_m


def _effective_density(pix_w: int, pix_h: int,
                       plane_x: float, plane_y: float) -> float:
    """Short-edge px / short-edge m. Because UV is 1:1 stretched, this equals
    the long-edge density."""
    short_px = float(min(pix_w, pix_h))
    short_m = float(min(plane_x, plane_y))
    return short_px / max(short_m, 1e-6)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", type=str, default="Data/textures_raw")
    p.add_argument("--processed_dir", type=str,
                   default="Data/textures_processed")
    p.add_argument("--train_dir", type=str, default="Data/textures_train")
    p.add_argument("--test_dir", type=str, default="Data/textures_test")
    p.add_argument("--plane_longer_m", type=float, default=PLANE_LONGER_M,
                   help="Plane's longer edge in metres (stretched_per_image_v1).")
    p.add_argument("--density_target_px_per_m", type=float,
                   default=DENSITY_TARGET_PX_PER_M,
                   help="Pixel-density warning target (not a hard floor).")
    p.add_argument("--report_altitude_m", type=float,
                   default=REPORT_ALTITUDE_M,
                   help="Altitude used when citing effective px/m in the report.")
    return p.parse_args()


def main() -> int:
    args = _parse()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    if not raw_dir.exists() or not raw_dir.is_dir():
        sys.stderr.write(
            f"[prepare_textures] ERROR: raw_dir {raw_dir} does not exist.\n"
            f"  Place your {NUM_TEXTURES_REQUIRED} carpet images "
            f"(.jpg/.jpeg/.png/.webp) in that folder.\n")
        return 2

    sources = _scan_raw(raw_dir)
    if len(sources) != NUM_TEXTURES_REQUIRED:
        sys.stderr.write(
            f"[prepare_textures] ERROR: found {len(sources)} image(s) in "
            f"{raw_dir}; need exactly {NUM_TEXTURES_REQUIRED}.\n"
            f"  Accepted extensions: {sorted(ACCEPTED_EXTS)}\n")
        return 2

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"textures": []}
    rejections: List[str] = []
    report_rows: List[tuple] = []

    for i, src in enumerate(sources):
        dst = processed_dir / f"tex_{i:02d}.webp"
        with Image.open(src) as img:
            orig_w, orig_h = img.size
            orig_mode = img.mode
            rgb = _ensure_rgb(img)
            # Lossless .webp, highest compression effort. No resize, no crop.
            rgb.save(dst, format="WEBP", lossless=True, method=6)

        longer_px = max(orig_w, orig_h)
        aspect = float(orig_w) / float(orig_h) if orig_h > 0 else float("nan")
        plane_x, plane_y = _plane_dims(orig_w, orig_h, args.plane_longer_m)
        eff_density = _effective_density(orig_w, orig_h, plane_x, plane_y)
        density_warn = eff_density < args.density_target_px_per_m

        if longer_px < MIN_LONGER_EDGE_PX_HARD_REJECT:
            rejections.append(
                f"{src.name}: longer edge {longer_px} px < hard-reject floor "
                f"{MIN_LONGER_EDGE_PX_HARD_REJECT} px ({eff_density:.1f} px/m "
                f"on {args.plane_longer_m:.1f} m plane — features unusable).")

        manifest["textures"].append({
            "index": i,
            "normalized_name": dst.name,
            "original_filename": src.name,
            "original_mode": orig_mode,
            "original_resolution": [orig_w, orig_h],
            "native_dimensions": [orig_w, orig_h],
            "aspect_ratio": aspect,
            "plane_dims_meters": [float(plane_x), float(plane_y)],
            "plane_longer_m": float(args.plane_longer_m),
            "effective_density_px_per_m": eff_density,
            "density_target_px_per_m": float(args.density_target_px_per_m),
            "density_warn": bool(density_warn),
            "sizing_policy": "stretched_per_image_v1",
        })
        report_rows.append((
            src.name, dst.name,
            f"{orig_w}x{orig_h}",
            f"{aspect:.3f}",
            f"{plane_x:.2f}x{plane_y:.2f}",
            f"{eff_density:.1f}",
            "WARN" if density_warn else "ok",
        ))
        print(f"[prepare_textures] {src.name} -> {dst.name}  "
              f"native={orig_w}x{orig_h} (aspect {aspect:.3f})  "
              f"plane={plane_x:.2f}x{plane_y:.2f} m  "
              f"density={eff_density:.1f} px/m"
              f"{' WARN' if density_warn else ''}")

    if rejections:
        sys.stderr.write(
            "[prepare_textures] ERROR: some images are too low-resolution "
            f"for the {args.plane_longer_m:.1f} m plane:\n")
        for r in rejections:
            sys.stderr.write(f"  - {r}\n")
        return 2

    print()
    print("[prepare_textures] stretched_per_image_v1 summary "
          f"(altitude report @ {args.report_altitude_m:.1f} m):")
    header = (f"  {'original':<58} {'normalized':<12} {'native':<12} "
              f"{'aspect':<7} {'plane (m)':<14} {'px/m':<6} note")
    print(header)
    for row in report_rows:
        print(f"  {row[0]:<58} {row[1]:<12} {row[2]:<12} "
              f"{row[3]:<7} {row[4]:<14} {row[5]:<6} {row[6]}")

    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Split: first 4 (sorted) → train, last 1 → test.
    split = {"train": [], "test": []}
    for i in range(NUM_TRAIN):
        name = f"tex_{i:02d}.webp"
        _link_or_copy(processed_dir / name, train_dir / name)
        split["train"].append({
            "normalized_name": name,
            "original_filename": manifest["textures"][i]["original_filename"],
        })
    for j in range(NUM_TEST):
        i = NUM_TRAIN + j
        name = f"tex_{i:02d}.webp"
        _link_or_copy(processed_dir / name, test_dir / name)
        split["test"].append({
            "normalized_name": name,
            "original_filename": manifest["textures"][i]["original_filename"],
        })

    split_path = Path(args.raw_dir).parent / "textures_split.json"
    split_path.write_text(json.dumps(split, indent=2))

    print(f"[prepare_textures] done.")
    print(f"  processed: {processed_dir}  ({NUM_TEXTURES_REQUIRED} files)")
    print(f"  train:     {train_dir}  ({NUM_TRAIN} files)")
    print(f"  test:      {test_dir}   ({NUM_TEST} file)")
    print(f"  split:     {split_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
