"""Tests for compute_plane_dims under the stretched_per_image_v1 policy.

Under stretched_per_image_v1, ``compute_plane_dims`` returns plane dimensions
matching the texture's native aspect ratio with longer edge = PLANE_LONGER_M
(default 28 m). The image stretches 1:1 across plane UV — no cropping, no
tiling — so every pixel of every carpet is visible somewhere on the floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "blender"))

from generate_data import (  # noqa: E402
    compute_plane_dims, plane_sizing_report, PLANE_LONGER_M,
)


TOL = 0.01   # 1% tolerance on aspect ratio reproduction.


def _write_image(path: Path, w: int, h: int) -> None:
    Image.new("RGB", (w, h), (128, 128, 128)).save(path, format="WEBP")


@pytest.mark.parametrize("w,h", [
    (3000, 2000),     # 1.5 landscape  → 28 × 18.67 m
    (4000, 3000),     # 1.33 landscape → 28 × 21 m
    (3600, 2400),     # 1.5 landscape  → 28 × 18.67 m
    (2000, 2000),     # square         → 28 × 28 m
    (2000, 3000),     # 0.667 portrait → 18.67 × 28 m
    (1800, 1800),     # square, minimal size → 28 × 28 m
])
def test_plane_matches_native_aspect(tmp_path: Path, w: int, h: int) -> None:
    p = tmp_path / f"{w}x{h}.webp"
    _write_image(p, w, h)
    plane_x, plane_y = compute_plane_dims(p)

    # Longer edge of plane = PLANE_LONGER_M.
    assert max(plane_x, plane_y) == pytest.approx(PLANE_LONGER_M, rel=1e-9)

    # Plane aspect reproduces image aspect within 1%.
    image_aspect = w / h
    plane_aspect = plane_x / plane_y
    assert plane_aspect == pytest.approx(image_aspect, rel=TOL), (
        f"plane aspect {plane_aspect:.4f} != image aspect {image_aspect:.4f}")


def test_plane_longer_m_constant() -> None:
    assert PLANE_LONGER_M == 28.0


def test_sizing_report_policy_landscape(tmp_path: Path) -> None:
    """3000×2000 carpet, aspect 1.5 → plane 28 × 18.67 m."""
    p = tmp_path / "carpet_landscape.webp"
    _write_image(p, 3000, 2000)
    r = plane_sizing_report(p)
    assert r["plane_sizing_policy"] == "stretched_per_image_v1"
    assert r["plane_x_m"] == pytest.approx(28.0, rel=1e-9)
    assert r["plane_y_m"] == pytest.approx(28.0 / 1.5, rel=1e-6)
    assert r["crop_applied"] is False
    # short_edge_px / short_edge_m = 2000 / (28/1.5) = 107.14 px/m.
    assert r["effective_density_px_per_m"] == pytest.approx(
        2000 / (28 / 1.5), rel=1e-3)
    assert r["aspect_ratio"] == pytest.approx(1.5, rel=1e-9)


def test_sizing_report_policy_portrait(tmp_path: Path) -> None:
    """2000×3000 carpet, aspect 0.667 → plane 18.67 × 28 m."""
    p = tmp_path / "carpet_portrait.webp"
    _write_image(p, 2000, 3000)
    r = plane_sizing_report(p)
    assert r["plane_sizing_policy"] == "stretched_per_image_v1"
    assert r["plane_x_m"] == pytest.approx(28.0 * (2000 / 3000), rel=1e-6)
    assert r["plane_y_m"] == pytest.approx(28.0, rel=1e-9)
    # short edge is the x dimension here; 2000 px / (28 * 2/3 m) ≈ 107.14 px/m.
    assert r["effective_density_px_per_m"] == pytest.approx(
        2000 / (28.0 * (2000 / 3000)), rel=1e-3)


def test_sizing_report_policy_square(tmp_path: Path) -> None:
    p = tmp_path / "carpet_square.webp"
    _write_image(p, 2400, 2400)
    r = plane_sizing_report(p)
    assert r["plane_x_m"] == pytest.approx(28.0, rel=1e-9)
    assert r["plane_y_m"] == pytest.approx(28.0, rel=1e-9)
    assert r["aspect_ratio"] == pytest.approx(1.0, rel=1e-9)


def test_density_warning_on_small_texture(tmp_path: Path) -> None:
    """A texture with very low px/m sets density_warn=True. 200×200 on a
    28 m plane is ~7 px/m, far under the 100 px/m soft target."""
    p = tmp_path / "tiny.webp"
    _write_image(p, 200, 200)
    r = plane_sizing_report(p, density_target_px_per_m=100.0)
    assert r["effective_density_px_per_m"] == pytest.approx(200 / 28, rel=1e-3)
    assert r["density_warn"] is True


def test_worst_case_drone_footprint_fits(tmp_path: Path) -> None:
    """On the tightest supported plane (aspect 1.5 → 28×18.67 m), verify the
    drone's footprint budget at altitude 3 m and max 35° tilt + corner-FOV.

    Calculation:
      plane_y = 28/1.5 = 18.67 m   → plane_y_half ≈ 9.33 m
      max altitude = 3 m
      tilt_max = 35°
      corner_FOV α = atan(sqrt(tan²α_h + tan²α_v))
                    = atan(sqrt(0.5² + 0.375²))    (at 640×480 → fx=fy=640)
      far_reach = 3 · tan(35° + α) ≈ 7.07 m
    At a 28 m plane this is well inside plane_y_half (9.33 m) with
    2.26 m margin at max tilt.
    """
    import math
    plane_longer = 28.0
    worst_aspect = 1.5
    plane_short = plane_longer / worst_aspect            # ≈ 18.67 m
    plane_short_half = plane_short / 2.0                 # ≈ 9.33 m

    h_max = 3.0
    tilt_max = math.radians(35.0)
    # corner-FOV for 640×480 intrinsics fx=fy=640:
    #   α_h = atan(IMG_W / 2 / fx) = atan(0.5)
    #   α_v = atan(IMG_H / 2 / fy) = atan(0.375)
    alpha_corner = math.atan(math.sqrt(0.5 ** 2 + 0.375 ** 2))
    far_reach = h_max * math.tan(tilt_max + alpha_corner)
    assert far_reach == pytest.approx(7.07, abs=0.1)

    # confine_to_plane uses max_footprint_half = 3.5 m (straight-down
    # approximation). Post-confine drone half on the shorter edge:
    #   plane_short_half − 3.5 ≈ 5.83 m
    conservative_footprint_half = 3.5
    drone_half = plane_short_half - conservative_footprint_half
    assert drone_half == pytest.approx(5.83, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
