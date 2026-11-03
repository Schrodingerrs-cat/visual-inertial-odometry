"""Headless Blender data generator for Phase 2.

Renders a single trajectory for a down-facing camera hovering over a plane
textured with one of the 5 user-provided carpet photographs. Invoked as:

    blender --background --python blender/generate_data.py -- \\
            --trajectory_id N --output_dir data/raw

Trajectory ID dispatch:
  0..3    → training, family = "sinusoid",    seed = 1000 + trajectory_id
  100     → test,     family = "flat_spiral", seed = 2000
  101     → test,     family = "figure_8",    seed = 2001
  102     → test,     family = "clover",      seed = 2002
  200     → validation (minitest, short clover), seed = 3000

Texture assignment (4-train / 1-test — see ``get_texture_for_trajectory``):
  traj 0   → textures_train/tex_00.webp
  traj 1   → textures_train/tex_01.webp
  traj 2   → textures_train/tex_02.webp
  traj 3   → textures_train/tex_03.webp
  traj 100 → textures_test/tex_04.webp    (held-out texture, unseen shape)
  traj 101 → textures_test/tex_04.webp    (held-out texture, unseen shape)
  traj 102 → textures_train/tex_00.webp   (seen texture, unseen shape)
  traj 200 → textures_test/tex_04.webp    (minitest validation; held-out)

Plane sizing policy ``stretched_per_image_v1``: per-image dimensions derived
from the native aspect ratio with longer edge = 18 m. Textures stretch 1:1
across plane UV (no cropping, no tiling). Trajectories are confined to the
plane with a camera-footprint margin.

Render target 640×480 via BLENDER_EEVEE with taa_render_samples=64.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

# bpy / mathutils only available inside Blender's Python. Guard imports so
# static analysis / unit imports outside Blender do not choke.
try:
    import bpy  # type: ignore
    from mathutils import Euler, Matrix, Vector  # type: ignore
except ImportError:  # pragma: no cover — meant to run inside Blender.
    bpy = None  # type: ignore
    Euler = Matrix = Vector = None  # type: ignore

# Make Phase2/blender importable whether we are run by Blender (cwd = Phase 2
# root) or directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from trajectory_library import (  # noqa: E402
    DT, DURATION_S, FPS, GRAVITY, N_SAMPLES,
    attitude_from_acceleration, generate_by_family, validate_trajectory,
)
from imu_simulation import simulate_imu  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RENDER_STRIDE = 10

# Render target 640×480 with 53° HFOV preserved from the 320×240 K.
# fx = IMG_W / (2·tan(HFOV/2)) = 640 / (2·tan(26.5°)) ≈ 640.
IMG_W, IMG_H = 640, 480
FX = FY = 640.0
CX, CY = 320.0, 240.0
K_MATRIX = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

# Plane-sizing policy: stretched_per_image_v1.
#
# Plane dimensions derived per-texture from the native aspect ratio, with
# longer edge = PLANE_LONGER_M = 28 m. Image content stretches 1:1 across
# plane UV (no cropping, no tiling). For typical carpets (aspect ~1.5), the
# plane is ~28 × 18.7 m. The narrowest supported aspect is 1.0 (square)
# which gives 28 × 28 m.
#
# Raised 18 → 28 m on 2026-04-24. At 18 m plane, flat_spiral (r_max ≈ 7 m)
# got confinement_scale ≈ 0.571; since confine_to_plane scales positions
# uniformly, velocity scales by the same factor → median optical flow
# dropped to ~3.3 px/frame, below S3's ~5 px floor. At 28 m plane,
# flat_spiral scale ≈ 0.89 → flow ≈ 5.1 px/frame.
#
# Known limitation: confine_to_plane's formula
# scale = available / (bbox + 2·footprint_half) doesn't fully guarantee
# footprint-inside-plane (it treats footprint as if scalable together with
# position). At 28 m plane this is modest — worst case on the narrow edge
# is ~1.3 m footprint overshoot past the plane edge during the flat_spiral
# outer phase, rendering gray background at one edge on ~5–15 % of traj
# 100 frames. Not fixed here: fixing the formula drops flat_spiral motion
# below the CNN floor. Accepted as a secondary artifact.
PLANE_LONGER_M = 28.0
DENSITY_TARGET_PX_PER_M_DEFAULT = 100.0
SIGMA_ACC = 0.02
SIGMA_GYRO = 0.002
GRAVITY_WORLD = np.array([0.0, 0.0, -GRAVITY])

# Trajectory ID → (split, family, seed) dispatch table.
#
# Step K (revised design 2026-05-01) extends the tuple to FOUR elements for
# new entries: (split, family, seed, target_peak_speed). Legacy 3-tuples
# (steps F/G/H/I/J — entries 0..307) keep their original form; ``_dispatch``
# pads them with ``target_peak_speed=None`` so the default sampler behavior
# is preserved.
TRAJ_ID_MAP = {
    0:   ("train", "sinusoid",       1000),
    1:   ("train", "sinusoid",       1001),
    2:   ("train", "sinusoid",       1002),
    3:   ("train", "sinusoid",       1003),
    100: ("test",  "flat_spiral",    2000),
    101: ("test",  "figure_8",       2001),
    102: ("test",  "clover",         2002),
    200: ("val",   "minitest",       3000),
    # Step S2 (2026-05-03) — IDs 300-307 retuned via target_peak_speed for
    # the corrected IMU pipeline (coordinated-turn attitude + analytic
    # vel/acc + R-chain gyro). Family + seed kept identical per ID; only
    # peak |v_xy| changes. Locked table from the audit report:
    #   train (301-304):  0.5 / 0.8 / 1.0 / 1.5 m/s across all 3 families.
    #   val   (305):      1.0 m/s on figure8 (in-distribution).
    #   test  (300/306/307): 0.3 / 0.5 / 1.2 m/s (OOD slow / in-dist / OOD fast).
    300: ("test",  "simple_figure8", 2003, 0.3),  # OOD slow
    301: ("train", "simple_figure8", 4001, 0.5),
    302: ("train", "simple_circle",  4002, 0.8),
    303: ("train", "simple_ellipse", 4003, 1.0),
    304: ("train", "simple_figure8", 4004, 1.5),
    # Held-out validation (best.pt selector):
    305: ("val",   "simple_figure8", 4005, 1.0),
    # Held-out within-family tests:
    306: ("test",  "simple_circle",  4006, 0.5),  # in-distribution
    307: ("test",  "simple_ellipse", 4007, 1.2),  # OOD fast
    # ----------------------------------------------------------------
    # Step K (revised) — structural-family + speed-diversity training.
    # ----------------------------------------------------------------
    # Step J's V failed cross-family (10.85 m on the original aggressive
    # sinusoids). Diagnosis: V never saw fast motion AND never saw shapes
    # with corners. Step K therefore expands training along two axes
    # simultaneously — shape (circle / figure-8 / diamond / zigzag) and
    # peak motion magnitude (0.1 — 0.5 m/s). Held-out tests use NEW
    # shapes (ellipse, triangle) at both seen and unseen speeds plus a
    # NEW shape (30 s controlled-speed sinusoid, traj 324) that has
    # never appeared in any prior step — the cleanest probe of shape
    # generalisation toward sinusoid-family motion at controlled
    # magnitude (decoupling shape from length / aggressive accel).
    #
    # Format: (split, family, seed, target_peak_speed). target_peak_speed
    # is m/s and is forwarded to the speed-scalable family samplers.
    312: ("train", "simple_circle",       5001, 0.1 ),  # circle, gentle
    313: ("train", "simple_circle",       5002, 0.4 ),  # circle, fast
    314: ("train", "simple_figure8",      5003, 0.15),  # fig-8, gentle
    315: ("train", "simple_figure8",      5004, 0.5 ),  # fig-8, fast
    316: ("train", "simple_diamond",      5005, 0.2 ),  # diamond, mid
    317: ("train", "simple_diamond",      5006, 0.4 ),  # diamond, fast
    318: ("train", "simple_zigzag",       5007, 0.2 ),  # zigzag, mid
    319: ("train", "simple_zigzag",       5008, 0.5 ),  # zigzag, fast
    # ----- held-out validation (drives best.pt; new shape, mid-speed)
    320: ("val",   "simple_ellipse",      5009, 0.3 ),
    # ----- held-out tests
    321: ("test",  "simple_ellipse",      5010, 0.5 ),  # smooth, OOD speed
    322: ("test",  "simple_triangle",     5011, 0.3 ),  # pointy, mid-speed
    323: ("test",  "simple_triangle",     5012, 0.5 ),  # pointy, fast (OOD spd)
    324: ("test",  "simple_sinusoid_30s", 5013, 0.5 ),  # NEW shape ablation
    # ----------------------------------------------------------------
    # Step K extended — held-out instances of the 4 *trained* shape
    # families at intermediate (unseen) speeds. Tests whether V's
    # within-family generalisation extends to ALL trained shapes
    # (not only ellipse / triangle / sinusoid_30s in the original
    # Step K test set). Seeds 6001-6004 do not collide with the
    # training seeds 5001-5008; speeds 0.25 / 0.30 lie in the gap
    # between trained speeds (0.10/0.15/0.20 vs 0.40/0.50).
    325: ("test",  "simple_figure8",     6001, 0.30),
    326: ("test",  "simple_circle",      6002, 0.25),
    327: ("test",  "simple_diamond",     6003, 0.30),
    328: ("test",  "simple_zigzag",      6004, 0.30),
    # ----------------------------------------------------------------
    # Step S — speed-extrapolation test for V/I/VI on Step K's
    # architecture. Per-family speed ceilings vary so the network sees
    # a graded distribution across motion magnitudes within training,
    # then is tested on a single held-out shape (simple_ellipse) at a
    # speed sweep that extends beyond every per-family training ceiling.
    # Same texture (tex_03), same 30 s length, same hyperparameters as
    # Step K. See configs/step_s.yaml and slurm/step_s_*.sh.
    #
    # Shape names use the canonical codebase forms (simple_figure8,
    # simple_sinusoid_30s) — generate_by_family lookup is case+spelling
    # sensitive.
    #
    # Train (8): 4 shapes × 2 speeds each. Speed ceilings differ per
    # family — circle reaches 1.0 m/s, the others top out at 0.8.
    # Step S2 (2026-05-03) — speed retune. Train spans 0.5 / 0.8 / 1.0 / 1.5
    # m/s, each speed appearing twice across different families so the
    # network can't memorise speed-shape pairs. Test sweeps the full
    # envelope (0.3 OOD-slow, 0.5 / 0.8 / 1.2 in-range) on the held-out
    # ellipse shape. Pre-S2 ceiling was 1.0 m/s on circle only (1.2 m/s
    # OOD); post-S2 ceiling is 1.5 m/s on circle + diamond (1.2 m/s now
    # interpolation).
    400: ("train", "simple_circle",       7001, 0.5),
    401: ("train", "simple_circle",       7002, 1.2),  # 1.5 → 1.2 (plane fit)
    402: ("train", "simple_figure8",      7003, 0.5),
    403: ("train", "simple_figure8",      7004, 1.0),
    404: ("train", "simple_diamond",      7005, 0.8),
    405: ("train", "simple_diamond",      7006, 1.05), # diamond plane ceiling, seed-dependent
    406: ("train", "simple_zigzag",       7007, 0.8),
    407: ("train", "simple_zigzag",       7008, 1.0),
    # Val / minitest — held-out ellipse @ 1.0 m/s (mid-range).
    408: ("val",   "simple_ellipse",      7009, 1.0),
    # Test — speed sweep on held-out ellipse: one OOD-slow, three in-range.
    410: ("test",  "simple_ellipse",      7010, 0.3),  # OOD slow
    411: ("test",  "simple_ellipse",      7011, 0.5),  # in-range
    412: ("test",  "simple_ellipse",      7012, 0.8),  # in-range
    413: ("test",  "simple_ellipse",      7013, 1.2),  # in-range (was OOD pre-S2)
    # Step S extra test — held-out simple_triangle at five speeds spanning
    # OOD-slow → train ceiling → OOD-fast. Triangle was never trained on at
    # Step S (Step K used it at 0.3/0.5 m/s under different IDs); these
    # entries probe shape generalisation across the Step S speed envelope.
    # Seeds 7020-7024 do not collide with any prior step. Same tex_03.
    420: ("test",  "simple_triangle",     7020, 0.3),  # OOD slow
    421: ("test",  "simple_triangle",     7021, 0.8),  # in-range
    422: ("test",  "simple_triangle",     7022, 1.0),  # in-range
    423: ("test",  "simple_triangle",     7023, 1.2),  # in-range
    424: ("test",  "simple_triangle",     7024, 1.5),  # OOD fast
}

# Texture assignment. 5 carpets total: tex_00..03 in train pool (one per
# training trajectory), tex_04 in test pool (shared by all held-out-texture
# trajectories). Traj 102 reuses a train texture (seen texture, unseen shape).
TEXTURE_ASSIGNMENT = {
    0:   ("train", "tex_00.webp"),
    1:   ("train", "tex_01.webp"),
    2:   ("train", "tex_02.webp"),
    3:   ("train", "tex_03.webp"),
    100: ("test",  "tex_04.webp"),  # held-out texture, unseen shape
    101: ("test",  "tex_04.webp"),  # held-out texture, unseen shape
    102: ("train", "tex_00.webp"),  # seen texture, unseen shape (intentional)
    200: ("test",  "tex_04.webp"),  # minitest validation (held-out)
    300: ("train", "tex_03.webp"),  # Step I: simple_figure8, render with
                                     # tex_03 to match data/raw_g/ texture
                                     # (forced via --force_texture as well).
    # Step J — all simple-shape trajectories share tex_03 (lives in
    # textures_train pool). Render scripts also pass --force_texture.
    301: ("train", "tex_03.webp"),
    302: ("train", "tex_03.webp"),
    303: ("train", "tex_03.webp"),
    304: ("train", "tex_03.webp"),
    305: ("train", "tex_03.webp"),
    306: ("train", "tex_03.webp"),
    307: ("train", "tex_03.webp"),
    # Step K (revised) — structural-family + speed-diversity training set.
    # All trajs share tex_03 so cross-traj diffs are purely trajectory
    # (shape × speed), not texture. Trajs 312-324 cover 4 train shapes ×
    # 2 speeds (gentle / fast) = 8 training trajs (312-319), plus 5
    # held-outs (320-324) including the new sinusoid_30s shape (324).
    312: ("train", "tex_03.webp"),
    313: ("train", "tex_03.webp"),
    314: ("train", "tex_03.webp"),
    315: ("train", "tex_03.webp"),
    316: ("train", "tex_03.webp"),
    317: ("train", "tex_03.webp"),
    318: ("train", "tex_03.webp"),
    319: ("train", "tex_03.webp"),
    320: ("train", "tex_03.webp"),
    321: ("train", "tex_03.webp"),
    322: ("train", "tex_03.webp"),
    323: ("train", "tex_03.webp"),
    324: ("train", "tex_03.webp"),
    # Step K extended — same texture as the rest of Step K.
    325: ("train", "tex_03.webp"),
    326: ("train", "tex_03.webp"),
    327: ("train", "tex_03.webp"),
    328: ("train", "tex_03.webp"),
    # Step S — all on tex_03 to keep texture as a constant; the only
    # variable vs Step K is target_peak_speed.
    400: ("train", "tex_03.webp"),
    401: ("train", "tex_03.webp"),
    402: ("train", "tex_03.webp"),
    403: ("train", "tex_03.webp"),
    404: ("train", "tex_03.webp"),
    405: ("train", "tex_03.webp"),
    406: ("train", "tex_03.webp"),
    407: ("train", "tex_03.webp"),
    408: ("train", "tex_03.webp"),
    410: ("train", "tex_03.webp"),
    411: ("train", "tex_03.webp"),
    412: ("train", "tex_03.webp"),
    413: ("train", "tex_03.webp"),
    # Step S extra test — simple_triangle speed sweep, same texture pool.
    420: ("train", "tex_03.webp"),
    421: ("train", "tex_03.webp"),
    422: ("train", "tex_03.webp"),
    423: ("train", "tex_03.webp"),
    424: ("train", "tex_03.webp"),
}

# Subdirectories under the Phase 2 root where normalized textures live.
TEXTURE_SUBDIRS = {
    "train": "Data/textures_train",
    "test":  "Data/textures_test",
}


# ---------------------------------------------------------------------------
# Texture assignment and plane dim helpers
# ---------------------------------------------------------------------------

def _phase2_root() -> Path:
    """Walk up from this file to find the Phase 2 project root."""
    return _HERE.parent


def get_texture_for_trajectory(trajectory_id: int,
                               project_root: Path | None = None) -> Path:
    """Return the absolute path to the .webp texture for the given trajectory.

    Raises AssertionError if the resolved file does not exist.
    """
    project_root = Path(project_root) if project_root is not None else _phase2_root()
    if trajectory_id not in TEXTURE_ASSIGNMENT:
        raise KeyError(
            f"trajectory_id {trajectory_id} has no texture assignment. "
            f"Valid IDs: {sorted(TEXTURE_ASSIGNMENT.keys())}")
    pool, fname = TEXTURE_ASSIGNMENT[trajectory_id]
    path = project_root / TEXTURE_SUBDIRS[pool] / fname
    assert path.exists(), (
        f"texture not found: {path}\n"
        f"  Did you run `python scripts/prepare_textures.py` after placing "
        f"7 images in Data/textures_raw/?")
    return path


def _resolve_forced_texture(name: str, project_root: Path) -> tuple[Path, str]:
    """Return (path, pool) for a forced-override texture basename.

    Searches train pool first, then test pool. Used by --force_texture so a
    caller can render multiple trajectories with one shared texture without
    editing TEXTURE_ASSIGNMENT (which is a project invariant referenced by
    sanity checks and other tooling).
    """
    for pool in ("train", "test"):
        candidate = project_root / TEXTURE_SUBDIRS[pool] / name
        if candidate.exists():
            return candidate, pool
    raise FileNotFoundError(
        f"--force_texture: {name!r} not found in either "
        f"{TEXTURE_SUBDIRS['train']} or {TEXTURE_SUBDIRS['test']}")


def compute_plane_dims(image_path: Path,
                       plane_longer_m: float = PLANE_LONGER_M,
                       ) -> tuple[float, float]:
    """Return (plane_x_m, plane_y_m) for the stretched_per_image_v1 policy.

    Plane dimensions are derived from the texture's native aspect ratio:
        aspect = image_W / image_H
        plane_longer_m = PLANE_LONGER_M          (configurable; default 18)
        if landscape (aspect ≥ 1):
            plane_W = plane_longer_m
            plane_H = plane_longer_m / aspect
        else (portrait, aspect < 1):
            plane_W = plane_longer_m * aspect
            plane_H = plane_longer_m

    Image content stretches 1:1 across plane UV. No cropping, no tiling.
    """
    from PIL import Image as _PILImage
    if not Path(image_path).exists():
        raise FileNotFoundError(f"texture not found: {image_path}")
    with _PILImage.open(image_path) as im:
        pix_w, pix_h = im.size
    if pix_h <= 0 or pix_w <= 0:
        raise ValueError(f"texture has invalid dimensions: {image_path}")
    aspect = float(pix_w) / float(pix_h)
    longer = float(plane_longer_m)
    if aspect >= 1.0:
        plane_x = longer
        plane_y = longer / aspect
    else:
        plane_x = longer * aspect
        plane_y = longer
    return plane_x, plane_y


def plane_sizing_report(image_path: Path,
                        density_target_px_per_m: float = DENSITY_TARGET_PX_PER_M_DEFAULT,
                        plane_longer_m: float = PLANE_LONGER_M,
                        ) -> dict:
    """Diagnostic dict for stretched_per_image_v1."""
    from PIL import Image as _PILImage
    with _PILImage.open(image_path) as im:
        pix_w, pix_h = im.size
    plane_x, plane_y = compute_plane_dims(image_path, plane_longer_m=plane_longer_m)
    # Effective density is "short-edge pixels per short-edge metre" — since
    # UV stretches 1:1, this is equivalent on the long edge.
    short_px = float(min(pix_w, pix_h))
    short_m = float(min(plane_x, plane_y))
    effective_density = short_px / max(short_m, 1e-6)
    aspect = float(pix_w) / float(pix_h) if pix_h > 0 else float("nan")
    return {
        "plane_sizing_policy": "stretched_per_image_v1",
        "pixel_size": [int(pix_w), int(pix_h)],
        "aspect_ratio": aspect,
        "plane_longer_m": float(plane_longer_m),
        "density_target_px_per_m": float(density_target_px_per_m),
        "plane_x_m": float(plane_x),
        "plane_y_m": float(plane_y),
        "effective_density_px_per_m": float(effective_density),
        "crop_applied": False,                      # never crop under this policy
        "density_warn": bool(effective_density < density_target_px_per_m),
    }


# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------

def _R_wb_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray
                   ) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    T = roll.shape[0]
    R = np.zeros((T, 3, 3), dtype=np.float64)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def _matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) → (N, 4) in [qx, qy, qz, qw] via Shepperd's method."""
    out = np.zeros((R.shape[0], 4))
    for i in range(R.shape[0]):
        m = R[i]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        out[i] = [qx, qy, qz, qw]
    return out


# ---------------------------------------------------------------------------
# Blender scene setup
# ---------------------------------------------------------------------------

def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _configure_render(out_dir: Path) -> None:
    """High-quality EEVEE at 640×480. Cycles is deliberately NOT used — the
    scene is a flat textured plane with diffuse shading; ray-tracing adds
    nothing here and costs ~20× render time.
    """
    scene = bpy.context.scene
    # Blender 4.2+ ships BLENDER_EEVEE_NEXT; earlier versions ship
    # BLENDER_EEVEE. Fall back gracefully.
    _eevee_engines = ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE')
    for _engine in _eevee_engines:
        try:
            scene.render.engine = _engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = IMG_W
    scene.render.resolution_y = IMG_H
    scene.render.resolution_percentage = 100
    scene.render.filter_size = 1.5                  # slightly soft AA filter
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'   # 8-bit PNG is enough
    scene.render.image_settings.compression = 15    # low compression = fast
    scene.render.use_motion_blur = False
    eevee = scene.eevee
    # taa_render_samples is the main EEVEE quality knob. 64 gives noticeably
    # cleaner samples than the default 32 without blowing up render time.
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 64
    if hasattr(eevee, "use_motion_blur"):
        eevee.use_motion_blur = False
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom = False                     # flat-plane scene; unused
    if hasattr(eevee, "use_gtao"):
        eevee.use_gtao = False                      # AO adds no structure here
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = False                       # no reflective surfaces
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'


def _add_camera_and_light() -> "bpy.types.Object":
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens_unit = 'MILLIMETERS'
    cam_data.sensor_width = 36.0
    cam_data.lens = 36.0                                  # 2·atan(18/36) ≈ 53° HFOV (resolution-independent)
    cam_data.clip_start = 0.1
    cam_data.clip_end = 1000.0
    cam_data.dof.use_dof = False
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new("Sun", type='SUN')
    light_data.energy = 3.0
    light = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, 0.0, 20.0)
    light.rotation_euler = (0.0, 0.0, 0.0)

    world = bpy.data.worlds.new("World") if "World" not in bpy.data.worlds else bpy.data.worlds["World"]
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.3, 0.3, 0.3, 1.0)
        bg.inputs[1].default_value = 1.0
    return cam_obj


def make_photo_material(image_path: Path) -> "bpy.types.Material":
    """Create a Blender material that maps ``image_path`` 1:1 onto the plane.

    Node graph: TexCoord(Generated) → Mapping(Scale 1,1,1) → ImageTexture
                (Linear interp, REPEAT extension) → Principled BSDF Base Color.
    The texture stretches to fill the plane. No tiling.
    """
    image_path = Path(image_path)
    mat_name = f"floor_{image_path.stem}_{uuid4().hex[:8]}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    tex_image = nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(image_path), check_existing=True)
    try:
        img.colorspace_settings.name = "sRGB"
    except Exception:  # pragma: no cover — old Blender builds
        pass
    tex_image.image = img
    # Cubic filtering preserves high-frequency detail (carpets) under oblique
    # views better than Linear, at negligible render cost. See S1 diagnostic.
    tex_image.interpolation = "Cubic"
    tex_image.extension = "EXTEND"                  # no tiling — plane UV is 1:1
    links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])
    links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])

    def _set_if_has(node, key: str, value) -> None:
        if key in node.inputs:
            node.inputs[key].default_value = value

    _set_if_has(bsdf, "Roughness", 0.6)
    _set_if_has(bsdf, "Specular", 0.3)
    _set_if_has(bsdf, "Specular IOR Level", 0.3)

    return mat


def _add_photo_plane(plane_x: float, plane_y: float, image_path: Path):
    """Create an axis-aligned plane with the photo material applied."""
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.scale = (plane_x / 2.0, plane_y / 2.0, 1.0)
    mat = make_photo_material(image_path)
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    return plane


# ---------------------------------------------------------------------------
# Camera pose placement
# ---------------------------------------------------------------------------

def _blender_camera_euler(roll_i: float, pitch_i: float, yaw_i: float):
    """Camera rotation as a Blender Euler.

    Blender cameras at rotation (0, 0, 0) already look along world -Z (down),
    since the camera's local forward is -Z_local and at identity local axes
    equal world axes. For a drone whose body frame matches world frame at
    identity hover (ENU), the camera's world pose IS the body's RPY — no
    offset needed.

    The previous implementation left-multiplied by rotation_x(π) ("down-facing
    offset"), which actually flipped the camera to look UP (+Z_world) at
    identity hover. Empirically verified with scripts/debug_blender_render.py:
    rotation (0,0,0) → forward (0,0,-1), rotation (π,0,0) → forward (0,0,+1).
    """
    return Euler((roll_i, pitch_i, yaw_i), 'XYZ')


def _place_camera(cam_obj, pos_i: np.ndarray, roll_i: float, pitch_i: float,
                  yaw_i: float) -> None:
    cam_obj.location = Vector((float(pos_i[0]), float(pos_i[1]), float(pos_i[2])))
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = _blender_camera_euler(roll_i, pitch_i, yaw_i)


def _render_frame(out_path: Path) -> None:
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _parse_cli_argv() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(
        description="Generate one Phase 2 trajectory via Blender.")
    p.add_argument("--trajectory_id", type=int, required=True,
                   help="Integer ID. 0..3 sinusoid train, 100/101/102 test, "
                        "200 minitest validation.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument("--force_texture", type=str, default=None,
                   help="Override TEXTURE_ASSIGNMENT for this trajectory. "
                        "Pass a basename (e.g. tex_03.webp); the script "
                        "searches Data/textures_train then Data/textures_test. "
                        "Used by Step G to render trajs 0-3 with one shared "
                        "texture for the disentangled within-family test.")
    return p.parse_args(argv)


def _dispatch(trajectory_id: int):
    """Return (split, family, seed, target_peak_speed). target_peak_speed
    defaults to None for legacy 3-tuple entries (Steps F/G/H/I/J)."""
    if trajectory_id not in TRAJ_ID_MAP:
        raise KeyError(
            f"trajectory_id {trajectory_id} has no family mapping. "
            f"Valid IDs: {sorted(TRAJ_ID_MAP.keys())}")
    entry = TRAJ_ID_MAP[trajectory_id]
    if len(entry) == 3:
        split, family, seed = entry
        return split, family, seed, None
    if len(entry) == 4:
        split, family, seed, target_peak_speed = entry
        return split, family, seed, target_peak_speed
    raise ValueError(
        f"TRAJ_ID_MAP[{trajectory_id}] has unexpected arity {len(entry)} "
        f"(expected 3 or 4): {entry!r}")


def _texture_manifest_entry(image_path: Path) -> dict:
    """Look up the original filename for a normalized texture from manifest.json."""
    manifest_path = image_path.parent.parent / "textures_processed" / "manifest.json"
    if not manifest_path.exists():
        return {"original_filename": "", "aspect_ratio": None}
    blob = json.loads(manifest_path.read_text())
    for entry in blob.get("textures", []):
        if entry["normalized_name"] == image_path.name:
            return entry
    return {"original_filename": "", "aspect_ratio": None}


def main() -> None:
    args = _parse_cli_argv()
    traj_id = int(args.trajectory_id)
    split, family, seed, target_peak_speed = _dispatch(traj_id)

    output_root = Path(args.output_dir)
    out_dir = output_root / f"traj_{traj_id:04d}"
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_if_exists and (out_dir / "metadata.json").exists():
        print(f"[gen] {out_dir} already complete — skipping.")
        return

    if target_peak_speed is not None:
        print(f"[gen] traj {traj_id} — split={split} family={family} "
              f"seed={seed} target_peak_speed={target_peak_speed} m/s")
    else:
        print(f"[gen] traj {traj_id} — split={split} family={family} "
              f"seed={seed} (no speed scaling)")

    project_root = _phase2_root()
    if args.force_texture:
        texture_path, texture_pool = _resolve_forced_texture(
            args.force_texture, project_root)
        print(f"[gen] --force_texture override: {args.force_texture} "
              f"(pool={texture_pool}; ignoring "
              f"TEXTURE_ASSIGNMENT[{traj_id}]={TEXTURE_ASSIGNMENT[traj_id]})")
    else:
        texture_path = get_texture_for_trajectory(traj_id, project_root)
        texture_pool = TEXTURE_ASSIGNMENT[traj_id][0]
    sizing = plane_sizing_report(
        texture_path,
        density_target_px_per_m=DENSITY_TARGET_PX_PER_M_DEFAULT,
        plane_longer_m=PLANE_LONGER_M,
    )
    plane_x = float(sizing["plane_x_m"])
    plane_y = float(sizing["plane_y_m"])
    print(f"[gen] texture: {texture_path.name} "
          f"-> plane {plane_x:.2f} x {plane_y:.2f} m "
          f"(stretched_per_image_v1, aspect {sizing['aspect_ratio']:.3f}, "
          f"density {sizing['effective_density_px_per_m']:.1f} px/m)")

    # Generate trajectory (already confined to the plane inside the sampler).
    # Step K: forward target_peak_speed to speed-scalable families.
    traj = generate_by_family(family, seed, plane_x, plane_y,
                              target_peak_speed=target_peak_speed)
    pos = traj["pos"]                                         # (T, 3)
    yaw_seq = traj["yaw"]                                     # (T,)
    n_samples = pos.shape[0]
    dt = DT
    ok, reason = validate_trajectory(traj, dt)
    if not ok:
        raise RuntimeError(f"trajectory validity re-check failed: {reason}")

    # Coupled attitude. The simple_* samplers (Step S — Option B) supply
    # analytic vel/acc; passing them through bypasses the finite-diff path
    # in attitude_from_acceleration entirely.
    traj_vel = traj.get("vel")
    traj_acc = traj.get("acc")
    att = attitude_from_acceleration(pos, yaw_seq, dt,
                                     vel=traj_vel, acc=traj_acc)
    roll  = att["roll"]
    pitch = att["pitch"]
    yaw   = att["yaw"]
    acc_world = att["acc_world"]
    ang_vel_body = att["ang_vel_body"]

    validity_stats = {
        "max_acc": float(np.max(np.linalg.norm(acc_world, axis=-1))),
        "median_dt": float(np.median(np.linalg.norm(
            pos[10:] - pos[:-10], axis=-1))),
        "max_roll_deg":  float(np.rad2deg(np.max(np.abs(roll)))),
        "max_pitch_deg": float(np.rad2deg(np.max(np.abs(pitch)))),
        "max_omega":     float(np.max(np.linalg.norm(ang_vel_body, axis=-1))),
    }

    # Pose quaternions (world<-body).
    R_wb = _R_wb_from_rpy(roll, pitch, yaw)
    quats = _matrix_to_quat_xyzw(R_wb)
    poses_full = np.concatenate([pos, quats], axis=-1)

    # IMU. Forward analytic vel/acc when present so the IMU pipeline stays
    # finite-diff-free on pos.
    imu = simulate_imu(pos, roll, pitch, yaw, dt,
                       sigma_acc=SIGMA_ACC, sigma_gyro=SIGMA_GYRO,
                       g=GRAVITY, seed=seed,
                       vel=traj_vel, acc=traj_acc)

    np.save(out_dir / "poses_full.npy", poses_full.astype(np.float64))
    np.save(out_dir / "imu.npy", imu.astype(np.float64))
    np.save(out_dir / "attitudes_full.npy",
            np.stack([roll, pitch, yaw], axis=-1).astype(np.float64))
    np.save(out_dir / "accel_world.npy", acc_world.astype(np.float64))

    # Blender scene.
    if bpy is None:
        raise RuntimeError("This script must be run inside Blender (bpy not importable).")

    _reset_scene()
    _configure_render(out_dir)
    cam_obj = _add_camera_and_light()
    _add_photo_plane(plane_x, plane_y, texture_path)

    altitude_initial = float(pos[0, 2])

    # Render every RENDER_STRIDE samples. For minitest (2001 samples) this is
    # still 201 rendered frames at 100 Hz.
    frame_indices = np.arange(0, n_samples, RENDER_STRIDE)
    n_frames = len(frame_indices)
    for k, idx in enumerate(frame_indices):
        _place_camera(cam_obj, pos[idx], float(roll[idx]), float(pitch[idx]),
                      float(yaw[idx]))
        _render_frame(frames_dir / f"{k:05d}.png")
        if (k % 50) == 0:
            print(f"[gen] traj {traj_id}: rendered {k+1}/{n_frames}",
                  flush=True)

    # Texture metadata from manifest.
    tex_manifest = _texture_manifest_entry(texture_path)
    # texture_pool is already resolved earlier (forced or from TEXTURE_ASSIGNMENT).
    texture_rel = texture_path.relative_to(project_root)
    image_pixel_dims = tex_manifest.get("original_resolution", [None, None])

    # Confinement diagnostics come back in traj["meta"].
    meta_traj = dict(traj.get("meta", {}))

    meta = {
        "trajectory_id": traj_id,
        "split": split,
        "family": family,
        "seed": seed,
        "n_samples": int(n_samples),
        "n_frames": int(n_frames),
        "fps_imu": FPS,
        "fps_image": FPS // RENDER_STRIDE,
        "K": K_MATRIX.tolist(),
        "image_size": [IMG_H, IMG_W],
        "texture_pool": texture_pool,
        "texture_path": str(texture_rel).replace("\\", "/"),
        "texture_name": texture_path.name,
        "texture_override": (args.force_texture if args.force_texture else None),
        "texture_assignment_original": list(TEXTURE_ASSIGNMENT[traj_id]),
        "texture_original_filename": tex_manifest.get("original_filename", ""),
        "image_pixel_dims": list(image_pixel_dims) if image_pixel_dims else [None, None],
        "plane_dims_meters": [float(plane_x), float(plane_y)],
        "plane_aspect_ratio": float(plane_x / plane_y) if plane_y > 0 else None,
        "plane_sizing_policy": sizing["plane_sizing_policy"],
        "plane_longer_m": sizing.get("plane_longer_m", float(max(plane_x, plane_y))),
        "density_target_px_per_m": sizing["density_target_px_per_m"],
        "effective_density_px_per_m": sizing["effective_density_px_per_m"],
        "crop_applied": sizing.get("crop_applied", False),
        "texture_pixel_size": sizing.get("pixel_size", [None, None]),
        "confinement_scale": float(meta_traj.get("confinement_scale", 1.0)),
        "bbox_xy_before": meta_traj.get("bbox_xy_before", None),
        "bbox_xy_after":  meta_traj.get("bbox_xy_after", None),
        "altitude_m": altitude_initial,  # legacy alias (some older tooling reads this)
        "altitude_initial": altitude_initial,
        "trajectory_params": meta_traj,
        "target_peak_speed": (float(target_peak_speed)
                              if target_peak_speed is not None else None),
        "empirical_peak_velocity": meta_traj.get("empirical_peak_velocity"),
        "speed_target_relative_error":
            meta_traj.get("speed_target_relative_error"),
        "validity_stats": validity_stats,
        "noise_params": {"sigma_acc": SIGMA_ACC, "sigma_gyro": SIGMA_GYRO},
        "gravity_world": GRAVITY_WORLD.tolist(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[gen] Wrote {out_dir} ({n_frames} frames).")


if __name__ == "__main__":
    main()
