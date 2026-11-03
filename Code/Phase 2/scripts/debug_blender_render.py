"""Single-frame Blender smoke test for the stretched_per_image_v1 + 640×480
rendering pipeline.

Invocation (inside Blender's Python):

    blender --background --python scripts/debug_blender_render.py -- \
        [--texture Data/textures_train/tex_00.webp] \
        [--altitude 2.0] [--output outputs/audit/debug_render.png]

Effect: reuses the production render pipeline (``blender/generate_data.py``):
reset scene, configure high-quality EEVEE at 640×480, build a photo plane
sized from the texture's native aspect, place a down-facing camera at the
requested altitude, render ONE frame, write it to ``--output``.

Purpose: verifies render quality before kicking off a full trajectory
generation shard. Prints the resulting plane dims, effective px/m, and
render time so carpet imagery can be eyeballed in the output PNG.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PHASE2 = _HERE.parent
sys.path.insert(0, str(_PHASE2))
sys.path.insert(0, str(_PHASE2 / "blender"))

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore
    Vector = None  # type: ignore

# Import after sys.path is set so Blender's Python can find them.
from generate_data import (  # noqa: E402
    IMG_W, IMG_H, PLANE_LONGER_M,
    _reset_scene, _configure_render, _add_camera_and_light, _add_photo_plane,
    _place_camera, _render_frame, compute_plane_dims, plane_sizing_report,
)


def _parse_cli_argv() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texture", type=str,
                   default="Data/textures_train/tex_00.webp",
                   help="Path (relative to Phase 2 root) to a prepared .webp "
                        "texture to render.")
    p.add_argument("--altitude", type=float, default=2.0,
                   help="Camera altitude in metres (midpoint of 1.5–3 m range).")
    p.add_argument("--output", type=str,
                   default="outputs/audit/debug_render.png",
                   help="Output PNG path (relative to Phase 2 root).")
    return p.parse_args(argv)


def main() -> int:
    if bpy is None:
        print("[debug_render] This script must be run inside Blender "
              "(bpy import failed).", file=sys.stderr)
        return 2

    args = _parse_cli_argv()
    tex_path = (_PHASE2 / args.texture).resolve()
    out_path = (_PHASE2 / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not tex_path.exists():
        print(f"[debug_render] texture not found: {tex_path}", file=sys.stderr)
        print("  Run scripts/prepare_textures.py first.", file=sys.stderr)
        return 2

    plane_x, plane_y = compute_plane_dims(tex_path)
    sizing = plane_sizing_report(tex_path)
    print(f"[debug_render] texture      : {tex_path}")
    print(f"[debug_render] plane dims   : {plane_x:.2f} × {plane_y:.2f} m  "
          f"(longer {PLANE_LONGER_M:.1f} m, aspect {sizing['aspect_ratio']:.3f})")
    print(f"[debug_render] density      : {sizing['effective_density_px_per_m']:.1f} px/m")
    print(f"[debug_render] resolution   : {IMG_W} × {IMG_H}")
    print(f"[debug_render] altitude     : {args.altitude:.2f} m")

    t0 = time.perf_counter()
    _reset_scene()
    _configure_render(out_path.parent)
    cam_obj = _add_camera_and_light()
    _add_photo_plane(plane_x, plane_y, tex_path)

    # Down-facing camera at (0, 0, altitude), zero yaw/roll/pitch.
    import numpy as np
    _place_camera(cam_obj, np.array([0.0, 0.0, float(args.altitude)]),
                  0.0, 0.0, 0.0)
    _render_frame(out_path)
    dt = time.perf_counter() - t0
    print(f"[debug_render] rendered     : {out_path} ({dt:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
