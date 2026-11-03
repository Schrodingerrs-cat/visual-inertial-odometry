"""Scene preview: 4 camera poses over textures_train/tex_00.webp + plane schematic.

Renders a 2×2 grid of rendered frames from four camera poses to validate that
the plane + texture + lighting look correct before running the full generator.
Also writes a top-down schematic showing the plane rectangle, an example
figure_8 trajectory, and footprint circles at each pose.

Invocation:
    blender --background --python scripts/preview_scene.py -- \\
            [--output outputs/sanity/scene_preview.png]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

try:
    import bpy  # type: ignore
    from mathutils import Euler, Vector  # type: ignore
except ImportError:  # pragma: no cover
    bpy = None
    Euler = Vector = None

from blender.generate_data import (  # noqa: E402
    PLANE_LONGER_M, _add_camera_and_light, _add_photo_plane,
    _configure_render, _place_camera, _reset_scene, compute_plane_dims,
)


def _parse_cli_argv() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=str,
                   default="outputs/sanity/scene_preview.png")
    p.add_argument("--train_dir", type=str, default="Data/textures_train")
    p.add_argument("--texture_name", type=str, default="tex_00.webp")
    p.add_argument("--schematic", type=str,
                   default="outputs/sanity/plane_geometry.png")
    return p.parse_args(argv)


# (pos, rpy) tuples.
_POSES = [
    # Centered z=3
    (np.array([0.0, 0.0, 3.0]), (0.0, 0.0, 0.0)),
    # Centered z=6
    (np.array([0.0, 0.0, 6.0]), (0.0, 0.0, 0.0)),
    # Off-center with roll+pitch+yaw
    (np.array([5.0, -3.0, 4.0]),
     (math.radians(8.0), math.radians(-5.0), math.radians(20.0))),
    # Opposite corner off-center
    (np.array([-6.0, 4.0, 4.5]), (0.0, 0.0, math.radians(-30.0))),
]


def _setup_scene(image_path: Path):
    _reset_scene()
    _configure_render(image_path.parent)
    _add_camera_and_light()
    plane_x, plane_y = compute_plane_dims(image_path, PLANE_LONGER_M)
    _add_photo_plane(plane_x, plane_y, image_path)
    return plane_x, plane_y


def _render_pose(pos: np.ndarray, rpy: tuple, out_png: Path) -> None:
    cam_obj = bpy.context.scene.camera
    _place_camera(cam_obj, pos, rpy[0], rpy[1], rpy[2])
    bpy.context.scene.render.filepath = str(out_png)
    bpy.ops.render.render(write_still=True)


def _assemble_2x2(paths: List[Path], out: Path) -> None:
    from PIL import Image as _PILImage
    imgs = [_PILImage.open(p).convert("RGB") for p in paths]
    w, h = imgs[0].size
    grid = _PILImage.new("RGB", (2 * w, 2 * h), (40, 40, 40))
    for i, im in enumerate(imgs):
        r, c = divmod(i, 2)
        grid.paste(im, (c * w, r * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


def _draw_schematic(plane_x: float, plane_y: float, out: Path) -> None:
    """Top-down view of the plane + example figure_8 traj + footprint circles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    # Example figure_8 trajectory in XY (parametric Lissajous).
    t = np.linspace(0.0, 2 * np.pi, 200)
    A = plane_x / 3.0
    B = plane_y / 4.0
    traj_x = A * np.sin(t)
    traj_y = B * np.sin(2 * t)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(mp.Rectangle((-plane_x / 2, -plane_y / 2),
                              plane_x, plane_y, fill=False,
                              edgecolor="black", linewidth=1.5,
                              label=f"plane {plane_x:.1f}×{plane_y:.1f} m"))
    ax.plot(traj_x, traj_y, color="blue", lw=1.2,
            label="example figure_8 trajectory")
    # Footprint circles at the four pose XY locations.
    half_tan = float(np.tan(np.deg2rad(53.0) / 2.0))
    for pos, _rpy in _POSES:
        r = pos[2] * half_tan
        ax.add_patch(mp.Circle((pos[0], pos[1]), radius=r, fill=False,
                               edgecolor="orange", alpha=0.7))
        ax.scatter([pos[0]], [pos[1]], color="red", s=30, zorder=5)
    ax.set_xlim(-plane_x * 0.6, plane_x * 0.6)
    ax.set_ylim(-plane_y * 0.6, plane_y * 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"plane schematic — {plane_x:.1f} × {plane_y:.1f} m")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)


def main() -> None:
    if bpy is None:
        raise RuntimeError("This script must run inside Blender.")
    args = _parse_cli_argv()
    project_root = HERE
    image_path = project_root / args.train_dir / args.texture_name
    if not image_path.exists():
        raise FileNotFoundError(
            f"scene preview needs {image_path}. "
            f"Run `python scripts/prepare_textures.py` first.")

    out_grid = Path(args.output)
    out_schematic = Path(args.schematic)

    plane_x, plane_y = _setup_scene(image_path)

    tmp_dir = out_grid.parent / "_scene_tiles"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tiles: List[Path] = []
    for i, (pos, rpy) in enumerate(_POSES):
        p = tmp_dir / f"pose_{i}.png"
        _render_pose(pos, rpy, p)
        tiles.append(p)

    _assemble_2x2(tiles, out_grid)
    _draw_schematic(plane_x, plane_y, out_schematic)
    print(f"[preview_scene] wrote {out_grid} and {out_schematic}")


if __name__ == "__main__":
    main()
