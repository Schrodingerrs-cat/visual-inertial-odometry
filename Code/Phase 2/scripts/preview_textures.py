"""Preview all 7 user textures in a 7-panel grid (5 train + 2 test).

Runs inside Blender (headless). For each texture:
  1. Create a plane sized to the texture's aspect ratio (longer edge = 75 m).
  2. Place a level, centered camera at altitude 4.5 m.
  3. Render one 320x240 frame.
  4. Label it with TRAIN/TEST + normalized filename + original filename.

Invocation:
    blender --background --python scripts/preview_textures.py -- \\
            [--output outputs/sanity/texture_preview_all.png]

Run ONCE after `prepare_textures.py` has produced textures_train/ and
textures_test/. If any panel shows a uniform colour / no texture, fix the
raw image before spending an hour on full data generation.
"""
from __future__ import annotations

import argparse
import json
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
    _configure_render, _reset_scene, compute_plane_dims,
)


PREVIEW_ALTITUDE = 4.5


def _parse_cli_argv() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(
        description="Render the 7 processed textures into a labelled grid.")
    p.add_argument("--output", type=str,
                   default="outputs/sanity/texture_preview_all.png")
    p.add_argument("--train_dir", type=str, default="Data/textures_train")
    p.add_argument("--test_dir", type=str, default="Data/textures_test")
    p.add_argument("--manifest", type=str,
                   default="Data/textures_processed/manifest.json")
    return p.parse_args(argv)


def _place_topdown_camera(altitude: float) -> None:
    cam = bpy.context.scene.camera
    cam.location = Vector((0.0, 0.0, altitude))
    cam.rotation_mode = "XYZ"
    # Rotation (0,0,0) has Blender camera looking along local −Z = world −Z
    # at identity, i.e. straight down onto the floor. The old
    # Euler((math.pi, 0, 0)) flipped it to look UP (regression; same bug
    # generate_data.py documents in _blender_camera_euler).
    cam.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")


def _render_one(image_path: Path, out_png: Path) -> None:
    _reset_scene()
    _configure_render(out_png.parent)
    _add_camera_and_light()
    _place_topdown_camera(PREVIEW_ALTITUDE)
    plane_x, plane_y = compute_plane_dims(image_path, PLANE_LONGER_M)
    _add_photo_plane(plane_x, plane_y, image_path)
    bpy.context.scene.render.filepath = str(out_png)
    bpy.ops.render.render(write_still=True)


def _label_panel(img_path: Path, label: str) -> None:
    """Overlay a caption on a rendered PNG via Pillow."""
    from PIL import Image as _PILImage, ImageDraw
    im = _PILImage.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    # Shaded bottom strip for readability.
    strip_h = max(28, h // 8)
    strip = _PILImage.new("RGB", (w, strip_h), (0, 0, 0))
    im.paste(strip, (0, h - strip_h))
    draw = ImageDraw.Draw(im)
    draw.text((6, h - strip_h + 4), label, fill=(255, 255, 255))
    im.save(img_path)


def _assemble_grid(panels: List[Tuple[Path, str]], out_path: Path) -> None:
    """5 TRAIN on row 1, 2 TEST on row 2."""
    from PIL import Image as _PILImage, ImageDraw
    train_imgs = [_PILImage.open(p).convert("RGB")
                  for (p, lbl) in panels if lbl.startswith("TRAIN")]
    test_imgs = [_PILImage.open(p).convert("RGB")
                 for (p, lbl) in panels if lbl.startswith("TEST")]
    if not train_imgs:
        raise RuntimeError("no TRAIN panels")
    w, h = train_imgs[0].size
    cols = max(len(train_imgs), len(test_imgs))
    grid = _PILImage.new("RGB", (cols * w, 2 * h), (40, 40, 40))
    for i, im in enumerate(train_imgs):
        grid.paste(im, (i * w, 0))
    for j, im in enumerate(test_imgs):
        grid.paste(im, (j * w, h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    if bpy is None:
        raise RuntimeError("This script must run inside Blender.")
    args = _parse_cli_argv()

    project_root = HERE
    train_dir = project_root / args.train_dir
    test_dir = project_root / args.test_dir
    manifest_path = project_root / args.manifest
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"textures": []}
    orig_by_name = {e["normalized_name"]: e["original_filename"]
                    for e in manifest.get("textures", [])}

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    panels: List[Tuple[Path, str]] = []

    for tex in sorted(train_dir.glob("tex_*.webp")):
        png = out_dir / f"preview_{tex.stem}.png"
        _render_one(tex, png)
        label = (f"TRAIN — {tex.name}  (orig: "
                 f"{orig_by_name.get(tex.name, '?')})")
        _label_panel(png, label)
        panels.append((png, label))
    for tex in sorted(test_dir.glob("tex_*.webp")):
        png = out_dir / f"preview_{tex.stem}.png"
        _render_one(tex, png)
        label = (f"TEST — {tex.name}  (orig: "
                 f"{orig_by_name.get(tex.name, '?')})")
        _label_panel(png, label)
        panels.append((png, label))

    _assemble_grid(panels, Path(args.output))
    print(f"[preview_textures] wrote {args.output}  ({len(panels)} panels)")


if __name__ == "__main__":
    main()
