"""Tests for get_texture_for_trajectory assignment policy."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "blender"))

from generate_data import (  # noqa: E402
    TEXTURE_ASSIGNMENT, get_texture_for_trajectory,
)


def _mock_pools(tmp_path: Path) -> Path:
    """Create dummy tex_*.webp files matching the 4-train / 1-test split.

    train: tex_00..03 (one per training trajectory)
    test:  tex_04     (held-out texture shared by trajs 100, 101, 200)
    """
    root = tmp_path / "proj"
    (root / "Data" / "textures_train").mkdir(parents=True)
    (root / "Data" / "textures_test").mkdir(parents=True)
    for i in range(4):
        (root / "Data" / "textures_train" / f"tex_{i:02d}.webp").write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
    (root / "Data" / "textures_test" / "tex_04.webp").write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
    return root


@pytest.fixture
def mock_root(tmp_path: Path) -> Path:
    return _mock_pools(tmp_path)


def test_train_ids_use_train_pool(mock_root: Path) -> None:
    for tid, expected_name in [(0, "tex_00.webp"), (1, "tex_01.webp"),
                               (2, "tex_02.webp"), (3, "tex_03.webp")]:
        p = get_texture_for_trajectory(tid, project_root=mock_root)
        assert p.name == expected_name
        assert p.parent.name == "textures_train"


def test_test_ids_100_101_use_test_pool(mock_root: Path) -> None:
    """Both held-out-texture test trajs share the one test carpet (tex_04)."""
    p100 = get_texture_for_trajectory(100, project_root=mock_root)
    assert p100.name == "tex_04.webp"
    assert p100.parent.name == "textures_test"
    p101 = get_texture_for_trajectory(101, project_root=mock_root)
    assert p101.name == "tex_04.webp"
    assert p101.parent.name == "textures_test"


def test_test_id_102_reuses_train_tex_00(mock_root: Path) -> None:
    """Traj 102 intentionally uses a TRAINING texture."""
    p = get_texture_for_trajectory(102, project_root=mock_root)
    assert p.name == "tex_00.webp"
    assert p.parent.name == "textures_train"


def test_minitest_uses_test_tex_04(mock_root: Path) -> None:
    """Minitest (traj 200) uses the held-out test carpet."""
    p = get_texture_for_trajectory(200, project_root=mock_root)
    assert p.name == "tex_04.webp"
    assert p.parent.name == "textures_test"


def test_assignment_table_complete() -> None:
    """Every documented trajectory ID has an assignment."""
    for tid in (0, 1, 2, 3, 100, 101, 102, 200):
        assert tid in TEXTURE_ASSIGNMENT, f"no texture assignment for {tid}"


def test_unknown_id_raises(mock_root: Path) -> None:
    with pytest.raises(KeyError):
        get_texture_for_trajectory(9999, project_root=mock_root)


def test_missing_file_assertion(tmp_path: Path) -> None:
    """If the file isn't on disk, assertion fires with a helpful message."""
    proj = tmp_path / "empty"
    proj.mkdir()
    with pytest.raises(AssertionError):
        get_texture_for_trajectory(0, project_root=proj)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
