"""PLY format checks: the export must match the displayed cloud (FR-031)."""

from __future__ import annotations

import numpy as np
import pytest

from v3d.errors import ResultMissingOrIncompatibleError
from v3d.ply_io import read_ply, write_ply


@pytest.fixture
def cloud() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(50, 3))
    rgb = rng.integers(0, 256, size=(50, 3), dtype=np.uint8)
    return xyz, rgb


def test_round_trip_preserves_points_and_colors(tmp_path, cloud):
    """The point count, coordinates and colours must survive writing and reading."""
    xyz, rgb = cloud
    written = write_ply(tmp_path / "c.ply", xyz, rgb, run_id="r1")
    back_xyz, back_rgb, _ = read_ply(tmp_path / "c.ply")

    assert written == len(xyz)
    assert len(back_xyz) == len(xyz)
    assert np.allclose(back_xyz, xyz, atol=1e-6), "coordinates distorted by writing"
    assert (back_rgb == rgb).all(), "colours distorted by writing"


def test_header_declares_type_and_scale(tmp_path, cloud):
    """The file must describe itself: a point cloud, scale not determined (FR-021)."""
    xyz, rgb = cloud
    write_ply(tmp_path / "c.ply", xyz, rgb, run_id="run-42")
    _, _, comments = read_ply(tmp_path / "c.ply")

    assert comments["run_id"] == "run-42", "the file must be identifiable by its run (FR-032)"
    assert comments["artifact_type"] == "point_cloud"
    assert comments["scale_status"] == "not_determined"


def test_mismatch_between_colors_and_points_is_forbidden(tmp_path, cloud):
    """A point without a colour would mean the export diverges from the displayed cloud."""
    xyz, rgb = cloud
    with pytest.raises(ValueError):
        write_ply(tmp_path / "c.ply", xyz, rgb[:10], run_id="r1")


def test_empty_cloud_is_written_without_error(tmp_path):
    """An empty result is a valid data case; the status decides whether it is usable."""
    written = write_ply(
        tmp_path / "c.ply", np.zeros((0, 3)), np.zeros((0, 3), np.uint8), run_id="r1"
    )
    back_xyz, _, _ = read_ply(tmp_path / "c.ply")
    assert written == 0
    assert len(back_xyz) == 0


def test_truncated_file_is_refused(tmp_path, cloud):
    """An incompletely written cloud must not be read as valid (code 7)."""
    xyz, rgb = cloud
    path = tmp_path / "c.ply"
    write_ply(path, xyz, rgb, run_id="r1")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 20])

    with pytest.raises(ResultMissingOrIncompatibleError):
        read_ply(path)


def test_foreign_format_is_refused(tmp_path):
    """A file not in PLY format is rejected with a reason."""
    path = tmp_path / "c.ply"
    path.write_bytes(b"not a ply file\n")
    with pytest.raises(ResultMissingOrIncompatibleError):
        read_ply(path)
