"""Web-cloud decimation checks: byte-identical determinism, no averaging (FR-051, FR-052)."""

from __future__ import annotations

import numpy as np
import pytest

from v3d.decimate import voxel_decimate, write_web_cloud
from v3d.ply_io import read_ply, write_ply


@pytest.fixture
def dense_cloud() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(200_000, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(200_000, 3), dtype=np.uint8)
    return xyz, rgb


def test_output_never_exceeds_target_and_is_close(dense_cloud):
    """The decimated cloud must fit the target and use most of the budget (FR-052)."""
    xyz, rgb = dense_cloud
    out_xyz, out_rgb, info = voxel_decimate(xyz, rgb, target=40_000)

    assert len(out_xyz) <= 40_000
    assert len(out_xyz) >= 0.9 * 40_000
    assert len(out_rgb) == len(out_xyz)
    assert info["points_out"] == len(out_xyz)
    assert info["points_in"] == 200_000
    assert info["target"] == 40_000
    assert info["method"] == "voxel_grid_nearest_to_centre"
    assert info["voxel_size"] is not None and info["voxel_size"] > 0.0


def test_every_output_point_is_an_exact_input_row(dense_cloud):
    """Decimation must keep real points: no averaged position or colour."""
    xyz, rgb = dense_cloud
    out_xyz, out_rgb, _ = voxel_decimate(xyz, rgb, target=10_000)

    row_to_index = {tuple(row): i for i, row in enumerate(xyz)}
    for point, color in zip(out_xyz, out_rgb, strict=True):
        index = row_to_index.get(tuple(point))
        assert index is not None, "output point is not an exact row of the input"
        assert tuple(rgb[index]) == tuple(color), "colour was altered for a kept point"


def test_deterministic_across_runs(dense_cloud):
    """Two runs on the same input must produce identical arrays (byte-for-byte)."""
    xyz, rgb = dense_cloud
    out_xyz_1, out_rgb_1, info_1 = voxel_decimate(xyz, rgb, target=15_000)
    out_xyz_2, out_rgb_2, info_2 = voxel_decimate(xyz, rgb, target=15_000)

    assert np.array_equal(out_xyz_1, out_xyz_2)
    assert np.array_equal(out_rgb_1, out_rgb_2)
    assert info_1 == info_2


def test_output_sorted_by_original_index(dense_cloud):
    """Output order must be deterministic: sorted by original index."""
    xyz, rgb = dense_cloud
    out_xyz, _, _ = voxel_decimate(xyz, rgb, target=5_000)

    # Recover original indices by matching rows back (float32, exact match since unmodified).
    row_to_index = {tuple(row): i for i, row in enumerate(xyz)}
    indices = [row_to_index[tuple(point)] for point in out_xyz]
    assert indices == sorted(indices)


def test_small_cloud_returned_unchanged():
    """A cloud already at or below the target must be returned unchanged, in original order."""
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(500, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(500, 3), dtype=np.uint8)

    out_xyz, out_rgb, info = voxel_decimate(xyz, rgb, target=1_000)

    assert np.array_equal(out_xyz, xyz)
    assert np.array_equal(out_rgb, rgb)
    assert info["points_in"] == 500
    assert info["points_out"] == 500
    assert info["voxel_size"] is None
    assert info["label"] == "500 of 500 points"


def test_non_finite_rows_are_dropped():
    """NaN/Inf points are not valid points and must not appear in the output."""
    xyz = np.array(
        [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 1.0, 1.0], [np.inf, 0.0, 0.0]],
        dtype=np.float32,
    )
    rgb = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint8)

    out_xyz, out_rgb, info = voxel_decimate(xyz, rgb, target=10)

    assert len(out_xyz) == 2
    assert info["points_in"] == 2
    assert np.isfinite(out_xyz).all()
    assert np.array_equal(out_xyz, xyz[[0, 2]])
    assert np.array_equal(out_rgb, rgb[[0, 2]])


def test_label_uses_thousands_separators():
    """FR-052 label must read e.g. '40,000 of 100,000 points'."""
    rng = np.random.default_rng(2)
    xyz = rng.normal(size=(100_000, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(100_000, 3), dtype=np.uint8)

    _, _, info = voxel_decimate(xyz, rgb, target=40_000)

    prefix, _, suffix = info["label"].partition(" of ")
    assert suffix == f"{info['points_in']:,} points"
    assert prefix == f"{info['points_out']:,}"
    assert "," in info["label"]


def test_spatial_coverage_keeps_both_clusters():
    """Decimating two separated clusters must keep points from both, not just one."""
    rng = np.random.default_rng(3)
    cluster_a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.01, size=(5_000, 3))
    cluster_b = rng.normal(loc=[100.0, 100.0, 100.0], scale=0.01, size=(5_000, 3))
    xyz = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    rgb = np.zeros((10_000, 3), dtype=np.uint8)

    out_xyz, _, _ = voxel_decimate(xyz, rgb, target=200)

    near_a = (np.linalg.norm(out_xyz - np.array([0.0, 0.0, 0.0]), axis=1) < 50.0).sum()
    near_b = (np.linalg.norm(out_xyz - np.array([100.0, 100.0, 100.0]), axis=1) < 50.0).sum()
    assert near_a > 0
    assert near_b > 0


@pytest.fixture
def normalized_ply(tmp_path):
    rng = np.random.default_rng(4)
    xyz = rng.normal(size=(50_000, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(50_000, 3), dtype=np.uint8)
    path = tmp_path / "normalized" / "points.ply"
    write_ply(
        path,
        xyz,
        rgb,
        run_id="run-web-1",
        comments=["filters_applied: []"],
        axes="opencv right-handed; units unknown",
    )
    return path


def test_write_web_cloud_deterministic_bytes(tmp_path, normalized_ply):
    """Writing the web cloud twice must produce byte-identical files."""
    out_1 = tmp_path / "web_1.ply"
    out_2 = tmp_path / "web_2.ply"
    write_web_cloud(normalized_ply, out_1, target=10_000)
    write_web_cloud(normalized_ply, out_2, target=10_000)

    assert out_1.read_bytes() == out_2.read_bytes()


def test_write_web_cloud_keeps_run_id_and_comment(tmp_path, normalized_ply):
    """The web cloud must keep run_id, note its decimation, and be smaller than the source."""
    out_path = tmp_path / "web.ply"
    info = write_web_cloud(normalized_ply, out_path, target=10_000)

    _, _, comments = read_ply(out_path)
    assert comments["run_id"] == "run-web-1"
    assert comments["web_decimation"] == info["label"]
    assert comments["derived_from"] == "normalized/points.ply"
    assert comments["axes"] == "opencv right-handed; units unknown"

    assert out_path.stat().st_size < normalized_ply.stat().st_size
    assert info["bytes"] == out_path.stat().st_size
    assert info["path"] == str(out_path)
