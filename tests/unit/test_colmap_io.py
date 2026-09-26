"""Unit tests of reading a COLMAP sparse model (T027).

The fixture is a synthetic scene from `assets/synthetic/make_scene.py`: it checks
**only file formats and geometry conventions**, not the quality of a real reconstruction
(it has no images, no feature detector, no optimization).
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from v3d.colmap_io import SparseModel, frame_id_from_image_name, read_sparse
from v3d.errors import ExitCode, ResultMissingOrIncompatibleError

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKE_SCENE = REPO_ROOT / "assets" / "synthetic" / "make_scene.py"

NUM_CAMERAS = 6
NUM_POINTS = 200
SEED = 7

# Point record format in points3D.bin up to the track: id, xyz, rgb, error.
_POINT_HEAD = "<Q3d3Bd"


@pytest.fixture(scope="session")
def scene(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Synthetic scene: the `sparse` directory and the parsed `expected.json`."""
    out_dir = tmp_path_factory.mktemp("colmap_scene")
    result = subprocess.run(
        [
            sys.executable,
            str(MAKE_SCENE),
            "--out",
            str(out_dir),
            "--cameras",
            str(NUM_CAMERAS),
            "--points",
            str(NUM_POINTS),
            "--seed",
            str(SEED),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    expected = json.loads((out_dir / "expected.json").read_text(encoding="utf-8"))
    return {"dir": out_dir, "sparse": out_dir / "sparse", "expected": expected}


@pytest.fixture(scope="session")
def model(scene: dict) -> SparseModel:
    """The model that was read — shared per session: reading has no side effects."""
    return read_sparse(scene["sparse"])


def test_counts_match_request(model: SparseModel) -> None:
    """The number of cameras and points matches what was requested at generation."""
    assert len(model.cameras) == NUM_CAMERAS
    assert len(model.image_names) == NUM_CAMERAS
    assert model.points_xyz.shape == (NUM_POINTS, 3)
    assert model.points_rgb.shape == (NUM_POINTS, 3)


def test_frame_id_from_image_name() -> None:
    """`frame_id` is the file name without extension, the subdirectory is dropped."""
    assert frame_id_from_image_name("0000_000012.jpg") == "0000_000012"
    assert frame_id_from_image_name("images/0003_000117.jpg") == "0003_000117"
    assert frame_id_from_image_name("0007_000254") == "0007_000254"


def test_frame_ids_follow_naming_format(model: SparseModel, scene: dict) -> None:
    """frame_id is derived from the model image names and has the format NNNN_SSSSSS."""
    expected_ids = sorted(Path(img["name"]).stem for img in scene["expected"]["images"])
    assert [cam.frame_id for cam in model.cameras] == expected_ids
    for frame_id in expected_ids:
        head, _, tail = frame_id.partition("_")
        assert len(head) == 4 and head.isdigit()
        assert len(tail) == 6 and tail.isdigit()


def test_camera_centers_match_expected(model: SparseModel, scene: dict) -> None:
    """Main convention check: our `C = -Rᵀt` matches the scene reference."""
    expected_centers = {
        Path(img["name"]).stem: np.asarray(img["center"], dtype=np.float64)
        for img in scene["expected"]["images"]
    }
    assert len(expected_centers) == len(model.cameras)
    for cam in model.cameras:
        got = np.asarray(cam.camera_center_world, dtype=np.float64)
        np.testing.assert_allclose(got, expected_centers[cam.frame_id], atol=1e-9, rtol=0.0)


def test_poses_are_world_to_camera(model: SparseModel, scene: dict) -> None:
    """R and t are stored as world_to_camera: ``R @ C + t == 0`` and match the reference."""
    expected = {Path(img["name"]).stem: img for img in scene["expected"]["images"]}
    for cam in model.cameras:
        R = np.asarray(cam.R_world_to_camera, dtype=np.float64)
        t = np.asarray(cam.t_world_to_camera, dtype=np.float64)
        center = np.asarray(cam.camera_center_world, dtype=np.float64)
        np.testing.assert_allclose(R @ center + t, np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(
            t, np.asarray(expected[cam.frame_id]["tvec"], dtype=np.float64), atol=1e-9
        )


def test_intrinsics_match_expected(model: SparseModel, scene: dict) -> None:
    """intrinsics and image_size are taken from the camera model, not guessed."""
    camera = scene["expected"]["camera"]
    fx, fy, cx, cy = camera["params"]
    for cam in model.cameras:
        assert cam.image_size == (camera["width"], camera["height"])
        assert cam.intrinsics == {
            "model": camera["model"],
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
        assert cam.source == "colmap_sparse"


def test_simple_pinhole_duplicates_focal_length(tmp_path: Path) -> None:
    """SIMPLE_PINHOLE has one focal parameter — it expands to ``fx == fy == f``."""
    pycolmap = pytest.importorskip("pycolmap")
    from v3d.colmap_io import intrinsics_from_camera

    camera = pycolmap.Camera.create_from_model_name(1, "SIMPLE_PINHOLE", 480.0, 640, 480)
    intrinsics = intrinsics_from_camera(camera)
    assert intrinsics == {
        "model": "SIMPLE_PINHOLE",
        "fx": 480.0,
        "fy": 480.0,
        "cx": 320.0,
        "cy": 240.0,
    }


def test_points_values_and_types(model: SparseModel, scene: dict) -> None:
    """Coordinates are finite and within the scene extents, colours are uint8 of shape (N, 3)."""
    assert model.points_xyz.dtype == np.float64
    assert np.isfinite(model.points_xyz).all()
    assert model.points_rgb.dtype == np.uint8
    assert model.points_rgb.shape == (NUM_POINTS, 3)

    bbox_min = np.asarray(scene["expected"]["points3D"]["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(scene["expected"]["points3D"]["bbox_max"], dtype=np.float64)
    np.testing.assert_allclose(model.points_xyz.min(axis=0), bbox_min, atol=1e-9)
    np.testing.assert_allclose(model.points_xyz.max(axis=0), bbox_max, atol=1e-9)


def test_point_order_is_deterministic(scene: dict) -> None:
    """Two consecutive reads give bitwise identical point arrays and the same camera order."""
    first = read_sparse(scene["sparse"])
    second = read_sparse(scene["sparse"])
    np.testing.assert_array_equal(first.points_xyz, second.points_xyz)
    np.testing.assert_array_equal(first.points_rgb, second.points_rgb)
    assert [c.frame_id for c in first.cameras] == [c.frame_id for c in second.cameras]
    assert first.image_names == second.image_names


def test_mean_reprojection_error_is_measured_zero(model: SparseModel) -> None:
    """Recorded actual behaviour of pycolmap 4.2.0 on this scene.

    The generator writes exactly 0.0 as the error into points3D.bin, because the 2D
    observations are built by exact projection. COLMAP considers such a point to have an error
    (`has_error() == True`), so the mean error here is a **measured zero**, not a substitution
    for an unavailable value: that is exactly what is checked. The unavailable case is
    checked separately (`test_mean_reprojection_error_none_when_unavailable`).
    """
    assert model.mean_reprojection_error == 0.0
    assert model.points_error is not None
    assert model.points_error.shape == (NUM_POINTS,)
    np.testing.assert_array_equal(model.points_error, np.zeros(NUM_POINTS))


def _rewrite_point_errors(src: Path, dst: Path, value: float) -> None:
    """Copy the sparse model, replacing the error of every point in points3D.bin with `value`."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin"):
        shutil.copy(src / name, dst / name)

    raw = (src / "points3D.bin").read_bytes()
    head_size = struct.calcsize(_POINT_HEAD)
    offset = 0
    (num_points,) = struct.unpack_from("<Q", raw, offset)
    offset += 8
    out = bytearray(struct.pack("<Q", num_points))
    for _ in range(num_points):
        point_id, x, y, z, r, g, b, _error = struct.unpack_from(_POINT_HEAD, raw, offset)
        offset += head_size
        (track_len,) = struct.unpack_from("<Q", raw, offset)
        offset += 8
        track = raw[offset : offset + track_len * 8]
        offset += track_len * 8
        out += struct.pack(_POINT_HEAD, point_id, x, y, z, r, g, b, value)
        out += struct.pack("<Q", track_len) + track
    (dst / "points3D.bin").write_bytes(bytes(out))


def test_mean_reprojection_error_none_when_unavailable(scene: dict, tmp_path: Path) -> None:
    """Error not set (-1 in the file) → `None`, not zero (FR-036).

    Important: `pycolmap.Reconstruction.compute_mean_reprojection_error()` itself in 4.2.0
    returns 0.0 in this case, so availability is determined by the points' `has_error()`,
    not by the returned number.
    """
    sparse_dir = tmp_path / "no_error" / "sparse"
    _rewrite_point_errors(scene["sparse"], sparse_dir, -1.0)

    model = read_sparse(sparse_dir)
    assert model.mean_reprojection_error is None
    assert model.points_error is None
    assert model.points_xyz.shape == (NUM_POINTS, 3)


def test_missing_directory_is_code_7(tmp_path: Path) -> None:
    """No model directory → ResultMissingOrIncompatibleError (code 7)."""
    with pytest.raises(ResultMissingOrIncompatibleError) as excinfo:
        read_sparse(tmp_path / "nope")
    assert excinfo.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE


def test_missing_points3d_is_code_7(scene: dict, tmp_path: Path) -> None:
    """A directory without points3D.bin → code 7, the missing file name is in the message."""
    partial = tmp_path / "partial"
    partial.mkdir()
    for name in ("cameras.bin", "images.bin"):
        shutil.copy(scene["sparse"] / name, partial / name)

    with pytest.raises(ResultMissingOrIncompatibleError) as excinfo:
        read_sparse(partial)
    assert excinfo.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE
    assert "points3D.bin" in excinfo.value.message


def test_corrupted_model_is_code_7(scene: dict, tmp_path: Path) -> None:
    """Files are in place but points3D.bin is truncated → code 7, pycolmap text kept in details."""
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy(scene["sparse"] / name, broken / name)
    data = (broken / "points3D.bin").read_bytes()
    (broken / "points3D.bin").write_bytes(data[: len(data) // 2])

    with pytest.raises(ResultMissingOrIncompatibleError) as excinfo:
        read_sparse(broken)
    assert excinfo.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE
    assert excinfo.value.details
