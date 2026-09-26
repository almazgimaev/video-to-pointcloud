"""Tests of the distortion options of the synthetic scene generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "assets" / "synthetic" / "make_scene.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_scene_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves the class module through sys.modules
    spec.loader.exec_module(module)
    return module


ms = _load_module()


def _generate(out: Path, **kwargs) -> dict:
    ms.make_scene(out, ms.SceneParams(**kwargs))
    return json.loads((out / "expected.json").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _centers(expected: dict) -> np.ndarray:
    return np.array([img["center"] for img in expected["images"]])


def _read_centers(sparse: Path) -> dict[str, np.ndarray]:
    rec = pycolmap.Reconstruction(str(sparse))
    return {
        img.name: np.asarray(img.cam_from_world().inverse().translation)
        for img in rec.images.values()
    }


def test_defaults_are_deterministic_and_unchanged(tmp_path: Path) -> None:
    _generate(tmp_path / "a")
    _generate(
        tmp_path / "b",
        arc_deg=360.0,
        height_jitter=0.0,
        roll_jitter_deg=0.0,
        tilt_deg=0.0,
        tilt_axis=(1.0, 0.0, 0.0),
        offset=(0.0, 0.0, 0.0),
        non_planar=0.0,
    )
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert _sha(tmp_path / "a" / "sparse" / name) == _sha(tmp_path / "b" / "sparse" / name)
    assert (tmp_path / "a" / "expected.json").read_bytes() == (
        tmp_path / "b" / "expected.json"
    ).read_bytes()

    expected = json.loads((tmp_path / "a" / "expected.json").read_text(encoding="utf-8"))
    assert expected["true_up_world"] == [0.0, 0.0, 1.0]
    assert expected["true_object_center"] == [0.0, 0.0, 0.0]


def test_distorted_scene_is_deterministic(tmp_path: Path) -> None:
    kwargs = dict(
        height_jitter=0.2, roll_jitter_deg=5.0, tilt_deg=30.0, offset=(1.0, 2.0, 3.0), seed=3
    )
    _generate(tmp_path / "a", **kwargs)
    _generate(tmp_path / "b", **kwargs)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert _sha(tmp_path / "a" / "sparse" / name) == _sha(tmp_path / "b" / "sparse" / name)


def test_tilt_and_offset_are_consistent_rigid_transform(tmp_path: Path) -> None:
    axis = (1.0, 1.0, 0.0)
    offset = (0.5, -1.5, 2.0)
    expected = _generate(tmp_path, tilt_deg=40.0, tilt_axis=axis, offset=offset)

    rot = ms.axis_angle_rotation(np.array(axis), np.deg2rad(40.0))
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    up = np.array(expected["true_up_world"])
    assert np.allclose(up, rot @ np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert np.isclose(np.linalg.norm(up), 1.0, atol=1e-12)
    assert not np.allclose(up, [0.0, 0.0, 1.0], atol=1e-3)
    assert np.allclose(expected["true_object_center"], offset, atol=1e-12)
    assert expected["distortion"]["tilt_deg"] == 40.0
    assert expected["distortion"]["tilt_axis"] == list(axis)
    assert expected["distortion"]["offset"] == list(offset)

    # Centres in the file agree with expected.json.
    read = _read_centers(tmp_path / "sparse")
    for img in expected["images"]:
        assert np.allclose(read[img["name"]], img["center"], atol=1e-9)

    # Each centre is the rigid image of the canonical one.
    canon = _generate(tmp_path / "canon")
    for img, base in zip(expected["images"], canon["images"], strict=True):
        moved = rot @ np.array(base["center"]) + np.array(offset)
        assert np.allclose(img["center"], moved, atol=1e-9)


def test_arc_span(tmp_path: Path) -> None:
    expected = _generate(tmp_path, arc_deg=120.0, num_cameras=9)
    centers = _centers(expected)
    # Canonical frame: ring centre is on the Z axis, the ring plane is XY.
    angles = np.rad2deg(np.arctan2(centers[:, 1], centers[:, 0]))
    assert np.isclose(angles.max() - angles.min(), 120.0, atol=1e-6)

    full = _centers(_generate(tmp_path / "full", num_cameras=9))
    full_angles = np.sort(np.rad2deg(np.arctan2(full[:, 1], full[:, 0])))
    assert full_angles.max() - full_angles.min() > 300.0


def test_arc_span_measured_after_tilt(tmp_path: Path) -> None:
    axis = (0.3, -1.0, 0.2)
    expected = _generate(tmp_path, arc_deg=120.0, num_cameras=9, tilt_deg=50.0, tilt_axis=axis)
    rot = ms.axis_angle_rotation(np.array(axis), np.deg2rad(50.0))
    canonical = (_centers(expected) - np.array(expected["true_object_center"])) @ rot
    angles = np.rad2deg(np.arctan2(canonical[:, 1], canonical[:, 0]))
    assert np.isclose(angles.max() - angles.min(), 120.0, atol=1e-6)


def test_roll_changes_orientation_not_centres(tmp_path: Path) -> None:
    base = _generate(tmp_path / "base", seed=5)
    rolled = _generate(tmp_path / "rolled", seed=5, roll_jitter_deg=8.0)

    assert np.allclose(_centers(base), _centers(rolled), atol=1e-12)
    base_q = np.array([img["qvec"] for img in base["images"]])
    rolled_q = np.array([img["qvec"] for img in rolled["images"]])
    assert not np.allclose(base_q, rolled_q, atol=1e-4)

    # The optical axis (camera z in the world) is untouched by a roll.
    for path in (tmp_path / "base", tmp_path / "rolled"):
        rec = pycolmap.Reconstruction(str(path / "sparse"))
        for img in rec.images.values():
            centre = np.asarray(img.cam_from_world().inverse().translation)
            axis = np.asarray(img.cam_from_world().rotation.matrix())[2]
            assert np.allclose(axis, -centre / np.linalg.norm(centre), atol=1e-9)


def test_height_jitter_and_non_planar_move_only_heights(tmp_path: Path) -> None:
    base = _centers(_generate(tmp_path / "base", seed=1))
    jitter = _centers(_generate(tmp_path / "jitter", seed=1, height_jitter=0.3))
    wave = _centers(_generate(tmp_path / "wave", seed=1, non_planar=0.4))

    for other in (jitter, wave):
        assert np.allclose(base[:, :2], other[:, :2], atol=1e-12)
        assert np.ptp(other[:, 2]) > 0.05
    assert np.allclose(base[:, 2], base[0, 2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "arc_deg": 150.0,
            "height_jitter": 0.1,
            "roll_jitter_deg": 6.0,
            "tilt_deg": 35.0,
            "tilt_axis": (1.0, 2.0, -0.5),
            "offset": (2.0, -1.0, 0.7),
            "non_planar": 0.2,
            "num_cameras": 15,
            "seed": 4,
        },
    ],
)
def test_reads_with_pycolmap_and_reprojects(tmp_path: Path, kwargs: dict) -> None:
    expected = _generate(tmp_path, **kwargs)
    rec = pycolmap.Reconstruction(str(tmp_path / "sparse"))
    assert rec.num_images() == expected["params"]["num_cameras"]
    fx, fy, cx, cy = rec.cameras[1].params

    checked = 0
    for img in rec.images.values():
        pose = img.cam_from_world().matrix()
        for p2d in img.points2D:
            if not p2d.has_point3D():
                continue
            cam = pose[:, :3] @ rec.points3D[p2d.point3D_id].xyz + pose[:, 3]
            assert cam[2] > 0.0
            uv = np.array([fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy])
            assert np.allclose(uv, p2d.xy, atol=1e-6)
            checked += 1
    assert checked > 0
