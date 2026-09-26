"""The saved precomputed example run (T036).

The tests check the **already generated** directory `assets/precomputed/sample_run/` and do
not run generation: it is slow and requires ffmpeg. If the directory is absent (a fresh
clone — raw upstream output is not stored in git, and the whole example may not be built),
the tests are skipped with an indication of how to enable them.

The main idea being checked: the example is marked as precomputed (A-07, FR-028), and it
cannot be used to judge the quality of a real reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from v3d.artifacts import require_same_run_id
from v3d.ply_io import read_ply

REPO = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO / "assets" / "precomputed" / "sample_run"

REQUIRED_FILES = (
    "manifest.json",
    "frames.json",
    "normalized/cameras.json",
    "normalized/points.ply",
    "normalized/diagnostics.json",
    "normalized/status.json",
)

pytestmark = pytest.mark.skipif(
    not SAMPLE_DIR.is_dir(),
    reason=(
        "the example is not generated, run assets/precomputed/make_sample.py "
        f"(expected directory {SAMPLE_DIR})"
    ),
)


def _read_json(relative: str) -> dict:
    return json.loads((SAMPLE_DIR / relative).read_text(encoding="utf-8"))


def test_example_directory_contains_required_artifacts() -> None:
    """The viewer, export and report read only these files — all of them must be present."""
    missing = [name for name in REQUIRED_FILES if not (SAMPLE_DIR / name).is_file()]
    assert not missing, f"the example lacks files: {missing}"


def test_example_is_marked_as_precomputed() -> None:
    """A-07 and FR-028: the mark in the metadata is what the example exists for.

    Without `precomputed_example`, a saved run can be mistaken for the result of a real
    reconstruction. The viewer must show this mark permanently, and it can take it only
    from here.
    """
    status = _read_json("normalized/status.json")
    assert status["precomputed_example"] is True, (
        "an example without the mark is indistinguishable from the result of a real reconstruction"
    )


def test_example_artifacts_belong_to_one_run() -> None:
    """Mixing files from different runs is forbidden (FR-029)."""
    manifest = _read_json("manifest.json")
    require_same_run_id(
        manifest["run_id"],
        ("frames.json", _read_json("frames.json")),
        ("normalized/cameras.json", _read_json("normalized/cameras.json")),
        ("normalized/diagnostics.json", _read_json("normalized/diagnostics.json")),
        ("normalized/status.json", _read_json("normalized/status.json")),
    )

    _, _, comments = read_ply(SAMPLE_DIR / "normalized" / "points.ply")
    assert comments["run_id"] == manifest["run_id"], "the point cloud is from another run"


def test_point_cloud_matches_diagnostics() -> None:
    """The point count in the PLY and in diagnostics must be the same (FR-031)."""
    xyz, rgb, _ = read_ply(SAMPLE_DIR / "normalized" / "points.ply")
    diagnostics = _read_json("normalized/diagnostics.json")

    assert len(xyz) == diagnostics["points"]["num_points_valid"]
    assert len(rgb) == len(xyz), "a colour without a point or a point without a colour"
    assert len(xyz) > 0, "an empty cloud is no good for developing the viewer"


def test_conventions_are_declared_not_implied() -> None:
    """Axes, units and scale status are written to the file (FR-021)."""
    conventions = _read_json("normalized/cameras.json")["conventions"]

    assert conventions["artifact_type"] == "point_cloud"
    assert conventions["transform_direction"] == "world_to_camera"
    assert conventions["units"] == "unknown"
    assert conventions["scale_status"] == "not_determined", "scale is not determined in v1"


def test_camera_poses_exist_and_agree_with_selected_frames() -> None:
    """The example is needed to show cameras (FR-025): they must exist and refer to frames."""
    cameras = _read_json("normalized/cameras.json")["cameras"]
    selected = {f["frame_id"] for f in _read_json("frames.json")["frames"] if f["selected"]}

    assert cameras, "without cameras the viewer has nothing to show"
    frame_ids = {c["frame_id"] for c in cameras}
    assert frame_ids <= selected, "the model has frames that are not among the selected ones"


def test_a_warning_sits_next_to_the_data() -> None:
    """The README is for a person who stumbles on the directory bypassing the viewer."""
    readme = (SAMPLE_DIR / "README.md").read_text(encoding="utf-8")
    assert "precomputed example" in readme
    assert "precomputed_example" in readme
