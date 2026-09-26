"""Stage 4 on a synthetic result (T032).

The synthetic scene checks formats, conventions and the handling of unusable results.
It does not prove, and cannot prove, the quality of a real reconstruction.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from v3d.artifacts import RunStatus
from v3d.errors import (
    ExitCode,
    ResultEmptyOrInvalidError,
    ResultIncompleteError,
    ResultMissingOrIncompatibleError,
)
from v3d.ingest import run_ingest
from v3d.ply_io import read_ply
from v3d.prepare import run_prepare

REPO = Path(__file__).resolve().parents[2]
MAKE_SCENE = REPO / "assets" / "synthetic" / "make_scene.py"
BUDGET = 12


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory):
    """A ready run of stages 1–2: frames selected, the portable package assembled."""
    base = tmp_path_factory.mktemp("cycle")
    clip = base / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    return run_prepare(clip, out_root=base / "runs", budget=BUDGET, long_side=320)


def _make_scene(out: Path, *, frames_json: Path, cameras: int = BUDGET, points: int = 300) -> Path:
    """A scene with image names from a specific run: imitates a return from the GPU."""
    subprocess.run(
        [
            sys.executable,
            str(MAKE_SCENE),
            "--out",
            str(out),
            "--cameras",
            str(cameras),
            "--points",
            str(points),
            "--seed",
            "5",
            "--names-from",
            str(frames_json),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture
def run_dir(prepared, tmp_path: Path) -> Path:
    """A copy of the prepared run: each test spoils the result in its own way."""
    target = tmp_path / prepared.layout.root.name
    shutil.copytree(prepared.layout.root, target)
    return target


def test_ingesting_a_synthetic_result_gives_success(run_dir: Path, tmp_path: Path) -> None:
    """A full set of files and all cameras registered — status success."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json")
    result = run_ingest(run_dir, source=scene)

    assert result.status is RunStatus.SUCCESS
    assert result.cameras_registered == BUDGET
    assert result.points_valid > 0
    assert (run_dir / "normalized" / "cameras.json").exists()
    assert (run_dir / "normalized" / "points.ply").exists()
    assert (run_dir / "normalized" / "diagnostics.json").exists()


def test_normalized_result_declares_conventions(run_dir: Path, tmp_path: Path) -> None:
    """Axes, units and scale status are written, not implied (FR-021)."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json")
    run_ingest(run_dir, source=scene)

    cameras = json.loads((run_dir / "normalized" / "cameras.json").read_text(encoding="utf-8"))
    conv = cameras["conventions"]
    assert conv["artifact_type"] == "point_cloud"
    assert conv["transform_direction"] == "world_to_camera"
    assert conv["units"] == "unknown"
    assert conv["scale_status"] == "not_determined", "scale is not determined in v1"


def test_camera_poses_match_the_scene_reference(run_dir: Path, tmp_path: Path) -> None:
    """Main convention check: camera centres agree with what was put into the scene."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json")
    run_ingest(run_dir, source=scene)

    expected = json.loads((scene / "expected.json").read_text(encoding="utf-8"))
    cameras = json.loads((run_dir / "normalized" / "cameras.json").read_text(encoding="utf-8"))
    by_frame = {c["frame_id"]: c for c in cameras["cameras"]}

    for image in expected["images"]:
        frame_id = Path(image["name"]).stem
        actual = np.array(by_frame[frame_id]["camera_center_world"])
        assert np.allclose(actual, np.array(image["center"]), atol=1e-9), (
            f"camera centre {frame_id} diverged from the scene reference"
        )


def test_point_cloud_is_readable_and_carries_identification(run_dir: Path, tmp_path: Path) -> None:
    """The PLY must be identifiable by its run and declare the artifact type (FR-032)."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json")
    result = run_ingest(run_dir, source=scene)

    xyz, rgb, comments = read_ply(run_dir / "normalized" / "points.ply")
    assert len(xyz) == result.points_valid
    assert rgb.dtype == np.uint8
    assert comments["run_id"] == result.run_id
    assert comments["artifact_type"] == "point_cloud"
    assert comments["background_present"] == "true", "the residual background must be declared"


def test_diagnostics_does_not_replace_the_unavailable_with_zero(
    run_dir: Path, tmp_path: Path
) -> None:
    """Measures are either available or marked with a reason — but not zeroed (FR-036)."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json")
    run_ingest(run_dir, source=scene)

    diag = json.loads((run_dir / "normalized" / "diagnostics.json").read_text(encoding="utf-8"))
    error = diag["reprojection_error"]
    assert error["availability"] in {"available", "unavailable"}
    if error["availability"] == "unavailable":
        assert error["value"] is None and error["reason"]
    assert diag["points"]["background_present"] is True
    assert len(diag["visual_check"]) == 6
    assert diag["interpretation_limits"], "the limits of conclusions must be present"


def test_partial_registration_gives_status_partial(run_dir: Path, tmp_path: Path) -> None:
    """Part of the frames without poses is partial, not success."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json", cameras=5)
    result = run_ingest(run_dir, source=scene)

    assert result.status is RunStatus.PARTIAL
    assert result.cameras_registered == 5
    status = json.loads((run_dir / "normalized" / "status.json").read_text(encoding="utf-8"))
    assert status["viewable"] is True, "a partial result may be viewed"
    codes = [w["code"] for w in status["warnings"]]
    assert "few_registered_cameras" in codes, "incomplete registration must be named"


def test_missing_model_file_is_refused(run_dir: Path, tmp_path: Path) -> None:
    """The directory was not fully transferred — code 7 with the file name."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json")
    (scene / "sparse" / "points3D.bin").unlink()

    with pytest.raises(ResultMissingOrIncompatibleError) as excinfo:
        run_ingest(run_dir, source=scene)
    assert excinfo.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE
    assert "points3D.bin" in excinfo.value.message


def test_truncated_result_gives_status_interrupted(run_dir: Path, tmp_path: Path) -> None:
    """An empty model file is a sign of an interrupted computation (code 5), not of absence."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json")
    (scene / "sparse" / "points3D.bin").write_bytes(b"")

    with pytest.raises(ResultIncompleteError) as excinfo:
        run_ingest(run_dir, source=scene)

    assert excinfo.value.exit_code == ExitCode.RESULT_INCOMPLETE
    status = json.loads((run_dir / "normalized" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == RunStatus.INTERRUPTED.value
    assert status["exportable"] is False, "an incomplete result is not exported"


def test_empty_model_gives_status_invalid_result(run_dir: Path, tmp_path: Path) -> None:
    """A reconstruction without points is not passed off as success (code 6)."""
    scene = _make_scene(tmp_path / "scene", frames_json=run_dir / "frames.json", points=1)
    # Empty the cloud: the model can be read, but it has no points.
    empty = tmp_path / "empty" / "sparse"
    empty.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin"):
        shutil.copy2(scene / "sparse" / name, empty / name)
    (empty / "points3D.bin").write_bytes((0).to_bytes(8, "little"))

    with pytest.raises(ResultEmptyOrInvalidError) as excinfo:
        run_ingest(run_dir, source=empty.parent)

    assert excinfo.value.exit_code == ExitCode.RESULT_EMPTY_OR_INVALID
    status = json.loads((run_dir / "normalized" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == RunStatus.INVALID_RESULT.value


def test_foreign_result_is_not_accepted(run_dir: Path, tmp_path: Path) -> None:
    """A model with names from another set of frames — mixing runs is forbidden (FR-029)."""
    foreign = tmp_path / "foreign"
    subprocess.run(
        [
            sys.executable,
            str(MAKE_SCENE),
            "--out",
            str(foreign),
            "--cameras",
            "4",
            "--points",
            "50",
            "--seed",
            "1",
        ],
        check=True,
        capture_output=True,
    )

    with pytest.raises(ResultMissingOrIncompatibleError) as excinfo:
        run_ingest(run_dir, source=foreign)
    assert excinfo.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE
