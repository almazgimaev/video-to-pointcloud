"""Tests of the normalized-result viewer (T033).

Requirements: spec.md FR-024…FR-029, data-model.md §3.4 and §3.7.

What is protected here:
* viewing reads only saved artifacts and writes `.rrd` without opening a window;
* a set of files from different runs and a missing `normalized/` are refused with code 7;
* the status line and the "precomputed example" mark are not lost;
* a camera without `intrinsics` does not crash the viewer.

No test opens a window: everywhere `spawn=False` and writing to a file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v3d.artifacts import (
    CameraRecord,
    RunStatus,
    available,
    default_conventions,
    new_diagnostics,
    unavailable,
    write_json,
)
from v3d.errors import ExitCode, ResultMissingOrIncompatibleError
from v3d.ply_io import write_ply
from v3d.status import PRECOMPUTED_EXAMPLE_MARK, build_status
from v3d.viewer import POINTS_PATH, POINTS_UNUSABLE_PATH, UNUSABLE_MARK, build_recording

RUN_ID = "20260919-120000-ab12cd"


# --- Fixtures: a minimal run directory assembled by hand --------------------------------


def _camera(frame_id: str, *, offset: float, intrinsics: dict | None) -> dict:
    """A camera record with identity rotation and a shift along X."""
    return CameraRecord(
        frame_id=frame_id,
        image_size=(640, 480),
        intrinsics=intrinsics,
        R_world_to_camera=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t_world_to_camera=[-offset, 0.0, 0.0],
        camera_center_world=[offset, 0.0, 0.0],
    ).to_dict()


INTRINSICS = {"model": "PINHOLE", "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}


def _build_run(
    root: Path,
    *,
    run_id: str = RUN_ID,
    status: RunStatus = RunStatus.SUCCESS,
    precomputed_example: bool = False,
    cameras: list[dict] | None = None,
    cameras_run_id: str | None = None,
    num_points: int = 12,
) -> Path:
    """A minimal run directory: manifest, frames and `normalized/`.

    The full `prepare`+`ingest` cycle is not needed here: the viewer only reads files.
    """
    run_dir = root / run_id
    normalized = run_dir / "normalized"
    normalized.mkdir(parents=True)

    camera_records = (
        cameras
        if cameras is not None
        else [_camera("0001_000000", offset=0.5, intrinsics=INTRINSICS)]
    )
    selected = max(len(camera_records), 1)

    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "1",
            "run_id": run_id,
            "created_utc": "2026-09-19T12:00:00+00:00",
            "input_video": {
                "filename": "sample.mp4",
                "sha256": "0" * 64,
                "duration_s": 12.5,
                "width": 1920,
                "height": 1080,
                "container": "mov,mp4",
                "video_codec": "h264",
                "rotation_deg": 0,
            },
            "image_transform": {"resize_mode": "long_side", "resized_size": [1024, 576]},
            "selection": {
                "strategy": "uniform",
                "budget": selected,
                "selected": selected,
                "candidates_total": 30,
                "params": {"strategy": "uniform", "budget": selected},
            },
            "seed": 42,
            "code_id": {"git_commit": "0123456", "dirty": False},
            "environment": {"stage": "mac_light_contour", "python": "3.11.9"},
            "commands": [],
        },
    )
    write_json(
        run_dir / "frames.json",
        {"schema_version": "1", "run_id": run_id, "frames": []},
    )
    write_json(
        normalized / "cameras.json",
        {
            "schema_version": "1",
            "run_id": cameras_run_id or run_id,
            "conventions": default_conventions(),
            "cameras": camera_records,
        },
    )

    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(num_points, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(num_points, 3), dtype=np.uint8)
    write_ply(normalized / "points.ply", xyz, rgb, run_id=run_id)

    diagnostics = new_diagnostics(run_id)
    diagnostics["frames"] = {"frames_total": 30, "frames_selected": selected}
    diagnostics["cameras"] = {
        "cameras_registered": len(camera_records),
        "registration_ratio": available(len(camera_records) / selected),
    }
    diagnostics["points"] = {
        "num_points_raw": num_points,
        "num_points_valid": num_points,
        "has_colors": True,
        "background_present": True,
        "bbox": available([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
    }
    diagnostics["stage_durations_s"] = {"ingest": 0.5}
    diagnostics["reprojection_error"] = unavailable("the model contains no mean error")
    write_json(normalized / "diagnostics.json", diagnostics)

    write_json(
        normalized / "status.json",
        build_status(
            run_id,
            status,
            precomputed_example=precomputed_example,
            cameras_registered=len(camera_records),
            selected_frames=selected + (1 if status is RunStatus.PARTIAL else 0),
        ),
    )
    return run_dir


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return _build_run(tmp_path / "runs")


# --- 1. Writing .rrd without a window ---------------------------------------------------


def test_writes_non_empty_rrd_without_window(run: Path, tmp_path: Path) -> None:
    file = tmp_path / "s.rrd"
    summary = build_recording(run, save_to=file, spawn=False)

    assert file.is_file()
    assert file.stat().st_size > 0
    assert summary["saved_to"] == str(file)
    assert summary["spawned"] is False


def test_summary_matches_input_files(run: Path, tmp_path: Path) -> None:
    cameras = [
        _camera("0001_000000", offset=0.5, intrinsics=INTRINSICS),
        _camera("0002_000010", offset=1.5, intrinsics=INTRINSICS),
    ]
    run_dir = _build_run(tmp_path / "runs2", cameras=cameras, num_points=25)
    summary = build_recording(run_dir, save_to=tmp_path / "s2.rrd", spawn=False)

    assert summary["points"] == 25
    assert summary["cameras"] == 2
    assert summary["cameras_with_intrinsics"] == 2
    assert summary["run_id"] == RUN_ID
    assert summary["entities"]["points"] == POINTS_PATH


def test_run_info_and_measures_in_panels(run: Path, tmp_path: Path) -> None:
    summary = build_recording(run, save_to=tmp_path / "s.rrd", spawn=False)

    assert "sample.mp4" in summary["panels"]["info"]
    assert "1920" in summary["panels"]["info"]
    # An unavailable measure — with a reason, not zero (FR-036).
    assert "unavailable: the model contains no mean error" in summary["panels"]["metrics"]
    assert "0 px" not in summary["panels"]["metrics"]


# --- 2. Refusal for an unusable set (FR-029) --------------------------------------------


def test_missing_normalized_is_refused_with_code_7(run: Path, tmp_path: Path) -> None:
    for file in (run / "normalized").iterdir():
        file.unlink()
    (run / "normalized").rmdir()

    with pytest.raises(ResultMissingOrIncompatibleError) as error:
        build_recording(run, save_to=tmp_path / "s.rrd", spawn=False)
    assert error.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE


def test_foreign_run_id_in_cameras_is_refused_with_code_7(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path / "runs", cameras_run_id="20260101-000000-ffffff")

    with pytest.raises(ResultMissingOrIncompatibleError) as error:
        build_recording(run_dir, save_to=tmp_path / "s.rrd", spawn=False)
    assert error.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE
    assert "cameras.json" in error.value.message


def test_missing_point_cloud_is_refused_with_code_7(run: Path, tmp_path: Path) -> None:
    (run / "normalized" / "points.ply").unlink()

    with pytest.raises(ResultMissingOrIncompatibleError) as error:
        build_recording(run, save_to=tmp_path / "s.rrd", spawn=False)
    assert error.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE


# --- 3. Status is always visible (FR-028) -----------------------------------------------


def test_partial_result_is_visible_in_banner(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path / "runs", status=RunStatus.PARTIAL)
    summary = build_recording(run_dir, save_to=tmp_path / "s.rrd", spawn=False)

    assert "partial" in summary["banner"]
    assert "partial" in summary["panels"]["status"]
    assert summary["warnings"] >= 1
    assert "few_registered_cameras" in summary["panels"]["warnings"]


def test_precomputed_example_mark_is_not_lost(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path / "runs", precomputed_example=True)
    summary = build_recording(run_dir, save_to=tmp_path / "s.rrd", spawn=False)

    assert summary["precomputed_example"] is True
    assert PRECOMPUTED_EXAMPLE_MARK in summary["banner"]
    assert PRECOMPUTED_EXAMPLE_MARK in summary["panels"]["status"]
    assert PRECOMPUTED_EXAMPLE_MARK in summary["marks"]


@pytest.mark.parametrize("status", [RunStatus.INTERRUPTED, RunStatus.INVALID_RESULT])
def test_unusable_result_is_opened_with_a_mark(status: RunStatus, tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path / "runs", status=status)
    file = tmp_path / "s.rrd"
    summary = build_recording(run_dir, save_to=file, spawn=False)

    assert file.is_file()
    assert summary["viewable"] is False
    assert UNUSABLE_MARK in summary["panels"]["status"]
    assert UNUSABLE_MARK in summary["marks"]
    # The cloud goes to a separate path so that it cannot be mistaken for a usable result.
    assert summary["entities"]["points"] == POINTS_UNUSABLE_PATH


# --- 4. Camera without intrinsics -------------------------------------------------------


def test_camera_without_intrinsics_does_not_crash_the_viewer(tmp_path: Path) -> None:
    cameras = [
        _camera("0001_000000", offset=0.5, intrinsics=None),
        _camera("0002_000010", offset=1.5, intrinsics=INTRINSICS),
    ]
    run_dir = _build_run(tmp_path / "runs", cameras=cameras)
    file = tmp_path / "s.rrd"
    summary = build_recording(run_dir, save_to=file, spawn=False)

    assert file.stat().st_size > 0
    assert summary["cameras"] == 2
    assert summary["cameras_with_intrinsics"] == 1


def test_up_axis_is_logged_at_the_3d_view_origin(tmp_path: Path) -> None:
    """Rerun reads the up axis at the view's origin; logging it elsewhere leaves the view tilted."""
    from v3d.artifacts import read_json, write_json
    from v3d.viewer import WORLD_PATH, build_recording

    run_dir = _build_run(tmp_path)
    cameras_path = run_dir / "normalized" / "cameras.json"
    cameras = read_json(cameras_path)
    cameras["world_alignment"] = {"status": "estimated", "quality": {}, "note": "estimated"}
    write_json(cameras_path, cameras)

    summary = build_recording(run_dir, save_to=tmp_path / "s.rrd", spawn=False)
    coords = summary["view_coordinates"]
    assert coords["up"] == "+Y"
    assert coords["path"] == WORLD_PATH
    assert coords["view_origin"] == f"/{WORLD_PATH}", "coordinates must sit on the view origin"
