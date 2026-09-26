"""Contract tests of point cloud export (T034, part of T037).

Sources of requirements:

* specs/001-video-3d-mvp/spec.md FR-029…FR-033 (match with the displayed cloud,
  file identification, ban on silent export of an unusable result);
* specs/001-video-3d-mvp/spec.md SC-003 (the PLY opens in a third-party viewer
  and contains the declared number of points);
* specs/001-video-3d-mvp/contracts/errors.md (exit codes 6, 7, 8).

The tests protect promises, not the implementation. The run directory is assembled by hand:
export reads only ``manifest.json`` and ``normalized/``, so the full preparation and ingest
cycle is not needed here and would only tie the tests to other stages.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from v3d.artifacts import RunStatus, default_conventions, new_diagnostics, new_status, write_json
from v3d.errors import (
    ExitCode,
    ResultEmptyOrInvalidError,
    ResultMissingOrIncompatibleError,
    RunDirExistsError,
)
from v3d.export import UNUSABLE_SUFFIX, export_point_cloud
from v3d.ply_io import read_ply, write_ply

RUN_ID = "20260101-120000_abc123"
OTHER_RUN_ID = "20260202-090000_def456"

# Points with clearly different coordinates and colours: a swap or recolouring would be visible.
POINTS_XYZ = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.5, -2.25, 3.125],
        [-10.0, 0.5, 42.0],
        [0.001, 0.002, 0.003],
        [7.75, 7.75, -7.75],
    ],
    dtype=np.float32,
)
POINTS_RGB = np.array(
    [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 128, 64],
        [17, 34, 51],
    ],
    dtype=np.uint8,
)

VERTEX_FORMAT = "<fffBBB"
VERTEX_STRIDE = struct.calcsize(VERTEX_FORMAT)


@dataclass(frozen=True)
class _Run:
    """A hand-assembled run directory and the paths inside it."""

    root: Path
    points: Path
    export_dir: Path


def _make_run(
    tmp_path: Path,
    *,
    status: RunStatus = RunStatus.SUCCESS,
    run_id: str = RUN_ID,
    points_run_id: str | None = None,
    xyz: np.ndarray = POINTS_XYZ,
    rgb: np.ndarray = POINTS_RGB,
    cameras: int = 4,
    scale_status: str = "not_determined",
) -> _Run:
    """A run directory with a manifest and a normalized result.

    ``points_run_id`` differs from ``run_id`` only where refusal on mixing files from
    different runs is tested.
    """
    root = tmp_path / "runs" / run_id
    normalized = root / "normalized"
    normalized.mkdir(parents=True)

    write_json(root / "manifest.json", {"run_id": run_id, "schema_version": "1"})

    conventions = default_conventions()
    conventions["scale_status"] = scale_status
    write_json(
        normalized / "cameras.json",
        {
            "schema_version": "1",
            "run_id": run_id,
            "conventions": conventions,
            "cameras": [{"frame_id": f"{i:04d}_000000"} for i in range(cameras)],
        },
    )
    write_json(normalized / "diagnostics.json", new_diagnostics(run_id))
    write_json(normalized / "status.json", new_status(run_id, status))

    write_ply(
        normalized / "points.ply",
        xyz,
        rgb,
        run_id=points_run_id or run_id,
        comments=["background_present: true", f"cameras: {cameras}"],
    )
    return _Run(root=root, points=normalized / "points.ply", export_dir=root / "export")


def _parse_ply(path: Path) -> tuple[dict[str, str], int, bytes]:
    """Independent PLY parsing: comments, declared point count and the binary tail.

    The parser is deliberately separate from ``v3d.ply_io``: it verifies that the file is
    readable by outside code from its header, and not only by our own reader (SC-003).
    """
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    assert end != -1, "the file has no end of header"
    header = raw[:end].decode("ascii")
    payload = raw[end + len(marker) :]

    comments: dict[str, str] = {}
    declared = -1
    properties: list[str] = []
    for line in header.splitlines():
        if line.startswith("comment "):
            key, _, value = line[len("comment ") :].partition(": ")
            comments[key] = value
        elif line.startswith("element vertex "):
            declared = int(line.split()[-1])
        elif line.startswith("property "):
            properties.append(line.split()[-1])
    assert properties == ["x", "y", "z", "red", "green", "blue"]
    return comments, declared, payload


def _exported_files(run: _Run) -> list[Path]:
    return sorted(p for p in run.export_dir.glob("*") if p.is_file())


# --- Match with the displayed cloud (FR-030, FR-031) ------------------------------------


def test_export_creates_file_matching_displayed_cloud(tmp_path: Path) -> None:
    """The export matches the displayed cloud: point count, coordinates and colours."""
    run = _make_run(tmp_path)

    exported = export_point_cloud(run.root)

    assert exported == run.export_dir / "object.ply"
    assert exported.is_file()

    shown_xyz, shown_rgb, _ = read_ply(run.points)
    got_xyz, got_rgb, _ = read_ply(exported)

    assert len(got_xyz) == len(shown_xyz)
    assert np.array_equal(got_xyz, shown_xyz)
    assert np.array_equal(got_rgb, shown_rgb)


def test_export_does_not_touch_source_cloud(tmp_path: Path) -> None:
    """The export source is not rewritten: the displayed cloud stays as it was."""
    run = _make_run(tmp_path)
    before = run.points.read_bytes()

    export_point_cloud(run.root)

    assert run.points.read_bytes() == before


# --- File identification (FR-032) -------------------------------------------------------


def test_header_carries_run_id_and_summary(tmp_path: Path) -> None:
    """The header has ``run_id`` and a run summary: the file cannot be confused with another."""
    run = _make_run(tmp_path, cameras=3, scale_status="not_determined")

    exported = export_point_cloud(run.root)
    comments, _, _ = _parse_ply(exported)

    assert comments["run_id"] == RUN_ID
    assert comments["scale_status"] == "not_determined"
    assert comments["points"] == str(len(POINTS_XYZ))
    assert comments["cameras"] == "3"
    assert comments["status"] == RunStatus.SUCCESS.value


def test_header_scale_status_comes_from_run_conventions(tmp_path: Path) -> None:
    """``scale_status`` is taken from the run conventions, not implied."""
    run = _make_run(tmp_path, scale_status="model_estimated")

    comments, _, _ = _parse_ply(export_point_cloud(run.root))

    assert comments["scale_status"] == "model_estimated"


# --- Unusable result (FR-033) -----------------------------------------------------------


@pytest.mark.parametrize("status", [RunStatus.INVALID_RESULT, RunStatus.INTERRUPTED])
def test_unusable_status_refused_without_force(tmp_path: Path, status: RunStatus) -> None:
    """An unusable result is not exported silently: code 6 and no file."""
    run = _make_run(tmp_path, status=status)

    with pytest.raises(ResultEmptyOrInvalidError) as err:
        export_point_cloud(run.root)

    assert err.value.exit_code == ExitCode.RESULT_EMPTY_OR_INVALID
    assert status.value in err.value.render()
    assert not run.export_dir.exists() or _exported_files(run) == []


@pytest.mark.parametrize("status", [RunStatus.INVALID_RESULT, RunStatus.INTERRUPTED])
def test_force_writes_unusable_marked_file(tmp_path: Path, status: RunStatus) -> None:
    """With ``force`` the file is written, but the name and header report unusability."""
    run = _make_run(tmp_path, status=status)

    exported = export_point_cloud(run.root, force=True)

    assert exported.is_file()
    assert "UNUSABLE" in exported.name
    assert exported.name.endswith(UNUSABLE_SUFFIX)
    assert _exported_files(run) == [exported]

    comments, _, _ = _parse_ply(exported)
    assert comments["unusable"] == "true"
    assert status.value in comments["unusable_reason"]
    assert comments["status"] == status.value


def test_force_marks_user_given_out_without_moving_it(tmp_path: Path) -> None:
    """The mark is added to the user-given name and does not replace the path."""
    run = _make_run(tmp_path, status=RunStatus.INTERRUPTED)
    requested = tmp_path / "elsewhere" / "my-cloud.ply"
    requested.parent.mkdir()

    exported = export_point_cloud(run.root, out=requested, force=True)

    assert exported.parent == requested.parent
    assert exported.name == "my-cloud" + UNUSABLE_SUFFIX
    assert not requested.exists()


def test_empty_cloud_refused_even_on_success(tmp_path: Path) -> None:
    """An empty cloud is refused with code 6 even under status ``success``: nothing to export."""
    run = _make_run(
        tmp_path,
        xyz=np.empty((0, 3), dtype=np.float32),
        rgb=np.empty((0, 3), dtype=np.uint8),
    )

    with pytest.raises(ResultEmptyOrInvalidError) as err:
        export_point_cloud(run.root)

    assert err.value.exit_code == ExitCode.RESULT_EMPTY_OR_INVALID
    assert not run.export_dir.exists() or _exported_files(run) == []


def test_empty_cloud_refused_even_with_force(tmp_path: Path) -> None:
    """An empty file would look like a valid result, so ``force`` does not create it."""
    run = _make_run(
        tmp_path,
        xyz=np.empty((0, 3), dtype=np.float32),
        rgb=np.empty((0, 3), dtype=np.uint8),
    )

    with pytest.raises(ResultEmptyOrInvalidError):
        export_point_cloud(run.root, force=True)

    assert not run.export_dir.exists() or _exported_files(run) == []


# --- Protection of a previous export (code 8) -------------------------------------------


def test_second_export_without_force_keeps_previous_file(tmp_path: Path) -> None:
    """A repeated export without ``force`` is code 8, the previous file is untouched."""
    run = _make_run(tmp_path)
    first = export_point_cloud(run.root)
    before = first.read_bytes()

    with pytest.raises(RunDirExistsError) as err:
        export_point_cloud(run.root)

    assert err.value.exit_code == ExitCode.RUN_DIR_EXISTS
    assert first.read_bytes() == before


def test_second_export_with_force_overwrites(tmp_path: Path) -> None:
    """With ``force`` the previous export is overwritten — an explicit user decision."""
    run = _make_run(tmp_path)
    first = export_point_cloud(run.root)

    again = export_point_cloud(run.root, force=True)

    assert again == first
    assert _exported_files(run) == [first]


# --- Run mismatch (FR-029, code 7) ------------------------------------------------------


def test_run_id_mismatch_refused(tmp_path: Path) -> None:
    """A cloud from another run is not exported: code 7 naming both ``run_id`` values."""
    run = _make_run(tmp_path, points_run_id=OTHER_RUN_ID)

    with pytest.raises(ResultMissingOrIncompatibleError) as err:
        export_point_cloud(run.root)

    assert err.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE
    rendered = err.value.render()
    assert RUN_ID in rendered and OTHER_RUN_ID in rendered
    assert not run.export_dir.exists() or _exported_files(run) == []


def test_missing_normalized_files_refused(tmp_path: Path) -> None:
    """A required result file is missing — code 7, not an empty export."""
    run = _make_run(tmp_path)
    (run.root / "normalized" / "status.json").unlink()

    with pytest.raises(ResultMissingOrIncompatibleError) as err:
        export_point_cloud(run.root)

    assert err.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE


# --- Readability by outside code (SC-003) -----------------------------------------------


def test_exported_file_is_readable_by_independent_parser(tmp_path: Path) -> None:
    """The file is readable by its own header: exactly as much data as declared."""
    run = _make_run(tmp_path)

    exported = export_point_cloud(run.root)
    comments, declared, payload = _parse_ply(exported)

    assert declared == len(POINTS_XYZ)
    assert comments["points"] == str(declared)
    assert len(payload) == declared * VERTEX_STRIDE

    for i in range(declared):
        x, y, z, r, g, b = struct.unpack_from(VERTEX_FORMAT, payload, i * VERTEX_STRIDE)
        assert (x, y, z) == tuple(POINTS_XYZ[i])
        assert (r, g, b) == tuple(int(c) for c in POINTS_RGB[i])
