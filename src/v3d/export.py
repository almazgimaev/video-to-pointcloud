"""Stage 5: export of the displayed point cloud to PLY (T034).

Runs on the Mac and recomputes nothing: only ``manifest.json`` and
``normalized/`` are read (FR-027). There is exactly one export source — ``normalized/points.ply``,
the very cloud that the viewer displays.

Rules of the stage:

* the export matches the displayed cloud (FR-031): no filtering, thinning
  or recolouring happens here;
* the file carries ``run_id`` and a run summary (FR-032), so that it cannot be confused
  with the result of another run;
* an unusable result is not passed off as a regular export under any flags (FR-033):
  without ``force`` — refusal with code 6, with ``force`` — a file with the suffix
  ``.UNUSABLE.ply`` and an unusability mark in the header.

The PLY header is written in ASCII (see :func:`v3d.ply_io.write_ply`), so the comments
here are ASCII-only: non-ASCII text would make the file unwritable.
"""

from __future__ import annotations

from pathlib import Path

from v3d import ply_io, runs
from v3d.artifacts import (
    EXPORTABLE_STATUSES,
    RunStatus,
    read_json,
    require_same_run_id,
)
from v3d.errors import (
    ResultEmptyOrInvalidError,
    ResultMissingOrIncompatibleError,
    RunDirExistsError,
)

DEFAULT_EXPORT_NAME = "object.ply"
SOURCE_RELPATH = "normalized/points.ply"
UNUSABLE_SUFFIX = ".UNUSABLE.ply"

# Statuses under which export is possible only with an explicit unusability mark (FR-033).
# This is the complement of `EXPORTABLE_STATUSES`: only `success` and `partial` are usable,
# everything else (including `invalid_result` and `interrupted`) requires the mark.
_UNUSABLE_STATUSES = frozenset(s for s in RunStatus if s not in EXPORTABLE_STATUSES)


def unusable_path(path: Path) -> Path:
    """File name for an unusable result: ``<name>.UNUSABLE.ply``.

    A user-specified path is not swapped for another directory and is not silently
    renamed — only the file name changes, so that unusability is visible in a file listing.
    """
    return path.with_name(path.stem + UNUSABLE_SUFFIX)


def _read_status(layout: runs.RunLayout) -> tuple[dict, RunStatus]:
    """Run status from ``normalized/status.json``.

    An unknown status value is code 7: this is a format we do not understand, and
    one must not guess the usability of a result from an unfamiliar string.
    """
    payload = read_json(layout.normalized / "status.json")
    raw = payload.get("status")
    try:
        return payload, RunStatus(raw)
    except ValueError as err:
        raise ResultMissingOrIncompatibleError(
            f"status.json has an unknown status: {raw!r}",
            details="the file was written by an incompatible format version",
        ) from err


def _read_cameras_count(layout: runs.RunLayout) -> tuple[dict, int, str]:
    """Camera count and scale status from ``normalized/cameras.json``."""
    payload = read_json(layout.normalized / "cameras.json")
    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        raise ResultMissingOrIncompatibleError(
            "cameras.json has no camera list",
            details="the file was written by an incompatible format version",
        )
    conventions = payload.get("conventions") or {}
    scale_status = str(conventions.get("scale_status", "not_determined"))
    return payload, len(cameras), scale_status


def _summary_comments(
    *,
    points: int,
    cameras: int,
    status: RunStatus,
    scale_status: str,
    unusable_reason: str | None,
) -> list[str]:
    """Run summary in header comments (FR-032). ASCII only.

    ``scale_status`` is written explicitly and after the placeholder line from
    :func:`ply_io.write_ply`: the value is taken from the run conventions, not implied.
    """
    comments = [
        f"export_of: {SOURCE_RELPATH}",
        f"points: {points}",
        f"cameras: {cameras}",
        f"status: {status.value}",
        f"scale_status: {scale_status}",
        "note: exported point cloud matches the displayed one; no extra filtering applied",
    ]
    if unusable_reason is not None:
        comments += [
            "unusable: true",
            f"unusable_reason: {unusable_reason}",
        ]
    return comments


def export_point_cloud(run_dir: Path, *, out: Path | None = None, force: bool = False) -> Path:
    """Export the displayed point cloud to PLY and return the path of the written file.

    Exactly ``normalized/points.ply`` is exported: the same number of points, the same
    coordinates, the same colours (FR-031). **No additional filtering, thinning, recomputation
    or recolouring is performed at this step** — otherwise the user would see one thing in the
    viewer and get another in the file.

    Args:
        run_dir: run directory ``runs/<run_id>``.
        out: export file path; without it — ``<run_dir>/export/object.ply``.
        force: write over an existing file and issue an unusable result
            with an explicit mark.

    Returns:
        Path of the written file. For an unusable result the name carries the suffix
        ``.UNUSABLE.ply``.

    Raises:
        ResultMissingOrIncompatibleError: required files are missing, ``run_id`` values do not
            match, or the format is incompatible (code 7).
        ResultEmptyOrInvalidError: the cloud is empty or the result is deemed unusable
            and ``force`` is not set (code 6).
        RunDirExistsError: the export file already exists and ``force`` is not set (code 8).
    """
    layout = runs.find_run_dir(run_dir)
    manifest = read_json(layout.manifest)
    run_id = manifest.get("run_id")
    if not run_id:
        raise ResultMissingOrIncompatibleError(
            f"the manifest has no run_id: {layout.manifest}",
            details="this is not a run directory or the manifest is damaged",
        )

    status_payload, status = _read_status(layout)
    cameras_payload, cameras_count, scale_status = _read_cameras_count(layout)

    source = layout.normalized / Path(SOURCE_RELPATH).name
    points_xyz, points_rgb, source_comments = ply_io.read_ply(source)

    # Files from different runs must not be mixed: the export must belong to a single run
    # as a whole (FR-029, FR-032).
    require_same_run_id(
        run_id,
        ("normalized/status.json", status_payload),
        ("normalized/cameras.json", cameras_payload),
        (SOURCE_RELPATH, source_comments),
    )

    unusable_reason: str | None = None
    if status in _UNUSABLE_STATUSES:
        unusable_reason = f"run status is {status.value}; exported only because --force was given"
        if not force:
            raise ResultEmptyOrInvalidError(
                f"the run result is unusable: status {status.value}",
                details=(
                    "exporting such a result was refused so that it cannot be mistaken "
                    "for a regular one; repeat with --force — the file will be written with the "
                    f"suffix {UNUSABLE_SUFFIX} and an unusability mark in the header"
                ),
            )

    # An empty cloud is not exported even with --force: a file without a single point would look
    # like a valid result, and there is nothing to show in it.
    if len(points_xyz) == 0:
        raise ResultEmptyOrInvalidError(
            f"the run cloud has no points: {source}",
            details="there is nothing to export; the result is unusable regardless of status",
        )

    target = Path(out) if out is not None else layout.export / DEFAULT_EXPORT_NAME
    if unusable_reason is not None:
        target = unusable_path(target)

    if target.exists() and not force:
        raise RunDirExistsError(
            f"export file already exists: {target}",
            details=(
                "a previous export is not silently overwritten; repeat with --force "
                "or specify a different --out"
            ),
        )

    written = ply_io.write_ply(
        target,
        points_xyz,
        points_rgb,
        run_id=run_id,
        comments=_summary_comments(
            points=len(points_xyz),
            cameras=cameras_count,
            status=status,
            scale_status=scale_status,
            unusable_reason=unusable_reason,
        ),
    )
    if written != len(points_xyz):
        raise ResultEmptyOrInvalidError(
            f"wrote {written} points instead of {len(points_xyz)}",
            details="the export must match the displayed cloud",
        )
    return target


def export_summary_lines(path: Path, *, points: int, unusable: bool) -> list[str]:
    """Lines for terminal output: facts without quality judgements."""
    lines = [f"export: {path}", f"  points: {points}"]
    if unusable:
        lines.append("  mark: the result was deemed unusable, the file was issued due to --force")
    return lines
