"""Stage 4: receiving the result from the GPU machine and converting it to our representation.

Runs on the Mac, no GPU needed: only files are read. Order of work —
integrity checks, model reading, normalization, diagnostics, status.

The key rule of the stage: a failure must remain readable. If the result is truncated or
empty, we still write the status and diagnostics and then report the error via the
exit code — silently passing off an unusable result as successful is forbidden (Principle I).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from v3d import colmap_io, diagnostics, geometry, ingest_checks, ply_io, runs, status
from v3d.artifacts import (
    RunStatus,
    default_conventions,
    read_json,
    write_json,
)
from v3d.config import DEFAULTS
from v3d.errors import ResultEmptyOrInvalidError, ResultIncompleteError

SPARSE_SUBDIR = "sparse"


@dataclass
class IngestResult:
    """Outcome of ingest: what happened and what the artifacts say about it."""

    run_id: str
    layout: runs.RunLayout
    status: RunStatus
    cameras_registered: int
    selected_frames: int
    points_raw: int
    points_valid: int
    banner: str


def _copy_result(source: Path, layout: runs.RunLayout) -> None:
    """Move the result returned from the GPU machine into the run directory.

    Only what belongs to the result is copied: the model and the upstream cloud. Frames
    and checksums are already in the run, no need to duplicate them.
    """
    if source.resolve() == layout.result.resolve():
        return
    layout.result.mkdir(parents=True, exist_ok=True)
    sparse_src = source / SPARSE_SUBDIR
    if sparse_src.is_dir():
        shutil.copytree(sparse_src, layout.result / SPARSE_SUBDIR, dirs_exist_ok=True)
    for name in ("points.ply", "stdout.log", "run_log.txt"):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, layout.result / name)


def _selected_frame_ids(frames_payload: dict) -> list[str]:
    return [f["frame_id"] for f in frames_payload["frames"] if f["selected"]]


def _write_failed(
    layout: runs.RunLayout,
    *,
    run_id: str,
    run_status: RunStatus,
    note: str,
) -> None:
    """Save the status of an unusable result: the run stays diagnosable."""
    payload = status.build_status(run_id, run_status, note=note)
    write_json(layout.normalized / "status.json", payload)


def run_ingest(
    run_dir: Path,
    *,
    source: Path | None = None,
    outlier_k: float = DEFAULTS.outlier_iqr_k,
    precomputed_example: bool = False,
) -> IngestResult:
    """Receive the result, check it and write the normalized representation."""
    layout = runs.find_run_dir(run_dir)
    manifest = read_json(layout.manifest)
    frames_payload = read_json(layout.frames)
    run_id = manifest["run_id"]
    ingest_checks.check_run_id_matches(manifest, ("frames.json", frames_payload))

    if source is not None:
        _copy_result(Path(source), layout)

    durations: dict[str, float] = {}
    with runs.stage_timer(durations, "ingest"):
        try:
            ingest_checks.check_result_files(layout.result)
        except ResultIncompleteError as err:
            _write_failed(layout, run_id=run_id, run_status=RunStatus.INTERRUPTED, note=err.message)
            raise

        model = colmap_io.read_sparse(layout.result / SPARSE_SUBDIR)
        selected_ids = _selected_frame_ids(frames_payload)
        ingest_checks.check_images_subset(model.image_names, selected_ids)

        try:
            ingest_checks.check_model_not_empty(model)
        except ResultEmptyOrInvalidError as err:
            _write_failed(
                layout, run_id=run_id, run_status=RunStatus.INVALID_RESULT, note=err.message
            )
            raise

        kept_xyz, kept_rgb, filter_info = geometry.filter_outliers_iqr(
            model.points_xyz, model.points_rgb, k=outlier_k
        )

        conventions = default_conventions()
        write_json(
            layout.normalized / "cameras.json",
            {
                "schema_version": "1",
                "run_id": run_id,
                "conventions": conventions,
                "cameras": [c.to_dict() for c in model.cameras],
            },
        )
        ply_io.write_ply(
            layout.normalized / "points.ply",
            kept_xyz,
            kept_rgb if kept_rgb is not None else model.points_rgb,
            run_id=run_id,
            comments=[
                "background_present: true",
                f"cameras: {len(model.cameras)}",
                "source: reconstructor output, outliers filtered",
            ],
        )

    artifact_sizes = _artifact_sizes(layout)
    diag = diagnostics.build_diagnostics(
        run_id,
        frames=frames_payload["frames"],
        model=model,
        valid_points_xyz=kept_xyz,
        filters_applied=[filter_info],
        durations_s=durations,
        artifact_sizes=artifact_sizes,
    )
    write_json(layout.normalized / "diagnostics.json", diag)

    run_status = status.determine_status(
        cameras_registered=len(model.cameras),
        points_valid=len(kept_xyz),
        selected_frames=len(selected_ids),
        result_complete=True,
    )
    status_payload = status.build_status(
        run_id,
        run_status,
        precomputed_example=precomputed_example,
        cameras_registered=len(model.cameras),
        selected_frames=len(selected_ids),
    )
    write_json(layout.normalized / "status.json", status_payload)

    return IngestResult(
        run_id=run_id,
        layout=layout,
        status=run_status,
        cameras_registered=len(model.cameras),
        selected_frames=len(selected_ids),
        points_raw=len(model.points_xyz),
        points_valid=len(kept_xyz),
        banner=status.status_banner(status_payload),
    )


def _artifact_sizes(layout: runs.RunLayout) -> dict[str, int]:
    """Sizes of output artifacts; a missing file is not counted as zero."""
    sizes: dict[str, int] = {}
    candidates: dict[str, Path] = {
        "normalized/points.ply": layout.normalized / "points.ply",
        "normalized/cameras.json": layout.normalized / "cameras.json",
        "result/points.ply": layout.result / "points.ply",
        "result/sparse": layout.result / SPARSE_SUBDIR,
    }
    for name, path in candidates.items():
        if path.is_dir():
            sizes[name] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            sizes[name] = path.stat().st_size
    return sizes


def ingest_summary_lines(result: IngestResult) -> list[str]:
    """Lines for terminal output: facts without quality judgements."""
    return [
        f"run: {result.run_id}",
        f"  status: {result.banner}",
        f"  cameras registered: {result.cameras_registered} of {result.selected_frames}",
        f"  points: {result.points_valid} valid of {result.points_raw} in the model",
        f"  normalized result: {result.layout.normalized}",
    ]
