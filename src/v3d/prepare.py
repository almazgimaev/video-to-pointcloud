"""Stages 1–2: video check, frame selection, building the transfer package.

This ties together the modules of the light contour. Everything runs on the Mac, no GPU is needed.
Command contract: specs/001-video-3d-mvp/contracts/cli.md
"""

from __future__ import annotations

import platform
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from v3d import metrics, package, probe, runid, runs, warnings_
from v3d.artifacts import FrameRecord, Manifest, RunStatus, new_status, write_json
from v3d.config import DEFAULTS, preliminary_note
from v3d.errors import NotEnoughFramesError
from v3d.extract import extract_candidates
from v3d.select.checks import check_enough_frames
from v3d.select.quality import select_quality_nonredundant
from v3d.select.uniform import select_uniform, selection_summary

# The upstream command a human runs on the GPU machine. Our code is not moved there.
# Working parameters measured on M1 (docs/m1-feasibility.md): the upstream default confidence
# threshold of 5.0 produced zero points on real footage, 1.6 kept the object and dropped most of
# the background. Bundle adjustment (--use_ba) needs pyceres and was not used.
UPSTREAM_COMMAND = "python demo_colmap.py --scene_dir=<path to package> --conf_thres_value 1.6"


@dataclass
class PrepareResult:
    """Preparation outcome: what was created and what is worth telling the human."""

    run_id: str
    layout: runs.RunLayout
    records: list[FrameRecord]
    warnings: list[dict]
    package_summary: dict[str, Any]
    candidates_total: int
    frames_total_in_video: int | None


def _selection_params(
    *,
    budget: int,
    selector: str,
    long_side: int | None,
    jpeg_quality: int,
    max_candidates: int,
    min_frames: int,
) -> dict:
    """Parameters that affect the selection result; they go into the run digest."""
    return {
        "selector": selector,
        "budget": budget,
        "long_side": long_side,
        "jpeg_quality": jpeg_quality,
        "max_candidates": max_candidates,
        "min_frames": min_frames,
    }


def _collect_metrics(candidates) -> dict[int, float]:
    """Sharpness of every candidate. The metric is computed on a downscaled copy of the frame."""
    return {c.src_index: metrics.sharpness(metrics.load_gray(c.path)) for c in candidates}


def _cached_difference(candidates) -> Callable[[int, int], float]:
    """Frame difference by ``src_index`` with grayscale images loaded once each."""
    by_index = {c.src_index: c for c in candidates}
    cache: dict[int, np.ndarray] = {}

    def gray(src: int) -> np.ndarray:
        if src not in cache:
            cache[src] = metrics.load_gray(by_index[src].path)
        return cache[src]

    return lambda a, b: metrics.frame_difference(gray(a), gray(b))


def _fill_diffs(records: list[FrameRecord], candidates) -> list[float]:
    """Fill in the difference from the previous selected frame; return these values.

    The value is a proxy for viewpoint change, not a measurement of camera displacement (FR-007).
    """
    by_index = {c.src_index: c for c in candidates}
    diffs: list[float] = []
    prev_gray = None
    for rec in records:
        if not rec.selected:
            continue
        gray = metrics.load_gray(by_index[rec.src_index].path)
        if prev_gray is not None:
            value = metrics.frame_difference(prev_gray, gray)
            rec.diff_to_prev_selected = value
            diffs.append(value)
        prev_gray = gray
    return diffs


def _build_warnings(*, info, records: list[FrameRecord], diffs: list[float]) -> list[dict]:
    """Observation warnings. None of them names a cause (FR-007)."""
    found = []
    total = len(records)
    rejected_blur = sum(1 for r in records if r.reject_reason == "blurry")
    if total:
        w = warnings_.many_blurry_frames(rejected_blur / total, rejected_blur, total)
        if w:
            found.append(w)
    if diffs:
        w = warnings_.low_viewpoint_diversity(statistics.median(diffs))
        if w:
            found.append(w)
    w = warnings_.short_video(info.duration_s)
    if w:
        found.append(w)
    return [x.to_dict() for x in found]


def _write_failed_run(
    layout: runs.RunLayout,
    *,
    run_id: str,
    records: list[FrameRecord],
    message: str,
    warnings: list[dict],
) -> None:
    """Save diagnostics of a failed preparation: no package built, but the run stays readable."""
    write_json(
        layout.frames,
        {"schema_version": "1", "run_id": run_id, "frames": [r.to_dict() for r in records]},
    )
    write_json(
        layout.normalized / "status.json",
        new_status(run_id, RunStatus.FAILED, warnings=warnings, note=message),
    )


def run_prepare(
    video: Path,
    *,
    out_root: Path,
    budget: int = DEFAULTS.frame_budget,
    selector: str = "uniform",
    seed: int = DEFAULTS.seed,
    long_side: int | None = DEFAULTS.long_side_px,
    jpeg_quality: int = DEFAULTS.jpeg_quality,
    max_candidates: int = 600,
    min_frames: int = DEFAULTS.min_frames,
    force: bool = False,
) -> PrepareResult:
    """Run stages 1–2 and build the transfer package.

    Order: file check → run directory → candidate extraction → metrics → selection →
    sufficiency check → package build → strict integrity check → manifest.
    """
    if selector not in ("uniform", "quality"):
        raise ValueError(f"unknown selection method '{selector}'; use uniform or quality")
    if selector == "quality" and DEFAULTS.redundancy_diff_threshold is None:
        raise NotImplementedError("tau for quality selection is not tuned yet (task T049)")

    started = datetime.now(UTC)
    durations: dict[str, float] = {}

    with runs.stage_timer(durations, "check"):
        info = probe.probe_video(video)

    params = _selection_params(
        budget=budget,
        selector=selector,
        long_side=long_side,
        jpeg_quality=jpeg_quality,
        max_candidates=max_candidates,
        min_frames=min_frames,
    )
    code_id = runid.compute_code_id()
    digest = runid.inputs_digest(
        video_sha256=info.sha256, params=params, code_id=code_id.value, seed=seed
    )
    run_id = runid.make_run_id(digest, started)
    layout, forced = runs.create_run_dir(out_root, run_id, force=force)
    layout.ensure_dirs()

    work_frames = layout.root / "work" / "frames"
    with runs.stage_timer(durations, "extract"):
        candidates, image_transform = extract_candidates(
            video,
            work_frames,
            info=info,
            max_candidates=max_candidates,
            long_side=long_side,
            jpeg_quality=jpeg_quality,
        )

    selection_info: dict = {}
    if selector == "uniform":
        # Uniform selection does not need sharpness; it is measured only to describe the
        # frames, so its cost is not part of the selection cost.
        with runs.stage_timer(durations, "metrics"):
            sharpness_by_index = _collect_metrics(candidates)
        with runs.stage_timer(durations, "select"):
            records = select_uniform(
                candidates, budget=budget, seed=seed, sharpness_by_index=sharpness_by_index
            )
            diffs = _fill_diffs(records, candidates)
    else:
        # Quality selection needs sharpness and frame differences to decide, so both are
        # timed as part of "select": the selection cost must not look smaller than it is.
        with runs.stage_timer(durations, "select"):
            sharpness_by_index = _collect_metrics(candidates)
            records, selection_info = select_quality_nonredundant(
                candidates,
                budget=budget,
                tau=float(DEFAULTS.redundancy_diff_threshold),
                sharpness_by_index=sharpness_by_index,
                diff=_cached_difference(candidates),
            )
            diffs = [r.diff_to_prev_selected for r in records if r.diff_to_prev_selected]

    with runs.stage_timer(durations, "check_frames"):
        try:
            check_enough_frames(records, min_frames=min_frames)
        except NotEnoughFramesError as err:
            # Contract (contracts/errors.md): no package is created, but the run must stay
            # diagnosable. Write what is already known and mark the directory unusable,
            # otherwise a retry with the same parameters would hit "directory already exists".
            _write_failed_run(
                layout,
                run_id=run_id,
                records=records,
                message=err.message,
                warnings=_build_warnings(info=info, records=records, diffs=diffs),
            )
            err.details = (err.details or "") + (
                f"\nrun directory marked unusable: {layout.root}"
                "\nto prepare again with the same parameters, add --force"
            )
            raise

    with runs.stage_timer(durations, "package"):
        summary = package.build_package(
            work_frames,
            records,
            candidates,
            package_dir=layout.package,
            run_id=run_id,
            command_hint=UPSTREAM_COMMAND,
        )
        # Strict mode is appropriate only here: the package has not been transferred yet, so
        # there can be no extra files. After a return from the GPU machine the check is
        # non-strict (see run-package.md).
        package.verify_sha256sums(layout.package, strict=True)

    warns = _build_warnings(info=info, records=records, diffs=diffs)

    write_json(
        layout.frames,
        {
            "schema_version": "1",
            "run_id": run_id,
            "frames": [r.to_dict() for r in records],
        },
    )

    selection = selection_summary(records, budget=budget, seed=seed) | {
        "params": params,
        "quality": selection_info or None,
        "preliminary": [preliminary_note("frame_budget"), preliminary_note("min_frames")],
    }

    manifest = Manifest(
        run_id=run_id,
        created_utc=started.isoformat(),
        input_video=info.to_dict(),
        image_transform=image_transform,
        selection=selection,
        seed=seed,
        code_id=code_id.to_dict(),
        environment={
            "stage": "mac_light_contour",
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        commands=[{"command": "prepare", "durations_s": durations}],
        reconstructor=None,
        forced_overwrite=forced,
    )
    write_json(layout.manifest, manifest.to_dict())
    write_json(
        layout.normalized / "status.json",
        new_status(run_id, RunStatus.PREPARED, warnings=warns),
    )

    return PrepareResult(
        run_id=run_id,
        layout=layout,
        records=records,
        warnings=warns,
        package_summary=summary,
        candidates_total=len(candidates),
        frames_total_in_video=info.nb_frames,
    )
