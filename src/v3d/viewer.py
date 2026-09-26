"""Viewing the normalized result in Rerun (T033).

Requirements: spec.md FR-024…FR-029, data-model.md §3.4 and §3.7.

The module **only reads** saved artifacts (`manifest.json`, `frames.json`,
`normalized/`) and assembles a Rerun recording from them. Nothing is recomputed, no GPU
and no network are needed (FR-027): no call here leads to launching the reconstructor.

What goes into the recording:

* `world/points` — the coloured point cloud from `normalized/points.ply`;
* `world/cameras/**` — camera poses as a separate path branch, so that the layer can be
  hidden and shown in the Rerun interface (FR-025);
* `run/info` — run information from the manifest (FR-026);
* `run/warnings` — run warnings from `normalized/status.json` (FR-026);
* `run/status` — the permanent status line with the "precomputed example" mark,
  if it is present in the metadata (FR-028, A-07);
* `run/metrics` — measures from `normalized/diagnostics.json`.

Decisions made here deliberately:

* **Wording is not invented.** The status line is taken from :func:`v3d.status.status_banner`,
  warning texts are carried over from `status.json` verbatim, measures — from
  `diagnostics.json` as is.
* **An unavailable measure is shown as an unavailability marker with a reason, never as zero**
  (FR-036): zero in the reprojection error field would read as a perfect reconstruction.
* **An unusable result is not passed off as usable.** If the `viewable` field in `status.json`
  is false (statuses `interrupted` and `invalid_result`), the result can still be opened —
  an interruption has to be diagnosed somehow — but the cloud is logged to the path
  ``world/points_unusable`` and accompanied by the mark "result unusable" in the status
  panel and in the returned summary. Silently showing it as a regular result is not allowed
  (FR-033).
* **Matrix arithmetic is not duplicated**: conversion of poses to camera_to_world is taken
  from :mod:`v3d.geometry`.

About the Rerun API: the module works with 0.38.1 and does not rely on global state
(`rr.init`/`rr.log` without a receiver) — the recording is assembled in an explicit
:class:`rerun.RecordingStream`, so parallel calls do not interfere with each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from v3d import geometry, ingest_checks, ply_io, runs
from v3d.artifacts import is_measure, read_json
from v3d.errors import ResultMissingOrIncompatibleError
from v3d.status import status_banner

#: Application identifier in the Rerun recording.
APPLICATION_ID = "v3d"

#: Entity paths. Cameras are placed in a separate branch so that the layer toggles as a whole.
POINTS_PATH = "world/points"
POINTS_UNUSABLE_PATH = "world/points_unusable"
CAMERAS_PATH = "world/cameras"
CAMERA_CENTERS_PATH = "world/cameras/centers"
STATUS_PATH = "run/status"
INFO_PATH = "run/info"
WARNINGS_PATH = "run/warnings"
METRICS_PATH = "run/metrics"

#: Mark of an unusable result (FR-033): statuses `interrupted` and `invalid_result`.
UNUSABLE_MARK = "result unusable"

#: Panel text when there are no warnings in the status.
NO_WARNINGS_TEXT = "there are no warnings in `status.json`"

_UNAVAILABLE_PREFIX = "unavailable"


def build_recording(
    run_dir: Path | str,
    *,
    save_to: Path | None = None,
    spawn: bool = True,
) -> dict:
    """Assemble a Rerun recording from a run directory and show or save it.

    Args:
        run_dir: run directory like ``runs/<run_id>``.
        save_to: ``.rrd`` path; if set, the recording is written to a file.
        spawn: open the viewer window. ``spawn=False`` together with ``save_to`` is a
            windowless mode, suitable for automated checks and for saving the view artifact.

    Returns:
        A summary of what was logged: status, status line, number of points, cameras and
        warnings, contents of the text panels and entity paths. The summary is
        self-sufficient: the marks (including "precomputed example") are visible in it,
        there is no need to parse the ``.rrd`` for checking.

    Raises:
        ResultMissingOrIncompatibleError: the ``normalized/`` directory or a required file
            in it is missing, or the artifacts' ``run_id`` does not match the manifest (code 7).
    """
    layout = runs.find_run_dir(run_dir)
    if not layout.normalized.is_dir():
        raise ResultMissingOrIncompatibleError(
            f"the run directory has no normalized/: {layout.root}",
            details="the result was not ingested or the run directory was not fully transferred",
        )

    manifest = read_json(layout.manifest)
    cameras_file = read_json(layout.normalized / "cameras.json")
    diagnostics = read_json(layout.normalized / "diagnostics.json")
    status_file = read_json(layout.normalized / "status.json")

    # FR-029: a set of files from different runs is not opened.
    ingest_checks.check_run_id_matches(
        manifest,
        ("normalized/cameras.json", cameras_file),
        ("normalized/diagnostics.json", diagnostics),
        ("normalized/status.json", status_file),
    )

    points, colors, _ = ply_io.read_ply(layout.normalized / "points.ply")

    banner = status_banner(status_file)
    usable = bool(status_file.get("viewable"))
    cloud_path = POINTS_PATH if usable else POINTS_UNUSABLE_PATH

    panels = {
        "status": _status_panel(banner, usable=usable),
        "info": _info_panel(manifest, diagnostics),
        "warnings": _warnings_panel(status_file),
        "metrics": _metrics_panel(diagnostics),
    }

    camera_records = list(cameras_file.get("cameras") or [])
    stream = rr.RecordingStream(APPLICATION_ID)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        stream.save(Path(save_to), default_blueprint=_blueprint(cloud_path))
    if spawn:
        stream.spawn(default_blueprint=_blueprint(cloud_path))

    _log_all(
        stream,
        cloud_path=cloud_path,
        points=points,
        colors=colors,
        cameras=camera_records,
        conventions=cameras_file.get("conventions") or {},
        panels=panels,
    )
    stream.flush()

    with_intrinsics = sum(1 for c in camera_records if c.get("intrinsics"))
    marks = [part for part in banner.split(" · ")[1:]]
    if not usable:
        marks.append(UNUSABLE_MARK)

    return {
        "run_id": manifest.get("run_id"),
        "status": status_file.get("status"),
        "banner": panels["status"],
        "precomputed_example": bool(status_file.get("precomputed_example")),
        "viewable": usable,
        "marks": marks,
        "points": int(len(points)),
        "cameras": len(camera_records),
        "cameras_with_intrinsics": with_intrinsics,
        "warnings": len(status_file.get("warnings") or []),
        "panels": panels,
        "entities": {
            "points": cloud_path,
            "cameras": CAMERAS_PATH,
            "status": STATUS_PATH,
            "info": INFO_PATH,
            "warnings": WARNINGS_PATH,
            "metrics": METRICS_PATH,
        },
        "saved_to": str(save_to) if save_to is not None else None,
        "spawned": spawn,
    }


# --- Logging to Rerun -------------------------------------------------------------------


def _log_all(
    stream: rr.RecordingStream,
    *,
    cloud_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    cameras: list[dict],
    conventions: dict,
    panels: dict[str, str],
) -> None:
    """Lay out the artifacts that were read into recording entities.

    Everything is logged as `static`: the view has no time axis, every entity is visible
    at any cursor position, including the permanent status line (FR-028).
    """
    # World axes are declared explicitly: the camera convention is opencv
    # (+X right, +Y down, +Z forward).
    stream.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    stream.log(
        cloud_path,
        rr.Points3D(positions=points, colors=colors),
        static=True,
    )

    _log_cameras(stream, cameras, conventions=conventions)

    stream.log(STATUS_PATH, _text(panels["status"]), static=True)
    stream.log(INFO_PATH, _text(panels["info"]), static=True)
    stream.log(WARNINGS_PATH, _text(panels["warnings"]), static=True)
    stream.log(METRICS_PATH, _text(panels["metrics"]), static=True)


def _log_cameras(
    stream: rr.RecordingStream,
    cameras: list[dict],
    *,
    conventions: dict,
) -> None:
    """Camera poses as a separate path branch (FR-025).

    The file stores `world_to_camera` transforms (data-model §3.4), while Rerun expects the
    camera→world transform: for an entity with a `Transform3D`, children are given in the
    camera coordinate system. The conversion is taken from :mod:`v3d.geometry`; there is no
    matrix arithmetic of our own here.

    A camera without `intrinsics` is logged with position and orientation only: there is
    nothing to draw a view frustum from, and a focal length must not be invented.
    """
    direction = conventions.get("transform_direction")
    if cameras and direction not in (None, "world_to_camera"):
        raise ResultMissingOrIncompatibleError(
            f"cameras.json declares transform_direction={direction!r}",
            details="the viewer only supports world_to_camera (data-model §3.4)",
        )

    centers: list[list[float]] = []
    labels: list[str] = []
    for camera in cameras:
        R_w2c = camera["R_world_to_camera"]
        t_w2c = camera["t_world_to_camera"]
        center = geometry.camera_center_from_world_to_camera(R_w2c, t_w2c)
        # The transform is an involution: applied to a world_to_camera pair, it gives a
        # camera_to_world pair. Only the rotation is taken from it — the centre is computed above.
        R_c2w, _ = geometry.world_to_camera_from_camera_to_world(R_w2c, t_w2c)

        frame = str(camera.get("frame_id") or f"camera_{len(centers):04d}")
        path = f"{CAMERAS_PATH}/{frame}"
        stream.log(
            path,
            rr.Transform3D(translation=center, mat3x3=R_c2w),
            static=True,
        )

        intrinsics = camera.get("intrinsics")
        size = camera.get("image_size")
        if intrinsics and size:
            stream.log(
                f"{path}/image",
                rr.Pinhole(
                    resolution=[int(size[0]), int(size[1])],
                    focal_length=[float(intrinsics["fx"]), float(intrinsics["fy"])],
                    principal_point=[float(intrinsics["cx"]), float(intrinsics["cy"])],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
                static=True,
            )

        centers.append([float(v) for v in center])
        labels.append(frame)

    if centers:
        # Camera positions as a single entity: they are visible even when there are no
        # intrinsics and there is nothing to draw a view frustum from.
        stream.log(
            CAMERA_CENTERS_PATH,
            rr.Points3D(positions=centers, labels=labels, show_labels=False),
            static=True,
        )


def _text(markup: str) -> rr.TextDocument:
    return rr.TextDocument(markup, media_type=rr.MediaType.MARKDOWN)


def _blueprint(cloud_path: str) -> rrb.Blueprint:
    """Window layout: the cloud with cameras on the left, text panels on the right.

    Cameras remain part of the 3D view but live in a separate path branch, so
    their layer is hidden and shown with a single toggle (FR-025).
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="cloud and cameras"),
            rrb.Vertical(
                rrb.TextDocumentView(origin=STATUS_PATH, name="status"),
                rrb.TextDocumentView(origin=METRICS_PATH, name="measures"),
                rrb.TextDocumentView(origin=WARNINGS_PATH, name="warnings"),
                rrb.TextDocumentView(origin=INFO_PATH, name="about the run"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


# --- Text panels ------------------------------------------------------------------------


def _status_panel(banner: str, *, usable: bool) -> str:
    """Permanent status line (FR-028); the marks from the banner are not reworded."""
    if usable:
        return banner
    return f"{banner} · {UNUSABLE_MARK}"


def _info_panel(manifest: dict, diagnostics: dict) -> str:
    """Run information: input, frames, selection parameters, environment, timings (FR-026)."""
    video = manifest.get("input_video") or {}
    selection = manifest.get("selection") or {}
    environment = manifest.get("environment") or {}
    frames = diagnostics.get("frames") or {}

    lines = [f"# run {manifest.get('run_id')}", ""]

    lines += ["## input"]
    lines += _pair_lines(
        [
            ("file", video.get("filename")),
            ("sha256", video.get("sha256")),
            ("duration, s", video.get("duration_s")),
            ("resolution", _resolution(video)),
            ("container", video.get("container")),
            ("codec", video.get("video_codec")),
            ("rotation, °", video.get("rotation_deg")),
        ]
    )

    lines += ["", "## frames"]
    lines += _pair_lines(
        [
            ("considered", frames.get("frames_total", selection.get("candidates_total"))),
            ("selected", frames.get("frames_selected", selection.get("selected"))),
        ]
    )

    lines += ["", "## selection parameters"]
    params = selection.get("params") or {}
    if params:
        lines += _pair_lines(list(params.items()))
    else:
        lines += _pair_lines([(key, selection.get(key)) for key in ("strategy", "budget", "seed")])
    lines += _pair_lines([("seed", manifest.get("seed"))])

    lines += ["", "## environment"]
    lines += _pair_lines(list(environment.items()))
    code = manifest.get("code_id") or {}
    if code:
        lines += _pair_lines(list(code.items()))

    lines += ["", "## stage timings, s"]
    durations = diagnostics.get("stage_durations_s") or {}
    if durations:
        lines += _pair_lines(list(durations.items()))
    else:
        lines += [f"- {_UNAVAILABLE_PREFIX}: diagnostics has no stage durations"]

    return "\n".join(lines)


def _warnings_panel(status_file: dict) -> str:
    """Run warnings verbatim: each is an observation, not an established cause."""
    warnings = status_file.get("warnings") or []
    if not warnings:
        return f"# warnings\n\n{NO_WARNINGS_TEXT}"

    lines = [f"# warnings ({len(warnings)})", ""]
    for warning in warnings:
        code = warning.get("code", "?")
        message = warning.get("message", "")
        lines.append(f"- **{code}** — {message}")
        if warning.get("observation_only"):
            lines.append("  (observation, no cause is established)")
    note = status_file.get("note")
    if note:
        lines += ["", f"run note: {note}"]
    return "\n".join(lines)


def _metrics_panel(diagnostics: dict) -> str:
    """Measures from diagnostics. An unavailable one — with a reason, never zero (FR-036)."""
    frames = diagnostics.get("frames") or {}
    cameras = diagnostics.get("cameras") or {}
    points = diagnostics.get("points") or {}

    lines = ["# measures", ""]
    lines += _pair_lines(
        [
            ("frames considered", frames.get("frames_total")),
            ("frames selected", frames.get("frames_selected")),
            ("cameras registered", cameras.get("cameras_registered")),
            ("registration share", cameras.get("registration_ratio")),
            ("points in the model result", points.get("num_points_raw")),
            ("valid points", points.get("num_points_valid")),
            ("points have colour", points.get("has_colors")),
            ("background in the cloud", points.get("background_present")),
            ("cloud extents", points.get("bbox")),
            ("reprojection error", diagnostics.get("reprojection_error")),
        ]
    )
    limits = diagnostics.get("interpretation_limits")
    if limits:
        lines += ["", "## limits of conclusions", limits]
    return "\n".join(lines)


# --- Formatting -------------------------------------------------------------------------


def _resolution(video: dict) -> str | None:
    """Input resolution ``width×height``; ``None`` when the data is incomplete.

    ``None`` here means "the field is not in the artifacts" and will be shown as an
    unavailability marker: zeros must not be substituted for an unknown size (FR-036).
    """
    width = video.get("width")
    height = video.get("height")
    if width is None or height is None:
        return None
    return f"{width}×{height}"


def _pair_lines(pairs: list[tuple[str, Any]]) -> list[str]:
    return [f"- {name}: {_format_value(value)}" for name, value in pairs]


def _format_value(value: Any) -> str:
    """Human-readable value. An unavailable measure — with a reason, not zero.

    A field missing from the file is also shown as an unavailability marker: substituting
    zero or an empty string for it would pass a gap off as a measurement (FR-036).
    """
    if is_measure(value):
        return _format_measure(value)
    if value is None:
        return f"{_UNAVAILABLE_PREFIX}: the field is not in the run artifacts"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        # Nested lists are bracketed: extents are two points, not six numbers.
        parts = [
            f"[{_format_value(v)}]" if isinstance(v, (list, tuple)) else _format_value(v)
            for v in value
        ]
        return ", ".join(parts)
    if isinstance(value, dict):
        return "; ".join(f"{k}={_format_value(v)}" for k, v in value.items())
    return str(value)


def _format_measure(measure: dict) -> str:
    """A measure from the artifacts: value with units or "unavailable: reason"."""
    if measure.get("availability") != "available":
        reason = measure.get("reason") or "no reason given in the artifact"
        return f"{_UNAVAILABLE_PREFIX}: {reason}"
    value = _format_value(measure.get("value"))
    units = measure.get("units")
    return f"{value} {units}" if units else value
