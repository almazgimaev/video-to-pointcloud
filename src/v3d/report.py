"""Human-readable run report `report.md` (T035).

The report is readable **without launching the viewer** (FR-035) and is assembled only from
already written artifacts: `manifest.json`, `frames.json` and the contents of `normalized/`.
Nothing is recomputed, no GPU is needed.

Rules deliberately built in here:

* **An unavailable measure is printed as "unavailable: <reason>", never as zero**
  (FR-036). Zero would read as a measured value.
* An available measure is printed together with an explanation of its meaning, if one is
  recorded in the artifact (FR-037). The module does not invent explanations of its own.
* Two separate sections: verified properties of the file and frames, and assumptions about
  the scene accepted on trust (FR-006). The second section is never called a verification.
* There are no quality judgements of the result ("good reconstruction", "high accuracy") in
  the text: the report states facts and observations (Principles I and V).
* The reason for a failure is not named if it was not measured (FR-007): wording is taken
  from ready-made builders (`v3d.status.status_banner`, warning texts) rather than
  composed here.
* The limits of conclusions are always given (FR-038, FR-039): without an independent
  reference, metric accuracy and surface completeness are not claimed.

The report must be built under statuses `partial`, `interrupted`, `invalid_result`, and
when some artifacts are missing: the missing section is marked unavailable with a reason,
not silently omitted and not replaced with zeros.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v3d import runs
from v3d.artifacts import (
    INTERPRETATION_LIMITS,
    VISUAL_CHECK_ITEMS,
    ScaleStatus,
    default_conventions,
    is_measure,
    read_json,
    require_same_run_id,
)
from v3d.config import preliminary_note
from v3d.errors import ResultMissingOrIncompatibleError
from v3d.status import PRECOMPUTED_EXAMPLE_MARK, status_banner

#: The artifact type in words. Other representations of the result are not named at all (SC-009).
ARTIFACT_TYPE_NOTE = (
    "The result is a **point cloud**: a set of individual coloured points. No surface is "
    "built from them, invisible areas are not filled in, and this version produces no other "
    "representation of the result."
)

#: Mark of the residual background: v1 does not remove it and declares its presence (FR-020).
BACKGROUND_NOTE = (
    "The background is not removed from the cloud: separating the object from the background "
    "is not performed in this version, so background points are present in the result."
)

#: Assumptions about the scene accepted from the user and not verified by the system (FR-006).
UNCHECKED_ASSUMPTIONS: tuple[str, ...] = (
    "the object is stationary during shooting",
    "the object is rigid: its shape did not change during shooting",
    "the object is opaque and not mirror-like",
    "the object's surface is textured enough for details on it to be distinguishable",
    "the scene lighting during shooting is stable",
    "neighbouring views overlap, the walk around the object is complete enough",
)

#: Explanation for the unverified-assumptions section: this is not a measurement.
UNCHECKED_ASSUMPTIONS_NOTE = (
    "The system **does not measure or verify** the items listed below. These are conditions "
    "of the scenario accepted on trust from the user. If any of them did not hold, the report "
    "will not notice it and will not report it."
)

#: Explanation for the verified section: only actually measured properties are listed.
CHECKED_NOTE = (
    "Below is only what the system actually read from the file and computed from the frames."
)

#: Limits of conclusions beyond the `INTERPRETATION_LIMITS` text (FR-038, FR-039).
NO_REFERENCE_NOTE = (
    "There is no independent reference and no described comparison protocol in this run, so "
    "metric accuracy and surface completeness are not claimed."
)

#: Scale status in words (FR-021).
SCALE_STATUS_WORDS: dict[str, str] = {
    ScaleStatus.MEASURED.value: "established by measurement",
    ScaleStatus.MODEL_ESTIMATED.value: "estimated by the model",
    ScaleStatus.NOT_DETERMINED.value: "not determined",
}

#: Names of the six visual-check items (spec.md, "Observable measures").
VISUAL_CHECK_LABELS: dict[str, str] = {
    "duplicated_geometry": "splitting or duplication of the object's geometry",
    "camera_trajectory_break": "break or jump in the camera pose trajectory",
    "point_soup_no_shape": '"point soup": no recognizable shape of the object is visible',
    "holes_on_low_texture": "gaps on weakly textured areas of the object",
    "background_dominates": "background dominating over the object",
    "color_inconsistency": "gross colour inconsistency between views",
}

#: States of a visual-check item in words.
VISUAL_CHECK_RESULTS: dict[str, str] = {
    "not_checked": "not performed",
    "absent": "not detected",
    "present": "detected",
    "unclear": "not determined unambiguously",
}

#: Labels of the input video fields.
INPUT_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("filename", "file name"),
    ("sha256", "sha256"),
    ("size_bytes", "size, bytes"),
    ("container", "container"),
    ("video_codec", "video track codec"),
    ("duration_s", "duration, s"),
    ("width", "width, px"),
    ("height", "height, px"),
    ("avg_fps", "average fps"),
    ("nb_frames", "frames per container data"),
    ("rotation_deg", "rotation from metadata, °"),
)

_MISSING = "there is no information about this in the run artifacts"


# --- Reading artifacts ------------------------------------------------------------------


def _read_optional(path: Path, name: str) -> tuple[dict | None, str | None]:
    """Read an optional artifact.

    Returns ``(payload, None)`` or ``(None, reason for unavailability)``. A missing file
    is not an error: the corresponding report section will be marked unavailable
    with a reason, but the report will be built (otherwise a failed run would stay undescribed).
    """
    if not path.is_file():
        return None, f"file {name} is not in the run directory"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, f"file {name} cannot be read as JSON"


# --- Value formatting -------------------------------------------------------------------


def _format_value(value: Any, units: str | None = None) -> str:
    """A measure's value on one line. A missing value is not replaced with zero."""
    if value is None:
        return "value not recorded"
    if isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, float):
        text = f"{value:.6g}"
    elif isinstance(value, list | tuple | dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return f"{text} {units}" if units else text


def _measure_lines(label: str, measure: Any) -> list[str]:
    """Measure lines: the value or an unavailability marker with a reason (FR-036, FR-037).

    An unavailable measure is printed as "unavailable: <reason>". Zero or any other
    invented value in its place is forbidden: the reader must be able to tell "measured zero"
    from "could not be measured".
    """
    if not is_measure(measure):
        return [f"- **{label}**: {_format_value(measure)}"]

    if measure.get("availability") == "available":
        lines = [f"- **{label}**: {_format_value(measure.get('value'), measure.get('units'))}"]
    else:
        reason = measure.get("reason") or "no reason for unavailability is recorded in the artifact"
        lines = [f"- **{label}**: unavailable: {reason}"]

    meaning = measure.get("meaning")
    if meaning:
        lines.append(f"  - meaning of the measure: {meaning}")
    note = measure.get("note")
    if note:
        lines.append(f"  - note: {note}")
    return lines


def _unavailable_section(reason: str) -> list[str]:
    """Body of a section for which there is no source data."""
    return [
        f"Section unavailable: {reason}.",
        "",
        "Values of this section are not filled in with zeros and are not reconstructed by guess.",
    ]


def _table(header: tuple[str, str], rows: list[tuple[str, str]]) -> list[str]:
    """A two-column table; an empty row list gives a note instead of a table."""
    if not rows:
        return ["_no records_"]
    lines = [f"| {header[0]} | {header[1]} |", "| --- | --- |"]
    lines += [f"| {left} | {right} |" for left, right in rows]
    return lines


# --- Report sections --------------------------------------------------------------------


def _header_lines(
    run_id: str, manifest: dict, status_payload: dict | None, reason: str | None
) -> list[str]:
    """Header: run, date, status. The status line is taken from `status_banner`."""
    lines = [
        f"# Run report `{run_id}`",
        "",
        f"- **Run**: `{run_id}`",
        f"- **Created (UTC)**: {manifest.get('created_utc', _MISSING)}",
    ]
    if status_payload is None:
        lines.append(f"- **Status**: unavailable: {reason}")
        return lines

    lines.append(f"- **Status**: {status_banner(status_payload)}")
    if status_payload.get("precomputed_example"):
        lines += [
            "",
            f"> **{PRECOMPUTED_EXAMPLE_MARK.upper()}.** The result shown was not produced by "
            "this run and has no relation to the current input.",
        ]
    note = status_payload.get("note")
    if note:
        lines.append(f"- **Status note**: {note}")
    return lines


def _conventions_lines(cameras_payload: dict | None, diag: dict | None) -> list[str]:
    """Result type, coordinate system, units, scale status, background (FR-021, FR-020)."""
    conventions = (cameras_payload or {}).get("conventions") or default_conventions()
    scale_status = conventions.get("scale_status", ScaleStatus.NOT_DETERMINED.value)
    in_words = SCALE_STATUS_WORDS.get(scale_status, scale_status)

    lines = [
        ARTIFACT_TYPE_NOTE,
        "",
        f"- **Artifact type**: {conventions.get('artifact_type', _MISSING)} (point cloud)",
        f"- **Transform direction**: {conventions.get('transform_direction', _MISSING)}",
        f"- **Formula**: `{conventions.get('formula', _MISSING)}`",
        f"- **Camera axes**: {conventions.get('camera_axes', _MISSING)}",
        f"- **World axes**: {conventions.get('world_axes', _MISSING)}",
        f"- **Units**: {conventions.get('units', _MISSING)} (unknown)",
        f"- **Scale status**: {in_words}",
    ]
    reference = conventions.get("scale_reference")
    lines.append(f"- **Scale reference**: {reference if reference else 'not used'}")

    background = ((diag or {}).get("points") or {}).get("background_present")
    if background is None:
        lines.append(f"- **Residual background**: {BACKGROUND_NOTE}")
    else:
        mark = "present" if background else "marked as absent in the artifact"
        lines.append(f"- **Residual background**: {mark}. {BACKGROUND_NOTE}")
    return lines


def _input_lines(manifest: dict) -> list[str]:
    """Properties of the input video from the manifest."""
    video = manifest.get("input_video") or {}
    if not video:
        return _unavailable_section("the manifest has no information about the input video")
    rows = [(label, _format_value(video.get(key))) for key, label in INPUT_FIELD_LABELS]
    return _table(("Property", "Value"), rows)


def _checked_lines(manifest: dict, frames_payload: dict | None, diag: dict | None) -> list[str]:
    """Section "verified by the system": only what was measured (FR-006, part "a")."""
    video = manifest.get("input_video") or {}
    transform = manifest.get("image_transform") or {}
    lines = [CHECKED_NOTE, ""]

    supported = video.get("supported")
    lines += [
        f"- file read, video track found: {_format_value(bool(video))}",
        f"- container and codec are in the supported list: {_format_value(supported)}",
        f"- duration: {_format_value(video.get('duration_s'), 's')}",
        f"- resolution: {_format_value(video.get('width'))}×{_format_value(video.get('height'))}",
        f"- average fps: {_format_value(video.get('avg_fps'))}",
        f"- frames per container data: {_format_value(video.get('nb_frames'))}",
        f"- rotation from metadata applied: "
        f"{_format_value(transform.get('rotation_applied_deg'), '°')}",
    ]

    frame_count = None if frames_payload is None else len(frames_payload.get("frames") or [])
    if frame_count is not None:
        with_sharpness = sum(
            1 for f in frames_payload.get("frames") or [] if f.get("sharpness") is not None
        )
        with_difference = sum(
            1
            for f in frames_payload.get("frames") or []
            if f.get("diff_to_prev_selected") is not None
        )
        lines += [
            f"- sharpness computed for frames: {with_sharpness} of {frame_count} considered",
            f"- difference from the previous selected frame computed for: {with_difference} frames",
        ]
    else:
        lines.append("- frame sharpness and difference: unavailable: file frames.json is missing")

    if diag is not None:
        lines.append(
            "- the number of points and camera poses was read from the reconstructor result "
            '(see the section "Reconstruction")'
        )
    lines += [
        "",
        "Sharpness and the difference between neighbouring frames are **indirect observations**: "
        "the frame difference is a proxy for viewpoint change, not a measurement of camera "
        "displacement.",
    ]
    return lines


def _assumptions_lines() -> list[str]:
    """Section "accepted from the user, not verified" (FR-006, part "b")."""
    return [
        UNCHECKED_ASSUMPTIONS_NOTE,
        "",
        *[f"- {text} — **not verified**" for text in UNCHECKED_ASSUMPTIONS],
    ]


def _frames_lines(
    manifest: dict, frames_payload: dict | None, reason: str | None, diag: dict | None
) -> list[str]:
    """Frames: how many considered and selected, rejection reasons, selection method and seed."""
    selection = manifest.get("selection") or {}
    params = selection.get("params") or {}

    diag_frames = (diag or {}).get("frames") or {}
    considered = diag_frames.get("frames_total", selection.get("candidates_considered"))
    selected = diag_frames.get("frames_selected", selection.get("selected"))
    reasons = _reject_reason_counts(diag_frames, selection, frames_payload)

    lines = [
        f"- **Candidates considered**: {_format_value(considered)}",
        f"- **Frames selected**: {_format_value(selected)}",
        f"- **Selection method**: {_format_value(selection.get('selector'))}",
        f"- **Seed**: {_format_value(manifest.get('seed'))}",
    ]
    if params:
        lines.append("- **Selection parameters**:")
        lines += [f"  - {key}: {_format_value(value)}" for key, value in params.items()]
    if frames_payload is None and reason is not None:
        lines.append(f"- **Frame list**: unavailable: {reason}")

    lines += ["", "**Rejection reasons**", ""]
    lines += _table(
        ("Reason", "Frames"),
        [(str(name), str(count)) for name, count in sorted(reasons.items())],
    )
    return lines


def _reject_reason_counts(diag_frames: dict, selection: dict, frames_payload: dict | None) -> dict:
    """Distribution of rejection reasons from the first source that contains it."""
    for source in (
        diag_frames.get("reject_reason_counts"),
        selection.get("reject_reasons"),
    ):
        if source:
            return dict(source)
    counts: dict[str, int] = {}
    for frame in (frames_payload or {}).get("frames") or []:
        reason = frame.get("reject_reason")
        if reason and not frame.get("selected"):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _reconstruction_lines(diag: dict | None, reason: str | None) -> list[str]:
    """Reconstruction: cameras, points, valid-point definition, applied filters."""
    if diag is None:
        return _unavailable_section(reason or _MISSING)

    cameras = diag.get("cameras") or {}
    points = diag.get("points") or {}
    lines = [
        f"- **Cameras registered**: {_format_value(cameras.get('cameras_registered'))}",
    ]
    lines += _measure_lines("Share of registered frames", cameras.get("registration_ratio"))
    lines += [
        f"- **Points in the model (raw)**: {_format_value(points.get('num_points_raw'))}",
        f"- **Valid points**: {_format_value(points.get('num_points_valid'))}",
        f"- **Definition of a valid point**: {points.get('valid_point_definition', _MISSING)}",
        f"- **Points have colour**: {_format_value(points.get('has_colors'))}",
    ]
    filters = points.get("filters_applied") or []
    if filters:
        lines.append("- **Applied filters**:")
        for entry in filters:
            name = entry.get("name", "unnamed")
            rest = {k: v for k, v in entry.items() if k != "name"}
            lines.append(f"  - `{name}`: {json.dumps(rest, ensure_ascii=False)}")
    else:
        lines.append("- **Applied filters**: no records")
    return lines


def _metrics_lines(diag: dict | None, reason: str | None) -> list[str]:
    """Measures with an explanation of meaning; unavailable ones — with a reason, never zero."""
    if diag is None:
        return _unavailable_section(reason or _MISSING)

    # The share of registered frames is output in the "Reconstruction" section together
    # with its explanation: there is no need to repeat it here.
    lines: list[str] = []
    lines += _measure_lines("Reprojection error", diag.get("reprojection_error"))
    points = diag.get("points") or {}
    if "bbox" in points:
        lines += _measure_lines("Cloud bounding box", points.get("bbox"))
    durations = diag.get("stage_durations_s") or {}
    lines += ["", "**Stage times, s**", ""]
    lines += _table(
        ("Stage", "Seconds"),
        [(stage, _format_value(value)) for stage, value in durations.items()],
    )
    if durations:
        total = sum(float(v) for v in durations.values())
        lines += ["", f"Total time of the recorded stages: {_format_value(total)} s."]

    sizes = diag.get("artifact_sizes_bytes") or {}
    lines += ["", "**Artifact sizes, bytes**", ""]
    lines += _table(
        ("Artifact", "Bytes"),
        [(name, _format_value(value)) for name, value in sizes.items()],
    )
    return lines


def _visual_check_lines(diag: dict | None, reason: str | None) -> list[str]:
    """Six visual-check items: an observation separate from quantitative conclusions."""
    check = (diag or {}).get("visual_check") or {}
    lines = [
        "The visual check is an **observation**, separate from the quantitative measures.",
        "",
    ]
    if not check:
        lines.append(f"Item states are unavailable: {reason or _MISSING}.")
        lines.append("")
    rows: list[tuple[str, str]] = []
    for item in VISUAL_CHECK_ITEMS:
        entry = check.get(item) or {}
        raw = entry.get("result", "not_checked")
        state = VISUAL_CHECK_RESULTS.get(raw, raw)
        remark = entry.get("note")
        if remark:
            state = f"{state} — {remark}"
        rows.append((VISUAL_CHECK_LABELS[item], state))
    lines += _table(("Check item", "State"), rows)
    return lines


def _warnings_lines(status_payload: dict | None, reason: str | None) -> list[str]:
    """Run warnings as a list. Each is an observation, not a diagnosis of a cause (FR-007)."""
    if status_payload is None:
        return _unavailable_section(reason or _MISSING)
    warnings = status_payload.get("warnings") or []
    if not warnings:
        return ["No warnings recorded."]
    lines = ["Each item is an observation; the cause of what is happening is not established.", ""]
    lines += [f"- `{w.get('code', 'no code')}`: {w.get('message', _MISSING)}" for w in warnings]
    return lines


def _limits_lines() -> list[str]:
    """Limits of conclusions (FR-038, FR-039)."""
    return [INTERPRETATION_LIMITS, "", NO_REFERENCE_NOTE]


def _reproduction_lines(layout: runs.RunLayout, manifest: dict, diag: dict | None) -> list[str]:
    """Information for repeating the run (FR-034)."""
    code = manifest.get("code_id") or {}
    environment = manifest.get("environment") or {}
    lines = [
        f"- **Code version**: {_format_value(code.get('value'))} "
        f"(source: {_format_value(code.get('source'))})",
    ]
    if code.get("working_tree_dirty"):
        changed = code.get("dirty_files") or []
        lines.append(
            "- **The working tree was dirty**: the code differs from the committed one; "
            f"changed files: {len(changed)}"
        )
        lines += [f"  - `{path}`" for path in changed]
    lines += [
        f"- **Seed**: {_format_value(manifest.get('seed'))}",
        f"- **Overwritten via --force**: {_format_value(manifest.get('forced_overwrite'))}",
    ]
    if environment:
        lines.append("- **Environment**:")
        lines += [f"  - {key}: {_format_value(value)}" for key, value in environment.items()]
    reconstructor = manifest.get("reconstructor")
    lines.append(
        f"- **Reconstructor**: {_format_value(reconstructor) if reconstructor else _MISSING}"
    )
    lines.append(f"- **Portable package**: `{layout.package}`")
    lines.append(f"- **Normalized result**: `{layout.normalized}`")

    commands = manifest.get("commands") or []
    lines += ["", "**Commands and stage times**", ""]
    rows: list[tuple[str, str]] = []
    for record in commands:
        durations = record.get("durations_s") or {}
        rows.append(
            (
                str(record.get("command", _MISSING)),
                _format_value(durations) if durations else "times not recorded",
            )
        )
    if diag and diag.get("stage_durations_s"):
        rows.append(("result ingest stages", _format_value(diag["stage_durations_s"])))
    lines += _table(("Command", "Durations, s"), rows)
    return lines


def _preliminary_lines(manifest: dict) -> list[str]:
    """Marks of preliminary values: the reader must know that a threshold was not measured."""
    selection = manifest.get("selection") or {}
    notes: list[str] = [text for text in (selection.get("preliminary") or []) if text]
    for name in selection.get("params") or {}:
        text = preliminary_note(name)
        if text and text not in notes:
            notes.append(text)
    if not notes:
        return ["No preliminary values are marked in the manifest."]
    return [
        "The values below were chosen **before measurement** and are subject to revision:",
        "",
        *[f"- {text}" for text in notes],
    ]


# --- Assembly ---------------------------------------------------------------------------


def _section(title: str, body: list[str]) -> list[str]:
    return [f"## {title}", "", *body, ""]


def build_report(run_dir: Path, *, out: Path | None = None) -> Path:
    """Build `report.md` from the run artifacts and return the path to it.

    :param run_dir: run directory ``runs/<run_id>``.
    :param out: path to the report file; by default ``<run_dir>/report.md``.

    Only `manifest.json`, `frames.json` and `normalized/` are read; the result is not
    recomputed, the viewer is not launched (FR-035, FR-027).

    Missing artifacts are not an error: the corresponding section is marked
    unavailable with a reason. The report is built under any status, including `partial`,
    `interrupted` and `invalid_result` — a failed run must stay readable.

    Raises:
        ResultMissingOrIncompatibleError: the run directory is missing, has no manifest,
            or the artifacts belong to different runs (FR-029).
    """
    layout = runs.find_run_dir(run_dir)
    manifest = read_json(layout.manifest)
    run_id = manifest.get("run_id")
    if not run_id:
        raise ResultMissingOrIncompatibleError(
            f"the run manifest has no run_id: {layout.manifest}",
            details="the report cannot be tied to a run",
        )

    frames_payload, frames_reason = _read_optional(layout.frames, "frames.json")
    status_payload, status_reason = _read_optional(
        layout.normalized / "status.json", "normalized/status.json"
    )
    diag, diag_reason = _read_optional(
        layout.normalized / "diagnostics.json", "normalized/diagnostics.json"
    )
    cameras_payload, _ = _read_optional(
        layout.normalized / "cameras.json", "normalized/cameras.json"
    )

    to_check = [
        (name, payload)
        for name, payload in (
            ("frames.json", frames_payload),
            ("normalized/status.json", status_payload),
            ("normalized/diagnostics.json", diag),
            ("normalized/cameras.json", cameras_payload),
        )
        if payload is not None
    ]
    require_same_run_id(run_id, *to_check)

    lines: list[str] = [
        *_header_lines(run_id, manifest, status_payload, status_reason),
        "",
        *_section("Result type and conventions", _conventions_lines(cameras_payload, diag)),
        *_section("Input", _input_lines(manifest)),
        *_section("Verified by the system", _checked_lines(manifest, frames_payload, diag)),
        *_section("Accepted from the user, not verified", _assumptions_lines()),
        *_section("Frames", _frames_lines(manifest, frames_payload, frames_reason, diag)),
        *_section("Reconstruction", _reconstruction_lines(diag, diag_reason)),
        *_section("Measures", _metrics_lines(diag, diag_reason)),
        *_section("Visual check", _visual_check_lines(diag, diag_reason)),
        *_section("Warnings", _warnings_lines(status_payload, status_reason)),
        *_section("Limits of conclusions", _limits_lines()),
        *_section("Reproduction", _reproduction_lines(layout, manifest, diag)),
        *_section("Preliminary values", _preliminary_lines(manifest)),
    ]

    target = Path(out) if out is not None else layout.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target
