"""Run artifact schemas: writing, reading and consistency checks.

Data model: specs/001-video-3d-mvp/data-model.md

Rules deliberately built in here:

* An unavailable measure is written as ``{"value": null, "availability": "unavailable",
  "reason": ...}``. Zero in place of unavailability is forbidden (FR-036).
* ``cameras.json`` always declares conventions explicitly: artifact type, transform
  direction, axes, units, scale status (FR-021).
* Every artifact carries ``run_id``; a mismatch is a refusal (FR-029).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from v3d.errors import ResultMissingOrIncompatibleError

SCHEMA_VERSION = "1"

# --- Measures: available and unavailable -----------------------------------------------


def available(value: Any, *, units: str | None = None, meaning: str | None = None) -> dict:
    """An available measure."""
    out: dict[str, Any] = {"value": value, "availability": "available"}
    if units is not None:
        out["units"] = units
    if meaning is not None:
        out["meaning"] = meaning
    return out


def unavailable(reason: str, *, meaning: str | None = None) -> dict:
    """An unavailable or inapplicable measure: only with a reason, never as zero."""
    if not reason:
        raise ValueError("an unavailable measure must carry a reason")
    out: dict[str, Any] = {"value": None, "availability": "unavailable", "reason": reason}
    if meaning is not None:
        out["meaning"] = meaning
    return out


def is_measure(obj: Any) -> bool:
    return isinstance(obj, dict) and "availability" in obj and "value" in obj


# --- Statuses --------------------------------------------------------------------------


class RunStatus(StrEnum):
    PREPARED = "prepared"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    INVALID_RESULT = "invalid_result"


VIEWABLE_STATUSES = {RunStatus.SUCCESS, RunStatus.PARTIAL}
EXPORTABLE_STATUSES = {RunStatus.SUCCESS, RunStatus.PARTIAL}


# --- Result conventions -----------------------------------------------------------------


class ScaleStatus(StrEnum):
    MEASURED = "measured"
    MODEL_ESTIMATED = "model_estimated"
    NOT_DETERMINED = "not_determined"


ARTIFACT_TYPE_POINT_CLOUD = "point_cloud"

# The list of visual-check defects was fixed by the specification before implementation (SC-013).
VISUAL_CHECK_ITEMS = (
    "duplicated_geometry",
    "camera_trajectory_break",
    "point_soup_no_shape",
    "holes_on_low_texture",
    "background_dominates",
    "color_inconsistency",
)


def default_conventions() -> dict:
    """v1 conventions. Values are not implied by default — they are written to the file."""
    return {
        "artifact_type": ARTIFACT_TYPE_POINT_CLOUD,
        "transform_direction": "world_to_camera",
        "formula": "x_cam = R * x_world + t",
        "camera_axes": "opencv: +X right, +Y down, +Z forward (looking direction)",
        "world_axes": "right-handed; world frame defined by the reconstructor",
        "units": "unknown",
        "scale_status": ScaleStatus.NOT_DETERMINED.value,
        "scale_reference": None,
    }


INTERPRETATION_LIMITS = (
    "The number of points, model confidence and reprojection error are not proof of "
    "geometric accuracy or surface completeness. There is no independent reference, so "
    "metric accuracy and completeness are not claimed."
)


# --- Artifact records -------------------------------------------------------------------


@dataclass
class FrameRecord:
    """Record for every frame considered (data-model.md §3.2)."""

    frame_id: str
    src_index: int
    timestamp_us: int
    # None means "metric was not computed"; zero is forbidden — it would read as a
    # measured zero sharpness (FR-036).
    sharpness: float | None
    diff_to_prev_selected: float | None
    selected: bool
    reject_reason: str | None = None
    image_file: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraRecord:
    """Pose of one registered frame (data-model.md §3.4)."""

    frame_id: str
    image_size: tuple[int, int]
    intrinsics: dict | None
    R_world_to_camera: list[list[float]]
    t_world_to_camera: list[float]
    camera_center_world: list[float]
    source: str = "colmap_sparse"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        return d


REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "run_id",
    "created_utc",
    "input_video",
    "image_transform",
    "selection",
    "seed",
    "code_id",
    "environment",
    "commands",
)


# --- I/O --------------------------------------------------------------------------------


def write_json(path: Path, payload: dict) -> None:
    """Write an artifact as strict JSON.

    ``allow_nan=False`` is deliberate: ``NaN`` and ``Infinity`` are not part of JSON and make
    the file unreadable to third-party tools, while the constitution requires open formats.
    A missing value is written as ``null`` or as an unavailability marker
    (see :func:`unavailable`), but never as ``NaN`` and never as zero.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise ResultMissingOrIncompatibleError(
            f"required file is missing: {path}",
            details="the run directory was not transferred in full or the stage was not executed",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Manifest:
    """Information sufficient to repeat a run (FR-034)."""

    run_id: str
    created_utc: str
    input_video: dict
    image_transform: dict
    selection: dict
    seed: int
    code_id: dict
    environment: dict
    commands: list = field(default_factory=list)
    reconstructor: dict | None = None
    forced_overwrite: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        missing = [k for k in REQUIRED_MANIFEST_FIELDS if k not in d]
        if missing:
            raise ResultMissingOrIncompatibleError(
                "manifest is incomplete: required fields are missing: " + ", ".join(missing)
            )
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def require_same_run_id(expected: str, *payloads: tuple[str, dict]) -> None:
    """Refuse if artifacts belong to different runs (FR-029).

    Mixing files from different runs is forbidden: silently continuing is not allowed.
    """
    for name, payload in payloads:
        actual = payload.get("run_id")
        if actual is None:
            raise ResultMissingOrIncompatibleError(f"file {name} has no run_id field")
        if actual != expected:
            raise ResultMissingOrIncompatibleError(
                f"file {name} belongs to a different run",
                details=f"expected run_id={expected}, found {actual}",
            )


def new_status(
    run_id: str,
    status: RunStatus,
    *,
    warnings: list[dict] | None = None,
    precomputed_example: bool = False,
    note: str | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status.value,
        "precomputed_example": precomputed_example,
        "exportable": status in EXPORTABLE_STATUSES,
        "viewable": status in VIEWABLE_STATUSES,
        "warnings": warnings or [],
        "note": note,
    }


def new_diagnostics(run_id: str) -> dict:
    """Diagnostics skeleton: every measure must be either available or marked."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "frames": {},
        "cameras": {},
        "points": {},
        "stage_durations_s": {},
        "artifact_sizes_bytes": {},
        "reprojection_error": unavailable(
            "the reconstruction stage has not been executed yet",
            meaning="mean reprojection error over the model; does not by itself prove accuracy",
        ),
        "visual_check": {
            item: {"result": "not_checked", "note": None} for item in VISUAL_CHECK_ITEMS
        },
        "interpretation_limits": INTERPRETATION_LIMITS,
    }
