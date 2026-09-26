"""Building `normalized/diagnostics.json` (T030).

The set of measures was fixed by the specification **before** implementation
(spec.md, section "Observable measures and how they are assessed"; data-model.md §3.6)
and is neither extended nor reduced here.

Rules deliberately built in:

* ``background_present`` is always ``True``: v1 does not remove the background, and the
  result must declare its presence (FR-020). The field is not computed and not configurable.
* An unavailable measure is a marker object with a reason, never zero (FR-036).
  Zero in the reprojection error field would read as a perfect reconstruction.
* An available error measure is accompanied by an explanation of its meaning and limits
  of applicability (FR-037).
* ``interpretation_limits`` is always present (FR-038, FR-039).

The module computes nothing on behalf of the caller: valid points and ``filters_applied``
come from outside (the filter lives in :mod:`v3d.geometry`). The only exception is
``bbox``: the extents are computed here, from the valid points passed in, and if there are
none — from the finite points of the original cloud with an explicit note that this is a
pre-filter box.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from v3d.artifacts import available, new_diagnostics, unavailable
from v3d.geometry import is_valid_points, valid_point_definition

#: Explanation of the meaning of reprojection error and its limits of applicability (FR-037).
REPROJECTION_ERROR_MEANING = (
    "Mean reprojection error over the model in pixels: how far, on average, the observed "
    "projections of points diverge from their projections through the recovered poses. This "
    "is a measure of the model's internal consistency; by itself it does not prove geometric "
    "accuracy, scale or surface completeness."
)

#: Reason for unavailability when upstream provides no mean reprojection error.
REPROJECTION_ERROR_UNAVAILABLE_REASON = (
    "the reconstructor ran without bundle adjustment, and the model contains no mean "
    "reprojection error; zero in its place would mean a perfect reconstruction"
)

#: Explanation of the cloud extents.
BBOX_MEANING = (
    "Axis-aligned bounding box of the cloud in the reconstructor's world coordinates: "
    "[[xmin, ymin, zmin], [xmax, ymax, zmax]]. Units are unknown, scale is not determined, "
    "so the box size is not a measurement of the object."
)

#: Note for extents computed from the original cloud: the outlier filter did not narrow them.
BBOX_BEFORE_FILTER_NOTE = (
    "The box was computed from the finite points of the model's original cloud, before the "
    "outlier filter: the filtered points themselves were not passed by the caller. These are "
    "not the extents of the displayed cloud — if the filter dropped extreme points, the box "
    "is wider."
)

#: Explanation of the share of registered frames.
REGISTRATION_RATIO_MEANING = (
    "Share of selected frames that received a camera pose: cameras_registered / "
    "frames_selected. Shows the coverage of the selected set, but not the quality of poses."
)


@runtime_checkable
class SparseModelLike(Protocol):
    """Structural type of a sparse model.

    There is deliberately no hard import of :class:`v3d.colmap_io.SparseModel` here: diagnostics
    needs only four fields, and tests supply their own stub.
    """

    cameras: Any
    points_xyz: Any
    points_rgb: Any
    mean_reprojection_error: float | None


def _reject_reason_counts(frames: list[dict]) -> dict[str, int]:
    """Distribution of rejection reasons over candidate frames.

    Only records with a non-empty reason are counted. A selected frame carries no reason.
    """
    counts: dict[str, int] = {}
    for frame in frames:
        if frame.get("selected"):
            continue
        reason = frame.get("reject_reason")
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _registration_ratio(cameras_registered: int, frames_selected: int) -> dict:
    """Share of registered frames or an unavailability marker.

    With zero selected frames the share is undefined: there is nothing to divide by. An
    unavailability marker is written, not zero — zero would read as "no frame of a
    non-empty set was registered" (FR-036).
    """
    if frames_selected <= 0:
        return unavailable(
            "there are no selected frames, the share of registered ones is undefined",
            meaning=REGISTRATION_RATIO_MEANING,
        )
    return available(cameras_registered / frames_selected, meaning=REGISTRATION_RATIO_MEANING)


def _bbox_from_points(points_xyz: Any, *, note: str | None = None) -> dict:
    """Bounding box over the finite points of the passed array or an unavailability marker.

    Non-finite coordinates (NaN, Inf) are ignored: they do not describe a point position.
    For an empty array, and when no finite points remain, the extents do not exist —
    a marker is written, not zeros: a zero-size box would read as a degenerate but
    measured cloud.

    ``note`` is attached to the measure when the box was not computed from the displayed
    cloud (see :func:`build_diagnostics`, the mode without ``valid_points_xyz``).
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        measure = unavailable(
            "there are no valid points, extents are undefined", meaning=BBOX_MEANING
        )
    else:
        finite = pts[is_valid_points(pts)]
        if len(finite) == 0:
            measure = unavailable(
                "no cloud point has finite coordinates, extents are undefined",
                meaning=BBOX_MEANING,
            )
        else:
            lo = finite.min(axis=0)
            hi = finite.max(axis=0)
            measure = available(
                [[float(v) for v in lo], [float(v) for v in hi]], meaning=BBOX_MEANING
            )
    if note is not None:
        measure["note"] = note
    return measure


def _reprojection_error(value: float | None) -> dict:
    """Reprojection error: a value with an explanation or an unavailability marker (FR-036, FR-037).

    Zero is never substituted here: it would mean a perfect reconstruction.
    A non-finite value (NaN, Inf) is also treated as unavailable — it cannot be written to
    strict JSON (see :func:`v3d.artifacts.write_json`).
    """
    if value is None:
        return unavailable(
            REPROJECTION_ERROR_UNAVAILABLE_REASON, meaning=REPROJECTION_ERROR_MEANING
        )
    as_float = float(value)
    if not np.isfinite(as_float):
        return unavailable(
            "the model returned a non-finite mean reprojection error; the value is unusable",
            meaning=REPROJECTION_ERROR_MEANING,
        )
    return available(as_float, units="px", meaning=REPROJECTION_ERROR_MEANING)


def _has_colors(points_rgb: Any, num_points_raw: int) -> bool:
    """Whether points have colour: the array exists, is non-empty and matches the point count."""
    if points_rgb is None:
        return False
    cols = np.asarray(points_rgb)
    if cols.size == 0:
        return False
    return cols.ndim == 2 and cols.shape[0] == num_points_raw and cols.shape[1] >= 3


def build_diagnostics(
    run_id: str,
    *,
    frames: list[dict],
    model: SparseModelLike,
    points_valid: int | None = None,
    valid_points_xyz: Any | None = None,
    filters_applied: list[dict],
    durations_s: dict[str, float],
    artifact_sizes: dict[str, int],
) -> dict:
    """Build the run diagnostics.

    :param run_id: run identifier; it is also checked when the result is opened (FR-029).
    :param frames: records of **all** candidates considered (``FrameRecord.to_dict()``).
    :param model: sparse model — any object with ``cameras``, ``points_xyz``,
        ``points_rgb``, ``mean_reprojection_error`` fields (see :class:`SparseModelLike`).
    :param points_valid: number of valid points, computed by the caller after the filter.
    :param valid_points_xyz: the valid points themselves (N, 3) after the filter, if the caller
        has them.
    :param filters_applied: records of applied filters (from :mod:`v3d.geometry`).
    :param durations_s: stage durations, seconds.
    :param artifact_sizes: artifact sizes, bytes.

    Two modes of supplying valid points:

    * **``valid_points_xyz`` passed** — ``bbox`` is computed from it, i.e. it describes exactly
      the cloud that will be displayed and exported; ``num_points_valid`` is taken as
      ``len(valid_points_xyz)``. If ``points_valid`` is passed as well and the numbers differ —
      ``ValueError``: the caller contradicts itself, and we must not choose for it.
    * **only ``points_valid`` passed** — the points themselves are unavailable, ``bbox`` is
      computed from the finite points of the model's **original** cloud and carries a ``note``
      saying it was computed before the outlier filter. If the filter dropped extreme points,
      such a box is wider than the extents of the displayed cloud, and the report reader must
      see that rather than guess.

    A valid point count greater than the model's point count is a caller error, ``ValueError``:
    the filter cannot produce points, and silently truncating the number would hide the
    discrepancy.
    """
    diagnostics = new_diagnostics(run_id)

    points_raw = np.asarray(model.points_xyz, dtype=np.float64).reshape(-1, 3)
    num_points_raw = int(len(points_raw))

    if valid_points_xyz is None:
        if points_valid is None:
            raise ValueError("either points_valid or valid_points_xyz must be passed")
        num_points_valid = int(points_valid)
        bbox = (
            _bbox_from_points(points_raw, note=BBOX_BEFORE_FILTER_NOTE)
            if num_points_valid > 0
            else _bbox_from_points(points_raw[:0], note=BBOX_BEFORE_FILTER_NOTE)
        )
    else:
        valid_pts = np.asarray(valid_points_xyz, dtype=np.float64).reshape(-1, 3)
        num_points_valid = int(len(valid_pts))
        if points_valid is not None and int(points_valid) != num_points_valid:
            raise ValueError(
                "points_valid differs from the length of valid_points_xyz: "
                f"{int(points_valid)} != {num_points_valid}"
            )
        bbox = _bbox_from_points(valid_pts)

    if num_points_valid < 0:
        raise ValueError(f"the number of valid points cannot be negative: {num_points_valid}")
    if num_points_valid > num_points_raw:
        raise ValueError(
            "the number of valid points exceeds the number of model points: "
            f"{num_points_valid} > {num_points_raw}; the filter cannot produce points"
        )

    frames_total = len(frames)
    frames_selected = sum(1 for frame in frames if frame.get("selected"))
    cameras_registered = int(len(model.cameras))

    diagnostics["frames"] = {
        "frames_total": frames_total,
        "frames_selected": frames_selected,
        "reject_reason_counts": _reject_reason_counts(frames),
    }
    diagnostics["cameras"] = {
        "cameras_registered": cameras_registered,
        "registration_ratio": _registration_ratio(cameras_registered, frames_selected),
    }
    diagnostics["points"] = {
        "num_points_raw": num_points_raw,
        "num_points_valid": num_points_valid,
        "valid_point_definition": valid_point_definition(),
        "has_colors": _has_colors(model.points_rgb, num_points_raw),
        # v1 does not remove the background; its presence is always declared (FR-020).
        "background_present": True,
        "bbox": bbox,
        "filters_applied": list(filters_applied),
    }
    diagnostics["stage_durations_s"] = dict(durations_s)
    diagnostics["artifact_sizes_bytes"] = dict(artifact_sizes)
    diagnostics["reprojection_error"] = _reprojection_error(model.mean_reprojection_error)
    return diagnostics
