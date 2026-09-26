"""Unit tests of diagnostics (T030): set of measures, unavailability marks, limits.

Checks the behaviour fixed by the specification before implementation: the set of sections,
honest unavailability marks instead of zeros (FR-036), the explanation of the meaning of the
reprojection error (FR-037), the mandatory limits of conclusions (FR-038) and the declared
presence of the background (FR-020).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from v3d.artifacts import VISUAL_CHECK_ITEMS
from v3d.diagnostics import build_diagnostics
from v3d.geometry import valid_point_definition

RUN_ID = "20250919-120000-abcdef"


@dataclass
class FakeModel:
    """Sparse model stub: only the fields diagnostics needs (structural typing).

    There is deliberately no hard dependency on `v3d.colmap_io` here — diagnostics
    accepts any object with these four fields.
    """

    cameras: list[Any] = field(default_factory=list)
    points_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    points_rgb: np.ndarray | None = None
    mean_reprojection_error: float | None = None


def make_model(
    *,
    num_cameras: int = 3,
    points: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    mean_reprojection_error: float | None = None,
) -> FakeModel:
    """A model with the given number of cameras and a point cloud."""
    pts = (
        points
        if points is not None
        else np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 1.0]], dtype=np.float64)
    )
    cols = colors
    if cols is None and len(pts) > 0:
        cols = np.full((len(pts), 3), 128, dtype=np.uint8)
    return FakeModel(
        cameras=[f"cam_{i}" for i in range(num_cameras)],
        points_xyz=pts,
        points_rgb=cols,
        mean_reprojection_error=mean_reprojection_error,
    )


def make_frames(*, selected: int, rejected: dict[str, int] | None = None) -> list[dict]:
    """A list of frame records: `selected` chosen ones plus those rejected by reason."""
    frames: list[dict] = []
    idx = 0
    for _ in range(selected):
        frames.append(
            {
                "frame_id": f"{idx:04d}_{idx:06d}",
                "src_index": idx,
                "selected": True,
                "reject_reason": None,
            }
        )
        idx += 1
    for reason, count in (rejected or {}).items():
        for _ in range(count):
            frames.append(
                {
                    "frame_id": f"----_{idx:06d}",
                    "src_index": idx,
                    "selected": False,
                    "reject_reason": reason,
                }
            )
            idx += 1
    return frames


def build(**overrides: Any) -> dict:
    """Diagnostics with reasonable defaults; overridden selectively."""
    kwargs: dict[str, Any] = {
        "frames": make_frames(selected=3, rejected={"blurry": 2}),
        "model": make_model(),
        "points_valid": 3,
        "filters_applied": [],
        "durations_s": {"prepare": 1.5},
        "artifact_sizes": {"points.ply": 1024},
    }
    kwargs.update(overrides)
    return build_diagnostics(RUN_ID, **kwargs)


def test_top_level_keys_and_run_id() -> None:
    """The top level has exactly the sections from the spec; the run is identified by run_id."""
    diag = build()
    assert diag["run_id"] == RUN_ID
    assert set(diag) == {
        "schema_version",
        "run_id",
        "frames",
        "cameras",
        "points",
        "stage_durations_s",
        "artifact_sizes_bytes",
        "reprojection_error",
        "visual_check",
        "interpretation_limits",
    }
    assert diag["stage_durations_s"] == {"prepare": 1.5}
    assert diag["artifact_sizes_bytes"] == {"points.ply": 1024}


def test_frames_section_counts_candidates_and_selected() -> None:
    """frames_total is all candidates considered, frames_selected is the chosen ones."""
    frames = make_frames(selected=4, rejected={"blurry": 3, "too_similar": 2})
    diag = build(frames=frames)
    assert diag["frames"]["frames_total"] == 9
    assert diag["frames"]["frames_selected"] == 4


def test_reject_reason_counts_aggregates_reasons() -> None:
    """Rejection reasons are aggregated into a reason → count dict."""
    frames = make_frames(selected=2, rejected={"blurry": 3, "too_similar": 1})
    diag = build(frames=frames)
    assert diag["frames"]["reject_reason_counts"] == {"blurry": 3, "too_similar": 1}


def test_registration_ratio_full_and_partial() -> None:
    """Registered share = cameras / selected; partial registration gives a fraction."""
    frames = make_frames(selected=4)
    full = build(frames=frames, model=make_model(num_cameras=4))
    assert full["cameras"]["cameras_registered"] == 4
    assert full["cameras"]["registration_ratio"]["value"] == pytest.approx(1.0)
    assert full["cameras"]["registration_ratio"]["availability"] == "available"

    partial = build(frames=frames, model=make_model(num_cameras=3))
    assert partial["cameras"]["registration_ratio"]["value"] == pytest.approx(0.75)


def test_registration_ratio_unavailable_without_selected_frames() -> None:
    """Zero selected frames neither crashes the function nor causes division by zero.

    The share is undefined — an unavailability mark with a reason is written, not zero:
    zero would read as "no frame of a non-empty set was registered" (FR-036).
    """
    diag = build(frames=[], model=make_model(num_cameras=0))
    ratio = diag["cameras"]["registration_ratio"]
    assert ratio["value"] is None
    assert ratio["availability"] == "unavailable"
    assert ratio["reason"]


def test_background_present_is_always_true() -> None:
    """FR-020: v1 does not remove the background, and the result must declare its presence.

    The field is neither computed nor configured — it is True for any cloud, including an empty one.
    """
    assert build()["points"]["background_present"] is True

    empty = build(
        model=make_model(num_cameras=0, points=np.zeros((0, 3), dtype=np.float64), colors=None),
        points_valid=0,
    )
    assert empty["points"]["background_present"] is True

    dense = build(
        model=make_model(points=np.random.default_rng(0).normal(size=(500, 3))),
        points_valid=500,
    )
    assert dense["points"]["background_present"] is True


def test_points_section_fields() -> None:
    """The points section has raw and valid counts, validity definition, colour and filters."""
    filters = [{"name": "iqr_per_axis", "k": 3.0, "num_points_before": 3, "num_points_after": 2}]
    diag = build(points_valid=2, filters_applied=filters)
    points = diag["points"]
    assert points["num_points_raw"] == 3
    assert points["num_points_valid"] == 2
    assert points["valid_point_definition"] == valid_point_definition()
    assert points["has_colors"] is True
    assert points["filters_applied"] == filters


def test_has_colors_false_without_rgb() -> None:
    """Without a colour array has_colors is False, not an invented True."""
    model = make_model(colors=np.zeros((0, 3), dtype=np.uint8))
    model.points_rgb = None
    diag = build(model=model)
    assert diag["points"]["has_colors"] is False


def test_reprojection_error_unavailable_is_never_zero() -> None:
    """FR-036: an unavailable reprojection error is a mark with a reason, never zero.

    Zero here would mean a perfect reconstruction — a lie about the result.
    """
    diag = build(model=make_model(mean_reprojection_error=None))
    err = diag["reprojection_error"]
    assert err["value"] is None
    assert err["value"] != 0
    assert err["availability"] == "unavailable"
    assert err["reason"]
    assert "bundle adjustment" in err["reason"]


def test_reprojection_error_available_with_meaning() -> None:
    """FR-037: an available value is written together with an explanation of meaning and limits."""
    diag = build(model=make_model(mean_reprojection_error=0.83))
    err = diag["reprojection_error"]
    assert err["value"] == pytest.approx(0.83)
    assert err["availability"] == "available"
    assert err["units"] == "px"
    assert "does not prove" in err["meaning"]


def test_reprojection_error_non_finite_is_unavailable() -> None:
    """A non-finite value is unusable and is marked unavailable rather than written as is."""
    diag = build(model=make_model(mean_reprojection_error=float("nan")))
    err = diag["reprojection_error"]
    assert err["value"] is None
    assert err["availability"] == "unavailable"


def test_visual_check_has_six_unchecked_items() -> None:
    """The visual check has exactly 6 spec items, all not yet checked."""
    visual = build()["visual_check"]
    assert len(visual) == 6
    assert set(visual) == set(VISUAL_CHECK_ITEMS)
    assert all(item["result"] == "not_checked" for item in visual.values())


def test_interpretation_limits_present_and_explicit() -> None:
    """FR-038: limits of conclusions are mandatory: the point count does not prove accuracy."""
    limits = build()["interpretation_limits"]
    assert limits
    assert "number of points" in limits
    assert "accuracy" in limits


def test_bbox_from_valid_points_ignores_nan_and_inf() -> None:
    """Extents are computed over finite points; NaN and Inf are ignored, not spoiling the box."""
    points = np.array(
        [
            [0.0, -1.0, 2.0],
            [np.nan, 5.0, 5.0],
            [3.0, 1.0, -4.0],
            [np.inf, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    diag = build(model=make_model(points=points), points_valid=2)
    bbox = diag["points"]["bbox"]
    assert bbox["availability"] == "available"
    assert bbox["value"] == [[0.0, -1.0, -4.0], [3.0, 1.0, 2.0]]


def test_bbox_unavailable_on_empty_cloud() -> None:
    """An empty cloud gives an unavailability mark, not a zero box."""
    diag = build(
        model=make_model(num_cameras=0, points=np.zeros((0, 3), dtype=np.float64), colors=None),
        points_valid=0,
    )
    bbox = diag["points"]["bbox"]
    assert bbox["value"] is None
    assert bbox["availability"] == "unavailable"
    assert bbox["reason"]


def test_bbox_unavailable_when_no_valid_points_left() -> None:
    """If the filter left no valid points, the extents are unavailable, not zeros."""
    diag = build(points_valid=0)
    assert diag["points"]["bbox"]["availability"] == "unavailable"


def test_points_valid_greater_than_raw_raises() -> None:
    """The filter cannot produce points: a mismatch is a caller error, not a silent fix."""
    with pytest.raises(ValueError, match="exceeds the number of model points"):
        build(points_valid=99)


def test_negative_points_valid_raises() -> None:
    """A negative number of valid points is meaningless."""
    with pytest.raises(ValueError, match="negative"):
        build(points_valid=-1)


def test_bbox_from_valid_points_array_is_narrower() -> None:
    """Array mode: the box is computed over the filtered points and is narrower than the raw one.

    That is the point of the parameter: the box must describe the displayed cloud, not what
    was there before the outlier filter.
    """
    raw = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [100.0, 100.0, 100.0]],
        dtype=np.float64,
    )
    kept = raw[:2]

    before = build(model=make_model(points=raw), points_valid=2)
    after = build(model=make_model(points=raw), points_valid=None, valid_points_xyz=kept)

    assert before["points"]["bbox"]["value"] == [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]
    assert after["points"]["bbox"]["value"] == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert after["points"]["num_points_valid"] == 2
    # A box over the displayed cloud carries no "before filter" note — it is computed over it.
    assert "note" not in after["points"]["bbox"]


def test_bbox_without_array_is_marked_as_before_filter() -> None:
    """Mode without an array: the box is marked as computed before the outlier filter."""
    bbox = build(points_valid=3)["points"]["bbox"]
    assert bbox["availability"] == "available"
    assert "before the outlier filter" in bbox["note"]


def test_points_valid_conflicting_with_array_raises() -> None:
    """The count and the array differ — the caller contradicts itself, we must not choose for it."""
    kept = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="differs"):
        build(points_valid=3, valid_points_xyz=kept)


def test_points_valid_matching_array_is_accepted() -> None:
    """A consistent count and array are accepted without remarks."""
    kept = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    diag = build(points_valid=2, valid_points_xyz=kept)
    assert diag["points"]["num_points_valid"] == 2


def test_valid_points_array_longer_than_raw_raises() -> None:
    """A valid-points array longer than the raw cloud — the filter cannot produce points."""
    kept = np.zeros((99, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="exceeds the number of model points"):
        build(points_valid=None, valid_points_xyz=kept)


def test_empty_valid_points_array_gives_unavailable_bbox() -> None:
    """An empty valid-points array gives an unavailability mark, not a zero box."""
    diag = build(points_valid=None, valid_points_xyz=np.zeros((0, 3), dtype=np.float64))
    bbox = diag["points"]["bbox"]
    assert diag["points"]["num_points_valid"] == 0
    assert bbox["value"] is None
    assert bbox["availability"] == "unavailable"
    assert bbox["reason"]


def test_neither_points_valid_nor_array_raises() -> None:
    """There is nothing to take the valid point count from — it must not be invented."""
    with pytest.raises(ValueError, match="points_valid"):
        build(points_valid=None)


def test_diagnostics_is_strict_json_serializable() -> None:
    """Diagnostics is written as strict JSON: NaN and Infinity are not allowed in it."""
    import json

    diag = build(model=make_model(mean_reprojection_error=0.5))
    json.dumps(diag, ensure_ascii=False, allow_nan=False)
