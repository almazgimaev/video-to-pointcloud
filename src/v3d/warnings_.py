"""Observation warnings for the preparation and result stages (FR-007).

WORDING RULE — do not break it when editing the texts:
a warning describes an **observation and its possible consequence** and does NOT name
a cause. "Many frames in the video are blurry — the result may be incomplete" is fine;
"the camera moved too fast" is not: camera motion was not measured, it is a guess.
The `FORBIDDEN_CAUSE_PHRASES` list records this rule in machine-readable form, and a test checks it.

Trigger thresholds are parameters with default values. All of them are **preliminary**:
they were chosen before the first end-to-end run and are not backed by a measurement (like
`min_frames`, task T045). A builder returns `None` if the threshold is not crossed.
"""

from __future__ import annotations

from v3d.errors import Warning_

# Phrases that establish a cause. They must not appear in warning texts.
FORBIDDEN_CAUSE_PHRASES: tuple[str, ...] = (
    "because",
    "since",
    "due to",
    "owing to",
    "caused by",
    "the cause",
    "to blame",
    "the camera moved",
    "the object moved",
    "you moved",
    "you shot",
    "shot too",
    "too fast",
    "should have",
)

# Preliminary thresholds (see the module docstring).
BLURRY_SHARE_THRESHOLD = 0.3
LOW_DIVERSITY_DIFF_THRESHOLD = 0.02
SHORT_VIDEO_DURATION_S = 5.0
FEW_REGISTERED_SHARE_THRESHOLD = 0.6


def thresholds_note() -> str:
    """Remark for the report: the warning thresholds are not backed by a measurement."""
    return (
        "the warning trigger thresholds are preliminary: they were chosen before the first "
        "end-to-end run and are not backed by a measurement (see M1, task T045)"
    )


# --- Preparation stage ------------------------------------------------------------------


def many_blurry_frames(
    share: float,
    rejected: int,
    total: int,
    *,
    threshold: float = BLURRY_SHARE_THRESHOLD,
) -> Warning_ | None:
    """Many frames were rejected for sharpness. `None` if the share is below the threshold."""
    if share < threshold:
        return None
    return Warning_(
        code="many_blurry_frames",
        message=(
            f"many frames in the video are blurry: {rejected} of {total} rejected for sharpness "
            f"({share:.0%}) — there may not be enough selected frames, the result may be incomplete"
        ),
    )


def low_viewpoint_diversity(
    median_diff: float,
    *,
    threshold: float = LOW_DIVERSITY_DIFF_THRESHOLD,
) -> Warning_ | None:
    """Neighbouring selected frames differ little. `None` if the difference exceeds it."""
    if median_diff > threshold:
        return None
    return Warning_(
        code="low_viewpoint_diversity",
        message=(
            f"neighbouring selected frames differ little (median difference "
            f"{median_diff:.4f}) — the reconstruction may fail"
        ),
    )


def short_video(
    duration_s: float,
    *,
    threshold: float = SHORT_VIDEO_DURATION_S,
) -> Warning_ | None:
    """The video is short. `None` if the duration is not less than the threshold."""
    if duration_s >= threshold:
        return None
    return Warning_(
        code="short_video",
        message=(
            f"the video is short: {duration_s:.1f} s — there may be few frames to select from, "
            "the result may be incomplete"
        ),
    )


def precomputed_example(run_id: str) -> Warning_:
    """A precomputed example is shown, not the result of this invocation."""
    return Warning_(
        code="precomputed_example",
        message=(
            f"a precomputed example is shown (run {run_id}) — it was not produced by this "
            "invocation and has no relation to the current input"
        ),
    )


# --- Result stage -----------------------------------------------------------------------


def few_registered_cameras(
    registered: int,
    total: int,
    *,
    threshold: float = FEW_REGISTERED_SHARE_THRESHOLD,
) -> Warning_ | None:
    """A small share of selected frames is registered. `None` if the share exceeds the threshold."""
    share = registered / total if total else 0.0
    if share > threshold:
        return None
    return Warning_(
        code="few_registered_cameras",
        message=(
            f"the camera position is not determined for all frames: {registered} of {total} "
            f"({share:.0%}) — the result may be partial"
        ),
    )


def background_dominates() -> Warning_:
    """The point cloud is dominated by background."""
    return Warning_(
        code="background_dominates",
        message=(
            "the point cloud is dominated by background — the object may be hard to distinguish, "
            "background separation is not performed in this version"
        ),
    )


def reprojection_error_unavailable(reason: str) -> Warning_:
    """The reprojection error was not obtained."""
    return Warning_(
        code="reprojection_error_unavailable",
        message=(
            f"reprojection error is unavailable: {reason} — it cannot be used to judge "
            "the consistency of the model in this run"
        ),
    )


def orientation_not_estimated(reason: str) -> Warning_:
    """The vertical of the scene could not be estimated from the camera trajectory."""
    return Warning_(
        code="orientation_not_estimated",
        message=(
            f"the vertical was not estimated from the camera trajectory ({reason}) — the result "
            "is re-centred on the object but may be shown tilted"
        ),
    )
