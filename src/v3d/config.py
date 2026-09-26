"""Default parameters.

Project rule: thresholds that affect the result are not assigned by eye.
Values marked PRELIMINARY are justified not by measurement but by the need to start
somewhere; they must be revisited after the first successful end-to-end run (M1,
task T045) and until then are shown in the report with a corresponding note.
"""

from __future__ import annotations

from dataclasses import dataclass

# Values not yet justified by measurement (see plan.md §2, question Q3).
#
# frame_budget is justified by the M1 measurement (2026-09-20, docs/m1-feasibility.md): the
# reconstructor's memory use is ≈ 1 GiB + 0.27 GiB per frame, and about 18 GB of GPU memory
# is available. 48 frames take ≈ 13.8 GiB — this is the measured ceiling, so it was chosen.
#
# min_frames remains preliminary: reconstruction has never been run with fewer than 24
# frames, so where the boundary of usability lies is unknown.
#
# The world-alignment gate thresholds are preliminary as well: measured on one real capture only
# (pumpkin, 48 frames: planarity 0.055, arc 334°, sign agreement 0.77), far from the limits.
PRELIMINARY_DEFAULTS = frozenset(
    {
        "min_frames",
        "align_max_planarity",
        "align_min_arc_deg",
        "align_min_sign_agreement",
        "align_min_cameras",
    }
)


@dataclass(frozen=True)
class Defaults:
    """Default values for data preparation."""

    frame_budget: int = 48
    min_frames: int = 12
    long_side_px: int = 1024
    jpeg_quality: int = 95
    seed: int = 42
    outlier_iqr_k: float = 3.0
    # Redundancy threshold tau for quality_nonredundant selection (task T049). Tuned on a
    # separate tuning video (meat360.mp4, not part of the evaluation set) by a rule declared
    # before computing it: tau = 0.5 x median frame difference between consecutive frames of a
    # uniform 48-frame selection. Median 0.0246 -> tau 0.0123. On the tuning video no uniform
    # pair fell below tau, so the redundancy check is expected to trigger rarely at this budget.
    redundancy_diff_threshold: float | None = 0.0123
    # World-alignment quality gate (FR-047). Out-of-plane spread of camera centres relative to
    # the smaller in-plane spread; angular span of the orbit; agreement between the ring normal
    # and the mean camera "up"; minimum number of registered cameras.
    align_max_planarity: float = 0.25
    align_min_arc_deg: float = 180.0
    align_min_sign_agreement: float = 0.3
    align_min_cameras: int = 6


DEFAULTS = Defaults()

# Upstream fact, not a tunable: VGGT demo_colmap.py keeps at most this many points
# (`max_points_for_colmap`). A run that reaches it has more points passing the confidence
# threshold than it reports, so the count is a lower bound.
UPSTREAM_POINT_CAP = 100_000


def is_preliminary(name: str) -> bool:
    """True if the parameter value is not yet justified by measurement."""
    return name in PRELIMINARY_DEFAULTS


def preliminary_note(name: str) -> str | None:
    """Note text for the report; None if the value is justified."""
    if not is_preliminary(name):
        return None
    return (
        f"value '{name}' is preliminary: chosen before the first end-to-end run and "
        "not justified by measurement (see M1, task T045)"
    )
