"""Sufficiency checks for the selected frames (FR-010, exit code 3 contract).

The `min_frames` threshold is set by the config and is currently **preliminary**: it was
chosen before the first end-to-end run and is not backed by a measurement (task T045). So
both the failure message and the report must carry the `min_frames_note()` remark —
otherwise the threshold reads as measured.
"""

from __future__ import annotations

from v3d.artifacts import FrameRecord
from v3d.config import preliminary_note
from v3d.errors import NotEnoughFramesError


def min_frames_note() -> str | None:
    """Remark that the threshold is unfounded; `None` if it is already backed by a measurement."""
    return preliminary_note("min_frames")


def check_enough_frames(records: list[FrameRecord], *, min_frames: int) -> None:
    """Fail with exit code 3 if fewer frames were selected than the threshold.

    The message must carry three numbers — considered, selected, minimum — because without
    them the user has nothing to change in the shooting or in the run parameters.

    Raises:
        NotEnoughFramesError: fewer than `min_frames` were selected. No package is
            created in this case.
    """
    considered = len(records)
    selected = sum(1 for record in records if record.selected)
    if selected >= min_frames:
        return

    note = min_frames_note()
    raise NotEnoughFramesError(
        f"fewer usable frames than the minimum: considered {considered}, "
        f"selected {selected}, minimum {min_frames}",
        details=note,
    )
