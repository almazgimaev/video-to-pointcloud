"""Frame selection "quality_nonredundant" (T046, T048; plan §7)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from v3d.select.quality import (
    REJECT_LESS_SHARP,
    REJECT_REDUNDANT,
    select_quality_nonredundant,
    window_bounds,
)


@dataclass(frozen=True)
class Candidate:
    src_index: int
    timestamp_us: int
    path: Path = Path("unused.jpg")


def _candidates(n: int, step: int = 2) -> list[Candidate]:
    return [Candidate(i * step, i * step * 33_333) for i in range(n)]


def _distance_diff(scale: float = 0.01):
    """Difference grows with the distance between frames, like a moving camera."""
    return lambda a, b: min(1.0, abs(a - b) * scale)


def test_budget_is_never_exceeded() -> None:
    cands = _candidates(100)
    sharp = {c.src_index: float(c.src_index % 7) for c in cands}
    records, info = select_quality_nonredundant(
        cands, budget=12, tau=0.0, sharpness_by_index=sharp, diff=_distance_diff()
    )
    assert sum(r.selected for r in records) == 12 == info["selected"]


def test_one_frame_per_window_keeps_time_coverage() -> None:
    """Windows protect viewpoints: every window contributes exactly one frame."""
    cands = _candidates(60)
    # Sharpness strongly favours the start of the video: a global ranking would pick only there.
    sharp = {c.src_index: 1000.0 - c.src_index for c in cands}
    records, _ = select_quality_nonredundant(
        cands, budget=6, tau=0.0, sharpness_by_index=sharp, diff=_distance_diff()
    )
    positions = [i for i, r in enumerate(records) if r.selected]
    for (start, end), pos in zip(window_bounds(60, 6), positions, strict=True):
        assert start <= pos < end


def test_sharpest_frame_of_each_window_wins_when_not_redundant() -> None:
    cands = _candidates(30)
    sharp = {c.src_index: 1.0 for c in cands}
    sharp[cands[7].src_index] = 9.0  # window 0 is [0, 10)
    records, _ = select_quality_nonredundant(
        cands, budget=3, tau=0.0, sharpness_by_index=sharp, diff=_distance_diff()
    )
    assert records[7].selected
    assert records[3].reject_reason == REJECT_LESS_SHARP


def test_too_similar_sharp_frame_is_skipped_for_the_next_one() -> None:
    """A sharper frame that looks like the previous pick yields to a less sharp one."""
    cands = _candidates(20, step=1)
    sharp = {c.src_index: 1.0 for c in cands}
    sharp[9] = 5.0  # window 0 [0,10): sharpest at 9
    sharp[10] = 9.0  # window 1 [10,20): sharpest at 10, right next to the previous pick
    sharp[15] = 4.0

    records, _ = select_quality_nonredundant(
        cands, budget=2, tau=0.03, sharpness_by_index=sharp, diff=_distance_diff(0.01)
    )
    assert records[9].selected
    assert not records[10].selected and records[10].reject_reason == REJECT_REDUNDANT
    assert records[15].selected


def test_window_is_never_left_empty_when_everything_is_similar() -> None:
    cands = _candidates(20, step=1)
    sharp = {c.src_index: float(c.src_index) for c in cands}
    records, info = select_quality_nonredundant(
        cands, budget=2, tau=0.9, sharpness_by_index=sharp, diff=lambda a, b: 0.0
    )
    assert sum(r.selected for r in records) == 2
    assert len(info["redundant_fallbacks"]) == 1


def test_less_sharp_is_not_reported_as_blurry() -> None:
    """Relative sharpness must not trigger the 'blurry video' warning."""
    cands = _candidates(40)
    sharp = {c.src_index: float(c.src_index % 5) for c in cands}
    records, _ = select_quality_nonredundant(
        cands, budget=4, tau=0.0, sharpness_by_index=sharp, diff=_distance_diff()
    )
    assert not any(r.reject_reason == "blurry" for r in records)


def test_selection_is_deterministic_with_ties() -> None:
    cands = _candidates(50)
    sharp = {c.src_index: 1.0 for c in cands}  # all equal: ties broken by src_index
    run = lambda: select_quality_nonredundant(  # noqa: E731
        cands, budget=5, tau=0.0, sharpness_by_index=sharp, diff=_distance_diff()
    )
    assert [r.to_dict() for r in run()[0]] == [r.to_dict() for r in run()[0]]


def test_fewer_candidates_than_budget_takes_all() -> None:
    cands = _candidates(4)
    sharp = {c.src_index: 1.0 for c in cands}
    records, _ = select_quality_nonredundant(
        cands, budget=10, tau=0.0, sharpness_by_index=sharp, diff=_distance_diff()
    )
    assert all(r.selected for r in records)


def test_missing_sharpness_is_an_error_not_a_zero() -> None:
    cands = _candidates(5)
    with pytest.raises(ValueError):
        select_quality_nonredundant(
            cands, budget=2, tau=0.0, sharpness_by_index={}, diff=_distance_diff()
        )


def test_cost_is_reported() -> None:
    """The number of difference evaluations is part of the selection cost (T047)."""
    cands = _candidates(30)
    sharp = {c.src_index: float(c.src_index) for c in cands}
    _, info = select_quality_nonredundant(
        cands, budget=5, tau=0.0, sharpness_by_index=sharp, diff=_distance_diff()
    )
    assert info["diff_evaluations"] == 4  # first window needs none
