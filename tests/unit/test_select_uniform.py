"""Tests for uniform selection, the frame sufficiency check and warnings.

Requirements: FR-007 (a warning is an observation, not a diagnosis), FR-008 (budget),
FR-010 (a record for every considered frame), FR-011 (determinism).
Data model: specs/001-video-3d-mvp/data-model.md §3.2.

The candidate is replaced by a local `FakeCandidate` with the same fields as
`v3d.extract.CandidateFrame`: selection reads only `src_index` and `timestamp_us` from a
candidate, so the test must not depend on the extraction module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from v3d import warnings_
from v3d.artifacts import read_json, write_json
from v3d.errors import WARNING_CODES, ExitCode, NotEnoughFramesError, Warning_
from v3d.select.checks import check_enough_frames, min_frames_note
from v3d.select.uniform import select_uniform, selection_summary

FRAME_ID_SELECTED = re.compile(r"^\d{4}_\d{6}$")
FRAME_ID_UNSELECTED = re.compile(r"^----_\d{6}$")


@dataclass(frozen=True)
class FakeCandidate:
    """Structural stand-in for `v3d.extract.CandidateFrame`."""

    src_index: int
    timestamp_us: int
    path: Path


def make_candidates(count: int, *, stride: int = 1, fps: float = 30.0) -> list[FakeCandidate]:
    step_us = int(round(1_000_000 / fps)) * stride
    return [
        FakeCandidate(
            src_index=i * stride,
            timestamp_us=i * step_us,
            path=Path(f"work/frames/{i * stride:06d}.jpg"),
        )
        for i in range(count)
    ]


# --- Budget -----------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [1, 2, 7, 12, 40])
def test_budget_is_respected_exactly(budget: int) -> None:
    records = select_uniform(make_candidates(100), budget=budget, seed=42)
    assert sum(1 for r in records if r.selected) == budget


def test_fewer_candidates_than_budget_are_all_taken() -> None:
    candidates = make_candidates(5)
    records = select_uniform(candidates, budget=80, seed=42)
    assert [r.selected for r in records] == [True] * 5


def test_empty_candidate_list_gives_empty_result() -> None:
    assert select_uniform([], budget=80, seed=42) == []


@pytest.mark.parametrize("budget", [0, -1])
def test_non_positive_budget_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        select_uniform(make_candidates(10), budget=budget, seed=42)


# --- Uniformity -------------------------------------------------------------------------


@pytest.mark.parametrize(("count", "budget"), [(100, 10), (100, 7), (57, 12), (31, 30)])
def test_selected_are_spread_uniformly(count: int, budget: int) -> None:
    """Steps between neighbouring selected `src_index` values must differ by at most 1."""
    records = select_uniform(make_candidates(count), budget=budget, seed=42)
    picked = [r.src_index for r in records if r.selected]
    steps = [b - a for a, b in zip(picked, picked[1:], strict=False)]
    assert min(steps) > 0
    assert max(steps) - min(steps) <= 1


def test_extreme_candidates_are_included() -> None:
    candidates = make_candidates(100)
    records = select_uniform(candidates, budget=10, seed=42)
    picked = [r.src_index for r in records if r.selected]
    assert picked[0] == candidates[0].src_index
    assert picked[-1] == candidates[-1].src_index


# --- A record for every candidate ---------------------------------------------------------


def test_a_record_is_returned_for_every_candidate() -> None:
    candidates = make_candidates(50, stride=4)
    records = select_uniform(candidates, budget=10, seed=42)
    assert len(records) == len(candidates)
    assert [r.src_index for r in records] == [c.src_index for c in candidates]
    assert [r.timestamp_us for r in records] == [c.timestamp_us for c in candidates]


def test_unselected_have_a_rejection_reason() -> None:
    records = select_uniform(make_candidates(50), budget=10, seed=42)
    rejected = [r for r in records if not r.selected]
    assert len(rejected) == 40
    assert {r.reject_reason for r in rejected} == {"budget_exhausted"}
    assert all(r.image_file is None for r in rejected)


def test_selected_have_no_rejection_reason() -> None:
    records = select_uniform(make_candidates(50), budget=10, seed=42)
    assert all(r.reject_reason is None for r in records if r.selected)


# --- Frame identifiers --------------------------------------------------------------------


def test_frame_id_and_image_file_format() -> None:
    records = select_uniform(make_candidates(30, stride=7), budget=5, seed=42)
    seq = 0
    for record in records:
        if record.selected:
            assert record.frame_id == f"{seq:04d}_{record.src_index:06d}"
            assert FRAME_ID_SELECTED.match(record.frame_id)
            assert record.image_file == f"{record.frame_id}.jpg"
            seq += 1
        else:
            assert record.frame_id == f"----_{record.src_index:06d}"
            assert FRAME_ID_UNSELECTED.match(record.frame_id)
    assert seq == 5


def test_sequence_numbers_of_selected_run_consecutively_from_zero() -> None:
    records = select_uniform(make_candidates(40), budget=8, seed=42)
    seqs = [r.frame_id.split("_")[0] for r in records if r.selected]
    assert seqs == [f"{i:04d}" for i in range(8)]


# --- Metrics ----------------------------------------------------------------------------


def test_sharpness_is_taken_from_dict_and_missing_stays_none() -> None:
    candidates = make_candidates(10)
    sharpness = {0: 12.5, 5: 3.25}
    records = select_uniform(candidates, budget=10, seed=42, sharpness_by_index=sharpness)
    by_index = {r.src_index: r.sharpness for r in records}
    assert by_index[0] == 12.5
    assert by_index[5] == 3.25
    # Neither 0.0 (zero would mean a measured zero sharpness, FR-036) nor nan (not JSON-safe).
    assert by_index[1] is None


def test_writing_frames_json_with_missing_metric_succeeds_and_gives_null(tmp_path: Path) -> None:
    """`write_json` writes with `allow_nan=False`: a skipped metric must be `null`, not `NaN`."""
    records = select_uniform(make_candidates(6), budget=3, seed=42, sharpness_by_index={0: 9.0})
    path = tmp_path / "frames.json"
    write_json(path, {"run_id": "test", "frames": [r.to_dict() for r in records]})

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    frames = read_json(path)["frames"]
    assert frames[0]["sharpness"] == 9.0
    assert frames[1]["sharpness"] is None
    assert all(f["sharpness"] != 0 for f in frames)
    # json.loads would accept NaN back by default — it is the strict writing that is checked.
    assert json.loads(raw)["frames"][1]["sharpness"] is None


def test_first_selected_has_no_difference() -> None:
    candidates = make_candidates(20)
    diffs = {c.src_index: 0.1 for c in candidates}
    records = select_uniform(candidates, budget=5, seed=42, diff_by_index=diffs)
    selected = [r for r in records if r.selected]
    assert selected[0].diff_to_prev_selected is None
    assert all(r.diff_to_prev_selected == 0.1 for r in selected[1:])


def test_difference_is_not_invented_without_metric_data() -> None:
    records = select_uniform(make_candidates(20), budget=5, seed=42)
    assert all(r.diff_to_prev_selected is None for r in records)


# --- Determinism (FR-011) -----------------------------------------------------------------


def test_determinism_with_identical_arguments() -> None:
    candidates = make_candidates(97, stride=3)
    sharpness = {c.src_index: c.src_index / 7.0 for c in candidates}
    first = select_uniform(candidates, budget=13, seed=42, sharpness_by_index=sharpness)
    second = select_uniform(candidates, budget=13, seed=42, sharpness_by_index=sharpness)
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]


def test_selection_summary_carries_method_parameters_and_reasons() -> None:
    records = select_uniform(make_candidates(50), budget=10, seed=42)
    summary = selection_summary(records, budget=10, seed=42)
    assert summary["selector"] == "uniform"
    assert summary["params"] == {"budget": 10, "seed": 42}
    assert summary["candidates_considered"] == 50
    assert summary["selected"] == 10
    assert summary["reject_reasons"] == {"budget_exhausted": 40}


# --- Frame sufficiency (exit code 3) ------------------------------------------------------


def test_check_passes_with_enough_frames() -> None:
    records = select_uniform(make_candidates(100), budget=20, seed=42)
    assert check_enough_frames(records, min_frames=12) is None


def test_too_few_frames_raise_code_3_with_three_numbers() -> None:
    records = select_uniform(make_candidates(30), budget=4, seed=42)
    with pytest.raises(NotEnoughFramesError) as excinfo:
        check_enough_frames(records, min_frames=12)
    error = excinfo.value
    assert error.exit_code == ExitCode.NOT_ENOUGH_FRAMES == 3
    numbers = re.findall(r"\d+", error.message)
    assert numbers == ["30", "4", "12"], error.message


def test_min_frames_threshold_is_marked_as_preliminary() -> None:
    """The threshold is not backed by a measurement (T045) — the failure must show that."""
    note = min_frames_note()
    assert note is not None and "preliminary" in note
    records = select_uniform(make_candidates(30), budget=4, seed=42)
    with pytest.raises(NotEnoughFramesError) as excinfo:
        check_enough_frames(records, min_frames=12)
    assert excinfo.value.details == note


# --- Observation warnings (FR-007) ---------------------------------------------------------


TRIGGERED_WARNINGS = [
    lambda: warnings_.many_blurry_frames(0.55, 55, 100),
    lambda: warnings_.low_viewpoint_diversity(0.004),
    lambda: warnings_.short_video(2.5),
    lambda: warnings_.few_registered_cameras(3, 20),
    lambda: warnings_.background_dominates(),
    lambda: warnings_.reprojection_error_unavailable("the reconstruction stage was not run"),
    lambda: warnings_.precomputed_example("20260101-000000-abcd"),
]


@pytest.mark.parametrize("builder", TRIGGERED_WARNINGS)
def test_builder_returns_a_warning_with_a_known_code(builder) -> None:
    warning = builder()
    assert isinstance(warning, Warning_)
    assert warning.code in WARNING_CODES
    assert warning.observation_only is True
    assert warning.to_dict()["observation_only"] is True
    assert warning.message


@pytest.mark.parametrize("builder", TRIGGERED_WARNINGS)
def test_warning_text_does_not_establish_a_cause(builder) -> None:
    """Meaning: a warning describes an observation and a possible consequence.

    A phrase explaining *why* it turned out that way ("because", "due to", "the camera moved")
    turns an indirect observation into a diagnosis nobody measured (FR-007).
    """
    text = builder().message.lower()
    found = [phrase for phrase in warnings_.FORBIDDEN_CAUSE_PHRASES if phrase in text]
    assert not found, f"diagnosis wording in the warning: {found} — {text}"


@pytest.mark.parametrize(
    "builder",
    [
        lambda: warnings_.many_blurry_frames(0.05, 5, 100),
        lambda: warnings_.low_viewpoint_diversity(0.5),
        lambda: warnings_.short_video(60.0),
        lambda: warnings_.few_registered_cameras(19, 20),
    ],
)
def test_no_warning_below_the_threshold(builder) -> None:
    assert builder() is None


def test_warning_thresholds_are_marked_as_preliminary() -> None:
    assert "not backed by a measurement" in warnings_.thresholds_note()


def test_forbidden_phrase_list_is_not_empty() -> None:
    """The constant is the machine-readable FR-007 rule; an empty list would disable the check."""
    assert len(warnings_.FORBIDDEN_CAUSE_PHRASES) >= 5
