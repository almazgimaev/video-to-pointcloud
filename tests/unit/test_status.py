"""Tests of the run result status.

Contracts: specs/001-video-3d-mvp/data-model.md §3.7, contracts/errors.md (case matrix),
spec.md FR-016, FR-028, FR-033.

What is protected here:
* a truncated result is not passed off as success under any numbers;
* viewing and export are open only to success and a partial result;
* the "precomputed example" mark is never lost, whatever the status;
* there is no numeric "how much partial is acceptable" threshold in the code.
"""

from __future__ import annotations

import pytest

from v3d.artifacts import EXPORTABLE_STATUSES, VIEWABLE_STATUSES, RunStatus
from v3d.status import (
    PRECOMPUTED_EXAMPLE_MARK,
    build_status,
    determine_status,
    status_banner,
)

RUN_ID = "20260919-120000-ab12cd"

# Transition table from data-model.md §3.7:
# (cameras, valid points, selected frames, result complete) → status.
STATUS_TABLE = [
    ((40, 15000, 40, True), RunStatus.SUCCESS, "all selected frames registered"),
    ((1, 1, 1, True), RunStatus.SUCCESS, "the only frame registered"),
    ((39, 15000, 40, True), RunStatus.PARTIAL, "part of the frames registered"),
    ((4, 500, 40, True), RunStatus.PARTIAL, "a small part of the frames registered"),
    ((0, 15000, 40, True), RunStatus.INVALID_RESULT, "no cameras"),
    ((40, 0, 40, True), RunStatus.INVALID_RESULT, "no valid points"),
    ((0, 0, 40, True), RunStatus.INVALID_RESULT, "neither cameras nor points"),
    ((0, 0, 0, True), RunStatus.INVALID_RESULT, "there was nothing to register"),
    ((40, 15000, 40, False), RunStatus.INTERRUPTED, "result incomplete with full registration"),
    ((20, 15000, 40, False), RunStatus.INTERRUPTED, "result incomplete with partial registration"),
    ((0, 0, 40, False), RunStatus.INTERRUPTED, "result incomplete and empty"),
]


def _determine(cameras: int, points: int, frames: int, complete: bool) -> RunStatus:
    return determine_status(
        cameras_registered=cameras,
        points_valid=points,
        selected_frames=frames,
        result_complete=complete,
    )


# --- 1. Transition table ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "expected_status", "case"),
    STATUS_TABLE,
    ids=[case for _, _, case in STATUS_TABLE],
)
def test_status_follows_transition_table(
    inputs: tuple[int, int, int, bool],
    expected_status: RunStatus,
    case: str,
) -> None:
    """data-model §3.7: each set of facts corresponds to one explicit status (FR-016)."""
    assert _determine(*inputs) is expected_status, f"case '{case}'"


@pytest.mark.parametrize("cameras", [0, 1, 7, 40])
@pytest.mark.parametrize("points", [0, 1, 15000])
def test_incomplete_result_is_never_success(cameras: int, points: int) -> None:
    """Protection against passing off a truncated result as successful: it is always interrupted."""
    status = _determine(cameras, points, 40, False)

    assert status is RunStatus.INTERRUPTED, (
        "an incomplete result overrides any numbers of cameras and points: one cannot "
        "judge from a fragment that the reconstruction succeeded"
    )
    assert status is not RunStatus.SUCCESS


def test_partial_has_no_numeric_threshold() -> None:
    """data-model §3.7: any under-registration is partial, "almost success" does not exist."""
    almost_all = _determine(39, 15000, 40, True)
    barely_any = _determine(2, 15000, 40, True)

    assert almost_all is RunStatus.PARTIAL
    assert barely_any is RunStatus.PARTIAL
    assert almost_all is barely_any, (
        "the 'how much partial is acceptable' threshold is not assigned: the status records "
        "a fact, the assessment is given by the baseline after the first run"
    )


# --- 2. Degenerate and invalid inputs ---------------------------------------------------


@pytest.mark.parametrize(
    ("cameras", "points", "frames", "field_name"),
    [
        (-1, 15000, 40, "cameras_registered"),
        (40, -1, 40, "points_valid"),
        (40, 15000, -1, "selected_frames"),
    ],
)
def test_negative_counters_are_rejected(
    cameras: int, points: int, frames: int, field_name: str
) -> None:
    """A counter is never negative — this is a call error, not a kind of status."""
    with pytest.raises(ValueError, match=field_name):
        _determine(cameras, points, frames, True)


def test_more_cameras_than_frames_is_an_inconsistency() -> None:
    """Registering more frames than were selected is impossible: an input error, not success."""
    with pytest.raises(ValueError, match="than frames selected"):
        _determine(41, 15000, 40, True)


def test_zero_selected_frames_is_not_success() -> None:
    """There was nothing to register: the share is undefined, the result is unusable."""
    assert _determine(0, 0, 0, True) is RunStatus.INVALID_RESULT


# --- 3. Viewing and export (FR-033) -----------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [RunStatus.INVALID_RESULT, RunStatus.INTERRUPTED, RunStatus.FAILED],
)
def test_unusable_result_is_neither_viewed_nor_exported(
    status: RunStatus,
) -> None:
    """FR-033: an empty, truncated or failed result does not go off to export."""
    payload = build_status(RUN_ID, status)

    assert payload["exportable"] is False, f"{status.value} is not exported"
    assert payload["viewable"] is False, f"{status.value} is not viewed"
    assert status not in EXPORTABLE_STATUSES
    assert status not in VIEWABLE_STATUSES


def test_success_is_viewed_and_exported() -> None:
    """contracts/errors.md: success is the only case without reservations."""
    payload = build_status(RUN_ID, RunStatus.SUCCESS)

    assert payload["exportable"] is True
    assert payload["viewable"] is True
    assert payload["status"] == "success"
    assert payload["run_id"] == RUN_ID


def test_partial_result_is_viewed_and_exported() -> None:
    """contracts/errors.md: a partial result may be viewed with a permanent mark."""
    payload = build_status(RUN_ID, RunStatus.PARTIAL, cameras_registered=30, selected_frames=40)

    assert payload["exportable"] is True
    assert payload["viewable"] is True


# --- 4. Incomplete registration warning -------------------------------------------------


def _codes(payload: dict) -> list[str]:
    return [w["code"] for w in payload["warnings"]]


@pytest.mark.parametrize(
    ("cameras", "frames"),
    [(39, 40), (30, 40), (2, 40), (1, 40)],
    ids=["almost all", "more than half", "few", "one frame"],
)
def test_partial_result_carries_a_warning(cameras: int, frames: int) -> None:
    """Under partial, the observation about incomplete registration is reported at any share."""
    payload = build_status(
        RUN_ID, RunStatus.PARTIAL, cameras_registered=cameras, selected_frames=frames
    )

    warnings = [w for w in payload["warnings"] if w["code"] == "few_registered_cameras"]
    assert len(warnings) == 1, "the warning is added exactly once"
    assert warnings[0]["observation_only"] is True, (
        "a warning cannot be presented as an established cause (FR-007)"
    )
    assert str(cameras) in warnings[0]["message"]
    assert str(frames) in warnings[0]["message"]


def test_passed_warning_is_not_duplicated() -> None:
    """If the builder already gave the warning upstream, a second one is not added."""
    own = {
        "code": "few_registered_cameras",
        "message": "the camera position is determined not for all frames: 30 of 40 (75%)",
        "observation_only": True,
    }

    payload = build_status(RUN_ID, RunStatus.PARTIAL, warnings=[own])

    assert _codes(payload).count("few_registered_cameras") == 1
    assert payload["warnings"][0]["message"] == own["message"], "foreign text is not rewritten"


def test_partial_without_numbers_and_without_warning_is_rejected() -> None:
    """A mandatory warning must not be lost silently."""
    with pytest.raises(ValueError, match="few_registered_cameras"):
        build_status(RUN_ID, RunStatus.PARTIAL)


@pytest.mark.parametrize(
    "status", [RunStatus.SUCCESS, RunStatus.INVALID_RESULT, RunStatus.INTERRUPTED]
)
def test_registration_warning_only_for_partial(status: RunStatus) -> None:
    """The observation about incomplete registration is not attributed to other statuses."""
    payload = build_status(RUN_ID, status)

    assert "few_registered_cameras" not in _codes(payload)


def test_passed_warnings_are_kept() -> None:
    """Run warnings are not lost and not mutated in place."""
    original = [
        {
            "code": "background_dominates",
            "message": "background dominates",
            "observation_only": True,
        }
    ]

    payload = build_status(
        RUN_ID, RunStatus.PARTIAL, warnings=original, cameras_registered=30, selected_frames=40
    )

    assert _codes(payload) == ["background_dominates", "few_registered_cameras"]
    assert len(original) == 1, "the caller's list is not modified"


def test_note_and_example_mark_go_into_the_file() -> None:
    """The `note` and `precomputed_example` fields are kept as passed (A-07)."""
    payload = build_status(
        RUN_ID, RunStatus.SUCCESS, precomputed_example=True, note="example from assets/"
    )

    assert payload["precomputed_example"] is True
    assert payload["note"] == "example from assets/"


# --- 5. Status line for permanent display (FR-028) --------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_wording"),
    [
        (RunStatus.PARTIAL, "partial result"),
        (RunStatus.INTERRUPTED, "result incomplete"),
        (RunStatus.INVALID_RESULT, "reconstruction failed"),
        (RunStatus.SUCCESS, "result obtained"),
        (RunStatus.FAILED, "reconstruction ended with an error"),
    ],
    ids=lambda value: value.value if isinstance(value, RunStatus) else value,
)
def test_status_line_names_the_state(status: RunStatus, expected_wording: str) -> None:
    """FR-028: the viewer permanently shows the state of the result, without diagnosing causes."""
    payload = build_status(RUN_ID, status, cameras_registered=30, selected_frames=40)

    line = status_banner(payload)

    assert expected_wording in line
    assert PRECOMPUTED_EXAMPLE_MARK not in line, "there must be no example mark here"


@pytest.mark.parametrize(
    "status",
    [
        RunStatus.SUCCESS,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.INTERRUPTED,
        RunStatus.INVALID_RESULT,
        RunStatus.PREPARED,
        RunStatus.RUNNING,
    ],
    ids=lambda status: status.value,
)
def test_example_mark_is_not_lost_under_any_status(status: RunStatus) -> None:
    """A-07 / FR-028: "precomputed example" is visible under any status, including success."""
    payload = build_status(
        RUN_ID,
        status,
        precomputed_example=True,
        cameras_registered=30,
        selected_frames=40,
    )

    line = status_banner(payload)

    assert PRECOMPUTED_EXAMPLE_MARK in line, (
        "the precomputed-example mark must not be lost under any status"
    )
    assert line != PRECOMPUTED_EXAMPLE_MARK, "the state of the result also stays in the line"


def test_status_line_does_not_name_a_cause() -> None:
    """FR-007: the line describes a state, it does not explain why it turned out that way."""
    payload = build_status(RUN_ID, RunStatus.PARTIAL, cameras_registered=5, selected_frames=40)

    line = status_banner(payload)

    for phrase in ("because", "due to", "the reason", "camera moved"):
        assert phrase not in line


def test_unknown_status_in_file_is_rejected() -> None:
    """A foreign or corrupted status.json does not turn into an empty line on screen."""
    with pytest.raises(ValueError, match="unknown run status"):
        status_banner({"status": "almost_succeeded", "precomputed_example": False})
