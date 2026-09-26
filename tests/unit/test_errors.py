"""Tests of exit codes, application errors and observation warnings.

Contract: specs/001-video-3d-mvp/contracts/errors.md

What is protected here: the exit code is part of the CLI contract (scripts read it),
and a warning (FR-007) has no right to look like an established cause.
"""

from __future__ import annotations

import pytest

from v3d.errors import (
    WARNING_CODES,
    ComparisonConditionsError,
    ExitCode,
    InputUnusableError,
    NotEnoughFramesError,
    PackageIntegrityError,
    ResultEmptyOrInvalidError,
    ResultIncompleteError,
    ResultMissingOrIncompatibleError,
    RunDirExistsError,
    V3dError,
    Warning_,
)

# Table from contracts/errors.md: error class → exit code → case.
CODE_TABLE = [
    (InputUnusableError, 2, "input unusable"),
    (NotEnoughFramesError, 3, "not enough usable frames"),
    (PackageIntegrityError, 4, "package integrity violated"),
    (ResultIncompleteError, 5, "result incomplete / computation interrupted"),
    (ResultEmptyOrInvalidError, 6, "result empty or invalid"),
    (ResultMissingOrIncompatibleError, 7, "files missing or incompatible"),
    (RunDirExistsError, 8, "run directory already exists"),
    (ComparisonConditionsError, 9, "comparison conditions violated"),
]


# --- 1. Error exit codes ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_class", "code", "case"),
    CODE_TABLE,
    ids=[cls.__name__ for cls, _, _ in CODE_TABLE],
)
def test_error_exit_code_matches_contract(error_class, code: int, case: str) -> None:
    """contracts/errors.md: scripts read the exit code, it must not be changed."""
    error = error_class("message")

    assert error.exit_code == code, f"case '{case}' is fixed by exit code {code}"
    assert int(error.exit_code) == code, "the exit code must be convertible to int for the CLI"


def test_unexpected_error_has_code_one() -> None:
    """contracts/errors.md: an unhandled exception is code 1, not a silent success."""
    assert V3dError("something went wrong").exit_code == ExitCode.UNEXPECTED, (
        "the base application error gives the unexpected-error code by default"
    )
    assert ExitCode.UNEXPECTED == 1, "the unexpected-error code is fixed as 1"


@pytest.mark.parametrize(
    ("error_class", "code", "case"),
    CODE_TABLE,
    ids=[cls.__name__ for cls, _, _ in CODE_TABLE],
)
def test_every_error_inherits_from_base(error_class, code: int, case: str) -> None:
    """contracts/errors.md: any application error is caught by a single handler."""
    assert issubclass(error_class, V3dError), (
        f"{error_class.__name__} must subclass V3dError, otherwise the CLI will not catch it"
    )


# --- 2. Integrity of the code list ------------------------------------------------------


def test_success_is_zero() -> None:
    """contracts/errors.md: success is code 0."""
    assert ExitCode.OK == 0, "the success code is fixed as 0"


def test_exit_codes_are_not_reused() -> None:
    """contracts/errors.md: code values are unique — otherwise cases are indistinguishable."""
    members = ExitCode.__members__

    values = [member.value for member in members.values()]

    assert len(values) == len(set(values)), (
        "exit codes must be unique: an alias would make two cases indistinguishable"
    )


def test_error_codes_cover_the_whole_contract() -> None:
    """contracts/errors.md: each code 2–9 corresponds to exactly one error class."""
    codes_from_table = {code for _, code, _ in CODE_TABLE}

    assert codes_from_table == set(range(2, 10)), (
        "the contract defines codes 2 through 9 inclusive; a gap means an uncovered case"
    )


# --- 3. Warnings: observation only (FR-007) ---------------------------------------------


@pytest.mark.parametrize("code", sorted(WARNING_CODES))
def test_known_warning_code_is_accepted(code: str) -> None:
    """FR-007: the set of v1 warning codes is fixed by the contract."""
    w = Warning_(code=code, message="observation")

    assert w.code == code, "the warning code is kept unchanged"


def test_unknown_warning_code_is_forbidden() -> None:
    """FR-007: an arbitrary code would turn warnings into an uncontrolled list."""
    with pytest.raises(ValueError):
        Warning_(code="camera_moved_too_fast", message="the camera moved too fast")


def test_warning_cannot_be_passed_off_as_a_cause() -> None:
    """FR-007: observation_only cannot be turned off — a warning does not establish a cause."""
    with pytest.raises(ValueError):
        Warning_(code="many_blurry_frames", message="many blurry frames", observation_only=False)


def test_warning_is_serialized_with_observation_mark() -> None:
    """FR-007: the observation mark reaches the report and the interface."""
    w = Warning_(code="low_viewpoint_diversity", message="neighbouring frames differ little")

    d = w.to_dict()

    assert d["observation_only"] is True, "the observation mark is mandatory in serialization"
    assert d["code"] == "low_viewpoint_diversity", "the warning code reaches the report"
    assert d["message"] == "neighbouring frames differ little", "the observation text is not lost"
    assert set(d) == {"code", "message", "observation_only"}, (
        "the warning structure is fixed by the contract: {code, message, observation_only}"
    )


def test_warning_is_observation_by_default() -> None:
    """FR-007: being an observation is the default value, not an option."""
    assert Warning_(code="short_video", message="the video is short").observation_only is True, (
        "observation_only must be true without being specified explicitly"
    )


# --- 4. Error rendering -----------------------------------------------------------------


def test_details_are_included_in_error_text() -> None:
    """FR-006: the error message explains exactly what did not match."""
    error = ResultMissingOrIncompatibleError(
        "file cameras.json belongs to a different run",
        details="expected run_id=A, found B",
    )

    text = error.render()

    assert "file cameras.json belongs to a different run" in text, (
        "the main message must be included in the output"
    )
    assert "expected run_id=A, found B" in text, "the details must be included in the output"


def test_without_details_only_the_message_is_rendered() -> None:
    """FR-006: there must be no empty lines or dangling separators in the output."""
    error = InputUnusableError("the file cannot be read")

    assert error.render() == "the file cannot be read", (
        "without details the output consists of the message alone"
    )
    assert error.details is None, "details are absent by default"
