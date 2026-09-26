"""Exit codes, application errors and observation warnings.

Contract: specs/001-video-3d-mvp/contracts/errors.md

Wording rule (FR-007): a message describes an observation and its possible
consequences but does not establish a cause. "Neighbouring frames differ little —
reconstruction may fail" is acceptable; "the camera moved too fast" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ExitCode(IntEnum):
    """CLI exit codes. Values are fixed by the contract and never reused."""

    OK = 0
    UNEXPECTED = 1
    INPUT_UNUSABLE = 2
    NOT_ENOUGH_FRAMES = 3
    PACKAGE_INTEGRITY = 4
    RESULT_INCOMPLETE = 5
    RESULT_EMPTY_OR_INVALID = 6
    RESULT_MISSING_OR_INCOMPATIBLE = 7
    RUN_DIR_EXISTS = 8
    COMPARISON_CONDITIONS_VIOLATED = 9


class V3dError(Exception):
    """Base application error: carries an exit code and a human-readable message."""

    exit_code: ExitCode = ExitCode.UNEXPECTED

    def __init__(self, message: str, *, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def render(self) -> str:
        return self.message if not self.details else f"{self.message}\n{self.details}"


class InputUnusableError(V3dError):
    """File is unreadable, has no video track, or container/codec is outside the supported list."""

    exit_code = ExitCode.INPUT_UNUSABLE


class NotEnoughFramesError(V3dError):
    """After selecting usable frames, fewer than the `min_frames` threshold remain."""

    exit_code = ExitCode.NOT_ENOUGH_FRAMES


class PackageIntegrityError(V3dError):
    """Checksums of the portable package do not match."""

    exit_code = ExitCode.PACKAGE_INTEGRITY


class ResultIncompleteError(V3dError):
    """Result is incomplete: some required files are missing, computation was interrupted."""

    exit_code = ExitCode.RESULT_INCOMPLETE


class ResultEmptyOrInvalidError(V3dError):
    """A model exists, but there are no camera poses or no valid points."""

    exit_code = ExitCode.RESULT_EMPTY_OR_INVALID


class ResultMissingOrIncompatibleError(V3dError):
    """Required files are missing, run_id does not match, or format version is unknown."""

    exit_code = ExitCode.RESULT_MISSING_OR_INCOMPATIBLE


class RunDirExistsError(V3dError):
    """Run directory already exists; overwriting only with an explicit --force."""

    exit_code = ExitCode.RUN_DIR_EXISTS


class ComparisonConditionsError(V3dError):
    """P2 comparison branches differ in something other than the frame selection method."""

    exit_code = ExitCode.COMPARISON_CONDITIONS_VIOLATED


# v1 warning codes (contracts/errors.md).
WARNING_CODES = frozenset(
    {
        "many_blurry_frames",
        "low_viewpoint_diversity",
        "short_video",
        "few_registered_cameras",
        "background_dominates",
        "reprojection_error_unavailable",
        "precomputed_example",
    }
)


@dataclass(frozen=True)
class Warning_:
    """Run warning.

    The observation_only field exists so that a warning cannot be presented as an
    established cause: it is always true and is serialized.
    """

    code: str
    message: str
    observation_only: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.code not in WARNING_CODES:
            raise ValueError(f"unknown warning code: {self.code}")
        if self.observation_only is not True:
            raise ValueError("observation_only is always True: a warning does not name a cause")

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "observation_only": True,
        }
