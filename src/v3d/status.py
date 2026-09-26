"""Run result status: determination, recording and the permanent display line.

Contracts: specs/001-video-3d-mvp/data-model.md §3.7 (transition table),
specs/001-video-3d-mvp/contracts/errors.md (case matrix), spec.md FR-016, FR-028, FR-033.

**No numeric threshold for "how much partial is acceptable" is assigned.** The status records
a fact: all selected frames were registered, or only some. Rules like "more than 80 % counts
as success" are forbidden here — such a threshold would be an invented measurement; the
assessment is given by the baseline after the first end-to-end run (Group B in spec, task T045).

The status diagnoses nothing: it does not name the reason why frames were not
registered and does not judge the quality of the model.
"""

from __future__ import annotations

from v3d import warnings_
from v3d.artifacts import EXPORTABLE_STATUSES, VIEWABLE_STATUSES, RunStatus, new_status

# Status line for permanent display in the viewer and in the report (FR-028).
# The wording states the state of the result, without diagnosing causes.
STATUS_BANNERS: dict[RunStatus, str] = {
    RunStatus.PREPARED: "data prepared, reconstruction not run",
    RunStatus.RUNNING: "reconstruction in progress",
    RunStatus.SUCCESS: "result obtained",
    RunStatus.PARTIAL: "partial result",
    RunStatus.FAILED: "reconstruction ended with an error",
    RunStatus.INTERRUPTED: "result incomplete",
    RunStatus.INVALID_RESULT: "reconstruction failed",
}

# Mark A-07 / FR-028: never lost, whatever the status.
PRECOMPUTED_EXAMPLE_MARK = "precomputed example"

_BANNER_SEPARATOR = " · "


def determine_status(
    *,
    cameras_registered: int,
    points_valid: int,
    selected_frames: int,
    result_complete: bool,
) -> RunStatus:
    """Determine the result status from the facts of ingest (data-model §3.7).

    The order of checks matters:

    1. `result_complete is False` → `INTERRUPTED`. An incomplete result overrides everything
       else: one cannot judge from a fragment that the reconstruction succeeded, even if
       the cameras and points that were read look acceptable.
    2. no cameras **or** no valid points → `INVALID_RESULT`.
    3. all selected frames registered → `SUCCESS`.
    4. part registered → `PARTIAL`.

    **No numeric threshold for "how much partial is acceptable" is assigned** (see the module
    docstring): any under-registration is `PARTIAL`, with no split into "almost success" and
    "bad partial".

    Degenerate inputs:

    * `selected_frames == 0` — there was nothing to register, the registration share is
      undefined, so the result can be considered neither complete nor partial:
      `INVALID_RESULT` (if execution reached this check, i.e. the result is complete and
      cameras with points exist).
    * `cameras_registered > selected_frames` — more frames were registered than were
      selected; this is an inconsistency of inputs (caught by `v3d.ingest_checks` via the
      name-subset check), not a kind of success: `ValueError`.
    * negative numbers — `ValueError`: a counter is never negative.
    """
    _require_non_negative(
        cameras_registered=cameras_registered,
        points_valid=points_valid,
        selected_frames=selected_frames,
    )
    if cameras_registered > selected_frames:
        raise ValueError(
            "more cameras registered than frames selected: "
            f"{cameras_registered} > {selected_frames}"
        )

    if not result_complete:
        return RunStatus.INTERRUPTED
    if cameras_registered == 0 or points_valid == 0:
        return RunStatus.INVALID_RESULT
    if selected_frames == 0:
        return RunStatus.INVALID_RESULT
    if cameras_registered == selected_frames:
        return RunStatus.SUCCESS
    return RunStatus.PARTIAL


def build_status(
    run_id: str,
    status: RunStatus,
    *,
    warnings: list[dict] | tuple[dict, ...] = (),
    precomputed_example: bool = False,
    note: str | None = None,
    cameras_registered: int | None = None,
    selected_frames: int | None = None,
) -> dict:
    """Build the contents of `normalized/status.json` on top of `artifacts.new_status`.

    Adds one thing to the skeleton: for status `PARTIAL`, the observation warning
    `few_registered_cameras`, if it is not yet in `warnings`. The text comes from the builder
    `v3d.warnings_.few_registered_cameras`; the module introduces no wording of its own.

    The builder's share threshold is preliminary and is **not applied** here: status `PARTIAL`
    already means that not all selected frames were registered, and this fact is reported
    regardless of whether the share is large or small (the "how much partial is acceptable"
    threshold is not assigned).

    `cameras_registered` and `selected_frames` are needed only for the text of this warning.
    If the status is `PARTIAL`, there is no such warning in `warnings` and the numbers are not
    passed — `ValueError`: a mandatory warning must not be silently lost.

    The `exportable`/`viewable` invariant (FR-033): true only for `SUCCESS` and `PARTIAL`.
    It is ensured by `new_status`; here it is re-checked so that an unusable result
    does not get exported because of an edit in a neighbouring module.
    """
    warning_list = [dict(w) for w in warnings]

    if status is RunStatus.PARTIAL and not _has_warning(warning_list, "few_registered_cameras"):
        warning_list.append(_few_registered_warning(cameras_registered, selected_frames))

    payload = new_status(
        run_id,
        status,
        warnings=warning_list,
        precomputed_example=precomputed_example,
        note=note,
    )

    _assert_export_invariant(payload, status)
    return payload


def status_banner(status_payload: dict) -> str:
    """Short status line for permanent display in the viewer and in the report (FR-028).

    Describes the state of the result and does not name a cause. If the metadata has
    `precomputed_example` set, the line necessarily carries the mark "precomputed example"
    (A-07): it is never lost, whatever the status, including success.
    """
    raw_status = status_payload.get("status")
    try:
        status_value = RunStatus(raw_status)
    except ValueError as exc:
        raise ValueError(f"unknown run status: {raw_status!r}") from exc

    parts = [STATUS_BANNERS[status_value]]
    if status_payload.get("precomputed_example"):
        parts.append(PRECOMPUTED_EXAMPLE_MARK)
    return _BANNER_SEPARATOR.join(parts)


# --- Helpers ----------------------------------------------------------------------------


def _require_non_negative(**values: int) -> None:
    """Counters are never negative: a negative value is a call error."""
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must not be negative: {value}")


def _has_warning(warnings: list[dict], code: str) -> bool:
    return any(w.get("code") == code for w in warnings)


def _few_registered_warning(
    cameras_registered: int | None,
    selected_frames: int | None,
) -> dict:
    """Incomplete-registration warning from the `v3d.warnings_` builder."""
    if cameras_registered is None or selected_frames is None:
        raise ValueError(
            "status partial requires cameras_registered and selected_frames: "
            "without them the mandatory few_registered_cameras warning cannot be built"
        )
    _require_non_negative(
        cameras_registered=cameras_registered,
        selected_frames=selected_frames,
    )
    # threshold=1.0: the share under partial is always below one, so the observation
    # is always reported, and the builder's preliminary threshold is not applied.
    warning = warnings_.few_registered_cameras(cameras_registered, selected_frames, threshold=1.0)
    if warning is None:  # pragma: no cover - unreachable under partial
        raise ValueError(
            "the few_registered_cameras builder gave no warning under status partial: "
            f"{cameras_registered} of {selected_frames}"
        )
    return warning.to_dict()


def _assert_export_invariant(payload: dict, status: RunStatus) -> None:
    """FR-033: viewing and export are open only for `SUCCESS` and `PARTIAL`."""
    if payload["exportable"] is not (status in EXPORTABLE_STATUSES):
        raise AssertionError(f"exportable invariant violated for status {status.value}")
    if payload["viewable"] is not (status in VIEWABLE_STATUSES):
        raise AssertionError(f"viewable invariant violated for status {status.value}")
