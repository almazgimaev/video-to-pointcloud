"""Contract tests of run artifact schemas and run identity.

Sources of requirements:

* specs/001-video-3d-mvp/data-model.md §2 (run identity), §3.6 (unavailability
  rule), §3.7 (statuses);
* specs/001-video-3d-mvp/contracts/errors.md (exit code 7).

The tests protect promises that must not be silently changed: the set of required fields,
the ban on zero in place of unavailability, the result conventions, refusal on run_id mismatch.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from v3d.artifacts import (
    REQUIRED_MANIFEST_FIELDS,
    SCHEMA_VERSION,
    VISUAL_CHECK_ITEMS,
    Manifest,
    RunStatus,
    default_conventions,
    new_diagnostics,
    new_status,
    read_json,
    require_same_run_id,
    unavailable,
    write_json,
)
from v3d.errors import ExitCode, ResultMissingOrIncompatibleError
from v3d.runid import inputs_digest, make_run_id, sha256_file

RUN_ID = "20260918-120000-deadbeef"


def full_manifest() -> dict:
    """A manifest dict with all required fields."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "created_utc": "2026-09-18T12:00:00Z",
        "input_video": {"filename": "video.mp4", "sha256": "a" * 64},
        "image_transform": {"rotation_applied_deg": 0, "crop": None},
        "selection": {"selector": "uniform", "frame_budget": 40},
        "seed": 17,
        "code_id": {"value": "c" * 40, "source": "git", "working_tree_dirty": False},
        "environment": {"python": "3.11.9", "platform": "darwin"},
        "commands": ["v3d prepare"],
    }


# --- 1. Required manifest fields --------------------------------------------------------


def test_manifest_is_built_from_a_full_dict() -> None:
    """FR-034: a full manifest is accepted without losing values."""
    m = Manifest.from_dict(full_manifest())

    assert m.run_id == RUN_ID, "run_id must survive manifest parsing"
    assert m.seed == 17, "seed must survive: it takes part in the inputs digest"
    assert m.schema_version == SCHEMA_VERSION, "the schema version must survive"


@pytest.mark.parametrize("field_name", REQUIRED_MANIFEST_FIELDS)
def test_missing_required_manifest_field_is_refused(field_name: str) -> None:
    """FR-034: an incomplete manifest is refused with code 7 and the name of the missing field."""
    d = full_manifest()
    del d[field_name]

    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        Manifest.from_dict(d)

    assert refusal.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE, (
        "an incomplete manifest must give exit code 7"
    )
    assert refusal.value.exit_code == 7, "code 7 is fixed by the errors.md contract"
    assert field_name in refusal.value.message, (
        f"the message must name the missing field {field_name}, not be generic"
    )


def test_manifest_ignores_unknown_fields() -> None:
    """FR-034: extra fields do not break parsing — the contract sets a minimum, not a maximum."""
    d = full_manifest()
    d["unknown_field"] = 1

    m = Manifest.from_dict(d)

    assert m.run_id == RUN_ID, "an unknown field must not interfere with manifest parsing"


# --- 2. Unavailability rule (FR-036) ----------------------------------------------------


def test_unavailable_measure_carries_a_reason() -> None:
    """FR-036: unavailability is value=null, availability and a non-empty reason."""
    m = unavailable("the reconstruction stage was not executed")

    assert m["value"] is None, "an unavailable measure cannot have a value"
    assert m["availability"] == "unavailable", "the unavailability marker is mandatory"
    assert m["reason"], "the reason for unavailability must be non-empty"


def test_unavailability_without_a_reason_is_forbidden() -> None:
    """FR-036: an unavailability mark without a reason is meaningless and forbidden."""
    with pytest.raises(ValueError):
        unavailable("")


def test_unavailable_measure_is_not_replaced_by_zero() -> None:
    """FR-036: in the diagnostics skeleton, unavailability is a mark, not zero."""
    d = new_diagnostics(RUN_ID)
    reprojection_error = d["reprojection_error"]

    assert reprojection_error["availability"] == "unavailable", (
        "before reconstruction the reprojection error is unavailable"
    )
    assert reprojection_error["value"] is None, "an unavailable value must be null"
    assert reprojection_error["value"] != 0, "zero in place of unavailability is forbidden (FR-036)"
    assert reprojection_error["reason"], "unavailability must explain itself"


def test_diagnostics_declares_limits_of_conclusions() -> None:
    """FR-038: diagnostics carries text about the limits of interpreting the measures."""
    d = new_diagnostics(RUN_ID)

    assert d["interpretation_limits"], "the interpretation_limits field must be non-empty"
    assert d["run_id"] == RUN_ID, "every artifact carries run_id (FR-029)"


# --- 3. Result conventions (FR-021) -----------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("artifact_type", "point_cloud"),
        ("transform_direction", "world_to_camera"),
        ("units", "unknown"),
        ("scale_status", "not_determined"),
    ],
)
def test_result_conventions_are_fixed(key: str, expected: str) -> None:
    """FR-021: artifact type, transform direction, units and scale are not implied.

    The test must fail on a silent change of the value: this protects against swapping the
    promise (for example, declaring a mesh instead of a point_cloud or metric units without a
    reference).
    """
    c = default_conventions()

    assert c[key] == expected, (
        f"convention {key} is fixed as {expected!r}; changing the value changes the promise"
    )


def test_conventions_declare_formula_and_axes() -> None:
    """FR-021: the transform direction is backed by a formula and a description of axes."""
    c = default_conventions()

    assert c["formula"] == "x_cam = R * x_world + t", "the formula fixes the meaning of R and t"
    assert c["camera_axes"], "camera axes must be declared explicitly"
    assert c["world_axes"], "world axes must be declared explicitly"
    assert c["scale_reference"] is None, "v1 has no scale reference, the reference must be null"


# --- 4. Refusal on run_id mismatch (FR-029) ---------------------------------------------


def test_consistent_artifacts_are_accepted() -> None:
    """FR-029: artifacts of one run pass the check."""
    require_same_run_id(
        RUN_ID,
        ("cameras.json", {"run_id": RUN_ID}),
        ("diagnostics.json", {"run_id": RUN_ID}),
    )


def test_foreign_run_id_is_refused() -> None:
    """FR-029: mixing files from different runs is refused with code 7."""
    foreign = "20260918-130000-cafebabe"

    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        require_same_run_id(RUN_ID, ("cameras.json", {"run_id": foreign}))

    assert refusal.value.exit_code == 7, "a run_id mismatch gives exit code 7"
    assert "cameras.json" in refusal.value.message, "the message must name the offending file"
    assert refusal.value.details is not None and foreign in refusal.value.details, (
        "the details must show the expected and the found run_id"
    )


def test_missing_run_id_field_is_refused() -> None:
    """FR-029: an artifact without run_id must not be accepted silently."""
    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        require_same_run_id(RUN_ID, ("status.json", {}))

    assert refusal.value.exit_code == 7, "a missing run_id gives exit code 7"
    assert "status.json" in refusal.value.message, "the message must name the file without run_id"


# --- 5. Statuses (data-model §3.7) ------------------------------------------------------


@pytest.mark.parametrize("status", [RunStatus.SUCCESS, RunStatus.PARTIAL])
def test_usable_result_is_viewed_and_exported(status: RunStatus) -> None:
    """data-model §3.7: success and partial allow viewing and export."""
    s = new_status(RUN_ID, status)

    assert s["viewable"] is True, f"status {status.value} allows viewing"
    assert s["exportable"] is True, f"status {status.value} allows export"


@pytest.mark.parametrize(
    "status",
    [RunStatus.INVALID_RESULT, RunStatus.INTERRUPTED, RunStatus.FAILED],
)
def test_unusable_result_is_neither_viewed_nor_exported(status: RunStatus) -> None:
    """data-model §3.7: an unusable result is not passed off as usable."""
    s = new_status(RUN_ID, status)

    assert s["viewable"] is False, f"status {status.value} does not allow viewing"
    assert s["exportable"] is False, f"status {status.value} does not allow export"


@pytest.mark.parametrize("status", list(RunStatus))
def test_status_always_declares_example_and_warnings(status: RunStatus) -> None:
    """FR-028: the precomputed example and warnings are always declared."""
    s = new_status(RUN_ID, status)

    assert "precomputed_example" in s, "the precomputed_example field is mandatory under any status"
    assert s["precomputed_example"] is False, "by default a run is not a ready-made example"
    assert s["warnings"] == [], "the warnings field must exist even when empty"
    assert s["status"] == status.value, "the status is written as a string value"


def test_status_marks_a_precomputed_example() -> None:
    """FR-028: a ready-made example is marked explicitly and carries a warning."""
    s = new_status(
        RUN_ID,
        RunStatus.SUCCESS,
        precomputed_example=True,
        warnings=[{"code": "precomputed_example", "message": "example", "observation_only": True}],
    )

    assert s["precomputed_example"] is True, "the example mark must reach status.json"
    assert s["warnings"][0]["observation_only"] is True, "a warning does not name a cause"


# --- 6. Visual check (SC-013) -----------------------------------------------------------


def test_visual_check_contains_six_items() -> None:
    """SC-013: the defect list is fixed by the specification — exactly six items."""
    d = new_diagnostics(RUN_ID)

    assert len(VISUAL_CHECK_ITEMS) == 6, "the specification fixes exactly 6 visual-check items"
    assert set(d["visual_check"]) == set(VISUAL_CHECK_ITEMS), (
        "the set of diagnostics items must match the VISUAL_CHECK_ITEMS list"
    )
    assert len(d["visual_check"]) == 6, "diagnostics has exactly 6 visual-check items"


def test_visual_check_is_not_performed_by_default() -> None:
    """SC-013: a check not performed is not passed off as passed."""
    d = new_diagnostics(RUN_ID)

    for item, entry in d["visual_check"].items():
        assert entry["result"] == "not_checked", f"item {item} is by default not checked, not 'ok'"
        assert entry["note"] is None, f"the unchecked item {item} has no remark"


# --- 7. Write-and-read round trip -------------------------------------------------------


def test_writing_and_reading_preserve_structure(tmp_path) -> None:
    """FR-029: an artifact reads back exactly as it was written."""
    path = tmp_path / "normalized" / "diagnostics.json"
    payload = new_diagnostics(RUN_ID)

    write_json(path, payload)

    assert path.exists(), "write_json must create intermediate directories"
    assert read_json(path) == payload, "the write-and-read round trip must not lose data"


def test_non_ascii_text_is_written_without_escaping(tmp_path) -> None:
    """FR-006: messages are human-readable — non-ASCII text is stored as is."""
    path = tmp_path / "report_meta.json"

    write_json(path, {"run_id": RUN_ID, "note": "frames differ little — naïve café"})
    text = path.read_text(encoding="utf-8")

    assert "frames differ little — naïve café" in text, (
        "non-ASCII text must be stored without \\u-escaping"
    )
    assert "\\u00" not in text, "escaping non-ASCII text makes the file unreadable to humans"


def test_reading_a_missing_file_is_refused(tmp_path) -> None:
    """FR-030: a required file is missing — refusal with code 7, not an empty result."""
    path = tmp_path / "normalized" / "cameras.json"

    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        read_json(path)

    assert refusal.value.exit_code == 7, "a missing required file gives exit code 7"
    assert "cameras.json" in refusal.value.message, "the message must name the missing file"


# --- 8. Stability of run identity (runid.py) --------------------------------------------

VIDEO_HASH = "b" * 64
CODE = "c" * 40
PARAMS = {"selector": "uniform", "frame_budget": 40, "min_frames": 12}


def digest(**overrides) -> str:
    arguments = {
        "video_sha256": VIDEO_HASH,
        "params": PARAMS,
        "code_id": CODE,
        "seed": 17,
    }
    arguments.update(overrides)
    return inputs_digest(**arguments)


def test_digest_does_not_depend_on_key_order() -> None:
    """data-model §2: parameters equal in meaning give one digest."""
    forward = {"selector": "uniform", "frame_budget": 40, "min_frames": 12}
    backward = {"min_frames": 12, "frame_budget": 40, "selector": "uniform"}

    assert digest(params=forward) == digest(params=backward), (
        "the key order in params is not a run input and must not change the digest"
    )


@pytest.mark.parametrize(
    ("description", "override"),
    [
        ("seed", {"seed": 18}),
        ("code_id", {"code_id": "d" * 40}),
        ("video hash", {"video_sha256": "e" * 64}),
        ("selection parameter", {"params": {**PARAMS, "frame_budget": 41}}),
        ("new parameter", {"params": {**PARAMS, "max_frames": 80}}),
    ],
)
def test_digest_changes_when_an_input_changes(description: str, override: dict) -> None:
    """data-model §2: different inputs cannot end up in the same run directory."""
    assert digest(**override) != digest(), (
        f"changing '{description}' must change inputs_digest, otherwise runs would merge"
    )


def test_run_id_contains_the_start_of_the_digest() -> None:
    """data-model §2: run_id = <UTC YYYYMMDD-HHMMSS>-<inputs_digest[:8]>."""
    d = digest()
    moment = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)

    run_id = make_run_id(d, moment)

    assert run_id == f"20260918-120000-{d[:8]}", "the run_id format is fixed by the data model"
    assert run_id.endswith(d[:8]), "run_id must carry the first 8 characters of the digest"


def test_file_sha256_matches_the_reference(tmp_path) -> None:
    """data-model §3.1: the content hash is the only reliable input identifier."""
    data = b"\x00\x01video bytes \xd0\xbf\xd1\x80\xd0\xbe\xd0\xb3\xd0\xbe\xd0\xbd" * 1000
    path = tmp_path / "video.mp4"
    path.write_bytes(data)

    assert sha256_file(path) == hashlib.sha256(data).hexdigest(), (
        "sha256_file must match the ordinary sha256 over the file contents"
    )


def test_different_files_give_different_hashes(tmp_path) -> None:
    """data-model §3.1: two different videos cannot get the same identifier."""
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.write_bytes(b"frame one")
    second.write_bytes(b"frame two")

    assert sha256_file(first) != sha256_file(second), "different content — different hashes"


# --- 9. Strictness of JSON --------------------------------------------------------------


def test_nan_is_not_written_to_an_artifact(tmp_path):
    """NaN is inexpressible in JSON: the artifact must fail rather than be born unreadable."""
    import math

    import pytest

    from v3d.artifacts import write_json

    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"run_id": "r1", "sharpness": math.nan})


def test_infinity_is_not_written_to_an_artifact(tmp_path):
    """Infinity is the same problem as NaN: strict JSON does not contain it."""
    import math

    import pytest

    from v3d.artifacts import write_json

    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"run_id": "r1", "value": math.inf})
