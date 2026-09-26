"""Contract tests of the `report.md` report (part of T037).

Sources of requirements:

* spec.md FR-006 (two separate sections), FR-007 (observations, not diagnoses),
  FR-021 (artifact type, axes, units, scale status), FR-020 (background mark),
  FR-034 (information for repetition), FR-035 (readable without the viewer),
  FR-036 (unavailable — with a reason, not zero), FR-037 (a measure with an explanation of
  meaning), FR-038, FR-039 (limits of conclusions), SC-009 (the result is not called anything
  but a point cloud);
* .specify/memory/constitution.md, Principles I and V.

The run directory is assembled by hand: the full `prepare` + `ingest` cycle is not needed
here, the promises of the report are checked, not the structure of the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from v3d.artifacts import (
    VISUAL_CHECK_ITEMS,
    RunStatus,
    available,
    default_conventions,
    new_diagnostics,
    unavailable,
    write_json,
)
from v3d.diagnostics import REPROJECTION_ERROR_MEANING, REPROJECTION_ERROR_UNAVAILABLE_REASON
from v3d.errors import ResultMissingOrIncompatibleError
from v3d.geometry import valid_point_definition
from v3d.ply_io import write_ply
from v3d.report import UNCHECKED_ASSUMPTIONS, VISUAL_CHECK_LABELS, build_report
from v3d.status import PRECOMPUTED_EXAMPLE_MARK, build_status

RUN_ID = "20260101-120000-abcd1234"

# Phrases claiming metric accuracy or judging the result.
# They must not appear in the report under any status (FR-038, FR-039, Principles I and V).
FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "millimeter",
    "millimetre",
    "accuracy is",
    "accuracy of the result",
    "high accuracy",
    "good reconstruction",
    "high-quality reconstruction",
    "confirmed by measurement",
    "surface completeness is achieved",
)

# Terms the result must not be called by (FR-023, SC-009).
FORBIDDEN_RESULT_TERMS: tuple[str, ...] = ("mesh", "splat", "textured model")

# Phrases under which a mention of surface completeness remains a negation, not a claim.
NEGATION_MARKERS: tuple[str, ...] = (
    "not claimed",
    "are not proof",
    "is not proof",
    "does not prove",
    "is not a measurement",
)


# --- Assembling the run directory -------------------------------------------------------


def _manifest(run_id: str = RUN_ID) -> dict:
    return {
        "schema_version": "1",
        "run_id": run_id,
        "created_utc": "2026-01-01T12:00:00+00:00",
        "input_video": {
            "path": "/video/walkaround.mp4",
            "filename": "walkaround.mp4",
            "sha256": "f" * 64,
            "size_bytes": 12_345_678,
            "container": "mp4",
            "video_codec": "h264",
            "duration_s": 18.5,
            "width": 1080,
            "height": 1920,
            "avg_fps": 30.0,
            "nb_frames": 555,
            "rotation_deg": 90,
            "supported": True,
        },
        "image_transform": {
            "rotation_applied_deg": 90,
            "source_size": [1080, 1920],
            "resize_mode": "long_side",
            "resized_size": [576, 1024],
            "scale_factor": 0.533,
            "crop": None,
            "jpeg_quality": 95,
            "candidate_stride": 3,
            "candidates_total": 185,
        },
        "selection": {
            "selector": "uniform",
            "params": {"budget": 80, "seed": 42, "min_frames": 12},
            "candidates_considered": 185,
            "selected": 4,
            "reject_reasons": {"not_on_grid": 180, "blurry": 1},
            "preliminary": [
                "value 'frame_budget' is preliminary: chosen before the first end-to-end "
                "run and not justified by measurement (see M1, task T045)"
            ],
        },
        "seed": 42,
        "code_id": {
            "value": "0123456789abcdef",
            "source": "git",
            "working_tree_dirty": True,
            "dirty_files": ["src/v3d/report.py"],
        },
        "environment": {
            "stage": "mac_light_contour",
            "python": "3.11.9",
            "platform": "macOS-15.5-arm64",
        },
        "commands": [{"command": "prepare", "durations_s": {"check": 0.4, "extract": 2.1}}],
        "reconstructor": {"name": "vggt+colmap", "options": "--use_ba"},
        "forced_overwrite": False,
    }


def _frames(run_id: str = RUN_ID) -> dict:
    records = []
    for seq, src_index in enumerate((10, 40, 70, 100), start=1):
        records.append(
            {
                "frame_id": f"{seq:04d}_{src_index:06d}",
                "src_index": src_index,
                "timestamp_us": src_index * 33_000,
                "sharpness": 120.5 + seq,
                "diff_to_prev_selected": None if seq == 1 else 0.031,
                "selected": True,
                "reject_reason": None,
                "image_file": f"images/{seq:04d}_{src_index:06d}.jpg",
            }
        )
    records.append(
        {
            "frame_id": "----_000025",
            "src_index": 25,
            "timestamp_us": 825_000,
            "sharpness": 3.2,
            "diff_to_prev_selected": None,
            "selected": False,
            "reject_reason": "blurry",
            "image_file": None,
        }
    )
    return {"schema_version": "1", "run_id": run_id, "frames": records}


def _diagnostics(
    run_id: str = RUN_ID,
    *,
    cameras_registered: int = 4,
    points_valid: int = 900,
    reprojection_available: bool = False,
) -> dict:
    diag = new_diagnostics(run_id)
    diag["frames"] = {
        "frames_total": 5,
        "frames_selected": 4,
        "reject_reason_counts": {"blurry": 1},
    }
    diag["cameras"] = {
        "cameras_registered": cameras_registered,
        "registration_ratio": available(
            cameras_registered / 4,
            meaning="Share of selected frames that received a camera pose.",
        ),
    }
    diag["points"] = {
        "num_points_raw": 1000,
        "num_points_valid": points_valid,
        "valid_point_definition": valid_point_definition(),
        "has_colors": True,
        "background_present": True,
        "bbox": available([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], meaning="Bounding box."),
        "filters_applied": [{"name": "iqr_outlier_filter", "k": 3.0, "num_points_before": 1000}],
    }
    diag["stage_durations_s"] = {"ingest": 1.25}
    diag["artifact_sizes_bytes"] = {"normalized/points.ply": 27_000}
    if reprojection_available:
        diag["reprojection_error"] = available(0.83, units="px", meaning=REPROJECTION_ERROR_MEANING)
    else:
        diag["reprojection_error"] = unavailable(
            REPROJECTION_ERROR_UNAVAILABLE_REASON, meaning=REPROJECTION_ERROR_MEANING
        )
    return diag


def _build_run(
    tmp_path: Path,
    *,
    status: RunStatus = RunStatus.SUCCESS,
    precomputed_example: bool = False,
    with_diagnostics: bool = True,
    reprojection_available: bool = False,
    cameras_registered: int = 4,
) -> Path:
    """A run directory assembled by hand from artifacts."""
    run_dir = tmp_path / "runs" / RUN_ID
    normalized = run_dir / "normalized"
    normalized.mkdir(parents=True)

    write_json(run_dir / "manifest.json", _manifest())
    write_json(run_dir / "frames.json", _frames())
    write_json(
        normalized / "cameras.json",
        {
            "schema_version": "1",
            "run_id": RUN_ID,
            "conventions": default_conventions(),
            "cameras": [],
        },
    )
    write_ply(
        normalized / "points.ply",
        np.zeros((3, 3), dtype=np.float32),
        np.full((3, 3), 128, dtype=np.uint8),
        run_id=RUN_ID,
    )
    if with_diagnostics:
        write_json(
            normalized / "diagnostics.json",
            _diagnostics(
                cameras_registered=cameras_registered,
                reprojection_available=reprojection_available,
            ),
        )
    write_json(
        normalized / "status.json",
        build_status(
            RUN_ID,
            status,
            precomputed_example=precomputed_example,
            cameras_registered=cameras_registered,
            selected_frames=4,
        ),
    )
    return run_dir


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return _build_run(tmp_path)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, title: str) -> str:
    """Body of the `## <title>` section up to the next heading of the same level."""
    match = re.search(rf"^## {re.escape(title)}$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match is not None, f"the report has no section '{title}'"
    return match.group(1)


def _line_with(text: str, needle: str) -> str:
    lines = [line for line in text.splitlines() if needle in line]
    assert lines, f"the report has no line containing '{needle}'"
    return lines[0]


# --- Tests ------------------------------------------------------------------------------


def test_report_is_created_next_to_the_run_and_names_the_run(run_dir: Path) -> None:
    """FR-035: the report is a separate file, readable without launching the viewer."""
    path = build_report(run_dir)

    assert path == run_dir / "report.md"
    text = _read(path)
    assert text.strip(), "the report is empty"
    assert RUN_ID in text, "the report does not name the run it belongs to"


def test_report_can_be_written_to_a_given_path(run_dir: Path, tmp_path: Path) -> None:
    """The report is available separately from the run directory: the path is set explicitly."""
    target = tmp_path / "export" / "report.md"
    path = build_report(run_dir, out=target)

    assert path == target
    assert RUN_ID in _read(target)


def test_result_type_scale_status_and_background_are_declared(run_dir: Path) -> None:
    """FR-021 and FR-020: artifact type, units, scale and residual background — in words."""
    text = _read(build_report(run_dir))
    section = _section(text, "Result type and conventions")

    assert "point cloud" in section
    assert "not determined" in section, "the scale status must be named in words"
    assert "unknown" in section, "units are not declared"
    assert "background" in section.lower(), "the residual background mark is not declared"


def test_result_is_not_called_anything_but_a_point_cloud(run_dir: Path) -> None:
    """SC-009 and FR-023: forbidden terms do not appear in the report at all."""
    text = _read(build_report(run_dir)).lower()

    for term in FORBIDDEN_RESULT_TERMS:
        assert term not in text, f"the report calls the result '{term}'"


def test_report_does_not_claim_metric_accuracy(run_dir: Path) -> None:
    """FR-038, FR-039: no quality judgements, no claims about accuracy and completeness."""
    text = _read(build_report(run_dir))
    lowered = text.lower()

    for phrase in FORBIDDEN_CLAIMS:
        assert phrase not in lowered, f"the report contains the claim '{phrase}'"

    for sentence in re.split(r"[.!?\n]", text):
        if "completeness" in sentence and "surface" in sentence:
            assert any(marker in sentence for marker in NEGATION_MARKERS), (
                f"surface completeness is mentioned not as a negation: {sentence!r}"
            )


def test_limits_of_conclusions_are_stated_explicitly(run_dir: Path) -> None:
    """FR-038, FR-039: the absence of an independent reference is spelled out."""
    section = _section(_read(build_report(run_dir)), "Limits of conclusions")

    assert "reference" in section
    assert "not claimed" in section


def test_unavailable_measure_is_printed_with_a_reason_not_zero(run_dir: Path) -> None:
    """FR-036 — the key check: the unavailable is not replaced by zero."""
    text = _read(build_report(run_dir))
    line = _line_with(text, "Reprojection error")

    assert "unavailable:" in line, "the measure is not marked as unavailable"
    assert REPROJECTION_ERROR_UNAVAILABLE_REASON in line, (
        "the reason for unavailability is not named"
    )
    assert re.search(r"(?<![\d,.])0(?![\d,.])", line) is None, (
        f"a zero stands in the line of an unavailable measure: {line!r}"
    )


def test_available_measure_is_printed_with_an_explanation_of_meaning(tmp_path: Path) -> None:
    """FR-037: the measure's value is accompanied by its meaning and limits of applicability."""
    run_dir = _build_run(tmp_path, reprojection_available=True)
    text = _read(build_report(run_dir))

    line = _line_with(text, "Reprojection error")
    assert "0.83 px" in line
    assert REPROJECTION_ERROR_MEANING in text, (
        "the explanation of the measure's meaning is not output"
    )


def test_verified_and_accepted_on_trust_are_separated_into_sections(run_dir: Path) -> None:
    """FR-006: assumptions about the scene are never passed off as a measurement."""
    text = _read(build_report(run_dir))
    verified = _section(text, "Verified by the system")
    accepted = _section(text, "Accepted from the user, not verified")

    assert "container" in verified and "resolution" in verified
    assert "not verified" in accepted or "does not measure" in accepted
    for assumption in UNCHECKED_ASSUMPTIONS:
        assert assumption in accepted, f"the assumption '{assumption}' is not listed"
        assert assumption not in verified, "an assumption ended up in the verified section"


def test_visual_check_is_output_with_all_six_items(run_dir: Path) -> None:
    """Principle V: a fixed list of six items, those not performed are marked."""
    section = _section(_read(build_report(run_dir)), "Visual check")

    assert len(VISUAL_CHECK_ITEMS) == 6
    for item in VISUAL_CHECK_ITEMS:
        assert VISUAL_CHECK_LABELS[item] in section, f"item '{item}' is not output"
    assert section.count("not performed") == 6, "the not_checked state is not output for all items"


def test_frames_and_rejection_reasons_are_output(run_dir: Path) -> None:
    """FR-010: considered, selected, rejection reasons, selection method and seed."""
    section = _section(_read(build_report(run_dir)), "Frames")

    assert "Candidates considered" in section
    assert "Frames selected" in section
    assert "uniform" in section
    assert "Seed" in section
    assert "blurry" in section, "rejection reasons are not output"


def test_reproduction_is_described(run_dir: Path) -> None:
    """FR-034: code version with a dirty-tree mark, seed, environment, commands, package."""
    section = _section(_read(build_report(run_dir)), "Reproduction")

    assert "0123456789abcdef" in section
    assert "dirty" in section, "the dirty working tree is not marked"
    assert "3.11.9" in section
    assert "prepare" in section
    assert "package" in section, "the path to the portable package is not given"


def test_preliminary_values_are_shown(run_dir: Path) -> None:
    """A threshold chosen before measurement must be visible to the reader."""
    section = _section(_read(build_report(run_dir)), "Preliminary values")

    assert "preliminary" in section
    assert "not justified by measurement" in section


def test_report_is_built_for_an_unusable_result(tmp_path: Path) -> None:
    """Principle I: a failure stays readable, the status banner is not lost."""
    run_dir = _build_run(
        tmp_path,
        status=RunStatus.INVALID_RESULT,
        cameras_registered=0,
        with_diagnostics=True,
    )
    text = _read(build_report(run_dir))

    assert "reconstruction failed" in text, "the status banner is not output"
    assert "Limits of conclusions" in text
    for term in FORBIDDEN_RESULT_TERMS:
        assert term not in text.lower()


def test_report_is_built_without_diagnostics(tmp_path: Path) -> None:
    """A missing diagnostics.json is no reason to crash: the section is marked unavailable."""
    run_dir = _build_run(tmp_path, with_diagnostics=False)
    text = _read(build_report(run_dir))

    measures = _section(text, "Measures")
    assert "unavailable" in measures, "the measures section is not marked unavailable"
    assert "diagnostics.json" in measures, "the reason for unavailability is not named"
    assert re.search(r"(?<![\d,.])0(?![\d,.])", measures) is None, (
        "zeros are output in place of unavailable measures"
    )
    assert "Visual check" in text


def test_precomputed_example_mark_is_not_lost(tmp_path: Path) -> None:
    """A-07, FR-028: the example mark is visible in the report under any status."""
    run_dir = _build_run(tmp_path, precomputed_example=True)
    text = _read(build_report(run_dir))

    assert PRECOMPUTED_EXAMPLE_MARK in text.lower() or PRECOMPUTED_EXAMPLE_MARK.upper() in text


def test_artifacts_of_a_foreign_run_are_rejected(run_dir: Path) -> None:
    """FR-029: files from different runs must not be mixed even for a report."""
    write_json(run_dir / "frames.json", _frames(run_id="20260101-120000-ffffffff"))

    with pytest.raises(ResultMissingOrIncompatibleError):
        build_report(run_dir)
