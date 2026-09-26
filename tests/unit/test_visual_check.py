"""Visual check recording (FR-049, SC-018)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from v3d import artifacts
from v3d.artifacts import VISUAL_CHECK_ITEMS
from v3d.cli import main
from v3d.errors import (
    InputUnusableError,
    ResultMissingOrIncompatibleError,
)
from v3d.visual_check import read_visual_check, record_visual_check

RUN_ID = "run-test"
NOW = datetime(2026, 9, 26, 21, 40, 0, tzinfo=UTC)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    root = tmp_path / RUN_ID
    artifacts.write_json(root / "manifest.json", {"run_id": RUN_ID})
    artifacts.write_json(
        root / "normalized" / "diagnostics.json", artifacts.new_diagnostics(RUN_ID)
    )
    return root


def _diagnostics(root: Path) -> dict:
    return json.loads((root / "normalized" / "diagnostics.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("result", ["ok", "defect"])
def test_record_writes_result_note_and_time(run_dir: Path, result: str) -> None:
    entry = record_visual_check(run_dir, "background_dominates", result, note="looks so", now=NOW)
    expected = {"result": result, "note": "looks so", "checked_at": "2026-09-26T21:40:00Z"}
    assert entry == expected
    assert _diagnostics(run_dir)["visual_check"]["background_dominates"] == expected


def test_record_leaves_other_items_and_fields_untouched(run_dir: Path) -> None:
    before = _diagnostics(run_dir)
    record_visual_check(run_dir, "duplicated_geometry", "defect", now=NOW)
    after = _diagnostics(run_dir)
    for item in VISUAL_CHECK_ITEMS:
        if item != "duplicated_geometry":
            assert after["visual_check"][item] == before["visual_check"][item]
    after["visual_check"].pop("duplicated_geometry")
    before["visual_check"].pop("duplicated_geometry")
    assert after == before


def test_not_checked_clears_time(run_dir: Path) -> None:
    record_visual_check(run_dir, "point_soup_no_shape", "ok", now=NOW)
    entry = record_visual_check(run_dir, "point_soup_no_shape", "not_checked", note="redo")
    assert entry == {"result": "not_checked", "note": "redo", "checked_at": None}


def test_non_utc_time_is_converted_to_utc(run_dir: Path) -> None:
    from datetime import timedelta, timezone

    local = datetime(2026, 9, 26, 23, 40, 0, tzinfo=timezone(timedelta(hours=2)))
    entry = record_visual_check(run_dir, "color_inconsistency", "ok", now=local)
    assert entry["checked_at"] == "2026-09-26T21:40:00Z"


def test_unknown_item_rejected_with_valid_list(run_dir: Path) -> None:
    with pytest.raises(InputUnusableError) as info:
        record_visual_check(run_dir, "nonsense", "ok")
    assert info.value.exit_code == 2
    assert "background_dominates" in (info.value.details or "")


def test_unknown_result_rejected(run_dir: Path) -> None:
    with pytest.raises(InputUnusableError):
        record_visual_check(run_dir, "background_dominates", "maybe")


def test_missing_diagnostics(run_dir: Path) -> None:
    (run_dir / "normalized" / "diagnostics.json").unlink()
    with pytest.raises(ResultMissingOrIncompatibleError) as info:
        record_visual_check(run_dir, "background_dominates", "ok")
    assert info.value.exit_code == 7


def test_missing_run_dir(tmp_path: Path) -> None:
    with pytest.raises(ResultMissingOrIncompatibleError):
        read_visual_check(tmp_path / "absent")


def test_run_id_mismatch(run_dir: Path) -> None:
    artifacts.write_json(
        run_dir / "normalized" / "diagnostics.json", artifacts.new_diagnostics("x")
    )
    with pytest.raises(ResultMissingOrIncompatibleError) as info:
        record_visual_check(run_dir, "background_dominates", "ok")
    assert info.value.exit_code == 7


def test_legacy_items_without_checked_at_are_readable(run_dir: Path) -> None:
    diagnostics = _diagnostics(run_dir)
    diagnostics["visual_check"]["duplicated_geometry"] = {"result": "defect", "note": "old"}
    artifacts.write_json(run_dir / "normalized" / "diagnostics.json", diagnostics)
    checks = read_visual_check(run_dir)
    assert checks["duplicated_geometry"] == {"result": "defect", "note": "old", "checked_at": None}
    assert set(checks) == set(VISUAL_CHECK_ITEMS)
    # Recording another item does not rewrite the legacy one.
    record_visual_check(run_dir, "color_inconsistency", "ok", now=NOW)
    legacy = _diagnostics(run_dir)["visual_check"]["duplicated_geometry"]
    assert legacy == {"result": "defect", "note": "old"}


def test_cli_round_trip(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["visual-check", str(run_dir), "background_dominates", "defect", "--note", "wall"])
    assert code == 0
    assert "visual check: background_dominates = defect (note: wall)" in capsys.readouterr().out

    assert main(["visual-check", str(run_dir), "--list", "--json"]) == 0
    checks = json.loads(capsys.readouterr().out)
    assert checks["background_dominates"]["result"] == "defect"
    assert checks["background_dominates"]["note"] == "wall"
    assert checks["background_dominates"]["checked_at"].endswith("Z")
    assert checks["duplicated_geometry"]["result"] == "not_checked"

    assert main(["visual-check", str(run_dir), "--list"]) == 0
    assert "background_dominates = defect" in capsys.readouterr().out


def test_cli_exit_codes(run_dir: Path, tmp_path: Path) -> None:
    assert main(["visual-check", str(run_dir), "nonsense", "ok"]) == 2
    assert main(["visual-check", str(run_dir), "background_dominates", "maybe"]) == 2
    assert main(["visual-check", str(run_dir)]) == 2
    assert main(["visual-check", str(tmp_path / "absent"), "--list"]) == 7
