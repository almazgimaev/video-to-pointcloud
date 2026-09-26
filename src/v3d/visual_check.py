"""Author's visual check of a run: recording and reading (FR-049, SC-018).

Data model: specs/001-video-3d-mvp/data-model.md §3.6 (``visual_check`` in diagnostics.json).

The check is a human judgement made while looking at the viewer. The system never fills it in
on its own: an item stays ``not_checked`` until the author records a result.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from v3d.artifacts import (
    VISUAL_CHECK_ITEMS,
    read_json,
    require_same_run_id,
    write_json,
)
from v3d.errors import InputUnusableError
from v3d.runs import MANIFEST_NAME, find_run_dir

VISUAL_CHECK_RESULTS = ("ok", "defect", "not_checked")
DIAGNOSTICS_NAME = "diagnostics.json"


def _validate(item: str, result: str) -> None:
    if item not in VISUAL_CHECK_ITEMS:
        raise InputUnusableError(
            f"unknown visual check item: {item}",
            details="valid items: " + ", ".join(VISUAL_CHECK_ITEMS),
        )
    if result not in VISUAL_CHECK_RESULTS:
        raise InputUnusableError(
            f"unknown visual check result: {result}",
            details="valid results: " + ", ".join(VISUAL_CHECK_RESULTS),
        )


def _load(run_dir: Path | str) -> tuple[Path, dict]:
    """Diagnostics path and content; the run_id must match the manifest (FR-029)."""
    layout = find_run_dir(run_dir)
    manifest = read_json(layout.manifest)
    path = layout.normalized / DIAGNOSTICS_NAME
    diagnostics = read_json(path)
    run_id = manifest.get("run_id")
    require_same_run_id(
        run_id if isinstance(run_id, str) else layout.run_id,
        (MANIFEST_NAME, manifest),
        (DIAGNOSTICS_NAME, diagnostics),
    )
    return path, diagnostics


def _normalized(entry: dict | None) -> dict:
    """Item entry with every field present; files written before ``checked_at`` still read."""
    entry = entry or {}
    return {
        "result": entry.get("result", "not_checked"),
        "note": entry.get("note"),
        "checked_at": entry.get("checked_at"),
    }


def record_visual_check(
    run_dir: Path | str,
    item: str,
    result: str,
    *,
    note: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Record the author's result for one item and return the stored entry.

    Read-modify-write of ``normalized/diagnostics.json``: other items and all other fields are
    left untouched. ``note`` is stored as given (``None`` clears an earlier note). For
    ``not_checked`` the time is ``None``: nothing was checked.

    Raises:
        InputUnusableError: unknown item or result (exit code 2).
        ResultMissingOrIncompatibleError: no run, no diagnostics or run_id mismatch (code 7).
    """
    _validate(item, result)
    path, diagnostics = _load(run_dir)

    checked_at = None
    if result != "not_checked":
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        checked_at = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {"result": result, "note": note, "checked_at": checked_at}

    diagnostics.setdefault("visual_check", {})[item] = entry
    write_json(path, diagnostics)
    return entry


def read_visual_check(run_dir: Path | str) -> dict:
    """The ``visual_check`` object: every known item, each with result, note and checked_at."""
    _, diagnostics = _load(run_dir)
    stored = diagnostics.get("visual_check") or {}
    items = [*VISUAL_CHECK_ITEMS, *(k for k in stored if k not in VISUAL_CHECK_ITEMS)]
    return {item: _normalized(stored.get(item)) for item in items}
