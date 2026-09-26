"""P2 frame-selection comparison: fixed conditions, then a pre-declared decision rule.

Data model: specs/001-video-3d-mvp/data-model.md §3.8 (SelectionComparison)
Plan: specs/001-video-3d-mvp/plan.md §7 (P2)
Contract: specs/001-video-3d-mvp/contracts/cli.md (``v3d compare``),
specs/001-video-3d-mvp/contracts/errors.md (exit code 9)

Two entry points:

* :func:`init_conditions` writes ``<exp_dir>/conditions.json`` **before** any run (FR-040,
  FR-041). The file is never edited afterwards: a changed condition means a new experiment
  directory, not a rewritten one (FR-043).
* :func:`compare_runs` reads two existing runs, refuses if they differ in anything but the
  frame selection method (FR-042, exit code 9), applies the decision rule exactly as it was
  stored in ``conditions.json``, and writes ``results.json`` and ``comparison.md`` (FR-044).

Rules deliberately built in here:

* A missing or unavailable metric is recorded as ``{"value": None, "availability":
  "unavailable", "reason": ...}``, never as zero (FR-036, reused from ``v3d.artifacts``).
* The decision rule is read from ``conditions.json`` and applied as-is; it is not
  recomputed or reinterpreted here beyond the fixed formulas it was written with.
* A negative or neutral verdict is written the same way as a positive one: no wording here
  softens ``regression`` or ``no_difference``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v3d.artifacts import available, is_measure, read_json, unavailable, write_json
from v3d.errors import ComparisonConditionsError, RunDirExistsError
from v3d.runid import sha256_file
from v3d.runs import RunLayout, find_run_dir

#: The two P2 branches, in the fixed order (plan.md §7).
BRANCHES: tuple[str, str] = ("uniform", "quality_nonredundant")

#: Metrics compared, in the order they are declared in conditions.json (FR-041).
METRICS_DECLARED: tuple[str, ...] = (
    "cameras_registered",
    "registration_ratio",
    "num_points_valid",
    "alignment_arc_coverage_deg",
    "alignment_planarity",
    "selection_time_s",
    "selection_time_share",
    "visual_check_defects",
)

#: The metric the decision rule is built on (data-model.md §3.8, plan.md §7).
PRIMARY_METRIC = "num_points_valid"

#: Decision-rule parameters, stored in conditions.json and never recomputed.
IMPROVEMENT_RATIO = 1.10
REGRESSION_RATIO = 0.90

#: Limits of interpretation, printed into every results.json / comparison.md (FR-044).
INTERPRETATION_LIMITS = (
    "One video, one run per branch; point count at a fixed confidence threshold is a proxy "
    "for depth confidence, not proof of geometric accuracy."
)


# --- Decision rule text -----------------------------------------------------------------


def _decision_rule() -> dict:
    """Fixed decision-rule text and parameters, written before any run (FR-041)."""
    return {
        "primary_metric": PRIMARY_METRIC,
        "parameters": {
            "improvement_ratio": IMPROVEMENT_RATIO,
            "regression_ratio": REGRESSION_RATIO,
        },
        "improvement": (
            "variant.num_points_valid >= baseline.num_points_valid * improvement_ratio "
            "AND variant.registration_ratio >= baseline.registration_ratio "
            "AND variant.visual_check_defects <= baseline.visual_check_defects"
        ),
        "regression": (
            "variant.num_points_valid <= baseline.num_points_valid * regression_ratio "
            "OR variant.registration_ratio < baseline.registration_ratio "
            "OR variant.visual_check_defects > baseline.visual_check_defects"
        ),
        "no_difference": "neither the improvement nor the regression condition holds",
        "inconclusive": (
            "repeat_baseline is true and the relative difference between the two baseline "
            "runs in the primary metric is >= the relative difference between variant and "
            "baseline: the effect is not larger than run-to-run variation"
        ),
    }


# --- init_conditions ----------------------------------------------------------------------


def init_conditions(
    exp_dir: Path | str,
    *,
    video: Path | str,
    budget: int,
    seed: int,
    reconstructor: str,
    repeat: bool = False,
) -> dict:
    """Write ``<exp_dir>/conditions.json`` before any run.

    Conditions are fixed once and never edited afterwards (FR-041, FR-043): a change in the
    video, budget, seed or reconstructor is a new experiment, not a rewrite of this file.

    Raises:
        RunDirExistsError: ``conditions.json`` already exists in ``exp_dir`` (exit code 8).
    """
    exp_dir = Path(exp_dir)
    conditions_path = exp_dir / "conditions.json"
    if conditions_path.exists():
        raise RunDirExistsError(
            f"conditions file already exists: {conditions_path}",
            details=(
                "conditions are never edited after being written; "
                "a changed condition means a new experiment directory"
            ),
        )

    video_path = Path(video)
    payload = {
        "exp_id": exp_dir.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "video": {"filename": video_path.name, "sha256": sha256_file(video_path)},
        "frame_budget": budget,
        "seed": seed,
        "reconstructor": reconstructor,
        "branches": list(BRANCHES),
        "repeat_baseline": bool(repeat),
        "metrics_declared": list(METRICS_DECLARED),
        "decision_rule": _decision_rule(),
    }
    write_json(conditions_path, payload)
    return payload


# --- Reading a run's manifest fields -------------------------------------------------------


def _manifest_selector(manifest: dict) -> str | None:
    selection = manifest.get("selection") or {}
    params = selection.get("params") or {}
    return params.get("selector") or selection.get("selector")


def _code_id(manifest: dict) -> str | None:
    return (manifest.get("code_id") or {}).get("value")


def _check_against_conditions(conditions: dict, manifest: dict, label: str) -> None:
    """Refuse if ``manifest`` differs from ``conditions`` in video, budget or seed (FR-042)."""
    video = manifest.get("input_video") or {}
    expected_sha256 = (conditions.get("video") or {}).get("sha256")
    if video.get("sha256") != expected_sha256:
        raise ComparisonConditionsError(
            f"{label}: video sha256 does not match conditions.json",
            details=f"expected {expected_sha256}, found {video.get('sha256')}",
        )

    params = (manifest.get("selection") or {}).get("params") or {}
    if params.get("budget") != conditions.get("frame_budget"):
        raise ComparisonConditionsError(
            f"{label}: frame budget does not match conditions.json",
            details=f"expected {conditions.get('frame_budget')}, found {params.get('budget')}",
        )

    if manifest.get("seed") != conditions.get("seed"):
        raise ComparisonConditionsError(
            f"{label}: seed does not match conditions.json",
            details=f"expected {conditions.get('seed')}, found {manifest.get('seed')}",
        )


# --- Collecting metrics from a normalized run ------------------------------------------------


def _require_normalized(layout: RunLayout) -> None:
    """Refuse if the run has no normalized result at all (FR-042)."""
    diagnostics_path = layout.normalized / "diagnostics.json"
    status_path = layout.normalized / "status.json"
    if not diagnostics_path.is_file() or not status_path.is_file():
        raise ComparisonConditionsError(
            f"run has no normalized result: {layout.root}",
            details="normalized/diagnostics.json and normalized/status.json are both required",
        )


def _selection_time(manifest: dict) -> tuple[float | None, float | None]:
    """Duration of the ``select`` stage and the total duration of stages in that command."""
    for command in manifest.get("commands") or []:
        durations = command.get("durations_s") or {}
        if "select" in durations:
            return float(durations["select"]), float(sum(durations.values()))
    return None, None


def _collect_metrics(layout: RunLayout) -> dict[str, dict]:
    """Every declared metric for one run, each as an available/unavailable measure (FR-036)."""
    _require_normalized(layout)
    manifest = read_json(layout.manifest)
    diag = read_json(layout.normalized / "diagnostics.json")
    cameras_payload = read_json(layout.normalized / "cameras.json")

    cameras = diag.get("cameras") or {}
    points = diag.get("points") or {}
    visual_check = diag.get("visual_check") or {}
    quality = (cameras_payload.get("world_alignment") or {}).get("quality") or {}
    selection_time, total_time = _selection_time(manifest)

    cameras_registered = cameras.get("cameras_registered")
    registration_ratio = cameras.get("registration_ratio")
    num_points_valid = points.get("num_points_valid")
    arc_coverage = quality.get("arc_coverage_deg")
    planarity = quality.get("planarity")
    defects = sum(1 for entry in visual_check.values() if entry.get("result") == "defect")

    return {
        "cameras_registered": (
            available(cameras_registered)
            if cameras_registered is not None
            else unavailable("diagnostics.json has no cameras.cameras_registered")
        ),
        "registration_ratio": (
            registration_ratio
            if is_measure(registration_ratio)
            else unavailable("diagnostics.json has no cameras.registration_ratio")
        ),
        "num_points_valid": (
            available(num_points_valid)
            if num_points_valid is not None
            else unavailable("diagnostics.json has no points.num_points_valid")
        ),
        "alignment_arc_coverage_deg": (
            available(arc_coverage, units="deg")
            if arc_coverage is not None
            else unavailable("cameras.json has no world_alignment.quality.arc_coverage_deg")
        ),
        "alignment_planarity": (
            available(planarity)
            if planarity is not None
            else unavailable("cameras.json has no world_alignment.quality.planarity")
        ),
        "selection_time_s": (
            available(selection_time, units="s")
            if selection_time is not None
            else unavailable("manifest has no recorded 'select' stage duration")
        ),
        "selection_time_share": (
            available(selection_time / total_time)
            if selection_time is not None and total_time
            else unavailable(
                "selection time or the total stage time of the same command is unavailable"
            )
        ),
        "visual_check_defects": available(defects),
    }


# --- Decision rule application -------------------------------------------------------------


def _value(measure: dict | None) -> Any:
    return measure.get("value") if measure else None


def _relative_difference(value: float | None, baseline: float | None) -> dict:
    """Relative difference of ``value`` from ``baseline``: ``(value - baseline) / baseline``."""
    if value is None or baseline is None:
        return unavailable("one of the compared values is unavailable")
    if baseline == 0:
        return unavailable("baseline value is zero, relative difference is undefined")
    return available((value - baseline) / baseline)


def _decide(
    metrics_baseline: dict[str, dict],
    metrics_variant: dict[str, dict],
    metrics_repeat: dict[str, dict] | None,
) -> tuple[str, dict, dict | None]:
    """Apply the fixed decision rule; returns (verdict, rel. diff. variant, rel. diff. repeat)."""
    baseline = _value(metrics_baseline[PRIMARY_METRIC])
    variant = _value(metrics_variant[PRIMARY_METRIC])
    rel_variant = _relative_difference(variant, baseline)

    rel_repeat = None
    if metrics_repeat is not None:
        repeat_value = _value(metrics_repeat[PRIMARY_METRIC])
        rel_repeat = _relative_difference(repeat_value, baseline)
        if (
            rel_repeat["availability"] == "available"
            and rel_variant["availability"] == "available"
            and abs(rel_repeat["value"]) >= abs(rel_variant["value"])
        ):
            return "inconclusive", rel_variant, rel_repeat

    if baseline is None or variant is None:
        return "inconclusive", rel_variant, rel_repeat

    reg_ratio_baseline = _value(metrics_baseline["registration_ratio"])
    reg_ratio_variant = _value(metrics_variant["registration_ratio"])
    defects_baseline = _value(metrics_baseline["visual_check_defects"])
    defects_variant = _value(metrics_variant["visual_check_defects"])

    registration_not_worse = (
        reg_ratio_variant is not None
        and reg_ratio_baseline is not None
        and reg_ratio_variant >= reg_ratio_baseline
    )
    registration_worse = (
        reg_ratio_variant is not None
        and reg_ratio_baseline is not None
        and reg_ratio_variant < reg_ratio_baseline
    )
    defects_not_worse = (
        defects_variant is not None
        and defects_baseline is not None
        and defects_variant <= defects_baseline
    )
    defects_worse = (
        defects_variant is not None
        and defects_baseline is not None
        and defects_variant > defects_baseline
    )

    if variant >= baseline * IMPROVEMENT_RATIO and registration_not_worse and defects_not_worse:
        return "improvement", rel_variant, rel_repeat
    if variant <= baseline * REGRESSION_RATIO or registration_worse or defects_worse:
        return "regression", rel_variant, rel_repeat
    return "no_difference", rel_variant, rel_repeat


# --- Rendering comparison.md ----------------------------------------------------------------


def _cell(measure: dict | None) -> str:
    if measure is None:
        return "unavailable"
    if measure.get("availability") == "available":
        value = measure.get("value")
        return f"{value:.6g}" if isinstance(value, float) else str(value)
    return f"unavailable: {measure.get('reason', 'no reason recorded')}"


def _render_comparison_md(results: dict) -> str:
    baseline_branch = results["branches"]["uniform"]
    variant_branch = results["branches"]["quality_nonredundant"]
    repeat_branch = results.get("repeat")
    baseline_metrics = baseline_branch["metrics"]
    variant_metrics = variant_branch["metrics"]
    repeat_metrics = repeat_branch["metrics"] if repeat_branch else None
    rel = results["relative_difference_primary"]

    lines = [
        f"# Comparison `{results['exp_id']}`",
        "",
        f"- **Conditions**: `{results['conditions_path']}`",
        f"- **Generated (UTC)**: {results['generated_utc']}",
        f"- **Baseline run (uniform)**: `{baseline_branch['run_id']}`",
        f"- **Variant run (quality_nonredundant)**: `{variant_branch['run_id']}`",
        f"- **Repeat baseline run**: `{repeat_branch['run_id']}`"
        if repeat_branch
        else "- **Repeat baseline run**: not run",
        f"- **Verdict**: **{results['verdict']}**",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Variant | Repeat | Relative difference (variant vs baseline) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in METRICS_DECLARED:
        rel_cell = _cell(rel["variant_vs_baseline"]) if name == PRIMARY_METRIC else ""
        repeat_cell = _cell(repeat_metrics.get(name)) if repeat_metrics is not None else "not run"
        lines.append(
            f"| {name} | {_cell(baseline_metrics.get(name))} | {_cell(variant_metrics.get(name))} "
            f"| {repeat_cell} | {rel_cell} |"
        )

    if repeat_branch:
        lines += [
            "",
            "**Relative difference between the two baseline runs (primary metric)**: "
            f"{_cell(rel['baseline_vs_repeat'])}",
        ]

    lines += [
        "",
        "## Decision rule",
        "",
        "```json",
        json.dumps(results["decision_rule"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Limits of interpretation",
        "",
        results["interpretation_limits"],
        "",
    ]
    return "\n".join(lines) + "\n"


# --- compare_runs ---------------------------------------------------------------------------


def compare_runs(
    run_a: Path | str,
    run_b: Path | str,
    conditions_path: Path | str,
    *,
    repeat: Path | str | None = None,
) -> dict:
    """Compare ``run_a`` (uniform) against ``run_b`` (quality_nonredundant).

    Refuses with :class:`ComparisonConditionsError` (exit code 9) if the two runs differ in
    anything but the frame selection method: video sha256, frame budget, seed, ``code_id``,
    branch order, or a missing repeat run when ``conditions.json`` requires one (FR-042).

    Applies the decision rule exactly as stored in ``conditions.json`` and writes
    ``<exp_dir>/results.json`` and ``<exp_dir>/comparison.md`` next to the conditions file;
    ``conditions.json`` itself is never modified.

    Returns the same dict that is written to ``results.json``.
    """
    conditions_path = Path(conditions_path)
    conditions = read_json(conditions_path)

    layout_a = find_run_dir(run_a)
    layout_b = find_run_dir(run_b)
    manifest_a = read_json(layout_a.manifest)
    manifest_b = read_json(layout_b.manifest)

    selector_a = _manifest_selector(manifest_a)
    selector_b = _manifest_selector(manifest_b)
    if selector_a != BRANCHES[0]:
        raise ComparisonConditionsError(
            f"run_a must be the {BRANCHES[0]} branch, found selector {selector_a!r}",
            details=str(layout_a.root),
        )
    if selector_b != BRANCHES[1]:
        raise ComparisonConditionsError(
            f"run_b must be the {BRANCHES[1]} branch, found selector {selector_b!r}",
            details=str(layout_b.root),
        )
    _check_against_conditions(conditions, manifest_a, "run_a")
    _check_against_conditions(conditions, manifest_b, "run_b")
    if _code_id(manifest_a) != _code_id(manifest_b):
        raise ComparisonConditionsError(
            "run_a and run_b were produced by different code_id values",
            details=f"{_code_id(manifest_a)} != {_code_id(manifest_b)}",
        )

    if conditions.get("repeat_baseline") and repeat is None:
        raise ComparisonConditionsError(
            "conditions.json requires a repeat baseline run (repeat_baseline: true), "
            "but --repeat was not given"
        )

    layout_a2 = manifest_a2 = None
    if repeat is not None:
        layout_a2 = find_run_dir(repeat)
        manifest_a2 = read_json(layout_a2.manifest)
        selector_a2 = _manifest_selector(manifest_a2)
        if selector_a2 != BRANCHES[0]:
            raise ComparisonConditionsError(
                f"the repeat run must be the {BRANCHES[0]} branch, found selector {selector_a2!r}",
                details=str(layout_a2.root),
            )
        _check_against_conditions(conditions, manifest_a2, "repeat")
        if _code_id(manifest_a2) != _code_id(manifest_a):
            raise ComparisonConditionsError(
                "the repeat run was produced by a different code_id than run_a",
                details=f"{_code_id(manifest_a2)} != {_code_id(manifest_a)}",
            )

    metrics_a = _collect_metrics(layout_a)
    metrics_b = _collect_metrics(layout_b)
    metrics_a2 = _collect_metrics(layout_a2) if layout_a2 is not None else None

    verdict, rel_variant, rel_repeat = _decide(metrics_a, metrics_b, metrics_a2)

    exp_dir = conditions_path.parent
    results_path = exp_dir / "results.json"
    comparison_path = exp_dir / "comparison.md"

    results = {
        "exp_id": conditions.get("exp_id"),
        "conditions_path": str(conditions_path),
        "generated_utc": datetime.now(UTC).isoformat(),
        "branches": {
            BRANCHES[0]: {
                "run_id": manifest_a.get("run_id"),
                "run_dir": str(layout_a.root),
                "metrics": metrics_a,
            },
            BRANCHES[1]: {
                "run_id": manifest_b.get("run_id"),
                "run_dir": str(layout_b.root),
                "metrics": metrics_b,
            },
        },
        "repeat": (
            {
                "run_id": manifest_a2.get("run_id"),
                "run_dir": str(layout_a2.root),
                "metrics": metrics_a2,
            }
            if metrics_a2 is not None
            else None
        ),
        "relative_difference_primary": {
            "variant_vs_baseline": rel_variant,
            "baseline_vs_repeat": rel_repeat,
        },
        "decision_rule": conditions.get("decision_rule"),
        "verdict": verdict,
        "interpretation_limits": INTERPRETATION_LIMITS,
        "results_path": str(results_path),
        "comparison_path": str(comparison_path),
    }
    write_json(results_path, results)
    comparison_path.write_text(_render_comparison_md(results), encoding="utf-8")
    return results
