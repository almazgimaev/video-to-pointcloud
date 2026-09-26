"""Contract tests for the P2 comparison module (`v3d.compare`).

Sources of requirements:

* spec.md FR-040..FR-044 (P2 comparison: fixed conditions, pre-declared decision rule);
* specs/001-video-3d-mvp/data-model.md §3.8 (SelectionComparison);
* specs/001-video-3d-mvp/plan.md §7 (P2 frame experiment);
* specs/001-video-3d-mvp/contracts/errors.md (exit code 9 = comparison conditions violated).

Run directories are assembled by hand from just the JSON files `compare.py` reads
(`manifest.json` and `normalized/{diagnostics,cameras,status}.json`); no reconstruction runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from v3d.artifacts import VISUAL_CHECK_ITEMS, available, write_json
from v3d.compare import INTERPRETATION_LIMITS, METRICS_DECLARED, compare_runs, init_conditions
from v3d.errors import ComparisonConditionsError, RunDirExistsError
from v3d.runid import sha256_file

# --- Building conditions and fake runs ----------------------------------------------------


def _make_conditions(tmp_path: Path, **overrides) -> tuple[Path, Path, dict]:
    """Write conditions.json via init_conditions and return (exp_dir, conditions_path, dict)."""
    video = tmp_path / "video.mp4"
    if not video.exists():
        video.write_bytes(b"fake mp4 bytes for the contract test")
    exp_dir = tmp_path / "experiments" / "exp1"
    kwargs = {
        "budget": 48,
        "seed": 0,
        "reconstructor": "vggt+colmap --conf_thres_value=1.6",
        "repeat": False,
    }
    kwargs.update(overrides)
    conditions = init_conditions(exp_dir, video=video, **kwargs)
    return exp_dir, exp_dir / "conditions.json", conditions


def _manifest(
    run_id: str,
    *,
    selector: str = "uniform",
    video_sha256: str,
    budget: int,
    seed: int,
    code_id: str = "code-a",
    select_time: float | None = 0.5,
) -> dict:
    durations = {"check": 0.1, "extract": 2.0, "package": 0.05}
    if select_time is not None:
        durations["select"] = select_time
    return {
        "run_id": run_id,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "input_video": {"filename": "video.mp4", "sha256": video_sha256},
        "selection": {"selector": selector, "params": {"selector": selector, "budget": budget}},
        "seed": seed,
        "code_id": {"value": code_id, "source": "git", "working_tree_dirty": False},
        "commands": [{"command": "prepare", "durations_s": durations}],
    }


def _diagnostics(
    run_id: str,
    *,
    cameras_registered: int | None = 48,
    registration_ratio: float | None = 1.0,
    points_valid: int | None = 1000,
    defects: int = 0,
) -> dict:
    cameras: dict = {}
    if cameras_registered is not None:
        cameras["cameras_registered"] = cameras_registered
    if registration_ratio is not None:
        cameras["registration_ratio"] = available(registration_ratio)
    points: dict = {}
    if points_valid is not None:
        points["num_points_valid"] = points_valid
    visual_check = {
        item: {"result": "defect" if i < defects else "ok"}
        for i, item in enumerate(VISUAL_CHECK_ITEMS)
    }
    return {"run_id": run_id, "cameras": cameras, "points": points, "visual_check": visual_check}


def _cameras(run_id: str, *, arc_coverage_deg: float = 334.0, planarity: float = 0.05) -> dict:
    return {
        "run_id": run_id,
        "world_alignment": {
            "quality": {"arc_coverage_deg": arc_coverage_deg, "planarity": planarity}
        },
    }


def _build_run(
    tmp_path: Path,
    run_id: str,
    *,
    selector: str = "uniform",
    video_sha256: str,
    budget: int,
    seed: int,
    code_id: str = "code-a",
    with_normalized: bool = True,
    **metric_overrides,
) -> Path:
    run_dir = tmp_path / "runs" / run_id
    normalized = run_dir / "normalized"
    normalized.mkdir(parents=True)
    write_json(
        run_dir / "manifest.json",
        _manifest(
            run_id,
            selector=selector,
            video_sha256=video_sha256,
            budget=budget,
            seed=seed,
            code_id=code_id,
        ),
    )
    if with_normalized:
        diag_keys = {"cameras_registered", "registration_ratio", "points_valid", "defects"}
        diag_kwargs = {k: v for k, v in metric_overrides.items() if k in diag_keys}
        camera_kwargs = {k: v for k, v in metric_overrides.items() if k not in diag_keys}
        points_valid = diag_kwargs.pop("points_valid", 1000)
        write_json(
            normalized / "diagnostics.json",
            _diagnostics(run_id, points_valid=points_valid, **diag_kwargs),
        )
        write_json(normalized / "cameras.json", _cameras(run_id, **camera_kwargs))
        write_json(normalized / "status.json", {"run_id": run_id, "status": "success"})
    return run_dir


# --- init_conditions ------------------------------------------------------------------------


def test_init_writes_all_fields_and_refuses_to_overwrite(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake mp4 bytes for the contract test")
    exp_dir = tmp_path / "experiments" / "exp1"

    conditions = init_conditions(
        exp_dir,
        video=video,
        budget=48,
        seed=0,
        reconstructor="vggt+colmap --conf_thres_value=1.6",
        repeat=True,
    )

    assert conditions["exp_id"] == "exp1"
    assert conditions["video"] == {"filename": "video.mp4", "sha256": sha256_file(video)}
    assert conditions["frame_budget"] == 48
    assert conditions["seed"] == 0
    assert conditions["reconstructor"] == "vggt+colmap --conf_thres_value=1.6"
    assert conditions["branches"] == ["uniform", "quality_nonredundant"]
    assert conditions["repeat_baseline"] is True
    assert conditions["metrics_declared"] == list(METRICS_DECLARED)
    assert conditions["decision_rule"]["primary_metric"] == "num_points_valid"
    assert conditions["decision_rule"]["parameters"] == {
        "improvement_ratio": 1.10,
        "regression_ratio": 0.90,
    }

    on_disk = json.loads((exp_dir / "conditions.json").read_text(encoding="utf-8"))
    assert on_disk == conditions

    with pytest.raises(RunDirExistsError):
        init_conditions(exp_dir, video=video, budget=48, seed=0, reconstructor="something else")


def test_conditions_file_unchanged_after_compare(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    before = conditions_path.read_text(encoding="utf-8")
    sha = conditions["video"]["sha256"]

    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0, points_valid=1000)
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1200,
    )

    compare_runs(run_a, run_b, conditions_path)

    assert conditions_path.read_text(encoding="utf-8") == before


# --- Refusals (exit code 9) -----------------------------------------------------------------


def test_refuses_on_different_video_sha256(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256="f" * 64, budget=48, seed=0)
    run_b = _build_run(
        tmp_path, "run-b", selector="quality_nonredundant", video_sha256=sha, budget=48, seed=0
    )

    with pytest.raises(ComparisonConditionsError):
        compare_runs(run_a, run_b, conditions_path)


def test_refuses_on_different_budget(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0)
    run_b = _build_run(
        tmp_path, "run-b", selector="quality_nonredundant", video_sha256=sha, budget=24, seed=0
    )

    with pytest.raises(ComparisonConditionsError):
        compare_runs(run_a, run_b, conditions_path)


def test_refuses_on_different_seed(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0)
    run_b = _build_run(
        tmp_path, "run-b", selector="quality_nonredundant", video_sha256=sha, budget=48, seed=7
    )

    with pytest.raises(ComparisonConditionsError):
        compare_runs(run_a, run_b, conditions_path)


def test_refuses_on_different_code_id(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0, code_id="code-a")
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        code_id="code-b",
    )

    with pytest.raises(ComparisonConditionsError):
        compare_runs(run_a, run_b, conditions_path)


def test_refuses_on_wrong_selector_order(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    # run_a is the variant, run_b is the baseline: swapped order.
    run_a = _build_run(
        tmp_path, "run-a", selector="quality_nonredundant", video_sha256=sha, budget=48, seed=0
    )
    run_b = _build_run(tmp_path, "run-b", selector="uniform", video_sha256=sha, budget=48, seed=0)

    with pytest.raises(ComparisonConditionsError):
        compare_runs(run_a, run_b, conditions_path)


def test_refuses_when_repeat_required_but_missing(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path, repeat=True)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0)
    run_b = _build_run(
        tmp_path, "run-b", selector="quality_nonredundant", video_sha256=sha, budget=48, seed=0
    )

    with pytest.raises(ComparisonConditionsError):
        compare_runs(run_a, run_b, conditions_path)


def test_refuses_when_a_run_has_no_normalized_result(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0)
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        with_normalized=False,
    )

    with pytest.raises(ComparisonConditionsError):
        compare_runs(run_a, run_b, conditions_path)


# --- Verdicts --------------------------------------------------------------------------------


def test_verdict_improvement(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(
        tmp_path,
        "run-a",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1000,
        registration_ratio=1.0,
        defects=0,
    )
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1200,
        registration_ratio=1.0,
        defects=0,
    )

    result = compare_runs(run_a, run_b, conditions_path)

    assert result["verdict"] == "improvement"


def test_verdict_regression_on_fewer_points(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0, points_valid=1000)
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=800,
    )

    result = compare_runs(run_a, run_b, conditions_path)

    assert result["verdict"] == "regression"


def test_verdict_regression_on_lower_registration_ratio(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(
        tmp_path,
        "run-a",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1000,
        registration_ratio=1.0,
    )
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1200,
        registration_ratio=0.5,
    )

    result = compare_runs(run_a, run_b, conditions_path)

    assert result["verdict"] == "regression"


def test_verdict_no_difference(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0, points_valid=1000)
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1050,
    )

    result = compare_runs(run_a, run_b, conditions_path)

    assert result["verdict"] == "no_difference"


def test_verdict_inconclusive_when_repeat_variance_dominates(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path, repeat=True)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0, points_valid=1000)
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1150,  # relative difference 0.15: would be "improvement" on its own
    )
    run_a2 = _build_run(
        tmp_path,
        "run-a2",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1250,  # relative difference from run_a is 0.25, >= 0.15
    )

    result = compare_runs(run_a, run_b, conditions_path, repeat=run_a2)

    assert result["verdict"] == "inconclusive"


# --- Unavailable metrics ---------------------------------------------------------------------


def test_unavailable_metric_is_recorded_as_unavailable_not_zero(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(
        tmp_path,
        "run-a",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1000,
        registration_ratio=None,  # not measured for this run
    )
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=1200,
    )

    result = compare_runs(run_a, run_b, conditions_path)

    metric = result["branches"]["uniform"]["metrics"]["registration_ratio"]
    assert metric["availability"] == "unavailable"
    assert metric["value"] is None
    assert metric.get("reason")


# --- comparison.md ---------------------------------------------------------------------------


def test_comparison_md_contains_verdict_and_limits(tmp_path: Path) -> None:
    exp_dir, conditions_path, conditions = _make_conditions(tmp_path)
    sha = conditions["video"]["sha256"]
    run_a = _build_run(tmp_path, "run-a", video_sha256=sha, budget=48, seed=0, points_valid=1000)
    run_b = _build_run(
        tmp_path,
        "run-b",
        selector="quality_nonredundant",
        video_sha256=sha,
        budget=48,
        seed=0,
        points_valid=800,
    )

    result = compare_runs(run_a, run_b, conditions_path)

    text = (exp_dir / "comparison.md").read_text(encoding="utf-8")
    assert result["verdict"] == "regression"
    assert "regression" in text
    assert INTERPRETATION_LIMITS in text
