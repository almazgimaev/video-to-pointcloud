"""Tests of the checks of the ingested result (`v3d.ingest_checks`).

Contracts: specs/001-video-3d-mvp/contracts/errors.md (case matrix and codes),
specs/001-video-3d-mvp/contracts/run-package.md §3, data-model.md §5.

What is protected here:

* the separation of code 7 (file missing) and 5 (file present but empty) — easy to merge
  by accident;
* partial frame registration is a normal outcome, not an error;
* the non-strict ``SHA256SUMS`` check after returning from the GPU machine: ``sparse/`` and
  ``points.ply`` written by upstream are not an integrity violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from v3d.errors import (
    ExitCode,
    PackageIntegrityError,
    ResultEmptyOrInvalidError,
    ResultIncompleteError,
    ResultMissingOrIncompatibleError,
)
from v3d.ingest_checks import (
    REQUIRED_RESULT_FILES,
    check_images_subset,
    check_model_not_empty,
    check_package_intact,
    check_result_files,
    check_run_id_matches,
)
from v3d.package import SHA256SUMS_NAME, write_sha256sums


@dataclass
class ModelStub:
    """Structural stub of a sparse model: the check needs only two fields.

    The real `v3d.colmap_io.SparseModel` is deliberately not imported — the checks are written
    against a structural type, and the test must use the same freedom.
    """

    cameras: list[Any] = field(default_factory=list)
    points_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))


def make_result(root: Path, *, empty: tuple[str, ...] = ()) -> Path:
    """Result directory with the full set of required files.

    Files listed in `empty` are created with zero size — this is what an interrupted
    computation looks like.
    """
    result = root / "result"
    for name in REQUIRED_RESULT_FILES:
        file = result / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"" if name in empty else b"COLMAP\x00")
    return result


def make_package(root: Path) -> Path:
    """The package as it goes off to the GPU machine: frames and `SHA256SUMS`."""
    package = root / "package"
    (package / "images").mkdir(parents=True)
    for number in range(3):
        (package / "images" / f"{number:04d}_000{number}00.jpg").write_bytes(
            b"jpeg" + bytes([number])
        )
    (package / "RUN_ON_GPU.txt").write_text("instructions", encoding="utf-8")
    write_sha256sums(package)
    return package


# --- 1. Presence of required result files -----------------------------------------------


def test_full_set_of_files_passes(tmp_path: Path) -> None:
    """contracts/errors.md: all required files are in place and non-empty — no refusal."""
    check_result_files(make_result(tmp_path))


@pytest.mark.parametrize("missing", REQUIRED_RESULT_FILES)
def test_missing_required_file_is_code_seven(tmp_path: Path, missing: str) -> None:
    """A file is missing entirely → code 7, and the message names the specific missing file."""
    result = make_result(tmp_path)
    (result / missing).unlink()

    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        check_result_files(result)

    assert refusal.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE, (
        "a missing result file is the case 'files missing or incompatible'"
    )
    assert missing in refusal.value.render(), (
        "a human needs the name of the specific file, not a statement that 'something is missing'"
    )


def test_zero_file_size_is_code_five_not_seven(tmp_path: Path) -> None:
    """A file exists but is empty → code 5: an interrupted computation, not a missing result.

    The separation of codes 5 and 7 is part of the contract, they must not be merged into one.
    """
    result = make_result(tmp_path, empty=("sparse/points3D.bin",))

    with pytest.raises(ResultIncompleteError) as refusal:
        check_result_files(result)

    assert refusal.value.exit_code == ExitCode.RESULT_INCOMPLETE, (
        "zero size is a sign of an interrupted computation (code 5), not a lost file (code 7)"
    )
    assert not isinstance(refusal.value, ResultMissingOrIncompatibleError)
    assert "sparse/points3D.bin" in refusal.value.render()


def test_missing_file_takes_precedence_over_empty_file(tmp_path: Path) -> None:
    """If one file is missing and another is empty, the loss is reported: no model without it."""
    result = make_result(tmp_path, empty=("sparse/points3D.bin",))
    (result / "sparse/cameras.bin").unlink()

    with pytest.raises(ResultMissingOrIncompatibleError):
        check_result_files(result)


# --- 2. Model image names ⊆ selected frames ---------------------------------------------


def test_subset_of_names_passes() -> None:
    """data-model.md §5: model names are among the selected frames — no refusal."""
    check_images_subset(
        ["0000_000100.jpg", "0001_000200.jpg"],
        ["0000_000100", "0001_000200", "0002_000300"],
    )


def test_partial_registration_is_not_an_error() -> None:
    """contracts/errors.md: part of the frames registered is the normal `partial` outcome."""
    check_images_subset(["0000_000100.jpg"], [f"{i:04d}_0001{i:02d}" for i in range(20)])


def test_match_without_extensions_passes() -> None:
    """Comparison is by `frame_id`, i.e. by name without extension."""
    check_images_subset(["0000_000100", "0001_000200"], ["0000_000100", "0001_000200"])


def test_extra_name_in_model_is_code_seven() -> None:
    """A name not among the selected frames → code 7: the result is from another set."""
    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        check_images_subset(
            ["0000_000100.jpg", "9999_999999.jpg"],
            ["0000_000100", "0001_000200"],
        )

    assert refusal.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE
    assert "9999_999999" in refusal.value.render(), (
        "the specific name is needed, not 'something extra'"
    )


def test_many_extra_names_are_collapsed_into_a_number() -> None:
    """The message has at most five names: the rest is given as a number."""
    extras = [f"{i:04d}_9999{i:02d}.jpg" for i in range(8)]

    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        check_images_subset(extras, ["0000_000100"])

    message = refusal.value.message
    assert message.count("_9999") == 5, "exactly five names are listed"
    assert "and 3 more" in message, "the rest are named by a number, not silently omitted"


# --- 3. Non-empty model -----------------------------------------------------------------


def test_model_with_cameras_and_points_passes() -> None:
    """Camera poses and points are present — the result is fit for ingest."""
    check_model_not_empty(
        ModelStub(cameras=["camera"], points_xyz=np.zeros((5, 3), dtype=np.float64))
    )


def test_model_without_cameras_is_code_six() -> None:
    """No camera poses → code 6, and it says exactly what is missing."""
    with pytest.raises(ResultEmptyOrInvalidError) as refusal:
        check_model_not_empty(ModelStub())

    assert refusal.value.exit_code == ExitCode.RESULT_EMPTY_OR_INVALID
    assert "camera poses" in refusal.value.message


def test_model_with_cameras_but_without_points_is_code_six() -> None:
    """Cameras present, no points → still code 6: the result is invalid."""
    with pytest.raises(ResultEmptyOrInvalidError) as refusal:
        check_model_not_empty(ModelStub(cameras=["camera"]))

    assert refusal.value.exit_code == ExitCode.RESULT_EMPTY_OR_INVALID
    assert "points" in refusal.value.message
    assert "camera poses" not in refusal.value.message, (
        "we do not report the absence of what is present"
    )


# --- 4. The same run --------------------------------------------------------------------


def test_matching_run_id_passes() -> None:
    """FR-029: artifacts of one run are accepted without remarks."""
    check_run_id_matches(
        {"run_id": "20260919-120000-abc"}, ("frames.json", {"run_id": "20260919-120000-abc"})
    )


def test_mismatching_run_id_is_code_seven() -> None:
    """An artifact of another run → code 7: mixing runs is forbidden."""
    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        check_run_id_matches(
            {"run_id": "20260919-120000-abc"},
            ("frames.json", {"run_id": "20260919-130000-zzz"}),
        )

    assert refusal.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE
    assert "frames.json" in refusal.value.render(), "we name the specific incompatible file"


def test_manifest_without_run_id_is_code_seven() -> None:
    """Without `run_id` in the manifest the check has no basis — a refusal, not a silent skip."""
    with pytest.raises(ResultMissingOrIncompatibleError) as refusal:
        check_run_id_matches({}, ("frames.json", {"run_id": "20260919-120000-abc"}))

    assert refusal.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE


# --- 5. Package integrity after return --------------------------------------------------


def test_untouched_package_passes(tmp_path: Path) -> None:
    """run-package.md §3: files from `SHA256SUMS` are in place and unchanged."""
    check_package_intact(make_package(tmp_path))


def test_files_added_by_upstream_do_not_violate_integrity(tmp_path: Path) -> None:
    """The key test of the non-strict mode (run-package.md §3).

    Upstream writes `sparse/` and `points.ply` into the directory it was given. The standard
    `shasum -a 256 -c` does not react to such files, and neither should our check.
    """
    package = make_package(tmp_path)
    (package / "sparse").mkdir()
    (package / "sparse" / "cameras.bin").write_bytes(b"COLMAP\x00")
    (package / "points.ply").write_text("ply\n", encoding="utf-8")

    check_package_intact(package)


def test_corrupted_package_file_is_code_four(tmp_path: Path) -> None:
    """A change to a file from `SHA256SUMS` → code 4 naming the specific file."""
    package = make_package(tmp_path)
    corrupted = package / "images" / "0001_000100.jpg"
    corrupted.write_text("different content", encoding="utf-8")

    with pytest.raises(PackageIntegrityError) as refusal:
        check_package_intact(package)

    assert refusal.value.exit_code == ExitCode.PACKAGE_INTEGRITY
    assert "images/0001_000100.jpg" in refusal.value.render()


def test_missing_package_file_is_code_four(tmp_path: Path) -> None:
    """A file from `SHA256SUMS` going missing is an integrity violation even in non-strict mode."""
    package = make_package(tmp_path)
    (package / "images" / "0002_000200.jpg").unlink()

    with pytest.raises(PackageIntegrityError) as refusal:
        check_package_intact(package)

    assert refusal.value.exit_code == ExitCode.PACKAGE_INTEGRITY
    assert "images/0002_000200.jpg" in refusal.value.render()


def test_package_without_sha256sums_is_code_four(tmp_path: Path) -> None:
    """`SHA256SUMS` itself is missing — nothing to check with, silently continuing is forbidden."""
    package = make_package(tmp_path)
    (package / SHA256SUMS_NAME).unlink()

    with pytest.raises(PackageIntegrityError) as refusal:
        check_package_intact(package)

    assert refusal.value.exit_code == ExitCode.PACKAGE_INTEGRITY
