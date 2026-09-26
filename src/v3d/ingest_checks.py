"""Checks of the result returned from the GPU machine.

Contracts: specs/001-video-3d-mvp/contracts/errors.md (case matrix and exit codes),
specs/001-video-3d-mvp/contracts/run-package.md §3 (integrity check),
data model: specs/001-video-3d-mvp/data-model.md §5.

Each check is a separate function with a single responsibility, so that the calling code
(`v3d ingest`) reads as a list of checks rather than one branching function.

The separation of exit codes here is essential and deliberate:

* 7 — a required file is missing entirely, or the result belongs to another run or
  another set of frames (files are missing or incompatible);
* 5 — files are in place, but some of them are empty: a sign of an interrupted computation;
* 6 — the model was read, but it has no camera poses or no points;
* 4 — package checksums do not match.

Message wording is subject to FR-007: we describe the observation and what to do about it,
without establishing a cause.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from v3d.artifacts import require_same_run_id
from v3d.errors import (
    ResultEmptyOrInvalidError,
    ResultIncompleteError,
    ResultMissingOrIncompatibleError,
)
from v3d.package import verify_sha256sums

#: Files without which the sparse model cannot be read (reconstructor-io.md §2).
#: ``points.ply`` is not included: it is optional — the cloud is built from ``points3D.bin``.
REQUIRED_RESULT_FILES: tuple[str, ...] = (
    "sparse/cameras.bin",
    "sparse/images.bin",
    "sparse/points3D.bin",
)

#: How many names to list in a message in full; the rest are collapsed into a number.
MAX_NAMES_IN_MESSAGE = 5


class SparseModelLike(Protocol):
    """Structural type of a sparse model: only cameras and points are needed.

    A protocol rather than an import of ``v3d.colmap_io.SparseModel``, deliberately: the check
    needs only two fields, and a hard link to the COLMAP reading module would force pulling
    ``pycolmap`` into the tests of these checks.
    """

    cameras: Any
    points_xyz: Any


def _format_names(names: list[str]) -> str:
    """List of names: at most :data:`MAX_NAMES_IN_MESSAGE`, the remainder as a number."""
    head = ", ".join(names[:MAX_NAMES_IN_MESSAGE])
    remainder = len(names) - MAX_NAMES_IN_MESSAGE
    return head if remainder <= 0 else f"{head} and {remainder} more"


def _is_empty(value: Any) -> bool:
    """Whether the value is empty: ``None`` or a zero-length container.

    Works both for a list of cameras and for a numpy array ``(0, 3)``.
    """
    if value is None:
        return True
    try:
        return len(value) == 0
    except TypeError:  # pragma: no cover - the model is expected to have sized fields
        return False


def check_result_files(result_dir: Path) -> None:
    """Checks presence and non-zero size of the required result files.

    Args:
        result_dir: directory with the returned result (``runs/<run_id>/result/``).

    Raises:
        ResultMissingOrIncompatibleError: a required file is missing entirely (exit code 7).
        ResultIncompleteError: the file exists but has zero size (exit code 5).
    """
    result_dir = Path(result_dir)
    missing = [name for name in REQUIRED_RESULT_FILES if not (result_dir / name).is_file()]
    if missing:
        raise ResultMissingOrIncompatibleError(
            f"the result lacks required files: {_format_names(missing)}",
            details=(
                f"expected relative to {result_dir}: {', '.join(REQUIRED_RESULT_FILES)}; "
                "perhaps the result directory was not fully transferred from the GPU machine"
            ),
        )

    empty = [name for name in REQUIRED_RESULT_FILES if (result_dir / name).stat().st_size == 0]
    if empty:
        raise ResultIncompleteError(
            f"result files are empty: {_format_names(empty)}",
            details=(
                f"the files exist in {result_dir} but have zero size — this is what an "
                "interrupted computation looks like; the result is considered incomplete, "
                "repeat the run on the GPU machine"
            ),
        )


def check_images_subset(model_names: list[str], selected_frame_ids: list[str]) -> None:
    """Model image names must be a subset of the selected frames (data-model.md §5).

    Comparison is by ``frame_id``, i.e. by file name without extension.

    Fewer names in the model than selected frames is **not** an error: it is partial
    registration, a normal outcome (`status = partial` in contracts/errors.md).

    Args:
        model_names: image names from the model (with or without extension).
        selected_frame_ids: identifiers of the selected frames from ``frames.json``.

    Raises:
        ResultMissingOrIncompatibleError: the model has a name that is not among the selected
            frames (exit code 7).
    """
    selected = set(selected_frame_ids)
    unexpected = sorted({Path(name).stem for name in model_names} - selected)
    if unexpected:
        raise ResultMissingOrIncompatibleError(
            f"the model has images that are not among the selected frames: "
            f"{_format_names(unexpected)}",
            details=(
                f"frames selected: {len(selected)}, names in the model: {len(model_names)}; "
                "the result belongs to another run or another set of frames — "
                "they must not be mixed"
            ),
        )


def check_model_not_empty(model: SparseModelLike) -> None:
    """Checks that the model has both camera poses and points.

    Args:
        model: object with ``cameras`` and ``points_xyz`` fields.

    Raises:
        ResultEmptyOrInvalidError: no cameras or no points, stating exactly what is missing
            (exit code 6).
    """
    absent: list[str] = []
    if _is_empty(getattr(model, "cameras", None)):
        absent.append("camera poses")
    if _is_empty(getattr(model, "points_xyz", None)):
        absent.append("points")
    if absent:
        raise ResultEmptyOrInvalidError(
            "the result was read, but it has no " + " and no ".join(absent),
            details=(
                "the run status is not 'success'; stage diagnostics are kept, "
                "exporting and viewing such a result makes no sense"
            ),
        )


def check_run_id_matches(manifest_payload: dict, *payloads: tuple[str, dict]) -> None:
    """Checks that all artifacts belong to one run (FR-029, data-model.md §5).

    A thin wrapper over :func:`v3d.artifacts.require_same_run_id`: the reference ``run_id``
    is taken from the manifest, so that at the call site it reads as one more ingest check.

    Args:
        manifest_payload: contents of the run's ``manifest.json``.
        payloads: "file name → contents" pairs of the other artifacts.

    Raises:
        ResultMissingOrIncompatibleError: the manifest has no ``run_id`` or it does not match
            (exit code 7).
    """
    expected = manifest_payload.get("run_id")
    if expected is None:
        raise ResultMissingOrIncompatibleError(
            "the run manifest has no run_id field",
            details="without it one cannot verify that the artifacts belong to one run",
        )
    require_same_run_id(expected, *payloads)


def check_package_intact(package_dir: Path) -> None:
    """Verifies the package against ``SHA256SUMS`` **non-strictly** (run-package.md §3).

    The non-strict mode is not an oversight and must not be "fixed" into a strict one: upstream
    ``demo_colmap.py --scene_dir=<package>`` writes its outputs (``sparse/``, ``points.ply``)
    into the same directory we gave it, so after returning from the GPU machine
    files that are not in ``SHA256SUMS`` are expected. The contract explicitly requires behaving
    like the standard ``shasum -a 256 -c``: it checks only the listed files and stays
    green. Strict mode is applied only right after the package is built, before transfer.

    Args:
        package_dir: package directory returned from the GPU machine.

    Raises:
        PackageIntegrityError: a file from ``SHA256SUMS`` was changed or is missing (exit code 4).
    """
    verify_sha256sums(Path(package_dir), strict=False)
