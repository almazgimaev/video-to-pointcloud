"""Building the transfer package for the GPU machine and verifying its integrity.

Contract: specs/001-video-3d-mvp/contracts/run-package.md
Data model: specs/001-video-3d-mvp/data-model.md §1, §5

Package layout::

    package/
    ├── images/<frame_id>.jpg   # selected frames only
    ├── SHA256SUMS
    └── RUN_ON_GPU.txt

Why the subdirectory is called exactly ``images/``: this is the layout expected by the
upstream VGGT ``demo_colmap.py --scene_dir=<package>``
(specs/001-video-3d-mvp/contracts/reconstructor-io.md §1, §3). This is a **deliberate
fit to someone else's contract**, not an accident: it lets us avoid moving our code to the
GPU machine — only the directory with frames and the instructions for a human go there.

What is intentionally **not** put into the package:

* the source video — only the selected frames are sent to the GPU machine (run-package.md §4);
* ``manifest.json`` — the GPU machine does not need it, and an extra file in ``scene_dir``
  must not get into processing (run-package.md §1).

``SHA256SUMS`` is written in the format of the standard OS utilities (``shasum -a 256`` /
``sha256sum``), so a human can verify the package before running **without our code**
(data-model.md §5).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from v3d.artifacts import FrameRecord
from v3d.errors import InputUnusableError, PackageIntegrityError, RunDirExistsError

if TYPE_CHECKING:  # pragma: no cover - for annotations only, v3d.extract is not imported
    from v3d.extract import CandidateFrame

SHA256SUMS_NAME = "SHA256SUMS"
RUN_ON_GPU_NAME = "RUN_ON_GPU.txt"
IMAGES_DIR_NAME = "images"
IMAGE_SUFFIX = ".jpg"

#: Files the reconstructor must return (reconstructor-io.md §2).
EXPECTED_RESULT_FILES = (
    "sparse/cameras.bin",
    "sparse/images.bin",
    "sparse/points3D.bin",
    "sparse/points.ply",
)


_CHUNK = 1 << 20


# --- Checksums ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hexadecimal SHA-256 sum of a single file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_files(package_dir: Path) -> list[Path]:
    """Relative paths of all package files except ``SHA256SUMS`` itself, sorted.

    Sorting provides determinism: the same set of frames gives a byte-identical
    ``SHA256SUMS``.
    """
    files = [
        p.relative_to(package_dir)
        for p in package_dir.rglob("*")
        if p.is_file() and p.name != SHA256SUMS_NAME
    ]
    return sorted(files, key=lambda p: p.as_posix())


def write_sha256sums(package_dir: Path) -> Path:
    """Write ``package_dir/SHA256SUMS`` covering all package files except itself.

    The line format is ``<64 hex><two spaces><relative path>``, i.e. the text mode of the
    ``shasum -a 256`` and ``sha256sum`` utilities. Paths are relative and sorted, so
    the file can be checked with ``shasum -a 256 -c SHA256SUMS`` from the package directory.

    Returns:
        Path to the written ``SHA256SUMS`` file.
    """
    package_dir = Path(package_dir)
    lines = [
        f"{sha256_file(package_dir / rel)}  {rel.as_posix()}" for rel in _package_files(package_dir)
    ]
    target = package_dir / SHA256SUMS_NAME
    target.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    return target


def read_sha256sums(package_dir: Path) -> dict[str, str]:
    """Read ``SHA256SUMS`` into a "relative path → sum" mapping.

    Raises:
        PackageIntegrityError: the file is missing or a line cannot be parsed (exit code 4).
    """
    package_dir = Path(package_dir)
    target = package_dir / SHA256SUMS_NAME
    if not target.is_file():
        raise PackageIntegrityError(
            f"the package has no {SHA256SUMS_NAME} file: {package_dir}",
            details="the package was assembled incompletely or transferred incompletely",
        )
    expected: dict[str, str] = {}
    for lineno, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64:
            raise PackageIntegrityError(
                f"{SHA256SUMS_NAME}: line {lineno} cannot be parsed",
                details=f"expected format '<64 hex><two spaces><path>', got: {line!r}",
            )
        expected[name.lstrip("*")] = digest.lower()
    return expected


def verify_sha256sums(package_dir: Path, *, strict: bool = False) -> None:
    """Compare the package contents with ``SHA256SUMS`` (run-package.md §3).

    A modified or missing file listed in ``SHA256SUMS`` is always an error.
    Both kinds of mismatch are collected in one pass so the human sees the full picture at once.

    Why **non-strict** by default: the upstream ``demo_colmap.py --scene_dir=<package>`` writes
    its outputs (``sparse/``, ``points.ply``) into the same directory we passed to it.
    After a run, files that were not in ``SHA256SUMS`` legitimately appear in the package.
    The standard ``shasum -a 256 -c`` does not notice this — it checks only the listed files
    and stays green; our check should not be stricter than the OS utility without a reason.

    Args:
        package_dir: package directory.
        strict: additionally forbid extra files (not in ``SHA256SUMS``). Applied
            **only** right after the package is built, before it is moved to the GPU machine.

    Raises:
        PackageIntegrityError: a mismatch, listing the specific files (exit code
            4). Silently continuing is forbidden.
    """
    package_dir = Path(package_dir)
    expected = read_sha256sums(package_dir)
    present = {rel.as_posix(): rel for rel in _package_files(package_dir)}

    missing = sorted(set(expected) - set(present))
    changed = sorted(
        name
        for name, rel in present.items()
        if name in expected and sha256_file(package_dir / rel) != expected[name]
    )
    extra = sorted(set(present) - set(expected)) if strict else []
    if not (missing or extra or changed):
        return

    parts: list[str] = []
    if changed:
        parts.append("modified: " + ", ".join(changed))
    if missing:
        parts.append("missing: " + ", ".join(missing))
    if extra:
        parts.append(f"extra (not in {SHA256SUMS_NAME}): " + ", ".join(extra))
    raise PackageIntegrityError(
        f"package integrity violated: {package_dir}",
        details="\n".join(parts),
    )


# --- Instructions for a human ------------------------------------------------------------


def render_run_on_gpu(run_id: str, command_hint: str) -> str:
    """Text of ``RUN_ON_GPU.txt``: what to run on the GPU machine and what to copy back.

    The instructions are human-readable and deliberately self-contained: manual transfer
    and launch is a regular step of the process, not a hidden one (contracts/cli.md,
    run-package.md §2).
    """
    outputs = "\n".join(f"  - {name}" for name in EXPECTED_RESULT_FILES)
    return f"""Run the reconstruction on the GPU machine

run_id: {run_id}

Before running:

    Check free GPU memory first (`nvidia-smi`). Measured cost of the reconstructor:
    about 1 GiB + 0.27 GiB per frame.

1. Upstream command (run in the directory of the reconstructor clone):

    {command_hint}

   The directory of this package is passed as scene_dir; the images/ subdirectory already
   contains only the selected frames in the right orientation.

2. Expected output files (relative to the package directory):
{outputs}

3. What to copy back:
   - the whole sparse/ directory,
   - the run log (upstream stdout) as is.
   Put the copied files into runs/{run_id}/result/ (or pass them via v3d ingest --from).
   Do not mix result files from different runs.

4. Integrity check before running (optional, does not require our code):

    sha256sum -c {SHA256SUMS_NAME}
"""


# --- Building the package ---------------------------------------------------------------


def _source_paths(images_dir: Path, candidates: list[CandidateFrame]) -> dict[int, Path]:
    """Mapping "frame index in the video → candidate file in the working directory".

    A candidate path may be absolute, relative to the current directory (which is how
    frame extraction returns it with a relative ``--out``), or relative to ``images_dir``.
    Previously the third case was applied to the second and the path doubled, so the order
    is now: take the path as is if the file exists, and only otherwise resolve it against
    ``images_dir``.
    """
    mapping: dict[int, Path] = {}
    for candidate in candidates:
        path = Path(candidate.path)
        if not path.is_absolute() and not path.exists():
            path = images_dir / path
        mapping[candidate.src_index] = path
    return mapping


def _require_clean(package_dir: Path) -> None:
    """The package is built only into a clean directory.

    The directory may exist (``RunLayout.ensure_dirs`` creates ``package/images``
    in advance) but must not contain a single file: leftovers of a previous build would end
    up in ``scene_dir`` and in ``SHA256SUMS``. Foreign files are never deleted under any
    conditions — instead of a silent overwrite a refusal is raised.

    Raises:
        RunDirExistsError: the directory already contains files (exit code 8).
    """
    if not package_dir.exists():
        return
    existing = sorted(
        p.relative_to(package_dir).as_posix() for p in package_dir.rglob("*") if p.is_file()
    )
    if existing:
        raise RunDirExistsError(
            f"package directory is not empty: {package_dir}",
            details=(
                "the package is built only into a clean directory, foreign files are not deleted; "
                "remove them manually or specify another directory. Found: "
                + ", ".join(existing[:10])
                + (" …" if len(existing) > 10 else "")
            ),
        )


def build_package(
    images_dir: Path,
    records: list[FrameRecord],
    candidates: list[CandidateFrame],
    *,
    package_dir: Path,
    run_id: str,
    command_hint: str,
) -> dict[str, Any]:
    """Build the transfer package ``package/`` from the selected frames.

    **Only** records with ``selected=True`` are copied, under the name ``<frame_id>.jpg``.
    The source video and ``manifest.json`` do not go into the package (see the module docstring).
    The ``image_file`` field of every copied record is updated to the path inside the package.

    Args:
        images_dir: working directory of candidate frames (``runs/<run_id>/work/frames/``);
            relative candidate paths are resolved against it.
        records: decisions for all considered frames.
        candidates: candidates; linked to records by ``src_index``.
        package_dir: package directory; must be empty (see :func:`_require_clean`).
        run_id: run identifier, goes into ``RUN_ON_GPU.txt``.
        command_hint: exact upstream command line for ``RUN_ON_GPU.txt``.

    Returns:
        Summary: ``package_dir``, ``file_count``, ``total_bytes``, ``image_count``.

    Raises:
        RunDirExistsError: the package directory is not empty (exit code 8).
        InputUnusableError: a selected record has no frame file (exit code 2).
    """
    images_dir = Path(images_dir)
    package_dir = Path(package_dir)
    _require_clean(package_dir)

    sources = _source_paths(images_dir, candidates)
    images_out = package_dir / IMAGES_DIR_NAME
    images_out.mkdir(parents=True, exist_ok=True)

    selected = [r for r in records if r.selected]
    for record in selected:
        source = sources.get(record.src_index)
        if source is None or not source.is_file():
            raise InputUnusableError(
                f"no frame file for {record.frame_id} (src_index={record.src_index})",
                details=(
                    f"expected file {source}"
                    if source is not None
                    else f"no candidate with src_index={record.src_index} in the candidate list"
                ),
            )
        target = images_out / f"{record.frame_id}{IMAGE_SUFFIX}"
        shutil.copyfile(source, target)
        record.image_file = f"{IMAGES_DIR_NAME}/{target.name}"

    (package_dir / RUN_ON_GPU_NAME).write_text(
        render_run_on_gpu(run_id, command_hint), encoding="utf-8"
    )
    write_sha256sums(package_dir)

    files = [package_dir / rel for rel in _package_files(package_dir)]
    total_bytes = sum(p.stat().st_size for p in files)
    sums = package_dir / SHA256SUMS_NAME
    return {
        "package_dir": str(package_dir),
        "file_count": len(files) + 1,
        "total_bytes": total_bytes + sums.stat().st_size,
        "image_count": len(selected),
    }
