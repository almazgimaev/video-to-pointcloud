"""Run directory: path layout, creation without silent overwrite, stage timing.

Data model: specs/001-video-3d-mvp/data-model.md §1–§2

No-overwrite rule: the ``runs/<run_id>/`` directory is created only if it does not exist.
Repeated preparation with the same parameters leaves an existing run untouched; overwriting
is possible only with an explicit ``--force``, and then the fact is recorded in the manifest
as ``forced_overwrite: true``. Foreign files are never deleted under any conditions.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from v3d.errors import ResultMissingOrIncompatibleError, RunDirExistsError

MANIFEST_NAME = "manifest.json"
FRAMES_NAME = "frames.json"
REPORT_NAME = "report.md"


@dataclass(frozen=True)
class RunLayout:
    """Paths inside the run directory. Paths only: creates and reads nothing."""

    run_id: str
    root: Path
    package: Path
    package_images: Path
    package_sha256sums: Path
    package_run_on_gpu: Path
    result: Path
    normalized: Path
    export: Path
    view: Path
    manifest: Path
    frames: Path
    report: Path

    def ensure_dirs(self) -> None:
        """Creates the run subdirectories. Does not clear existing directories."""
        for directory in (
            self.root,
            self.package,
            self.package_images,
            self.result,
            self.normalized,
            self.export,
            self.view,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def run_dir_layout(root: Path | str, run_id: str) -> RunLayout:
    """Layout of the run directory ``<root>/<run_id>/`` per data-model.md §1."""
    run_root = Path(root) / run_id
    package = run_root / "package"
    return RunLayout(
        run_id=run_id,
        root=run_root,
        package=package,
        package_images=package / "images",
        package_sha256sums=package / "SHA256SUMS",
        package_run_on_gpu=package / "RUN_ON_GPU.txt",
        result=run_root / "result",
        normalized=run_root / "normalized",
        export=run_root / "export",
        view=run_root / "view",
        manifest=run_root / MANIFEST_NAME,
        frames=run_root / FRAMES_NAME,
        report=run_root / REPORT_NAME,
    )


def create_run_dir(root: Path | str, run_id: str, *, force: bool = False) -> tuple[RunLayout, bool]:
    """Creates the run directory and returns the layout and a forced-overwrite flag.

    The returned flag is ``True`` only if the directory already existed and work
    continued because of ``--force``; it is written to the manifest as ``forced_overwrite``.

    Raises:
        RunDirExistsError: the directory exists and ``--force`` is not set (exit code 8).
    """
    layout = run_dir_layout(root, run_id)
    existed = layout.root.exists()
    if existed and not force:
        raise RunDirExistsError(
            f"run directory already exists: {layout.root}",
            details=(
                "repeated preparation does not overwrite a finished run; "
                "repeat with --force to write into this directory, "
                "or specify a different --out"
            ),
        )
    layout.ensure_dirs()
    return layout, existed and force


def find_run_dir(path: Path | str) -> RunLayout:
    """Layout of an existing run directory from its path.

    ``run_id`` is taken from the directory name; the presence of the manifest is checked,
    its content is not: that is the concern of the code that reads artifacts.

    Raises:
        ResultMissingOrIncompatibleError: the directory is missing or has no manifest (code 7).
    """
    run_root = Path(path)
    if not run_root.is_dir():
        raise ResultMissingOrIncompatibleError(
            f"run directory not found: {run_root}",
            details="specify a path like runs/<run_id>",
        )
    layout = run_dir_layout(run_root.parent, run_root.name)
    if not layout.manifest.is_file():
        raise ResultMissingOrIncompatibleError(
            f"run directory has no {MANIFEST_NAME}: {run_root}",
            details="this is not a run directory or the preparation stage did not finish",
        )
    return layout


@contextmanager
def stage_timer(durations: MutableMapping[str, float], stage: str) -> Iterator[None]:
    """Records the stage duration in seconds into ``diagnostics.stage_durations_s``.

    The duration is recorded on exceptions too: an interrupted stage also cost time.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        durations[stage] = round(time.perf_counter() - started, 3)
