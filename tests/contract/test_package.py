"""Contract tests for the ``package/`` transfer package.

Sources of requirements:

* specs/001-video-3d-mvp/contracts/run-package.md (package contents, prohibitions, verification);
* specs/001-video-3d-mvp/data-model.md §1, §5 (layout, integrity);
* specs/001-video-3d-mvp/contracts/reconstructor-io.md §1 (the ``images/`` subdirectory);
* specs/001-video-3d-mvp/contracts/cli.md (GPU machine instructions, exit code 4).

The tests protect promises, not the implementation: the package contents, the absence of
the video and the manifest, the ``SHA256SUMS`` format, suitability of the file for the
standard OS utility, both integrity check modes (non-strict by default, strict right after
the build), failure with exit code 4 on mismatch, and determinism of a rebuild.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from v3d.artifacts import FrameRecord
from v3d.errors import ExitCode, InputUnusableError, PackageIntegrityError, RunDirExistsError
from v3d.package import (
    RUN_ON_GPU_NAME,
    SHA256SUMS_NAME,
    build_package,
    verify_sha256sums,
    write_sha256sums,
)

RUN_ID = "20260101-120000_abc123"
COMMAND_HINT = "python demo_colmap.py --scene_dir=/path/to/package --use_ba"

SUMS_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


@dataclass(frozen=True)
class _Candidate:
    """Stub for ``v3d.extract.CandidateFrame``: the package needs only these three fields."""

    src_index: int
    timestamp_us: int
    path: Path


def _make_frames(tmp_path: Path, count: int = 5) -> tuple[Path, list[_Candidate]]:
    """Candidate frames in the working directory: ``.jpg`` files distinguishable by content."""
    work = tmp_path / "work" / "frames"
    work.mkdir(parents=True)
    candidates: list[_Candidate] = []
    for i in range(count):
        src_index = 12 + i * 35
        path = work / f"cand_{src_index:06d}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + f"frame-{src_index}".encode() * 4)
        candidates.append(
            _Candidate(src_index=src_index, timestamp_us=src_index * 33333, path=path)
        )
    return work, candidates


def _make_records(candidates: list[_Candidate], selected_idx: set[int]) -> list[FrameRecord]:
    """Selection records: selected ones get a ``frame_id`` with a sequence number."""
    records: list[FrameRecord] = []
    seq = 0
    for i, cand in enumerate(candidates):
        selected = i in selected_idx
        frame_id = f"{seq:04d}_{cand.src_index:06d}" if selected else f"----_{cand.src_index:06d}"
        if selected:
            seq += 1
        records.append(
            FrameRecord(
                frame_id=frame_id,
                src_index=cand.src_index,
                timestamp_us=cand.timestamp_us,
                sharpness=100.0 + i,
                diff_to_prev_selected=None if not selected else 0.2,
                selected=selected,
                reject_reason=None if selected else "over frame budget",
            )
        )
    return records


def _simulate_upstream_result(package_dir: Path) -> list[str]:
    """Simulate a return from the GPU machine: upstream writes outputs into the given scene_dir."""
    (package_dir / "sparse").mkdir(parents=True, exist_ok=True)
    (package_dir / "sparse" / "cameras.bin").write_bytes(b"\x00cameras")
    (package_dir / "points.ply").write_bytes(b"ply\n")
    return ["sparse/cameras.bin", "points.ply"]


@pytest.fixture
def built(tmp_path: Path):
    """A built package: 5 candidates, 3 selected."""
    work, candidates = _make_frames(tmp_path)
    records = _make_records(candidates, selected_idx={0, 2, 4})
    package_dir = tmp_path / "runs" / RUN_ID / "package"
    summary = build_package(
        work,
        records,
        candidates,
        package_dir=package_dir,
        run_id=RUN_ID,
        command_hint=COMMAND_HINT,
    )
    return package_dir, records, summary


# --- Package contents ---------------------------------------------------------------------


def test_package_layout_matches_upstream_contract(built) -> None:
    """The package consists of exactly images/, SHA256SUMS and RUN_ON_GPU.txt."""
    package_dir, _, _ = built
    assert (package_dir / "images").is_dir()
    assert (package_dir / SHA256SUMS_NAME).is_file()
    assert (package_dir / RUN_ON_GPU_NAME).is_file()
    assert sorted(p.name for p in package_dir.iterdir()) == [
        RUN_ON_GPU_NAME,
        SHA256SUMS_NAME,
        "images",
    ]


def test_only_selected_frames_with_frame_id_names(built) -> None:
    """images/ holds only the selected frames, the file name equals frame_id."""
    package_dir, records, _ = built
    selected = [r for r in records if r.selected]
    images = sorted(p.name for p in (package_dir / "images").iterdir())
    assert images == sorted(f"{r.frame_id}.jpg" for r in selected)
    assert len(images) == len(selected)


def test_rejected_frames_absent(built) -> None:
    """Unselected frames do not get into the package."""
    package_dir, records, _ = built
    for record in records:
        if not record.selected:
            assert not (package_dir / "images" / f"{record.frame_id}.jpg").exists()


def test_no_manifest_and_no_source_video(built) -> None:
    """The package has no manifest.json and no source video (run-package.md §1, §4)."""
    package_dir, _, _ = built
    names = {p.name for p in package_dir.rglob("*") if p.is_file()}
    assert "manifest.json" not in names
    assert not any(n.lower().endswith((".mp4", ".mov")) for n in names)
    suffixes = {p.suffix for p in (package_dir / "images").iterdir()}
    assert suffixes == {".jpg"}


def test_summary_reports_files_size_and_path(built) -> None:
    """Summary: file count, total size in bytes, package path."""
    package_dir, records, summary = built
    files = [p for p in package_dir.rglob("*") if p.is_file()]
    assert summary["package_dir"] == str(package_dir)
    assert summary["file_count"] == len(files)
    assert summary["total_bytes"] == sum(p.stat().st_size for p in files)
    assert summary["image_count"] == len([r for r in records if r.selected])


# --- SHA256SUMS ---------------------------------------------------------------------------


def test_sha256sums_format_and_coverage(built) -> None:
    """Line format and coverage: all package files except SHA256SUMS itself."""
    package_dir, _, _ = built
    lines = (package_dir / SHA256SUMS_NAME).read_text(encoding="utf-8").splitlines()
    parsed = {}
    for line in lines:
        match = SUMS_LINE.match(line)
        assert match is not None, f"line is not in shasum format: {line!r}"
        parsed[match.group(2)] = match.group(1)

    expected = {
        p.relative_to(package_dir).as_posix()
        for p in package_dir.rglob("*")
        if p.is_file() and p.name != SHA256SUMS_NAME
    }
    assert set(parsed) == expected
    for rel, digest in parsed.items():
        actual = hashlib.sha256((package_dir / rel).read_bytes()).hexdigest()
        assert actual == digest, f"sum mismatch for {rel}"


def test_sha256sums_paths_are_relative_and_sorted(built) -> None:
    """Paths are relative and sorted — otherwise there is no determinism."""
    package_dir, _, _ = built
    names = [
        SUMS_LINE.match(line).group(2)
        for line in (package_dir / SHA256SUMS_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert names == sorted(names)
    assert all(not Path(n).is_absolute() for n in names)


@pytest.mark.skipif(
    shutil.which("shasum") is None, reason="the shasum utility is not available on this system"
)
def test_sha256sums_checked_by_os_utility(built) -> None:
    """A human must be able to verify the package with the standard OS utility, without our code."""
    package_dir, _, _ = built
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", SHA256SUMS_NAME],
        cwd=package_dir,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"shasum -c exited with {proc.returncode}:\n{proc.stdout}{proc.stderr}"
    )


# --- Integrity check -----------------------------------------------------------------------


def test_verify_passes_on_untouched_package(built) -> None:
    package_dir, _, _ = built
    verify_sha256sums(package_dir)


@pytest.mark.parametrize("strict", [False, True], ids=["non-strict", "strict"])
def test_verify_detects_modified_file(built, strict: bool) -> None:
    """A modified file from SHA256SUMS → code 4 with its name in both modes."""
    package_dir, records, _ = built
    victim = next(r for r in records if r.selected)
    rel = f"images/{victim.frame_id}.jpg"
    (package_dir / rel).write_bytes(b"\xff\xd8broken")

    with pytest.raises(PackageIntegrityError) as excinfo:
        verify_sha256sums(package_dir, strict=strict)
    assert excinfo.value.exit_code == ExitCode.PACKAGE_INTEGRITY == 4
    assert rel in excinfo.value.render()


@pytest.mark.parametrize("strict", [False, True], ids=["non-strict", "strict"])
def test_verify_detects_missing_file(built, strict: bool) -> None:
    """A missing file from SHA256SUMS → code 4 with its name in both modes."""
    package_dir, records, _ = built
    victim = next(r for r in records if r.selected)
    rel = f"images/{victim.frame_id}.jpg"
    (package_dir / rel).unlink()

    with pytest.raises(PackageIntegrityError) as excinfo:
        verify_sha256sums(package_dir, strict=strict)
    assert excinfo.value.exit_code == ExitCode.PACKAGE_INTEGRITY == 4
    assert rel in excinfo.value.render()


def test_verify_tolerates_upstream_outputs_by_default(built) -> None:
    """Upstream outputs in the same directory are not an error: it writes to scene_dir."""
    package_dir, _, _ = built
    _simulate_upstream_result(package_dir)
    verify_sha256sums(package_dir)


def test_verify_strict_refuses_extra_files(built) -> None:
    """Strict mode (right after the build, before transfer) forbids extra files."""
    package_dir, _, _ = built
    extra = _simulate_upstream_result(package_dir)

    with pytest.raises(PackageIntegrityError) as excinfo:
        verify_sha256sums(package_dir, strict=True)
    assert excinfo.value.exit_code == ExitCode.PACKAGE_INTEGRITY == 4
    rendered = excinfo.value.render()
    for rel in extra:
        assert rel in rendered


@pytest.mark.skipif(
    shutil.which("shasum") is None, reason="the shasum utility is not available on this system"
)
def test_os_utility_stays_green_after_upstream_run(built) -> None:
    """Rationale for the non-strict default: the standard utility does not notice extra files."""
    package_dir, _, _ = built
    _simulate_upstream_result(package_dir)
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", SHA256SUMS_NAME],
        cwd=package_dir,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"shasum -c exited with {proc.returncode}:\n{proc.stdout}{proc.stderr}"
    )


def test_verify_requires_sha256sums(built) -> None:
    """No SHA256SUMS → code 4, no silent continuation."""
    package_dir, _, _ = built
    (package_dir / SHA256SUMS_NAME).unlink()
    with pytest.raises(PackageIntegrityError) as excinfo:
        verify_sha256sums(package_dir)
    assert excinfo.value.exit_code == ExitCode.PACKAGE_INTEGRITY


# --- RUN_ON_GPU.txt -------------------------------------------------------------------------


def test_run_on_gpu_contains_run_id_command_and_outputs(built) -> None:
    package_dir, _, _ = built
    text = (package_dir / RUN_ON_GPU_NAME).read_text(encoding="utf-8")
    assert RUN_ID in text
    assert COMMAND_HINT in text
    for name in (
        "sparse/cameras.bin",
        "sparse/images.bin",
        "sparse/points3D.bin",
        "sparse/points.ply",
    ):
        assert name in text
    assert "sparse/" in text
    assert "sha256sum -c SHA256SUMS" in text


def test_run_on_gpu_states_gpu_memory_hint(built) -> None:
    """The instructions tell the operator to check free GPU memory and give the measured cost."""
    package_dir, _, _ = built
    text = (package_dir / RUN_ON_GPU_NAME).read_text(encoding="utf-8")
    assert "nvidia-smi" in text
    assert "0.27 GiB per frame" in text


# --- Determinism and refusals ----------------------------------------------------------------


def test_rebuild_is_deterministic(tmp_path: Path) -> None:
    """Rebuilding the same set gives a byte-identical SHA256SUMS."""
    work, candidates = _make_frames(tmp_path)
    selected = {0, 2, 4}
    sums = []
    for n in ("a", "b"):
        package_dir = tmp_path / "out" / n / "package"
        build_package(
            work,
            _make_records(candidates, selected),
            candidates,
            package_dir=package_dir,
            run_id=RUN_ID,
            command_hint=COMMAND_HINT,
        )
        sums.append((package_dir / SHA256SUMS_NAME).read_bytes())
    assert sums[0] == sums[1]


def test_write_sha256sums_is_idempotent(built) -> None:
    """A repeated write_sha256sums call neither changes the file nor includes it in itself."""
    package_dir, _, _ = built
    before = (package_dir / SHA256SUMS_NAME).read_bytes()
    write_sha256sums(package_dir)
    assert (package_dir / SHA256SUMS_NAME).read_bytes() == before
    assert SHA256SUMS_NAME not in before.decode("utf-8")


def test_missing_source_frame_is_refused(tmp_path: Path) -> None:
    """No source frame file → refusal, the package is not built silently."""
    work, candidates = _make_frames(tmp_path)
    records = _make_records(candidates, selected_idx={0, 2})
    candidates[2].path.unlink()

    with pytest.raises(InputUnusableError) as excinfo:
        build_package(
            work,
            records,
            candidates,
            package_dir=tmp_path / "package",
            run_id=RUN_ID,
            command_hint=COMMAND_HINT,
        )
    assert records[2].frame_id in excinfo.value.render()


def test_non_empty_target_dir_is_refused(tmp_path: Path) -> None:
    """The package is built only into a clean directory: foreign files are never overwritten."""
    work, candidates = _make_frames(tmp_path)
    records = _make_records(candidates, selected_idx={0})
    package_dir = tmp_path / "package"
    (package_dir / "images").mkdir(parents=True)
    (package_dir / "images" / "old.jpg").write_bytes(b"old")

    with pytest.raises(RunDirExistsError) as excinfo:
        build_package(
            work,
            records,
            candidates,
            package_dir=package_dir,
            run_id=RUN_ID,
            command_hint=COMMAND_HINT,
        )
    assert excinfo.value.exit_code == ExitCode.RUN_DIR_EXISTS
    assert (package_dir / "images" / "old.jpg").read_bytes() == b"old"


def test_empty_dirs_do_not_block_build(tmp_path: Path) -> None:
    """RunLayout.ensure_dirs creates package/images in advance — this does not hinder the build."""
    work, candidates = _make_frames(tmp_path)
    records = _make_records(candidates, selected_idx={1})
    package_dir = tmp_path / "package"
    (package_dir / "images").mkdir(parents=True)

    summary = build_package(
        work,
        records,
        candidates,
        package_dir=package_dir,
        run_id=RUN_ID,
        command_hint=COMMAND_HINT,
    )
    assert summary["image_count"] == 1
    verify_sha256sums(package_dir)


def test_relative_candidate_path_is_not_doubled(tmp_path, monkeypatch):
    """A relative frame path is taken as is; regression of doubling with --out runs."""
    from v3d.artifacts import FrameRecord
    from v3d.package import build_package

    monkeypatch.chdir(tmp_path)
    work = Path("runs/r1/work/frames")
    work.mkdir(parents=True)
    (work / "cand_000000.jpg").write_bytes(b"\xff\xd8\xff\xdb frame")

    class _Candidate:
        def __init__(self, src_index: int, path: Path) -> None:
            self.src_index = src_index
            self.timestamp_us = 0
            self.path = path

    records = [
        FrameRecord(
            frame_id="0000_000000",
            src_index=0,
            timestamp_us=0,
            sharpness=1.0,
            diff_to_prev_selected=None,
            selected=True,
        )
    ]
    summary = build_package(
        work,
        records,
        [_Candidate(0, work / "cand_000000.jpg")],
        package_dir=Path("runs/r1/package"),
        run_id="r1",
        command_hint="python demo_colmap.py --scene_dir=<package>",
    )
    assert summary["image_count"] == 1
    assert (Path("runs/r1/package/images/0000_000000.jpg")).exists()
