"""Candidate extraction with the real ffmpeg.

The video is generated in the test (`testsrc`, 1–2 seconds), there are no external files.
Tests are skipped if ffmpeg or ffprobe are missing from PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from v3d.extract import extract_candidates
from v3d.probe import probe_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required in PATH",
)

FPS = 10
DURATION_S = 2
TOTAL_FRAMES = FPS * DURATION_S


def _make_video(path: Path, *, size: str = "320x240") -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}:rate={FPS}:duration={DURATION_S}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _make_rotated_video(source: Path, target: Path, degrees: int) -> Path:
    """Stream copy with a display matrix: rotation goes into metadata, pixels are untouched."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-display_rotation",
            str(degrees),
            "-i",
            str(source),
            "-c",
            "copy",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_video(tmp_path_factory.mktemp("video") / "clip.mp4")


def test_probe_reads_real_file(video: Path) -> None:
    info = probe_video(video)
    assert (info.width, info.height) == (320, 240)
    assert info.video_codec == "h264"
    assert info.container in {"mov", "mp4"}
    assert info.avg_fps == pytest.approx(FPS)
    assert info.duration_s == pytest.approx(DURATION_S, abs=0.2)
    assert info.rotation_deg == 0
    assert info.supported is True
    assert info.size_bytes == video.stat().st_size


def test_candidates_are_subset_with_constant_stride(video: Path, tmp_path: Path) -> None:
    info = probe_video(video)
    candidates, transform = extract_candidates(
        video, tmp_path / "frames", info=info, max_candidates=5, long_side=None
    )

    assert 0 < len(candidates) <= 5
    assert transform["candidates_total"] == len(candidates)
    stride = transform["candidate_stride"]
    assert stride == 4  # ceil(20 / 5)

    indices = [c.src_index for c in candidates]
    assert indices == sorted(indices)
    assert indices[0] == 0
    steps = {b - a for a, b in zip(indices, indices[1:], strict=False)}
    assert steps == {stride}
    assert all(c.path.is_file() for c in candidates)
    assert [c.path.name for c in candidates] == [f"cand_{i:06d}.jpg" for i in indices]


def test_timestamps_increase_and_fit_duration(video: Path, tmp_path: Path) -> None:
    info = probe_video(video)
    candidates, transform = extract_candidates(
        video, tmp_path / "frames", info=info, max_candidates=5, long_side=None
    )

    stamps = [c.timestamp_us for c in candidates]
    assert stamps[0] == 0
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))
    assert max(stamps) <= info.duration_s * 1_000_000
    assert transform["timestamp_source"] == "container_pts"

    # Timestamps are actual: the candidate step is stride frames of 1/FPS seconds each.
    expected_step_us = round(transform["candidate_stride"] / FPS * 1_000_000)
    assert stamps[1] - stamps[0] == pytest.approx(expected_step_us, abs=2000)


def test_resize_by_long_side(video: Path, tmp_path: Path) -> None:
    info = probe_video(video)
    candidates, transform = extract_candidates(
        video, tmp_path / "frames", info=info, max_candidates=3, long_side=160
    )

    assert transform["resize_mode"] == "long_side"
    assert transform["source_size"] == [320, 240]
    assert transform["resized_size"] == [160, 120]
    assert transform["scale_factor"] == pytest.approx(0.5)
    assert transform["crop"] is None
    for candidate in candidates:
        with Image.open(candidate.path) as image:
            assert image.size == (160, 120)


def test_no_resize_keeps_source_size(video: Path, tmp_path: Path) -> None:
    info = probe_video(video)
    _, transform = extract_candidates(
        video, tmp_path / "frames", info=info, max_candidates=3, long_side=None
    )
    assert transform["resize_mode"] == "none"
    assert transform["resized_size"] == [320, 240]
    assert transform["scale_factor"] == pytest.approx(1.0)


def test_extraction_is_deterministic(video: Path, tmp_path: Path) -> None:
    info = probe_video(video)
    first, transform_a = extract_candidates(
        video, tmp_path / "a", info=info, max_candidates=6, long_side=200
    )
    second, transform_b = extract_candidates(
        video, tmp_path / "b", info=info, max_candidates=6, long_side=200
    )

    assert transform_a == transform_b
    assert [(c.src_index, c.timestamp_us, c.path.name) for c in first] == [
        (c.src_index, c.timestamp_us, c.path.name) for c in second
    ]
    assert [c.path.read_bytes() for c in first] == [c.path.read_bytes() for c in second]


def test_rotation_metadata_applied_to_saved_frames(video: Path, tmp_path: Path) -> None:
    rotated = _make_rotated_video(video, tmp_path / "rotated.mp4", 90)
    info = probe_video(rotated)
    if info.rotation_deg != 90:
        pytest.skip(
            "the installed ffmpeg did not write a 90° rotation into the display matrix: "
            f"probe_video returned rotation_deg={info.rotation_deg}"
        )

    assert (info.width, info.height) == (320, 240)  # stream dimensions, before rotation
    candidates, transform = extract_candidates(
        rotated, tmp_path / "frames", info=info, max_candidates=3, long_side=None
    )

    assert transform["rotation_applied_deg"] == 90
    assert transform["source_size"] == [240, 320]  # sides swapped
    for candidate in candidates:
        with Image.open(candidate.path) as image:
            assert image.size == (240, 320)
