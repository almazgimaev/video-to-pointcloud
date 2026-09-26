"""Extraction of candidate frames from a video (`v3d prepare`, stage 1).

**Not all frames** are taken from the video, but a subset with a constant stride: the
stride is chosen so that the number of candidates does not exceed `max_candidates`.
Selection among the candidates is the job of `v3d.select`; this module only extracts
frames and honestly describes what was done to them.

Timestamps (FR-005). `timestamp_us` is taken from the actual frame timestamps of the
container, not computed by dividing the index by fps. The mapping works like this:

1. `ffprobe -show_entries frame=best_effort_timestamp_time,pts_time` lists the frames
   of the video stream in display order; the position in this list is the `src_index`
   (the order is checked for non-decreasing timestamps; on violation the list is sorted
   by timestamp).
2. The `select=not(mod(n,stride))` filter in ffmpeg counts `n` over the same input
   frames in the same display order, so the k-th extracted frame is `src_index = k*stride`.
3. `timestamp_us` = the timestamp of this `src_index` minus the timestamp of the first
   frame, in microseconds.

If the container does not provide actual timestamps for at least one candidate, the
module refuses to work (`InputUnusableError`) instead of substituting computed time:
a substituted time would be indistinguishable from a measured one.

Orientation (FR-004). When re-encoding, ffmpeg applies the rotation from the display
matrix itself. The module does not take the flag on faith: it decodes one frame and
compares the actual size with the one expected from metadata. `rotation_applied_deg`
records the rotation confirmed by this check; `source_size` is the actually observed
frame size after rotation and before resize.

Data model: specs/001-video-3d-mvp/data-model.md §3.2, §3.3.
Interface: docs/internal-interfaces.md.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from v3d.errors import InputUnusableError
from v3d.probe import VideoInfo

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
_TIMEOUT_S = 1800
_STDERR_TAIL_CHARS = 4000


@dataclass(frozen=True)
class CandidateFrame:
    """An extracted candidate frame.

    `src_index` is the position of the frame in the source video in display order, from zero.
    `timestamp_us` is the actual timestamp of this frame from the start of the video, microseconds.
    """

    src_index: int
    timestamp_us: int
    path: Path


def _tail(text: str) -> str:
    text = (text or "").strip()
    return text[-_STDERR_TAIL_CHARS:]


def _run(cmd: list[str], *, what: str) -> bytes:
    """Run ffmpeg/ffprobe with the error text from stderr (stderr is not swallowed)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT_S)
    except FileNotFoundError as exc:
        raise InputUnusableError(f"{cmd[0]} not found: it is required to {what}") from exc
    except subprocess.TimeoutExpired as exc:
        raise InputUnusableError(
            f"{cmd[0]} did not respond within the allotted time: {what}"
        ) from exc
    if proc.returncode != 0:
        raise InputUnusableError(
            f"{cmd[0]} failed: {what}",
            details=_tail(proc.stderr.decode("utf-8", "replace")),
        )
    return proc.stdout


def _frame_timestamps(video: Path) -> list[int]:
    """Actual timestamps of all video stream frames in microseconds, in display order."""
    raw = _run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            "-show_entries",
            "frame=best_effort_timestamp_time,pts_time",
            str(video),
        ],
        what="read frame timestamps",
    )
    try:
        frames = json.loads(raw.decode("utf-8", "replace")).get("frames", [])
    except json.JSONDecodeError as exc:
        raise InputUnusableError(
            "ffprobe frame timestamp output could not be parsed as JSON", details=str(exc)
        ) from exc

    seconds: list[float] = []
    missing = 0
    for frame in frames:
        value = frame.get("best_effort_timestamp_time")
        if value in (None, "N/A"):
            value = frame.get("pts_time")
        if value in (None, "N/A"):
            missing += 1
            continue
        try:
            seconds.append(float(value))
        except (TypeError, ValueError):
            missing += 1

    if missing:
        raise InputUnusableError(
            f"the container does not report timestamps for {missing} of {len(frames)} frames: "
            "the actual frame time cannot be recovered, and a computed one must not be substituted"
        )
    if not seconds:
        raise InputUnusableError("ffprobe did not list any frames of the video stream")

    if any(b < a for a, b in zip(seconds, seconds[1:], strict=False)):
        seconds.sort()

    origin = seconds[0]
    return [round((value - origin) * 1_000_000) for value in seconds]


def _decoded_frame_size(video: Path) -> tuple[int, int]:
    """Actual size of the first decoded frame, after the ffmpeg auto-rotation."""
    raw = _run(
        [
            FFMPEG,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "pipe:1",
        ],
        what="check the actual frame orientation",
    )
    if not raw:
        raise InputUnusableError("ffmpeg produced no frames during the orientation check")
    with Image.open(io.BytesIO(raw)) as image:
        return image.size


def _verified_rotation(info: VideoInfo, decoded_size: tuple[int, int]) -> int:
    """Rotation confirmed by comparing the metadata with the actual frame size.

    The size only distinguishes whether the sides are swapped (90/270 versus 0/180), so
    on a mismatch with the metadata the module refuses to guess and reports it.
    """
    expected_swap = info.rotation_deg in (90, 270)
    actual_swap = decoded_size == (info.height, info.width)
    unchanged = decoded_size == (info.width, info.height)

    if expected_swap and actual_swap:
        return info.rotation_deg
    if not expected_swap and unchanged:
        return info.rotation_deg
    raise InputUnusableError(
        "the actual frame rotation does not match the metadata: "
        f"stream {info.width}x{info.height}, metadata {info.rotation_deg}°, "
        f"decoded frame {decoded_size[0]}x{decoded_size[1]}"
    )


def _target_size(source: tuple[int, int], long_side: int | None) -> tuple[int, int]:
    """Size after resizing by the long side. No upscaling."""
    width, height = source
    if long_side is None or max(width, height) <= long_side:
        return width, height
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _read_exactly(stream, size: int) -> bytes | None:
    """Exactly `size` bytes from the stream; None if there are no more frames."""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if remaining == size:
        return None
    if remaining:
        raise InputUnusableError(
            f"ffmpeg returned an incomplete frame: {remaining} of {size} bytes missing"
        )
    return b"".join(chunks)


def extract_candidates(
    video: Path,
    out_dir: Path,
    *,
    info: VideoInfo,
    max_candidates: int = 600,
    long_side: int | None = 1024,
    jpeg_quality: int = 95,
) -> tuple[list[CandidateFrame], dict]:
    """Extract candidate frames with a constant stride.

    Stride: `stride = ceil(total_frames / max_candidates)`, minimum 1; candidates are
    the frames with indices `0, stride, 2*stride, ...`, at most `max_candidates` of them.
    Files are written to `out_dir` as `cand_<src_index:06d>.jpg`. The same input and
    parameters give the same set of files: the name is determined by the frame index,
    not by the write order, and the size is computed, not chosen by ffmpeg.

    Returns the list of candidates and the `image_transform` dict for the manifest
    (data-model §3.3) with `candidate_stride`, `candidates_total` and
    `timestamp_source` added.

    `InputUnusableError` if ffmpeg/ffprobe failed, if the container does not provide
    actual timestamps, or if the actual frame rotation disagrees with the metadata.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if long_side is not None and long_side < 1:
        raise ValueError("long_side must be positive or None")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = _frame_timestamps(video)
    total_frames = len(timestamps)
    stride = max(1, math.ceil(total_frames / max_candidates))
    indices = list(range(0, total_frames, stride))

    decoded_size = _decoded_frame_size(video)
    rotation_applied = _verified_rotation(info, decoded_size)
    source_size = decoded_size
    target_size = _target_size(source_size, long_side)
    resized = target_size != source_size

    filters = [f"select=not(mod(n\\,{stride}))"]
    if resized:
        filters.append(f"scale={target_size[0]}:{target_size[1]}")

    cmd = [
        FFMPEG,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        ",".join(filters),
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]

    frame_bytes = target_size[0] * target_size[1] * 3
    candidates: list[CandidateFrame] = []

    with tempfile.TemporaryFile() as err_file:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file)
        except FileNotFoundError as exc:
            raise InputUnusableError("ffmpeg not found: it is required to extract frames") from exc
        if proc.stdout is None:  # pragma: no cover - Popen with PIPE always provides stdout
            proc.kill()
            raise InputUnusableError("could not get the frame stream from ffmpeg")
        try:
            for src_index in indices:
                buffer = _read_exactly(proc.stdout, frame_bytes)
                if buffer is None:
                    break
                path = out_dir / f"cand_{src_index:06d}.jpg"
                image = Image.frombytes("RGB", target_size, buffer)
                image.save(path, format="JPEG", quality=jpeg_quality, subsampling="4:2:0")
                candidates.append(
                    CandidateFrame(
                        src_index=src_index,
                        timestamp_us=timestamps[src_index],
                        path=path,
                    )
                )
            proc.stdout.read()
        finally:
            proc.stdout.close()
            returncode = proc.wait()
        err_file.seek(0)
        stderr_text = _tail(err_file.read().decode("utf-8", "replace"))

    if returncode != 0:
        raise InputUnusableError("ffmpeg failed during frame extraction", details=stderr_text)
    if len(candidates) != len(indices):
        raise InputUnusableError(
            f"ffmpeg returned {len(candidates)} frames instead of {len(indices)}",
            details=stderr_text,
        )

    source_long = max(source_size)
    image_transform = {
        "rotation_applied_deg": rotation_applied,
        "source_size": [source_size[0], source_size[1]],
        "resize_mode": "none" if long_side is None else "long_side",
        "resized_size": [target_size[0], target_size[1]],
        "scale_factor": max(target_size) / source_long if source_long else 1.0,
        "crop": None,
        "jpeg_quality": jpeg_quality,
        "candidate_stride": stride,
        "candidates_total": len(candidates),
        "timestamp_source": "container_pts",
    }
    return candidates, image_transform
