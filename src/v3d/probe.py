"""Verifiable properties of the input video file (`v3d check`).

The module reads the file through `ffprobe` and reports only what is actually recorded
in the file: container, video codec, duration, resolution, average frame rate,
frame count, orientation metadata, size and sha256 of the content.

Module boundaries (FR-006, FR-007). There are no judgements about the scene here and
there must never be: "the object is static", "lighting is stable", "there is little
light", "the texture is poor" are assumptions about the shot content, they cannot be
verified by reading the file and never end up in `VideoInfo`. Everything the module
returns can be verified by running `ffprobe` again on the same file.

Rotation convention. `rotation_deg` is the angle **counter-clockwise** by which the
frame must be rotated to display correctly. This is the display matrix convention
(`ffprobe` `rotation` field in side data, matches `av_display_rotation_get`). The
legacy `tags.rotate` tag gives an angle *clockwise*, so its sign is flipped on reading.

Command contract: specs/001-video-3d-mvp/contracts/cli.md (`v3d check`).
Data model: specs/001-video-3d-mvp/data-model.md §3.1.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from v3d.errors import InputUnusableError
from v3d.runid import sha256_file

# Supported v1 input (contracts/cli.md, A-02). "Any video is supported" is not claimed.
SUPPORTED_CONTAINERS: frozenset[str] = frozenset({"mp4", "mov"})
SUPPORTED_CODECS: frozenset[str] = frozenset({"h264", "hevc"})

FFPROBE = "ffprobe"
_FFPROBE_TIMEOUT_S = 120


@dataclass(frozen=True)
class VideoInfo:
    """Verifiable properties of the input video. No field describes the scene content."""

    path: str
    filename: str
    sha256: str
    size_bytes: int
    container: str
    video_codec: str
    duration_s: float
    width: int
    height: int
    avg_fps: float
    nb_frames: int | None  # None if the container does not report the frame count
    rotation_deg: int  # 0/90/180/270, counter-clockwise, from side data
    supported: bool

    def to_dict(self) -> dict:
        """Exactly the keys listed in data-model §3.1 (InputVideo)."""
        return {
            "path": self.path,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "container": self.container,
            "video_codec": self.video_codec,
            "duration_s": self.duration_s,
            "width": self.width,
            "height": self.height,
            "avg_fps": self.avg_fps,
            "nb_frames": self.nb_frames,
            "rotation_deg": self.rotation_deg,
            "supported": self.supported,
        }


def _run_ffprobe(args: list[str]) -> str:
    """Run `ffprobe` with ready-made arguments; returns stdout.

    A separate function for a single patch point in tests and a single point for
    launch error handling. stderr is not swallowed: it ends up in the exception `details`.
    """
    try:
        proc = subprocess.run(
            [FFPROBE, *args],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise InputUnusableError(
            "ffprobe not found: it is required to read video file properties",
            details=str(exc),
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise InputUnusableError(
            "ffprobe did not respond within the allotted time",
            details=str(exc),
        ) from exc
    if proc.returncode != 0:
        raise InputUnusableError(
            "ffprobe could not read the file",
            details=(proc.stderr or "").strip(),
        )
    return proc.stdout


def _probe_json(path: Path) -> dict:
    """Container and stream information as parsed JSON."""
    raw = _run_ffprobe(["-v", "error", "-of", "json", "-show_format", "-show_streams", str(path)])
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputUnusableError(
            "ffprobe output could not be parsed as JSON",
            details=str(exc),
        ) from exc
    if not isinstance(data, dict):
        raise InputUnusableError("ffprobe output has an unexpected structure")
    return data


def _video_stream(data: dict) -> dict:
    """First video stream. A missing video track means unusable input."""
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise InputUnusableError(
        "the file has no video track: there is nothing to process",
    )


def _parse_rate(value: str | None) -> float | None:
    """Parse a fraction such as `30000/1001`. None if the rate is undefined."""
    if not value or value == "N/A":
        return None
    if "/" in value:
        num_text, _, den_text = value.partition("/")
        try:
            num, den = float(num_text), float(den_text)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        rate = float(value)
    except ValueError:
        return None
    return rate or None


def _parse_float(value: object) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _snap_to_quadrant(degrees: float) -> int:
    """Snap an angle to 0/90/180/270. The container should not specify intermediate angles."""
    return int(round(degrees / 90.0)) % 4 * 90


def _rotation_deg(stream: dict) -> int:
    """Rotation from stream metadata, counter-clockwise.

    Sources in decreasing order of reliability: display matrix in side data, then the
    legacy `rotate` tag (it gives clockwise rotation, so the sign is flipped).
    No metadata means 0.
    """
    for side_data in stream.get("side_data_list") or []:
        if not isinstance(side_data, dict):
            continue
        rotation = _parse_float(side_data.get("rotation"))
        if rotation is not None:
            return _snap_to_quadrant(rotation)
    tag = _parse_float((stream.get("tags") or {}).get("rotate"))
    if tag is not None:
        return _snap_to_quadrant(-tag)
    return 0


def _containers(format_block: dict) -> list[str]:
    """`format_name` is a comma-separated list (`mov,mp4,m4a,3gp,3g2,mj2`)."""
    raw = format_block.get("format_name") or ""
    return [name.strip() for name in raw.split(",") if name.strip()]


def probe_video(path: Path) -> VideoInfo:
    """Read the verifiable properties of a video file.

    Raises `InputUnusableError` (exit code 2) if the file is missing, `ffprobe` cannot
    read it, it has no video track, or the container or codec is outside the supported
    list. The returned `VideoInfo.supported` is always true: unusable input never gets
    as far as a return. The field is kept because it is required in the manifest.
    """
    path = Path(path)
    if not path.exists():
        raise InputUnusableError(f"file not found: {path}")
    if not path.is_file():
        raise InputUnusableError(f"not a file: {path}")

    data = _probe_json(path)
    stream = _video_stream(data)
    format_block = data.get("format") or {}

    containers = _containers(format_block)
    # ffprobe returns the format family as a single list ("mov,mp4,m4a,3gp,3g2,mj2"), and
    # its first element does not have to match the actual file. Pick the supported
    # container that matches the file extension, otherwise the first from the list.
    # Otherwise .mp4 would be shown in the report as "mov", which reads as a detection error.
    by_suffix = path.suffix.lower().lstrip(".")
    if by_suffix in containers and by_suffix in SUPPORTED_CONTAINERS:
        container = by_suffix
    else:
        container = containers[0] if containers else ""
    codec = (stream.get("codec_name") or "").lower()

    container_ok = bool(SUPPORTED_CONTAINERS.intersection(containers))
    codec_ok = codec in SUPPORTED_CODECS
    if not container_ok or not codec_ok:
        raise InputUnusableError(
            _unsupported_message(
                containers=containers,
                codec=codec,
                container_ok=container_ok,
                codec_ok=codec_ok,
            )
        )

    width = _parse_int(stream.get("width"))
    height = _parse_int(stream.get("height"))
    if not width or not height:
        raise InputUnusableError("ffprobe did not report the frame size")

    duration = _parse_float(stream.get("duration"))
    if duration is None:
        duration = _parse_float(format_block.get("duration"))
    if duration is None:
        raise InputUnusableError("ffprobe did not report the video duration")

    avg_fps = _parse_rate(stream.get("avg_frame_rate"))
    if avg_fps is None:
        avg_fps = _parse_rate(stream.get("r_frame_rate"))
    if avg_fps is None:
        raise InputUnusableError("ffprobe did not report the frame rate")

    size_bytes = _parse_int(format_block.get("size"))
    if size_bytes is None:
        size_bytes = path.stat().st_size

    return VideoInfo(
        path=str(path),
        filename=path.name,
        sha256=sha256_file(path),
        size_bytes=size_bytes,
        container=container,
        video_codec=codec,
        duration_s=duration,
        width=width,
        height=height,
        avg_fps=avg_fps,
        # A missing nb_frames is exactly that, missing: no zero, no estimate from fps.
        nb_frames=_parse_int(stream.get("nb_frames")),
        rotation_deg=_rotation_deg(stream),
        supported=True,
    )


def _unsupported_message(
    *, containers: list[str], codec: str, container_ok: bool, codec_ok: bool
) -> str:
    """The message names exactly what is outside the list and lists what is supported."""
    parts: list[str] = []
    if not container_ok:
        shown = ",".join(containers) or "unknown"
        parts.append(f"container '{shown}' is not supported")
    if not codec_ok:
        shown = codec or "unknown"
        parts.append(f"video codec '{shown}' is not supported")
    supported = (
        f"supported containers: {', '.join(sorted(SUPPORTED_CONTAINERS))}; "
        f"video codecs: {', '.join(sorted(SUPPORTED_CODECS))}"
    )
    return f"{'; '.join(parts)}. {supported}"
