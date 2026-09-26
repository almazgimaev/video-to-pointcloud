"""Parsing of ffprobe output in `v3d.probe`.

ffprobe invocation is patched: the parsing is tested, not the behaviour of the external tool.
Behaviour on a real file is tested in tests/integration/test_extract.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from v3d import probe
from v3d.errors import ExitCode, InputUnusableError


def _stream(**overrides) -> dict:
    stream = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30000/1001",
        "r_frame_rate": "30000/1001",
        "duration": "12.500000",
        "nb_frames": "375",
    }
    stream.update(overrides)
    return stream


def _payload(*, streams: list[dict] | None = None, format_name: str = "mov,mp4,m4a") -> dict:
    return {
        "streams": [_stream()] if streams is None else streams,
        "format": {
            "format_name": format_name,
            "duration": "12.500000",
            "size": "1234567",
        },
    }


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not a real video, ffprobe is patched")
    return path


def _patch(monkeypatch: pytest.MonkeyPatch, payload: dict) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(args: list[str]) -> str:
        calls.append(args)
        return json.dumps(payload)

    monkeypatch.setattr(probe, "_run_ffprobe", fake)
    return calls


def test_avg_fps_computed_from_fraction(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    _patch(monkeypatch, _payload())
    info = probe.probe_video(video_file)
    assert info.avg_fps == pytest.approx(30000 / 1001)
    assert info.nb_frames == 375
    # From the "mov,mp4,m4a,…" family the container matching the file extension is chosen:
    # otherwise .mp4 would be shown to the user as "mov".
    assert info.container == "mp4"
    assert info.video_codec == "h264"
    assert info.supported is True


def test_avg_fps_falls_back_to_r_frame_rate(
    monkeypatch: pytest.MonkeyPatch, video_file: Path
) -> None:
    _patch(monkeypatch, _payload(streams=[_stream(avg_frame_rate="0/0")]))
    info = probe.probe_video(video_file)
    assert info.avg_fps == pytest.approx(30000 / 1001)


def test_nb_frames_absent_is_none_not_zero(
    monkeypatch: pytest.MonkeyPatch, video_file: Path
) -> None:
    stream = _stream()
    del stream["nb_frames"]
    _patch(monkeypatch, _payload(streams=[stream]))
    info = probe.probe_video(video_file)
    assert info.nb_frames is None
    assert info.to_dict()["nb_frames"] is None


def test_nb_frames_na_is_none(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    _patch(monkeypatch, _payload(streams=[_stream(nb_frames="N/A")]))
    assert probe.probe_video(video_file).nb_frames is None


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotation_from_display_matrix(
    monkeypatch: pytest.MonkeyPatch, video_file: Path, rotation: int
) -> None:
    stream = _stream(side_data_list=[{"side_data_type": "Display Matrix", "rotation": rotation}])
    _patch(monkeypatch, _payload(streams=[stream]))
    assert probe.probe_video(video_file).rotation_deg == rotation


def test_rotation_negative_display_matrix_normalized(
    monkeypatch: pytest.MonkeyPatch, video_file: Path
) -> None:
    stream = _stream(side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}])
    _patch(monkeypatch, _payload(streams=[stream]))
    assert probe.probe_video(video_file).rotation_deg == 270


def test_rotation_from_legacy_tag_changes_sign(
    monkeypatch: pytest.MonkeyPatch, video_file: Path
) -> None:
    # The rotate tag gives clockwise rotation, VideoInfo counter-clockwise.
    _patch(monkeypatch, _payload(streams=[_stream(tags={"rotate": "90"})]))
    assert probe.probe_video(video_file).rotation_deg == 270


def test_rotation_absent_is_zero(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    _patch(monkeypatch, _payload())
    assert probe.probe_video(video_file).rotation_deg == 0


def test_unsupported_codec_rejected(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    _patch(monkeypatch, _payload(streams=[_stream(codec_name="vp9")]))
    with pytest.raises(InputUnusableError) as excinfo:
        probe.probe_video(video_file)
    message = excinfo.value.message
    assert excinfo.value.exit_code == ExitCode.INPUT_UNUSABLE
    assert "vp9" in message
    assert "h264" in message and "hevc" in message


def test_unsupported_container_rejected(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    _patch(monkeypatch, _payload(format_name="matroska,webm"))
    with pytest.raises(InputUnusableError) as excinfo:
        probe.probe_video(video_file)
    message = excinfo.value.message
    assert excinfo.value.exit_code == ExitCode.INPUT_UNUSABLE
    assert "matroska" in message
    assert "mp4" in message and "mov" in message


def test_no_video_stream_rejected(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    audio = {"codec_type": "audio", "codec_name": "aac"}
    _patch(monkeypatch, _payload(streams=[audio]))
    with pytest.raises(InputUnusableError) as excinfo:
        probe.probe_video(video_file)
    assert "video track" in excinfo.value.message
    assert excinfo.value.exit_code == ExitCode.INPUT_UNUSABLE


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(InputUnusableError) as excinfo:
        probe.probe_video(tmp_path / "no-such-file.mp4")
    assert excinfo.value.exit_code == ExitCode.INPUT_UNUSABLE
    assert "not found" in excinfo.value.message


def test_ffprobe_failure_keeps_stderr(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    def failing(args: list[str]) -> str:
        raise InputUnusableError("ffprobe could not read the file", details="moov atom not found")

    monkeypatch.setattr(probe, "_run_ffprobe", failing)
    with pytest.raises(InputUnusableError) as excinfo:
        probe.probe_video(video_file)
    assert "moov atom not found" in excinfo.value.render()


def test_duration_falls_back_to_format_block(
    monkeypatch: pytest.MonkeyPatch, video_file: Path
) -> None:
    stream = _stream()
    del stream["duration"]
    _patch(monkeypatch, _payload(streams=[stream]))
    assert probe.probe_video(video_file).duration_s == pytest.approx(12.5)


def test_to_dict_keys_match_data_model(monkeypatch: pytest.MonkeyPatch, video_file: Path) -> None:
    _patch(monkeypatch, _payload())
    assert set(probe.probe_video(video_file).to_dict()) == {
        "path",
        "filename",
        "sha256",
        "size_bytes",
        "container",
        "video_codec",
        "duration_s",
        "width",
        "height",
        "avg_fps",
        "nb_frames",
        "rotation_deg",
        "supported",
    }


def test_container_chosen_by_extension_within_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A .mov file in the same format family is shown as mov, not as the first in the list."""
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"\x00" * 16)
    _patch(monkeypatch, _payload())
    info = probe.probe_video(clip)
    assert info.container == "mov"
