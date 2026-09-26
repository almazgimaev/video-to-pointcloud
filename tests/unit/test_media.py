"""Tests of presentation media assembly from a stored run (T080/T081).

Requirements: spec.md FR-050…FR-052.

The run used here is a copy of ``assets/precomputed/sample_run``: it already has
``normalized/`` and ``package/images/`` with real (small) frames, so the frame-versus-render
pairs and the trajectory can be built without a synthetic fixture. Sizes are kept small so
the test runs fast and does not depend on ffmpeg being fast, only on it being present.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from v3d.errors import ExitCode, ResultMissingOrIncompatibleError
from v3d.media import build_media

SAMPLE_RUN = Path(__file__).resolve().parents[2] / "assets" / "precomputed" / "sample_run"


@pytest.fixture
def run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "sample"
    shutil.copytree(SAMPLE_RUN, run_dir)
    return run_dir


def test_build_media_writes_all_expected_files(run: Path) -> None:
    index = build_media(run, size=160, turntable_frames=8)
    media_dir = run / "media"

    expected = {
        "view_front.png",
        "view_side.png",
        "view_three_quarter.png",
        "view_top.png",
        "turntable.webp",
        "turntable.mp4",
        "trajectory.png",
        "cloud_web.ply",
        "media.json",
    }
    present = {p.name for p in media_dir.iterdir()}
    assert expected <= present

    files_in_index = {asset["file"] for asset in index["assets"]}
    assert files_in_index <= present
    for name in expected - {"media.json"}:
        assert (media_dir / name).is_file()
        assert (media_dir / name).stat().st_size > 0


def test_index_lists_every_asset_with_what_and_caption(run: Path) -> None:
    index = build_media(run, size=160, turntable_frames=8)

    assert index["schema_version"] == "1"
    assert index["source"] == "normalized/points.ply + normalized/cameras.json + package/images"
    assert len(index["assets"]) >= 1
    for asset in index["assets"]:
        assert asset["what"]
        assert asset["caption"]
        assert asset["file"]
        assert "run " in asset["caption"]


def test_precomputed_example_mark_is_in_every_caption(run: Path) -> None:
    index = build_media(run, size=160, turntable_frames=8)

    assert index["precomputed_example"] is True
    for asset in index["assets"]:
        assert "precomputed example" in asset["caption"]


def test_pairs_are_roughly_twice_the_frame_width(run: Path) -> None:
    from PIL import Image

    index = build_media(run, size=160, turntable_frames=8)
    pairs = [a for a in index["assets"] if a["kind"] == "frame_vs_render"]
    assert pairs  # the sample run has registered cameras with matching images

    for pair in pairs:
        frame_width = Image.open(run / "package" / "images" / f"{pair['frame_id']}.jpg").width
        image = Image.open(run / "media" / pair["file"])
        assert abs(image.width - 2 * frame_width) <= frame_width * 0.2


def test_web_cloud_label_is_present(run: Path) -> None:
    index = build_media(run, size=160, turntable_frames=8, web_target=50)
    web = [a for a in index["assets"] if a["kind"] == "web_cloud"]
    assert len(web) == 1
    assert "of" in web[0]["label"]
    assert "points" in web[0]["label"]

    web_ply = run / "media" / "cloud_web.ply"
    assert web_ply.is_file()
    assert web_ply.stat().st_size > 0


def test_missing_normalized_is_refused_with_code_7(run: Path) -> None:
    for file in (run / "normalized").iterdir():
        file.unlink()
    (run / "normalized").rmdir()

    with pytest.raises(ResultMissingOrIncompatibleError) as error:
        build_media(run)
    assert error.value.exit_code == ExitCode.RESULT_MISSING_OR_INCOMPATIBLE


def test_out_dir_override(run: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "elsewhere"
    index = build_media(run, out_dir=out_dir, size=160, turntable_frames=8)

    assert (out_dir / "media.json").is_file()
    assert index["run_id"]


def test_build_media_is_deterministic_except_timestamp(run: Path, tmp_path: Path) -> None:
    first = build_media(run, size=160, turntable_frames=8)
    view_bytes_1 = (run / "media" / "view_front.png").read_bytes()

    out2 = tmp_path / "second"
    second = build_media(run, out_dir=out2, size=160, turntable_frames=8)
    view_bytes_2 = (out2 / "view_front.png").read_bytes()

    assert view_bytes_1 == view_bytes_2
    assert first["assets"] == second["assets"]
    assert first["generated_utc"] != "" and second["generated_utc"] != ""


def _resize_package_frames(run: Path, size: tuple[int, int]) -> None:
    from PIL import Image

    for path in (run / "package" / "images").glob("*.jpg"):
        with Image.open(path) as frame:
            frame.convert("RGB").resize(size).save(path)


def test_uniformly_resized_frames_are_rescaled_and_recorded(run: Path) -> None:
    """Same aspect ratio: intrinsics are scaled to the frame, and the index says so."""
    _resize_package_frames(run, (160, 120))
    index = build_media(run, size=120, turntable_frames=4)
    pairs = [a for a in index["assets"] if a["kind"] == "frame_vs_render"]
    assert pairs, "pairs must still be produced for a uniform resize"
    assert all(p["intrinsics_rescaled_from"] == [320, 240] for p in pairs)


def test_frames_with_another_aspect_ratio_are_skipped_not_distorted(run: Path) -> None:
    """A different aspect ratio cannot be fixed by scaling: the pair is skipped with a reason."""
    _resize_package_frames(run, (200, 100))
    index = build_media(run, size=120, turntable_frames=4)
    assert not [a for a in index["assets"] if a["kind"] == "frame_vs_render"]
    reasons = " ".join(item["reason"] for item in index.get("skipped_pairs", []))
    assert "aspect ratio" in reasons
    assert "/Users/" not in reasons and "/home/" not in reasons, "no absolute paths in the index"
