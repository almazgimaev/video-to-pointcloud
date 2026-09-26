"""World alignment through the whole light stage: prepare → tilted "GPU result" → ingest (T075).

The synthetic scene is moved by a known tilt and offset, the way a reconstructor anchors its world
to an arbitrary first camera, with an uneven handheld orbit. After ingest the result must stand
upright, be centred on the object, and the export must still equal the shown cloud.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from v3d import geometry as g
from v3d.export import export_point_cloud
from v3d.ingest import run_ingest
from v3d.ply_io import read_ply
from v3d.prepare import run_prepare

REPO = Path(__file__).resolve().parents[2]
MAKE_SCENE = REPO / "assets" / "synthetic" / "make_scene.py"
BUDGET = 12


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("align")
    clip = base / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc=size=320x240:rate=10:duration=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True,
    )  # fmt: skip
    return run_prepare(clip, out_root=base / "runs", budget=BUDGET, long_side=320)


@pytest.fixture
def run_dir(prepared, tmp_path: Path) -> Path:
    target = tmp_path / prepared.layout.root.name
    shutil.copytree(prepared.layout.root, target)
    return target


def _scene(out: Path, frames_json: Path, *extra: str) -> tuple[Path, dict]:
    subprocess.run(
        [
            sys.executable, str(MAKE_SCENE), "--out", str(out),
            "--cameras", str(BUDGET), "--points", "400", "--seed", "11",
            "--names-from", str(frames_json), *extra,
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return out, json.loads((out / "expected.json").read_text(encoding="utf-8"))


HANDHELD = (
    "--tilt-deg", "52", "--tilt-axis", "1", "0.4", "0.2",
    "--offset", "0.7", "-1.3", "2.1",
    "--height-jitter", "0.06", "--roll-jitter-deg", "10",
)  # fmt: skip


def _cameras(run_dir: Path) -> dict:
    return json.loads((run_dir / "normalized" / "cameras.json").read_text(encoding="utf-8"))


def test_tilted_handheld_result_is_brought_upright(run_dir: Path, tmp_path: Path) -> None:
    """SC-015: the scene's true up ends on +Y and the object centre on the origin."""
    scene, expected = _scene(tmp_path / "scene", run_dir / "frames.json", *HANDHELD)
    run_ingest(run_dir, source=scene)

    cameras = _cameras(run_dir)
    alignment = cameras["world_alignment"]
    T = np.array(alignment["transform"])
    assert alignment["status"] == "estimated"
    up = T[:3, :3] @ np.array(expected["true_up_world"])
    angle = np.degrees(np.arccos(np.clip(up @ [0.0, 1.0, 0.0], -1.0, 1.0)))
    assert angle < 5.0, f"true up is {angle:.2f}° away from +Y"
    centre = g.apply_transform_to_points(T, np.array([expected["true_object_center"]]))[0]
    assert np.linalg.norm(centre) < 0.05


def test_conventions_declare_the_estimated_frame(run_dir: Path, tmp_path: Path) -> None:
    """FR-046: the frame is declared an estimate; scale stays not determined."""
    scene, _ = _scene(tmp_path / "scene", run_dir / "frames.json", *HANDHELD)
    run_ingest(run_dir, source=scene)

    cameras = _cameras(run_dir)
    assert "+Y up (estimated" in cameras["conventions"]["world_axes"]
    assert cameras["conventions"]["scale_status"] == "not_determined"
    assert "not measured" in cameras["world_alignment"]["note"]
    _, _, comments = read_ply(run_dir / "normalized" / "points.ply")
    assert comments["world_alignment"] == "estimated"


def test_raw_result_is_left_untouched(run_dir: Path, tmp_path: Path) -> None:
    """FR-046: alignment never rewrites the reconstructor's own files."""
    scene, _ = _scene(tmp_path / "scene", run_dir / "frames.json", *HANDHELD)
    before = (scene / "sparse" / "images.bin").read_bytes()
    run_ingest(run_dir, source=scene)
    assert (run_dir / "result" / "sparse" / "images.bin").read_bytes() == before


def test_export_equals_the_shown_aligned_cloud(run_dir: Path, tmp_path: Path) -> None:
    """SC-017: after alignment the export is still exactly the shown cloud (FR-031)."""
    scene, _ = _scene(tmp_path / "scene", run_dir / "frames.json", *HANDHELD)
    run_ingest(run_dir, source=scene)
    exported = export_point_cloud(run_dir)

    shown_xyz, shown_rgb, _ = read_ply(run_dir / "normalized" / "points.ply")
    out_xyz, out_rgb, _ = read_ply(exported)
    assert np.array_equal(shown_xyz, out_xyz)
    assert np.array_equal(shown_rgb, out_rgb)


def test_partial_orbit_is_only_recentred_and_warned(run_dir: Path, tmp_path: Path) -> None:
    """SC-016 / FR-047: a quarter orbit keeps the frame, centres it, and says so."""
    scene, _ = _scene(tmp_path / "scene", run_dir / "frames.json", *HANDHELD, "--arc-deg", "90")
    run_ingest(run_dir, source=scene)

    cameras = _cameras(run_dir)
    assert cameras["world_alignment"]["status"] == "not_estimated"
    assert "vertical not estimated" in cameras["conventions"]["world_axes"]
    status = json.loads((run_dir / "normalized" / "status.json").read_text(encoding="utf-8"))
    codes = [w["code"] for w in status["warnings"]]
    assert "orientation_not_estimated" in codes
