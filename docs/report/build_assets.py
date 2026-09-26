"""Copy the published images out of local run directories into ``docs/assets/`` (FR-050, FR-051).

Run directories are local and not versioned, so everything the README and the project page show
must be copied into the repository. Images are converted to WebP at a web-friendly size; each
asset is listed in ``docs/assets/assets.json`` with what it shows, its source run and whether it
is a selected example (Principle I).

Usage: ``python docs/report/build_assets.py`` (after ``v3d media`` for the runs listed below).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from v3d import geometry, render
from v3d.ply_io import read_ply

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
OUT = REPO / "docs" / "assets"

# The typical run: 48 uniformly selected frames at the working threshold 1.6 — what the tool
# produces by default. Not the best of many runs; the only run with these settings.
MAIN = "20260920-084926-91b5ed03"
# P2 branches at the stress-test threshold 1.8.
P2_UNIFORM = "20260926-221214-e80ab63c"
P2_QUALITY = "20260926-221232-494c1365"

MAX_SIDE = 1400


def _webp(src: Path, dst: Path, *, max_side: int = MAX_SIDE, quality: int = 84) -> dict:
    with Image.open(src) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        im.save(dst, "WEBP", quality=quality, method=6)
        return {"width": im.width, "height": im.height}


def _alignment_before_after(run: str, dst: Path) -> dict:
    """The same cloud in the reconstructor frame and after alignment, rendered side by side."""
    run_dir = RUNS / run
    xyz, rgb, _ = read_ply(run_dir / "normalized" / "points.ply")
    cameras = json.loads((run_dir / "normalized" / "cameras.json").read_text(encoding="utf-8"))
    undo = geometry.invert_transform(cameras["world_alignment"]["transform"])
    raw = geometry.apply_transform_to_points(undo, xyz)

    centre, radius = render.robust_extent(xyz)
    distance = render.framing_distance(radius * 1.25)
    size = 640

    def view(points: np.ndarray, target: np.ndarray, caption: str) -> Image.Image:
        cam = render.orbit_camera(
            target=target, distance=distance, azimuth_deg=0, elevation_deg=12,
            width=size, height=size,
        )  # fmt: skip
        return render.add_caption(render.render_points(points, rgb, cam), caption)

    raw_centre, _ = render.robust_extent(raw)
    left = view(raw, raw_centre, f"reconstructor frame (as returned) · run {run}")
    right = view(xyz, centre, f"after alignment: +Y up, centred · run {run}")
    combined = render.side_by_side(left, right)
    combined.save(dst, "WEBP", quality=84, method=6)
    return {"width": combined.width, "height": combined.height}


def _video(src: Path, dst: Path, *, width: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
            "-vf", f"scale={width}:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "26", "-movflags", "+faststart", "-an", str(dst),
        ],
        check=True,
    )  # fmt: skip


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_media = RUNS / MAIN / "media"
    assets = []

    def add(file: str, what: str, run: str, *, selected: bool = False, **extra) -> None:
        assets.append(
            {"file": file, "what": what, "source_run": run, "selected_example": selected, **extra}
        )

    for name, what in [
        ("pair_0000_000000", "first video frame next to a render from the same camera pose"),
        ("pair_0032_000432", "a later video frame next to a render from the same camera pose"),
        ("view_front", "point cloud, front view"),
        ("view_three_quarter", "point cloud, three-quarter view"),
        ("view_top", "point cloud, top view"),
        ("trajectory", "camera trajectory: 48 registered cameras around the object"),
    ]:
        size = _webp(main_media / f"{name}.png", OUT / f"{name}.webp")
        add(f"{name}.webp", what, MAIN, **size)

    _video(main_media / "turntable.mp4", OUT / "turntable.mp4", width=640)
    add("turntable.mp4", "turntable animation of the point cloud, 72 frames", MAIN)
    shutil.copy2(main_media / "cloud_web.ply", OUT / "cloud_web.ply")
    add("cloud_web.ply", "point cloud decimated for the web: 40,000 of 100,000 points", MAIN)

    size = _alignment_before_after(MAIN, OUT / "alignment_before_after.webp")
    add("alignment_before_after.webp", "the same cloud before and after world alignment", MAIN,
        **size)  # fmt: skip

    for run, label in ((P2_UNIFORM, "uniform"), (P2_QUALITY, "quality")):
        size = _webp(RUNS / run / "media" / "view_front.png", OUT / f"p2_{label}_front.webp")
        add(f"p2_{label}_front.webp", f"P2 at threshold 1.8, {label} selection, front view", run,
            **size)  # fmt: skip

    index = {
        "note": (
            "Images copied from local run directories by docs/report/build_assets.py. "
            f"The main run {MAIN} is the only run with the default settings (48 uniform frames, "
            "threshold 1.6), not the best of several."
        ),
        "assets": assets,
    }
    (OUT / "assets.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"{len(assets)} assets, {total / 1e6:.1f} MB in {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
