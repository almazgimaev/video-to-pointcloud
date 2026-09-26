"""Assembling presentation media from a stored run, without a GPU (FR-050…FR-052).

Requirements: spec.md FR-050…FR-052, tasks.md T078/T080/T081, data-model.md §3.4a (aligned
world: +Y up, object at the origin, first reconstructed camera in front on +Z).

Everything here is computed from artifacts already on disk (`normalized/points.ply`,
`normalized/cameras.json`, `package/images/`): no reconstruction, no GPU, no network call.
The renderer that does the actual projection lives in :mod:`v3d.render` and is not
duplicated here.

Every produced asset is captioned with its source (`render.add_caption`) and described in
the returned/written index `media/media.json` (FR-051): what it shows and where it came
from. A precomputed example gets an extra mark in every caption, taken from
`normalized/status.json` — never assumed.

Determinism: same run artifacts and the same parameters produce byte-identical PNGs (the
index differs only in `generated_utc`). The renderer itself is deterministic (no random
sampling); ffmpeg is invoked with a fixed frame rate and fixed encoder settings.
"""

from __future__ import annotations

import datetime
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from v3d import ingest_checks, ply_io, render, runs
from v3d.artifacts import read_json, write_json
from v3d.errors import ResultMissingOrIncompatibleError

SCHEMA_VERSION = "1"

MEDIA_DIRNAME = "media"
INDEX_NAME = "media.json"

#: Views rendered around the aligned world (§3.4a); azimuth 0 is the front (+Z side).
VIEWS: tuple[tuple[str, str, float, float], ...] = (
    ("view_front.png", "front view", 0.0, 15.0),
    ("view_side.png", "side view", 90.0, 15.0),
    ("view_three_quarter.png", "three-quarter view", 35.0, 30.0),
    ("view_top.png", "top view", 0.0, 85.0),
)

#: Elevation of the turntable orbit, degrees.
TURNTABLE_ELEVATION_DEG = 20.0
#: Rendering size of turntable frames is capped so the animation stays light (T078).
TURNTABLE_MAX_SIZE = 640
#: Encoding frame rate for both the WebP and the MP4 turntable.
TURNTABLE_FPS = 24

#: Elevation and extra framing margin of the camera-trajectory overview.
TRAJECTORY_ELEVATION_DEG = 75.0
TRAJECTORY_MARGIN = 1.25

FFMPEG = "ffmpeg"
_FFMPEG_TIMEOUT_S = 600


def _caption_suffix(precomputed_example: bool) -> str:
    return " · precomputed example" if precomputed_example else ""


def _caption(what: str, run_id: str, *, precomputed_example: bool) -> str:
    return f"{what} · run {run_id}{_caption_suffix(precomputed_example)}"


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _run_ffmpeg(cmd: list[str], *, what: str) -> None:
    """Run ffmpeg with the error text from stderr surfaced, not swallowed."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT_S)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg not found: it is required to {what}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg did not respond within the allotted time: {what}") from exc
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip()[-4000:]
        raise RuntimeError(f"ffmpeg failed to {what}", tail)


def _render_views(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    centre: np.ndarray,
    distance: float,
    size: int,
    run_id: str,
    precomputed_example: bool,
    media_dir: Path,
) -> list[dict]:
    assets = []
    for filename, what, azimuth, elevation in VIEWS:
        camera = render.orbit_camera(
            target=centre,
            distance=distance,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            width=size,
            height=size,
        )
        image = render.render_points(points, colors, camera)
        caption_text = _caption(what, run_id, precomputed_example=precomputed_example)
        image = render.add_caption(image, caption_text)
        _save(image, media_dir / filename)
        assets.append(
            {
                "file": filename,
                "kind": "view",
                "what": what,
                "caption": caption_text,
                "source": "normalized/points.ply",
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
            }
        )
    return assets


def _render_turntable(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    centre: np.ndarray,
    distance: float,
    size: int,
    frames: int,
    media_dir: Path,
) -> dict:
    turntable_size = min(size, TURNTABLE_MAX_SIZE)
    step = 360.0 / frames
    with tempfile.TemporaryDirectory(prefix="v3d-turntable-") as tmp:
        tmp_dir = Path(tmp)
        for i in range(frames):
            camera = render.orbit_camera(
                target=centre,
                distance=distance,
                azimuth_deg=i * step,
                elevation_deg=TURNTABLE_ELEVATION_DEG,
                width=turntable_size,
                height=turntable_size,
            )
            image = render.render_points(points, colors, camera)
            image.save(tmp_dir / f"frame_{i:04d}.png")

        webp_path = media_dir / "turntable.webp"
        mp4_path = media_dir / "turntable.mp4"
        media_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(tmp_dir / "frame_%04d.png")

        _run_ffmpeg(
            [
                FFMPEG,
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-framerate",
                str(TURNTABLE_FPS),
                "-i",
                pattern,
                "-c:v",
                "libwebp_anim",
                "-loop",
                "0",
                "-q:v",
                "75",
                str(webp_path),
            ],
            what="encode the turntable animation as WebP",
        )
        _run_ffmpeg(
            [
                FFMPEG,
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-framerate",
                str(TURNTABLE_FPS),
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "22",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ],
            what="encode the turntable animation as MP4",
        )

    return {
        "webp_bytes": webp_path.stat().st_size,
        "mp4_bytes": mp4_path.stat().st_size,
        "frames": frames,
        "size": turntable_size,
    }


def _record_for_image_size(record: dict, actual_size: tuple[int, int]) -> dict:
    """Scale ``record``'s intrinsics to the frame's actual on-disk size, if it differs.

    ``package/images/<frame_id>.jpg`` and ``normalized/cameras.json`` are written by separate
    stages and can end up disagreeing about resolution (e.g. a resize applied to one but not
    recorded consistently in the other). Rescaling the intrinsics keeps the render's camera
    model exact for the frame actually on disk, instead of silently rendering at the wrong
    size or upscaling/blurring the photograph to match.
    """
    declared = record.get("image_size")
    if not declared or (int(declared[0]), int(declared[1])) == tuple(actual_size):
        return record
    scale_x = actual_size[0] / declared[0]
    scale_y = actual_size[1] / declared[1]
    intr = record["intrinsics"]
    return {
        **record,
        "image_size": [actual_size[0], actual_size[1]],
        "intrinsics": {
            **intr,
            "fx": intr["fx"] * scale_x,
            "fy": intr["fy"] * scale_y,
            "cx": intr["cx"] * scale_x,
            "cy": intr["cy"] * scale_y,
        },
    }


def _same_aspect(a: tuple[int, int], b: tuple[int, int], tolerance: float = 0.01) -> bool:
    """True when two image sizes have the same aspect ratio (a uniform resize)."""
    return abs(a[0] / a[1] - b[0] / b[1]) <= tolerance * (a[0] / a[1])


def _pair_indices(n: int) -> list[int]:
    """Deterministic indices spread along the camera list: first, ~1/3, ~2/3."""
    if n <= 0:
        return []
    picks = sorted({0, n // 3, (2 * n) // 3} & set(range(n)))
    return picks


def _render_pairs(
    cameras: list[dict],
    points: np.ndarray,
    colors: np.ndarray,
    *,
    images_dir: Path,
    run_id: str,
    precomputed_example: bool,
    media_dir: Path,
) -> tuple[list[dict], list[dict]]:
    assets = []
    skipped = []
    for idx in _pair_indices(len(cameras)):
        record = cameras[idx]
        frame_id = record.get("frame_id", f"camera_{idx:04d}")
        image_path = images_dir / f"{frame_id}.jpg"
        if not image_path.is_file():
            skipped.append(
                {
                    "frame_id": frame_id,
                    "reason": f"image file is missing: package/images/{frame_id}.jpg",
                }
            )
            continue
        if not record.get("intrinsics"):
            skipped.append({"frame_id": frame_id, "reason": "camera has no intrinsics"})
            continue

        left = Image.open(image_path).convert("RGB")
        declared = tuple(int(v) for v in record.get("image_size") or left.size)
        if not _same_aspect(declared, left.size):
            skipped.append(
                {
                    "frame_id": frame_id,
                    "reason": (
                        f"frame is {left.size[0]}x{left.size[1]} but the camera was estimated "
                        f"for {declared[0]}x{declared[1]} with a different aspect ratio; "
                        "a render would be distorted"
                    ),
                }
            )
            continue
        camera = render.camera_from_record(_record_for_image_size(record, left.size))
        right = render.render_points(points, colors, camera)
        combined = render.side_by_side(left, right)
        what = f"video frame {frame_id} versus a render from the same camera pose"
        caption_text = (
            f"video frame {frame_id} | render from the same camera pose · run {run_id}"
            f"{_caption_suffix(precomputed_example)}"
        )
        combined = render.add_caption(combined, caption_text)
        filename = f"pair_{frame_id}.png"
        _save(combined, media_dir / filename)
        assets.append(
            {
                "file": filename,
                "kind": "frame_vs_render",
                "what": what,
                "caption": caption_text,
                "source": (
                    f"package/images/{frame_id}.jpg + "
                    "normalized/cameras.json + normalized/points.ply"
                ),
                "frame_id": frame_id,
                "intrinsics_rescaled_from": list(declared) if declared != left.size else None,
            }
        )
    return assets, skipped


def _render_trajectory(
    points: np.ndarray,
    colors: np.ndarray,
    cameras: list[dict],
    *,
    centre: np.ndarray,
    cloud_radius: float,
    size: int,
    run_id: str,
    precomputed_example: bool,
    media_dir: Path,
) -> dict:
    centers = np.asarray(
        [c["camera_center_world"] for c in cameras if c.get("camera_center_world")],
        dtype=np.float64,
    )
    ring_radius = (
        float(np.max(np.linalg.norm(centers - centre, axis=1))) if len(centers) else cloud_radius
    )
    distance = render.framing_distance(max(cloud_radius, ring_radius)) * TRAJECTORY_MARGIN

    camera = render.orbit_camera(
        target=centre,
        distance=distance,
        azimuth_deg=0.0,
        elevation_deg=TRAJECTORY_ELEVATION_DEG,
        width=size,
        height=size,
    )
    image = render.render_points(points, colors, camera)
    if len(centers):
        image = render.mark_points(image, centers, camera)
    what = f"camera trajectory: {len(centers)} cameras"
    caption_text = _caption(what, run_id, precomputed_example=precomputed_example)
    image = render.add_caption(image, caption_text)
    filename = "trajectory.png"
    _save(image, media_dir / filename)
    return {
        "file": filename,
        "kind": "trajectory",
        "what": what,
        "caption": caption_text,
        "source": "normalized/points.ply + normalized/cameras.json",
        "num_cameras": len(centers),
    }


def build_media(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    size: int = 900,
    turntable_frames: int = 72,
    web_target: int = 40_000,
) -> dict:
    """Build presentation media for a stored run and write/return the `media.json` index.

    Reads only `normalized/` and `package/images/`; nothing is recomputed and no GPU is
    needed (FR-050). Raises :class:`ResultMissingOrIncompatibleError` (exit code 7) if
    `normalized/` is missing, mirroring the other read-only commands (`view`, `export`).
    """
    layout = runs.find_run_dir(run_dir)
    if not layout.normalized.is_dir():
        raise ResultMissingOrIncompatibleError(
            f"the run directory has no normalized/: {layout.root}",
            details="the result was not ingested or the run directory was not fully transferred",
        )

    manifest = read_json(layout.manifest)
    cameras_file = read_json(layout.normalized / "cameras.json")
    status_file = read_json(layout.normalized / "status.json")
    ingest_checks.check_run_id_matches(
        manifest,
        ("normalized/cameras.json", cameras_file),
        ("normalized/status.json", status_file),
    )

    run_id = manifest["run_id"]
    precomputed_example = bool(status_file.get("precomputed_example"))
    points, colors, _ = ply_io.read_ply(layout.normalized / "points.ply")
    cameras = list(cameras_file.get("cameras") or [])

    media_dir = Path(out_dir) if out_dir is not None else layout.root / MEDIA_DIRNAME
    media_dir.mkdir(parents=True, exist_ok=True)

    centre, radius = render.robust_extent(points)
    distance = render.framing_distance(radius)

    assets: list[dict] = []

    assets += _render_views(
        points,
        colors,
        centre=centre,
        distance=distance,
        size=size,
        run_id=run_id,
        precomputed_example=precomputed_example,
        media_dir=media_dir,
    )

    turntable_info = _render_turntable(
        points,
        colors,
        centre=centre,
        distance=distance,
        size=size,
        frames=turntable_frames,
        media_dir=media_dir,
    )
    turntable_what = (
        f"turntable animation, {turntable_frames} frames at "
        f"{TURNTABLE_ELEVATION_DEG:.0f}° elevation"
    )
    assets.append(
        {
            "file": "turntable.webp",
            "kind": "turntable",
            "what": turntable_what,
            "caption": _caption(turntable_what, run_id, precomputed_example=precomputed_example),
            "source": "normalized/points.ply",
            "frames": turntable_info["frames"],
            "fps": TURNTABLE_FPS,
            "bytes": turntable_info["webp_bytes"],
        }
    )
    assets.append(
        {
            "file": "turntable.mp4",
            "kind": "turntable",
            "what": turntable_what,
            "caption": _caption(turntable_what, run_id, precomputed_example=precomputed_example),
            "source": "normalized/points.ply",
            "frames": turntable_info["frames"],
            "fps": TURNTABLE_FPS,
            "bytes": turntable_info["mp4_bytes"],
        }
    )

    pair_assets, skipped_pairs = _render_pairs(
        cameras,
        points,
        colors,
        images_dir=layout.package_images,
        run_id=run_id,
        precomputed_example=precomputed_example,
        media_dir=media_dir,
    )
    assets += pair_assets

    assets.append(
        _render_trajectory(
            points,
            colors,
            cameras,
            centre=centre,
            cloud_radius=radius,
            size=size,
            run_id=run_id,
            precomputed_example=precomputed_example,
            media_dir=media_dir,
        )
    )

    web_cloud_asset: dict | None = None
    try:
        from v3d.decimate import write_web_cloud
    except ImportError:
        write_web_cloud = None  # type: ignore[assignment]

    if write_web_cloud is not None:
        web_info = write_web_cloud(
            layout.normalized / "points.ply", media_dir / "cloud_web.ply", target=web_target
        )
        what = f"decimated point cloud for the web ({web_info['label']})"
        web_cloud_asset = {
            "file": "cloud_web.ply",
            "kind": "web_cloud",
            "what": what,
            "caption": _caption(what, run_id, precomputed_example=precomputed_example),
            "source": "normalized/points.ply",
            "label": web_info["label"],
            "points_out": web_info["points_out"],
            "points_in": web_info["points_in"],
        }
        assets.append(web_cloud_asset)

    index = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "precomputed_example": precomputed_example,
        "generated_utc": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "source": "normalized/points.ply + normalized/cameras.json + package/images",
        "assets": assets,
    }
    if skipped_pairs:
        index["skipped_pairs"] = skipped_pairs

    write_json(media_dir / INDEX_NAME, index)
    return index
