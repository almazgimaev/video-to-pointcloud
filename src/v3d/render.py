"""CPU rendering of a normalized point cloud for the README and the project page (FR-050).

No GPU, no OpenGL, no plotting library: points are projected with a pinhole camera and splatted
into an image with a z-buffer, rendered at a higher resolution and downscaled for smooth edges.
A mild depth shading darkens far points so the shape reads as three-dimensional.

Cameras follow the project convention (world_to_camera, OpenCV axes: +X right, +Y down,
+Z forward). Synthetic orbit cameras assume the aligned world of data-model §3.4a: +Y up, object
at the origin, first reconstructed camera in front (+Z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (14, 16, 20)
CAPTION_BACKGROUND = (24, 27, 33)
CAPTION_TEXT = (196, 202, 212)
CAMERA_MARK = (255, 107, 74)


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: world_to_camera pose and intrinsics in pixels of ``width × height``."""

    R: np.ndarray
    t: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def look_at(position: Any, target: Any, up: Any = (0.0, 1.0, 0.0)) -> np.ndarray:
    """world_to_camera rotation of an OpenCV camera at ``position`` looking at ``target``."""
    p = np.asarray(position, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - p
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(x) < 1e-9:  # looking straight along "up": pick any right vector
        x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def orbit_camera(
    *,
    target: Any = (0.0, 0.0, 0.0),
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    fov_deg: float = 35.0,
) -> Camera:
    """Camera on a sphere around ``target`` in the aligned world (+Y up).

    Azimuth 0 is the front (+Z side, where the first reconstructed camera stands); positive
    azimuth turns counter-clockwise seen from above; positive elevation looks from above.
    """
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    direction = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    tgt = np.asarray(target, dtype=np.float64)
    position = tgt + distance * direction
    R = look_at(position, tgt)
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    return Camera(R, -R @ position, f, f, width / 2.0, height / 2.0, width, height)


def camera_from_record(record: dict) -> Camera:
    """Camera of a reconstructed frame from a ``cameras.json`` entry (needs intrinsics)."""
    intr = record.get("intrinsics")
    if not intr:
        raise ValueError(f"camera {record.get('frame_id')} has no intrinsics")
    width, height = (int(v) for v in record["image_size"])
    return Camera(
        np.asarray(record["R_world_to_camera"], dtype=np.float64),
        np.asarray(record["t_world_to_camera"], dtype=np.float64),
        float(intr["fx"]),
        float(intr["fy"]),
        float(intr["cx"]),
        float(intr["cy"]),
        width,
        height,
    )


def robust_extent(points: Any, *, percentile: float = 98.0) -> tuple[np.ndarray, float]:
    """Centre and radius that enclose most of the cloud, ignoring a few far outliers."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if not len(P):
        return np.zeros(3), 1.0
    centre = np.median(P, axis=0)
    radius = float(np.percentile(np.linalg.norm(P - centre, axis=1), percentile))
    return centre, max(radius, 1e-6)


def framing_distance(radius: float, *, fov_deg: float = 35.0, margin: float = 1.12) -> float:
    """Distance at which a sphere of ``radius`` fills the view with a small margin."""
    return radius * margin / np.sin(np.radians(fov_deg) / 2.0)


def _disk_offsets(radius: float) -> np.ndarray:
    r = max(radius, 0.5)
    span = int(np.ceil(r))
    dy, dx = np.mgrid[-span : span + 1, -span : span + 1]
    keep = dx**2 + dy**2 <= r**2 + 1e-9
    return np.stack([dx[keep], dy[keep]], axis=1)


def render_points(
    points: Any,
    colors: Any,
    camera: Camera,
    *,
    point_size: float = 2.2,
    background: tuple[int, int, int] = BACKGROUND,
    depth_shading: float = 0.35,
    supersample: int = 2,
) -> Image.Image:
    """Render coloured points seen by ``camera`` into an RGB image of the camera's size.

    ``point_size`` is the splat diameter in output pixels. The nearest point wins per pixel
    (z-buffer); ``depth_shading`` darkens the farthest points by up to that fraction.
    """
    ss = max(int(supersample), 1)
    W, H = camera.width * ss, camera.height * ss
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    C = np.asarray(colors, dtype=np.float64).reshape(-1, 3)

    cam = P @ camera.R.T + camera.t
    z = cam[:, 2]
    ok = np.isfinite(cam).all(axis=1) & (z > 1e-6)
    cam, z, C = cam[ok], z[ok], C[ok]
    image = np.empty((H, W, 3), dtype=np.uint8)
    image[:] = background
    if not len(z):
        return Image.fromarray(image).resize((camera.width, camera.height), Image.LANCZOS)

    u = (camera.fx * cam[:, 0] / z + camera.cx) * ss
    v = (camera.fy * cam[:, 1] / z + camera.cy) * ss

    if depth_shading > 0 and z.max() > z.min():
        near, far = np.percentile(z, 2), np.percentile(z, 98)
        depth = np.clip((z - near) / max(far - near, 1e-9), 0.0, 1.0)
        C = C * (1.0 - depth_shading * depth)[:, None]

    offsets = _disk_offsets(point_size * ss / 2.0)
    px = (np.round(u)[:, None] + offsets[None, :, 0]).astype(np.int64).ravel()
    py = (np.round(v)[:, None] + offsets[None, :, 1]).astype(np.int64).ravel()
    depth_all = np.repeat(z, len(offsets))
    src = np.repeat(np.arange(len(z)), len(offsets))
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    pix = py[inside] * W + px[inside]
    depth_all, src = depth_all[inside], src[inside]

    # z-buffer: for every pixel keep the splat with the smallest depth.
    order = np.lexsort((depth_all, pix))
    pix, src = pix[order], src[order]
    first = np.ones(len(pix), dtype=bool)
    first[1:] = pix[1:] != pix[:-1]
    flat = image.reshape(-1, 3)
    flat[pix[first]] = np.clip(C[src[first]], 0, 255).astype(np.uint8)

    return Image.fromarray(image).resize((camera.width, camera.height), Image.LANCZOS)


def mark_points(
    image: Image.Image,
    world_points: Any,
    camera: Camera,
    *,
    radius: float = 3.5,
    color: tuple[int, int, int] = CAMERA_MARK,
) -> Image.Image:
    """Draw small discs at the projections of ``world_points`` (e.g. camera centres)."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    P = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    cam = P @ camera.R.T + camera.t
    for x, y, zc in cam:
        if zc <= 1e-6:
            continue
        u = camera.fx * x / zc + camera.cx
        v = camera.fy * y / zc + camera.cy
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color)
    return out


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without scalable default font
        return ImageFont.load_default()


def add_caption(image: Image.Image, text: str, *, height: int = 30) -> Image.Image:
    """Append a caption strip under the image (FR-051: every asset names its source)."""
    out = Image.new("RGB", (image.width, image.height + height), CAPTION_BACKGROUND)
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text(
        (12, image.height + height // 2), text, fill=CAPTION_TEXT, font=_font(14), anchor="lm"
    )
    return out


def side_by_side(left: Image.Image, right: Image.Image, *, gap: int = 6) -> Image.Image:
    """Two images of equal height next to each other."""
    height = max(left.height, right.height)
    out = Image.new("RGB", (left.width + gap + right.width, height), BACKGROUND)
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    return out
