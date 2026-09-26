"""Tests of the CPU point-cloud renderer (T081).

Requirements: spec.md FR-050…FR-052; tasks.md T081.

Synthetic data only: no run directory is needed here, only :mod:`v3d.render` itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from v3d.render import (
    BACKGROUND,
    Camera,
    add_caption,
    camera_from_record,
    orbit_camera,
    render_points,
)


def _brightest_pixel(image: Image.Image) -> tuple[int, int]:
    arr = np.asarray(image.convert("L"), dtype=np.int64)
    flat_index = int(np.argmax(arr))
    y, x = divmod(flat_index, arr.shape[1])
    return x, y


# --- Projection lands on the principal point --------------------------------------------


def test_point_on_optical_axis_lands_on_principal_point() -> None:
    camera = Camera(
        R=np.eye(3),
        t=np.array([0.0, 0.0, 5.0]),
        fx=200.0,
        fy=200.0,
        cx=64.0,
        cy=48.0,
        width=128,
        height=96,
    )
    points = np.array([[0.0, 0.0, 0.0]])
    colors = np.array([[255.0, 255.0, 255.0]])

    image = render_points(points, colors, camera, point_size=1.5, depth_shading=0.0)
    x, y = _brightest_pixel(image)

    assert abs(x - camera.cx) <= 1
    assert abs(y - camera.cy) <= 1


# --- Z-buffer: nearer point wins ----------------------------------------------------------


def test_zbuffer_keeps_the_nearer_point() -> None:
    camera = Camera(
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.0]),
        fx=200.0,
        fy=200.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
    )
    # Both points lie on the same ray through the principal point; red is nearer.
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]])
    colors = np.array([[220.0, 20.0, 20.0], [20.0, 20.0, 220.0]])

    image = render_points(points, colors, camera, point_size=3.0, depth_shading=0.0)
    pixel = image.getpixel((32, 32))

    assert pixel[0] > pixel[2]  # red channel dominates over blue


# --- Determinism ---------------------------------------------------------------------------


def test_render_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 3))
    colors = rng.integers(0, 256, size=(500, 3))
    camera = orbit_camera(distance=5.0, azimuth_deg=20.0, elevation_deg=10.0, width=80, height=60)

    first = render_points(points, colors, camera)
    second = render_points(points, colors, camera)

    assert np.array_equal(np.asarray(first), np.asarray(second))


# --- orbit_camera geometry -----------------------------------------------------------------


def test_orbit_camera_azimuth_zero_sits_on_positive_z() -> None:
    camera = orbit_camera(distance=10.0, azimuth_deg=0.0, elevation_deg=0.0, width=64, height=64)
    # world_to_camera: cam = R @ world + t; camera position in world is -R^T @ t.
    position = -camera.R.T @ camera.t
    assert position[2] > 0
    assert abs(position[0]) < 1e-6
    assert abs(position[1]) < 1e-6

    # Looking at the origin: the origin must project near the principal point.
    points = np.array([[0.0, 0.0, 0.0]])
    colors = np.array([[255.0, 255.0, 255.0]])
    image = render_points(points, colors, camera, depth_shading=0.0)
    x, y = _brightest_pixel(image)
    assert abs(x - camera.cx) <= 1
    assert abs(y - camera.cy) <= 1


def test_orbit_camera_high_elevation_looks_down() -> None:
    camera = orbit_camera(distance=10.0, azimuth_deg=0.0, elevation_deg=89.0, width=64, height=64)
    position = -camera.R.T @ camera.t
    # Almost straight above the target on the Y axis (world is +Y up, §3.4a).
    assert position[1] > 9.0


# --- Points behind the camera and empty input ------------------------------------------


def test_points_behind_camera_are_not_drawn() -> None:
    camera = Camera(
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.0]),
        fx=100.0,
        fy=100.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
    )
    points = np.array([[0.0, 0.0, -5.0]])  # negative z: behind the camera
    colors = np.array([[255.0, 0.0, 0.0]])

    image = render_points(points, colors, camera)
    arr = np.asarray(image)

    # Every pixel equals the background colour: nothing was drawn.
    background = np.asarray(image.getpixel((0, 0)))
    assert np.array_equal(arr, np.broadcast_to(background, arr.shape))


def test_empty_input_gives_background_image() -> None:
    camera = orbit_camera(distance=5.0, azimuth_deg=0.0, elevation_deg=0.0, width=32, height=24)
    points = np.zeros((0, 3))
    colors = np.zeros((0, 3))

    image = render_points(points, colors, camera)

    assert image.size == (32, 24)
    arr = np.asarray(image)
    assert np.array_equal(arr, np.broadcast_to(np.array(BACKGROUND, dtype=np.uint8), arr.shape))


# --- add_caption ---------------------------------------------------------------------------


def test_add_caption_increases_height_and_keeps_image_on_top() -> None:
    base = Image.new("RGB", (40, 30), (10, 20, 30))
    captioned = add_caption(base, "hello", height=20)

    assert captioned.size == (40, 50)
    top = captioned.crop((0, 0, 40, 30))
    assert np.array_equal(np.asarray(top), np.asarray(base))


# --- camera_from_record ----------------------------------------------------------------


def test_camera_from_record_needs_intrinsics() -> None:
    record = {
        "frame_id": "0001_000000",
        "image_size": [64, 48],
        "intrinsics": None,
        "R_world_to_camera": np.eye(3).tolist(),
        "t_world_to_camera": [0.0, 0.0, 0.0],
    }
    with pytest.raises(ValueError):
        camera_from_record(record)
