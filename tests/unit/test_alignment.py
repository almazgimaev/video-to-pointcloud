"""World alignment: vertical from the camera ring, re-centring, quality gate (FR-045…FR-048).

Scenes are built in a canonical frame (up = +Z, object at the origin) and then moved by a known
rigid transform, the way a reconstructor anchors its world to an arbitrary first camera. The
alignment must bring the canonical up to +Y and the object centre to the origin.
"""

from __future__ import annotations

import numpy as np
import pytest

from v3d import geometry as g

GATE = {
    "max_planarity": 0.25,
    "min_arc_deg": 180.0,
    "min_sign_agreement": 0.3,
    "min_cameras": 6,
}


def _look_at(centre: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """world_to_camera rotation for an OpenCV camera at ``centre`` looking at ``target``."""
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, up)  # OpenCV: +X right, +Y down, +Z forward, det = +1
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _rot(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _orbit(
    *,
    n: int = 24,
    radius: float = 1.0,
    height: float = 0.6,
    arc_deg: float = 360.0,
    height_jitter: float = 0.0,
    roll_deg: float = 0.0,
    wave: float = 0.0,
    below: bool = False,
    seed: int = 0,
    tilt=((1.0, 1.0, 0.3), 47.0),
    offset=(0.4, -1.2, 2.5),
) -> dict:
    """Canonical orbit (up +Z) moved by a known rigid transform."""
    rng = np.random.default_rng(seed)
    up = np.array([0.0, 0.0, 1.0])
    step = arc_deg / n if arc_deg >= 360 else arc_deg / (n - 1)
    centres, rotations = [], []
    for i in range(n):
        phi = np.radians(i * step)
        z = (-height if below else height) + rng.normal(0, height_jitter) + wave * np.sin(2 * phi)
        c = np.array([radius * np.cos(phi), radius * np.sin(phi), z])
        R = _look_at(c, np.zeros(3), up)
        if roll_deg:
            R = _rot([0, 0, 1], rng.normal(0, roll_deg)) @ R
        centres.append(c)
        rotations.append(R)
    G = _rot(*tilt)
    o = np.asarray(offset, dtype=float)
    moved_c = [G @ c + o for c in centres]
    moved_R = [R @ G.T for R in rotations]
    return {
        "centres": np.array(moved_c),
        "rotations": np.array(moved_R),
        "true_up": G @ up,
        "true_centre": o,
    }


def _align(scene: dict, points=None) -> dict:
    return g.estimate_alignment(scene["centres"], scene["rotations"], points=points, **GATE)


def _angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# --- Vertical and centre -------------------------------------------------------------------


def test_tilted_clean_orbit_brings_true_up_to_plus_y() -> None:
    """SC-015: a known tilt of the whole scene is undone."""
    scene = _orbit()
    result = _align(scene)
    T = np.array(result["transform"])
    assert result["status"] == "estimated"
    assert _angle_deg(T[:3, :3] @ scene["true_up"], [0, 1, 0]) < 0.01


@pytest.mark.parametrize("seed", range(5))
def test_handheld_orbit_with_jitter_and_roll_is_recovered(seed: int) -> None:
    """SC-015 under A-13: uneven height and an unlevelled horizon still give a good vertical."""
    scene = _orbit(height_jitter=0.08, roll_deg=12.0, seed=seed)
    result = _align(scene)
    T = np.array(result["transform"])
    assert result["status"] == "estimated"
    assert _angle_deg(T[:3, :3] @ scene["true_up"], [0, 1, 0]) < 5.0


def test_roll_does_not_change_the_estimate() -> None:
    """The vertical comes from camera positions; per-frame roll only chooses the sign."""
    base = _align(_orbit(seed=3))
    rolled = _align(_orbit(seed=3, roll_deg=25.0))
    assert np.allclose(
        np.array(base["transform"])[:3, :3] @ [0, 1, 0],
        np.array(rolled["transform"])[:3, :3] @ [0, 1, 0],
        atol=1e-9,
    )


def test_object_centre_moves_to_the_origin() -> None:
    scene = _orbit()
    T = np.array(_align(scene)["transform"])
    centre = g.apply_transform_to_points(T, scene["true_centre"][None, :])[0]
    assert np.linalg.norm(centre) < 1e-6


def test_centre_ignores_background_points() -> None:
    """The centre comes from the optical axes, so a heavy background does not move it."""
    scene = _orbit()
    rng = np.random.default_rng(1)
    background = scene["true_centre"] + np.array([3.0, 0.0, 0.0]) + rng.normal(0, 0.5, (5000, 3))
    result = _align(scene, points=background)
    T = np.array(result["transform"])
    assert result["centre_method"] == g.CENTRE_OPTICAL_AXES
    centre = g.apply_transform_to_points(T, scene["true_centre"][None, :])[0]
    assert np.linalg.norm(centre) < 1e-6


# --- Sign and yaw --------------------------------------------------------------------------


def test_cameras_above_the_object_end_up_above_it() -> None:
    """Shooting from above: after alignment the cameras have positive height."""
    scene = _orbit(height=0.6)
    T = np.array(_align(scene)["transform"])
    heights = g.apply_transform_to_points(T, scene["centres"])[:, 1]
    assert heights.mean() > 0


def test_sign_follows_camera_up_not_camera_height() -> None:
    """Shooting from below: the vertical still points to the cameras' up, not towards them."""
    scene = _orbit(below=True)
    result = _align(scene)
    T = np.array(result["transform"])
    assert result["status"] == "estimated"
    assert _angle_deg(T[:3, :3] @ scene["true_up"], [0, 1, 0]) < 0.01
    assert g.apply_transform_to_points(T, scene["centres"])[:, 1].mean() < 0


def test_first_camera_is_in_front() -> None:
    """FR-048: rotation about the vertical is fixed by putting the first camera on +Z."""
    scene = _orbit()
    T = np.array(_align(scene)["transform"])
    first = g.apply_transform_to_points(T, scene["centres"][:1])[0]
    assert abs(first[0]) < 1e-9
    assert first[2] > 0


# --- Quality gate --------------------------------------------------------------------------


def test_partial_arc_is_rejected() -> None:
    """SC-016: a quarter orbit does not support a reliable vertical."""
    result = _align(_orbit(arc_deg=90.0))
    assert result["status"] == "not_estimated"
    assert "min_arc_deg" in result["reason"]
    assert np.allclose(np.array(result["transform"])[:3, :3], np.eye(3))


def test_non_planar_path_is_rejected() -> None:
    """SC-016: a strongly wavy path is not a ring."""
    result = _align(_orbit(wave=0.9))
    assert result["status"] == "not_estimated"
    assert "max_planarity" in result["reason"]


def test_too_few_cameras_are_rejected() -> None:
    result = _align(_orbit(n=4))
    assert result["status"] == "not_estimated"
    assert "min_cameras" in result["reason"]


def test_rejected_estimate_still_recentres() -> None:
    """FR-047: without a vertical the result is at least centred on the object."""
    scene = _orbit(arc_deg=90.0)
    T = np.array(_align(scene)["transform"])
    assert np.linalg.norm(T[:3, 3]) > 0


def test_record_declares_estimate_and_thresholds() -> None:
    """FR-046: method, quality, gate and a note that this is not a measurement."""
    result = _align(_orbit())
    assert result["method"] == g.ALIGNMENT_METHOD
    assert result["gate"]["thresholds_preliminary"] is True
    assert "not measured" in result["note"]
    assert set(result["quality"]) == {
        "num_cameras",
        "planarity",
        "arc_coverage_deg",
        "sign_agreement",
    }


# --- Transform properties ------------------------------------------------------------------


def test_transform_is_rigid_and_invertible() -> None:
    """FR-046: no scaling, and the original frame is recoverable exactly."""
    T = np.array(_align(_orbit(roll_deg=10.0, height_jitter=0.05))["transform"])
    A = T[:3, :3]
    assert np.allclose(A @ A.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(A), 1.0)
    assert np.allclose(g.invert_transform(T) @ T, np.eye(4), atol=1e-12)


def test_poses_and_points_stay_consistent() -> None:
    """A world point projects identically before and after alignment."""
    scene = _orbit()
    T = np.array(_align(scene)["transform"])
    point = scene["true_centre"] + np.array([0.1, -0.05, 0.2])
    moved = g.apply_transform_to_points(T, point[None, :])[0]
    for R, C in zip(scene["rotations"], scene["centres"], strict=True):
        t = -R @ C
        R2, t2 = g.apply_transform_to_pose(T, R, t)
        assert np.allclose(R @ point + t, R2 @ moved + t2, atol=1e-9)


@pytest.mark.parametrize(("arc", "expected"), [(360.0, 345.0), (180.0, 180.0), (90.0, 90.0)])
def test_arc_coverage_is_measured_around_the_object(arc: float, expected: float) -> None:
    """A partial orbit must not look bigger than it is (measured around the object centre)."""
    scene = _orbit(arc_deg=arc)
    _, normal, _ = g.fit_ring_plane(scene["centres"])
    coverage = g.arc_coverage_deg(scene["centres"], scene["true_centre"], normal)
    assert coverage == pytest.approx(expected, abs=1.0)


def test_centroid_would_overestimate_a_partial_arc() -> None:
    """Documents why the object centre is used: around the centroid a half orbit inflates."""
    scene = _orbit(arc_deg=180.0)
    centroid, normal, _ = g.fit_ring_plane(scene["centres"])
    assert g.arc_coverage_deg(scene["centres"], centroid, normal) > 220.0
