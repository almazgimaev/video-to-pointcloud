"""Unit tests of geometry (T014): quaternions, poses, projection, validity, filter."""

from __future__ import annotations

import numpy as np
import pytest

from v3d.geometry import (
    OUTLIER_FILTER_NAME,
    VALID_POINT_DEFINITION,
    camera_center_from_world_to_camera,
    filter_outliers_iqr,
    is_valid_points,
    project_points,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    valid_point_definition,
    world_to_camera_from_camera_to_world,
)

INTRINSICS = {"model": "PINHOLE", "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0}

# Several distinct rotations for round-trip checks.
QUATERNIONS = [
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.5, 0.5, 0.5, 0.5),
    (0.9238795, 0.3826834, 0.0, 0.0),
    (0.1830127, -0.6830127, 0.6830127, 0.1830127),
]


def look_at_world_to_camera(
    center: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World_to_camera pose of a camera at `center` looking at `target` (OpenCV axes)."""
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])  # rows are the camera axes in the world
    return R, -R @ center


# --- Quaternions ---------------------------------------------------------------------------


@pytest.mark.parametrize("qvec", QUATERNIONS)
def test_quaternion_roundtrip_is_canonical(qvec: tuple[float, float, float, float]) -> None:
    q = np.array(qvec) / np.linalg.norm(qvec)
    if q[0] < 0:
        q = -q
    back = rotation_matrix_to_quaternion(quaternion_to_rotation_matrix(q))
    assert back[0] >= 0.0
    np.testing.assert_allclose(back, q, atol=1e-12)


@pytest.mark.parametrize("qvec", QUATERNIONS)
def test_quaternion_to_rotation_matrix_is_orthonormal(
    qvec: tuple[float, float, float, float],
) -> None:
    R = quaternion_to_rotation_matrix(qvec)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_input_is_normalized() -> None:
    R_scaled = quaternion_to_rotation_matrix((2.0, 0.0, 0.0, 2.0))
    R_unit = quaternion_to_rotation_matrix((1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)))
    np.testing.assert_allclose(R_scaled, R_unit, atol=1e-12)


def test_zero_quaternion_rejected() -> None:
    with pytest.raises(ValueError):
        quaternion_to_rotation_matrix((0.0, 0.0, 0.0, 0.0))


def test_negated_quaternion_gives_same_matrix_and_canonical_sign() -> None:
    q = np.array([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(q), quaternion_to_rotation_matrix(-q), atol=1e-12
    )
    np.testing.assert_allclose(
        rotation_matrix_to_quaternion(quaternion_to_rotation_matrix(-q)), q, atol=1e-12
    )


# --- Camera poses ---------------------------------------------------------------------------


def test_camera_centers_on_synthetic_ring() -> None:
    """Cameras on a circle look at the centre: recovered centres match the given ones."""
    radius = 3.0
    target = np.array([0.0, 0.0, 0.0])
    for angle in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
        center = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.5])
        R, t = look_at_world_to_camera(center, target)
        np.testing.assert_allclose(camera_center_from_world_to_camera(R, t), center, atol=1e-12)


def test_camera_looks_at_target_along_plus_z() -> None:
    center = np.array([2.0, -1.0, 0.5])
    target = np.array([0.0, 0.0, 0.0])
    R, t = look_at_world_to_camera(center, target)
    target_in_camera = R @ target + t
    assert target_in_camera[2] > 0.0
    np.testing.assert_allclose(target_in_camera[:2], [0.0, 0.0], atol=1e-12)


def test_camera_to_world_conversion_is_an_involution() -> None:
    """Applying the transform twice returns the original pair."""
    R_c2w = quaternion_to_rotation_matrix((0.1830127, -0.6830127, 0.6830127, 0.1830127))
    t_c2w = np.array([1.5, -2.0, 0.25])
    R_w2c, t_w2c = world_to_camera_from_camera_to_world(R_c2w, t_c2w)
    R_back, t_back = world_to_camera_from_camera_to_world(R_w2c, t_w2c)
    np.testing.assert_allclose(R_back, R_c2w, atol=1e-12)
    np.testing.assert_allclose(t_back, t_c2w, atol=1e-12)


def test_camera_to_world_translation_is_the_camera_center() -> None:
    """t of a camera_to_world pair is the camera centre; after conversion it is also -Rᵀt."""
    R_c2w = quaternion_to_rotation_matrix((0.9238795, 0.3826834, 0.0, 0.0))
    t_c2w = np.array([4.0, 0.0, -1.0])
    R_w2c, t_w2c = world_to_camera_from_camera_to_world(R_c2w, t_c2w)
    np.testing.assert_allclose(camera_center_from_world_to_camera(R_w2c, t_w2c), t_c2w, atol=1e-12)


# --- Projection -------------------------------------------------------------------------------


def test_point_on_optical_axis_projects_to_principal_point() -> None:
    R = np.eye(3)
    t = np.zeros(3)
    uv, valid = project_points([[0.0, 0.0, 5.0]], R, t, INTRINSICS)
    assert bool(valid[0])
    np.testing.assert_allclose(uv[0], [INTRINSICS["cx"], INTRINSICS["cy"]], atol=1e-12)


def test_point_behind_camera_is_invalid() -> None:
    R = np.eye(3)
    t = np.zeros(3)
    points = [[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0]]
    _, valid = project_points(points, R, t, INTRINSICS)
    assert valid.tolist() == [True, False, False]


def test_projection_offset_matches_pinhole_model() -> None:
    R = np.eye(3)
    t = np.zeros(3)
    uv, valid = project_points([[1.0, 2.0, 4.0]], R, t, INTRINSICS)
    assert bool(valid[0])
    expected = [800.0 * 1.0 / 4.0 + 320.0, 800.0 * 2.0 / 4.0 + 240.0]
    np.testing.assert_allclose(uv[0], expected, atol=1e-12)


def test_projection_uses_world_to_camera_pose() -> None:
    center = np.array([0.0, -4.0, 0.0])
    R, t = look_at_world_to_camera(center, np.zeros(3))
    uv, valid = project_points([[0.0, 0.0, 0.0]], R, t, INTRINSICS)
    assert bool(valid[0])
    np.testing.assert_allclose(uv[0], [INTRINSICS["cx"], INTRINSICS["cy"]], atol=1e-9)


def test_non_finite_points_are_invalid_in_projection() -> None:
    _, valid = project_points(
        [[np.nan, 0.0, 1.0], [0.0, np.inf, 1.0], [0.0, 0.0, 1.0]],
        np.eye(3),
        np.zeros(3),
        INTRINSICS,
    )
    assert valid.tolist() == [False, False, True]


# --- Point validity ------------------------------------------------------------------------


def test_is_valid_points_rejects_nan_and_inf() -> None:
    points = [
        [0.0, 0.0, 0.0],
        [np.nan, 1.0, 1.0],
        [1.0, np.inf, 1.0],
        [1.0, 1.0, -np.inf],
        [1e9, -1e9, 0.5],
    ]
    assert is_valid_points(points).tolist() == [True, False, False, False, True]


def test_is_valid_points_on_empty_input() -> None:
    assert is_valid_points(np.empty((0, 3))).tolist() == []


def test_valid_point_definition_mentions_all_three_conditions() -> None:
    text = valid_point_definition()
    assert text == VALID_POINT_DEFINITION
    assert "finite" in text
    assert "confidence" in text
    assert "outlier" in text


# --- Outlier filter ----------------------------------------------------------------------


def _cluster(n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(loc=0.0, scale=0.1, size=(n, 3))


def test_iqr_filter_removes_explicit_outlier_and_keeps_normal_points() -> None:
    normal = _cluster()
    points = np.vstack([normal, [[1000.0, 0.0, 0.0]]])
    kept, colors, info = filter_outliers_iqr(points, k=3.0)
    assert colors is None
    assert len(kept) == len(normal)
    np.testing.assert_allclose(kept, normal)
    assert info["num_points_before"] == len(normal) + 1
    assert info["num_points_after"] == len(normal)
    assert info["num_outliers_removed"] == 1


def test_iqr_filter_drops_non_finite_and_reports_their_count() -> None:
    normal = _cluster()
    points = np.vstack([normal, [[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [500.0, 0.0, 0.0]]])
    kept, _, info = filter_outliers_iqr(points, k=3.0)
    assert len(kept) == len(normal)
    assert info["name"] == OUTLIER_FILTER_NAME
    assert info["k"] == 3.0
    assert info["num_points_before"] == len(normal) + 3
    assert info["num_points_after"] == len(normal)
    assert info["num_non_finite_removed"] == 2
    assert info["num_outliers_removed"] == 1
    assert [b["axis"] for b in info["bounds_per_axis"]] == [0, 1, 2]
    assert all(np.isfinite([b["lower"], b["upper"]]).all() for b in info["bounds_per_axis"])


def test_iqr_filter_keeps_colors_aligned_with_points() -> None:
    points = np.vstack([_cluster(8), [[0.0, 0.0, 900.0]]])
    colors = np.arange(len(points) * 3, dtype=np.uint8).reshape(-1, 3)
    kept, kept_colors, _ = filter_outliers_iqr(points, colors, k=3.0)
    assert kept_colors is not None
    assert len(kept_colors) == len(kept)
    np.testing.assert_array_equal(kept_colors, colors[:-1])


def test_iqr_filter_is_deterministic() -> None:
    points = np.vstack([_cluster(), [[1000.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]])
    first_pts, _, first_info = filter_outliers_iqr(points, k=3.0)
    second_pts, _, second_info = filter_outliers_iqr(points, k=3.0)
    np.testing.assert_array_equal(first_pts, second_pts)
    assert first_info == second_info


def test_iqr_filter_does_not_mutate_input() -> None:
    points = np.vstack([_cluster(8), [[1000.0, 0.0, 0.0]]])
    snapshot = points.copy()
    filter_outliers_iqr(points, k=3.0)
    np.testing.assert_array_equal(points, snapshot)


def test_iqr_filter_on_empty_input() -> None:
    kept, colors, info = filter_outliers_iqr(np.empty((0, 3)))
    assert len(kept) == 0
    assert colors is None
    assert info["num_points_before"] == 0
    assert info["num_points_after"] == 0
    assert info["num_non_finite_removed"] == 0


def test_iqr_filter_rejects_negative_k() -> None:
    with pytest.raises(ValueError):
        filter_outliers_iqr(_cluster(8), k=-1.0)
