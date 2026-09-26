"""Geometry of camera poses and points: quaternions, transform direction change, projection,
point validity and the outlier filter.

Data model: specs/001-video-3d-mvp/data-model.md §3.4, §3.5; plan — plan.md §6.

The v1 conventions are fixed by documents and are not discussed here
(:func:`v3d.artifacts.default_conventions`):

* poses are stored as ``world_to_camera``, formula ``x_cam = R @ x_world + t``;
* camera axes are OpenCV (+X right, +Y down, +Z forward), the world is right-handed;
* units are unknown, ``scale_status = not_determined``.

A reconstructor that outputs camera-to-world is brought to the contract by
:func:`world_to_camera_from_camera_to_world`; the contract itself does not change.

All functions are pure: input arrays are not modified, there are no files or global state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Written definition of a valid point (data-model.md §3.5). Stored in the diagnostics field
# `valid_point_definition` so that the meaning of num_points_valid is explicit.
VALID_POINT_DEFINITION = (
    "A point is valid if: (a) all three coordinates are finite (not NaN and not Inf); "
    "(b) the point made it into the reconstructor's result after its own confidence "
    "threshold; (c) the point was not discarded by our outlier filter."
)

#: Name of the v1 outlier filter, recorded in `filters_applied`.
OUTLIER_FILTER_NAME = "iqr_per_axis"

#: Default k value for the outlier filter.
DEFAULT_IQR_K = 3.0


def valid_point_definition() -> str:
    """Definition of a valid point as one string — for the `valid_point_definition` field."""
    return VALID_POINT_DEFINITION


# --- Quaternions and rotation matrices --------------------------------------------------


def quaternion_to_rotation_matrix(qvec: Any) -> np.ndarray:
    """Quaternion in COLMAP order ``(qw, qx, qy, qz)`` → 3×3 rotation matrix.

    The input is normalized; zero norm is a ``ValueError``, no rotation is defined by it.
    """
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion has zero or non-finite norm: rotation is undefined")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(R: Any) -> np.ndarray:
    """3×3 rotation matrix → quaternion ``(qw, qx, qy, qz)``.

    Stable branching scheme based on the matrix trace. The sign is canonicalized: ``qw >= 0``,
    so that a round trip is deterministic (q and -q define the same rotation).
    """
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


# --- Camera poses -----------------------------------------------------------------------


def camera_center_from_world_to_camera(R: Any, t: Any) -> np.ndarray:
    """Camera centre in the world: ``C = -Rᵀ t`` for a world_to_camera pair."""
    R_arr = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t_arr = np.asarray(t, dtype=np.float64).reshape(3)
    return -R_arr.T @ t_arr


def world_to_camera_from_camera_to_world(R_c2w: Any, t_c2w: Any) -> tuple[np.ndarray, np.ndarray]:
    """camera_to_world pair → world_to_camera pair: ``R = R_c2wᵀ``, ``t = -R_c2wᵀ t_c2w``.

    Needed by adapters of reconstructors that output camera-to-world: the storage contract
    (`transform_direction = world_to_camera`) does not change.
    """
    R_arr = np.asarray(R_c2w, dtype=np.float64).reshape(3, 3)
    t_arr = np.asarray(t_c2w, dtype=np.float64).reshape(3)
    R_w2c = R_arr.T
    return R_w2c, -R_w2c @ t_arr


# --- Projection ---------------------------------------------------------------------------


def project_points(
    points_world: Any,
    R: Any,
    t: Any,
    intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Projection of world points to pixels by a pinhole camera (``fx, fy, cx, cy``).

    The pose is given as a world_to_camera pair. Returns ``(uv, valid_mask)``: an (N, 2) array
    of pixel coordinates and a boolean mask of points strictly in front of the camera
    (``z > 0``) with finite coordinates. For invalid points the ``uv`` values are undefined.
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    R_arr = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t_arr = np.asarray(t, dtype=np.float64).reshape(3)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    cam = pts @ R_arr.T + t_arr
    z = cam[:, 2]
    in_front = np.isfinite(cam).all(axis=1) & (z > 0.0)

    safe_z = np.where(in_front, z, 1.0)
    uv = np.empty((pts.shape[0], 2), dtype=np.float64)
    uv[:, 0] = fx * cam[:, 0] / safe_z + cx
    uv[:, 1] = fy * cam[:, 1] / safe_z + cy
    valid_mask = in_front & np.isfinite(uv).all(axis=1)
    return uv, valid_mask


# --- Point validity and the outlier filter ----------------------------------------------


def is_valid_points(points: Any) -> np.ndarray:
    """Boolean array of length N: all three point coordinates are finite (not NaN, not Inf)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.isfinite(pts).all(axis=1)


def filter_outliers_iqr(
    points: Any,
    colors: Any | None = None,
    k: float = DEFAULT_IQR_K,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Outlier filter by interquartile range, separately for each axis.

    Non-finite points are dropped first (:func:`is_valid_points`). Then quartiles are computed
    per axis and a point is dropped if it falls outside ``[q1 - k*iqr, q3 + k*iqr]``
    on at least one axis. Point order is preserved, the result is deterministic.

    Returns ``(points_kept, colors_kept, info)``; ``info`` is the record for the diagnostics
    field `filters_applied`.
    """
    if k < 0.0:
        raise ValueError("outlier filter k must not be negative")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = None if colors is None else np.asarray(colors).reshape(len(pts), -1)

    num_input = int(len(pts))
    finite_mask = is_valid_points(pts)
    num_non_finite = int(num_input - int(finite_mask.sum()))
    finite_pts = pts[finite_mask]

    keep_finite = np.ones(len(finite_pts), dtype=bool)
    bounds: list[dict[str, float]] = []
    for axis in range(3):
        if len(finite_pts) == 0:
            bounds.append({"axis": axis, "lower": float("nan"), "upper": float("nan")})
            continue
        q1, q3 = np.percentile(finite_pts[:, axis], [25.0, 75.0])
        iqr = float(q3 - q1)
        lower = float(q1) - k * iqr
        upper = float(q3) + k * iqr
        bounds.append({"axis": axis, "lower": lower, "upper": upper})
        keep_finite &= (finite_pts[:, axis] >= lower) & (finite_pts[:, axis] <= upper)

    keep = np.zeros(num_input, dtype=bool)
    keep[np.flatnonzero(finite_mask)[keep_finite]] = True

    points_kept = pts[keep]
    colors_kept = None if cols is None else cols[keep]
    info = {
        "name": OUTLIER_FILTER_NAME,
        "k": float(k),
        "num_points_before": num_input,
        "num_points_after": int(len(points_kept)),
        "num_non_finite_removed": num_non_finite,
        "num_outliers_removed": int(num_input - num_non_finite - len(points_kept)),
        "bounds_per_axis": bounds,
    }
    return points_kept, colors_kept, info
