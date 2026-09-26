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


# --- World alignment (FR-045…FR-048, plan §13.1) -----------------------------------------
#
# The reconstructor anchors its world to the first camera, so the result is tilted by however
# the phone was held. There is no gravity measurement, so "up" is *estimated*: a handheld orbit
# keeps the camera centres close to one plane, and the normal of that plane is the vertical.
# Camera positions do not depend on per-frame roll, so an unlevelled horizon does not bias the
# estimate; per-frame "up" vectors only choose the sign of the normal.

ALIGNMENT_METHOD = "camera_ring_plane"
YAW_RULE = "first_camera_in_front"
CENTRE_OPTICAL_AXES = "optical_axes_nearest_point"
CENTRE_POINTS_MEDIAN = "points_median"
CENTRE_CAMERA_CENTROID = "camera_centroid"

ALIGNMENT_NOTE = (
    "Vertical estimated from the camera trajectory, not measured: the reconstruction contains "
    "no gravity information. Rotation about the vertical is arbitrary and fixed by the rule "
    "'first camera in front'. Scale is not determined."
)

_WORLD_UP = np.array([0.0, 1.0, 0.0])
_WORLD_FRONT = np.array([0.0, 0.0, 1.0])


def camera_up_vectors(rotations_w2c: Any) -> np.ndarray:
    """Per-camera "up" direction in the world: ``-Rᵀ·e_y`` (OpenCV +Y points down)."""
    R = np.asarray(rotations_w2c, dtype=np.float64).reshape(-1, 3, 3)
    return -R.transpose(0, 2, 1) @ np.array([0.0, 1.0, 0.0])


def camera_forward_vectors(rotations_w2c: Any) -> np.ndarray:
    """Per-camera viewing direction in the world: ``Rᵀ·e_z``."""
    R = np.asarray(rotations_w2c, dtype=np.float64).reshape(-1, 3, 3)
    return R.transpose(0, 2, 1) @ np.array([0.0, 0.0, 1.0])


def fit_ring_plane(centres: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plane through the camera centres: ``(centroid, unit normal, singular values)``.

    Singular values are in descending order; ``s[2] / s[1]`` measures how far the orbit leaves
    the plane (0 for a perfectly planar orbit).
    """
    C = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    centroid = C.mean(axis=0)
    _, s, vt = np.linalg.svd(C - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    return centroid, normal, s


def arc_coverage_deg(centres: Any, around: Any, normal: Any) -> float:
    """Angular span of the orbit around the object, measured in the orbit plane.

    Computed as 360° minus the largest empty gap between consecutive camera azimuths, so a full
    orbit gives close to 360° and a half orbit close to 180°, regardless of camera order.

    ``around`` must be the object centre, not the centroid of the cameras: for a partial orbit
    the centroid slides inside the arc and a half orbit would look like about 240°.
    """
    C = np.asarray(centres, dtype=np.float64).reshape(-1, 3) - np.asarray(around, dtype=float)
    n = np.asarray(normal, dtype=np.float64)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    angles = np.sort(np.degrees(np.arctan2(C @ e2, C @ e1)) % 360.0)
    if len(angles) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([angles, [angles[0] + 360.0]]))
    return float(360.0 - gaps.max())


def nearest_point_to_rays(origins: Any, directions: Any) -> np.ndarray | None:
    """Least-squares point closest to all rays; ``None`` if the rays are (nearly) parallel.

    Every camera of an orbit looks at the object, so this point is the object centre. It does not
    depend on the point cloud, whose background may dominate.
    """
    starts = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(starts, dirs, strict=True):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    if np.linalg.cond(A) > 1e8:
        return None
    return np.linalg.solve(A, b)


def rotation_between(a: Any, b: Any) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` to unit vector ``b`` (Rodrigues)."""
    u = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    v = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    c = float(np.clip(u @ v, -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # Opposite vectors: rotate by 180° about any axis orthogonal to u.
        helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.cross(u, v)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def rigid_transform(rotation: Any, translation: Any) -> np.ndarray:
    """4×4 matrix of ``x' = R·x + t``."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def apply_transform_to_points(transform: Any, points: Any) -> np.ndarray:
    """Apply ``x' = T·x`` to an (N, 3) array; returns a new array."""
    T = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def apply_transform_to_pose(
    transform: Any, R_w2c: Any, t_w2c: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Re-express a world_to_camera pose in the transformed world ``x' = A·x + a``.

    From ``x = Aᵀ(x' - a)``: ``R' = R·Aᵀ`` and ``t' = t - R'·a``. The camera itself (its image
    and its OpenCV axes) is unchanged.
    """
    T = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    A, a = T[:3, :3], T[:3, 3]
    R = np.asarray(R_w2c, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_w2c, dtype=np.float64).reshape(3)
    R_new = R @ A.T
    return R_new, t - R_new @ a


def invert_transform(transform: Any) -> np.ndarray:
    """Inverse of a rigid 4×4 transform: recovers the original frame (FR-046)."""
    T = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    A, a = T[:3, :3], T[:3, 3]
    return rigid_transform(A.T, -A.T @ a)


def _object_centre(
    centres: np.ndarray, forwards: np.ndarray, points: np.ndarray | None
) -> tuple[np.ndarray, str]:
    """Object centre: nearest point to the optical axes if it lies in front of the cameras."""
    candidate = nearest_point_to_rays(centres, forwards)
    if candidate is not None and np.all(np.isfinite(candidate)):
        in_front = np.einsum("ij,ij->i", candidate - centres, forwards) > 0
        if in_front.mean() >= 0.5:
            return candidate, CENTRE_OPTICAL_AXES
    if points is not None and len(points):
        valid = points[is_valid_points(points)]
        if len(valid):
            return np.median(valid, axis=0), CENTRE_POINTS_MEDIAN
    return centres.mean(axis=0), CENTRE_CAMERA_CENTROID


def estimate_alignment(
    centres: Any,
    rotations_w2c: Any,
    *,
    points: Any | None = None,
    max_planarity: float,
    min_arc_deg: float,
    min_sign_agreement: float,
    min_cameras: int,
) -> dict:
    """Estimate the world alignment: re-centre on the object and, if reliable, set +Y up.

    Returns the ``world_alignment`` record of data-model §3.4a with the 4×4 ``transform``
    (``x_aligned = T·x_original``). When the quality gate fails, only re-centring is applied and
    ``status = not_estimated`` with the reason. Gate thresholds are passed in by the caller.
    """
    C = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(rotations_w2c, dtype=np.float64).reshape(-1, 3, 3)
    pts = None if points is None else np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n_cam = len(C)
    forwards = camera_forward_vectors(R)
    centre, centre_method = _object_centre(C, forwards, pts)

    quality: dict[str, Any] = {
        "num_cameras": n_cam,
        "planarity": None,
        "arc_coverage_deg": None,
        "sign_agreement": None,
    }
    normal = None
    if n_cam >= 3:
        centroid, normal, s = fit_ring_plane(C)
        mean_up = camera_up_vectors(R).mean(axis=0)
        if normal @ mean_up < 0:
            normal = -normal
        quality["planarity"] = float(s[2] / s[1]) if s[1] > 1e-12 else None
        quality["arc_coverage_deg"] = arc_coverage_deg(C, centre, normal)
        quality["sign_agreement"] = float(normal @ mean_up)

    checks = {
        "min_cameras": {"threshold": min_cameras, "passed": n_cam >= min_cameras},
        "max_planarity": {
            "threshold": max_planarity,
            "passed": quality["planarity"] is not None and quality["planarity"] <= max_planarity,
        },
        "min_arc_deg": {
            "threshold": min_arc_deg,
            "passed": quality["arc_coverage_deg"] is not None
            and quality["arc_coverage_deg"] >= min_arc_deg,
        },
        "min_sign_agreement": {
            "threshold": min_sign_agreement,
            "passed": quality["sign_agreement"] is not None
            and quality["sign_agreement"] >= min_sign_agreement,
        },
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    gate = {"checks": checks, "passed": not failed, "thresholds_preliminary": True}

    if failed or normal is None:
        rotation = np.eye(3)
        status = "not_estimated"
        reason = "quality gate not passed: " + ", ".join(failed)
    else:
        tilt = rotation_between(normal, _WORLD_UP)
        # Yaw: put the first camera in front (+Z) so that a render from its pose matches the
        # first video frame. Its horizontal direction from the centre is rotated onto +Z.
        first = tilt @ (C[0] - centre)
        horizontal = np.array([first[0], 0.0, first[2]])
        yaw = (
            rotation_between(horizontal, _WORLD_FRONT)
            if np.linalg.norm(horizontal) > 1e-9
            else np.eye(3)
        )
        rotation = yaw @ tilt
        status = "estimated"
        reason = None

    transform = rigid_transform(rotation, -rotation @ centre)
    return {
        "status": status,
        "method": ALIGNMENT_METHOD,
        "centre_method": centre_method,
        "transform": transform.tolist(),
        "quality": quality,
        "gate": gate,
        "yaw_rule": YAW_RULE if status == "estimated" else None,
        "note": ALIGNMENT_NOTE
        if status == "estimated"
        else "Re-centred on the object only; the vertical was not estimated.",
        "reason": reason,
    }
