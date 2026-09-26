"""Deterministic decimation of a normalized point cloud for the web page (FR-051, FR-052).

The web viewer must not ship every reconstructed point: the cloud is decimated to a target
count before publishing. To keep the shown cloud honest (FR-031: exported data must match what
is displayed), decimation never averages positions or colours — it keeps a subset of the real
points, one per occupied voxel of a grid sized so the number of occupied voxels is as close to
the target as possible without exceeding it. The voxel representative is the point nearest to
the voxel centre, ties broken by the smallest original index, so the whole process is
deterministic and reproducible byte-for-byte across runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from v3d.geometry import is_valid_points
from v3d.ply_io import read_ply, write_ply

#: Name of the decimation method, recorded in the returned info dict.
METHOD = "voxel_grid_nearest_to_centre"

#: Maximum number of bisection steps used to search the voxel edge length.
_MAX_BISECTION_STEPS = 40

#: Maximum number of doublings/halvings used to bracket the bisection interval.
_MAX_BRACKET_STEPS = 30


def _label(points_out: int, points_in: int) -> str:
    """Format the FR-052 label, e.g. ``"40,000 of 100,000 points"``."""
    return f"{points_out:,} of {points_in:,} points"


def _occupied_representatives(
    points: np.ndarray, original_index: np.ndarray, bbox_min: np.ndarray, voxel_size: float
) -> np.ndarray:
    """Positions (into ``points``) of one representative per occupied voxel.

    The representative of a voxel is the point nearest to its centre; ties are broken by the
    smallest original index. Both rules make the choice deterministic and independent of input
    order.
    """
    ijk = np.floor((points - bbox_min) / voxel_size).astype(np.int64)
    # Encode each voxel's (i, j, k) as a single int64 key: 1-D `unique` is much cheaper than
    # `unique(..., axis=0)`, which sorts row-wise.
    shifted = ijk - ijk.min(axis=0)
    ranges = shifted.max(axis=0) + 1
    key = (shifted[:, 0] * ranges[1] + shifted[:, 1]) * ranges[2] + shifted[:, 2]
    _, inverse = np.unique(key, return_inverse=True)
    inverse = inverse.reshape(-1)
    centres = bbox_min + (ijk.astype(np.float64) + 0.5) * voxel_size
    dist2 = np.sum((points - centres) ** 2, axis=1)

    # Sort by (voxel, distance to its centre, original index): the first row of each voxel
    # group in this order is the representative we want.
    order = np.lexsort((original_index, dist2, inverse))
    group_sorted = inverse[order]
    _, first_pos = np.unique(group_sorted, return_index=True)
    return order[first_pos]


def voxel_decimate(
    points_xyz: np.ndarray, points_rgb: np.ndarray, *, target: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decimate a point cloud to at most ``target`` points, keeping real points unchanged.

    Non-finite points are dropped first. If what remains already fits within ``target`` it is
    returned unchanged, in its original order. Otherwise the cloud is downsampled on a voxel
    grid: the voxel edge length is found by bisection so the number of occupied voxels is as
    close to ``target`` as possible without exceeding it, one point per voxel (nearest to its
    centre), and the output is sorted by original index so two runs are byte-identical.

    Returns ``(points_out, colors_out, info)``; ``info`` matches FR-052's labelling requirement.
    """
    xyz = np.asarray(points_xyz)
    rgb = np.asarray(points_rgb)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected a points array (N, 3), got {xyz.shape}")
    if len(rgb) != len(xyz):
        raise ValueError(f"colours ({len(rgb)}) do not match points ({len(xyz)}) in count")
    if target < 0:
        raise ValueError("target must not be negative")

    valid_mask = is_valid_points(xyz)
    valid_index = np.flatnonzero(valid_mask)
    valid_xyz = xyz[valid_index]
    valid_rgb = rgb[valid_index]
    points_in = int(len(valid_index))

    if points_in <= target:
        info = {
            "method": METHOD,
            "points_in": points_in,
            "points_out": points_in,
            "voxel_size": None,
            "target": int(target),
            "label": _label(points_in, points_in),
        }
        return valid_xyz.copy(), valid_rgb.copy(), info

    pts64 = valid_xyz.astype(np.float64)
    bbox_min = pts64.min(axis=0)
    bbox_max = pts64.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    if diag <= 0.0:
        # All valid points sit at the same location: they all fall in one voxel regardless of
        # size. The representative is the smallest original index (target >= 1 here, since the
        # points_in <= target branch above already handled target >= points_in).
        winners = np.array([0], dtype=np.int64)
        voxel_size_used = 0.0
    else:
        # Volume-based heuristic: with `target` voxels evenly filling the bounding box, the
        # voxel edge is about diag / cbrt(target). Bracket generously around it so the safety
        # loops below rarely need to run.
        guess = diag / (max(target, 1) ** (1.0 / 3.0))
        lo = max(guess / 1000.0, diag * 1e-12)
        hi = min(guess * 1000.0, diag * 10.0)

        def occupied_count(s: float) -> int:
            return len(_occupied_representatives(pts64, valid_index, bbox_min, s))

        tries = 0
        while occupied_count(lo) <= target and tries < _MAX_BRACKET_STEPS:
            lo /= 2.0
            tries += 1
        tries = 0
        while occupied_count(hi) > target and tries < _MAX_BRACKET_STEPS:
            hi *= 2.0
            tries += 1

        best_s = hi
        best_winners = _occupied_representatives(pts64, valid_index, bbox_min, hi)
        for _ in range(_MAX_BISECTION_STEPS):
            mid = (lo + hi) / 2.0
            if mid <= 0.0 or mid in (lo, hi):
                break
            winners = _occupied_representatives(pts64, valid_index, bbox_min, mid)
            if len(winners) <= target:
                hi = mid
                best_s = mid
                best_winners = winners
            else:
                lo = mid
        winners = best_winners
        voxel_size_used = float(best_s)

    out_original_index = valid_index[winners]
    order = np.argsort(out_original_index, kind="stable")
    winners_sorted = winners[order]

    points_out = valid_xyz[winners_sorted]
    colors_out = valid_rgb[winners_sorted]
    n_out = int(len(winners_sorted))

    info = {
        "method": METHOD,
        "points_in": points_in,
        "points_out": n_out,
        "voxel_size": voxel_size_used,
        "target": int(target),
        "label": _label(n_out, points_in),
    }
    return points_out, colors_out, info


def write_web_cloud(src_ply: Path, out_ply: Path, *, target: int = 40_000) -> dict:
    """Decimate ``src_ply`` (the normalized cloud) and write the web cloud to ``out_ply``.

    Preserves ``run_id`` and the ``axes`` comment from the source, and adds ``web_decimation``
    (the FR-052 label) and ``derived_from`` comments so the file documents where it came from.
    Returns the decimation info plus ``path`` and ``bytes``.
    """
    xyz, rgb, comments = read_ply(src_ply)
    points_out, colors_out, info = voxel_decimate(xyz, rgb, target=target)

    run_id = comments.get("run_id", "")
    axes = comments.get("axes", "opencv right-handed; units unknown")
    write_ply(
        out_ply,
        points_out,
        colors_out,
        run_id=run_id,
        comments=[
            f"web_decimation: {info['label']}",
            "derived_from: normalized/points.ply",
        ],
        axes=axes,
    )

    return {**info, "path": str(out_ply), "bytes": out_ply.stat().st_size}
