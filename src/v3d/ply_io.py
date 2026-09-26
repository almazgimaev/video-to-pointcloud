"""Writing and reading a coloured point cloud in PLY.

Format: binary little endian, properties ``x y z`` (float32) and ``red green blue`` (uchar).
Chosen because it can be read by third-party viewers (MeshLab, CloudCompare) without our
tools — this is a requirement for open formats.

The header carries comments with ``run_id`` and a summary, so that an exported file cannot be
confused with the result of another run (FR-032). The artifact type is declared explicitly:
this is a point cloud, not a mesh or a splat.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from v3d.errors import ResultMissingOrIncompatibleError

HEADER_ARTIFACT_TYPE = "artifact_type: point_cloud"


def write_ply(
    path: Path,
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    *,
    run_id: str,
    comments: list[str] | None = None,
) -> int:
    """Write a coloured point cloud. Returns the number of points written.

    Points and colours must match in length: a colour without a point and a point without a
    colour would mean the export diverges from the displayed cloud (FR-031).
    """
    xyz = np.asarray(points_xyz, dtype=np.float32)
    rgb = np.asarray(points_rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected a points array (N, 3), got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"colours {rgb.shape} do not match points {xyz.shape} in count")

    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment run_id: {run_id}",
        f"comment {HEADER_ARTIFACT_TYPE}",
        "comment scale_status: not_determined",
        "comment axes: opencv right-handed; units unknown",
    ]
    for text in comments or []:
        lines.append(f"comment {text}")
    lines += [
        f"element vertex {len(xyz)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    header = ("\n".join(lines) + "\n").encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(xyz, rgb, strict=True):
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))
    return len(xyz)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a cloud written by :func:`write_ply`. Returns points, colours and comments.

    Needed to check that the export matches the displayed cloud, and to reopen a result
    without recomputation.
    """
    if not path.exists():
        raise ResultMissingOrIncompatibleError(f"point cloud file is missing: {path}")

    comments: dict[str, str] = {}
    count = 0
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ResultMissingOrIncompatibleError(f"file is not in PLY format: {path}")
        while True:
            raw = f.readline()
            if not raw:
                raise ResultMissingOrIncompatibleError(f"PLY has no end of header: {path}")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            if line.startswith("comment "):
                body = line[len("comment ") :]
                key, _, value = body.partition(": ")
                comments[key] = value if value else body
            elif line.startswith("element vertex "):
                count = int(line.split()[-1])
        payload = f.read()

    stride = struct.calcsize("<fffBBB")
    if len(payload) != count * stride:
        raise ResultMissingOrIncompatibleError(
            f"point cloud is truncated: expected {count} points, data for {len(payload) // stride}"
        )

    xyz = np.empty((count, 3), dtype=np.float32)
    rgb = np.empty((count, 3), dtype=np.uint8)
    for i in range(count):
        x, y, z, r, g, b = struct.unpack_from("<fffBBB", payload, i * stride)
        xyz[i] = (x, y, z)
        rgb[i] = (r, g, b)
    return xyz, rgb, comments
