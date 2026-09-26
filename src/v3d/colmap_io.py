"""Reading a binary COLMAP sparse model and converting it to project structures.

Intake contract: `specs/001-video-3d-mvp/contracts/reconstructor-io.md`;
data model — `data-model.md` §3.4 (Camera) and §3.5 (PointCloud).

The module does exactly two things: reads `sparse/{cameras,images,points3D}.bin` via
`pycolmap` and converts what was read into :class:`SparseModel`. There are no business-rule
checks here (empty model, frame names matching the selected ones) — that is `v3d.ingest_checks`.

Conventions are not reinvented: in the COLMAP format the pose is already `world_to_camera`
(``x_cam = R @ x_world + t``), so the matrix and vector go into `CameraRecord`
as is, and `camera_center_world` is computed via :mod:`v3d.geometry`
(``C = -Rᵀ t``).

What is established about the actual pycolmap 4.2.0 API (verified by calls, not from memory):

* `pycolmap.Reconstruction(path)` reads the whole directory; a missing directory or any
  of the required files, as well as a truncated file, gives a `ValueError` with a
  C++-level text — the raw exception is not let out, its text goes into `details`;
* `Image.cam_from_world()` is a **method** (in 3.x it was the `cam_from_world` property),
  returns a `Rigid3d` with `.rotation.matrix()` and `.translation`;
* `Point3D.error` equals ``-1`` if the error is not set; the indicator is `Point3D.has_error()`;
* `Reconstruction.compute_mean_reprojection_error()` returns **0.0** also when
  no point has an error. So availability is determined not by this number but by the
  number of points with `has_error()`; see :attr:`SparseModel.mean_reprojection_error`;
* `Camera.focal_length_x` / `principal_point_x` for a model without a focal length
  (`EQUIRECTANGULAR`) crash the process with a signal rather than an exception, so the
  parameters are parsed by the indices `focal_length_idxs()` / `principal_point_idxs()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from v3d.artifacts import CameraRecord
from v3d.errors import ResultMissingOrIncompatibleError
from v3d.geometry import camera_center_from_world_to_camera

#: Files without which a directory is not a COLMAP sparse model.
REQUIRED_MODEL_FILES: tuple[str, ...] = ("cameras.bin", "images.bin", "points3D.bin")

#: Value of the `source` field in camera records (data-model.md §3.4).
CAMERA_SOURCE = "colmap_sparse"


@dataclass(frozen=True)
class SparseModel:
    """A sparse model read into project structures.

    Fields:

    * `cameras` — one record per **registered** image (one that has a pose);
      the order is deterministic (by image name, then by `image_id`);
    * `points_xyz` — (N, 3) float64, `points_rgb` — (N, 3) uint8; points are ordered
      by ascending `point3D_id`, so two reads give identical arrays;
    * `points_error` — (N,) float64 or ``None`` if no point has an error set;
      for individual points without an error the array holds ``NaN`` (an absent value,
      not a measured zero);
    * `image_names` — names of **all** model images, including unregistered ones,
      in the same sort order as `cameras`;
    * `mean_reprojection_error` — mean reprojection error over the points that have one,
      or ``None`` if no point has it (FR-036: an unavailable metric is not
      replaced by zero). It is unavailable, for example, when upstream ran without bundle
      adjustment and did not write per-point errors: the file then holds ``-1``. A value
      of ``0.0`` here means a measured zero, not missing data.
    """

    cameras: list[CameraRecord]
    points_xyz: np.ndarray
    points_rgb: np.ndarray
    points_error: np.ndarray | None
    image_names: list[str]
    mean_reprojection_error: float | None


def frame_id_from_image_name(name: str) -> str:
    """Image name → `frame_id`: file name without extension (`0000_000012.jpg` → `0000_000012`).

    Parsed as a POSIX path: a name in a COLMAP model may contain a subdirectory
    (`images/0000_000012.jpg`), and only the file itself is needed.
    """
    return PurePosixPath(name).stem


def intrinsics_from_camera(camera: Any) -> dict | None:
    """COLMAP camera parameters → ``{model, fx, fy, cx, cy}`` or ``None``.

    Parsing goes by the indices `focal_length_idxs()` and `principal_point_idxs()`, not by
    the properties `focal_length_x` / `principal_point_x`: for a model without a focal length
    the latter in pycolmap 4.2.0 crash the process with a signal instead of raising.

    A model with a single focal parameter (`SIMPLE_PINHOLE`, `SIMPLE_RADIAL`,
    `SIMPLE_FISHEYE`, …) gives ``fx = fy = f``: this spells out its own convention,
    not a guess. If the focal length or principal point cannot be decomposed, ``None`` is
    returned; made-up numbers are not substituted.
    """
    try:
        params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
        focal_idxs = tuple(int(i) for i in camera.focal_length_idxs())
        pp_idxs = tuple(int(i) for i in camera.principal_point_idxs())
        model_name = str(camera.model_name)
    except Exception:
        return None

    if len(focal_idxs) not in (1, 2) or len(pp_idxs) != 2:
        return None
    if any(i < 0 or i >= params.size for i in focal_idxs + pp_idxs):
        return None

    return {
        "model": model_name,
        "fx": float(params[focal_idxs[0]]),
        "fy": float(params[focal_idxs[-1]]),
        "cx": float(params[pp_idxs[0]]),
        "cy": float(params[pp_idxs[1]]),
    }


def _camera_record(image: Any, camera: Any) -> CameraRecord:
    """One registered image → :class:`CameraRecord` (world_to_camera pose)."""
    cam_from_world = image.cam_from_world()
    R = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64).reshape(3, 3)
    t = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3)
    center = camera_center_from_world_to_camera(R, t)
    return CameraRecord(
        frame_id=frame_id_from_image_name(str(image.name)),
        image_size=(int(camera.width), int(camera.height)),
        intrinsics=intrinsics_from_camera(camera),
        R_world_to_camera=[[float(v) for v in row] for row in R],
        t_world_to_camera=[float(v) for v in t],
        camera_center_world=[float(v) for v in center],
        source=CAMERA_SOURCE,
    )


def _read_reconstruction(sparse_dir: Path) -> Any:
    """Open the model via pycolmap, turning any failure into code 7."""
    try:
        import pycolmap
    except Exception as exc:  # pragma: no cover — depends on the environment, not on data
        raise ResultMissingOrIncompatibleError(
            "failed to load pycolmap: nothing to read the COLMAP model with",
            details=f"{type(exc).__name__}: {exc}",
        ) from exc

    if not sparse_dir.is_dir():
        raise ResultMissingOrIncompatibleError(
            f"sparse model directory is missing: {sparse_dir}",
            details="the result directory is incomplete or the reconstructor did not run",
        )
    missing = [name for name in REQUIRED_MODEL_FILES if not (sparse_dir / name).is_file()]
    if missing:
        raise ResultMissingOrIncompatibleError(
            f"{sparse_dir} lacks required model files: {', '.join(missing)}",
            details=f"all of these were expected: {', '.join(REQUIRED_MODEL_FILES)}",
        )

    try:
        return pycolmap.Reconstruction(str(sparse_dir))
    except Exception as exc:
        raise ResultMissingOrIncompatibleError(
            f"failed to read the COLMAP sparse model from {sparse_dir}",
            details=f"{type(exc).__name__}: {exc}",
        ) from exc


def _read_points(reconstruction: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Model points in deterministic order (by ascending `point3D_id`)."""
    point_ids = sorted(int(i) for i in reconstruction.points3D.keys())
    xyz = np.zeros((len(point_ids), 3), dtype=np.float64)
    rgb = np.zeros((len(point_ids), 3), dtype=np.uint8)
    error = np.full(len(point_ids), np.nan, dtype=np.float64)
    num_with_error = 0

    for row, point_id in enumerate(point_ids):
        point = reconstruction.points3D[point_id]
        xyz[row] = np.asarray(point.xyz, dtype=np.float64).reshape(3)
        rgb[row] = np.asarray(point.color, dtype=np.uint8).reshape(3)
        if bool(point.has_error()):
            error[row] = float(point.error)
            num_with_error += 1

    return xyz, rgb, (error if num_with_error > 0 else None)


def read_sparse(sparse_dir: Path) -> SparseModel:
    """Read a binary COLMAP sparse model from `sparse_dir` into :class:`SparseModel`.

    Files `cameras.bin`, `images.bin`, `points3D.bin` are expected. Poses are taken as is
    (`world_to_camera`), the camera centre is computed via :mod:`v3d.geometry`.

    A missing directory or files, as well as any pycolmap parse failure (truncated file,
    unknown format version, inconsistent tracks), raises
    :class:`~v3d.errors.ResultMissingOrIncompatibleError` (code 7); the text of the original
    exception is kept in `details`.

    The module performs no content checks: an empty model, a model without points and
    a model with foreign image names are read normally, the decision to refuse is made by
    `v3d.ingest_checks`.
    """
    reconstruction = _read_reconstruction(Path(sparse_dir))

    try:
        images = [reconstruction.images[i] for i in sorted(reconstruction.images.keys())]
        images.sort(key=lambda im: (str(im.name), int(im.image_id)))
        image_names = [str(im.name) for im in images]
        cameras = [
            _camera_record(im, reconstruction.cameras[int(im.camera_id)])
            for im in images
            if bool(im.has_pose)
        ]
        points_xyz, points_rgb, points_error = _read_points(reconstruction)
        num_with_error = 0 if points_error is None else int(np.isfinite(points_error).sum())
        mean_error = (
            float(reconstruction.compute_mean_reprojection_error()) if num_with_error else None
        )
    except ResultMissingOrIncompatibleError:
        raise
    except Exception as exc:
        raise ResultMissingOrIncompatibleError(
            f"sparse model in {sparse_dir} was read, but its content cannot be parsed",
            details=f"{type(exc).__name__}: {exc}",
        ) from exc

    return SparseModel(
        cameras=cameras,
        points_xyz=points_xyz,
        points_rgb=points_rgb,
        points_error=points_error,
        image_names=image_names,
        mean_reprojection_error=mean_error,
    )
