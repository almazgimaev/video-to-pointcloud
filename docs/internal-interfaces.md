# Internal interfaces of stages 1–2 (US1)

This document fixes module boundaries so that parts written in parallel fit together.
It describes the **code**, not a user promise; user contracts are
in `specs/001-video-3d-mvp/contracts/`.

## Data flow

```
video ──probe──> VideoInfo
      ──extract──> work/frames/*.jpg + [CandidateFrame]   (candidates, not all frames of the video)
      ──metrics──> sharpness, diff
      ──select──> [FrameRecord]  (selected=True/False, reject_reason)
      ──package──> package/images/<frame_id>.jpg + SHA256SUMS + RUN_ON_GPU.txt
```

Intermediate frames live in `runs/<run_id>/work/frames/` and are not part of the transfer package.

## `v3d.probe`

```python
@dataclass(frozen=True)
class VideoInfo:
    path: str; filename: str; sha256: str; size_bytes: int
    container: str; video_codec: str
    duration_s: float; width: int; height: int; avg_fps: float
    nb_frames: int | None            # None if the container does not report it
    rotation_deg: int                # 0/90/180/270, from side data
    supported: bool
    def to_dict(self) -> dict: ...

SUPPORTED_CONTAINERS: frozenset[str]  # mp4, mov
SUPPORTED_CODECS: frozenset[str]      # h264, hevc

def probe_video(path: Path) -> VideoInfo: ...   # InputUnusableError on unusable input
```

## `v3d.extract`

```python
@dataclass(frozen=True)
class CandidateFrame:
    src_index: int          # frame index in the source video
    timestamp_us: int       # PTS from the start of the video, microseconds
    path: Path              # frame file in work/frames/

def extract_candidates(
    video: Path, out_dir: Path, *, info: VideoInfo,
    max_candidates: int = 600, long_side: int | None = 1024, jpeg_quality: int = 95,
) -> tuple[list[CandidateFrame], dict]: ...
```

The second element is the `image_transform` for the manifest: `rotation_applied_deg`, `source_size`,
`resize_mode`, `resized_size`, `scale_factor`, `crop` (always `None` in v1), `jpeg_quality`,
as well as `candidate_stride` and `candidates_total`.

## `v3d.metrics`

```python
def load_gray(path: Path, *, max_side: int = 256) -> np.ndarray   # float32 [0,1]
def sharpness(gray: np.ndarray) -> float                          # variance of the Laplacian
def frame_difference(a: np.ndarray, b: np.ndarray) -> float       # MAD on 64x64
```

`frame_difference` is a **proxy** for camera motion, not a measurement of viewpoint shift.

## `v3d.select.uniform`

```python
def select_uniform(
    candidates: list[CandidateFrame], *, budget: int, seed: int,
    sharpness_by_index: dict[int, float] | None = None,
    diff_by_index: dict[int, float] | None = None,
    diff_threshold: float | None = None,
) -> list[FrameRecord]        # FrameRecord from v3d.artifacts, one record per candidate
```

`diff_by_index` was added after implementation: without it there is nothing to fill the
`diff_to_prev_selected` field with, and it must not be invented. `seed` does not affect uniform
sampling (the grid is deterministic) and is accepted for uniformity — this is stated in the
docstring, not imitated. `diff_threshold` is used only by the `quality_nonredundant` variant.

**A missing metric value is `None`, not `NaN` and not `0.0`.** Zero would mean a
measured zero sharpness, and `NaN` cannot be expressed in strict JSON: `write_json` writes with
`allow_nan=False` and will fail. The pipeline MUST supply metrics for all candidates.

`frame_id` = `f"{seq:04d}_{src_index:06d}"` for selected frames, `f"----_{src_index:06d}"` for the rest.

## `v3d.select.checks`

```python
def check_enough_frames(records: list[FrameRecord], *, min_frames: int) -> None
```

Raises `NotEnoughFramesError` (code 3) with the source/selected/threshold numbers.

## `v3d.warnings_`

Warning builders, they return `v3d.errors.Warning_`. The phrasings are observations,
without naming a cause.

## `v3d.package`

```python
def build_package(
    images_dir: Path, records: list[FrameRecord], candidates: list[CandidateFrame],
    *, package_dir: Path, run_id: str, command_hint: str,
) -> dict                      # summary: number of files, total size

def write_sha256sums(package_dir: Path) -> Path
def verify_sha256sums(package_dir: Path) -> None     # PackageIntegrityError (code 4)
```

---

# Stage 4: result ingest (T027–T032)

## Data flow

```
result/sparse/*.bin ──colmap_io──> SparseModel
frames.json + result/ ──ingest_checks──> refusal (codes 5/6/7) or continue
SparseModel ──ingest──> normalized/{cameras.json, points.ply}
           ──diagnostics──> normalized/diagnostics.json
           ──status──> normalized/status.json
```

## `v3d.colmap_io` (T027)

```python
@dataclass(frozen=True)
class SparseModel:
    cameras: list[CameraRecord]        # from v3d.artifacts, world_to_camera poses
    points_xyz: np.ndarray             # (N, 3) float64
    points_rgb: np.ndarray             # (N, 3) uint8
    points_error: np.ndarray | None    # (N,) or None if the model does not provide it
    image_names: list[str]
    mean_reprojection_error: float | None   # None if unavailable (not zero!)

def read_sparse(sparse_dir: Path) -> SparseModel
```

Reads the COLMAP binary model via `pycolmap`. Converts poses to `world_to_camera`
(in the COLMAP format they already are), fills `camera_center_world` via `v3d.geometry`.
`frame_id` is taken from the image name without the extension. An unavailable reprojection error is
`None`, never zero.

## `v3d.ingest_checks` (T028)

```python
REQUIRED_RESULT_FILES: tuple[str, ...]   # sparse/cameras.bin, sparse/images.bin, sparse/points3D.bin

def check_result_files(result_dir: Path) -> None
def check_images_subset(model_names: list[str], selected_frame_ids: list[str]) -> None
def check_model_not_empty(model: SparseModel) -> None
def check_run_id_matches(manifest_payload: dict, *payloads: tuple[str, dict]) -> None
def check_package_intact(package_dir: Path) -> None     # non-strict checksum check
```

Codes: required files missing → 7; files exist but the result is truncated (zero size) → 5;
the model's image names are not a subset of the selected frames → 7; no cameras or no points → 6.

## `v3d.diagnostics` (T030)

```python
def build_diagnostics(
    run_id: str, *, frames: list[dict], model: SparseModel,
    points_valid: int | None = None,
    valid_points_xyz: np.ndarray | None = None,
    filters_applied: list[dict],
    durations_s: dict[str, float], artifact_sizes: dict[str, int],
) -> dict
```

Two modes. With `valid_points_xyz`, the `bbox` dimensions describe exactly the displayed point cloud, and
`num_points_valid` is taken as the array length; a mismatch with the passed `points_valid` is a
`ValueError`. With only `points_valid`, the box is computed over the raw point cloud before the outlier filter
and is marked with a corresponding note. With neither — `ValueError`: there is nowhere to get the number of
valid points from, and it must not be invented. `ingest` calls the first mode.

Fills the template from `artifacts.new_diagnostics`: frames, cameras and `registration_ratio`,
points (`num_points_raw`, `num_points_valid`, `valid_point_definition`, `has_colors`,
`background_present = True`), timings, sizes, the reprojection error or an unavailability mark
with a reason, `visual_check` (6 items `not_checked`), `interpretation_limits`.

## `v3d.status` (T031)

```python
def determine_status(
    *, cameras_registered: int, points_valid: int,
    selected_frames: int, result_complete: bool,
) -> RunStatus

def build_status(run_id: str, status: RunStatus, *, warnings: list[dict],
                 precomputed_example: bool = False, note: str | None = None,
                 cameras_registered: int | None = None,
                 selected_frames: int | None = None) -> dict

def status_banner(status_payload: dict) -> str
```

`cameras_registered` and `selected_frames` are needed to assemble the text of the mandatory
`few_registered_cameras` warning with the ready-made builder from `v3d.warnings_`, rather than
inventing our own phrasing. With status `partial` and without these numbers and without an already built
warning — `ValueError`: the observation about incomplete registration must not be lost silently.

Rules: result incomplete → `interrupted`; no cameras or no valid points →
`invalid_result`; all selected frames registered → `success`; some of them → `partial`.
No numeric threshold for "how much partial is acceptable" is assigned.

---

# Stage 5: viewing, export, report (T033–T038)

All three read only `normalized/` and `manifest.json`. None recomputes the result
or requires the GPU machine (FR-027).

## `v3d.viewer` (T033)

```python
def build_recording(run_dir: Path, *, save_to: Path | None = None, spawn: bool = True) -> dict
```

Logs to Rerun: the point cloud with colors, camera poses as a separate toggleable layer,
a run information panel, a warnings panel, a persistent status indicator.
`spawn=False` + `save_to` — record an `.rrd` without opening a window (a mode for checks).
Returns a summary: what was logged and how much.

## `v3d.export` (T034)

```python
def export_point_cloud(run_dir: Path, *, out: Path | None = None, force: bool = False) -> Path
```

Writes a PLY matching the displayed point cloud: the same point count, coordinates, colors.
Statuses `invalid_result` and `interrupted` — refusal with code 6; with `force=True` the file is written
with the `.UNUSABLE.ply` suffix and a mark in the header.

## `v3d.report` (T035)

```python
def build_report(run_dir: Path, *, out: Path | None = None) -> Path
```

Writes `report.md`: input, frames, parameters, environment, timings, metrics with explanations,
unavailable metrics with reasons, warnings, scale status, the background flag,
interpretation limits. Two separate sections: verified properties of the file and unverifiable
assumptions about the scene (FR-006).
