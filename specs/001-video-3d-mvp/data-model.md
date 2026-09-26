# Data Model: Video-to-3D MVP (Phase 1)

**Date**: 2026-09-18 · **Feature**: [spec.md](./spec.md) · **Plan**: [plan.md](./plan.md)

This document defines the entities, required and unavailable fields, geometric conventions,
and the rules for run identity. Schemas of specific files are in [contracts/](./contracts/).

---

## 1. Run directory

```text
runs/<run_id>/
├── manifest.json          # run information (required)
├── frames.json            # all frames considered and the decision for each
├── package/               # TRANSFERABLE to the GPU machine
│   ├── images/            # selected frames only, .jpg
│   ├── SHA256SUMS         # checksums of all package files
│   └── RUN_ON_GPU.txt     # instructions for a human: which upstream command to run
├── result/                # RETURNED from the GPU machine, upstream content as is
│   ├── sparse/{cameras,images,points3D}.bin
│   ├── points.ply
│   └── stdout.log
├── normalized/            # normalized representation (computed on the Mac)
│   ├── cameras.json
│   ├── points.ply         # normalized colored point cloud
│   ├── diagnostics.json
│   └── status.json
├── export/                # output of the export command
│   └── object.ply
├── view/
│   └── session.rrd        # recording for the viewer (an artifact, not the source of truth)
└── report.md
```

**No-overwrite rule**: the directory `runs/<run_id>/` is created only if it does not exist. Repeated
preparation with the same parameters does not overwrite an existing run; overwriting is possible
only with the explicit `--force` flag, which is written to the manifest as `forced_overwrite: true`.

---

## 2. Run identity

```text
run_id = <UTC YYYYMMDD-HHMMSS>-<inputs_digest[:8]>

inputs_digest = sha256(
    video_sha256 ||
    canonical_json(selection_params) ||
    code_id ||
    seed
)
```

- `code_id` — the code version identifier: a git commit, or (outside git) the sha256 of the contents
  of the `v3d` package. Additionally `working_tree_dirty: true|false` is recorded and, if dirty,
  the list of changed files.
- Two runs with the same `inputs_digest` are a reproduction of the same run; two with different
  ones cannot land in the same directory.
- **Every artifact inside a run carries `run_id`**: `manifest.json`, `cameras.json`,
  `diagnostics.json`, `status.json`, the header of `report.md`, a comment in the exported PLY.
  The viewer and export refuse to work on a `run_id` mismatch between files (FR-029).

---

## 3. Entities

### 3.1 InputVideo (in `manifest.json`)

| Field | Type | Req. | Description |
|---|---|---|---|
| `path` | str | yes | path at the time of preparation |
| `filename` | str | yes | file name |
| `sha256` | str | yes | content hash — the only reliable identifier of the input |
| `size_bytes` | int | yes | |
| `container` | str | yes | from `ffprobe` (`mov,mp4,m4a,...`) |
| `video_codec` | str | yes | `h264`, `hevc`, … |
| `duration_s` | float | yes | |
| `width`, `height` | int | yes | dimensions as in the stream, before rotation is applied |
| `avg_fps` | float | yes | |
| `nb_frames` | int \| null | no | may be absent in the container → `null` + `unavailable_reason` |
| `rotation_deg` | int | yes | 0/90/180/270 from side data; 0 if there is no metadata |
| `supported` | bool | yes | whether it is in the supported list (see contracts/cli.md) |

### 3.2 Frame (in `frames.json`) — a record for **every frame considered**

| Field | Type | Req. | Description |
|---|---|---|---|
| `frame_id` | str | yes | `NNNN_SSSSSS`: NNNN — sequence number among the selected (or `----` for an unselected one), SSSSSS — frame index in the source video |
| `src_index` | int | yes | frame index in the source video, zero-based |
| `timestamp_us` | int | yes | PTS in microseconds from the start of the video |
| `sharpness` | float \| null | yes | variance of the Laplacian on a downscaled copy; `null` — the metric was not computed (zero is prohibited: it would mean a measured zero sharpness) |
| `diff_to_prev_selected` | float \| null | yes/null | mean absolute 64×64 grayscale difference to the last selected one; `null` for the first |
| `selected` | bool | yes | |
| `reject_reason` | str \| null | yes/null | `blurry` \| `redundant` \| `budget_exhausted` \| `outside_window` \| null |
| `image_file` | str \| null | yes/null | file name in `package/images/`, if selected |

Frame file name: `package/images/<frame_id>.jpg`. **The frame file name is the key linking to the camera**
(COLMAP stores the image `name`) — FR-019.

### 3.3 ImageTransform (in `manifest.json`, common to all v1 frames)

| Field | Type | Req. | Description |
|---|---|---|---|
| `rotation_applied_deg` | int | yes | rotation actually applied during extraction |
| `source_size` | [int,int] | yes | frame size after rotation, before resize |
| `resize_mode` | str | yes | `none` \| `long_side` |
| `resized_size` | [int,int] \| null | yes/null | size of the saved frame |
| `scale_factor` | float | yes | `resized_long_side / source_long_side`; 1.0 with `none` |
| `crop` | null | yes | **crop is not applied in v1**; the field is present and equals `null` |
| `jpeg_quality` | int | yes | frame saving quality |
| `candidate_stride` | int | yes | stride over source video frames when choosing candidates |
| `candidates_total` | int | yes | how many frames were considered (candidates), see `frames.json` |
| `timestamp_source` | str | yes | where the timestamps come from; `container_pts` — the actual container PTS, not a recomputation from fps |

All internal sizes (intrinsics and points) refer to the **saved** frame, i.e. after
rotation and resize. This is declared in `cameras.json` by the `image_size` field.

### 3.4 Camera (in `normalized/cameras.json`)

One record per **registered** frame.

| Field | Type | Req. | Description |
|---|---|---|---|
| `frame_id` | str | yes | link to Frame and the image file |
| `image_size` | [int,int] | yes | size of the image the intrinsics refer to (pixels) |
| `intrinsics` | object \| null | yes/null | `{model, fx, fy, cx, cy}`; `null` + `unavailable_reason` if the reconstructor does not provide them |
| `R_world_to_camera` | 3×3 float | yes | rotation matrix |
| `t_world_to_camera` | 3 float | yes | translation vector |
| `camera_center_world` | 3 float | yes | derived `C = -Rᵀ t`, for the viewer |
| `source` | str | yes | `colmap_sparse` — where the values come from |

**Conventions (fixed in the header of `cameras.json` and not implied by default)**:

```json
"conventions": {
  "artifact_type": "point_cloud",
  "transform_direction": "world_to_camera",
  "formula": "x_cam = R * x_world + t",
  "camera_axes": "opencv: +X right, +Y down, +Z forward (looking direction)",
  "world_axes": "right-handed; world frame defined by the reconstructor",
  "units": "unknown",
  "scale_status": "not_determined",
  "scale_reference": null
}
```

- `artifact_type` — the actual artifact type (FR-021). In v1 always `point_cloud`;
  the values `mesh` and `splat` are not allowed in this feature.
- `scale_status` ∈ {`measured`, `model_estimated`, `not_determined`}. **In v1 always
  `not_determined`** — there is no independent reference (Principle I, FR-021, FR-039).
- Switching to a reconstructor that outputs camera-to-world **does not change the contract**: the adapter must
  convert to `world_to_camera` and leave the `transform_direction` field unchanged.

### 3.4a WorldAlignment (in `normalized/cameras.json`, key `world_alignment`) — FR-045…FR-048

The reconstructor anchors its world to the first camera, so the result is tilted by however the
phone was held. Alignment re-centres the result on the object and estimates "up" from the camera
ring. All poses and points in `normalized/` are expressed in the **aligned** frame; the raw
reconstructor output in `result/sparse/` is never modified.

| Field | Type | Req. | Description |
|---|---|---|---|
| `status` | str | yes | `estimated` — vertical estimated and applied; `not_estimated` — quality gate failed, only re-centring applied |
| `method` | str | yes | `camera_ring_plane` for the vertical; centre method recorded separately |
| `centre_method` | str | yes | `optical_axes_nearest_point` or `points_median` (fallback) |
| `transform` | 4×4 float | yes | rigid transform from the reconstructor frame to the aligned frame: `x_aligned = T · x_original` (no scaling) |
| `quality` | object | yes | `planarity` (s3/s2 of camera centres), `arc_coverage_deg`, `sign_agreement`, `num_cameras` |
| `gate` | object | yes | thresholds used and whether each passed; thresholds marked preliminary |
| `yaw_rule` | str | yes | deterministic rule fixing rotation about the vertical, e.g. `first_camera_in_front` |
| `note` | str | yes | human-readable statement: "estimated from the camera trajectory, not measured; rotation about the vertical is arbitrary" |
| `reason` | str \| null | yes/null | why the estimate was rejected, when `status = not_estimated` |

With alignment, the `conventions` block changes as follows: `world_axes` becomes
`"right-handed, +Y up (estimated from camera trajectory), origin at object centre"` when
`status = estimated`, and `"right-handed, origin at object centre; vertical not estimated"`
otherwise. `units` and `scale_status` are unchanged (`unknown`, `not_determined`).
Camera axes stay OpenCV inside each camera frame; `R_world_to_camera` and `t_world_to_camera`
are re-expressed in the aligned world. The original pose is `T⁻¹` applied to the aligned one.

### 3.5 PointCloud (`normalized/points.ply` + metadata in `diagnostics.json`)

| Field | Type | Req. | Description |
|---|---|---|---|
| `num_points_raw` | int | yes | points in the original result |
| `num_points_valid` | int | yes | points after the validity filter |
| `valid_point_definition` | str | yes | text definition, see below |
| `has_colors` | bool | yes | must be `true` in v1 |
| `background_present` | bool | yes | **always `true` in v1** — the background is not removed (FR-020) |
| `bbox` | [[3],[3]] | yes | extent of the cloud in world units |
| `filters_applied` | list | yes | list of applied filters with parameters |

**Definition of a valid point (fixed here, before implementation)**: a point is considered valid
if (a) all three coordinates are finite (not NaN/Inf), (b) the point made it into the reconstructor's result
after its own confidence threshold, (c) the point was not discarded by our outlier filter.
The v1 outlier filter is a standard IQR filter on each axis independently: a point is discarded
if on at least one axis it falls outside `[q1 - k·iqr, q3 + k·iqr]`, where `q1`, `q3` are quartiles,
`iqr = q3 - q1`. The value of `k` is fixed as a parameter (default 3.0) and written to
`filters_applied` together with the number of points before and after and the actual per-axis bounds.
Non-finite points (NaN/Inf) are discarded before the quartiles are computed, and their number is counted separately.
No other filters are applied.

Storage format: PLY (binary little endian), properties `x y z` (float32) and `red green blue`
(uchar). Color is taken from the reconstructor's result; no custom coloring is applied.

### 3.6 Diagnostics (`normalized/diagnostics.json`)

The metrics are exactly those fixed in the spec, section "Observable metrics":

| Group | Fields |
|---|---|
| Frames | `frames_total`, `frames_selected`, `reject_reason_counts{}` |
| Cameras | `cameras_registered`, `registration_ratio` (= registered / selected; recorded as a metric with an availability flag — with zero selected frames the ratio is unavailable, not zero) |
| Points | `num_points_raw`, `num_points_valid`, `valid_point_definition`, `has_colors`, `background_present`, `bbox` (extent of the displayed cloud; for an empty cloud — an unavailability flag, not zeros), `filters_applied` |
| Time | `stage_durations_s{check, prepare, select, reconstruct, ingest, view, export}` |
| Volume | `artifact_sizes_bytes{}` |
| Error | `reprojection_error` → `{value, units, meaning, availability}` |
| Visual check | `visual_check` → a list of the 6 spec items with the value `ok` \| `defect` \| `not_checked`, a `note` field, and `checked_at` (UTC ISO time of the human check, `null` while `not_checked`) — FR-049 |

**Unavailability rule (FR-036)**: any unavailable field is recorded as
`{"value": null, "availability": "unavailable", "reason": "<text>"}`. Zero in place of
unavailability is prohibited. For example, without `--use_ba` the reprojection error is unavailable, and the reason
is recorded explicitly.

**Interpretation limits rule (FR-038)**: `diagnostics.json` contains the field
`"interpretation_limits"` with text stating that the number of points, confidence, and reprojection error
are not proof of geometric accuracy or completeness.

### 3.7 RunStatus (`normalized/status.json`)

```text
status ∈ { prepared, running, success, partial, failed, interrupted, invalid_result }
```

Transitions:

```text
prepared ──(heavy stage started)──> running
running ──> success        : a sparse model exists, cameras_registered > 0, num_points_valid > 0
running ──> partial        : cameras_registered > 0, but only part of the frames registered
running ──> invalid_result : a model exists, but there are no cameras or no valid points
running ──> failed         : the reconstructor finished with an error
running ──> interrupted    : the process was interrupted / the result is incomplete (required files missing)
```

Additional `status.json` fields: `precomputed_example: bool` (FR-028, A-07),
`warnings: [{code, message, observation_only: true}]`, `exportable: bool`.

`partial` is determined by the threshold `registration_ratio < 1.0`. **No numeric threshold for "how much partial
is acceptable" is assigned** — the status records the fact, the evaluation is given by the baseline (Group B in the spec).

### 3.8 SelectionComparison (P2 experiment, `experiments/<exp_id>/`)

| Field | Type | Description |
|---|---|---|
| `exp_id` | str | comparison identifier |
| `video_sha256` | str | the same video for both branches |
| `frame_budget` | int | the same for both branches |
| `reconstructor` | object | the same (model, weights, version) |
| `seed` | int | the same |
| `branches` | [{`selector`, `run_id`}] | exactly two: `uniform` and `quality_nonredundant` |
| `metrics_declared` | list | the list of metrics being compared, **recorded before the runs** |
| `results` | object | metric values per branch + unavailability flags |
| `verdict` | str | `improvement` \| `no_difference` \| `regression` \| `inconclusive` |

The file `experiments/<exp_id>/conditions.json` is created **before** the runs and is not
edited afterwards; changing the conditions = a new `exp_id` (FR-041, FR-043).

---

## 4. Artifact reuse

| Artifact | Reused if | Otherwise |
|---|---|---|
| `package/images/` | `video_sha256`, selection parameters, and the `code_id` of the selection section match | recomputed |
| `result/` | **never recomputed automatically** — this is the expensive stage on another machine | only on an explicit human command |
| `normalized/` | cheap, always recomputed on `ingest` | — |
| `view/session.rrd`, `export/`, `report.md` | cheap, recomputed on demand | — |

**Resuming inside the reconstructor is not supported and not promised**: VGGT is a single forward
pass, there is no intermediate state. An interrupted heavy stage is restarted in full; an already
obtained `result/` is preserved and not deleted automatically.

---

## 5. Integrity of the transfer package

- During preparation on the Mac, `package/SHA256SUMS` is generated over all package files.
- Before the run on the GPU machine a human can verify them with the OS's standard utility
  (`shasum -a 256 -c` / `sha256sum -c`) — **no code of ours is required for this**.
- After `result/` is returned to the Mac, the `ingest` command checks:
  1. image names in `sparse/images.bin` ⊆ names in `frames.json` (`selected == true`);
  2. the number of images in the model ≤ the number of selected frames;
  3. presence of all required result files;
  4. the `run_id` of the directory matches the manifest.
- Any discrepancy → `status = invalid_result` or a refusal naming the specific file/name
  (FR-029). Mixing artifacts of different runs is prohibited.
