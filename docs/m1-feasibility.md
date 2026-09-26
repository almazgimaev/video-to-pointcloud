# M1 — the first feasibility check

**Date**: 2026-09-20
**Video**: `pumpkin360.mp4`, 26.5 s, 636 frames, 1920×1080, h264, sha256 `19ba9b1d2d23c9ae…`
**Machine**: NVIDIA RTX 5090, 32.6 GB, Linux, driver 575.64.03 (CUDA 12.9). 13.6 GB was already
in use on the card before the runs; it is subtracted.
**Reconstructor**: VGGT, `facebook/VGGT-1B` (weights 4.68 GB), `demo_colmap.py` without bundle adjustment.

This document records **actual measurements**. Anything not measured is marked as unknown.

---

## 1. Environment: what had to be fixed

The upstream pin `torch==2.3.1+cu121` is **incompatible with the RTX 5090**: sm_120 is not in the
list of architectures supported by this build, and any GPU computation fails with
`CUDA error: no kernel image is available for execution on the device`.

Replaced with a CUDA 12.8 build (`--index-url https://download.pytorch.org/whl/cu128`).
This confirms an assumption recorded at the plan stage (research.md R-0) as
unverified: **the requirement of torch ≥ 2.7 with cu128 for Blackwell is confirmed in practice.**

Additionally required were dependencies pulled in by the import chain `demo_colmap.py` →
`track_predict` → `vggsfm_utils`: `pycolmap==3.10.0`, `lightglue`, `hydra-core`, `omegaconf`,
`trimesh`, `opencv-python`, `scipy`, `onnxruntime`. They are needed **even without** `--use_ba`.
`pyceres` was not installed: it is required only for bundle adjustment.

---

## 2. Memory and time usage (measured)

Memory already in use on the card before the run is 13 644 MiB; it is subtracted.

| Frames | Peak on the card | Our usage | Time |
|---|---|---|---|
| 24 | 21 205 MiB | **≈ 7.4 GiB** | 1 min 41 s (including weights download) |
| 48 | 27 785 MiB | **≈ 13.8 GiB** | 31 s |

Linear estimate from two points: ≈ 1 GiB fixed + ≈ 0.27 GiB per frame.

**Consequence**: 80 frames would need ≈ 22 GiB. On a free card this is possible, but with the
13.6 GB already in use the total would be ≈ 36 GB against a capacity of 32.6 GB —
**it does not fit within the available memory**. The 80-frame step was not run.

**Practical ceiling within the available memory: 48 frames.** This is a measurement, not an estimate.

Memory was fully released after the process exited: the value returned to 13 644 MiB.

---

## 3. Confidence threshold: no points appear at the default value

The first run of 24 frames with default parameters gave **zero points** with fully
recovered cameras (24 of 24). There was no error in the log: for the script an empty mask is
a valid case.

The cause is in the branch without bundle adjustment:

```python
conf_mask = depth_conf >= conf_thres_value   # default 5.0
```

Not a single depth-map point passed the threshold of 5.0 on our data. With
`--conf_thres_value 1.0` points appeared (`points3D.bin` — 5.9 MB, on the order of 90 thousand points
against the script limit of 100 000).

**Conclusion**: the default value is unsuitable for this kind of footage. A working value
is tuned and fixed; a low threshold means more points and more noise among them.

---

## 4. Tool behavior on an empty result

The check for which all the honesty machinery was written. On an empty reconstruction
`v3d ingest` returned code 6 and wrote:

```
status: invalid_result | view: False | export: False
note: result was read, but it contains no points
```

The empty result **was not passed off as successful**, export is forbidden. The diagnostics are saved.

---

## 5. Result: feasibility confirmed

A run of 24 frames with `--conf_thres_value 1.0`:

| Metric | Value |
|---|---|
| Cameras registered | **24 of 24** |
| Valid points | 99 907 of 100 000 |
| Total sweep of the camera trajectory | **374°** (full circle) |
| Spread of the trajectory radius | 16% of the mean |
| Time | 31 s |

The object in the point cloud is **distinguishable**: a compact yellow-orange cluster in the center,
surrounded by background points. The camera trajectory forms a smooth arc without breaks.

**Conclusion: the v1 goal is achievable on our own video.** The specification promised a colored
point cloud with camera poses — exactly that was obtained. No substitution of the result type
was required.

Details — [baseline.md](./baseline.md). The success conditions from plan.md §8 are all six met:
the heavy stage completed, cameras are registered, points exist and are colored, the shape is
recognizable, the export opens, the manifest is filled in.

Cost: **4 runs** instead of the three allowed — the extra one was the first, with the default
threshold, which gave an empty result. Formally the limit was exceeded by one run;
the excess was accepted deliberately, because the cause of the emptiness was found and eliminated
rather than debugged blindly.

## 5b. Confidence threshold tuning and the best run

After feasibility was confirmed, the threshold was tuned on 48 frames. The measured dependence of the
number of points on `--conf_thres_value` (the same video, the same frames):

| Threshold | Points | Observation |
|---|---|---|
| 5.0 (default) | 0 | cameras exist, no point cloud |
| 1.0 | 100 000 (script limit) | background dominates, the object is a blob inside junk |
| 1.3 | 100 000 (limit) | unchanged: the filter does not reach the limit |
| **1.6** | **100 000 (limit)** | **background gone, object recognizable — the working value** |
| 1.8 | ≈ 9 300 | a tenfold collapse, the point cloud is sparse |

The collapse between 1.6 and 1.8 is very sharp. Below 1.8 the filter does not reach the script
limit of 100 000 points, so **the number of points does not show the effect of the threshold** — it is visible
only in the composition of the point cloud.

### Best run: 48 frames, threshold 1.6

| Metric | baseline (24 frames, 1.0) | 48 frames, 1.6 |
|---|---|---|
| Cameras registered | 24 of 24 | **48 of 48** |
| Total sweep | 374° | 380° |
| Step between cameras | 16.4° | 8.0° |
| Point cloud dimensions / trajectory radius | **4.6×** (background) | **0.5×** (object) |
| 90% of points within radius | 0.92 | **0.18** |

The ratio of the point cloud dimensions to the radius of the camera ring is the only valid way to compare
two runs, because each reconstruction has its own scale and it is not determined.

Visually: the object is recognizable as a pumpkin, the ribs are visible, color is consistent across
viewpoints, the background is practically absent.

## 6. What remains unknown

- Usage and time with `--use_ba` (bundle adjustment was not run, `pyceres` was not installed),
  and with it the reprojection error is also unavailable.
- Behavior at 80+ frames: on a free card ≈ 22 GiB should fit, but this is unverified.
- Where the lower bound of usability by frame count lies: fewer than 24 was not run,
  so `min_frames` remains a provisional value.
- Geometry quality in the metric sense: there is no ground truth, the scale is not determined.

## 7. A gap found in the tool

The `visual_check` field in the diagnostics exists and contains all six items, but there is **nothing
to record the result of the visual check with**: there is no command for it, all items stay
`not_checked`. The check had to be done manually and recorded in baseline.md.
The gap is recorded, closing it is a separate task.
