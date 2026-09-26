# Baseline

The first successful end-to-end run. Before it, no time, memory or quality thresholds were
assigned; from now on, versions are compared against these values.

**Established**: 2026-09-20
**Run**: `20260920-084918-43d38b2d`
**Input**: `pumpkin360.mp4` (26.5 s, 636 frames, 1920×1080, h264), sha256 `19ba9b1d2d23c9ae…`
**Reconstructor**: VGGT `facebook/VGGT-1B`, `demo_colmap.py --conf_thres_value 1.0`,
without bundle adjustment, RTX 5090.

## Measured values

| Metric | Value |
|---|---|
| Frames in the video | 636 |
| Candidates considered | 318 |
| Frames selected | 24 |
| Cameras registered | **24 of 24** (ratio 1.0) |
| Points in the model | 100 000 (upstream script limit) |
| Valid points after outlier filter | 99 907 |
| Point cloud dimensions | 3.46 × 2.77 × 3.06 (units unknown) |
| VRAM usage | ≈ 7.4 GiB |
| Reconstruction time | 31 s (excluding weights download) |
| Reprojection error | unavailable: run without bundle adjustment |

## Camera trajectory geometry

| Metric | Value |
|---|---|
| Radius from the centroid | 0.750 ± 0.121 (spread 16%) |
| Step between adjacent cameras | median 16.4°, from 6.1° to 28.8° |
| Total sweep | **374°** — a full circle |

## Visual check (against the six defects from the spec)

Performed on orthogonal projections of the point cloud with the camera poses overlaid.

| Defect | Result | Observation |
|---|---|---|
| Geometry doubling | not detected | — |
| Camera trajectory break | not detected | the arc is smooth, no jumps in the step |
| Shapeless "point soup" | not detected | the object is distinguishable as a compact cluster |
| Gaps on weak texture | not assessed | not visible in the projections |
| Background dominance | **detected** | the background makes up most of the points |
| Color inconsistency | not detected | colors are consistent across viewpoints |

Background dominance is an expected property of v1: no segmentation is performed, and the
confidence threshold was lowered to 1.0 just to get any points at all.

## What this does NOT prove

Metric accuracy and surface completeness were not measured: there is no independent ground truth,
and the scale is not determined. The number of points and their density are not proof of accuracy.
