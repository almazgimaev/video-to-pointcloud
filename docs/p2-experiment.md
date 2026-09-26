# P2 — does smarter frame selection improve the reconstruction?

**Status**: design fixed, runs pending. This document is written before any P2 reconstruction
was run; results are appended below the line "Results" only after both branches are ingested.

## Question

With the same frame budget, the same video and the same reconstructor, does selecting frames by
sharpness while avoiding near-duplicate viewpoints (`quality_nonredundant`) give a better
reconstruction than uniform sampling in time (`uniform`)?

No gain is promised. A negative or neutral result is a valid outcome and is reported as such.

## Method

**Variant `quality_nonredundant`** (`src/v3d/select/quality.py`): the candidate frames are split
into `budget` consecutive windows; in each window the sharpest frame is taken unless its frame
difference from the previously selected frame is below `tau`, in which case the next sharpest
acceptable frame is taken; if none is acceptable, the sharpest is kept and reported. One frame per
window keeps the time coverage — a global sharpness ranking would crowd the selection where the
camera moved slowly and lose viewpoints.

**Baseline `uniform`**: frames evenly spaced over the candidates.

## Tuning (T049) — on a separate video only

`tau` was tuned on `meat360.mp4`, which is **not** part of the evaluation set. The rule was fixed
before computing it: `tau = 0.5 × median frame difference between consecutive frames of a
uniform 48-frame selection`.

| Tuning video | Pairs | Median difference | Quartiles | tau |
|---|---|---|---|---|
| meat360.mp4 | 47 | 0.0246 | 0.0222 / 0.0257 | **0.0123** |

On the tuning video no uniform pair fell below `tau`, and a trial `quality_nonredundant`
selection triggered the redundancy rule 8 times out of 48 windows. Expectation written in
advance: at this budget any difference between the branches will come mostly from sharpness,
not from redundancy.

The evaluation video was not used for tuning.

## Conditions (fixed before the runs)

Stored in `experiments/p2_pumpkin_48/conditions.json`, committed before any P2 reconstruction.

| Item | Value |
|---|---|
| Evaluation video | `pumpkin360.mp4` |
| Frame budget | 48 |
| Seed | 42 |
| Reconstructor | VGGT `facebook/VGGT-1B`, `demo_colmap.py --conf_thres_value 1.8`, no bundle adjustment |
| Branches | `uniform`, `quality_nonredundant`, plus a repeat of `uniform` |

**Why confidence threshold 1.8 and not the working 1.6.** At 1.6 both branches hit the upstream
cap of 100,000 points, so the primary metric would be saturated and could not show a difference.
At 1.8 the M1 run gave about 9,300 points — well below the cap. The threshold is a reconstructor
parameter, the same for both branches; it does not favour either selection method.

**Why a repeat of the baseline.** The reconstructor may not be perfectly deterministic on a GPU.
Running the uniform branch twice on identical frames measures run-to-run variation; an effect
smaller than that variation is reported as `inconclusive`.

## Metrics and decision rule (declared in advance)

Metrics: `cameras_registered`, `registration_ratio`, **`num_points_valid` (primary)**,
`alignment_arc_coverage_deg`, `alignment_planarity`, `selection_time_s`, `selection_time_share`,
`visual_check_defects`.

- `improvement`: variant points ≥ baseline × 1.10, registration not lower, no more visual defects;
- `regression`: variant points ≤ baseline × 0.90, or registration lower, or more visual defects;
- otherwise `no_difference`;
- `inconclusive` when the two baseline runs differ at least as much as variant and baseline.

**Selection cost.** For the variant, computing sharpness is part of the selection cost, because
it needs sharpness to decide. For the baseline, sharpness is only descriptive and is excluded.

## Limits

One video, one run per branch (plus one repeat). The number of points at a fixed confidence
threshold is a proxy for how confident the reconstructor is about depth; it is not a measurement
of geometric accuracy, and no independent reference exists.

## Results

*Pending.*
