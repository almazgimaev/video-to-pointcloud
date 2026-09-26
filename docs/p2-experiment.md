# P2 — does smarter frame selection improve the reconstruction?

**Status**: done — verdict **improvement** by the rule fixed before the runs. Everything above
"Results" was written and committed before any P2 reconstruction was run.

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

Runs on 2026-09-26, code `b9404702` for all three branches, conditions committed beforehand.
Full table: [`experiments/p2_pumpkin_48/comparison.md`](../experiments/p2_pumpkin_48/comparison.md).

| Metric | uniform | quality_nonredundant | uniform, repeat |
|---|---|---|---|
| Cameras registered | 48 / 48 | 48 / 48 | 48 / 48 |
| Points at confidence 1.8 (primary) | 9,346 | **100,000 — at the upstream cap** | 9,297 |
| Arc coverage / planarity | 334° / 0.055 | 328° / 0.050 | 334° / 0.055 |
| Visual defects | 2 | 1 | 2 |
| Selection cost | 0.16 s | 1.19 s | — |
| Reconstruction time | 41.6 s | 34.3 s | 36.6 s |

**Verdict: `improvement`.** Variant ≥ 1.10 × baseline in the primary metric, the same
registration ratio, fewer visual defects. The two baseline runs differ by 0.5 % and render
identically, so the effect is not run-to-run noise.

### What was actually seen

At confidence 1.8 the uniform branch keeps only scattered fragments of the object — the shape
is not recognizable. The variant keeps the complete object, upright, ribs visible, with the top
almost closed, plus a disc of table surface under it. The variant's cloud is wider for that
reason (median distance from the centre 0.30 of the camera-ring radius against 0.23).

### How to read the numbers

- **The "×10" is a lower bound and it is threshold-dependent.** The variant hit the upstream cap
  of 100,000 points, so its true count is higher. At the same time the threshold 1.8 sits on a
  cliff of the confidence distribution (on M1, going from 1.6 to 1.8 cut the count tenfold), so a
  moderate increase in depth confidence becomes a huge jump in the count. The number does not
  mean "ten times better geometry"; it means the variant's depth is confident enough that the
  whole object survives a strict threshold, while the baseline's is not.
- **At the working threshold 1.6 the uniform branch is already complete** (the M1 run on the same
  48 uniform frames). The variant was not run at 1.6. The practical gain shown here is robustness:
  with the variant's frames the object survives a stricter threshold.
- **The mechanism is not established.** The variant's frames are only 5 % sharper by median,
  and 39 of the 48 frames differ between the branches; the redundancy rule never triggered.
  The effect may come from a few blurred or occluded frames in the uniform set rather than from
  sharpness in general. One video cannot tell these apart.
- **Selection costs about one second** (sharpness for every candidate) — about 3 % of one
  reconstruction; reconstruction times of all branches are within the spread of repeats.
- **The visual check was not blind**: done by the assistant knowing the branch. Borderline items
  were marked against the variant (its small hole at the top counts as a defect).
- **Generalization is not claimed.** One video, one object, one budget, one threshold.

### What would make this stronger

The same comparison on the harder and the deliberately problematic videos of the test set; a
second budget; counting points below the upstream cap (a lower threshold for both branches) to
measure the effect without saturation.
