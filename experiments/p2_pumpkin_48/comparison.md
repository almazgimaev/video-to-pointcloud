# Comparison `p2_pumpkin_48`

- **Conditions**: `conditions.json`
- **Generated (UTC)**: 2026-09-26T22:47:21.342866+00:00
- **Baseline run (uniform)**: `20260926-221214-e80ab63c`
- **Variant run (quality_nonredundant)**: `20260926-221232-494c1365`
- **Repeat baseline run**: `20260926-221223-e80ab63c`
- **Verdict**: **improvement**

## Metrics

| Metric | Baseline | Variant | Repeat | Relative difference (variant vs baseline) |
| --- | --- | --- | --- | --- |
| cameras_registered | 48 | 48 | 48 |  |
| registration_ratio | 1 | 1 | 1 |  |
| num_points_valid | 9346 | 100000 | 9297 | 9.69976 |
| alignment_arc_coverage_deg | 334.457 | 327.675 | 334.457 |  |
| alignment_planarity | 0.0546391 | 0.0504477 | 0.0546391 |  |
| selection_time_s | 0.159 | 1.191 | 0.159 |  |
| selection_time_share | 0.0207952 | 0.156669 | 0.0208006 |  |
| visual_check_defects | 2 | 1 | 2 |  |

**Relative difference between the two baseline runs (primary metric)**: -0.00524288

## Decision rule

```json
{
  "primary_metric": "num_points_valid",
  "parameters": {
    "improvement_ratio": 1.1,
    "regression_ratio": 0.9
  },
  "improvement": "variant.num_points_valid >= baseline.num_points_valid * improvement_ratio AND variant.registration_ratio >= baseline.registration_ratio AND variant.visual_check_defects <= baseline.visual_check_defects",
  "regression": "variant.num_points_valid <= baseline.num_points_valid * regression_ratio OR variant.registration_ratio < baseline.registration_ratio OR variant.visual_check_defects > baseline.visual_check_defects",
  "no_difference": "neither the improvement nor the regression condition holds",
  "inconclusive": "repeat_baseline is true and the relative difference between the two baseline runs in the primary metric is >= the relative difference between variant and baseline: the effect is not larger than run-to-run variation"
}
```

## Limits of interpretation

**Saturation**: the primary metric reached the upstream cap of 100,000 points in: variant. The true count is higher, so the relative difference is a lower bound; the verdict holds because the rule is satisfied even by the lower bound.

One video, one run per branch; point count at a fixed confidence threshold is a proxy for depth confidence, not proof of geometric accuracy.

