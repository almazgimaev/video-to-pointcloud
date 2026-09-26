# Contract: states, errors and return codes

**Status: designed contract.**

General rule (FR-007): a message describes the **observation and its possible consequences**,
and does not assert a cause. Phrasings such as "the camera moved too fast" are prohibited;
acceptable is "adjacent frames differ only slightly — reconstruction may fail".

## Return codes

| Code | Meaning | Typical cause |
|---|---|---|
| 0 | success | |
| 1 | unexpected error | unhandled exception |
| 2 | input unusable | file unreadable, container/codec outside the supported list, no video track |
| 3 | not enough usable frames | fewer frames than the minimum after frame selection |
| 4 | package integrity violated | checksums do not match |
| 5 | result incomplete / computation interrupted | some required files missing, log truncated |
| 6 | result empty or invalid | no camera poses or no valid points |
| 7 | result files missing or incompatible | wrong `run_id`, no `normalized/`, unknown format version |
| 8 | run directory already exists | overwrite protection, `--force` needed |
| 9 | comparison conditions violated | P2 branches differ in more than the frame selection method |

## Case matrix

| Case | Behavior | Code | `status` |
|---|---|---|---|
| Corrupted/unsupported file | refuse before processing, list of supported formats, run directory is not created | 2 | — |
| Few usable frames | reported: source / passed selection / minimum; package is not created | 3 | — |
| Many blurry frames | **does not block**, observation warning, share in the report | 0 | prepared |
| Nearly identical viewpoints | **does not block**, observation warning about low diversity | 0 | prepared |
| GPU unavailable / insufficient resources | heavy stage was not started or failed; `package/` remains usable for a rerun; work on the Mac is not blocked | 5 on `ingest` | prepared / failed |
| Interrupted computation | `result/` is marked incomplete; viewing only with a persistent "result incomplete" banner; export is forbidden without `--force` | 5 | interrupted |
| Empty/invalid result | status is **not** "success"; the report with stage diagnostics is kept | 6 | invalid_result |
| Partial result | viewing allowed, persistent "partial" mark, share of used frames in the report | 0 | partial |
| Missing/incompatible files | refuse, stating which file is missing or which `run_id`s do not match | 7 | — |
| Background not separated | this is normal for v1: `background_present = true`, mark in the report and in the viewer | 0 | success |
| Precomputed example | `precomputed_example = true`, persistent badge in the viewer and a line in the report | 0 | as is |

## Warnings

Each warning is an object `{code, message, observation_only: true}`. The `observation_only` field
exists so that a warning cannot be presented in the report and the interface as an established
cause.

Set of v1 codes: `many_blurry_frames`, `low_viewpoint_diversity`, `short_video`,
`few_registered_cameras`, `background_dominates`, `reprojection_error_unavailable`,
`precomputed_example`.
