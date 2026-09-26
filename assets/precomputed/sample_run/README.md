# Precomputed example run

**WARNING: this is a precomputed example, not the result of a real reconstruction.**

The directory was built by the script `assets/precomputed/make_sample.py` entirely from
nothing: the frames come from the synthetic `ffmpeg testsrc` video, the "reconstructor
result" from the analytically defined scene `assets/synthetic/make_scene.py`. There is no
footage of a real object here, no feature detector and no optimization, so these files
**cannot be used to judge the quality of a reconstruction**: they only check formats,
conventions and the operation of the viewer, export and report without a reconstruction
environment (FR-027).

The honesty mark: `normalized/status.json` has `precomputed_example: true`
(assumption A-07, requirement FR-028). The viewer and the report must show this
mark permanently.

## What is inside

| File | What it is |
| --- | --- |
| `manifest.json` | run information: input, frame transformation, selection parameters |
| `frames.json` | a record of every frame considered |
| `normalized/cameras.json` | camera poses and the declared conventions |
| `normalized/points.ply` | the coloured point cloud |
| `normalized/diagnostics.json` | run measures and the limits of conclusions |
| `normalized/status.json` | the result status and the example mark |
| `package/` | the portable set of frames (in a real run it goes to the reconstructor) |
| `result/` | raw upstream output; **not stored in git** (`.gitignore`) |

## Numbers of this example

- `run_id`: `20260926-214913-70604779`
- status: `success`
- frames selected: 12
- cameras registered: 12
- points: 300 valid of 300 in the model

## How to rebuild

```
python assets/precomputed/make_sample.py
```

The script is idempotent: the directory is recreated entirely. No network, GPU or manual
steps are needed, `ffmpeg` in PATH is enough.
