# Contract: CLI commands (designed interface)

**Status: designed interface. No command has been implemented or executed.**

All commands are the light stage and run **on the Mac**, except the explicitly marked step on the
GPU machine, which is run by the **upstream command**, not ours.

General rules:
- The first positional argument is the subcommand. Common flags: `--json` (machine-readable output),
  `--quiet`, `--force`.
- Any command that modifies a run directory writes an entry about itself to `manifest.json`
  (command, arguments, time, duration).
- No command accesses the GPU machine over the network.

---

## `v3d check <video>`

**Purpose**: verifiable properties of the file before any processing (FR-002, FR-006).

| | |
|---|---|
| Input | path to a video file |
| Action | `ffprobe`: container, codec, duration, resolution, fps, frame count, rotation |
| Output | report to stdout (+`--json`), creates no files |
| Errors | file unreadable → code 2; container/codec outside the list → code 2 listing the supported ones |

Supported v1 input (A-02, to be confirmed at implementation): containers `mp4`, `mov`;
video codecs `h264`, `hevc`; progressive scan. Anything else is rejected — "support for any
video" is not claimed.

---

## `v3d prepare <video> [--budget N] [--selector uniform|quality] [--seed S] [--long-side PX] [--out runs/]`

**Purpose**: stages 1–2 (check, extraction, frame selection, building the transfer package).

| | |
|---|---|
| Input | video file, frame budget, selection method, seed |
| Action | `ffprobe` → `ffmpeg` (extraction with auto-rotation) → frame metrics → frame selection → write package and manifest |
| Output | `runs/<run_id>/` with `manifest.json`, `frames.json`, `package/` (see [run-package.md](./run-package.md)) |
| Location | Mac |
| Errors | not enough usable frames → code 3, package is not created; run directory already exists → code 8 unless `--force` is given |

`--selector quality` is the frame selection variant based on sharpness and non-redundancy
(algorithm in plan.md §7). The default `--budget` value is fixed at implementation and is **not
assigned now**.

Prints: number of source frames, number of selected frames, a summary of rejection reasons,
warnings (as observations, not as a diagnosis — FR-007).

---

## Heavy stage — an upstream command on the GPU machine (our code is not moved there)

**Access conditions**: a Linux machine with an RTX 5090 (32 GB), available intermittently. Runs
on the GPU machine are started manually by the author; the working memory budget is **≤ 24 GB**;
free GPU memory is checked before a run and `nvidia-smi` output is recorded.

```bash
# on the GPU machine, in a clone of facebookresearch/vggt
python demo_colmap.py --scene_dir=/path/to/package
# or with bundle adjustment (gives a reprojection error):
python demo_colmap.py --scene_dir=/path/to/package --use_ba
```

| | |
|---|---|
| Input | the `package/` directory with an `images/` subdirectory |
| Output | `package/sparse/{cameras,images,points3D}.bin`, `package/points.ply` |
| Location | GPU machine |
| Errors | out of VRAM / missing environment → upstream error, code 5 on the `ingest` side; see [errors.md](./errors.md) |

The fallback variant under the same contract (COLMAP, Mac or GPU machine) — see
[reconstructor-io.md](./reconstructor-io.md).

The exact command text is duplicated in `package/RUN_ON_GPU.txt`, so that the manual transfer and
launch are not a hidden step.

---

## `v3d ingest <run_dir> [--from <path to the returned result>]`

**Purpose**: stage 4 — integrity check and conversion to the normalized representation.

| | |
|---|---|
| Input | run directory + the returned `result/` |
| Action | check checksums and name consistency; read the sparse model (`pycolmap`); convert poses to `world_to_camera`; outlier filter; compute diagnostics |
| Output | `normalized/{cameras.json, points.ply, diagnostics.json, status.json}` |
| Location | Mac (no GPU required) |
| Errors | required files missing → code 7; names/`run_id` do not match → code 7; model is empty → code 6, `status=invalid_result`; result is incomplete → code 5, `status=interrupted` |

---

## `v3d view <run_dir>`

| | |
|---|---|
| Input | run directory with `normalized/` |
| Action | log the point cloud, camera poses, run information and warnings to Rerun; save `view/session.rrd` |
| Output | viewer window + `view/session.rrd` |
| Location | Mac, no GPU machine, no recomputation (FR-027) |
| Errors | no `normalized/` → code 7; `status ∈ {interrupted, invalid_result}` → open only with a persistent banner, or refuse (see errors.md) |

Required view elements: the point cloud with colors; camera poses (toggleable); a run
information panel; a warnings panel; a persistent status indicator and, if applicable, a
"precomputed example" badge (FR-028).

---

## `v3d export <run_dir> [--format ply] [--out FILE]`

| | |
|---|---|
| Input | `normalized/points.ply` |
| Action | write a PLY with the same point count, coordinates and colors as displayed (FR-031); write `run_id` and a summary into the PLY header comment |
| Output | `export/object.ply` or the specified file |
| Errors | `status ∈ {invalid_result, interrupted}` → refuse (code 6), or with `--force` a file with the `.UNUSABLE.ply` suffix and a mark in the header (FR-033) |

---

## `v3d report <run_dir>`

A human-readable report `report.md`: input, frames, parameters, environment, timings, metrics
with explanations, unavailable metrics with reasons, warnings, interpretation limits,
scale status and the background-present flag. Available without launching the viewer (FR-035).

---

## `v3d compare <run_dir_a> <run_dir_b> --conditions <conditions.json>`

Comparison of two runs of experiment P2. Refuses to compare if anything other than the frame
selection method differs (video, budget, reconstructor, seed) → code 9. Output:
`experiments/<exp_id>/comparison.md` + `results.json`.

---

## End-to-end launch

- `v3d run <video> ...` **on the Mac** is available only in the fallback variant with local COLMAP,
  where all stages can run on one machine.
- In the primary variant there is **no** single-command end-to-end launch and none is promised:
  between `prepare` and `ingest` sits a manual transfer and an upstream launch on the GPU machine.
  This gap is described explicitly (see quickstart.md), not hidden.

---

## Return codes

Common to all commands — see [errors.md](./errors.md).
