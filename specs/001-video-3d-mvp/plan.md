# Implementation Plan: Video-to-3D MVP

**Feature**: `001-video-3d-mvp` | **Date**: 2026-09-18 | **Spec**: [spec.md](./spec.md)

**Input**: specification `specs/001-video-3d-mvp/spec.md`, constitution v3.1.0,
checklist `checklists/requirements.md`

**Plan status**: **ready**. Hardware questions Q1 and Q2 were resolved by the user's answer on 2026-09-18
(see §2 Constraints and §10). Q3 remains — actual VRAM and time consumption, which is measured
on M1 and is not replaced by assumed values until then.

---

## 1. Summary

A tool with two stages. **Mac**: video validation, frame extraction and selection, assembly of the
transfer package, receiving the result, normalization, viewing, export, report, comparison of
selection methods. **GPU machine**: the single heavy step — reconstruction, executed by the
**standard upstream script**, without moving our code there.

Technical approach: **VGGT** via `demo_colmap.py`, the result in **COLMAP sparse** format
(`cameras.bin`, `images.bin`, `points3D.bin` + `points.ply`). This format is chosen as the single
point of ingestion because classic COLMAP produces it too — the fallback path needs no second
integration. Viewing is the ready-made **Rerun** viewer; no custom frontend is created.
The rationale for the choice and rejected alternatives are in [research.md](./research.md).

The v1 result stays what is fixed in the spec: a colored point cloud, camera poses,
viewing, PLY export, metadata, and a report. Mesh, textures, splat, metric scale —
out of the feature. Scale is marked `not_determined` in all artifacts.

---

## 2. Technical Context

**Language/Version**: Python 3.11+ (Mac). The version is fixed during implementation.

**Primary Dependencies (Mac)**: `numpy`, `Pillow`, `pycolmap` (reading the sparse model),
`rerun-sdk` (viewer), external `ffmpeg`/`ffprobe` (Homebrew).

**Updated 2026-09-19 (implement stage, T002/T006)**: the environment was built and verified on
macOS ARM, Python 3.12.10 — `numpy 2.5.3`, `pillow 12.3.0`, `pycolmap 4.2.0`,
`rerun-sdk 0.38.1`, `pytest 9.1.1`, `ruff 0.16.8`; all import successfully. Versions are fixed
in `requirements.lock`. This section previously carried a note "nothing was installed" — it
matched the state at the plan stage and is now outdated. Compatibility of the **heavy** stage
(PyTorch for sm_120 on the GPU machine) is still **not verified**.

**Primary Dependencies (GPU machine)**: upstream only — a clone of `facebookresearch/vggt`,
its `requirements.txt`, the `facebook/VGGT-1B` weights (CC-BY-NC-4.0), PyTorch built for sm_120.
**Our code is not moved there.**

**Storage**: files on disk, run directory `runs/<run_id>/`. No database, queues, or services.

**Testing**: `pytest` — unit tests of deterministic logic (frame selection, geometric
transformations, reading/writing formats, package integrity) + checks on a synthetic
example and on saved precomputed results. Real GPU runs are manual,
per the quickstart checklist.

**Target Platform**: macOS (Apple Silicon) — development and the whole light stage;
an NVIDIA GPU machine — the heavy step only.

**Project Type**: single project — a Python package + CLI + a viewing subcommand.

**Performance Goals**: **not assigned.** Thresholds for time, VRAM, and quality are set
after the baseline (spec, criteria Group B). Assigning them now would be invention.

**Constraints**:
- The heavy stage runs on another machine, exchange is via files, transfer is manual.
- Our own code is **not deployed** on the GPU machine (user requirement).
- GPU: **NVIDIA RTX 5090, 32 GB VRAM** (clarified by the user on 2026-09-18).
- The GPU machine is available intermittently. A working memory budget of **≤ 24 GB**
  is used — a deliberate self-imposed limit, not the "card maximum". Memory already in use on the
  card before a run is subtracted from the available budget.
- **Runs on the GPU machine are started manually by the author.**
  Automatic connection is prohibited.
- The GPU machine OS is **Linux**; the driver, CUDA, and Python environment are present (per
  the user, "most likely" — verified at first login), internet access is available,
  so cloning the VGGT repository and downloading the weights are feasible.
- PyTorch requirements for sm_120 (CUDA 12.8, torch ≥ 2.7) remain a **documentation claim**
  until verified on the machine.
- Actual VGGT VRAM consumption on N frames is **unknown**, measured on M1 (Q3).

**Scale/Scope**: one video = one object = one run; expected order of 50–150
selected frames (an assumption, not a measurement). One user, offline.

---

## 3. Constitution Check (before design)

Constitution v3.1.0.

| Gate / principle | How it is met | Status |
|---|---|---|
| **I. Honesty of results** | artifact type `point_cloud` in metadata; `scale_status = not_determined`; axis conventions recorded explicitly; completion of invisible surfaces is not applied; the words mesh/splat are not used | PASS |
| **II. Reproducibility** | run manifest with video hash, parameters, seed, `code_id`, environment, times; `run_id` from a digest of inputs; directories are not overwritten; each stage is one command; the cross-machine transfer is described as a standard step (contracts/run-package.md) | PASS |
| **III. One scenario** | the plan introduces no capability outside the spec; the "Out of scope" section is not expanded | PASS |
| **IV. Stage separation** | §5: the execution location is stated for each stage; the light stage does not require the GPU machine; the COLMAP-on-Mac fallback gives an end-to-end run without it entirely | PASS |
| **Hardware gate** | §2 Constraints: Linux + RTX 5090 32 GB are known; a ≤ 24 GB budget as a self-imposed limit; torch requirements are marked "per documentation, not verified"; memory consumption is measured, not assumed | PASS |
| **V. Measurability** | metrics are taken from the spec unchanged; unavailable ones are marked with a reason; interpretation limits are recorded in diagnostics | PASS |
| **VI. Engineering contribution** | borrowings are listed (§9) stating what was taken and what was done by the author; no abstractions are introduced without a user | PASS |

**Conclusion before Phase 0**: all gates are passable. Hardware information was obtained before the start
of designing the heavy stage; the only thing left unverified is actual resource consumption,
which is measured on M1.

---

## 4. Project Structure

### Feature documentation

```text
specs/001-video-3d-mvp/
├── plan.md              # this file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/           # Phase 1
│   ├── cli.md
│   ├── run-package.md
│   ├── reconstructor-io.md
│   └── errors.md
├── spec.md
└── checklists/requirements.md
```

### Source code (repository root) — planned layout

```text
src/v3d/
├── cli.py                 # subcommand parsing, return codes
├── probe.py               # ffprobe: file properties, supportability
├── extract.py             # ffmpeg: frame extraction, orientation, resize
├── metrics.py             # sharpness, frame difference
├── select/
│   ├── uniform.py         # baseline
│   └── quality.py         # selection by sharpness and non-redundancy
├── package.py             # assembling package/, SHA256SUMS, RUN_ON_GPU.txt
├── ingest.py              # integrity checks, reading COLMAP sparse
├── geometry.py            # conventions, world↔camera, camera centers, outlier filter
├── artifacts.py           # manifest/frames/cameras/diagnostics/status: writing and reading
├── viewer.py              # logging to Rerun
├── export.py              # PLY
├── report.py              # report.md
└── compare.py             # P2 experiment

tests/
├── unit/                  # metrics, selection, geometry, formats, error codes
├── contract/              # schemas of manifest/cameras/diagnostics, PLY, SHA256SUMS
└── integration/           # synthetic scene + saved precomputed results

assets/
├── synthetic/             # synthetic scene generator (formats and geometry)
└── precomputed/           # saved results for interface development, marked as such
```

**Structure Decision**: a single Python package `src/v3d` with a CLI and a viewing subcommand.
No separate frontend is created (R-5). There is no "swappable backend" abstraction module: the boundary around the
reconstructor is a single COLMAP sparse format reader in `ingest.py` + `geometry.py` (R-3).

---

## 5. Processing stages

Format: **input → action → output → location → error check**.

### Stage 1. Video validation, frame extraction and selection

- **Input**: a video file.
- **Action**: `ffprobe` (container, codec, duration, resolution, fps, rotation) →
  supportability decision → `ffmpeg` extracts frames with auto-rotation and optional resize
  by the long side → computing sharpness and frame difference → selection by the chosen method.
- **Output**: frames on disk, `frames.json` (a record for every frame considered),
  a draft `manifest.json`.
- **Location**: **Mac**, no GPU needed.
- **Error check**: an unreadable file or a format outside the list → code 2, nothing is created;
  fewer than the minimum number of usable frames → code 3, the package is not created; many blurry frames or low
  viewpoint diversity → an observation warning, execution continues.

### Stage 2. Transfer package

- **Input**: selected frames, parameters, `run_id`.
- **Action**: assembling `package/images/`, computing `SHA256SUMS`, generating `RUN_ON_GPU.txt`
  with the exact upstream command; finalizing `manifest.json`.
- **Output**: `runs/<run_id>/package/` — self-contained, transferable.
- **Location**: **Mac**.
- **Error check**: the run directory already exists → code 8 (`--force` needed); a write
  error → the run is marked incomplete, `package/` is not declared valid.

### Stage 3. Heavy reconstruction

- **Input**: `package/` (a directory with `images/`).
- **Action**: an **upstream command**, started by a human:
  `python demo_colmap.py --scene_dir=<package>` (optionally `--use_ba`).
- **Output**: `sparse/{cameras,images,points3D}.bin`, `points.ply`, a log.
- **Location**: **GPU machine**. Our code is not there.
- **Error check**: environment errors and lack of VRAM show up as a nonzero upstream exit code
  and incomplete output; they are detected at stage 4 (codes 5/6/7). Interruption → the result is incomplete,
  `package/` remains valid for a full re-run. Resuming from the middle
  is not supported and not promised.

### Stage 4. Bringing the result to a consistent representation

- **Input**: `result/` + `frames.json` + `manifest.json`.
- **Action**: integrity and name-match checks; reading the sparse model; converting poses
  to `world_to_camera` (OpenCV) and computing camera centers; outlier filter; collecting diagnostics;
  determining the status.
- **Output**: `normalized/{cameras.json, points.ply, diagnostics.json, status.json}`.
- **Location**: **Mac**, no GPU.
- **Error check**: required files missing or `run_id` mismatch → code 7; the model is empty
  (no cameras or no valid points) → code 6, `status=invalid_result`; output incomplete →
  code 5, `status=interrupted`; only part of the frames registered → `status=partial` (not an error).

### Stage 5. Local viewing, export, and report

- **Input**: `normalized/`.
- **Action**: logging to Rerun (cloud, camera poses, panels for information, warnings,
  and status); exporting a PLY that matches the displayed cloud; generating `report.md`.
- **Output**: the viewer window, `view/session.rrd`, `export/object.ply`, `report.md`.
- **Location**: **Mac**, without the GPU machine and without recomputation.
- **Error check**: no `normalized/` → code 7; status `interrupted`/`invalid_result` →
  a permanent banner and export prohibited without `--force`; `run_id` mismatch between files →
  refusal naming the file.

### Stage 6. Comparison of frame selection methods (P2)

- **Input**: one video, the fixed `conditions.json`, two completed runs.
- **Action**: verifying that only the selection method differs; a side-by-side of the pre-declared
  metrics; recording the verdict.
- **Output**: `experiments/<exp_id>/{conditions.json, results.json, comparison.md}`.
- **Location**: **Mac** (the heavy runs of both branches were done beforehand on the GPU machine).
- **Error check**: anything other than the selection method differs → code 9, the comparison
  is not performed; an unavailable metric → marked unavailable, not replaced by zero.

---

## 6. Geometry and data — key fixations

Details are in [data-model.md](./data-model.md). Here only what determines correctness:

1. **The direction of the transformation** is recorded explicitly: `transform_direction = world_to_camera`,
   formula `x_cam = R·x_world + t`. A reconstructor that outputs camera-to-world is converted
   by an adapter; the contract does not change.
2. **Axes**: OpenCV (+X right, +Y down, +Z forward), a right-handed world, the origin is set
   by the reconstructor.
3. **Units**: `unknown`; **`scale_status = not_determined`** in all v1 artifacts.
4. **Intrinsics** refer to the saved frame (after rotation and resize); the size is stated
   next to them in the `image_size` field. Image transformations (rotation, resize, `crop = null`)
   are recorded in the manifest.
5. **Camera-to-frame link** — by the image file name, which matches `frame_id`.
6. **Point color** — from the reconstructor's result; no custom coloring is introduced.
7. **A valid point** is defined in writing before implementation (finite coordinates, passed the
   reconstructor's confidence threshold, not discarded by the outlier filter).
8. **Unavailable fields** are recorded as `null` + `unavailable_reason`, zero is prohibited.
9. **Two runs do not overwrite each other**: `run_id` from a digest
   (video hash + parameters + `code_id` + seed), the directory is not overwritten without `--force`.
10. **Expensive stages are preserved**: `result/` is never recomputed automatically;
    `normalized/`, the report, and the export are cheap and are recomputed freely.

---

## 7. Frame experiment (P2)

Performed **only after P1 works**.

**Baseline** — uniform sampling of `budget` frames over time.

**Variant "quality_nonredundant"** (all computation on the Mac):
1. Split the video into `budget` windows of equal duration — this guarantees coverage of all parts
   of the recording.
2. In each window choose the frame with maximum sharpness.
3. If the chosen frame is too similar to the previous selected one (`diff < τ`), take in this window
   the next frame by sharpness that satisfies the threshold; if there is none — keep the best
   by sharpness and mark it `redundant`.
4. The budget is never exceeded; a shortfall of frames is allowed and recorded.

**Why not "just the sharpest"**: global selection by sharpness pulls the sample toward areas
of slow motion and **loses viewpoints** — a direct risk for camera recovery. The windowed scheme
exists precisely against this risk; time coverage is recorded in the report.

**Comparison conditions** (fixed in `conditions.json` before the runs): the same video,
the same `budget`, one reconstructor and the same weights, one seed, one `code_id`, a pre-declared
list of metrics.

**Comparison metrics** (from the spec only): `cameras_registered` and `registration_ratio`;
`num_points_valid`; reprojection error with `--use_ba` (otherwise — "unavailable" with a reason);
reconstruction time; **the time of frame selection itself** and its share of total time; visual
check against the six pre-described defects.

**Analysis order**: first the conditions are fixed → the baseline run → the variant run →
only then reading the metrics → the verdict `improvement | no_difference | regression |
inconclusive`. Changing the conditions retroactively = a new `exp_id`.

**Threshold tuning** (`τ`, window) is done on a **separate tuning video** not included
in the fixed verification set from the spec. Tuning on the verification set is prohibited.

**No gain is promised in advance.** A negative result is a valid outcome and is stored
as such.

---

## 8. Checks and the first feasibility check

### M1 — the first technical milestone (only designed at the plan stage)

- **Input**: one own video shot per the guide: a textured static object,
  a slow full walk-around, stable lighting, 30–60 s.
- **Expected artifacts**: `package/` → `result/sparse/*` + `points.ply` → `normalized/*` →
  opened viewing → `export/object.ply` → `report.md`.
- **Success conditions** (observable, no invented thresholds):
  1. the heavy stage finished without error and wrote all required files;
  2. `cameras_registered > 0`, and the share of registered frames is recorded;
  3. `num_points_valid > 0`, the cloud is colored;
  4. visual check: the object's shape is recognizable, the camera trajectory has no obvious breaks
     (items 1–3 and 6 of the defect list from the spec);
  5. the exported PLY opens in a third-party viewer;
  6. the manifest is fully filled in, unavailable fields are marked with a reason.
- **The first successful run is declared the baseline** — no thresholds are assigned before it.
- **Reasons to revisit the approach**: lack of VRAM at a frame budget ≥ 40 → VGGT-Omega or
  a smaller budget; implausible poses with compliant shooting → checking the same input via
  COLMAP; an unsuitable GPU machine environment → COLMAP.
- **Cost limit**: no more than **three** GPU runs and one working session on M1.
  When exhausted — switch to the fallback path, rather than debugging someone else's model.

### Check levels

| Level | Where | What it checks | With what |
|---|---|---|---|
| A. Quick checks | Mac, automatic | ffprobe parsing, orientation, resize, frame metrics, selection determinism, geometry (world↔camera, camera centers, round trip), reading/writing formats, error codes, `SHA256SUMS`, refusal on `run_id` mismatch | `pytest`, synthetic scene |
| B. Viewer and export check | Mac, on a saved example | opening without GPU, showing cameras and panels, `partial`/`precomputed` badges, export matching the displayed cloud | saved precomputed results (marked as such) |
| C. Integration GPU runs | GPU machine, manual | the real heavy stage, actual VRAM and time consumption, completeness of the result | quickstart checklist |
| D. Quality comparison | Mac + previously obtained results | the fixed set of three videos from the spec, comparison of applicable metrics and visual check | `v3d compare`, reports |

**Special attention** (per the task statement): coordinates and direction of transformations;
image transformations (rotation/resize and the consistency of intrinsics); invalid data
(NaN/Inf, an empty model); an interrupted run; lack of resources.

**The synthetic example** is used only to check formats and geometric operations.
It **does not prove** the quality of real reconstruction, and is labeled accordingly.

**No conflicts with the spec metrics**: the plan uses exactly the metrics fixed
in the specification. If the chosen reconstructor requires changing their meaning or the user-facing
promise (for example, metric scale appears), this is handled by an explicit spec amendment,
not a silent change.

---

## 9. Borrowings and own contribution (Principle VI)

| Taken | Why | Alternatives | Our contribution |
|---|---|---|---|
| VGGT + `VGGT-1B` weights | reconstruction in a single pass | Pi3X, COLMAP (research.md R-1, R-2) | data preparation, transfer contract, result normalization, diagnostics |
| COLMAP sparse format | a single result contract | a custom format | reader, integrity checks, normalization of conventions |
| `pycolmap` | reading `*.bin` | a custom parser (fallback) | — |
| Rerun | interactive viewing | viser, Open3D | logging schema, information and status panels |
| ffmpeg / ffprobe | decoding, orientation | OpenCV, PyAV | property parsing, metrics, selection |
| ffmpeg animation encoders (`libwebp_anim`, `libx264`) | turntable animations for the page (US3) | GIF only, matplotlib animation | frame rendering, captions, the video-frame-versus-render pairing |
| Astro + Tailwind | static project page (US3) | plain HTML/CSS, Next.js static export | page structure, content, report rendering |
| three.js (`PLYLoader`, `OrbitControls`) | interactive 3D view on the page (US3) | a pre-rendered video only, Potree | deterministic web decimation, on-demand loading, labelling |

The own contribution as a whole: frame selection and its experiment, the transfer and integrity contract,
the data model and conventions, diagnostics and the report, error and status handling, the viewer schema.

---

## 10. Constitution Check (after design)

| Gate / principle | Check after Phase 1 | Status |
|---|---|---|
| I. Honesty | `scale_status=not_determined` is baked into the data-model; `background_present=true` is mandatory; segmentation and completion are absent; the contracts contain no words mesh/splat | PASS |
| II. Reproducibility | `run_id` from a digest, `code_id` + `working_tree_dirty`, seed, `SHA256SUMS`, a ban on silent overwriting, recording weights versions and the upstream command; the run rule is met per the v3.0.0 wording (see §12) | PASS |
| III. Boundaries | no new capabilities introduced; a segmentation model is explicitly rejected; end-to-end launch with one command is not promised where it does not exist | PASS |
| IV. Stages | §5 states the location of each stage; the light stage is entirely on the Mac; a COLMAP-on-Mac fallback path; our code is not moved to the GPU machine | PASS |
| **Hardware gate** | Linux, RTX 5090 32 GB, environment and internet are available; a ≤ 24 GB budget as a self-imposed limit; runs are started manually; torch requirements are marked "per documentation, not verified" | PASS |
| V. Measurability | spec metrics unchanged; the unavailability rule; `interpretation_limits`; visual check kept separate from quantitative; the failing video is verified as a failure | PASS |
| VI. Contribution | §9 | PASS |

**State of the questions**:

- **Q1 — closed** (2026-09-18): the GPU machine runs Linux, the driver/CUDA and Python environment are present,
  internet is available, and ssh access is performed manually by the user. The "most likely" caveat on the environment
  is verified at first login, before M1.
- **Q2 — closed** (2026-09-18): RTX 5090, 32 GB. The GPU machine is available intermittently, and the working budget
  is limited to **≤ 24 GB**; runs are started manually by the author.
- **Q3 — open, non-blocking**: actual VRAM and time consumption on N frames is measured
  on M1. The default `--budget` value is not assigned until then.

**Rules for working with the GPU machine** (a consequence of the user's answer, in force at all
subsequent stages):

1. Runs on the GPU machine are started manually by the author.
2. Connecting over ssh is done manually; automatic connections are prohibited.
3. A run is planned to fit within ≤ 24 GB; when short — reduce the frame
   budget rather than take the whole card.
4. Before a run, the `nvidia-smi` output is recorded (including memory already in use on the card
   before the run, which is subtracted from the budget); if there is not enough free memory, the run is postponed.

---

## 11. Implementation order in major stages

Not tasks — stages. `tasks.md` is created by a separate command and is not produced here.

> **Superseded for the remaining work by the 2026-09-26 amendment (§13).** Stages 1–4 below are
> done. The remaining order is: repository hygiene → world alignment → media rendering →
> P2 experiment → report, README and page → polish. `tasks.md` "Order after the 2026-09-26
> amendment" is the authoritative sequence.

1. **Skeleton and the light input stage**: `probe`, `extract`, `metrics`, uniform selection,
   `package` + `SHA256SUMS` + `RUN_ON_GPU.txt`, the manifest. Level A checks.
2. **Ingestion and normalization**: the COLMAP sparse reader, geometry and conventions, outlier filter,
   diagnostics, statuses, error codes. Level A checks on the synthetic example.
3. **Viewing, export, report**: the Rerun schema, PLY export, `report.md`. Level B checks
   on a saved precomputed result.
4. **M1 — the first feasibility check** on the GPU machine (level C). Establishing the baseline.
5. **The COLMAP fallback path** (if needed after M1, or as insurance for independence
   from the GPU).
6. **The P2 experiment**: the `quality_nonredundant` selection variant, `conditions.json`, `compare`,
   runs on the fixed set (level D).
7. **README and honesty gates**: a description of limitations, the weights license (NC), scale status,
   marking of selected examples.

---

## 12. Resolved conflict: one-command launch (finding D1)

`/speckit-analyze` found a direct conflict: constitution v2.0.0 required "launching the pipeline
with one command", while Principle IV requires separating the light and heavy stages across machines. With an
intermittently available GPU machine and manual transfer, both requirements cannot be met at once.

**How it was resolved**: by amending the constitution to **v3.0.0** (MAJOR), not by reinterpreting it
in the plan. The new wording of Principle II requires:

1. each pipeline stage is launched with **one command** with a config;
2. an end-to-end launch with one command is mandatory **where all stages can run on one machine**
   (for us — the COLMAP fallback on the Mac, §R-4 of research.md);
3. the cross-machine transfer is described as a **standard pipeline step**: what is transferred, which command
   continues the work, how integrity is verified.

**How it is met in this plan**: item 1 — all `v3d` subcommands in contracts/cli.md; item 2 —
the COLMAP-on-Mac fallback; item 3 — contracts/run-package.md and quickstart.md §5, where the transfer
is described as a separate numbered step with a `SHA256SUMS` check, not as a default.

---

## 13. Amendment 2026-09-26: alignment, presentation, English-only repo (US3)

Constitution v3.1.0. Scope added to the spec: User Story 3, FR-045…FR-055, SC-015…SC-020,
A-13…A-15. Decisions below; tasks T063+ in `tasks.md`.

### 13.1 World alignment (FR-045…FR-048)

**Problem, measured.** On `pumpkin360.mp4` the normal of the camera ring was 44.8° away from the
frame's "up", and the cloud centre was 0.90 from the origin: the object looked tilted and orbited
around an empty point. Cause: VGGT anchors the world to the first camera; there is no gravity
information (IMU unused, A-03).

**Decision.** In `ingest`, before writing `normalized/`:

1. **Vertical** = normal of the plane fitted (SVD) to camera **centres**. Positions do not depend
   on per-frame roll, so an unlevelled horizon (A-13) does not bias the estimate.
2. **Sign** = the side the mean camera "up" vector (`-Rᵀ·e_y`) points to; roll averages out over
   the orbit. Agreement is recorded.
3. **Quality gate** (preliminary thresholds, like `min_frames`): planarity `s3/s2`, angular arc
   coverage of the orbit, sign agreement, minimum number of cameras. Fail → re-centre only,
   `status = not_estimated`, warning `orientation_not_estimated`.
4. **Centre** = least-squares point nearest to all optical axes: every camera looks at the object,
   so this does not depend on the noisy background. Fallback: median of the points.
5. **Output frame**: right-handed, **+Y up** (external viewers such as MeshLab and CloudCompare
   then show the object upright). Yaw fixed by `first_camera_in_front`, so a render from the first
   camera pose can be placed next to the first video frame.
6. **Record** as `world_alignment` (data-model §3.4a). Raw `result/sparse/` untouched. Viewer and
   export read the same aligned files, so FR-031 holds without extra work.

Measured on the M1 capture: out-of-plane spread of camera centres 0.175 against 3.6 / 3.2 in
plane — a strong planarity signal for a handheld orbit.

**Rejected**: fitting the plane to points (background dominates at low thresholds); using per-frame
camera up vectors as the vertical (unlevelled horizon, A-13); asking the user to rotate manually
(not reproducible).

### 13.2 Visual check command (FR-049)

`v3d visual-check <run> <item> ok|defect|not_checked [--note TEXT]` updates `diagnostics.json`
with `checked_at`. The report already renders the field. Closes the gap found in M1.

### 13.3 Media rendering (FR-050…FR-052)

`src/v3d/render.py` + `v3d media <run>` → `runs/<id>/media/` (gitignored). CPU only: numpy + Pillow
perspective splatting with depth sort; ffmpeg for animations (`libwebp_anim`, `libx264`
available locally). Outputs: views, turntable, video-frame-versus-render pair from the same camera
pose (uses the stored intrinsics), camera trajectory, a deterministically decimated web PLY with
"N of M points". Every asset captioned with `run_id` and source. Charts as hand-styled SVG.

**Rejected**: Open3D offscreen (extra heavy dependency, macOS rendering quirks); screenshots from
the Rerun viewer (manual, not reproducible); default matplotlib styling (explicitly unwanted).

### 13.4 Project page and README (FR-053, FR-054)

- `site/`: **Astro** (static output by default, Markdown content, near-zero JS) + **Tailwind**.
  Landing page: visuals only; the detailed report is a separate page; the interactive 3D view is
  a three.js island (`PLYLoader` + `OrbitControls`) that loads the decimated PLY on click.
- Output `site/dist/` (gitignored); opens from disk without a server. No analytics.
- Hosting (GitHub Pages) only on explicit request.
- `docs/report/`: the detailed project report (M1 → P2 → decisions → failures), rendered by the
  site and linked from the README.

### 13.5 Repository language and infrastructure privacy (FR-055)

- Repository content is English (constitution v3.1.0).
- Run logistics for the GPU machine live in a local notes file excluded through
  `.git/info/exclude` — not `.gitignore`, so its name never appears in the repository.
- `tests/contract/test_repo_hygiene.py` fails on letters outside the Latin and Greek scripts,
  absolute home paths, wording about how the GPU machine is accessed or shared, and on private
  patterns read from the local notes file.
- The generated sample stores the video filename, not an absolute temp path.

### 13.6 Remote repository

Private remote `video-to-pointcloud`. The hygiene test must pass before every push.

---

## Complexity Tracking

There are no constitution violations requiring justification. Partial passing of the hardware gate —
that was the open question Q1, not a deviation: the plan is explicitly not declared fully ready.

| Potential complication | Decision |
|---|---|
| A framework of swappable 3D backends | **not introduced**; instead a single COLMAP sparse format reader |
| A custom viewing frontend | **not introduced**; the ready-made Rerun viewer |
| A segmentation model | **not introduced**; background is allowed and marked (research.md R-8) |
| A DB, queue, service, SSH automation | **not introduced**; file exchange, manual transfer |
