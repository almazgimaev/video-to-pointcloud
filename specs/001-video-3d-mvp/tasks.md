---
description: "Implementation task list for the video-3d-mvp feature"
---

# Tasks: Video-to-3D MVP

**Input**: design documents from `/specs/001-video-3d-mvp/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md),
[data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md),
constitution v3.1.0

**Tests**: included **as mandatory**, but not as TDD dogma — as required by the constitution:
"Automated checks are mandatory for deterministic logic (frame selection, geometry,
I/O, export formats)". There are no automated tests for reconstruction quality — there it is a
comparison of applicable metrics and a visual check.

**Organization**: tasks are grouped by the user stories of spec.md (US1 = P1, US2 = P2),
so that each can be completed and verified independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no unfinished dependencies)
- **[Story]**: `[US1]` / `[US2]`; the Setup, Foundational and Polish phases carry no story labels

## Path conventions

Single project: `src/v3d/`, `tests/`, `assets/` at the repository root (plan.md §4).

## GPU machine rule (applies in all phases)

The Linux + RTX 5090 (32 GB) machine is available intermittently. Runs on the GPU machine are
started manually by the author; the working memory budget is **≤ 24 GB**; before a run, check
free GPU memory and record `nvidia-smi`. Tasks that require a GPU are marked **(GPU)**.

---

## Phase 1: Setup (shared infrastructure)

**Goal**: a project skeleton on which checks can be written and run on the Mac.

- [X] T001 Create the repository structure per plan.md §4: `src/v3d/`, `src/v3d/select/`, `tests/unit/`, `tests/contract/`, `tests/integration/`, `assets/synthetic/`, `assets/precomputed/`, `runs/` (in `.gitignore`)
- [X] T002 Create `pyproject.toml`: package `v3d`, Python 3.11+, dependencies `numpy`, `Pillow`, `pycolmap`, `rerun-sdk`, extra `dev` (`pytest`, linter), console entry point `v3d`
- [X] T003 [P] Configure the linter and formatter (ruff + black or equivalent) in `pyproject.toml` and add a check command to the `Makefile`
- [X] T004 [P] Create `.gitignore` (exclude `runs/`, `.venv/`, large videos and weights; **do not ignore** `experiments/**/conditions.json`, `results.json`, `comparison.md`) and initialize the git repository — **done 2026-09-18**, branch `main`, commit `ba9f480`
- [X] T005 [P] Create a `README.md` skeleton with sections: purpose, scope limitations, license of the `facebook/VGGT-1B` weights (CC-BY-NC-4.0), scale status `not determined`, "what was borrowed / what was built by us" and a **glossary of terms** (uniform names: "transfer package" = the `package/` directory; "run" = `runs/<run_id>/`; "normalized result" = `normalized/`) — from here on, use only these names in all documents and messages
- [X] T006 Record the actually installed dependency versions in `requirements.lock` and document how to update it in `README.md`

**Checkpoint**: `pip install -e ".[dev]"` succeeds, `v3d --help` runs (empty skeleton), `pytest` collects.

---

## Phase 2: Foundational (blocking prerequisites)

**Goal**: shared entities, conventions and error rules on which both stories depend.

**⚠️ CRITICAL**: until this phase is complete, work on US1 and US2 does not begin.

- [X] T007 Implement return codes and the application error class in `src/v3d/errors.py` strictly per [contracts/errors.md](./contracts/errors.md) (codes 0–9, warning object with `observation_only: true`)
- [X] T008 Implement computation of `run_id` and `inputs_digest` in `src/v3d/runid.py` per data-model.md §2 (sha256 of the video hash + canonical JSON of parameters + `code_id` + seed), including determination of `code_id` (git commit or sha256 of the package) and the `working_tree_dirty` flag
- [X] T009 Implement schemas and artifact read/write in `src/v3d/artifacts.py`: `manifest.json`, `frames.json`, `cameras.json`, `diagnostics.json`, `status.json` — with required fields and the unavailability rule (`{value: null, availability: "unavailable", reason: ...}`), forbidding zero in place of unavailability
- [X] T010 [P] Implement geometry in `src/v3d/geometry.py`: OpenCV conventions, storing `R,t` as `world_to_camera`, computing `camera_center_world = -Rᵀt`, converting camera-to-world → world-to-camera, IQR outlier filter with parameters recorded in `filters_applied`
- [X] T011 [P] Implement the CLI skeleton in `src/v3d/cli.py`: subcommands `check, prepare, ingest, view, export, report, compare`, common flags `--json/--quiet/--force`, mapping exceptions to return codes from `errors.py`
- [X] T012 [P] Implement silent-overwrite protection in `src/v3d/runs.py`: creating `runs/<run_id>/`, refusing with code 8 if the directory exists, writing `forced_overwrite: true` with `--force`
- [X] T013 [P] Create a synthetic scene generator in `assets/synthetic/make_scene.py`: known camera poses on a circle, points with colors, written in COLMAP sparse format; note in the docstring that the example checks **only formats and geometry** and does not prove reconstruction quality
- [X] T014 [P] Geometry tests in `tests/unit/test_geometry.py`: forward/inverse transform, camera centers on the synthetic scene, NaN/Inf handling, determinism of the outlier filter
- [X] T015 [P] Artifact and `run_id` tests in `tests/contract/test_artifacts.py`: required fields, the unavailability rule, refusal on `run_id` mismatch between files, stability of `inputs_digest`
- [X] T016 [P] Return code test in `tests/unit/test_errors.py`: conformance to the case matrix in contracts/errors.md

**Checkpoint**: shared conventions are fixed in code and tests; US1 can begin.

---

## Phase 3: User Story 1 — Obtain and inspect a reconstruction (Priority: P1) 🎯 MVP

**Goal**: an end-to-end path "video → preparation → (GPU) reconstruction → transfer → viewing,
export, report" with the run information saved.

**Independent Test**: one suitable video goes through the whole path; the result opens on the Mac
without a GPU and without recomputation; the exported PLY matches the displayed point cloud.

### Stage 1 — video check, frame extraction and selection

- [X] T017 [P] [US1] Implement `src/v3d/probe.py`: parse `ffprobe` (container, codec, duration, resolution, fps, `nb_frames`, rotation from side data), decide supportability (mp4/mov, h264/hevc) and refuse with code 2 with a list of supported formats
- [X] T018 [US1] Implement `src/v3d/extract.py`: frame extraction via `ffmpeg` with auto-rotation, optional resize by the long side, write `ImageTransform` (applied rotation, sizes, `scale_factor`, `crop = null`, jpeg quality) to the manifest; **for each extracted frame, write `src_index` and `timestamp_us` (PTS in microseconds from the start of the video)** to `frames.json` (FR-005, data-model.md §3.2)
- [X] T019 [P] [US1] Implement `src/v3d/metrics.py`: sharpness as the variance of the Laplacian and adjacent-frame difference as MAD on grayscale 64×64; state explicitly in the docstring that the frame difference is a **proxy** for motion, not a measurement of viewpoint shift
- [X] T020 [US1] Implement uniform selection in `src/v3d/select/uniform.py` (baseline) with a frame budget and seed; fill `frames.json` for every considered frame (including `reject_reason`)
- [X] T021 [US1] Implement the frame sufficiency check in `src/v3d/config.py` + `src/v3d/select/`: the `min_frames` threshold is **set by config**, the initial value is marked in code and in the report as **provisional, not justified by measurement**; if the number of selected frames is below the threshold — code 3, the package is not created, the message contains source/selected/threshold
- [X] T022 [US1] Implement observation warnings in `src/v3d/warnings_.py`: `many_blurry_frames`, `low_viewpoint_diversity`, `short_video` — phrasings without naming a cause (FR-007), all with `observation_only: true`
- [X] T023 [P] [US1] Selection and metrics tests in `tests/unit/test_select_uniform.py` and `tests/unit/test_metrics.py`: determinism with the same seed and inputs, budget compliance, correctness of rejection reasons

### Stage 2 — transfer package

- [X] T024 [US1] Implement `src/v3d/package.py`: layout `package/images/<frame_id>.jpg` (compatible with `demo_colmap.py --scene_dir`), compute `SHA256SUMS`, generate `RUN_ON_GPU.txt` with the exact upstream command, expected outputs and `run_id`
- [X] T025 [US1] Wire `check` and `prepare` in `src/v3d/cli.py`: `check` = probe + a suitability verdict (codes 0/2, `--json`); `prepare` = probe → extract → metrics → select → package → finalize `manifest.json`; print a summary (source/selected frames, rejection reasons, warnings)
- [X] T026 [P] [US1] Package contract test in `tests/contract/test_package.py`: package contents, correctness of `SHA256SUMS`, no manifest inside `package/`, presence of `RUN_ON_GPU.txt`

### Stage 4 — result ingest and normalization

- [X] T027 [US1] Implement COLMAP sparse reading in `src/v3d/ingest.py` via `pycolmap`: cameras, poses, points with colors; intrinsics tied to the `image_size` of the saved frame
- [X] T028 [US1] Implement integrity checks in `src/v3d/ingest.py`: required files present, the model's image names ⊆ selected frames, `run_id` match, non-zero sizes → codes 5/6/7 per contracts/errors.md
- [X] T029 [US1] Implement normalization in `src/v3d/ingest.py`: write `normalized/cameras.json` with a `conventions` block (`world_to_camera`, OpenCV axes, `units: unknown`, `scale_status: not_determined`) and the **mandatory field `artifact_type: "point_cloud"`** (FR-021), `normalized/points.ply` (binary LE, xyz + rgb), apply the outlier filter
- [X] T030 [US1] Implement diagnostics collection in `src/v3d/diagnostics.py`: frames, cameras and `registration_ratio`, `num_points_raw/valid` with a written `valid_point_definition`, **`has_colors` and `background_present: true`** (FR-020), stage timings, artifact sizes, reprojection error or an "unavailable" mark with a reason, an `interpretation_limits` block, **a `visual_check` field — the six items from spec.md with values `ok | defect | not_checked` and a `note` field** (SC-013)
- [X] T031 [US1] Implement status determination in `src/v3d/status.py`: `success | partial | invalid_result | interrupted | failed`, fields `precomputed_example`, `warnings`, `exportable` per data-model.md §3.7
- [X] T032 [P] [US1] Ingest tests in `tests/integration/test_ingest_synthetic.py`: normalization of the synthetic scene, refusal when files are missing, on a foreign `run_id`, on an empty model, on an incomplete result

### Stage 5 — viewing, export, report

- [X] T033 [P] [US1] Implement `src/v3d/viewer.py` on Rerun: point cloud with colors, camera poses as a toggleable layer, a run information panel, a warnings panel, a persistent status indicator and a "precomputed example" badge; save `view/session.rrd`
- [X] T034 [P] [US1] Implement `src/v3d/export.py`: write a PLY matching the displayed point cloud (same point count, coordinates, colors), `run_id` and a summary in the header comment; refuse with code 6 on `invalid_result`/`interrupted`, with `--force` — a file with the `.UNUSABLE.ply` suffix
- [X] T035 [P] [US1] Implement `src/v3d/report.py`: `report.md` with input, frames, parameters, environment, timings, metrics with explanations, unavailable metrics with reasons, scale status, the background-present flag, interpretation limits; **two separate sections: (a) verified properties of the file and frames, (b) assumptions about the scene accepted from the user and not verifiable by the system** (FR-006); available without launching the viewer
- [X] T036 [US1] Build the precomputed example in `assets/precomputed/sample_run/` (from the synthetic scene until a real result exists) with `precomputed_example: true` in all artifacts
- [X] T037 [P] [US1] Export and report tests in `tests/contract/test_export_report.py`: point count and colors equal the displayed cloud, PLY readable by a third-party parser, required report sections present, export of an unusable result is forbidden
- [X] T038 [US1] Manual level-B check per quickstart.md §3 on `assets/precomputed/sample_run`: viewing, panels, badges, export, report — **without the GPU machine**

### Stage 3 — M1, the first feasibility check (GPU)

- [X] T039 [US1] Shoot our own video following the shooting instructions (a textured static object, slow full walk-around, stable lighting, 30–60 s) and put it in `assets/videos/` (outside git)
- [X] T040 [US1] Write `docs/shooting-guide.md` — the shooting instructions shipped with v1 (A-10)
- [X] T041 [US1] **(GPU)** Started manually by the author: prepare the environment per quickstart.md §6: `nvidia-smi` (model, VRAM, free memory), clone `facebookresearch/vggt`, install requirements; record the torch/CUDA versions, the repository commit, the weights revision
- [X] T042 [US1] **(GPU)** M1 run: `python demo_colmap.py --scene_dir=<package> --use_ba`; record the actual VRAM usage (within ≤ 24 GB), execution time, the full stdout; on `CUDA out of memory` — reduce the frame budget in `prepare` and repeat the transfer
- [X] T043 [US1] Transfer the result to the Mac, run `v3d ingest/view/export/report`, evaluate the M1 success conditions from plan.md §8 and record the outcome in `docs/m1-feasibility.md` (including a negative outcome, if that is what it is)
- [X] T044 [US1] Declare the first successful run the **baseline**: save its metrics in `docs/baseline.md`; until then, do not assign time/memory/quality thresholds
- [X] T045 [US1] After M1, fix the default values of `--budget` and `min_frames` in `src/v3d/config.py` and justify them by measurement, not by assumption; remove the "provisional" mark from `min_frames` (resolves question Q3 and finding A1)

**Checkpoint**: US1 is complete — the end-to-end path works, the result opens and exports
on the Mac without a GPU; the baseline is fixed. This is the MVP.

---

## Phase 4: User Story 2 — Compare two frame selection methods (Priority: P2)

**Goal**: one bounded experiment — uniform sampling versus selection that accounts for sharpness
and redundancy, with an equal budget and one reconstructor.

**Independent Test**: two runs on one video under conditions fixed in advance produce a saved
comparison of the declared metrics; a negative result is accepted.

**Dependency**: starts only after US1 is complete (a working end-to-end path is needed).

- [X] T046 [US2] Implement `quality_nonredundant` selection in `src/v3d/select/quality.py` per the algorithm in plan.md §7: windowed scheme (`budget` windows equal in time) → sharpest in the window → check the difference threshold `τ` → permitted shortfall of frames, recorded
- [X] T047 [US2] Measure and record the **cost of the selection itself** in `diagnostics.stage_durations_s.select` and as a separate line in the report (share of total time)
- [X] T048 [P] [US2] Selection tests in `tests/unit/test_select_quality.py`: the budget is never exceeded, coverage over time is preserved (at most one frame per window), determinism with a fixed seed, correct `redundant` marking
- [X] T049 [US2] Tune the threshold `τ` and window parameters on a **separate tuning video** in `assets/videos/tuning/`; forbid tuning on the evaluation set and record this decision in `docs/p2-experiment.md`
- [X] T050 [US2] Implement fixing of conditions in `src/v3d/compare.py`: `v3d compare --init` creates `experiments/<exp_id>/conditions.json` (video, budget, reconstructor and weights, seed, `code_id`, list of declared metrics) and makes it non-editable — changing the conditions = a new `exp_id`
- [X] T051 [US2] Implement comparison in `src/v3d/compare.py`: refuse with code 9 if anything other than the frame selection method differs; output `experiments/<exp_id>/{results.json, comparison.md}` with unavailable metrics marked and a verdict `improvement | no_difference | regression | inconclusive`
- [X] T052 [P] [US2] Comparison tests in `tests/contract/test_compare.py`: refusal when the video/budget/seed/reconstructor differ, correct verdict recording, immutability of `conditions.json`
- [X] T053 [US2] **(GPU)** Started manually by the author: run both branches of the experiment on one video with the same budget and seed; transfer the results, run `ingest` for each
- [X] T054 [US2] Run the comparison and record the outcome in `docs/p2-experiment.md`, including a visual check against the six defects and an explicit verdict; **keep a negative or neutral result as a valid outcome**, do not change conditions retroactively

**Checkpoint**: US1 and US2 work independently; the experiment is reproducible from the saved conditions.

---

## Phase 5: Polish and cross-cutting tasks

**Purpose**: what concerns both stories and the honesty gates before publication.

- [ ] T055 [P] Shoot the remaining two videos of the fixed set (a more difficult shoot; a deliberately problematic case) and describe them in `docs/test-set.md` with the expected nature of the behavior
- [ ] T056 **(GPU)** Started manually by the author: run the fixed set of three videos, save artifacts and metrics for comparison between versions; **the deliberately problematic video is checked as a failure**, not as a successful reconstruction
- [ ] T057 [P] Implement the COLMAP fallback path in `docs/fallback-colmap.md`: commands `feature_extractor → sequential_matcher → mapper → model_converter`, ingest by the same `v3d ingest`, limitations; verify on the Mac with at least one video
- [ ] T058 [P] Extend `README.md` per the honesty gates: actual behavior of the code, known limitations, artifact type (point cloud, not mesh/splat), `scale_status = not_determined`, presence of residual background, NC weights license, marking of selected examples; **describe the COLMAP fallback path as a development and verification tool, not as a second supported backend of the product**
- [ ] T059 [P] Record in `docs/decisions.md` the decisions that changed direction (choice of VGGT, COLMAP sparse format as the contract, Rerun instead of our own frontend, dropping segmentation) — briefly: what was decided, why, what was rejected
- [ ] T060 [P] Add a wording check in `tests/contract/test_terminology.py`: (a) the result is nowhere called a mesh, textured model or splat (SC-009, FR-023); (b) artifacts, reports and documentation contain no claims of metric accuracy or surface completeness — prohibited phrasings ("millimeter accuracy", "surface completeness", "accuracy of N mm") without a reference to an independent ground truth (FR-038, FR-039); (c) every warning carries `observation_only: true` (FR-007)
- [ ] T061 Deliberately reproduce each case from the Edge Cases section of spec.md and make sure none ends with a "success" status; record the results in `docs/edge-cases-check.md` (SC-004)
- [ ] T062 Go through `quickstart.md` in full on a clean copy of the environment and fix discrepancies between the document and the actual behavior of the commands

---

## Phase 6: Repository hygiene and private remote (amendment 2026-09-26)

**Purpose**: keep the repository publishable: English only, no author infrastructure. Blocks
every push.

- [X] T063 Repository content in English: docs, specs, code, comments, messages, test names (FR-055)
- [X] T064 Add `tests/contract/test_repo_hygiene.py`: only Latin/Greek-script letters, no absolute home paths, no wording about how the GPU machine is accessed or shared, no private patterns from the local notes file (FR-055, SC-020)
- [X] T065 Amend the constitution to v3.1.0: English-only repository rule and infrastructure-privacy rule in "Scope and hardware constraints"
- [X] T066 Create the local run-logistics notes file, excluded through `.git/info/exclude`, including the private-pattern block used by T064
- [X] T067 Push the repository to the private remote `video-to-pointcloud`; the hygiene test (T064) must pass before every push

**Checkpoint**: the repository is on the private remote; the hygiene test runs before every push.

---

## Phase 7: User Story 3 — World alignment and visual check (Priority: P3)

**Goal**: the normalized result is re-centred on the object and shown upright, with the transform recorded as an estimate (FR-045…FR-048); the visual check can be recorded (FR-049).

**Independent Test**: synthetic tilted scene → vertical recovered within tolerance; partial arc → `not_estimated`; re-ingested pumpkin run stands upright and rotates in place in `v3d view`; export still equals the shown cloud.

- [X] T068 [P] [US3] Extend `assets/synthetic/make_scene.py` with scene tilt, camera height jitter, per-camera roll and partial-arc options, recording the true vertical in `expected.json`
- [X] T069 [US3] Implement in `src/v3d/geometry.py`: plane fit to camera centres, sign from mean camera up vector, quality metrics (planarity, arc coverage, sign agreement, number of cameras), optical-axes nearest point, and the rigid alignment transform with the `first_camera_in_front` yaw rule; apply a rigid transform to poses and points
- [X] T070 [P] [US3] Tests in `tests/unit/test_alignment.py`: tilt recovered within tolerance under height jitter and roll (SC-015); partial arc and non-planar path rejected (SC-016); transform is rigid and invertible; sign correct; centre robust to background points
- [X] T071 [US3] Add preliminary gate thresholds to `src/v3d/config.py` and warning code `orientation_not_estimated` to `src/v3d/errors.py` / `src/v3d/warnings_.py`
- [X] T072 [US3] Apply alignment in `src/v3d/ingest.py` before writing `normalized/`; write the `world_alignment` block and updated `conventions` into `cameras.json` (data-model §3.4a); keep `result/sparse/` untouched
- [X] T073 [US3] Switch `src/v3d/viewer.py` to a Y-up world when `world_alignment.status = estimated`; show the alignment note in the info panel; show alignment in `src/v3d/report.py`
- [X] T074 [US3] Add `v3d visual-check <run> <item> ok|defect|not_checked [--note]` in `src/v3d/cli.py` writing `checked_at` into `diagnostics.json` (FR-049, SC-018), with tests
- [X] T075 [US3] Integration test `tests/integration/test_ingest_alignment.py`: aligned ingest of a tilted synthetic scene; export equals shown cloud after alignment (SC-017)
- [X] T076 [US3] Re-ingest the local pumpkin runs and regenerate `assets/precomputed/sample_run`; confirm upright orientation visually and record the result with `v3d visual-check`

**Checkpoint**: aligned results everywhere; visual check is recordable.

---

## Phase 8: User Story 3 — Media rendering (Priority: P3)

**Goal**: reproducible media from stored run artifacts, without a GPU (FR-050…FR-052).

- [X] T077 [US3] Implement `src/v3d/render.py`: perspective point splatting with depth sort (numpy + Pillow), fixed views, point size and background; caption with `run_id` and source
- [X] T078 [US3] Turntable animation (frames → WebP and MP4 via ffmpeg) and a video-frame-versus-render pair from the same camera pose using stored intrinsics
- [X] T079 [P] [US3] Camera-trajectory figure and deterministic web decimation of `normalized/points.ply` labelled "N of M points" (FR-052)
- [X] T080 [US3] Add `v3d media <run>` in `src/v3d/cli.py` writing to `runs/<id>/media/` with a `media.json` index (what each file shows, source, `run_id`)
- [X] T081 [P] [US3] Tests `tests/unit/test_render.py`: deterministic output, captions present, decimation count and labelling, render from a synthetic camera pose projects the scene centre to the principal point

**Checkpoint**: `v3d media` produces all assets for the README and the page.

---

## Phase 9: User Story 3 — Report, README and project page (Priority: P3)

**Goal**: a reader understands the work in a minute and can dig into the details on click (FR-053, FR-054). Built after P2 so the content is written once.

- [X] T082 [US3] Write the detailed project report in `docs/report/` (M1 → alignment → P2 → decisions → measurements → failures), linking stored artifacts
- [X] T083 [US3] SVG charts for the report and the page (memory versus frames, confidence threshold versus points, P2 comparison) with a consistent palette
- [X] T084 [US3] Scaffold `site/` with Astro + Tailwind; static output to `site/dist/` (gitignored); no analytics
- [X] T085 [US3] Landing page: input video frames next to renders, turntable, key numbers, links; detailed report page rendered from `docs/report/`
- [X] T086 [US3] Interactive 3D island (three.js `PLYLoader` + `OrbitControls`) loading the decimated PLY only on click, with the "N of M points" label (SC-019)
- [X] T087 [US3] Reader-facing `README.md`: what, how, results with visuals, honest limitations, links to the report and the page (supersedes T058)
- [X] T088 [US3] Verify: `npm run build`, serve `site/dist/` with a static file server, 3D not downloaded until requested, no third-party requests, hygiene test green

**Checkpoint**: the project is presentable; publication happens only on explicit request.

---

## Dependencies & Execution Order

### Order after the 2026-09-26 amendment

Phase 6 → Phase 7 (alignment) → Phase 8 (media) → Phase 4 (P2, T046–T054) → Phase 9 (report,
README, page) → Phase 5 (polish: test set, COLMAP fallback, decisions log, terminology check,
edge-case sweep, quickstart walk-through). Alignment comes first because every image depends on orientation; media before P2
so the experiment can be inspected visually; the report and page last so they are written once.
T058 is superseded by T087; T060 is partly covered by T064 and stays for the terminology check.

### Phase dependencies

- **Setup (Phase 1)**: no dependencies
- **Foundational (Phase 2)**: after Setup; **blocks both stories**
- **US1 (Phase 3)**: after Foundational
- **US2 (Phase 4)**: **after US1** — the experiment is meaningless without a working end-to-end path
  (spec.md, "Why this priority" for US2)
- **Polish (Phase 5)**: after US1; completeness — after US2

### Within US1

```text
T017 → T018 → T019/T020 → T021/T022 → T024 → T025          (preparation)
T027 → T028 → T029 → T030 → T031                           (ingest)
T033/T034/T035 → T036 → T038                               (viewing/export/report)
T039/T040 → T041 → T042 → T043 → T044 → T045               (M1, GPU)
```

Viewing, export and report (T033–T038) can be done **before** a real result exists —
on the synthetic and precomputed example. This is a direct consequence of Principle IV: progress
is not blocked by GPU availability.

### Parallel opportunities

- Setup: T003, T004, T005 are parallel
- Foundational: T010–T016 are parallel to each other after T007–T009
- US1: T017 and T019 are parallel; T033, T034, T035 are parallel; tests T023, T026, T032, T037 are parallel to their modules
- US2: T048 and T052 are parallel to the implementation
- Polish: T055, T057, T058, T059, T060 are parallel

---

## Parallel Example: User Story 1

```bash
# Three independent modules at the viewing/export/report level:
Task: "Implement src/v3d/viewer.py on Rerun"
Task: "Implement src/v3d/export.py (PLY)"
Task: "Implement src/v3d/report.py (report.md)"

# Tests that do not conflict on files:
Task: "tests/unit/test_select_uniform.py and tests/unit/test_metrics.py"
Task: "tests/contract/test_package.py"
Task: "tests/integration/test_ingest_synthetic.py"
```

---

## Implementation Strategy

### MVP (US1 only)

1. Phase 1 → Phase 2 → Phase 3.
2. **Stop and verify**: quickstart.md §§2–4, 7 on the Mac, then M1 on the GPU machine
   (started manually by the author).
3. The first successful run = baseline. Before it, no thresholds are assigned.

### Incremental delivery

1. Setup + Foundational → foundation.
2. + US1 → the end-to-end path works → MVP.
3. + US2 → a frame selection experiment, any correctly measured outcome.
4. + Polish → fixed set, fallback path, honesty gates, README.

### Cost limit for M1

No more than **three** GPU runs and one session. When exhausted — the COLMAP fallback path (T057),
not debugging someone else's model (plan.md §8).

---

## Notes

- `[P]` = different files, no unfinished dependencies.
- A single developer: `[P]` means "can be reordered", not "do simultaneously".
- Commit after each task or logical group.
- **(GPU)** tasks are started manually by the author; memory budget ≤ 24 GB.
- Prohibited: assigning time/memory/quality thresholds before the baseline; replacing an unavailable
  metric with zero; calling the result a mesh or splat; claiming metric accuracy or surface
  completeness without an independent ground truth; removing a failing test from the suite; changing
  the conditions of experiment P2 retroactively; describing the COLMAP fallback path as a second
  supported backend of the product.
- Terms are uniform per the README glossary (T005): "transfer package" = `package/`, "run" =
  `runs/<run_id>/`, "normalized result" = `normalized/`.
