# Feature Specification: Video-to-3D MVP

**Feature Branch**: `001-video-3d-mvp` (the repository is not initialized under git — no branch was created)

**Created**: 2026-09-18

**Status**: Draft

**Input**: User description: "First complete version of the personal Video-to-3D project: an ordinary RGB video of a static object → a colored point cloud with camera poses, interactive viewing on a Mac, export to PLY, metadata and a run report. The heavy stage runs separately on an intermittently available GPU machine. Second scenario — a limited comparison of two frame selection methods."

## Overview

The tool takes a phone-recorded RGB video of a single static object and gives the
user (the project author) a reconstructed 3D representation that can be inspected,
exported, and evaluated by honest metrics together with a description of its limitations.

**The v1 result is fixed**:

1. a colored point cloud of the object (with acceptable residual background);
2. camera poses for the viewpoints used;
3. interactive viewing of the result on a Mac;
4. export of a plain colored point cloud to PLY;
5. accompanying metadata and a run report.

**Feasibility status (feasibility gate, Constitution item 1 of the "Workflow" section)**:
the listed result is a chosen implementation target, NOT a confirmed capability.
Achievability on the author's own video **must be verified by the first end-to-end experiment**.
Until the first successful end-to-end run, the project does not publicly claim this result
as achieved.

**Out of scope for this feature**: mesh, textures, Gaussian Splatting, a physically usable
(metrically calibrated) model. Their absence is a deliberate decision, not a default.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Obtain and inspect a reconstruction (Priority: P1)

The user has shot a video of an object following the shooting guide. They point the tool
on the Mac at the video file and receive a conclusion on input suitability: which verifiable
properties of the file and frames are satisfied, which raise doubts. Then the tool prepares
data for the heavy computation: it selects frames and puts the transfer package (frames +
parameters + input information) into a single directory.

The user manually moves the transfer package to the GPU machine, runs the heavy stage
there with a separate command, and manually moves the result back to the Mac. The transfer
procedure is described in the documentation; remote control, a scheduler, and SSH integration
are not required.

On the Mac the user opens the result: rotates and zooms the point cloud, sees the camera poses
of the viewpoints used, reads the run information (input, frames, parameters, environment,
time, metrics) and warnings. The result can be exported to PLY and reopened
later — without a GPU and without recomputation.

**Why this priority**: this is the declared v1 product. Without an end-to-end path from video
to an openable result, the remaining work is meaningless.

**Independent Test**: fully verified by a run of one suitable video from the test set:
preparation on the Mac → heavy stage on the GPU machine → transfer → opening, viewing,
and export on the Mac. Value — a working end-to-end tool.

**Acceptance Scenarios**:

1. **Given** a supported video of a static object that follows the shooting guide,
   **When** the user starts preparation on the Mac, **Then** the system reports the number of
   source and selected frames and the result of the verifiable file and frame properties, and creates
   a prepared-data directory suitable for transfer.
2. **Given** the prepared directory has been moved to the GPU machine, **When** the user
   starts the heavy stage, **Then** on completion the point cloud, camera
   poses, metadata, and run report are written to disk, and the process reports the final status (success /
   partial result / error).
3. **Given** the result directory has been moved to the Mac, **When** the user opens it
   in the viewer, **Then** they can rotate and zoom the cloud, see the camera poses,
   the run information, and all warnings of this run.
4. **Given** an open result, **When** the user performs an export, **Then** a PLY
   file with the colored point cloud is created, whose content matches the displayed cloud
   (the same number of points, the same coordinates and colors).
5. **Given** a previously exported/saved result, **When** the user opens
   it on the Mac some time later without access to the GPU machine, **Then** viewing and reading of the
   information work without recomputation.
6. **Given** the result of any run, **When** the user reads the run report,
   **Then** it states the scale status (confirmed by measurement / estimated by the model /
   not determined), the coordinate system and units, and the residual-background indicator.

---

### User Story 2 - Compare two frame selection methods (Priority: P2)

After P1 works, the user runs one limited experiment: compares
uniform frame sampling (baseline) with one selection variant that accounts for frame quality
(sharpness) and their redundancy (similarity of neighboring viewpoints).

The comparison is run on the same videos, with the same reconstructor, under pre-set
conditions and the same frame budget. The results of both methods are stored
side by side and compared by pre-declared metrics.

**Why this priority**: this is the author's educational engineering contribution on top of ready-made components, but
it is meaningless without a working P1. A positive improvement is not guaranteed; a correctly
measured and honestly described negative result is a valid and complete outcome.

**Independent Test**: verified separately from changes to viewing and export — two runs
on the same video from the test set with the same frame budget and a stored
comparison of metrics.

**Acceptance Scenarios**:

1. **Given** a video from the test set and pre-fixed comparison conditions (frame
   budget, parameters, code version, seed), **When** the user performs a run with the method
   "uniform sampling" and a run with the method "selection by quality and redundancy", **Then**
   both runs use the same frame budget and the same reconstructor, and their
   artifacts and information are stored separately and in full.
2. **Given** two completed runs on one video, **When** the user requests
   the comparison, **Then** they get a side-by-side of the pre-declared metrics for both
   methods with an explicit marking of inapplicable or unavailable metrics and the reason.
3. **Given** the comparison shows a degradation or no difference, **When** the user
   records the outcome, **Then** the result is stored and described as negative/neutral
   without changing the comparison conditions retroactively.
4. **Given** fixed comparison conditions, **When** the comparison is repeated on the same
   code version and the same inputs, **Then** frame selection yields the same set of frames, and discrepancies
   in the other metrics are explained by explicitly named nondeterminism.

---

### User Story 3 - Present the work to a reader (Priority: P3)

A reader (a reviewer of the author's portfolio) opens the repository or the project page and,
within a minute, understands what was built: a phone video goes in, a colored point cloud with
camera poses comes out, and the steps in between are the author's own work. A light landing page
shows the input video frames next to renders of the result, a turntable animation and a few
charts. One click opens the detailed project report with every step, measurement, decision and
failure. Another click loads an interactive 3D view of the point cloud that the reader can rotate.

Before any of this, the result is brought into a frame a human expects: the object stands
upright and rotates in place. Because the reconstruction carries no gravity information, "up"
is estimated from the camera trajectory, and this is stated wherever it matters.

**Why this priority**: the project exists for learning and for a portfolio; without a readable
presentation the engineering contribution stays invisible. It depends on P1 (something to show)
and benefits from P2 (a result of the author's own experiment), hence P3.

**Independent Test**: built from the stored artifacts of an existing run, with no GPU: generate
the media, build the static site, open it from disk, and check that every image names its source
run and that the 3D view loads only on demand.

**Acceptance Scenarios**:

1. **Given** a normalized result of a handheld orbit capture, **When** it is ingested, **Then**
   the point cloud is re-centred on the object and its vertical axis is estimated from the camera
   ring; the applied transform, its quality numbers and the label "estimated from the camera
   trajectory, not measured" are recorded, and the original frame is recoverable.
2. **Given** a capture whose camera trajectory does not allow a reliable vertical estimate (partial
   arc, non-planar path), **When** it is ingested, **Then** only re-centring is applied, the
   vertical is marked `not_estimated` with the reason, and a warning is shown.
3. **Given** a run, **When** the author records a visual check item, **Then** it is stored with
   the time of the check and appears in the report.
4. **Given** stored run artifacts, **When** the media command is run, **Then** renders, a
   turntable animation, and a video-frame-versus-render pair are produced, each captioned with
   the `run_id` and its source.
5. **Given** the built site, **When** it is opened as local files without a server, **Then** the
   landing page shows the visuals, the report opens on click, and the interactive 3D view loads
   only when requested and states how many of the points it shows.
6. **Given** the repository, **When** the hygiene check runs, **Then** it finds only English
   text and no details of the author's infrastructure.

---

### Edge Cases

For each case the expected behavior is stated. A warning about a possible problem
**must not** pose as a reliable automatic diagnosis of the cause: the system reports an
observation and its possible consequences, and does not establish the cause.

- **Corrupted or unsupported file**: the input is rejected before preparation starts;
  the message names what exactly failed (file cannot be read / container or codec outside the
  supported list / no video track) and lists the supported formats.
  No prepared directory is created.
- **The video is readable, but there are too few usable frames** (short video, almost all frames
  rejected): preparation ends with the status "insufficient data", reports the number of
  source frames, the number that passed selection, and the minimum required number; the prepared
  directory is either not created or is marked as unfit for a run.
- **Noticeable blur**: frames with low sharpness are rejected; if the share rejected for
  this reason is large, the user gets the warning "the video contains many blurry frames,
  the result may be incomplete" — as an observation, without asserting a cause.
- **Nearly identical viewpoints** (the camera barely moved): redundant frames are not
  duplicated in the sample; if viewpoint diversity is low, the warning
  "viewpoints differ little, reconstruction may fail" is issued. The run is not blocked
  if the formal conditions are met.
- **GPU machine unavailable or insufficient resources when starting the heavy stage**: the stage
  ends with an error with a clear message and preserves what has already been written to disk;
  the prepared directory remains usable for a later re-run. Work on the Mac
  (viewing previously obtained results, preparing new data) is not blocked.
- **Interrupted computation** (stopped by the user, process crash, machine going offline):
  the result directory is marked as incomplete; the viewer opens it only with an explicit
  warning "result is incomplete" or refuses to open it, stating the reason.
  An incomplete result is not passed off as successful.
- **Empty or invalid result** (reconstruction finished, but there are no camera poses,
  no points or a negligible number of them, coordinates are invalid): the run status is
  "reconstruction failed", not "success"; the report is saved with stage diagnostics;
  export of an empty cloud is not passed off as a valid result.
- **Partial result** (only a part of the viewpoints was registered or part of the
  object was reconstructed): status "partial result", the report states what share of frames
  was used; viewing is allowed with a permanently visible "partial" mark.
- **Missing or incompatible result files** (not the whole directory was transferred, files
  from different runs, a result from a different format version): the viewer refuses to
  open it, stating which file is missing or which run identifiers do not
  match. Mixing artifacts of different runs is prohibited.
- **Object not separated from the background**: residual background is acceptable; the report and the viewer
  state that the result contains background and that automatic segmentation is not guaranteed.
- **Precomputed examples**: saved results used for interface development
  are always marked as precomputed — both in the viewer and in the report.

## Requirements *(mandatory)*

### Functional requirements — input and validation (FR-0xx)

- **FR-001**: The system MUST accept one video file as the scenario input and treat it
  as the unit of processing (one run = one video).
- **FR-002**: The system MUST explicitly define and document the list of supported
  containers and codecs and reject input outside that list. Support for "any video"
  MUST NOT be claimed.
- **FR-003**: The system MUST support typical phone video: handheld shooting, portrait
  or landscape orientation, variable sharpness.
- **FR-004**: The system MUST correctly account for frame orientation metadata so that frames
  are fed to processing in the correct rotation.
- **FR-005**: The system MUST use the timestamps/frame order of the source video and
  store for each selected frame its position in the time of the source video.
- **FR-006**: The system MUST separate in the report (a) verifiable properties of the file and frames
  (readability, container/codec, duration, resolution, fps, frame count, sharpness,
  difference between neighboring frames) and (b) assumptions about scene content (object is static,
  rigid, opaque, textured, lighting is stable) which the system takes
  on trust from the user and does NOT verify automatically.
- **FR-007**: The system MUST NOT claim that a violation of shooting conditions has been detected if
  only an indirect observation was checked; warning wording MUST remain
  observations ("few distinct viewpoints"), not diagnoses of the cause.

### Functional requirements — preparation and frame selection

- **FR-008**: The system MUST extract frames from the video and select a subset of them for
  reconstruction within a pre-set frame budget.
- **FR-009**: The system MUST support at least two selection methods: uniform sampling
  (baseline) and selection that accounts for frame quality and redundancy of neighboring viewpoints.
- **FR-010**: The system MUST record for each run: the number of source frames, the number of
  selected frames, the selection method applied and its parameters, and the aggregated
  rejection reasons.
- **FR-011**: Frame selection MUST be deterministic for identical inputs, parameters,
  and seed.
- **FR-012**: The system MUST put the preparation result into a single self-contained directory
  suitable for manual transfer to another machine.

### Functional requirements — separation of stages and execution

- **FR-013**: The system MUST split processing into a light stage (input validation, preparation
  and frame selection, viewing, export, reports), executed on the Mac without an external GPU machine,
  and a heavy stage (reconstruction), started by a separate command.
- **FR-014**: The heavy stage MUST take all inputs from the prepared-data directory and
  write all outputs to the result directory, with no dependence on the state of the
  light stage process.
- **FR-015**: The system MUST NOT require automated or network access to the GPU machine;
  manual transfer of directories MUST be the standard and documented way of exchange.
- **FR-016**: The heavy stage MUST end with one of the explicit statuses: success, partial
  result, error, interrupted — and record this status in the report.

### Functional requirements — reconstruction result

- **FR-017**: On successful completion the system MUST produce a colored point cloud of the scene
  in which the object is distinguishable.
- **FR-018**: The system MUST produce camera poses for the viewpoints used, tied
  to the same coordinates as the point cloud.
- **FR-019**: The system MUST record for each camera pose which selected frame
  it corresponds to.
- **FR-020**: The system MUST allow residual background in the result and MUST explicitly indicate its
  presence. Perfect automatic segmentation of the object is NOT a v1 requirement.
- **FR-021**: The system MUST state in the result metadata the actual artifact type
  (point cloud), the coordinate system, units, and scale status (confirmed by measurement /
  estimated by the model / not determined).
- **FR-022**: The system MUST NOT pass off as observed geometry points obtained by
  completing invisible surfaces; if such completion is applied, it MUST be
  marked. In v1 generative completion is not applied.
- **FR-023**: The system MUST NOT produce a mesh, a textured model, or a splat within
  this feature, and MUST NOT call the point cloud by any of these terms.

### Functional requirements — viewing

- **FR-024**: The user MUST be able to open the result directory on the Mac and
  interactively rotate and zoom in/out the point cloud.
- **FR-025**: The viewer MUST show the camera poses of the viewpoints used together
  with the cloud and allow hiding/showing them.
- **FR-026**: The viewer MUST show the run information (input, frames, parameters,
  environment, time, available metrics) and all warnings of the run.
- **FR-027**: The viewer MUST work without access to the GPU machine and without recomputation,
  reading only saved artifacts.
- **FR-028**: The viewer MUST permanently display the result status (success / partial /
  incomplete) and the "precomputed example" mark if it is present in the metadata.
- **FR-029**: The viewer MUST refuse to open a set of files with mismatched
  run identifiers or with missing required files, naming the reason.

### Functional requirements — export

- **FR-030**: The user MUST be able to export the colored point cloud
  to PLY format.
- **FR-031**: The exported file MUST match the displayed cloud: the same number
  of points, the same coordinates and colors (to the precision defined by the storage format).
- **FR-032**: Export MUST be accompanied by saving or referencing the run metadata,
  so that the exported file cannot be confused with the result of another run.
- **FR-033**: Export of an empty or invalid result MUST be either
  prohibited or performed only with an explicit "result unusable" mark.

### Functional requirements — reproducibility and reporting

- **FR-034**: Each run MUST store information sufficient to repeat it:
  identification of the input video (name, hash, duration, resolution, fps), the list of
  frames used, all stage parameters, seed, identification of the model/method applied
  and weights (if applicable), code version, versions of key dependencies, execution environment
  (Mac / GPU machine), start time and stage durations, the list of output artifacts
  with their sizes.
- **FR-035**: The run report MUST be human-readable and available both from the viewer
  and separately, without launching the viewer.
- **FR-036**: Unavailable or inapplicable metrics MUST be marked as unavailable
  with a reason, and MUST NOT be replaced by zero or an invented value.
- **FR-037**: The system MUST record and show the available error metrics together
  with an explanation of their meaning and limits of applicability.
- **FR-038**: The system MUST NOT present the number of points, model confidence, or error metrics
  as proof of geometric accuracy or surface completeness.
- **FR-039**: The system MUST NOT claim millimeter accuracy or measured surface
  completeness in the absence of an independent reference and a described comparison protocol.

### Functional requirements — frame selection experiment (P2)

- **FR-040**: The system MUST allow two runs on the same video
  that differ only in the frame selection method, with the same frame budget and the same
  reconstructor.
- **FR-041**: The comparison conditions (video, frame budget, parameters, code version, seed, the set
  of metrics being compared) MUST be fixed before the runs and stored together with
  the results.
- **FR-042**: The system MUST produce a side-by-side of the pre-declared metrics of the two
  methods, with inapplicable metrics marked.
- **FR-043**: A negative or neutral comparison result MUST be stored and
  described as such; changing the comparison conditions retroactively MUST NOT be applied.
- **FR-044**: The experiment MUST remain a limited comparison of two selection methods
  and MUST NOT expand into model training or a large benchmark.

### Functional requirements — world alignment (US3)

- **FR-045**: The system MUST re-centre the normalized result on the object and MUST estimate a
  vertical axis from the camera trajectory, so that the object is shown upright and rotates in
  place. The reconstruction contains no gravity measurement; the estimate MUST NOT be presented
  as a measurement.
- **FR-046**: The applied transform MUST be recorded with the result: method, the transform from
  the reconstructor's frame to the aligned frame, quality numbers of the estimate, a status
  (`estimated` / `not_estimated`) and an explanation. The original frame MUST be recoverable, and
  the raw reconstructor output MUST remain unchanged.
- **FR-047**: If the camera trajectory does not support a reliable estimate (quality gate not
  passed), the system MUST apply re-centring only, mark the vertical as `not_estimated` with the
  reason, and emit an observation warning. Gate thresholds are preliminary until measured.
- **FR-048**: Rotation about the estimated vertical is arbitrary; the system MUST fix it by a
  deterministic, documented rule and MUST declare that it is arbitrary. The viewer and the export
  MUST use the same aligned frame (FR-031 continues to hold).
- **FR-049**: The author MUST be able to record the result of each visual check item
  (`ok` / `defect` / `not_checked` with an optional note); the record MUST include the time of
  the check and MUST appear in the report.

### Functional requirements — presentation (US3)

- **FR-050**: Media for the presentation (renders, turntable animation, video-frame-versus-render
  pairs, camera trajectory, charts) MUST be produced by a command from stored run artifacts,
  without a GPU and without re-running the reconstruction.
- **FR-051**: Every published image, animation and 3D asset MUST name its source run (`run_id`)
  and what it shows. Synthetic and precomputed examples MUST be marked as such; a hand-picked
  example MUST be labelled as selected (Principle I).
- **FR-052**: A point cloud decimated for the web MUST be labelled with the number of points
  shown out of the total, and MUST be derived deterministically from the normalized result.
- **FR-053**: The project MUST provide a reader-facing README with visuals and links to the
  detailed project report and to the project page.
- **FR-054**: The project page MUST build into static files that open without a server: a light
  landing page with visuals, a detailed report opened on click, and an interactive 3D view loaded
  only on demand. It MUST NOT include analytics or third-party tracking.
- **FR-055**: Repository and published materials MUST be in English and MUST NOT contain details
  of the author's infrastructure (host names, user names, absolute home paths, how the GPU
  machine is accessed or shared). This MUST be checked automatically.

### Key Entities

- **Input video**: the source file. Attributes: name, hash, container/codec, duration,
  resolution, fps, frame count, orientation.
- **Frame**: an extracted image. Attributes: position in the time of the source video, index,
  quality metrics (sharpness), selection flag and rejection reason.
- **Transfer package (package for the heavy stage)**: a self-contained directory. Contains the
  selected frames, input information, selection parameters, seed, run identifier.
- **Run**: the unit of work from preparation to result. Attributes: identifier, status
  (prepared / in progress / success / partial / error / interrupted), environment, stage
  times, warnings.
- **Point cloud**: colored points of the scene. Attributes: number of points, coordinate system, units,
  scale status, background-presence flag.
- **Camera pose**: position and orientation of the camera for one used frame. Tied
  to a specific selected frame.
- **Run report**: a human-readable summary of the run information, metrics, and warnings.
- **Selection comparison (P2)**: a pair of runs on one video with fixed conditions
  and a side-by-side of metrics.
- **Test video set**: a fixed small set of inputs with expected behavior
  (see "Test set").

## Observable metrics and evaluation method

Metrics are fixed here, before implementation, and MUST NOT be chosen after the fact.

- **Scenario completion**: the end-to-end path reached an openable result (yes/no),
  with the stopping stage stated on failure.
- **Output correctness**: presence of all required artifacts, consistency of
  run identifiers, readability of the exported PLY by a third-party viewer.
- **Frames**: number of source frames, number of selected frames, distribution of rejection reasons.
- **Camera poses**: number of viewpoints used, share of selected frames that received a pose.
- **Points**: number of valid points in the result (with an explicit definition of "valid point").
- **Time and volume**: duration of each stage, total time, artifact size.
- **Error metrics**: available metrics (for example, reprojection error — if the method
  provides them), each with an explanation of meaning and limits of applicability; unavailable ones
  marked as unavailable.
- **Visual check** of pre-described defects from a fixed list:
  1. doubling/duplication of the object's geometry;
  2. a break or jump in the camera pose trajectory;
  3. "point soup" — no recognizable shape of the object;
  4. gaps in weakly textured areas of the object;
  5. background dominating the object;
  6. gross color inconsistency between viewpoints.
  The result of the visual check is recorded as an observation and MUST remain separate
  from quantitative conclusions.

**Interpretation limits**: the number of points and confidence are not proof of accuracy.
Without an independent reference, claims of metric accuracy and surface completeness
are not made.

## Test set

A fixed set of three videos. **Status: planned — not yet shot at the time of writing
the specification. No trials have been run.**

1. **Good shooting**: a textured object, a slow full walk-around, stable
   lighting, minimal blur. Expectation: a successful result.
2. **More difficult shooting**: faster movement, some frames blurred, a less even
   walk-around. Expectation: success or a partial result with warnings.
3. **Deliberately problematic case**: a weakly textured/glossy object or a nearly
   stationary camera. Expectation: **failure or a clearly failed reconstruction**. This case
   is verified as a failure and is NOT considered a failed check; removing it from the set is prohibited.

The first successful end-to-end run establishes the metrics baseline; target levels are not
assigned before that.

## Out of scope for this feature

Not included and not implemented: mesh and its generation, textures and UV, Gaussian Splatting,
a metrically calibrated ("physically usable") model, generative completion of invisible
surfaces, accounts and authorization, a cloud service, a mobile app, a 3D model
editor, a robot and its control, physical simulation, model training, a large benchmark,
real time, multi-user mode, remote control of the GPU machine, a job scheduler,
and SSH integration, automatic segmentation of the object from the background as a guarantee.

The static project page (US3) is documentation, not a cloud service: it has no backend, no
accounts, no uploads and no analytics. Estimating "up" from the camera trajectory (FR-045) is a
presentation aid, not a metric calibration: scale stays `not_determined`.

The choice of specific models, libraries, architecture, metadata storage formats, and file
schemas belongs to the plan stage and is not defined here.

## Success Criteria *(mandatory)*

The criteria are split into two groups. Group A is verifiable immediately after implementation. Group B is
a measurement mechanism whose readiness is verifiable, whereas the **target level of
performance and quality is NOT assigned until a baseline is obtained** on the first successful
end-to-end run.

### Group A — verifiable results

- **SC-001**: On the "good shooting" video from the test set the user goes through the path
  "video → preparation → heavy stage → transfer → viewing" and gets an openable result
  without manually editing files between stages.
- **SC-002**: A previously obtained result directory opens on the Mac with the GPU machine
  completely unavailable, shows the cloud, camera poses, and run information,
  without recomputation.
- **SC-003**: The exported PLY opens in a third-party point cloud viewer
  and contains the same number of points as shown in the viewer and stated in the report.
- **SC-004**: Each case listed in the Edge Cases section is reproduced deliberately
  and leads to the behavior and message described there; none of them ends with the status
  "success".
- **SC-005**: From the saved run information another person (or the author some time later)
  can reproduce the same run: all fields from FR-034 are present and filled in or
  explicitly marked as unavailable with a reason.
- **SC-006**: The report of each successful run contains the artifact type, coordinate system,
  units, scale status, and the residual-background indicator.
- **SC-007**: Re-running preparation with the same inputs, parameters, and seed yields the same
  set of selected frames.
- **SC-008** (P2): The comparison of two selection methods on one video is performed under pre-fixed
  conditions, yields a side-by-side of the declared metrics, and is stored;
  a negative result is accepted as a valid outcome.
- **SC-009**: No project document, screen, or file name calls the result a mesh,
  a textured model, or a splat.
- **SC-010**: The deliberately problematic video from the test set does not produce a result with status
  "success".

### Group B — readiness of the measurement mechanism (thresholds not assigned)

- **SC-011**: For each run the following are recorded and available: the number of source and selected frames,
  the number of camera poses, the number of valid points, stage times, artifact size. Threshold values
  are not set before the baseline.
- **SC-012**: For each run the available error metrics are recorded together with an explanation of
  their meaning; inapplicable ones are marked with a reason. Target values are not set before the baseline.
- **SC-013**: The visual check against the fixed defect list is performed and
  recorded for each run of the test set. The acceptable defect level
  is not set before the baseline.
- **SC-014**: The test set results are stored so that comparison between versions
  is possible; the first successful run is declared the baseline.

### Group C — presentation and alignment (US3)

- **SC-015**: On a synthetic scene tilted by a known angle, with height jitter and per-camera
  roll, the estimated vertical differs from the true one by no more than a tolerance fixed in
  the tests; on a real capture the aligned result shows the object upright (visual check).
- **SC-016**: On a synthetic partial arc or a non-planar trajectory, the estimate is rejected
  (`not_estimated`) and only re-centring is applied.
- **SC-017**: After alignment the export still contains exactly the shown cloud (FR-031).
- **SC-018**: A visual check item written by the command is visible in `diagnostics.json` and
  in the report.
- **SC-019**: The built project page opens from local files without a server; the 3D view is
  not downloaded until requested.
- **SC-020**: The repository hygiene check reports zero lines outside the Latin and Greek
  scripts and zero infrastructure details.

## Assumptions

Accepted assumptions (may be refined at the plan stage or by a separate specification amendment):

- **A-01**: The only v1 user is the project author. They are familiar with the shooting guide
  and deliberately shoot suitable video. Training a casual user is not required.
- **A-02**: Supported input is limited to the typical output of a phone camera:
  MP4 and MOV containers, H.264 and HEVC video codecs, progressive scan. The list
  is **fixed by the plan** (plan.md §2, contracts/cli.md) and cannot be changed without an amendment
  to the specification. Actual verification of decoding for each item on the list is performed
  during implementation by an automated test on real files; until then the list is considered declared
  but unverified. The promise of "any video" is excluded.
- **A-03**: The audio track is ignored. Phone sensor data (IMU, gyroscope) is not used
  in v1.
- **A-04**: One video = one object = one run. Combining several videos
  into one reconstruction is not supported.
- **A-05**: Transfer of directories between the Mac and the GPU machine is done by the user manually
  (external drive, shared folder, any convenient method). The transfer method is not part of the
  product, but is documented.
- **A-06**: The characteristics of the GPU machine (model, VRAM, OS, environment) are unknown and are not
  fixed here; requirements for them are determined at the plan stage after verification.
- **A-07**: Precomputed results are used for interface development; they are
  always marked with the corresponding flag in the metadata.
- **A-08**: There is no independent geometry reference (a measured object, a scanner) in v1,
  so metric accuracy is neither claimed nor measured.
- **A-09**: The v1 interface may be minimal (including a command line for processing
  and a separate viewer); the specific form of the interface is determined at the plan stage.
- **A-10**: The shooting guide is part of the v1 deliverable (a document), but its execution is verified
  by a human, not by the system.
- **A-11**: The frame budget is set by a configuration parameter; the specific default value
  is determined at the plan stage and fixed before the P2 experiment.
- **A-12**: Reconstruction in v1 uses a ready-made external method/model; per Principle VI
  the borrowing is documented, and the author's own contribution (preparation, frame selection, diagnostics,
  viewing, evaluation, error handling) is described separately.
- **A-13**: Captures are handheld: the orbit is uneven in height and the horizon is not
  levelled. The vertical estimate therefore relies on camera positions (the orbit plane), not on
  per-frame camera orientation, which only chooses the sign.
- **A-14**: Nothing is published publicly without an explicit request from the author.
- **A-15**: Run logistics for the GPU machine are kept in local notes excluded from version
  control.
