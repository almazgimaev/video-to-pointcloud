<!--
Sync Impact Report
- Version: 3.0.0 → 3.1.0
- Change type: MINOR (two new rules added to "Scope and hardware constraints")
- Reason: the repository is intended to be published as a portfolio piece; it must be readable by
  an international audience and must not expose how or where the heavy computation is run.
- Modified principles:
  - IV. Separation of the light and heavy stages — access rule for the GPU machine clarified:
    manual start by the author or on the author's explicit request
- Modified sections:
  - Scope and hardware constraints — added the English-language rule and the
    infrastructure-privacy rule
- Added sections: none
- Removed sections: none
- Consequences: an automated hygiene test guards both rules; run logistics live in local notes
  excluded from version control.
- Deferred TODOs: none
- Template check: plan/spec/tasks templates read the constitution at runtime; no edits needed
-->

<!--
Previous report (3.0.0):

Sync Impact Report
- Version: 2.0.0 → 3.0.0
- Change type: MAJOR (incompatible redefinition of the launch rule in Principle II:
  previously an end-to-end launch with one command was required unconditionally, now this
  requirement applies to each stage, and to the end-to-end path only where it is feasible on one machine)
- Reason: Principle II required "launching the pipeline with one command", Principle IV requires
  splitting the light and heavy stages across different machines. With an intermittently available
  GPU machine and manual file transfer, both requirements cannot be met at once. The conflict was
  found by /speckit-analyze on feature 001-video-3d-mvp (finding D1, CRITICAL) and is resolved here,
  not worked around in the plan
- Changed principles:
  - II. Run reproducibility — the launch rule is reworded; added a requirement to
    document the cross-machine handoff as a standard pipeline step
- Changed sections: none
- Added sections: none
- Removed sections: none
- Scope and supported scene types unchanged
- Consequences: a pipeline broken up by a manual transfer between machines is allowed, provided
  that each stage is launched with one command and the handoff is described (what is transferred,
  how integrity is checked, which command continues the work); hidden and undocumented manual
  steps remain prohibited
- Deferred TODOs: none
- Template check: the plan/spec/tasks templates read the constitution at runtime and need no edits;
  plan.md and tasks.md of feature 001-video-3d-mvp are updated to reference v3.0.0
-->

# Video-to-3D Project Constitution

Video-to-3D is a personal single-developer project: reconstructing a 3D representation of an
object from ordinary RGB phone video. The goals are learning (AI Engineering, ML Engineering, 3D
perception as a foundation for further Robotics) and a portfolio. The project's value is a
finished reproducible tool, a clear engineering contribution by the author and honestly measured
results. The constitution sets the invariant rules for all specify, plan, tasks and
implement stages.

## Core Principles

### I. Result honesty (NON-NEGOTIABLE)

The type of result the project promises and the type of result the project delivers MUST
match literally.

- It is prohibited to silently replace a promised mesh with a point cloud, a Gaussian splat or any
  other representation. A change of result type MUST be formalized as an explicit amendment
  to the specification stating the reason.
- Every artifact MUST carry metadata about its actual type (point cloud / mesh /
  splat), how it was obtained, known defects, coordinate system, units and scale
  status (confirmed by measurement / estimated by the model / not determined).
- Invisible surfaces completed by the model or an algorithm MUST NOT be passed off as
  observed geometry; any completion applied MUST be labeled.
- Demonstration materials (README, screenshots, videos) MUST show the result of a
  typical run, not the best of many. A selected successful example MUST be labeled
  as selected.
- Failures and limitations MUST be documented on a par with successes. "It did not work" is a
  valid stage result; hidden degradation is not.

Rationale: the only value of a learning portfolio project is trust in the claimed
results. A single undeclared substitution devalues the whole project.

### II. Run reproducibility

Any published result MUST be reproducible from fixed inputs.

- Every run MUST save a manifest: the input video (name, hash, duration, resolution,
  fps), the code version (git commit), stage parameters, versions of key dependencies, the
  execution environment (Mac / GPU machine), execution time.
- Randomness MUST be controlled: the seed is fixed and recorded in the manifest.
  Nondeterminism that cannot be eliminated MUST be named explicitly.
- Intermediate artifacts (selected frames, camera poses, point cloud, stage log) MUST
  be saved to disk in open formats and survive a process restart.
- Every pipeline stage MUST be launchable with one command and a config. An end-to-end launch with
  one command MUST be possible where all stages can run on one machine.
- If stages are split across machines (Principle IV), the handoff between them MUST be described as
  a standard pipeline step: what is transferred, which command continues the work, how the
  integrity of the transferred package is checked. Hidden and undocumented manual steps are prohibited.

Rationale: without a manifest and artifacts it is impossible to compare versions, analyze
regressions, or show the work to another person.

### III. One scenario, explicit boundaries

The first supported scenario is fixed and MUST be completed in full before a second
appears.

Supported scenario:
- one static rigid opaque object;
- the camera moves around the object;
- sufficient texture, viewpoint overlap, stable lighting;
- processing of recorded video, with no real-time requirements.

Outside the initial scope (MUST NOT be implemented before a separate constitution amendment):
dynamic and deformable objects, joint recovery, hand–object interaction, robot
control, physical simulation, training a foundation model, multi-user SaaS,
infrastructure "for future projects".

- A possible future large robotics project MUST remain separate. The current project MUST
  have its own achievable result and not depend on it.
- Input outside the scenario MUST be rejected or marked as unsupported with a
  clear message, and not processed "as best it can".

Rationale: a one-person project dies from scope expansion, not from a lack of ideas.

### IV. Separation of the light and heavy stages

Development MUST NOT be blocked by GPU availability.

- The interface, frame preparation and selection, viewing, export, diagnostics and quick checks
  MUST be developed and run locally on the Mac without access to an external GPU machine.
  Using the Mac's built-in graphics for visualization is allowed.
- Heavy computation MUST be launched as a separate step, write its result to files and allow
  subsequent viewing and analysis of those files on the Mac without recomputation.
- The boundary between the stages MUST be explicit: the heavy stage takes inputs from disk and gives
  outputs to disk, with no hidden dependence on process state.
- The characteristics of the GPU machine (model, VRAM, OS, environment) were unknown at the time of
  ratification. It is prohibited to fix CUDA, MPS, specific memory requirements or other
  hardware properties without verification at the plan stage. Unverified assumptions MUST be
  marked as requiring verification.
- Runs on the GPU machine are started manually by the author or on the author's explicit
  request; automatic access to it is not assumed.

Rationale: GPU access is intermittent, while progress must be daily.

### V. Measurability and diagnostics

Reconstruction quality MUST be assessed by applicable measurements and a visual check of
3D artifacts. Visual observations MUST be separated from quantitative conclusions.

- Every pipeline stage MUST produce diagnostics relevant to it: for example, the number
  and reasons of frame selection, the number of camera poses and points obtained, available error
  metrics, stage time. Inapplicable or unavailable metrics MUST be marked with a reason,
  and not replaced with an invented value or zero.
- The number of points, the model's confidence and the reprojection error MUST NOT by themselves be considered
  proof of the accuracy or completeness of the geometry. Claims about these properties MUST
  rely on an independent ground truth and a described comparison protocol; without it only
  explicitly named diagnostic indicators and observations are acceptable.
- The set of metrics and the way they are computed MUST be fixed at the specify stage before implementation,
  and not chosen after the fact to fit the result obtained.
- There MUST be a small fixed set of test videos (easy / medium /
  failure case) on which changes that can affect the
  reconstruction are checked; the results are saved and compared between versions. Changes to the UI,
  documentation and viewing that do not change the reconstruction are checked locally with saved
  artifacts and do not require a new GPU run.
- The failure case MUST remain in the set. Removing a test that has stopped passing is
  prohibited; only an explicit declaration of the limitation is permitted.

Rationale: without a baseline it is impossible to tell an improvement from random luck on
a single video.

### VI. Clear engineering contribution

The project MUST remain readable as engineering work, and not as a stitching together of other people's commands.

- Using ready-made libraries and models is allowed and encouraged. But for each
  borrowed component it MUST be recorded: what was taken, why, which alternatives were
  considered, what the author did themselves.
- The own contribution (pipeline, frame selection, diagnostics, interface, quality assessment,
  error handling) MUST be separable and described in the README.
- Code MUST be simple by default: complexity is introduced only for a specific measured
  problem and accompanied by an explanation in the decision record.
- Decisions that changed the direction of the project MUST be recorded in a short note (what was decided,
  why, what was rejected).

Rationale: for learning and a portfolio, the value is not the mere fact of a working pipeline but the visible
chain of engineering decisions.

## Scope and hardware constraints

- The primary development environment is the local Mac. Everything that is not heavy computation
  MUST work on it.
- Heavy tests run on a separate GPU machine, available intermittently. Its parameters at the time of
  ratification are not determined and MUST be established at the plan stage, not assumed.
- The first end-to-end technical result MAY be a colored point cloud with camera poses.
- The result type of the first public version, the need for a mesh and textures, and export formats
  MUST be explicitly defined at the specify stage after a feasibility check. Until then
  the project does not claim them publicly.
- The user path around which the product is built: video → frame preparation and selection →
  reconstruction → interactive viewing → export and diagnostics of the result.
  A change to the composition of this path MUST go through a constitution amendment.
- The project is not required to work in real time, serve multiple users or
  provide infrastructure to other projects.
- All repository content — documentation, specifications, code, comments, messages, test
  names and generated artifact text — MUST be written in English.
- Public materials MUST NOT reveal the author's infrastructure: host names, user names,
  absolute home paths, or details of how the GPU machine is accessed or shared. Run logistics
  live in local notes excluded from version control. The GPU model and measured resource
  usage MAY be stated, because the results cannot be interpreted without them.

## Workflow and quality gates

Work order: specify → plan → tasks → implement. The order MUST be observed for every
notable feature.

Gates that MUST be passed:

1. **Feasibility gate (specify).** A promise of a result type is permitted only after
   confirming its achievability on at least one test video, or an explicit
   "requires verification" mark. A violation is a direct violation of Principle I.
2. **Hardware gate (plan).** The plan MUST list its assumptions about hardware and
   environment and mark the unverified ones. An unverified assumption cannot become
   a requirement.
3. **Stage gate (plan).** The plan MUST state for each stage where it runs
   (Mac / GPU) and which files it passes on.
4. **Reproducibility gate (implement).** A change is considered done after checks of the
   affected behavior. For reconstruction changes, a run on the fixed
   set is mandatory, saving the manifest and artifacts and comparing applicable metrics; the first
   run establishes the baseline. An expected failure on the failure example is checked as a
   failure, and not as a successful reconstruction. For changes that do not affect the reconstruction,
   corresponding local checks on saved artifacts are sufficient.
5. **Honesty gate (before publication).** The README and demonstrations MUST match the
   actual behavior of the current code, including a section on known limitations.

Additionally:
- Automated checks are mandatory for deterministic logic (frame selection, geometry,
  I/O, export formats). For reconstruction quality, instead of unit tests, a
  comparison of applicable metrics and a visual check on the fixed set of videos is used.
- A change that worsens the metrics on the test set MUST either be fixed or accepted
  explicitly with a recorded reason and magnitude of the degradation.

## Governance

- This constitution takes precedence over any other practices, habits and convenient
  solutions within the Video-to-3D project. In a conflict with a plan, a task or code,
  the constitution wins.
- An amendment is made by changing this file and MUST contain: what changes, why, what
  becomes permitted or prohibited, which existing artifacts need updating.
- Versioning is semantic:
  - MAJOR — removal or incompatible redefinition of a principle, expansion of scope beyond
    the "Scope and hardware constraints" section;
  - MINOR — a new principle or section, a substantial expansion of the rules;
  - PATCH — clarifications of wording, typos, non-semantic edits.
- Scope expansion (for example, the start of a robotics direction, dynamic objects,
  multi-user mode) MUST be formalized as a MAJOR amendment, and not added in the course of
  implementation.
- A compliance check is performed by the author at every specify → plan → tasks →
  implement transition and before any publication. A violation found is either fixed or
  recorded as a conscious deviation with a reason and a deadline.
- The constitution as a whole is revised at the completion of each public version of the project.

**Version**: 3.1.0 | **Ratified**: 2026-09-18 | **Last Amended**: 2026-09-26
