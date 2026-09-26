# Research: Video-to-3D MVP (Phase 0)

**Date**: 2026-09-18
**Feature**: [spec.md](./spec.md)

Rule of this document: documentation claims are separated from our own assumptions.
Nothing here has been verified by running — dependencies were not installed, weights were not downloaded,
the GPU machine was not used. The phrase "per documentation" means reading a source,
not verifying.

---

## R-0. State of the hardware information

**Clarified by the user on 2026-09-18** (questions Q1 and Q2 are closed):

| Parameter | Status | Value / comment |
|---|---|---|
| GPU model | **known** | NVIDIA RTX 5090 (Blackwell, compute capability sm_120) |
| VRAM | **known** | 32 GB. The GPU machine is available intermittently. A working budget of **≤ 24 GB** is planned — a self-imposed limit, not the card's limit. Memory already in use on the card before a run is subtracted |
| Access mode | **known** | ssh, connected manually by the user. **Runs are started manually by the author.** Automatic connection is prohibited |
| GPU machine OS | **known** | Linux |
| Driver / CUDA / Python environment | **probably present** | per the user, "most likely"; verified at first login with `nvidia-smi`, `python -V`, `nvcc --version` — before M1 |
| Internet access from the machine | **known** | available → cloning the VGGT repository and downloading the weights are feasible |
| Actual VGGT VRAM consumption on N frames | **unknown** | measured on M1 (Q3); no default values are assigned until then |

**Consequence for PyTorch (per documentation)**: sm_120 requires PyTorch builds with CUDA 12.8;
official sm_120 support appeared starting with PyTorch 2.7.0 (sources below). This is a
**documentation claim, not a verification on the machine**: actual versions are fixed at
first login and recorded in the manifest.

Sources: [pytorch/pytorch#159207](https://github.com/pytorch/pytorch/issues/159207),
[PyTorch Forums: sm_120](https://discuss.pytorch.org/t/is-there-a-pytorch-build-that-supports-nvidia-rtx-5090-compute-capability-12-0-sm-120/223536),
[SaladCloud: PyTorch on RTX 5090](https://docs.salad.com/container-engine/tutorials/machine-learning/pytorch-rtx5090).

---

## R-1. Comparison of reconstructor candidates

Verified sources (read on 2026-09-18):
[facebookresearch/vggt](https://github.com/facebookresearch/vggt),
[VGGT README (raw)](https://raw.githubusercontent.com/facebookresearch/vggt/main/README.md),
[demo_colmap.py (raw)](https://raw.githubusercontent.com/facebookresearch/vggt/main/demo_colmap.py),
[HF facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B),
[yyfz/Pi3](https://github.com/yyfz/Pi3),
[Pi3 README (raw)](https://raw.githubusercontent.com/yyfz/Pi3/main/README.md),
[colmap.github.io/install](https://colmap.github.io/install.html),
COLMAP documentation (cli.rst, features.rst, faq.rst) via Context7 `/colmap/colmap`.

| Criterion | VGGT | Pi3 / Pi3X | COLMAP (classic) |
|---|---|---|---|
| Input | a sequence of images (1…hundreds of views) | a B×N×3×H×W tensor, values in [0,1]; the demo accepts a folder or a video with `--interval` | a folder of images |
| Output | extrinsics + intrinsics, depth maps + conf, point maps + conf, 3D tracks | `points` (global), `local_points`, `conf` (logits), `camera_poses` (4×4, camera-to-world, OpenCV) | sparse model: cameras/images/points3D, colored points; optionally dense MVS |
| Ready export without our code | **yes**: `demo_colmap.py --scene_dir=...` → `sparse/{cameras,images,points3D}.bin` + `points.ply` | `example_mm.py --data_path … --save_path …` → `.ply` | **yes**: the standard CLI + `model_converter … --output_type PLY` |
| Camera convention | OpenCV, "camera from world" (world→camera) — stated in the README | OpenCV, camera-to-world 4×4 — stated in the README | world→camera (quaternion + t) in `images.bin` |
| Scale status | scale is not metric; metric scale is not claimed in the documentation → **scale_status = not determined** | Pi3X: "approximate metric scale"; Pi3: scale-invariant | not metric (SfM up-to-scale) unless a reference dimension is given |
| Hardware / VRAM requirements | **no figures in the README**; it is stated that after the 2026-05-15 optimization "2–3 times more frames" fit in the same memory budget; VGGT-Omega was released (2026-05-18). bfloat16 at CC ≥ 8.0 | **no figures** | the GPU is needed for SIFT extraction/matching (acceleration); the FAQ has a matching memory estimate formula: `4·M² + 4·M·256` bytes at `max_num_matches M` (≈400 MB at M=10000). Works on CPU too |
| Limit on frame count / resolution | processing resolution 518 px (hard-coded inside VGGT), loading in `demo_colmap.py` — 1024 px, the result is scaled back to the original size. The frame limit is not set by the documentation → determined by memory | no limit set; there is `--interval` for thinning | the practical limit is time (matching grows quadratically with exhaustive; `sequential_matcher` exists for video) |
| Installation complexity | git clone + `requirements.txt`; weights from HuggingFace; a working PyTorch for sm_120 is needed | git clone + `requirements.txt`; weights `yyfz233/Pi3`, `yyfz233/Pi3X` | build from source for CUDA; on Mac `brew install colmap`; distribution packages have no CUDA |
| Code license | "commercial-use-friendly" (since 2025-07-29), with a military-use exclusion | BSD 3-Clause | BSD (the COLMAP project) |
| Weights license | `facebook/VGGT-1B` — **CC-BY-NC-4.0** (non-commercial); there is a separate `facebook/VGGT-1B-Commercial` with request approval | **CC BY-NC 4.0**, strictly non-commercial | no weights |
| Suitability for a static textured object | yes (a typical feed-forward reconstruction scenario) | yes | yes, this is a classic SfM scenario; suffers on weak texture |
| Error diagnostics | with `--use_ba` reprojection error is available (`--max_reproj_error`, default 8.0); without BA — only confidence | confidence (logits) | reprojection error, number of registered images, tracks — standard |

**Our own assumptions (not from the documentation)**: (a) our scenario will need
on the order of 50–150 frames; (b) at 24 GB and 518 px resolution VGGT will fit within this amount.
Both values are **not verified** and are to be measured on M1.

---

## R-2. Decision: the main reconstructor is VGGT (via `demo_colmap.py`)

**Decision**: VGGT, run on the GPU machine by the standard upstream script
`python demo_colmap.py --scene_dir=<package>`; weights `facebook/VGGT-1B`.

**Rationale**:
1. **Delivers exactly what the spec promises**: a colored point cloud + camera poses, in a single
   pass, without tuning SfM parameters.
2. **Does not require our code on the GPU machine** (the user's key constraint): on the remote
   machine only the upstream repository and the weights live; our project is not moved there.
3. **The result format is COLMAP sparse** (`cameras.bin`, `images.bin`, `points3D.bin`) plus
   `points.ply`. This is a stable documented contract with ready readers, and it is also
   what classic COLMAP produces. So the fallback needs no second integration code —
   there is a single reader on the Mac.
4. **Conventions are stated explicitly** (OpenCV, camera-from-world), which is necessary for an honest record
   of axes and transformations (Principle I).
5. The CC-BY-NC-4.0 weights license is **sufficient** for a personal educational/portfolio project;
   the restriction is recorded in the project README.

**Alternatives considered**:
- **Pi3X** — rejected as the main one: approximate metric scale is attractive, but
  (a) the output is in its own `.ply` format + a dictionary of tensors, which requires our code on the
  GPU machine or our own converter, (b) "approximate metric scale" cannot be claimed to the user
  without our own independent reference (Principle I, FR-039), (c) the weights license
  is the same NC, no advantage. Kept as a candidate if a scale requirement appears.
- **Classic COLMAP** — rejected as the main one due to setup effort and high
  sensitivity to weak texture, but **kept as a fallback path**: it gives the same file format,
  installs on the Mac via Homebrew (i.e. gives an end-to-end run without the GPU machine at all,
  albeit slowly), needs no weights, and has no license restrictions.
- **VGGT-Omega** — not taken for M1: too fresh (2026-05-18), the memory gain is needed
  only if 24 GB turns out to be too little. Candidate No. 1 for revisiting.

**Condition for revisiting the choice** (any of):
1. VGGT does not fit into the actual VRAM at a frame budget ≥ 40 even after reducing the number of
   frames → first VGGT-Omega / a smaller budget, then COLMAP.
2. On the own M1 video the camera poses are systematically implausible (the trajectory breaks,
   the cloud does not form a recognizable shape) with shooting conditions met → COLMAP as a check
   of "is it the model or the shooting".
3. The GPU machine environment does not allow installing PyTorch for sm_120 → COLMAP (CPU/CUDA) or
   postpone until the environment is clarified.
4. A metric scale requirement appears → revisit toward Pi3X + an independent reference
   (and a mandatory spec amendment).

**Not planned**: integrating all candidates, a universal framework of swappable backends,
a second 3D backend for the sake of the P2 experiment baseline. The P2 baseline uses the same VGGT.

---

## R-3. Decision: the boundary around the reconstructor

**Decision**: a "thin boundary" — one adapter module on the Mac that can read the **COLMAP
sparse format** and convert it to our normalized representation. The reconstructor
is invoked by a human as an external program; our code neither imports nor wraps it.

**Rationale**: the reconstructor physically lives on another machine; the only contract that
crosses the boundary is files on disk. A "backend plugins" abstraction would be code without
a user (Principle VI — simplicity by default).

**Alternatives considered**: (a) a Python wrapper importing VGGT — requires our code and
heavy dependencies on the GPU machine, rejected; (b) a registry of backends with a common interface —
premature with a single implemented path.

---

## R-4. Decision: a fallback end-to-end path without the GPU machine

**Decision**: a documented fallback path — COLMAP on the Mac (`brew install colmap`,
`feature_extractor` → `sequential_matcher` → `mapper` → `model_converter --output_type PLY`).
Slow, but it gives the same file contract and allows developing and verifying stages 4–6
without access to the GPU.

**Rationale**: Principle IV — progress must not be blocked by GPU availability. Plus insurance
in case the GPU machine environment turns out to be unsuitable.

**Alternatives considered**: only saved precomputed examples — not enough,
it does not allow verifying result normalization on new inputs.

---

## R-5. Decision: the viewer is Rerun

**Decision**: interactive viewing — the ready-made **Rerun** viewer (`rerun-sdk`, Python + the desktop
viewer on the Mac). Our code only logs entities: the point cloud (`Points3D` with colors), camera
poses (`Pinhole` + `Transform3D`), text panels with run information, warnings,
and status. No custom frontend is created.

**Rationale**:
- We need the cloud, camera poses, text, and a permanently visible status at the same time — MeshLab or
  CloudCompare show only the cloud.
- Works locally on the Mac with standard graphics, the GPU machine is not needed (Principle IV).
- Allows saving a recording (`.rrd`) as an artifact and opening it later without recomputation (FR-027).

**Alternatives considered**:
- **viser** (used in VGGT's `demo_viser.py`) — a browser server; it effectively turns
  into a separate frontend, and the panels and statuses would have to be written ourselves. Kept as a fallback.
- **Open3D visualizer** — the cloud and cameras can be shown, but text panels and statuses are
  inconvenient; plus a separate question of wheel compatibility on macOS ARM.
- **PLY export only + a third-party viewer** — does not satisfy FR-025, FR-026, FR-028.

**Not verified**: compatibility of `rerun-sdk` versions with Python on the Mac. Installation was not performed.
Versions are fixed at the implementation stage.

---

## R-6. Decision: frame extraction and orientation

**Decision**: decoding and frame extraction — external `ffmpeg` / `ffprobe` (Homebrew).
`ffprobe` gives the container, codec, duration, fps, frame count, and `side_data` with the rotation
matrix; `ffmpeg` applies auto-rotation by default when re-encoding frames.

**Rationale**: phone video orientation is a known source of errors; ffmpeg handles
it in a standard, documented way. The alternative — OpenCV `VideoCapture`, which historically
ignores the display matrix, causing frames to come out rotated 90°.

**Alternatives considered**: PyAV (the same ffmpeg, but rotation would have to be applied by us);
OpenCV (risk of silent loss of orientation); a pure Python decoder — unrealistic.

**Consequence for the contract**: the manifest records the rotation actually applied
and an "orientation applied" flag, not the original flag.

---

## R-7. Decision: frame selection metrics (light stage)

**Decision**: two observable per-frame indicators, both computed on the Mac on a downscaled copy of the frame:
1. **Sharpness** — variance of the Laplacian, a 3×3 numpy convolution.
2. **Difference between neighboring viewpoints** — the mean absolute difference of normalized 64×64
   grayscale frames between the candidate and the last selected one.

**Rationale**: both metrics are cheap, deterministic, need no additional dependencies
(numpy + Pillow), and directly match FR-006 ("verifiable properties of frames").

**Honest caveat**: the frame difference is a **proxy** for camera motion, not a measurement of viewpoint
shift; it is worded that way in the report and warnings (FR-007).

**Alternatives considered**: optical flow / ORB matching between frames (more accurate, but
more expensive and pulls in OpenCV); dropping the metrics and using pure uniform sampling (does not allow running
the P2 experiment).

---

## R-8. Decision: object segmentation

**Decision**: a segmentation model is **not introduced**. Residual background is allowed and marked
(FR-020). If separating the object is needed later, the minimal way is **geometric
filtering after reconstruction** (cropping by radius/box around the dominant cluster of points,
set by the user), and **only at the viewing/export stage**.

**Rationale**: removing the background *before* reconstruction removes the features from which camera
poses are recovered: the background provides parallax and connectivity between viewpoints. Masking the input will almost certainly
degrade camera registration. Post-filtering does not affect the cameras at all.

**Alternatives considered**: SAM/rembg on the input frames — rejected for the reason above
plus an extra heavy dependency; manual masking — laborious and not reproducible.

---

## R-9. Decision: language, structure, dependencies

**Decision**: Python (3.11+), a single package `v3d`, a CLI on `argparse`, a separate viewer as
a subcommand that launches Rerun. Mac-stage dependencies: `numpy`, `Pillow`, `rerun-sdk`,
`pycolmap` (reading the sparse model) — exact versions are fixed at implementation, none is
installed or verified.

**Rationale**: a minimal set; `pycolmap` spares us from writing our own parser of the binary
COLMAP format (Principle VI: we borrow — we record it).

**Alternatives considered**: our own `*.bin` parser (extra work and a source of errors,
but remains a fallback if `pycolmap` does not install on the Mac); Typer/Click instead of argparse
(an extra dependency for cosmetics).

---

## R-10. Open questions blocking the declaration of the plan as ready

| # | Question | State | How it is resolved |
|---|---|---|---|
| Q1 | GPU machine OS and environment | **closed** 2026-09-18: Linux, environment and internet available, ssh access on the user's side | confirming versions at first login |
| Q2 | VRAM amount | **closed** 2026-09-18: 32 GB, working budget ≤ 24 GB, runs started manually by the author | — |
| Q3 | Actual VGGT VRAM and time consumption on N frames | **open, non-blocking** | measured on M1; until then the default frame budget is not assigned |

No blocking questions remain. The R-2 decision and the heavy stage are confirmed at the hardware level;
the only thing left unknown is actual resource consumption, which is measured rather than
assumed.
