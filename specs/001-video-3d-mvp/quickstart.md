# Quickstart: verifying Video-to-3D MVP

**IMPORTANT. Everything below is a planned interface and a planned verification sequence.
No `v3d` command is implemented. Nothing here has been executed: dependencies were not
installed, weights were not downloaded, the GPU machine was not used. Figures for time, memory,
and quality are deliberately absent from this document — they will appear only after the first
actual run (M1).**

---

## 0. What runs where

| Step | Machine | Requires the GPU machine |
|---|---|---|
| 1. Preparing the Mac environment | Mac | no |
| 2. Video validation, frame selection, package assembly | Mac | no |
| 3. Manual transfer of the package | human | — |
| 4. Reconstruction with the upstream command | GPU machine | yes |
| 5. Manual transfer of the result back | human | — |
| 6. Ingestion, viewing, export, report | Mac | no |
| 7. Comparison of selection methods (P2) | Mac | no (the runs are already done) |

---

## 1. Mac: preparing the environment (planned)

```bash
brew install ffmpeg          # decoding and frame extraction
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # numpy, Pillow, pycolmap, rerun-sdk, pytest
```

**Expected**: `v3d --help` lists the subcommands `check, prepare, ingest, view, export,
report, compare`.

**Typical failures**: no `ffmpeg` in PATH → `v3d check` ends with a clear message;
`pycolmap` or `rerun-sdk` do not install on this Python version → record the actual versions
and, if needed, apply the fallbacks from research.md (our own `*.bin` parser,
viser).

---

## 2. Mac: quick checks without any video (level A)

```bash
pytest tests/unit tests/contract
```

**Expected**: checks pass for frame metrics, selection determinism, geometry
(`world_to_camera` ↔ camera centers, inverse transformation), reading/writing formats,
computing and verifying `SHA256SUMS`, return codes, refusal on `run_id` mismatch.

The synthetic scene is used here only for formats and geometry.
**It does not prove the quality of real reconstruction.**

---

## 3. Mac: checking the viewer and export on a saved example (level B)

```bash
v3d view   assets/precomputed/sample_run
v3d export assets/precomputed/sample_run --out /tmp/sample.ply
v3d report assets/precomputed/sample_run
```

**Expected**:
- a viewer window with the point cloud, camera poses (toggleable), a panel with run
  information, and a warnings panel;
- a permanent badge **"precomputed example"**;
- the result status is visible at all times;
- `/tmp/sample.ply` opens in a third-party viewer, the point count matches the displayed one
  and the report;
- `report.md` contains the scale status `not determined`, the background-presence flag, and a section
  "interpretation limits".

**Typical failures**: no `normalized/` directory → code 7; `run_id` in files does not match →
refusal naming the file.

---

## 4. Mac: preparing your own video (stages 1–2)

```bash
v3d check   ~/videos/object_01.mov
v3d prepare ~/videos/object_01.mov --budget 80 --selector uniform --seed 42 --out runs/
```

`--budget 80` here is an **illustration of the syntax, not a recommended value**: the default
value is assigned after M1.

**Expected**: a message about whether the file is supported; the number of source and selected frames; a summary
of rejection reasons; a created `runs/<run_id>/` with `manifest.json`, `frames.json`, and
`package/{images,SHA256SUMS,RUN_ON_GPU.txt}`.

**Typical failures**: a format outside the list → code 2 with the list of supported ones; too few
usable frames → code 3, the package is not created; the run directory already exists → code 8.

**Warnings are observations**: "many blurry frames", "viewpoints differ little".
They do not block work and do not name a cause.

---

## 5. Manual transfer to the GPU machine (standard step)

```bash
# by any convenient means: external drive, shared folder, manual scp
cp -R runs/<run_id>/package /Volumes/USB/<run_id>_package
# optionally — verification on the receiving side, with the OS's standard utility:
cd <package> && sha256sum -c SHA256SUMS      # on macOS: shasum -a 256 -c SHA256SUMS
```

Our code is not on the GPU machine and does not appear there.

---

## 6. GPU machine: reconstruction with the upstream command (level C)

**Rules for working with the GPU machine (mandatory):** the Linux machine with an RTX 5090 (32 GB) is
available intermittently. Runs on it are started manually by the author; no automatic connections
are made. Before a run, take `nvidia-smi` and check the free GPU memory: memory already in
use on the card before the run is subtracted from the budget; if there is not enough free memory, postpone the run.
The working memory budget is **≤ 24 GB** and the whole card is not used; when short, reduce
the frame budget.

One-time machine setup (done by the user; **versions not verified**):

```bash
nvidia-smi                       # record the GPU model and the actual amount of VRAM
git clone https://github.com/facebookresearch/vggt && cd vggt
pip install -r requirements.txt  # PyTorch must be built for sm_120 (CUDA 12.8), see research.md R-0
```

The run:

```bash
python demo_colmap.py --scene_dir=/path/to/<run_id>_package
# or with bundle adjustment — then reprojection error becomes available:
python demo_colmap.py --scene_dir=/path/to/<run_id>_package --use_ba
```

**Expected artifacts**: `<package>/sparse/{cameras,images,points3D}.bin`,
`<package>/points.ply`, the saved stdout.

**What to record manually**: the `nvidia-smi` output before and during the run (actual
VRAM consumption), the torch and CUDA versions, the repository commit, the weights identifier, the full command line,
the execution time. This information is transferred back together with the result and goes into the manifest.

**Typical failures**:
- `CUDA error: no kernel image is available` / sm_120 incompatibility → PyTorch is not built for
  Blackwell (research.md R-0);
- `CUDA out of memory` → reduce the frame budget at `prepare` and repeat the transfer;
  if exhausted — VGGT-Omega or the COLMAP fallback path;
- interruption (Ctrl-C, machine going offline) → the result is incomplete; **resuming from the middle
  is not supported**, the run is repeated in full; the already assembled `package/` is valid as is.

---

## 7. Mac: ingesting the result, viewing, export (stages 4–5)

```bash
v3d ingest runs/<run_id> --from /Volumes/USB/<run_id>_package
v3d view   runs/<run_id>
v3d export runs/<run_id> --out runs/<run_id>/export/object.ply
v3d report runs/<run_id>
```

**Expected on success**: `status = success`; `normalized/cameras.json` declares
`world_to_camera`, OpenCV axes, `units = unknown`, `scale_status = not_determined`;
`diagnostics.json` has the number of selected frames, the number of registered cameras, the number of valid
points, stage times, reprojection error (or an "unavailable" flag with a reason), the result of the
visual check against the six defects, and an "interpretation limits" block.

**Expected on a partial result**: `status = partial`, viewing is allowed, the "partial"
badge is visible at all times, the share of frames used is in the report.

**Typical failures**:
- required files missing or not everything transferred → code 7 naming the file;
- the model's image names do not match the selected frames → code 7;
- no cameras or no valid points → code 6, `status = invalid_result`, export prohibited;
- the result is cut off → code 5, `status = interrupted`, viewing only with a banner.

---

## 8. M1 — the first feasibility check

Performed once, on your own video shot per the guide.

**Success signs** (no invented thresholds): the heavy stage finished and wrote all
required files; `cameras_registered > 0` with the share recorded; `num_points_valid > 0`,
the cloud is colored; visual check — the shape is recognizable, the camera trajectory has no obvious breaks;
the export opens in a third-party viewer; the manifest is filled in, unavailable fields are marked
with a reason.

**The first successful run is declared the baseline.** Before it, thresholds for time, memory, and quality
are not assigned.

**Cost limit**: no more than three GPU runs and one session. When exhausted — the fallback
path (COLMAP), rather than debugging someone else's model.

---

## 9. Fallback path without the GPU machine (level C on the Mac)

```bash
brew install colmap
colmap feature_extractor  --database_path db.db --image_path runs/<run_id>/package/images
colmap sequential_matcher --database_path db.db
colmap mapper             --database_path db.db --image_path runs/<run_id>/package/images --output_path sparse
# then the same ingestion of the result:
v3d ingest runs/<run_id> --from .
```

Slow, but it gives the **same result format** and allows verifying stages 4–6 without access
to the GPU machine.

---

## 10. The P2 experiment (level D)

```bash
# 1) conditions are fixed BEFORE the runs and are not edited afterwards
v3d compare --init experiments/exp_01/conditions.json \
            --video ~/videos/object_01.mov --budget 80 --seed 42

# 2) two branches: the same budget, one reconstructor, one seed
v3d prepare ~/videos/object_01.mov --budget 80 --selector uniform            --seed 42
v3d prepare ~/videos/object_01.mov --budget 80 --selector quality            --seed 42
#    → transfer, upstream run, transfer back, ingest for each branch

# 3) comparison
v3d compare runs/<run_A> runs/<run_B> --conditions experiments/exp_01/conditions.json
```

**Expected**: `comparison.md` with a side-by-side of the pre-declared metrics, including
**the time of frame selection itself**, and a verdict `improvement | no_difference | regression |
inconclusive`.

**Typical failures**: anything other than the selection method differs → code 9, the comparison
is not performed.

**A negative result is a valid outcome.** "Improvement" thresholds are not assigned in advance,
conditions are not changed retroactively, threshold tuning is done on a separate tuning video
that is not part of the verification set.
