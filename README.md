# Video-to-3D

**From a phone video of an object to a coloured 3D point cloud — with honest diagnostics.**

![First video frame next to a render of the reconstruction from the same camera pose](docs/assets/pair_0000_000000.webp)

*Left: a frame of the input video. Right: the reconstructed point cloud, rendered from the camera
pose estimated for that frame.*

A small, end-to-end tool built as a learning and portfolio project in 3D perception. You record a
short orbit around a static object; the tool selects frames, hands them to a reconstruction model
running on a GPU machine, brings the result back, stands the object upright, and lets you inspect,
export and measure it.

- **Detailed report** — every step, measurement and failure: [`docs/report/report.md`](docs/report/report.md)
- **Project page** — a visual walk-through with an interactive 3D view: <https://almazgimaev.github.io/video-to-pointcloud/> (source in [`site/`](site/))

## Results

| | |
|---|---|
| Input | 26.5 s phone video, 1920×1080 |
| Camera poses | 48 of 48 frames registered, full orbit |
| Output | coloured point cloud, upright and centred; PLY export |
| Frame selection experiment | selecting sharp, non-redundant frames keeps the whole object under a strict threshold where uniform sampling keeps fragments — judged by a rule fixed before the runs |
| Tests | 552, all on a laptop, no GPU |

| Front | Three-quarter | Top |
|---|---|---|
| ![](docs/assets/view_front.webp) | ![](docs/assets/view_three_quarter.webp) | ![](docs/assets/view_top.webp) |

## How it works

```
video ─► check ─► select frames ─► transfer package ─► reconstruct (GPU) ─► ingest ─► view / export / report
        └──────────────── laptop ────────────────┘   └── separate machine ──┘  └────── laptop ──────┘
```

- **Reconstruction**: [VGGT](https://github.com/facebookresearch/vggt) (`VGGT-1B`), run with its
  own script — no project code on the GPU machine. The two sides exchange files, with checksums.
- **World alignment**: the model anchors its world to the first camera, so the raw result is
  tilted. The vertical is estimated from the ring of camera positions, the object is centred, and
  the transform is recorded as an estimate.

  ![Before and after alignment](docs/assets/alignment_before_after.webp)

- **Frame selection**: uniform in time, or sharpest-per-window without near-duplicate views.
- **Diagnostics**: registered cameras, valid points, timings, a visual-check record, and explicit
  "unavailable: reason" instead of made-up numbers.

## What the numbers do not claim

- **Scale is not determined** — there is no independent reference, so no metric accuracy or
  surface completeness is claimed.
- **The result is a point cloud**, not a mesh. Residual background is kept and marked.
- **The vertical is an estimate** from the camera trajectory, not a measurement.
- **One real capture** was measured. The frame-selection result is not generalized.

## Try it

```bash
brew install ffmpeg
make install                                   # venv + dependencies
make test

.venv/bin/v3d check video.mp4
.venv/bin/v3d prepare video.mp4 --out runs     # prints run_id and the transfer package path
# reconstruct the package on a GPU machine (see the report, section 7), copy sparse/ back, then:
.venv/bin/v3d ingest runs/<run_id>
.venv/bin/v3d view runs/<run_id>
.venv/bin/v3d media runs/<run_id>
```

A ready-made example that needs no GPU: `.venv/bin/v3d view assets/precomputed/sample_run`
(a synthetic scene, marked as a precomputed example).

Recording tips: [`docs/shooting-guide.md`](docs/shooting-guide.md).

## Project layout

| Path | What |
|---|---|
| `src/v3d/` | the tool: frame selection, transfer package, ingest, alignment, viewer, export, rendering |
| `tests/` | unit, contract and integration tests, incl. a synthetic scene with known ground truth |
| `docs/report/` | the detailed report, charts and the scripts that build them |
| `docs/` | shooting guide, feasibility notes, the P2 experiment |
| `experiments/` | conditions and results of the frame-selection experiment |
| `specs/` | specification, plan, data model, contracts and tasks |
| `site/` | the project page (static) |

Terms used throughout: a **run** is one video processed end to end (`runs/<run_id>/`); the
**transfer package** is what goes to the GPU machine (`package/`); the **normalized result** is the
aligned output the tool works with (`normalized/`).

## Licences

- Project code: [MIT](LICENSE). The licence covers this repository only, not the VGGT code or weights.
- VGGT code: licence of the upstream repository. **`VGGT-1B` weights: CC-BY-NC-4.0** — non-commercial
  use only; this project is non-commercial.
- COLMAP format tooling (`pycolmap`): BSD.
