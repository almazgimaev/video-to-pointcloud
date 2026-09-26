# Video-to-3D

A personal learning project: reconstructing a 3D representation of a single static object
from ordinary RGB phone video.

> **Status: in development, the end-to-end path has not yet been verified on real video.**
> No reconstruction has been run yet. The capabilities below describe the designed behavior,
> not an achieved result.

## What it does (v1 goal)

Input — one video of a static textured object, shot by walking around it.
Output:

- a colored **point cloud** (`point_cloud`) — **not** a mesh, **not** a textured model,
  **not** a Gaussian splat;
- camera poses for the viewpoints used;
- interactive viewing on the Mac;
- export of the point cloud to PLY;
- metadata and a run report.

## Honest limitations

- **The scale is not determined** (`scale_status: not_determined`): there is no independent ground truth,
  so metric accuracy and surface completeness are neither claimed nor measured.
- **The background is not removed.** Residual background in the result is normal for v1, and it is marked
  (`background_present: true`). Automatic segmentation of the object is not promised.
- The number of points, the model's confidence and the reprojection error are **not** proof
  of geometric accuracy.
- Input is limited: MP4/MOV containers, H.264/HEVC codecs. "Any video"
  is not supported.
- There is no real-time operation; the heavy stage runs separately on a machine with a GPU.

## Architecture in brief

| Stage | Where | What it does |
|---|---|---|
| Light | Mac | video check, frame extraction and selection, building the transfer package, ingesting the result, normalization, viewing, export, report |
| Heavy | GPU machine | reconstruction by the **upstream's standard script**; the project's own code is not moved there |

Exchange between the stages is via files, the transfer is manual and documented
(see `specs/001-video-3d-mvp/contracts/run-package.md`).

## Glossary

These names are used in all documents, messages and directory names. No synonyms
are introduced.

| Term | Meaning | Where on disk |
|---|---|---|
| **run** | one unit of work: one video from preparation to result | `runs/<run_id>/` |
| **transfer package** (also: package) | a self-contained directory for transfer to the GPU machine | `runs/<run_id>/package/` |
| **reconstruction result** | the upstream's output as is, COLMAP sparse format | `runs/<run_id>/result/` |
| **normalized result** | the representation converted to our conventions | `runs/<run_id>/normalized/` |
| **run report** | a human-readable summary of the run | `runs/<run_id>/report.md` |

## Borrowed / built by the author

| Taken | Why |
|---|---|
| VGGT + `facebook/VGGT-1B` weights | single-pass reconstruction |
| COLMAP sparse format | a single result contract |
| `pycolmap` | reading the binary model |
| Rerun | interactive viewing |
| ffmpeg / ffprobe | decoding and frame orientation |

Built by the author: frame preparation and selection, the transfer contract and integrity check, the data
model and coordinate conventions, diagnostics and report, error and status handling, the viewing
layout, an experiment comparing frame selection methods.

## Licenses

- Project code — see `LICENSE` (to be added).
- The `facebook/VGGT-1B` weights are distributed under **CC-BY-NC-4.0** — non-commercial
  use only. The project is educational, and this restriction is respected.
- VGGT code — the upstream license; COLMAP — BSD.

## Installation (Mac)

```bash
brew install ffmpeg
make install          # venv + dependencies
make test             # automated checks
```

Pinned versions are in `requirements.lock`, updated with
`.venv/bin/pip freeze --exclude-editable > requirements.lock`.

## Shooting

Before your first shoot, read the [guide](docs/shooting-guide.md): which object
is suitable, how to move around it, what the system checks itself and what it takes from you
on trust. Shooting conditions are not checked automatically — meeting them is up to you.

## Documents

The specification, plan, data model and contracts are in `specs/001-video-3d-mvp/`.
Project rules — `.specify/memory/constitution.md`.
