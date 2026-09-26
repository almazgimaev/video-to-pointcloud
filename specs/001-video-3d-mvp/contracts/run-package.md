# Contract: transfer package and manual transfer

**Status: designed contract, not implemented.**

## 1. Package contents (created on the Mac by the `v3d prepare` command)

```text
package/
├── images/
│   ├── 0000_000012.jpg
│   ├── 0001_000047.jpg
│   └── ...
├── SHA256SUMS
└── RUN_ON_GPU.txt
```

- `images/` contains **only the selected** frames, already rotated to the correct orientation
  and, if necessary, downscaled. File name = `frame_id` (see data-model.md §3.2).
- The subdirectory is called `images/` because that is exactly the layout expected by the VGGT
  upstream `demo_colmap.py --scene_dir=...`. This is a deliberate adaptation to someone else's
  contract, so that **our code is not moved to the GPU machine**.
- `SHA256SUMS` — checksums of all package files, in the `shasum`/`sha256sum` utility format.
- `RUN_ON_GPU.txt` — a human-readable instruction: the exact upstream command, the expected output
  files, what to return, the `run_id`.

The run manifest (`manifest.json`) stays **outside** the package: the GPU machine does not need it,
and an extra file in `scene_dir` must not end up in processing.

## 2. Manual transfer — a standard step, not a hidden one

Direction "there": the `package/` directory is copied by any means (external drive, shared folder,
`scp` run manually by the user). Transfer automation, SSH integration and a scheduler are out of
scope (FR-015).

Direction "back": a directory is returned containing
`sparse/{cameras,images,points3D}.bin`, `points.ply`, the run log. It is placed in
`runs/<run_id>/result/` (or specified via `v3d ingest --from`).

## 3. Integrity check

| Moment | Who checks | With what | Mode |
|---|---|---|---|
| right after the package is built | `v3d prepare` | all files present, no extras | strict |
| before the run on the GPU machine | a person, optionally | `shasum -a 256 -c SHA256SUMS` — a standard OS utility, none of our code required | — |
| after the result is returned | `v3d ingest` | files listed in `SHA256SUMS` are unchanged and not missing; the model's image names ⊆ selected frames; required result files are present; `run_id` matches | non-strict |

**Important about "extra" files.** Upstream (`demo_colmap.py --scene_dir=<package>`) writes its
outputs — `sparse/` and `points.ply` — **into the same directory** we passed. Therefore after a
run, files that were not in `SHA256SUMS` legitimately appear in the directory.

- The standard `shasum -a 256 -c` utility does not notice this: it checks only the listed
  files. Its result stays green.
- Our check MUST behave the same way: **by default extra files are not an error**.
  An error (code 4) is only a change to or disappearance of a file listed in `SHA256SUMS`.
- Strict mode (extra files forbidden) applies **only** right after the package is built,
  before it is transferred.

A mismatch in any item is a refusal naming the specific file or name; silently
continuing is prohibited.

## 4. Prohibited

- Automatic connection to the GPU machine, a permanently running service, a job queue.
- Transferring the source video to the GPU machine (only the selected frames are transferred).
- Mixing result files from different `run_id`s in one run directory.
