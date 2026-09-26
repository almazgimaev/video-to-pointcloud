"""Generator of the precomputed example run (T036).

IMPORTANT (A-07, FR-028, the project constitution — honesty of the result):
    The directory this script creates is **not the result of a real reconstruction**.
    The frames come from the synthetic `ffmpeg testsrc` test video, and the "reconstructor
    result" is an analytically defined scene from `assets/synthetic/make_scene.py`: it has
    neither a feature detector nor optimization. Therefore the example is marked with the
    `precomputed_example: true` flag in `normalized/status.json`, and this mark must reach
    the viewer and the report.

Why it is needed
----------------
The viewer (T033), export (T034) and report (T035) read only `normalized/` and
`manifest.json` and must work without the reconstruction environment and without
recomputation (FR-027). To be able to write and check them on a Mac without a real video,
a saved ready-made run is needed. This is it.

What happens
------------
1. `ffmpeg` generates a short test video (`testsrc`, a few seconds, 320x240).
2. `v3d.prepare.run_prepare` goes through stages 1–2: candidates, metrics, selection, package.
3. `assets/synthetic/make_scene.py --names-from <frames.json>` builds a synthetic
   COLMAP model **under the frame names of this run** — this way result ingest is checked
   without forging `frames.json`.
4. `v3d.ingest.run_ingest(..., precomputed_example=True)` ingests the result and writes the
   normalized representation with the mandatory mark.
5. The finished product is placed into `assets/precomputed/sample_run/` together with
   `README.md`.

Idempotence and failures
------------------------
All work goes in a temporary directory; the target directory is recreated only after the
whole run succeeded. A half-empty result is not created on any error: any failure is an
exception with a clear text.

Usage
-----
    python assets/precomputed/make_sample.py
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO / "assets" / "precomputed" / "sample_run"
MAKE_SCENE = REPO / "assets" / "synthetic" / "make_scene.py"

# The example parameters are chosen for one requirement: it must be light. The budget is
# measured in megabytes, so both the frame is small and there are few points.
VIDEO_SIZE = "320x240"
VIDEO_FPS = 10
VIDEO_DURATION_S = 4
FRAME_BUDGET = 12
LONG_SIDE_PX = 320
JPEG_QUALITY = 85
SCENE_POINTS = 300
SCENE_SEED = 5

# What is carried into the example directory. `work/` (intermediate frames) is not carried:
# it is not part of the portable package and is not needed in the example. `result/` is the
# raw upstream output; it is copied for completeness but does not get into git: the root
# .gitignore contains the line `assets/precomputed/**/result/`.
COPIED_ENTRIES = ("manifest.json", "frames.json", "normalized", "package", "result")

PRECOMPUTED_WARNING = (
    "WARNING: this is a precomputed example, not the result of a real reconstruction."
)


def _require_tool(name: str) -> None:
    """Check that an external program is in PATH before anything is created."""
    if shutil.which(name) is None:
        raise RuntimeError(
            f"program '{name}' is required in PATH: the example cannot be built without it"
        )


def _run(command: list[str], *, what: str) -> None:
    """Run an external command and turn its failure into a clear error."""
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{what}: the command finished with code {completed.returncode}\n"
            f"  command: {' '.join(command)}\n"
            f"  stderr: {completed.stderr.strip()}"
        )


def make_video(path: Path) -> Path:
    """Generate the `testsrc` test video. No external files or network are required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={VIDEO_SIZE}:rate={VIDEO_FPS}:duration={VIDEO_DURATION_S}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        what="test video generation",
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg finished, but the video file is missing or empty: {path}")
    return path


def make_scene(out_dir: Path, *, frames_json: Path, cameras: int) -> Path:
    """A synthetic COLMAP model under the frame names of a specific run.

    The names are taken from `frames.json` via the standard `--names-from` flag: putting them
    in by hand would mean forging the correspondence between the model and the selected frames.
    """
    _run(
        [
            sys.executable,
            str(MAKE_SCENE),
            "--out",
            str(out_dir),
            "--cameras",
            str(cameras),
            "--points",
            str(SCENE_POINTS),
            "--seed",
            str(SCENE_SEED),
            "--names-from",
            str(frames_json),
        ],
        what="synthetic scene generation",
    )
    return out_dir


def _copy_run(run_root: Path, target: Path) -> None:
    """Carry the needed parts of the run into the example directory.

    The target directory is recreated: the script is idempotent, a repeated run does not
    mix files of two runs.
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in COPIED_ENTRIES:
        source = run_root / name
        if not source.exists():
            raise RuntimeError(f"the run lacks the expected part '{name}': {source}")
        if source.is_dir():
            shutil.copytree(source, target / name)
        else:
            shutil.copy2(source, target / name)


def _dir_size_bytes(path: Path) -> int:
    """Total size of the directory's files."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _check_mark(target: Path) -> dict:
    """Make sure the precomputed-example mark is in place (A-07, FR-028)."""
    status_path = target / "normalized" / "status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if payload.get("precomputed_example") is not True:
        raise RuntimeError(
            f"{status_path} has no precomputed_example flag: an example without the mark "
            "could be mistaken for the result of a real reconstruction, so it is not saved"
        )
    return payload


def write_readme(target: Path, summary: dict) -> Path:
    """A warning next to the data: a person must not take it for a real result."""
    path = target / "README.md"
    text = f"""# Precomputed example run

**{PRECOMPUTED_WARNING}**

The directory was built by the script `assets/precomputed/make_sample.py` entirely from
nothing: the frames come from the synthetic `ffmpeg testsrc` video, the "reconstructor
result" from the analytically defined scene `assets/synthetic/make_scene.py`. There is no
footage of a real object here, no feature detector and no optimization, so these files
**cannot be used to judge the quality of a reconstruction**: they only check formats,
conventions and the operation of the viewer, export and report without a reconstruction
environment (FR-027).

The honesty mark: `normalized/status.json` has `precomputed_example: true`
(assumption A-07, requirement FR-028). The viewer and the report must show this
mark permanently.

## What is inside

| File | What it is |
| --- | --- |
| `manifest.json` | run information: input, frame transformation, selection parameters |
| `frames.json` | a record of every frame considered |
| `normalized/cameras.json` | camera poses and the declared conventions |
| `normalized/points.ply` | the coloured point cloud |
| `normalized/diagnostics.json` | run measures and the limits of conclusions |
| `normalized/status.json` | the result status and the example mark |
| `package/` | the portable set of frames (in a real run it goes to the reconstructor) |
| `result/` | raw upstream output; **not stored in git** (`.gitignore`) |

## Numbers of this example

- `run_id`: `{summary["run_id"]}`
- status: `{summary["status"]}`
- frames selected: {summary["selected_frames"]}
- cameras registered: {summary["cameras_registered"]}
- points: {summary["points_valid"]} valid of {summary["points_raw"]} in the model

## How to rebuild

```
python assets/precomputed/make_sample.py
```

The script is idempotent: the directory is recreated entirely. No network, GPU or manual
steps are needed, `ffmpeg` in PATH is enough.
"""
    path.write_text(text, encoding="utf-8")
    return path


def build_sample(target: Path = SAMPLE_DIR) -> dict:
    """Build the whole example and return a summary.

    The entire run is executed in a temporary directory. The target directory is touched only
    after result ingest succeeded: no half-empty example is left.
    """
    _require_tool("ffmpeg")
    _require_tool("ffprobe")
    if not MAKE_SCENE.is_file():
        raise RuntimeError(f"the synthetic scene generator is missing: {MAKE_SCENE}")

    # Import inside the function: first check the environment, then touch the package.
    from v3d.ingest import run_ingest
    from v3d.prepare import run_prepare

    with tempfile.TemporaryDirectory(prefix="v3d-sample-") as tmp:
        work = Path(tmp)
        video = make_video(work / "sample.mp4")
        # The manifest records the video path exactly as given. Passing a bare file name
        # (relative to the working directory) keeps temporary absolute paths out of it.
        with contextlib.chdir(work):
            prepared = run_prepare(
                Path(video.name),
                out_root=work / "runs",
                budget=FRAME_BUDGET,
                long_side=LONG_SIDE_PX,
                jpeg_quality=JPEG_QUALITY,
            )
        selected = [r for r in prepared.records if r.selected]
        scene = make_scene(
            work / "scene",
            frames_json=prepared.layout.frames,
            cameras=len(selected),
        )
        ingested = run_ingest(prepared.layout.root, source=scene, precomputed_example=True)
        summary = {
            "run_id": ingested.run_id,
            "status": ingested.status.value,
            "selected_frames": ingested.selected_frames,
            "cameras_registered": ingested.cameras_registered,
            "points_raw": ingested.points_raw,
            "points_valid": ingested.points_valid,
            "banner": ingested.banner,
        }
        _copy_run(prepared.layout.root, target)

    _check_mark(target)
    write_readme(target, summary)
    summary["target"] = target
    summary["size_bytes"] = _dir_size_bytes(target)
    return summary


def summary_lines(summary: dict) -> list[str]:
    """Summary lines for the terminal: facts and the mandatory warning."""
    size_mb = summary["size_bytes"] / (1024 * 1024)
    return [
        f"precomputed example created: {summary['target']}",
        f"  run_id: {summary['run_id']}",
        f"  status: {summary['banner']}",
        f"  frames selected: {summary['selected_frames']}",
        f"  cameras registered: {summary['cameras_registered']}",
        f"  points: {summary['points_valid']} valid of {summary['points_raw']} in the model",
        f"  directory size: {size_mb:.2f} MB",
        "  mark precomputed_example: true (A-07, FR-028)",
        PRECOMPUTED_WARNING,
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. No arguments: the example is built in one way."""
    if argv:
        raise SystemExit("the script takes no arguments")
    summary = build_sample()
    for line in summary_lines(summary):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
