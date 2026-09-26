"""Command-line skeleton of ``v3d``.

Command contract: specs/001-video-3d-mvp/contracts/cli.md
Exit codes: specs/001-video-3d-mvp/contracts/errors.md

Only argument parsing and mapping of exceptions to exit codes live here. The stage logic
lives in separate modules and is hooked up by their own tasks; until then the subcommands
honestly report that a stage is not implemented instead of pretending the work was done.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

from v3d.errors import ExitCode, V3dError

PROG = "v3d"
DEFAULT_OUT = "runs"
DEFAULT_SEED = 0

SUPPORTED_INPUT = "containers mp4, mov; video codecs h264, hevc; progressive scan"


def package_version() -> str:
    """Installed package version; outside an installation — a marker, not an invented number."""
    try:
        return version("v3d")
    except PackageNotFoundError:
        return "unknown (package not installed)"


def say(args: argparse.Namespace, message: str) -> None:
    """Normal output to stdout. Silent with ``--quiet``; errors are not affected."""
    if not getattr(args, "quiet", False):
        print(message)


# --- Subcommands: stubs -------------------------------------------------------------------
#
# Each one raises NotImplementedError with the number of the task that implements it.


def cmd_check(args: argparse.Namespace) -> int:
    """Check the video file properties before any processing.

    Only verifiable file properties are reported. Assumptions about the scene (the object
    is static, texture is sufficient, lighting is stable) are neither checked nor assessed (FR-006).
    """
    from pathlib import Path

    from v3d.probe import probe_video

    info = probe_video(Path(args.video))
    if getattr(args, "json", False):
        print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
        return ExitCode.OK

    say(args, f"file: {info.filename}")
    say(args, f"  container/codec: {info.container} / {info.video_codec}")
    say(args, f"  duration: {info.duration_s:.2f} s, {info.avg_fps:.3f} fps")
    say(args, f"  resolution: {info.width}x{info.height}, rotation: {info.rotation_deg}°")
    frames = info.nb_frames if info.nb_frames is not None else "not reported by the container"
    say(args, f"  frames: {frames}")
    say(args, f"  sha256: {info.sha256[:16]}…")
    say(args, "input is supported")
    say(args, "not checked by the system: object being static, texture, lighting stability")
    return ExitCode.OK


def cmd_prepare(args: argparse.Namespace) -> int:
    """Extract and select frames, build the ``package/`` transfer package."""
    from pathlib import Path

    from v3d.config import DEFAULTS
    from v3d.prepare import run_prepare

    result = run_prepare(
        Path(args.video),
        out_root=Path(getattr(args, "out", None) or DEFAULT_OUT),
        budget=getattr(args, "budget", None) or DEFAULTS.frame_budget,
        selector=getattr(args, "selector", None) or "uniform",
        seed=getattr(args, "seed", None)
        if getattr(args, "seed", None) is not None
        else DEFAULTS.seed,
        long_side=getattr(args, "long_side", None) or DEFAULTS.long_side_px,
        force=getattr(args, "force", False),
    )
    selected = [r for r in result.records if r.selected]
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "run_dir": str(result.layout.root),
                    "candidates": result.candidates_total,
                    "selected": len(selected),
                    "package": result.package_summary,
                    "warnings": result.warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return ExitCode.OK

    say(args, f"run: {result.run_id}")
    say(args, f"  directory: {result.layout.root}")
    total = result.frames_total_in_video
    say(args, f"  frames in video: {total if total is not None else 'not reported by container'}")
    say(args, f"  candidates considered: {result.candidates_total}")
    say(args, f"  frames selected: {len(selected)}")
    reasons: dict[str, int] = {}
    for rec in result.records:
        if rec.reject_reason:
            reasons[rec.reject_reason] = reasons.get(rec.reject_reason, 0) + 1
    if reasons:
        pairs = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))
        say(args, f"  rejection reasons: {pairs}")
    say(args, f"  transfer package: {result.layout.package}")
    for warn in result.warnings:
        say(args, f"  warning ({warn['code']}): {warn['message']}")
    say(args, "next: transfer package/ to the GPU machine, see RUN_ON_GPU.txt inside the package")
    return ExitCode.OK


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest the result from the GPU machine and convert it to the normalized representation."""
    from pathlib import Path

    from v3d.ingest import ingest_summary_lines, run_ingest

    source = getattr(args, "from_path", None)
    result = run_ingest(
        Path(args.run_dir),
        source=Path(source) if source else None,
        precomputed_example=getattr(args, "precomputed_example", False),
    )
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status.value,
                    "cameras_registered": result.cameras_registered,
                    "selected_frames": result.selected_frames,
                    "points_raw": result.points_raw,
                    "points_valid": result.points_valid,
                    "normalized": str(result.layout.normalized),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return ExitCode.OK

    for line in ingest_summary_lines(result):
        say(args, line)
    say(args, "next: v3d view <run directory> — view the result on the Mac")
    return ExitCode.OK


def cmd_view(args: argparse.Namespace) -> int:
    """Open the normalized result: point cloud, camera poses, details and status."""
    from pathlib import Path

    from v3d.viewer import build_recording

    save_to = getattr(args, "save", None)
    summary = build_recording(
        Path(args.run_dir),
        save_to=Path(save_to) if save_to else None,
        spawn=not getattr(args, "no_window", False),
    )
    if getattr(args, "json", False):
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return ExitCode.OK

    say(args, f"run: {summary['run_id']}")
    say(args, f"  {summary['banner']}")
    say(args, f"  points: {summary['points']}, cameras: {summary['cameras']}")
    for mark in summary.get("marks", []):
        say(args, f"  mark: {mark}")
    if summary.get("saved_to"):
        say(args, f"  view recording: {summary['saved_to']}")
    return ExitCode.OK


def cmd_export(args: argparse.Namespace) -> int:
    """Export the displayed point cloud to PLY."""
    from pathlib import Path

    from v3d.export import export_point_cloud
    from v3d.ply_io import read_ply

    out = getattr(args, "out", None)
    path = export_point_cloud(
        Path(args.run_dir),
        out=Path(out) if out else None,
        force=getattr(args, "force", False),
    )
    points, _, comments = read_ply(path)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {"path": str(path), "points": len(points), "run_id": comments.get("run_id")},
                ensure_ascii=False,
                indent=2,
            )
        )
        return ExitCode.OK

    say(args, f"points exported: {len(points)}")
    say(args, f"  file: {path}")
    if "UNUSABLE" in path.name:
        say(args, "  attention: the result is marked unusable, the file was saved due to --force")
    return ExitCode.OK


def cmd_report(args: argparse.Namespace) -> int:
    """Build a human-readable report on the run."""
    from pathlib import Path

    from v3d.report import build_report

    out = getattr(args, "out", None)
    path = build_report(Path(args.run_dir), out=Path(out) if out else None)
    if getattr(args, "json", False):
        print(json.dumps({"path": str(path)}, ensure_ascii=False, indent=2))
        return ExitCode.OK
    say(args, f"report: {path}")
    return ExitCode.OK


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two runs of experiment P2 under pre-fixed conditions."""
    raise NotImplementedError("the comparison stage is not implemented yet (task T051)")


# --- Argument parsing ----------------------------------------------------------------------


def _common_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Common flags. On subcommands the default is suppressed to keep the shared value."""
    default = argparse.SUPPRESS if suppress else False
    parser.add_argument(
        "--json", action="store_true", default=default, help="machine-readable output"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=default, help="suppress normal output"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=default,
        help="allow an action that would otherwise be rejected (overwriting a run directory)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Argument parser per contracts/cli.md."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Video-to-3D: a colored point cloud and camera poses from an RGB video "
            "of a static object. The heavy stage is run manually on the GPU machine "
            "and is not part of these commands."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {package_version()}")
    _common_flags(parser, suppress=False)

    common = argparse.ArgumentParser(add_help=False)
    _common_flags(common, suppress=True)

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    p_check = subparsers.add_parser(
        "check",
        parents=[common],
        help="check that a video file is usable",
        description=f"Verifiable file properties before processing. Supported: {SUPPORTED_INPUT}.",
    )
    p_check.add_argument("video", help="path to the video file")
    p_check.set_defaults(func=cmd_check)

    p_prepare = subparsers.add_parser(
        "prepare",
        parents=[common],
        help="extract and select frames, build the transfer package",
    )
    p_prepare.add_argument("video", help="path to the video file")
    p_prepare.add_argument(
        "--budget",
        type=int,
        default=None,
        help="frame budget for selection; the default is assigned after M1 (T045)",
    )
    p_prepare.add_argument(
        "--selector",
        choices=("uniform", "quality"),
        default="uniform",
        help="frame selection method (default uniform)",
    )
    p_prepare.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="selection seed (default 0)"
    )
    p_prepare.add_argument(
        "--long-side",
        type=int,
        default=None,
        help="long side of the saved frame in pixels; without the flag no resize is applied",
    )
    p_prepare.add_argument(
        "--out", default=DEFAULT_OUT, help=f"root of run directories (default {DEFAULT_OUT})"
    )
    p_prepare.set_defaults(func=cmd_prepare)

    p_ingest = subparsers.add_parser(
        "ingest",
        parents=[common],
        help="ingest the result from the GPU machine and normalize it",
    )
    p_ingest.add_argument("run_dir", help="run directory runs/<run_id>")
    p_ingest.add_argument(
        "--from",
        dest="from_path",
        default=None,
        help="path to the returned result; without the flag <run_dir>/result is used",
    )
    p_ingest.add_argument(
        "--precomputed-example",
        action="store_true",
        help="mark the run as a precomputed example (for interface development)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_view = subparsers.add_parser(
        "view", parents=[common], help="open the normalized result in the viewer"
    )
    p_view.add_argument("run_dir", help="run directory runs/<run_id>")
    p_view.add_argument("--save", default=None, help="save the view recording to an .rrd file")
    p_view.add_argument(
        "--no-window",
        action="store_true",
        help="do not open a window (for checks and saving the recording)",
    )
    p_view.set_defaults(func=cmd_view)

    p_export = subparsers.add_parser(
        "export", parents=[common], help="export the displayed point cloud"
    )
    p_export.add_argument("run_dir", help="run directory runs/<run_id>")
    p_export.add_argument(
        "--format", choices=("ply",), default="ply", help="export format (default ply)"
    )
    p_export.add_argument(
        "--out", default=None, help="export file; without the flag — <run_dir>/export/object.ply"
    )
    p_export.set_defaults(func=cmd_export)

    p_report = subparsers.add_parser("report", parents=[common], help="build report.md for a run")
    p_report.add_argument("run_dir", help="run directory runs/<run_id>")
    p_report.add_argument(
        "--out", default=None, help="report file; without the flag — <run_dir>/report.md"
    )
    p_report.set_defaults(func=cmd_report)

    p_compare = subparsers.add_parser(
        "compare",
        parents=[common],
        help="compare two runs of experiment P2",
        description=(
            "The comparison is rejected if the branches differ in anything other than the "
            "frame selection method."
        ),
    )
    p_compare.add_argument("run_dir_a", help="directory of the first run")
    p_compare.add_argument("run_dir_b", help="directory of the second run")
    p_compare.add_argument(
        "--conditions", required=True, help="file of pre-fixed conditions conditions.json"
    )
    p_compare.add_argument(
        "--init",
        action="store_true",
        help="create the conditions file instead of comparing; changed conditions = new experiment",
    )
    p_compare.set_defaults(func=cmd_compare)

    return parser


# --- Error output --------------------------------------------------------------------------


def _fail(message: str, code: ExitCode, *, details: str | None, as_json: bool) -> int:
    payload = {"error": message, "exit_code": int(code), "details": details}
    if as_json:
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    else:
        print(f"error: {message}", file=sys.stderr)
        if details:
            print(details, file=sys.stderr)
    return int(code)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns only codes from ``ExitCode``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    as_json = bool(getattr(args, "json", False))

    if getattr(args, "func", None) is None:
        parser.print_help(sys.stderr)
        return int(ExitCode.UNEXPECTED)

    try:
        return int(args.func(args) or ExitCode.OK)
    except V3dError as err:
        return _fail(err.message, err.exit_code, details=err.details, as_json=as_json)
    except NotImplementedError as err:
        return _fail(
            str(err),
            ExitCode.UNEXPECTED,
            details="the command is declared by the contract, but its stage is not written yet",
            as_json=as_json,
        )
    except KeyboardInterrupt:
        return _fail(
            "execution interrupted by the user",
            ExitCode.UNEXPECTED,
            details=None,
            as_json=as_json,
        )
    except Exception as err:  # unexpected: code 1, no cause is guessed
        return _fail(
            f"unexpected error: {err}",
            ExitCode.UNEXPECTED,
            details=type(err).__name__,
            as_json=as_json,
        )


if __name__ == "__main__":
    raise SystemExit(main())
