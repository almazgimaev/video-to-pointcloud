"""Run identity: code_id, inputs digest and run_id.

Data model: specs/001-video-3d-mvp/data-model.md §2

    run_id = <UTC YYYYMMDD-HHMMSS>-<inputs_digest[:8]>
    inputs_digest = sha256(video_sha256 || canonical_json(params) || code_id || seed)

Two runs with the same inputs_digest are a reproduction of the same run; two with
different digests cannot end up in the same directory.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Hash of file contents. The only reliable identifier of the input video."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> str:
    """Canonical representation of parameters: sorted keys, no extra whitespace.

    Needed so that parameters equal in meaning give the same digest regardless of
    key order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class CodeId:
    """Code version that takes part in the inputs digest.

    In a git repository — the commit. Outside git — sha256 over the v3d package contents,
    so that an identifier exists anyway.
    """

    value: str
    source: str  # "git" | "package_hash"
    working_tree_dirty: bool
    dirty_files: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source": self.source,
            "working_tree_dirty": self.working_tree_dirty,
            "dirty_files": list(self.dirty_files),
        }


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=15
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return out.stdout.strip()


def hash_package(package_dir: Path) -> str:
    """Hash of package sources: sorted walk over *.py with names and contents."""
    h = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        h.update(str(path.relative_to(package_dir)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def compute_code_id(repo_root: Path | None = None, package_dir: Path | None = None) -> CodeId:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    package_dir = package_dir or Path(__file__).resolve().parent

    commit = _git(["rev-parse", "HEAD"], repo_root)
    if commit:
        status = _git(["status", "--porcelain"], repo_root) or ""
        dirty_files = tuple(line[3:] for line in status.splitlines() if line.strip())
        return CodeId(
            value=commit,
            source="git",
            working_tree_dirty=bool(dirty_files),
            dirty_files=dirty_files,
        )

    # Outside git, a code identifier must still exist (Principle II).
    return CodeId(value=hash_package(package_dir), source="package_hash", working_tree_dirty=False)


def inputs_digest(*, video_sha256: str, params: dict, code_id: str, seed: int) -> str:
    h = hashlib.sha256()
    h.update(video_sha256.encode())
    h.update(canonical_json(params).encode())
    h.update(code_id.encode())
    h.update(str(seed).encode())
    return h.hexdigest()


def make_run_id(digest: str, when: datetime | None = None) -> str:
    when = when or datetime.now(UTC)
    return f"{when.strftime('%Y%m%d-%H%M%S')}-{digest[:8]}"
