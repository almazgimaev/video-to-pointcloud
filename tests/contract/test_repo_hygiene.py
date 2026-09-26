"""Repository hygiene: English-only content and no author infrastructure details.

Constitution v3.1.0 requires that everything in the repository is written in English and that
public materials do not reveal the author's infrastructure (host names, user names, absolute
home paths, or how the GPU machine is used). These tests scan every tracked and untracked,
non-ignored text file.

Private patterns (host alias, user name, remote directory) are never hardcoded here: they are
read from the local, git-excluded ``OPERATIONS.local.md`` when it exists.
"""

from __future__ import annotations

import re
import subprocess
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
LOCAL_NOTES = REPO / "OPERATIONS.local.md"

# Letters are allowed only from these Unicode groups: Latin for text; Greek and modifier letters
# (for example the superscript T in a transpose) for math notation.
ALLOWED_SCRIPTS = ("LATIN", "GREEK", "MODIFIER LETTER")

# Generic patterns that must never appear in repository files.
GENERIC_PRIVATE_PATTERNS = (
    re.compile(r"/Users/[A-Za-z0-9_.-]+"),
    re.compile(r"/home/[A-Za-z0-9_.-]+"),
    re.compile(r"/var/folders/"),
    re.compile(r"shared (GPU )?machine", re.IGNORECASE),
    re.compile(r"work(ing)? (GPU )?machine", re.IGNORECASE),
    re.compile(r"other users'? processes", re.IGNORECASE),
    re.compile(r"owner'?s? permission", re.IGNORECASE),
)

BEGIN = "<!-- PRIVATE-PATTERNS-BEGIN -->"
END = "<!-- PRIVATE-PATTERNS-END -->"


def _repo_files() -> list[Path]:
    """Tracked plus untracked-but-not-ignored files, so new files are checked before commit."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout: repository hygiene cannot be checked")
    files = []
    for line in sorted(set(out.splitlines())):
        path = (REPO / line).resolve()
        if path == THIS_FILE or not path.is_file():
            continue
        files.append(path)
    return files


def _read_text(path: Path) -> str | None:
    """Text content, or None for binary files."""
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _private_patterns() -> list[str]:
    if not LOCAL_NOTES.exists():
        return []
    text = LOCAL_NOTES.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        return []
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    return [line.strip() for line in block.splitlines() if line.strip()]


def _violations(predicate) -> list[str]:
    found = []
    for path in _repo_files():
        text = _read_text(path)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            hit = predicate(line)
            if hit:
                found.append(f"{path.relative_to(REPO)}:{number}: {hit}")
    return found


def _foreign_letter(line: str) -> str | None:
    """First letter in the line that belongs to a script other than Latin or Greek."""
    for char in line:
        if char.isalpha() and not unicodedata.name(char, "").startswith(ALLOWED_SCRIPTS):
            return char
    return None


def test_repository_text_is_english_only() -> None:
    """Every repository file is written in English (constitution v3.1.0)."""

    def predicate(line: str) -> str | None:
        return line.strip()[:80] if _foreign_letter(line) else None

    violations = _violations(predicate)
    assert not violations, "non-English text found:\n" + "\n".join(violations[:30])


def test_repository_contains_no_generic_private_details() -> None:
    """No absolute home paths and no wording about how the GPU machine is shared."""

    def predicate(line: str) -> str | None:
        for pattern in GENERIC_PRIVATE_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(0)
        return None

    violations = _violations(predicate)
    assert not violations, "private details found:\n" + "\n".join(violations[:30])


def test_repository_contains_no_author_private_patterns() -> None:
    """Host alias, user name and remote directory from the local notes never leak."""
    patterns = _private_patterns()
    if not patterns:
        pytest.skip("OPERATIONS.local.md with private patterns is not present on this machine")

    def predicate(line: str) -> str | None:
        for pattern in patterns:
            if pattern in line:
                return "<private pattern>"
        return None

    violations = _violations(predicate)
    assert not violations, "private patterns found:\n" + "\n".join(violations[:30])
