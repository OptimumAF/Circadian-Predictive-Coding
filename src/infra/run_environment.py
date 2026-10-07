"""Capture local execution provenance for a versioned result bundle.

Inputs are a repository path. Outputs are Git, dependency, and CPU facts
collected before training. This module does not train, select metrics,
inspect evaluation roles, or write experiment files.
"""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess
import sys

from src.core.run_manifest import RunEnvironment


def _git_output(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=10,
    )
    return completed.stdout


def _hash_untracked_file(repository: Path, name: bytes) -> bytes:
    relative = Path(os.fsdecode(name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Git returned an unsafe untracked path")
    candidate = repository / relative
    digest = sha256()
    if candidate.is_symlink():
        digest.update(b"symlink\0")
        digest.update(os.fsencode(os.readlink(candidate)))
    elif candidate.is_file():
        digest.update(b"file\0")
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    else:
        raise ValueError("untracked source file vanished during provenance capture")
    return digest.digest()


def _capture_source(repository: Path) -> dict[str, object]:
    try:
        commit = _git_output(repository, "rev-parse", "HEAD").decode("ascii").strip()
        status = _git_output(repository, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        tracked_diff = _git_output(
            repository, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD"
        )
        untracked = tuple(
            sorted(
                item
                for item in _git_output(
                    repository, "ls-files", "--others", "--exclude-standard", "-z"
                ).split(b"\0")
                if item
            )
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, UnicodeError):
        return {
            "commit_sha": None,
            "dirty": None,
            "status_sha256": None,
            "tracked_diff_sha256": None,
            "workspace_sha256": None,
            "untracked_file_count": None,
            "unavailable_reason": "Git metadata is unavailable for this execution directory.",
        }
    workspace = sha256(b"circadian_workspace_v1\0" + commit.encode("ascii") + b"\0" + tracked_diff)
    for name in untracked:
        workspace.update(b"\0" + name + b"\0" + _hash_untracked_file(repository, name))
    return {
        "commit_sha": commit,
        "dirty": bool(status),
        "status_sha256": sha256(status).hexdigest(),
        "tracked_diff_sha256": sha256(tracked_diff).hexdigest(),
        "workspace_sha256": workspace.hexdigest(),
        "untracked_file_count": len(untracked),
        "unavailable_reason": None,
    }


def capture_run_environment(repository: str | Path) -> RunEnvironment:
    """Record source, required runtime versions, and CPU execution scope."""
    root = Path(repository)
    if not root.is_dir():
        raise ValueError(f"repository directory does not exist: {root}")
    processor = platform.processor().strip() or None
    return RunEnvironment(
        source=_capture_source(root),
        dependency_versions={"python": sys.version.split()[0], "numpy": version("numpy")},
        hardware={
            "system": platform.system() or "unknown",
            "release": platform.release() or "unknown",
            "machine": platform.machine() or "unknown",
            "processor": processor,
            "processor_unavailable_reason": (
                None if processor is not None else "Operating system did not report a CPU model."
            ),
            "logical_cpu_count": os.cpu_count(),
            "compute_device": "cpu",
        },
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
    )
