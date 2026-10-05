"""Inspect all retained physical and local Git seed evidence without science.

Inputs are complete prospective file/history pins. Outputs preserve every
physical alias, whole-byte identity, semantic declaration and source/text
reference. Exact duplicate bodies are parsed once with all aliases retained.
Opaque assets and parse/symbolic/history uncertainty remain explicit. No data
or checkpoint is deserialized, and no source, model, RNG or score is invoked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import subprocess
import tokenize
from typing import Any

from src.core.seed_declaration_context import audit_seed_metadata
from src.core.seed_source_evidence import inspect_seed_csv, inspect_seed_text
from src.core.seed_usage import strict_seed_metadata_json


OWNED_EVIDENCE_DIRECTORY = "artifacts/runs/p67-untouched-seed-usage/prior-evidence"
OWNERSHIP_BYTES = b"p67_complete_prior_seed_evidence_transaction_v1\n"
_CACHE_DIRECTORIES = {".git", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"}
_OPAQUE_SUFFIXES = {".bin", ".ckpt", ".checkpoint", ".gif", ".png", ".gz"}


@dataclass(frozen=True)
class PriorSeedFile:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class GitSeedObject:
    object_id: str
    kind: str
    byte_count: int
    sha256: str
    commit_path_aliases: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SeedGitHistory:
    commits: tuple[str, ...]
    objects: tuple[GitSeedObject, ...]


def _paths(root: Path) -> tuple[str, ...]:
    paths = []
    for current, folders, files in os.walk(root, followlinks=False):
        directory = Path(current)
        retained = []
        for name in folders:
            child = directory / name
            if child.is_symlink() or not child.resolve(strict=True).is_relative_to(root):
                raise ValueError(
                    "prior seed evidence cannot follow external or symbolic directories"
                )
            if child.relative_to(root).as_posix() == OWNED_EVIDENCE_DIRECTORY:
                owner = child / "ownership.txt"
                if (
                    owner.is_symlink()
                    or not owner.resolve(strict=False).is_relative_to(root)
                    or not owner.is_file()
                    or owner.read_bytes() != OWNERSHIP_BYTES
                ):
                    raise ValueError(
                        "prior seed evidence cannot exclude an unowned or symbolic transaction"
                    )
            elif name not in _CACHE_DIRECTORIES and not (child / "pyvenv.cfg").is_file():
                retained.append(name)
        folders[:] = retained
        for name in files:
            path = directory / name
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("prior seed evidence cannot follow external or symbolic files")
            paths.append(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


def _file_identity(root: Path, name: str) -> PriorSeedFile:
    digest, count = sha256(), 0
    with (root / name).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(chunk)
            digest.update(chunk)
    return PriorSeedFile(name, count, digest.hexdigest())


def freeze_prior_seed_files(root: Path) -> tuple[PriorSeedFile, ...]:
    root = root.resolve(strict=True)
    names = _paths(root)
    files = tuple(_file_identity(root, name) for name in names)
    if _paths(root) != names:
        raise ValueError("prior seed evidence membership changed during freeze")
    return files


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot read complete local Git seed evidence: {args[0]}") from error


def _git_membership(root: Path) -> tuple[tuple[str, ...], tuple[tuple[str, str, int], ...]]:
    commits = tuple(sorted(_git(root, "rev-list", "--all").decode("ascii").splitlines()))
    objects = tuple(
        sorted(
            (name, kind, int(size))
            for name, kind, size in (
                line.split()
                for line in _git(
                    root,
                    "cat-file",
                    "--batch-all-objects",
                    "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                )
                .decode("ascii")
                .splitlines()
            )
        )
    )
    return commits, objects


def _git_bodies(root: Path, objects: tuple[tuple[str, str, int], ...]) -> dict[str, bytes]:
    if not objects:
        return {}
    raw = _git(
        root,
        "cat-file",
        "--batch",
        input_bytes=("\n".join(name for name, _, _ in objects) + "\n").encode("ascii"),
    )
    result, position = {}, 0
    for wanted, kind, count in objects:
        end = raw.index(b"\n", position)
        observed, actual_kind, actual_count = raw[position:end].decode("ascii").split()
        if (observed, actual_kind, int(actual_count)) != (wanted, kind, count):
            raise ValueError("local Git object batch differs from complete frozen membership")
        position = end + 1
        result[wanted] = raw[position : position + count]
        position += count
        if raw[position : position + 1] != b"\n":
            raise ValueError("local Git object batch is truncated")
        position += 1
    if position != len(raw):
        raise ValueError("local Git object batch has unexpected trailing data")
    return result


def _git_aliases(root: Path, commits: tuple[str, ...]) -> dict[str, tuple[tuple[str, str], ...]]:
    aliases: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for commit in commits:
        for row in _git(root, "ls-tree", "-r", "--full-tree", "-z", commit).split(b"\0"):
            if not row:
                continue
            entry, path = row.split(b"\t", 1)
            _, kind, object_id = entry.decode("ascii").split()
            if kind == "blob":
                aliases[object_id].append((commit, os.fsdecode(path)))
    return {name: tuple(sorted(rows)) for name, rows in aliases.items()}


def _read_git(root: Path) -> tuple[SeedGitHistory, dict[str, bytes]]:
    commits, objects = _git_membership(root)
    bodies = _git_bodies(root, objects)
    aliases = _git_aliases(root, commits)
    pins = tuple(
        GitSeedObject(name, kind, count, sha256(bodies[name]).hexdigest(), aliases.get(name, ()))
        for name, kind, count in objects
    )
    if _git_membership(root) != (commits, objects):
        raise ValueError("local Git evidence membership changed while reading")
    return SeedGitHistory(commits, pins), bodies


def freeze_seed_git_history(root: Path) -> SeedGitHistory:
    return _read_git(root.resolve(strict=True))[0]


def _decoded_text(raw: bytes, paths: tuple[str, ...]) -> str:
    encoding = (
        tokenize.detect_encoding(BytesIO(raw).readline)[0]
        if any(Path(p).suffix.lower() == ".py" for p in paths)
        else "utf-8-sig"
    )
    return raw.decode(encoding)


def _json_lines(text: str) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            rows.append(
                {"line": index, "parse_status": "blank", "metadata": None, "parse_error": None}
            )
            continue
        try:
            body = asdict(audit_seed_metadata(strict_seed_metadata_json(line)))
            rows.append(
                {"line": index, "parse_status": "parsed", "metadata": body, "parse_error": None}
            )
        except ValueError as error:
            rows.append(
                {
                    "line": index,
                    "parse_status": "unparsed",
                    "metadata": None,
                    "parse_error": str(error),
                }
            )
    return rows


def _inspect(raw: bytes, paths: tuple[str, ...]) -> dict[str, Any]:
    suffixes = {Path(path).suffix.lower() for path in paths}
    opaque = bool(suffixes) and suffixes.issubset(_OPAQUE_SUFFIXES)
    body: dict[str, Any] = {
        "byte_count": len(raw),
        "sha256": sha256(raw).hexdigest(),
        "aliases": list(paths),
        "text": None,
        "json": None,
        "jsonl": None,
        "csv": None,
        "issues": [],
    }
    if opaque:
        body["issues"].append(
            {
                "kind": "opaque_asset_or_checkpoint",
                "reason": "whole bytes bound; no deserialization or absence inference",
            }
        )
        return body
    try:
        text = _decoded_text(raw, paths)
    except (UnicodeError, SyntaxError, LookupError) as error:
        body["issues"].append({"kind": "opaque_or_undecodable", "reason": str(error)})
        return body
    if "\x00" in text:
        body["issues"].append(
            {
                "kind": "opaque_binary",
                "reason": "whole bytes bound; binary is not executed or deserialized",
            }
        )
        return body
    unknown_blob = any(path.startswith("git-unreachable:") for path in paths)
    body["text"] = inspect_seed_text(text, python_source=".py" in suffixes or unknown_blob)
    if unknown_blob:
        body["issues"].append(
            {
                "kind": "unreachable_blob_without_path",
                "reason": "format and past usage unrecorded; source/text inspection is not execution proof",
            }
        )
    if (
        ".json" in suffixes
        or text.lstrip().startswith(("{", "["))
        and not suffixes.intersection({".py", ".csv", ".jsonl"})
    ):
        try:
            body["json"] = asdict(audit_seed_metadata(strict_seed_metadata_json(text)))
        except ValueError as error:
            body["issues"].append({"kind": "unparsed_json", "reason": str(error)})
    if ".jsonl" in suffixes:
        body["jsonl"] = _json_lines(text)
    if ".csv" in suffixes:
        try:
            body["csv"] = inspect_seed_csv(text)
        except ValueError as error:
            body["issues"].append({"kind": "unparsed_csv", "reason": str(error)})
    return body


def _validate_file_pins(root: Path, files: tuple[PriorSeedFile, ...]) -> None:
    if (
        type(files) is not tuple
        or not files
        or any(type(file) is not PriorSeedFile for file in files)
    ):
        raise ValueError("prior seed evidence requires its whole nonempty file tuple")
    for file in files:
        if (
            type(file.path) is not str
            or type(file.byte_count) is not int
            or file.byte_count < 0
            or type(file.sha256) is not str
            or len(file.sha256) != 64
            or any(c not in "0123456789abcdef" for c in file.sha256)
        ):
            raise ValueError("prior seed evidence requires valid exact file identities")
    if tuple(file.path for file in files) != _paths(root):
        raise ValueError("prior seed evidence requires complete unchanged file membership")


def _validate_history(history: SeedGitHistory) -> None:
    if (
        type(history) is not SeedGitHistory
        or type(history.commits) is not tuple
        or type(history.objects) is not tuple
    ):
        raise ValueError("prior seed evidence requires its complete typed Git history")
    for obj in history.objects:
        if (
            type(obj) is not GitSeedObject
            or type(obj.byte_count) is not int
            or obj.byte_count < 0
            or type(obj.sha256) is not str
            or len(obj.sha256) != 64
            or any(c not in "0123456789abcdef" for c in obj.sha256)
        ):
            raise ValueError("prior seed evidence requires valid exact Git identities")


def read_prior_seed_evidence(
    root: Path, files: tuple[PriorSeedFile, ...], history: SeedGitHistory
) -> dict[str, Any]:
    """Read the entire frozen checkout/history; keep all aliases and uncertainty."""
    root = root.resolve(strict=True)
    _validate_file_pins(root, files)
    _validate_history(history)
    actual_history, objects = _read_git(root)
    if actual_history != history:
        raise ValueError("prior seed evidence Git identities, aliases or membership changed")
    groups: dict[str, list[str]] = defaultdict(list)
    representatives: dict[str, Path | str] = {}
    for file in files:
        if _file_identity(root, file.path) != file:
            raise ValueError(f"prior seed evidence changed before parsing: {file.path}")
        groups[file.sha256].append(file.path)
        representatives.setdefault(file.sha256, root / file.path)
    for obj in history.objects:
        aliases = [f"git:{commit}:{path}" for commit, path in obj.commit_path_aliases]
        fallback = (
            f"git-unreachable:{obj.object_id}"
            if obj.kind == "blob"
            else f"git-metadata:{obj.kind}:{obj.object_id}"
        )
        groups[obj.sha256].extend(aliases or [fallback])
        representatives.setdefault(obj.sha256, obj.object_id)
    # Whole duplicate identities let us load one payload at a time, preserving
    # every alias without holding multiple gigabytes of raw checkout bytes.
    rows = []
    for digest in sorted(groups):
        representative = representatives[digest]
        raw = (
            representative.read_bytes()
            if isinstance(representative, Path)
            else objects[representative]
        )
        if sha256(raw).hexdigest() != digest:
            raise ValueError("prior seed evidence representative changed before parsing")
        rows.append(_inspect(raw, tuple(sorted(groups[digest]))))
    if freeze_prior_seed_files(root) != files or freeze_seed_git_history(root) != history:
        raise ValueError("prior seed evidence whole physical/history pins changed after parsing")
    return {
        "schema_id": "complete_retained_prior_seed_evidence_v1",
        "files": [asdict(file) for file in files],
        "history": asdict(history),
        "contents": rows,
        "physical_file_count": len(files),
        "physical_byte_count": sum(file.byte_count for file in files),
        "git_object_count": len(history.objects),
        "git_object_byte_count": sum(obj.byte_count for obj in history.objects),
        "distinct_complete_content_count": len(rows),
        "all_aliases_retained": True,
        "exact_duplicate_parsing_only": True,
        "limitations": "declarations_and_static_expressions_not_execution_or_release_proof;opaque_unparsed_symbolic_unrecorded_history_remains",
        "complete_prior_usage_acceptance": False,
        "fresh_roles_authorized": False,
        "original_p67_acceptance_complete": False,
    }
