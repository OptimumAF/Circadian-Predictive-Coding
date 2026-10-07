"""Read a whole prospectively pinned metadata corpus without scientific access.

Only runtime/cache directories and the new owned inventory output namespace
are excluded. Membership and full physical bytes are checked before and after.
Outputs retain unparsed files and unresolved fields; they never authorize a
fresh seed, role, model, training or score. No file is modified here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import Any

from src.core.seed_usage import collect_seed_usage, strict_seed_metadata_json


OWNED_OUTPUT_DIRECTORY = "artifacts/runs/p67-untouched-seed-usage"
OWNERSHIP_BYTES = b"p67_seed_usage_inventory_transaction_v1\n"
_CACHE_DIRECTORIES = {".git", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"}


@dataclass(frozen=True)
class SeedMetadataFile:
    path: str
    byte_count: int
    sha256: str


def _paths(root: Path) -> tuple[str, ...]:
    root = root.resolve(strict=True)
    paths = []
    for current, folders, files in os.walk(root, followlinks=False):
        directory = Path(current)
        output = directory / "p67-untouched-seed-usage"
        if output.relative_to(root).as_posix() == OWNED_OUTPUT_DIRECTORY and output.exists():
            owner = output / "ownership.txt"
            if (
                output.is_symlink()
                or not output.resolve(strict=True).is_relative_to(root)
                or owner.is_symlink()
                or not owner.resolve(strict=False).is_relative_to(root)
            ):
                raise ValueError("cannot exclude an external or symbolic owned inventory path")
            if not owner.is_file() or owner.read_bytes() != OWNERSHIP_BYTES:
                raise ValueError("cannot exclude an unowned seed inventory output directory")
        folders[:] = [
            name
            for name in folders
            if name not in _CACHE_DIRECTORIES
            and not (directory / name / "pyvenv.cfg").is_file()
            and (directory / name).relative_to(root).as_posix() != OWNED_OUTPUT_DIRECTORY
        ]
        for name in folders:
            child = directory / name
            if child.is_symlink() or not child.resolve(strict=True).is_relative_to(root):
                raise ValueError(
                    "seed metadata corpus cannot follow external or symbolic directories"
                )
        for name in files:
            if Path(name).suffix.lower() != ".json":
                continue
            path = directory / name
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("seed metadata corpus cannot follow external or symbolic files")
            paths.append(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


def _identity(root: Path, name: str) -> SeedMetadataFile:
    raw = (root / name).read_bytes()
    return SeedMetadataFile(name, len(raw), sha256(raw).hexdigest())


def freeze_seed_metadata_files(root: Path) -> tuple[SeedMetadataFile, ...]:
    """Freeze every current JSON file in the declared corpus, including failures."""
    root = root.resolve(strict=True)
    names = _paths(root)
    pinned = tuple(_identity(root, name) for name in names)
    if _paths(root) != names:
        raise ValueError("seed metadata corpus membership changed during freeze")
    return pinned


def _validate_pins(root: Path, files: tuple[SeedMetadataFile, ...]) -> None:
    if type(files) is not tuple or not files or any(type(f) is not SeedMetadataFile for f in files):
        raise ValueError("seed metadata requires its whole nonempty file tuple")
    if tuple(f.path for f in files) != _paths(root):
        raise ValueError("seed metadata requires the complete unchanged corpus membership")
    for file in files:
        if type(file.byte_count) is not int or file.byte_count < 0:
            raise ValueError("seed metadata byte count must be a nonnegative integer")
        if (
            type(file.sha256) is not str
            or len(file.sha256) != 64
            or any(c not in "0123456789abcdef" for c in file.sha256)
        ):
            raise ValueError("seed metadata digest must be lowercase SHA-256")


def read_seed_usage_inventory(root: Path, files: tuple[SeedMetadataFile, ...]) -> dict[str, Any]:
    """Complete bytes and membership, with no fresh-source authority from a scan."""
    root = root.resolve(strict=True)
    _validate_pins(root, files)
    rows: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for file in files:
        raw = (root / file.path).read_bytes()
        if SeedMetadataFile(file.path, len(raw), sha256(raw).hexdigest()) != file:
            raise ValueError(f"seed metadata file changed before parsing: {file.path}")
        try:
            usage = collect_seed_usage(strict_seed_metadata_json(raw))
            declarations = [asdict(d) for d in usage.declarations]
            unresolved = [asdict(d) for d in usage.unresolved]
            seeds.update(d.seed for d in usage.declarations)
            row = {
                "file": asdict(file),
                "parse_status": "parsed",
                "declarations": declarations,
                "unresolved": unresolved,
                "parse_error": None,
            }
        except ValueError as error:
            row = {
                "file": asdict(file),
                "parse_status": "unparsed",
                "declarations": [],
                "unresolved": [],
                "parse_error": str(error),
            }
        rows.append(row)
    _validate_pins(root, files)
    if tuple(_identity(root, file.path) for file in files) != files:
        raise ValueError("seed metadata full physical bytes changed after parsing")
    return {
        "schema_id": "complete_retained_json_seed_declarations_v1",
        "files": rows,
        "declared_seed_values": sorted(seeds),
        "file_count": len(rows),
        "physical_byte_count": sum(file.byte_count for file in files),
        "declaration_count": sum(len(row["declarations"]) for row in rows),
        "unresolved_declaration_count": sum(len(row["unresolved"]) for row in rows),
        "unparsed_file_count": sum(row["parse_status"] == "unparsed" for row in rows),
        "scope": "all_retained_JSON_metadata_in_checkout_including_failures_canonical_mappings_embedded_JSON_and_CLI_arguments",
        "limitations": "declarations_not_execution_proof_or_complete_unrecorded_historical_usage_text_factory_stream_and_role_audits_remain",
        "fresh_roles_authorized": False,
        "complete_prior_usage_acceptance": False,
        "original_p67_acceptance_complete": False,
    }
