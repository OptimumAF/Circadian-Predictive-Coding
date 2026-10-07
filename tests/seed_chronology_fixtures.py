"""Complete fabricated saved corpus; no source, execution or release proof."""

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

from src.core.seed_declaration_context import audit_seed_metadata
from src.core.seed_source_evidence import inspect_seed_text


def saved_seed_corpus(metadata: Any = None) -> dict[str, Any]:
    raw = b"seed=7\n"
    digest = sha256(raw).hexdigest()
    commit = "b" * 40
    aliases = [f"git:{commit}:defaults.py", "retained.json"]
    files = [{"path": "retained.json", "byte_count": len(raw), "sha256": digest}]
    history = {
        "commits": [commit],
        "objects": [
            {
                "object_id": "a" * 40,
                "kind": "blob",
                "byte_count": len(raw),
                "sha256": digest,
                "commit_path_aliases": [[commit, "defaults.py"]],
            }
        ],
    }
    report = {
        "schema_id": "complete_retained_prior_seed_evidence_v1",
        "files": files,
        "history": history,
        "contents": [
            {
                "byte_count": len(raw),
                "sha256": digest,
                "aliases": aliases,
                "text": inspect_seed_text(raw.decode(), python_source=True),
                "json": asdict(
                    audit_seed_metadata(metadata if metadata is not None else {"seed": 7})
                ),
                "jsonl": None,
                "csv": None,
                "issues": [],
            }
        ],
        "physical_file_count": 1,
        "physical_byte_count": len(raw),
        "git_object_count": 1,
        "git_object_byte_count": len(raw),
        "distinct_complete_content_count": 1,
        "all_aliases_retained": True,
        "exact_duplicate_parsing_only": True,
        "limitations": "declarations_and_static_expressions_not_execution_or_release_proof;opaque_unparsed_symbolic_unrecorded_history_remains",
        "complete_prior_usage_acceptance": False,
        "fresh_roles_authorized": False,
        "original_p67_acceptance_complete": False,
    }
    return json.loads(json.dumps(report))
