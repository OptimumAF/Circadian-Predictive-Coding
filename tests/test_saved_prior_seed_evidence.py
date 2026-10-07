from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from src.core.seed_stream_screening import EvidenceIdentity
from src.infra.saved_prior_seed_evidence import load_saved_seed_evidence
from seed_chronology_fixtures import saved_seed_corpus


def _write(root: Path, name: str, value: dict) -> EvidenceIdentity:
    raw = json.dumps(value, sort_keys=True).encode()
    (root / name).write_bytes(raw)
    return EvidenceIdentity(len(raw), sha256(raw).hexdigest())


def _inputs(root: Path) -> tuple[EvidenceIdentity, EvidenceIdentity]:
    report = saved_seed_corpus()
    return _write(root, "report.json", report), _write(
        root,
        "source.json",
        {
            "schema_id": "p67_complete_prior_seed_evidence_source_v1",
            "files": report["files"],
            "history": report["history"],
        },
    )


def test_should_read_complete_saved_bodies_without_scientific_or_source_execution(
    tmp_path: Path,
) -> None:
    report_id, source_id = _inputs(tmp_path)
    result = load_saved_seed_evidence(
        tmp_path,
        evidence_path="report.json",
        evidence_identity=report_id,
        source_path="source.json",
        source_identity=source_id,
    )
    assert result.report == saved_seed_corpus()
    assert result.evidence_identity == report_id and result.source_identity == source_id
    assert not result.report["fresh_roles_authorized"]


@pytest.mark.parametrize("which", ["report.json", "source.json"])
def test_should_reject_changed_whole_input_bytes(tmp_path: Path, which: str) -> None:
    report_id, source_id = _inputs(tmp_path)
    with (tmp_path / which).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="saved seed"):
        load_saved_seed_evidence(
            tmp_path,
            evidence_path="report.json",
            evidence_identity=report_id,
            source_path="source.json",
            source_identity=source_id,
        )


def test_should_reject_source_membership_that_differs_from_the_complete_report(
    tmp_path: Path,
) -> None:
    report_id, _ = _inputs(tmp_path)
    source = deepcopy(saved_seed_corpus())
    source["history"]["objects"] = []
    source_id = _write(
        tmp_path,
        "source.json",
        {
            "schema_id": "p67_complete_prior_seed_evidence_source_v1",
            "files": source["files"],
            "history": source["history"],
        },
    )
    with pytest.raises(ValueError, match="membership"):
        load_saved_seed_evidence(
            tmp_path,
            evidence_path="report.json",
            evidence_identity=report_id,
            source_path="source.json",
            source_identity=source_id,
        )


def test_should_reject_late_mutation_after_complete_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_id, source_id = _inputs(tmp_path)
    from src.infra import saved_prior_seed_evidence as boundary

    original = boundary.strict_seed_metadata_json

    def corrupt(raw: bytes) -> dict:
        value = original(raw)
        if value.get("schema_id") == "complete_retained_prior_seed_evidence_v1":
            with (tmp_path / "report.json").open("ab") as stream:
                stream.write(b" ")
        return value

    monkeypatch.setattr(boundary, "strict_seed_metadata_json", corrupt)
    with pytest.raises(ValueError, match="after"):
        load_saved_seed_evidence(
            tmp_path,
            evidence_path="report.json",
            evidence_identity=report_id,
            source_path="source.json",
            source_identity=source_id,
        )


def test_should_reject_paths_outside_the_requested_root(tmp_path: Path) -> None:
    report_id, source_id = _inputs(tmp_path)
    with pytest.raises(ValueError, match="path"):
        load_saved_seed_evidence(
            tmp_path,
            evidence_path="../report.json",
            evidence_identity=report_id,
            source_path="source.json",
            source_identity=source_id,
        )
