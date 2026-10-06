"""Whole file boundary behavior; synthetic inputs supply no original proof."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from test_prospective_confirmation_evidence import saved_contract_inputs as saved_contract_inputs
from src.core.seed_stream_screening import EvidenceIdentity
from src.infra import prospective_confirmation_inputs as boundary


@pytest.fixture
def whole_inputs(
    tmp_path: Path, saved_contract_inputs: tuple[dict[str, Any], dict[str, Any]]
) -> tuple[Path, dict[str, str], dict[str, EvidenceIdentity]]:
    inputs, _ = saved_contract_inputs
    paths = {name: name + ".json" for name in boundary.INPUT_NAMES}
    identities: dict[str, EvidenceIdentity] = {}
    # Chronicle is written first because both complete witness bodies link its bytes.
    for name in (
        "chronology",
        "precision",
        "precision_repeat",
        "canonical_witness",
        "repeat_witness",
    ):
        if name.endswith("_witness"):
            inputs[name]["complete_prior_ledger_identity"] = asdict(identities["chronology"])
        raw = (json.dumps(inputs[name], sort_keys=True, allow_nan=False) + "\n").encode()
        (tmp_path / paths[name]).write_bytes(raw)
        identities[name] = EvidenceIdentity(len(raw), sha256(raw).hexdigest())
    return tmp_path, paths, identities


def test_should_read_every_complete_file_and_leave_all_inputs_unchanged(whole_inputs: Any) -> None:
    root, paths, identities = whole_inputs
    before = {name: (root / path).read_bytes() for name, path in paths.items()}
    result = boundary.load_pending_confirmation_contract(root, paths, identities)
    assert result["whole_declared_input_identities"] == {
        name: asdict(value) for name, value in identities.items()
    }
    assert len(result["every_original_prior_effect"]) == 4471
    assert result["actual_current_source_proof_verified"] is False
    assert result["fresh_roles_authorized"] is False
    assert before == {name: (root / path).read_bytes() for name, path in paths.items()}


@pytest.mark.parametrize("name", sorted(boundary.INPUT_NAMES))
def test_should_reject_whole_drift_in_every_input_kind(whole_inputs: Any, name: str) -> None:
    root, paths, identities = whole_inputs
    (root / paths[name]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="whole saved input drift"):
        boundary.load_pending_confirmation_contract(root, paths, identities)


@pytest.mark.parametrize("marker", [".claim", ".failure.json"])
def test_should_refuse_unfinished_or_failed_input_producers(whole_inputs: Any, marker: str) -> None:
    root, paths, identities = whole_inputs
    (root / paths["repeat_witness"]).with_suffix(marker).write_bytes(b"retained")
    with pytest.raises(ValueError, match="producer marker"):
        boundary.load_pending_confirmation_contract(root, paths, identities)


@pytest.mark.parametrize("late_change", ["bytes", "marker"])
def test_should_refuse_late_input_mutation_after_complete_projection(
    whole_inputs: Any, monkeypatch: pytest.MonkeyPatch, late_change: str
) -> None:
    root, paths, identities = whole_inputs
    original = boundary.build_pending_confirmation_contract

    def mutate(*args: Any) -> dict[str, Any]:
        result = original(*args)
        path = root / paths["repeat_witness"]
        if late_change == "bytes":
            path.write_bytes(b"changed late")
        else:
            path.with_suffix(".failure.json").write_bytes(b"retained late failure")
        return result

    monkeypatch.setattr(boundary, "build_pending_confirmation_contract", mutate)
    with pytest.raises(ValueError):
        boundary.load_pending_confirmation_contract(root, paths, identities)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.json",
        "/outside.json",
        "C:/foreign.json",
        "nested/../chronology.json",
        "nested\\file.json",
        "./chronology.json",
    ],
)
def test_should_reject_external_or_noncanonical_paths(whole_inputs: Any, name: str) -> None:
    root, paths, identities = whole_inputs
    paths["chronology"] = name
    with pytest.raises(ValueError, match="root-relative"):
        boundary.load_pending_confirmation_contract(root, paths, identities)


@pytest.mark.parametrize(
    "raw", [b'{"duplicate":1,"duplicate":2}', b'{"value":NaN}', b"[]", b'{"bad":']
)
def test_should_reject_duplicate_nonfinite_nonobject_or_malformed_whole_json(
    whole_inputs: Any, raw: bytes
) -> None:
    root, paths, identities = whole_inputs
    (root / paths["repeat_witness"]).write_bytes(raw)
    identities["repeat_witness"] = EvidenceIdentity(len(raw), sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        boundary.load_pending_confirmation_contract(root, paths, identities)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "bad_identity"])
def test_should_require_distinct_complete_input_paths_and_exact_identities(
    whole_inputs: Any, kind: str
) -> None:
    root, paths, identities = deepcopy(whole_inputs)
    if kind == "missing":
        paths.pop("precision_repeat")
    elif kind == "duplicate":
        paths["precision_repeat"] = paths["precision"]
    else:
        identities["precision"] = EvidenceIdentity(True, "a" * 64)
    with pytest.raises(ValueError):
        boundary.load_pending_confirmation_contract(root, paths, identities)


def test_should_refuse_a_symlink_detected_at_the_input_boundary(
    whole_inputs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, paths, identities = whole_inputs
    original = Path.is_symlink
    target = root / paths["repeat_witness"]
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    with pytest.raises(ValueError, match="without symlinks"):
        boundary.load_pending_confirmation_contract(root, paths, identities)
