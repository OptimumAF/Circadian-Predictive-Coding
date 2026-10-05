"""Complete pure byte-binding behavior with small fabricated bodies and records."""

from hashlib import sha256
from copy import deepcopy
from typing import Any

import pytest

from src.app import continual_findings_synthesis_inputs as module
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_findings_synthesis import build_complete_findings


def _fixture(monkeypatch) -> dict[str, Any]:
    content = "keep CRLF\r\nnegative=-0.12345678901234568\r\n"
    raw = content.encode()
    identity = {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}
    bodies = {
        name: {"raw": {"negative": -0.12345678901234568, "unmeasured": None}}
        for name in (
            "primary",
            "development",
            "matrix",
            "activity",
            "cost_inspection",
            "outcome_costs",
        )
    }
    catalog = {
        "bodies": {
            name: {"identity": canonical_body_identity(body)} for name, body in bodies.items()
        },
        "records": {"original.txt": identity},
    }
    monkeypatch.setattr(module, "CATALOG_ID", canonical_body_identity(catalog))
    return {
        "catalog": catalog,
        "bodies": bodies,
        "preserved_records": {"original.txt": {"content": content, "identity": identity.copy()}},
    }


def test_should_bind_full_raw_record_bytes_without_normalizing_newlines(monkeypatch):
    inputs = _fixture(monkeypatch)
    module.verify_synthesis_identities(inputs)
    inputs["preserved_records"]["original.txt"]["content"] = inputs["preserved_records"][
        "original.txt"
    ]["content"].replace("\r\n", "\n")
    with pytest.raises(ValueError, match="whole preserved"):
        module.verify_synthesis_identities(inputs)


@pytest.mark.parametrize(
    "name", ["primary", "development", "matrix", "activity", "cost_inspection", "outcome_costs"]
)
def test_should_reject_late_negative_null_or_raw_body_corruption(monkeypatch, name):
    inputs = _fixture(monkeypatch)
    inputs["bodies"][name]["raw"]["unmeasured"] = 0
    with pytest.raises(ValueError, match="whole synthesis"):
        module.verify_synthesis_identities(inputs)


@pytest.mark.parametrize(
    "section,key", [("bodies", "matrix"), ("preserved_records", "original.txt")]
)
def test_should_reject_missing_whole_inputs_before_interpretation(monkeypatch, section, key):
    inputs = _fixture(monkeypatch)
    del inputs[section][key]
    with pytest.raises(ValueError, match="all"):
        module.verify_synthesis_identities(inputs)


def test_should_reject_changed_catalog_even_if_remaining_subset_is_self_consistent(monkeypatch):
    inputs = _fixture(monkeypatch)
    del inputs["catalog"]["bodies"]["matrix"]
    del inputs["bodies"]["matrix"]
    with pytest.raises(ValueError, match="catalog"):
        module.verify_synthesis_identities(inputs)


def test_should_reject_unpinned_public_input_before_any_declaration_or_io(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.app.continual_findings_synthesis.verify_synthesis_declarations",
        lambda *_: calls.append("derive"),
    )
    with pytest.raises(ValueError, match="catalog"):
        build_complete_findings({"catalog": {}, "bodies": {}, "preserved_records": {}})
    assert calls == []


@pytest.mark.parametrize("value", [None, [], {"catalog": {}}])
def test_should_fail_loudly_on_malformed_public_inputs(value):
    with pytest.raises(ValueError):
        build_complete_findings(value)


def _matrix_fixture(monkeypatch) -> tuple[dict[str, Any], dict[str, Any]]:
    source: dict[str, Any] = {
        "source_sha256": {"original.py": "f" * 64},
        "source_map_sha256": "e" * 64,
        "request_template": {"inputs": {"whole-report": {"byte_count": 100, "sha256": "d" * 64}}},
    }
    matrix: dict[str, Any] = {
        "rows": [{"raw_negative": -0.025}],
        "provenance": {
            "validation_scope": "current_source_bound_complete_original_report_readback_and_stored_matrix",
            "source_sha256": deepcopy(source["source_sha256"]),
            "source_map_sha256": source["source_map_sha256"],
            "inputs": deepcopy(source["request_template"]["inputs"]),
            "new_training_or_final_source_access": False,
        },
    }
    monkeypatch.setattr(
        module, "build_confirmation_matrix", lambda _: {"rows": [{"raw_negative": -0.025}]}
    )
    return matrix, source


def test_should_preserve_historical_provenance_separately_from_whole_pure_matrix(monkeypatch):
    matrix, source = _matrix_fixture(monkeypatch)
    original = deepcopy(matrix)
    module._verify_matrix_declarations(matrix, {}, source)
    assert matrix == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", {}),
        ("source_map_sha256", "0" * 64),
        ("inputs", {}),
        ("new_training_or_final_source_access", True),
        ("validation_scope", "freshly_reconstructed_reader"),
    ],
)
def test_should_reject_changed_whole_historical_matrix_provenance(monkeypatch, field, value):
    matrix, source = _matrix_fixture(monkeypatch)
    matrix["provenance"][field] = value
    with pytest.raises(ValueError, match="historical matrix provenance"):
        module._verify_matrix_declarations(matrix, {}, source)


def test_should_reject_changed_matrix_declarations_even_with_intact_provenance(monkeypatch):
    matrix, source = _matrix_fixture(monkeypatch)
    matrix["rows"][0]["raw_negative"] = 0
    with pytest.raises(ValueError, match="pure matrix declarations"):
        module._verify_matrix_declarations(matrix, {}, source)
