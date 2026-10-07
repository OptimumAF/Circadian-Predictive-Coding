"""IO-only fabricated reader spies; no reserved training or final evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from test_continual_confirmation_scoring import _forbid
from src.app import continual_arrived_benchmark as arrived
from src.app.continual_confirmation_execution import digest_json
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra import continual_confirmation_final as final_adapter
from src.infra import continual_confirmation_training_references as references
from src.infra.continual_confirmation_io import read_json, write_exclusive


@pytest.fixture(autouse=True)
def seal_reference_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    for model in (
        BackpropMLP,
        PredictiveCodingNetwork,
        CircadianPredictiveCodingNetwork,
        ParentControlledCircadianNetwork,
    ):
        monkeypatch.setattr(model, "__init__", _forbid)
        monkeypatch.setattr(model, "train_epoch", _forbid)
        monkeypatch.setattr(model, "predict_proba", _forbid)
    monkeypatch.setattr(final_adapter, "release_confirmation_final", _forbid)
    monkeypatch.setattr(final_adapter, "evaluate_confirmation_final", _forbid)


@pytest.fixture
def fabricated_bundles(tmp_path: Path) -> tuple[Any, Any, list[str]]:
    manifest = fixed_scoring_manifest()
    source_map = {"src/app/fabricated.py": "1" * 64}
    manifest = replace(manifest, training_source_map_sha256=digest_json(source_map))
    declared = []
    for reference in manifest.training_bundles:
        directory = tmp_path / reference.directory
        directory.mkdir(parents=True)
        # Deliberately incomplete scientific facts. Only private IO seams
        # accept these byte-bound fixtures; the public fixed manifest cannot.
        request = {
            "manifest_sha256": manifest.analysis_contract.train_manifest_sha256,
            "scope_record_sha256": manifest.scope_record_sha256,
            "source_sha256": source_map,
            "adapter_sha256": manifest.training_adapter_sha256,
            "fixture_only": True,
        }
        result = {"fixture_only": True, "unscored": list(range(80))}
        write_exclusive(directory / "confirmation-train.request.json", request)
        write_exclusive(directory / "confirmation-train.result.json", result)
        request_sha = sha256(
            (directory / "confirmation-train.request.json").read_bytes()
        ).hexdigest()
        result_bytes = (directory / "confirmation-train.result.json").read_bytes()
        result_sha = sha256(result_bytes).hexdigest()
        audit = {
            **request,
            "request_sha256": request_sha,
            "result_sha256": result_sha,
            "work": {"by_seed": [{"fixture_only": True}], "totals": {"cells": 560}},
            "observed_updates": {"fixture_only": True},
            "process_rss": {"fixture_only": True},
            "worker_elapsed_seconds": 1.0,
            "elapsed_seconds": 2.0,
        }
        write_exclusive(directory / "confirmation-train.audit.json", audit)
        audit_sha = sha256((directory / "confirmation-train.audit.json").read_bytes()).hexdigest()
        declared.append(
            replace(
                reference,
                request_sha256=request_sha,
                result_sha256=result_sha,
                result_bytes=len(result_bytes),
                audit_sha256=audit_sha,
            )
        )
    fixture_manifest = replace(manifest, training_bundles=tuple(declared))
    calls: list[str] = []

    def reader(directory: Path) -> Any:
        calls.append(directory.name)
        return tuple(
            read_json(directory / f"confirmation-train.{name}.json")
            for name in ("request", "result", "audit")
        )

    return fixture_manifest, reader, calls


def test_should_read_both_references_sequentially_and_return_only_small_metadata(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path
) -> None:
    manifest, reader, calls = fabricated_bundles
    report = references._read_reference_bundles(tmp_path, manifest, reader)
    assert calls == ["p67-confirmation-train", "p67-confirmation-train-repeat"]
    assert len(report["bundles"]) == 2
    assert report["bundles"][0]["work"] == report["bundles"][1]["work"]
    for bundle in report["bundles"]:
        assert "result" not in bundle and "request" not in bundle and "audit" not in bundle
        assert bundle["complete_readback"] is True
        assert bundle["files"]["result"]["sha256"] == manifest.training_bundles[0].result_sha256
    assert "unscored" not in str(report)


@pytest.mark.parametrize("bundle", [0, 1])
@pytest.mark.parametrize("name", ["request", "result", "audit", "failure", "claim"])
def test_should_refuse_missing_bytes_or_failure_claim_markers_before_any_reader(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path, bundle: int, name: str
) -> None:
    manifest, reader, calls = fabricated_bundles
    directory = tmp_path / manifest.training_bundles[bundle].directory
    path = directory / (
        "confirmation-train.claim" if name == "claim" else f"confirmation-train.{name}.json"
    )
    if name in {"request", "result", "audit"}:
        path.unlink()
    else:
        path.write_text("fixture occupied\n", encoding="utf-8")
    with pytest.raises(ValueError):
        references._read_reference_bundles(tmp_path, manifest, reader)
    assert not calls


@pytest.mark.parametrize("bundle", [0, 1])
@pytest.mark.parametrize("name", ["request", "result", "audit"])
def test_should_reject_changed_reference_bytes_before_any_reader(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path, bundle: int, name: str
) -> None:
    manifest, reader, calls = fabricated_bundles
    path = (
        tmp_path / manifest.training_bundles[bundle].directory / f"confirmation-train.{name}.json"
    )
    with path.open("a", encoding="utf-8") as output:
        output.write(" ")
    with pytest.raises(ValueError, match="bytes"):
        references._read_reference_bundles(tmp_path, manifest, reader)
    assert not calls


def test_should_bind_the_exact_result_length_as_well_as_hash(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path
) -> None:
    manifest, reader, calls = fabricated_bundles
    first = replace(
        manifest.training_bundles[0], result_bytes=manifest.training_bundles[0].result_bytes + 1
    )
    changed = replace(manifest, training_bundles=(first, manifest.training_bundles[1]))
    with pytest.raises(ValueError, match="length"):
        references._read_reference_bundles(tmp_path, changed, reader)
    assert not calls


@pytest.mark.parametrize("name", ["request", "result", "audit"])
def test_should_reject_detached_decoded_bodies_even_when_file_bytes_stay_bound(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path, name: str
) -> None:
    manifest, reader, calls = fabricated_bundles

    def detached(directory: Path) -> Any:
        parts = list(reader(directory))
        parts[("request", "result", "audit").index(name)]["late_drift"] = True
        return tuple(parts)

    with pytest.raises(ValueError, match="decoded"):
        references._read_reference_bundles(tmp_path, manifest, detached)
    assert len(calls) == 1


@pytest.mark.parametrize("stage", ["first_read", "last_read", "earlier_after_last"])
@pytest.mark.parametrize("kind", ["bytes", "failure", "claim"])
def test_should_recheck_reference_bytes_and_markers_through_the_last_reader(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path, stage: str, kind: str
) -> None:
    manifest, reader, calls = fabricated_bundles

    def drift(directory: Path) -> Any:
        parts = reader(directory)
        trigger = len(calls) == (1 if stage == "first_read" else 2)
        if trigger:
            target = (
                tmp_path / manifest.training_bundles[0].directory
                if stage == "earlier_after_last"
                else directory
            )
            path = target / (
                "confirmation-train.result.json"
                if kind == "bytes"
                else "confirmation-train.failure.json"
                if kind == "failure"
                else "confirmation-train.claim"
            )
            with path.open("a", encoding="utf-8") as output:
                output.write(" ")
        return parts

    with pytest.raises(ValueError):
        references._read_reference_bundles(tmp_path, manifest, drift)
    assert len(calls) == (1 if stage == "first_read" else 2)


@pytest.mark.parametrize("parts", [None, [], ({},), ({}, {}, []), [{}, {}, {}]])
def test_should_reject_invalid_completed_reader_contracts(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path, parts: Any
) -> None:
    manifest, _, _ = fabricated_bundles
    with pytest.raises(ValueError, match="reader"):
        references._read_reference_bundles(tmp_path, manifest, lambda directory: parts)


def test_should_propagate_complete_reader_errors_without_reading_the_repeat(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path
) -> None:
    manifest, _, _ = fabricated_bundles
    calls = []

    def failed(directory: Path) -> Any:
        calls.append(directory)
        raise RuntimeError("fabricated full validator rejection")

    with pytest.raises(RuntimeError, match="full validator"):
        references._read_reference_bundles(tmp_path, manifest, failed)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field", ["manifest_sha256", "scope_record_sha256", "adapter_sha256", "source_sha256"]
)
def test_should_reject_resealed_old_declaration_drift(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path, field: str
) -> None:
    manifest, reader, _ = fabricated_bundles
    reference = manifest.training_bundles[1]
    directory = tmp_path / reference.directory
    request_path = directory / "confirmation-train.request.json"
    request = read_json(request_path)
    request[field] = {} if field == "source_sha256" else "0" * 64
    request_path.unlink()
    write_exclusive(request_path, request)
    request_sha = sha256(request_path.read_bytes()).hexdigest()
    audit_path = directory / "confirmation-train.audit.json"
    audit = read_json(audit_path)
    audit[field] = deepcopy(request[field])
    audit["request_sha256"] = request_sha
    audit_path.unlink()
    write_exclusive(audit_path, audit)
    altered = replace(
        reference,
        request_sha256=request_sha,
        audit_sha256=sha256(audit_path.read_bytes()).hexdigest(),
    )
    changed = replace(manifest, training_bundles=(manifest.training_bundles[0], altered))
    with pytest.raises(ValueError, match="binding"):
        references._read_reference_bundles(tmp_path, changed, reader)


def test_should_refuse_private_fixture_manifest_at_public_reference_gates(
    fabricated_bundles: tuple[Any, Any, list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _, _ = fabricated_bundles
    monkeypatch.setattr(Path, "open", _forbid)
    with pytest.raises(ValueError, match="frozen complete"):
        references.read_training_references(tmp_path, manifest, _forbid)
    with pytest.raises(ValueError, match="frozen complete"):
        references.verify_training_reference_bytes(tmp_path, manifest)


def test_should_stream_large_file_identity_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"fabricated local bytes\n" * 100000
    path = tmp_path / "large-fixture.bin"
    path.write_bytes(body)
    monkeypatch.setattr(Path, "read_bytes", _forbid)
    assert references.stream_file_identity(path) == {
        "sha256": sha256(body).hexdigest(),
        "byte_count": len(body),
    }
