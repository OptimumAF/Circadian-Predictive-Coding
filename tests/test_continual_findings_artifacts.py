"""Exclusive publication, fresh readbacks and failure/ownership behavior."""

import json

import pytest

from src.infra import continual_findings_artifacts as module
from findings_publication_fixtures import make_artifact_boundary, write_body


def test_should_publish_then_dispatch_all_fresh_ports_again_for_independent_readback(
    tmp_path, monkeypatch
):
    fixture, readers, events = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    audit = module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    request, body, rebuilt = module.read_completed_findings(
        tmp_path, output, tmp_path / "scope", readers
    )
    assert body == fixture["body"] and audit == rebuilt
    assert request["protocol_id"] == "fixed"
    assert events == ["outcome_costs", "matrix", "development"] * 2
    assert not module.artifact_paths(output)["claim"].exists()
    assert not module.artifact_paths(output)["failure"].exists()


@pytest.mark.parametrize("part", ["request", "result", "markdown", "audit", "failure", "claim"])
def test_should_preserve_occupied_output_without_dispatching_readers(part, tmp_path, monkeypatch):
    _, readers, events = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    path = module.artifact_paths(output)[part]
    path.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    assert path.read_bytes() == b"foreign" and events == []


@pytest.mark.parametrize("part", ["result", "markdown", "audit"])
def test_should_reject_changed_complete_result_markdown_or_audit_on_fresh_readback(
    part, tmp_path, monkeypatch
):
    _, readers, events = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    paths = module.artifact_paths(output)
    if part == "markdown":
        paths[part].write_bytes(b"selected winner\n")
    else:
        body = json.loads(paths[part].read_bytes())
        body["selected_winner"] = True
        write_body(paths[part], body)
    with pytest.raises(ValueError):
        module.read_completed_findings(tmp_path, output, tmp_path / "scope", readers)
    assert events == ["outcome_costs", "matrix", "development"] * 2


@pytest.mark.parametrize("part", ["claim", "failure"])
def test_should_reject_partial_bundle_before_any_fresh_reader_call(part, tmp_path, monkeypatch):
    _, readers, events = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    module.artifact_paths(output)[part].write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="complete successful"):
        module.read_completed_findings(tmp_path, output, tmp_path / "scope", readers)
    assert events == ["outcome_costs", "matrix", "development"]


def test_should_preserve_foreign_claim_after_mid_publication_replacement(tmp_path, monkeypatch):
    _, readers, _ = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    paths = module.artifact_paths(output)
    original = module._body_files

    def replacing(*args):
        result = original(*args)
        paths["claim"].write_bytes(b"foreign-owner")
        return result

    monkeypatch.setattr(module, "_body_files", replacing)
    with pytest.raises(ValueError, match="claim ownership"):
        module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    assert paths["claim"].read_bytes() == b"foreign-owner"
    assert paths["failure"].is_file()


def test_should_detect_late_source_input_or_environment_change_after_body_write(
    tmp_path, monkeypatch
):
    fixture, readers, _ = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    original = module._body_files

    def drifting(*args):
        result = original(*args)
        fixture["state"]["input"] = "late subset"
        return result

    monkeypatch.setattr(module, "_body_files", drifting)
    with pytest.raises(ValueError, match="current bindings"):
        module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    assert module.artifact_paths(output)["failure"].is_file()
    with pytest.raises(ValueError, match="complete successful"):
        module.read_completed_findings(tmp_path, output, tmp_path / "scope", readers)


def test_should_revoke_only_owned_audit_if_failure_marker_write_also_fails(tmp_path, monkeypatch):
    fixture, readers, _ = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    original = module._body_files

    def drifting(*args):
        result = original(*args)
        fixture["state"]["source"] = "drift"
        return result

    def broken_marker(*_):
        raise OSError("failure marker unavailable")

    monkeypatch.setattr(module, "_body_files", drifting)
    monkeypatch.setattr(module, "_failure", broken_marker)
    with pytest.raises(OSError, match="marker unavailable"):
        module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    assert not module.artifact_paths(output)["audit"].exists()
    with pytest.raises(ValueError, match="complete successful"):
        module.read_completed_findings(tmp_path, output, tmp_path / "scope", readers)


def test_should_preserve_foreign_audit_when_failure_marker_cannot_be_written(tmp_path, monkeypatch):
    _, readers, _ = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    paths = module.artifact_paths(output)

    def foreign_audit(*_):
        paths["audit"].write_bytes(b"another owner's audit")
        raise ValueError("late ownership change")

    def broken_marker(*_):
        raise OSError("failure marker unavailable")

    monkeypatch.setattr(module, "_finish", foreign_audit)
    monkeypatch.setattr(module, "_failure", broken_marker)
    with pytest.raises(OSError, match="marker unavailable"):
        module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    assert paths["audit"].read_bytes() == b"another owner's audit"


def test_should_reject_environment_drift_during_independent_readback(tmp_path, monkeypatch):
    fixture, readers, events = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    original = module.rebuild_current_findings

    def drifting(*args):
        result = original(*args)
        fixture["state"]["environment"] = "changed during fresh readback"
        return result

    monkeypatch.setattr(module, "rebuild_current_findings", drifting)
    with pytest.raises(ValueError, match="current bindings"):
        module.read_completed_findings(tmp_path, output, tmp_path / "scope", readers)
    assert events == ["outcome_costs", "matrix", "development"] * 2


def test_should_fail_publication_exceeding_the_declared_outer_cap(tmp_path, monkeypatch):
    _, readers, _ = make_artifact_boundary(tmp_path, monkeypatch)
    output = tmp_path / "output"
    clock = {"now": 0.0}
    original = module._body_files

    def delayed(*args):
        result = original(*args)
        clock["now"] = 841.0
        return result

    monkeypatch.setattr(module, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(module, "_body_files", delayed)
    with pytest.raises(ValueError):
        module.publish_complete_findings(tmp_path, output, tmp_path / "scope", readers)
    paths = module.artifact_paths(output)
    assert paths["failure"].is_file() and not paths["claim"].exists()
