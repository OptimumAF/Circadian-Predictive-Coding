"""Complete reader dispatch, canonical corruption and original budget checks."""

from copy import deepcopy
from hashlib import sha256

import pytest

from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_findings_readers import FindingsReaders
from src.infra import continual_findings_current_inputs as module
from findings_publication_fixtures import make_repository, record_call, write_body


@pytest.mark.parametrize("parts", [({}, {}), {"result": {}}, ({}, {}, []), None])
def test_should_reject_partial_or_wrong_complete_reader_parts(parts, tmp_path, monkeypatch):
    fixture = make_repository(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="whole current"):
        module._bundle_reader("matrix", lambda: parts, fixture["catalog"]["current_bundles"][2])


@pytest.mark.parametrize("part", [0, 1, 2])
def test_should_reject_corrupted_whole_request_result_or_audit_return(part, tmp_path, monkeypatch):
    fixture = make_repository(tmp_path, monkeypatch)
    parts = deepcopy(fixture["returned"]["matrix"])
    parts[part]["changed"] = True
    with pytest.raises(ValueError, match="whole returned"):
        module._bundle_reader("matrix", lambda: parts, fixture["catalog"]["current_bundles"][2])


@pytest.mark.parametrize("kind,elapsed", [("outcome_costs", 241.0), ("matrix", 181.0)])
def test_should_keep_each_original_reader_cap_independent_of_larger_outer_cap(
    kind, elapsed, tmp_path, monkeypatch
):
    fixture = make_repository(tmp_path, monkeypatch)
    values = iter([0.0, elapsed])
    monkeypatch.setattr(module, "monotonic", lambda: next(values))
    with pytest.raises(ValueError):
        module._bundle_reader(
            kind,
            lambda: fixture["returned"][kind],
            fixture["catalog"]["current_bundles"][0 if kind == "outcome_costs" else 2],
        )


def test_should_dispatch_both_independent_bundle_ports_and_complete_development_each_time(
    tmp_path, monkeypatch
):
    fixture = make_repository(tmp_path, monkeypatch)
    development = {"development": "all original", "negative": -0.025, "null": None}
    synthesis = {"bodies": {"development": {"identity": canonical_body_identity(development)}}}
    calls: list[str] = []
    readers = FindingsReaders(
        lambda: record_call(calls, "outcomes", fixture["returned"]["outcome_costs"]),
        lambda: record_call(calls, "matrix", fixture["returned"]["matrix"]),
        lambda: record_call(calls, "development", deepcopy(development)),
    )
    for _ in range(2):
        bodies, evidence = module._current_reader_inputs(readers, fixture["catalog"], synthesis)
        assert bodies["development"] == development and len(evidence) == 3
    assert calls == ["outcomes", "matrix", "development"] * 2


def test_should_reject_partial_development_instead_of_substituting_saved_ledger(
    tmp_path, monkeypatch
):
    fixture = make_repository(tmp_path, monkeypatch)
    readers = FindingsReaders(
        lambda: fixture["returned"]["outcome_costs"],
        lambda: fixture["returned"]["matrix"],
        lambda: {"subset": True},
    )
    with pytest.raises(ValueError, match="whole returned current development"):
        module._current_reader_inputs(
            readers,
            fixture["catalog"],
            {"bodies": {"development": {"identity": canonical_body_identity({"whole": True})}}},
        )


def test_should_keep_literal_raw_record_newlines_and_reject_changed_bytes(tmp_path):
    path = tmp_path / "source.py"
    raw = b"literal -0.025\r\nNone\r\n"
    path.write_bytes(raw)
    expected = {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}
    assert module._raw_record(tmp_path, "source.py", expected)["content"] == raw.decode()
    path.write_bytes(raw.replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="preserved record"):
        module._raw_record(tmp_path, "source.py", expected)


def test_should_reject_late_bindings_after_fresh_readers_before_loading_pure_inputs(
    tmp_path, monkeypatch
):
    fixture = make_repository(tmp_path, monkeypatch)
    request = {"bindings": {"environment": "original"}}
    request_path = tmp_path / "request.json"
    write_body(request_path, request)
    catalog = {"synthesis_catalog": {"path": "synthesis.json", "identity": {}}}
    monkeypatch.setattr(module, "publication_declarations", lambda _: (catalog, {}))
    monkeypatch.setattr(module, "read_bound_json", lambda *_: {})
    calls = []

    def fresh_readers(*_):
        calls.append("fresh complete ports")
        return {}, {}

    monkeypatch.setattr(module, "_current_reader_inputs", fresh_readers)
    monkeypatch.setattr(
        module,
        "checked_findings_request",
        lambda *_: {"bindings": {"environment": "changed after readers"}},
    )

    def forbidden_pure_inputs(*_):
        raise AssertionError("must reject late drift before pure input loading")

    monkeypatch.setattr(module, "_synthesis_inputs", forbidden_pure_inputs)
    readers = FindingsReaders(lambda: ({}, {}, {}), lambda: ({}, {}, {}), lambda: {})
    with pytest.raises(ValueError, match="changed after fresh readers"):
        module.rebuild_current_findings(
            tmp_path, {"request": request_path}, fixture["scope"], readers, request
        )
    assert calls == ["fresh complete ports"]


def test_should_keep_the_original_development_reader_cap(tmp_path, monkeypatch):
    fixture = make_repository(tmp_path, monkeypatch)
    development = {"whole": True}
    values = iter([0.0, 0.0, 0.0, 0.0, 0.0, 121.0])
    monkeypatch.setattr(module, "monotonic", lambda: next(values))
    readers = FindingsReaders(
        lambda: fixture["returned"]["outcome_costs"],
        lambda: fixture["returned"]["matrix"],
        lambda: development,
    )
    with pytest.raises(ValueError):
        module._current_reader_inputs(
            readers,
            fixture["catalog"],
            {"bodies": {"development": {"identity": canonical_body_identity(development)}}},
        )
