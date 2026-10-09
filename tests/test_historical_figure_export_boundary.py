"""IO refusals precede optional plotting imports and directory creation."""

from hashlib import sha256
import json

import pytest

from scripts import export_historical_outcome_figures as exporter


@pytest.mark.parametrize("raw", [b"{}", b"", b"wrong source identity"])
def test_should_refuse_unaccepted_source_without_creating_output(tmp_path, raw):
    source, output = tmp_path / "source.json", tmp_path / "output"
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="whole bytes/SHA256"):
        exporter.export_figures(source, output)
    assert not output.exists()
    assert source.read_bytes() == raw


def test_should_bind_complete_bytes_not_a_prefix(tmp_path, monkeypatch):
    raw = json.dumps({"fixture": "tiny non-scientific body"}).encode()
    source = tmp_path / "source.json"
    source.write_bytes(raw)
    monkeypatch.setattr(exporter, "SOURCE_BYTES", len(raw))
    monkeypatch.setattr(exporter, "SOURCE_SHA256", sha256(raw).hexdigest())
    assert exporter.read_pinned_body(source) == (raw, json.loads(raw))
    source.write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        exporter.read_pinned_body(source)


def test_should_refuse_different_body_even_with_same_length(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b'{"x":2}')
    monkeypatch.setattr(exporter, "SOURCE_BYTES", 7)
    monkeypatch.setattr(exporter, "SOURCE_SHA256", sha256(b'{"x":1}').hexdigest())
    with pytest.raises(ValueError):
        exporter.read_pinned_body(source)


def test_should_write_once_and_keep_prior_artifact(tmp_path):
    output = tmp_path / "result.json"
    assert exporter._write(output, b"first")["sha256"] == sha256(b"first").hexdigest()
    with pytest.raises(FileExistsError):
        exporter._write(output, b"second")
    assert output.read_bytes() == b"first"


def test_should_bind_metric_table_columns_by_name_not_dictionary_order():
    values = dict(zip(exporter.CELL_METRICS, range(1, 7), strict=True))
    row = {
        "family": "fixture",
        "seed": 23,
        "arm": "ordinary",
        "metrics": {
            name: {"value": values[name], "reason": None}
            for name in reversed(exporter.CELL_METRICS)
        },
        "resource_fields": {
            name: {"value": n}
            for name, n in zip(
                (
                    "executed_optimizer_updates",
                    "optimizer_latent_iterations",
                    "rejected_replay_updates",
                    "parameters_peak",
                ),
                range(31, 35),
                strict=True,
            )
        },
        "owned_retention": {"checkpoints": {"after_b": {"owned_array_bytes": 0}}},
        "outcome": {"failure": None},
    }
    report = exporter._markdown({"rows": [row], "stage_storage_totals": {}}, [])
    assert (
        "| fixture | 23 | ordinary | 1 | 2 | 3 | 4 | 5 | 6 | 31 | 32 | 33 | 34 | 0 | None |"
        in report
    )
