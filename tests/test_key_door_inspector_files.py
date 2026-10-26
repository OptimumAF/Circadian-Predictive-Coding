"""Strict file admission and offline HTML publication; no model or environment work."""

from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.build_key_door_inspector as cli
import src.infra.key_door_inspector_files as files


class Page(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=False)
        self.payload_parts: list[str] = []
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.in_payload = False
        self.payload_count = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag == "script" and attributes.get("id") == "inspector-data":
            assert attributes.get("type") == "application/json"
            self.in_payload = True
            self.payload_count += 1

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_payload = False

    def handle_data(self, data):
        if self.in_payload:
            self.payload_parts.append(data)

    def payload(self) -> Any:
        assert self.payload_count == 1
        return json.loads("".join(self.payload_parts))


def stub_projection(monkeypatch, payload=None):
    projected = {"source": {"label": "recorded"}, "conditions": []} if payload is None else payload
    calls: list[Any] = []

    def project(report, *, source_label):
        calls.append((deepcopy(report), source_label))
        return deepcopy(projected)

    monkeypatch.setattr(files, "project_key_door_report", project)
    return projected, calls


def source_file(tmp_path, payload=b'{"status":"complete"}'):
    source = tmp_path / "recorded & quoted.json"
    source.write_bytes(payload)
    return source


def test_should_publish_one_offline_page_preserving_source_bytes_and_projection(
    tmp_path, monkeypatch
):
    projected, calls = stub_projection(monkeypatch)
    source = source_file(tmp_path)
    before = source.read_bytes()
    destination = tmp_path / "derived" / "inspector.html"
    written = files.write_key_door_inspector(source, destination)
    assert written == destination.resolve()
    assert source.read_bytes() == before
    assert calls == [({"status": "complete"}, source.name)]
    page = Page(written.read_text(encoding="utf-8"))
    assert page.payload() == projected
    for _, attributes in page.tags:
        for attribute in ("src", "href"):
            value = attributes.get(attribute)
            assert value is None or not value.lower().startswith(("http:", "https:", "//"))
    assert files.MAX_INPUT_BYTES == 8 * 1024**2
    assert files.MAX_OUTPUT_BYTES == 16 * 1024**2


def test_should_escape_script_terminators_markup_entities_and_unicode_without_changing_data(
    tmp_path, monkeypatch
):
    dangerous = '</script><img id="injected" src="x" onerror="alert(1)"> & < > \u2028\u2029'
    projected, _ = stub_projection(
        monkeypatch,
        {
            "source": {"label": dangerous},
            "conditions": [],
            "literal": "\\u003c/script\\u003e",
            "nested": [dangerous, {"<&>": dangerous}],
        },
    )
    source = source_file(tmp_path)
    destination = tmp_path / "inspector.html"
    files.write_key_door_inspector(str(source), str(destination))
    page = Page(destination.read_text(encoding="utf-8"))
    assert page.payload() == projected
    assert not any(attributes.get("id") == "injected" for _, attributes in page.tags)
    embedded = "".join(page.payload_parts)
    assert "<" not in embedded and ">" not in embedded and "&" not in embedded
    assert "</script>" not in embedded.lower()


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"\xef\xbb\xbf{}",
        b'{"broken":',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":1e999}',
        b'{"x":-1e999}',
        b'{"x":1,"x":2}',
        b'{"outer":{"x":1,"x":2}}',
    ],
)
def test_should_refuse_invalid_utf8_json_duplicate_keys_and_nonfinite_before_output(
    tmp_path, monkeypatch, payload
):
    source = source_file(tmp_path, payload)
    _, calls = stub_projection(monkeypatch)
    parent = tmp_path / "uncreated"
    destination = parent / "inspector.html"
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, destination)
    assert source.read_bytes() == payload and not parent.exists()
    assert calls == []


def test_should_reject_oversized_input_before_projection_or_output_allocation(
    tmp_path, monkeypatch
):
    source = tmp_path / "too-large.json"
    with source.open("wb") as stream:
        stream.write(b"{}")
        stream.truncate(8 * 1024**2 + 1)
    _, calls = stub_projection(monkeypatch)
    destination = tmp_path / "uncreated" / "inspector.html"
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, destination)
    assert calls == [] and not destination.parent.exists()
    assert source.stat().st_size == 8 * 1024**2 + 1


def test_should_reject_oversized_render_before_creating_parent_or_output(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    stub_projection(
        monkeypatch, {"source": {"label": "large"}, "conditions": [], "text": "x" * 2048}
    )
    monkeypatch.setattr(files, "MAX_OUTPUT_BYTES", 1024)
    destination = tmp_path / "uncreated" / "inspector.html"
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, destination)
    assert not destination.parent.exists() and source.read_bytes() == b'{"status":"complete"}'


def test_should_preserve_source_when_destination_resolves_to_source(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    before = source.read_bytes()
    _, calls = stub_projection(monkeypatch)
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, tmp_path / "." / source.name)
    assert source.read_bytes() == before and calls == []


def test_should_preserve_existing_destination_without_projecting_or_overwriting(
    tmp_path, monkeypatch
):
    source = source_file(tmp_path)
    destination = tmp_path / "existing.html"
    destination.write_bytes(b"unrelated existing output")
    _, calls = stub_projection(monkeypatch)
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, destination)
    assert destination.read_bytes() == b"unrelated existing output" and calls == []


def test_should_use_exclusive_publication_when_destination_appears_during_projection(
    tmp_path, monkeypatch
):
    source = source_file(tmp_path)
    destination = tmp_path / "raced.html"

    def project(*args, **kwargs):
        destination.write_bytes(b"created by another writer")
        return {"source": {"label": "race"}, "conditions": []}

    monkeypatch.setattr(files, "project_key_door_report", project)
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, destination)
    assert destination.read_bytes() == b"created by another writer"
    assert source.read_bytes() == b'{"status":"complete"}'


def test_should_leave_no_output_when_projection_rejects_corrupt_records(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    destination = tmp_path / "uncreated" / "inspector.html"

    def reject(*args, **kwargs):
        raise ValueError("recorded transition is inconsistent")

    monkeypatch.setattr(files, "project_key_door_report", reject)
    with pytest.raises(ValueError, match="transition"):
        files.write_key_door_inspector(source, destination)
    assert not destination.parent.exists()


def test_should_build_actual_accepted_paired_report_without_modifying_recorded_evidence(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/r21-return-fresh-task-probes-20261016/native.json"
    )
    before = sha256(source.read_bytes()).hexdigest()
    expected = files.project_key_door_report(
        json.loads(source.read_text(encoding="utf-8")), source_label=source.name
    )
    destination = tmp_path / "paired.html"
    files.write_key_door_inspector(source, destination)
    assert sha256(source.read_bytes()).hexdigest() == before
    page = Page(destination.read_text(encoding="utf-8"))
    assert page.payload() == expected
    assert len(expected["conditions"]) == 2
    assert 0 < destination.stat().st_size <= 16 * 1024**2


def test_should_dispatch_cli_with_explicit_paths_and_print_saved_output_metadata(
    tmp_path, monkeypatch, capsys
):
    source, destination = tmp_path / "source.json", tmp_path / "inspector.html"
    calls = []

    def write(source_path, destination_path):
        calls.append((Path(source_path), Path(destination_path)))
        destination.write_bytes(b"offline-page")
        return destination.resolve()

    monkeypatch.setattr(cli, "write_key_door_inspector", write)
    monkeypatch.setattr(
        sys, "argv", ["inspector", "--source-artifact", str(source), "--output", str(destination)]
    )
    cli.main()
    assert calls == [(source, destination)]
    assert json.loads(capsys.readouterr().out) == dict(
        output=str(destination.resolve()),
        source=str(source.resolve()),
        bytes=12,
    )


@pytest.mark.parametrize(
    "error", [ValueError("invalid recorded transitions"), FileExistsError("output already exists")]
)
def test_should_exit_with_cli_error_without_publishing_when_file_boundary_refuses(
    tmp_path, monkeypatch, capsys, error
):
    destination = tmp_path / "inspector.html"

    def reject(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "write_key_door_inspector", reject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspector",
            "--source-artifact",
            str(tmp_path / "source.json"),
            "--output",
            str(destination),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and not destination.exists()
    output = capsys.readouterr()
    assert output.out == "" and str(error) in output.err
