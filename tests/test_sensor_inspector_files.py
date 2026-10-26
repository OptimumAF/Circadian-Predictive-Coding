"""Recorded sensor publication uses the existing bounded exclusive file boundary."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import scripts.build_key_door_inspector as cli
from src.infra import key_door_inspector_files as files
from test_key_door_inspector_files import Page

STAGE = Path("artifacts/runs/r46-sensor-heldout-20261017")
CASES = ("bp", "pc", "neutral_cpc", "gated_cpc", "scheduled_bp", "replay_bp")


@pytest.mark.parametrize("case", CASES)
def test_should_publish_saved_sensor_and_separate_resources_without_source_writes(tmp_path, case):
    source, resources = STAGE / f"{case}-result.json", STAGE / f"{case}-resources.json"
    before = source.read_bytes(), resources.read_bytes()
    output = tmp_path / f"{case}.html"
    files.write_key_door_inspector(source, output, resource_artifact=resources)
    html = output.read_text(encoding="utf8")
    projection = Page(html).payload()
    assert projection["schema_id"] == "sensor_inspector_v1"
    assert projection["source"] == json.loads(before[0])
    assert projection["resources"] == json.loads(before[1])
    assert projection["source_label"] == source.name
    assert projection["resource_label"] == resources.name
    assert "Sensor run inspector" in html
    assert "association" in html and "unverified" in html
    assert (source.read_bytes(), resources.read_bytes()) == before
    assert 0 < output.stat().st_size <= files.MAX_OUTPUT_BYTES
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, output)


def test_should_build_sensor_from_existing_cli_without_resources(tmp_path, monkeypatch, capsys):
    source = STAGE / "bp-result.json"
    output = tmp_path / "sensor.html"
    monkeypatch.setattr(
        sys, "argv", ["builder", "--source-artifact", str(source), "--output", str(output)]
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "output": str(output.resolve()),
        "source": str(source.resolve()),
        "bytes": output.stat().st_size,
    }
    projection = Page(output.read_text(encoding="utf8")).payload()
    assert projection["resources"] is None


def test_should_embed_injected_text_as_inert_json_and_display_with_text_content(tmp_path):
    report = deepcopy(json.loads((STAGE / "bp-result.json").read_text()))
    danger = '</script><img id="injected" src="https://example.invalid" onerror="alert(1)"> &'
    report.update(status="incomplete", reason=danger)
    source, output = tmp_path / "unsafe.json", tmp_path / "safe.html"
    source.write_text(json.dumps(report), encoding="utf8")
    files.write_key_door_inspector(source, output)
    page = Page(output.read_text(encoding="utf8"))
    assert page.payload()["source"]["reason"] == danger
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert "<" not in "".join(page.payload_parts)
    for _, attrs in page.tags:
        assert not any(
            (attrs.get(k) or "").lower().startswith(("https:", "http:", "//"))
            for k in ("src", "href")
        )


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"value":NaN}', '{"value":1e999}', "[1,2]"])
def test_should_refuse_malformed_optional_resources_before_creating_output_parent(
    tmp_path, payload
):
    resources = tmp_path / "bad.json"
    resources.write_text(payload)
    output = tmp_path / "new" / "sensor.html"
    with pytest.raises(ValueError):
        files.write_key_door_inspector(
            STAGE / "bp-result.json", output, resource_artifact=resources
        )
    assert not output.parent.exists()


def test_should_refuse_resource_output_alias_without_modifying_resources(tmp_path):
    resources = tmp_path / "resources.json"
    resources.write_text("{}")
    before = resources.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        files.write_key_door_inspector(
            STAGE / "bp-result.json", resources, resource_artifact=resources
        )
    assert resources.read_bytes() == before
