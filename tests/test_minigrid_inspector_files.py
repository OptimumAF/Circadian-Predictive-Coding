"""Offline publication of recorded MiniGrid reports through existing file ports."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.build_key_door_inspector as cli
from src.infra import key_door_inspector_files as files
from test_key_door_inspector_files import Page
from test_minigrid_inspector_projection import ARTIFACT, METHODS, pointer, recorded as recorded


def test_should_publish_actual_recorded_source_and_all_conditions_without_touching_source(
    tmp_path, recorded
):
    before = ARTIFACT.read_bytes()
    destination = tmp_path / "native-mini-grid.html"
    written = files.write_key_door_inspector(ARTIFACT, destination)
    assert written.resolve() == destination.resolve() and ARTIFACT.read_bytes() == before
    assert 0 < len(before) <= files.MAX_INPUT_BYTES
    assert 0 < destination.stat().st_size <= files.MAX_OUTPUT_BYTES
    page = Page(destination.read_text(encoding="utf8"))
    payload = page.payload()
    assert payload["schema_id"] == "minigrid_inspector_v1" and payload["source"] == recorded
    assert payload["source_label"] == ARTIFACT.name
    assert [condition["key"] for condition in payload["conditions"]] == list(METHODS)
    assert sum(len(condition["episodes"]) for condition in payload["conditions"]) == 20
    for condition in payload["conditions"]:
        for slot in condition["episodes"]:
            assert slot["record"] == pointer(payload["source"], slot["pointer"])
    for _, attributes in page.tags:
        assert not any(
            (attributes.get(field) or "").lower().startswith(("http:", "https:", "//"))
            for field in ("src", "href")
        )


@pytest.mark.parametrize(
    "schema,template",
    [("minigrid_inspector_v1", "minigrid_inspector.html"), (None, "key_door_inspector.html")],
)
def test_should_select_new_template_only_for_exact_minigrid_schema(monkeypatch, schema, template):
    calls = []

    def read(path, *args, **kwargs):
        calls.append(path.name)
        return f'<html data-template="{path.name}"><script id="inspector-data" type="application/json">__INSPECTOR_DATA__</script></html>'

    monkeypatch.setattr(Path, "read_text", read)
    projection: dict[str, Any] = {"source": {}, "conditions": []}
    if schema is not None:
        projection["schema_id"] = schema
    rendered = files.render_key_door_inspector(projection).decode("utf8")
    assert calls == [template]
    assert f'data-template="{template}"' in rendered
    assert Page(rendered).payload() == projection


def test_should_build_saved_minigrid_through_cli_with_inert_script_terminators(
    tmp_path, recorded, monkeypatch, capsys
):
    report = deepcopy(recorded)
    dangerous = '</script><img id="injected" src="x" onerror="alert(1)"> & < > \u2028\u2029'
    report["recorded_note"] = dangerous
    source, destination = tmp_path / "recorded & quoted.json", tmp_path / "offline.html"
    source.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    before = source.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["builder", "--source-artifact", str(source), "--output", str(destination)]
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "output": str(destination.resolve()),
        "source": str(source.resolve()),
        "bytes": destination.stat().st_size,
    }
    assert source.read_bytes() == before
    page = Page(destination.read_text(encoding="utf8"))
    payload = page.payload()
    assert payload["source"] == report
    assert payload["source"]["recorded_note"] == dangerous
    assert not any(attributes.get("id") == "injected" for _, attributes in page.tags)
    embedded = "".join(page.payload_parts)
    assert "<" not in embedded and ">" not in embedded and "&" not in embedded


def test_should_refuse_invalid_causal_history_before_creating_destination_parent(
    tmp_path, recorded
):
    report = deepcopy(recorded)
    report["evaluations"][0]["episodes"][0]["predictions"][1]["features"][0] = 0.2
    source, destination = tmp_path / "corrupt.json", tmp_path / "uncreated" / "inspector.html"
    source.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    before = source.read_bytes()
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, destination)
    assert source.read_bytes() == before and not destination.parent.exists()
