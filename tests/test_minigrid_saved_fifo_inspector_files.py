"""Offline FIFO36 inspector publication uses existing bounded exclusive IO ports."""

import builtins
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import scripts.build_key_door_inspector as cli
from src.infra import key_door_inspector_files as files
from test_key_door_inspector_files import Page
from test_minigrid_saved_fifo_inspector import (
    ARTIFACT,
    lost_helper_report,
    recorded_fifo as recorded_fifo,
)


def test_should_publish_actual_saved_fifo_source_without_modification_and_refuse_overwrite(
    tmp_path, recorded_fifo
):
    before = ARTIFACT.read_bytes()
    destination = tmp_path / "derived" / "fifo36.html"
    written = files.write_key_door_inspector(ARTIFACT, destination)
    assert written.resolve() == destination.resolve()
    assert ARTIFACT.read_bytes() == before
    assert 0 < destination.stat().st_size <= files.MAX_OUTPUT_BYTES
    page = Page(destination.read_text(encoding="utf8"))
    p = page.payload()
    assert p["source"] == recorded_fifo
    assert p["source_label"] == ARTIFACT.name
    assert p["schema_id"] == "minigrid_saved_fifo_inspector_v1"
    assert len(p["conditions"][0]["curves"]) == 64
    assert p["conditions"][0]["episodes"] == []
    for _, attrs in page.tags:
        assert not any(
            (attrs.get(k) or "").lower().startswith(("http:", "https:", "//"))
            for k in ("src", "href")
        )
    rendered = destination.read_bytes()
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(ARTIFACT, destination)
    assert destination.read_bytes() == rendered


@pytest.mark.parametrize(
    "schema,template",
    [
        ("minigrid_saved_fifo_inspector_v1", "minigrid_inspector.html"),
        ("minigrid_inspector_v1", "minigrid_inspector.html"),
        ("minigrid_learnability_inspector_v1", "minigrid_inspector.html"),
        (None, "key_door_inspector.html"),
    ],
)
def test_should_route_fifo_schema_to_existing_minigrid_template_without_changing_old_views(
    monkeypatch, schema, template
):
    calls = []

    def read(path, *args, **kwargs):
        calls.append(path.name)
        return '<script id="inspector-data" type="application/json">__INSPECTOR_DATA__</script>'

    monkeypatch.setattr(Path, "read_text", read)
    p = {"schema_id": schema, "conditions": [], "source": {}}
    assert Page(files.render_key_door_inspector(p).decode("utf8")).payload() == p
    assert calls == [template]


def test_should_build_fifo_cli_with_inert_saved_markup_without_model_or_environment_imports(
    tmp_path, recorded_fifo, monkeypatch, capsys
):
    report = deepcopy(recorded_fifo)
    dangerous = '</script><img id="injected" src="x" onerror="alert(1)"> & < > \u2028\u2029'
    report["recorded_note"] = dangerous
    source, destination = tmp_path / "source & quoted.json", tmp_path / "view.html"
    source.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    before = source.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["builder", "--source-artifact", str(source), "--output", str(destination)]
    )
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torch", "numpy", "gymnasium", "minigrid")
        assert not name.startswith("src.adapters")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    cli.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {
        "output": str(destination.resolve()),
        "source": str(source.resolve()),
        "bytes": destination.stat().st_size,
    }
    assert source.read_bytes() == before
    page = Page(destination.read_text(encoding="utf8"))
    assert page.payload()["source"]["recorded_note"] == dangerous
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert not any(c in "".join(page.payload_parts) for c in "<>&")


def test_should_publish_partial_unknown_curves_without_invented_game_outcomes(
    tmp_path, recorded_fifo
):
    report = lost_helper_report(recorded_fifo)
    source, destination = tmp_path / "partial.json", tmp_path / "partial.html"
    source.write_text(json.dumps(report), encoding="utf8")
    files.write_key_door_inspector(source, destination)
    p = Page(destination.read_text(encoding="utf8")).payload()
    unknown = [row for row in p["conditions"][0]["curves"] if row["source_updates"] == 128]
    assert len(unknown) == 32
    assert all(row["accuracy"] is None and row["pointer"] is None for row in unknown)
    assert p["conditions"][0]["episodes"] == []


@pytest.mark.parametrize(
    "payload",
    [
        b'{"protocol":1,"protocol":2}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1e9999}',
        b"\xff",
    ],
)
def test_should_refuse_ambiguous_nonfinite_or_non_utf8_json_before_publication(tmp_path, payload):
    source, destination = tmp_path / "bad.json", tmp_path / "uncreated" / "view.html"
    source.write_bytes(payload)
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, destination)
    assert not destination.parent.exists()


def test_should_refuse_contradictory_fifo_before_creating_output_directory(tmp_path, recorded_fifo):
    report = deepcopy(recorded_fifo)
    report["cohorts"][1]["phase_charges"]["F"] = 15
    source, destination = tmp_path / "bad.json", tmp_path / "uncreated" / "view.html"
    source.write_text(json.dumps(report), encoding="utf8")
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, destination)
    assert not destination.parent.exists()


def test_should_keep_absolute_sixteen_mib_input_limit_for_saved_fifo(tmp_path, monkeypatch):
    assert files.MAX_LEARNABILITY_INPUT_BYTES == files.MAX_OUTPUT_BYTES == 16 * 1024**2
    monkeypatch.setattr(files, "MAX_LEARNABILITY_INPUT_BYTES", 1024)
    source, destination = tmp_path / "large.json", tmp_path / "uncreated" / "view.html"
    source.write_bytes(b" " * 1025)
    with pytest.raises(ValueError, match="16 MiB"):
        files.write_key_door_inspector(source, destination)
    assert not destination.parent.exists()


def test_should_refuse_output_over_limit_before_publishing_any_file(
    tmp_path, recorded_fifo, monkeypatch
):
    source, destination = ARTIFACT, tmp_path / "uncreated" / "view.html"
    monkeypatch.setattr(files, "MAX_OUTPUT_BYTES", 1024)
    with pytest.raises(ValueError, match="16 MiB"):
        files.write_key_door_inspector(source, destination)
    assert not destination.parent.exists()


def test_should_refuse_source_as_destination_before_reading_it(recorded_fifo):
    before = ARTIFACT.read_bytes()
    with pytest.raises(ValueError, match="differ"):
        files.write_key_door_inspector(ARTIFACT, ARTIFACT)
    assert ARTIFACT.read_bytes() == before
