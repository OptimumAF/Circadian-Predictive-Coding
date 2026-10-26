"""Exact saved examples and resource lifetimes; no learner or model imports."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import Path as ZipPath, ZipFile

import pytest

from scripts import build_key_door_inspector as cli
from src.infra.key_door_inspector_files import write_key_door_inspector
from test_key_door_inspector_files import Page

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
SOURCES = {
    "bp": ROOT / "artifacts/runs/bp-new-contexts-20261023/actual.json",
    **{
        m: ROOT / f"artifacts/runs/native-ticket-family-controls-20261023/{m}.json"
        for m in METHODS[1:]
    },
}


@pytest.mark.parametrize("method", METHODS)
def test_exact_checkout_copy_and_public_cli_preserve_full_results(
    method, tmp_path, monkeypatch, capsys
):
    original = SOURCES[method].read_bytes()
    monkeypatch.setattr(
        cli.resources, "files", lambda *a: pytest.fail("checkout used package lookup")
    )
    with cli._example_ticket(method) as source:
        assert source.name == f"{method}.json" and source.read_bytes() == original
    output = tmp_path / f"{method}.html"
    monkeypatch.setattr(
        sys, "argv", ["builder", "--example-ticket", method, "--output", str(output)]
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["source"] == f"bundled saved ticket {method}"
    view = Page(output.read_text(encoding="utf8")).payload()
    assert view["schema_id"] == "ticket_workflow_inspector_v1"
    assert view["source"] == json.loads(original)
    assert SOURCES[method].read_bytes() == original
    assert view["source"]["summary"]["new_context_gain"] == (1 if method == "bp" else 0)
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and output.read_bytes() == before


@pytest.mark.parametrize("method", METHODS)
def test_installed_zip_resource_lifetime_for_real_family(method, tmp_path, monkeypatch):
    archive_path = tmp_path / "installed.zip"
    raw = SOURCES[method].read_bytes()
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{method}.json", raw)
    calls: list[str] = []
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts/builder.py"))
    with ZipFile(archive_path) as archive:

        def files(package):
            calls.append(package)
            assert package == "src.infra.ticket_workflow_data"
            return ZipPath(archive)

        monkeypatch.setattr(cli.resources, "files", files)
        with cli._example_ticket(method) as source:
            assert source.is_file() and sha256(source.read_bytes()).digest() == sha256(raw).digest()
            output = tmp_path / f"installed-{method}.html"
            write_key_door_inspector(source, output)
            assert Page(output.read_text(encoding="utf8")).payload()["source"] == json.loads(raw)
        assert not source.exists()
    assert calls == ["src.infra.ticket_workflow_data"]


@pytest.mark.parametrize("method", ["../bp", "BP", "invented", "", None])
def test_unknown_or_pathlike_method_refused_before_resource_access(method, monkeypatch):
    monkeypatch.setattr(
        cli.resources, "files", lambda *a: pytest.fail("invalid method reached lookup")
    )
    with pytest.raises(ValueError, match="ticket example"):
        with cli._example_ticket(method):
            pytest.fail("invalid method admitted")


def test_missing_zip_resource_fails_without_creating_output(tmp_path, monkeypatch):
    archive_path = tmp_path / "empty.zip"
    with ZipFile(archive_path, "w"):
        pass
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts/builder.py"))
    with ZipFile(archive_path) as archive:
        monkeypatch.setattr(cli.resources, "files", lambda *a: ZipPath(archive))
        with pytest.raises(FileNotFoundError):
            with cli._example_ticket("bp"):
                pytest.fail("missing resource yielded")


def test_corrupt_checkout_example_does_not_fall_back_to_other_evidence(tmp_path, monkeypatch):
    data = tmp_path / "examples/ticket_workflow"
    data.mkdir(parents=True)
    (data / "bp.json").write_text('{"complete":NaN}', encoding="utf8")
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts/builder.py"))
    monkeypatch.setattr(
        cli.resources, "files", lambda *a: pytest.fail("bad checkout silently replaced")
    )
    with cli._example_ticket("bp") as source:
        with pytest.raises(ValueError):
            write_key_door_inspector(source, tmp_path / "bad.html")
    assert not (tmp_path / "bad.html").exists()


@pytest.mark.parametrize(
    "other", [["--example-comparison"], ["--source-artifact", "recorded.json"]]
)
def test_source_choices_are_exclusive_before_any_export(other, monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "write_key_door_inspector", lambda *a, **kw: pytest.fail("ambiguous source exported")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["builder", "--example-ticket", "bp", *other, "--output", str(tmp_path / "bad.html")],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
