"""Package-resource and CLI precedence checks without executing a learner.

Real zip resources exercise temporary extraction and the existing bounded JSON
reader. All demo/native entrypoints are doubles: zero model or build work.
"""

import json
from pathlib import Path
from typing import Any
from zipfile import Path as ZipPath, ZipFile

import pytest

from scripts import run_taskspace_memory_demo as cli
from scripts import taskspace_memory_native as native


def forbidden(*args, **kwargs):
    pytest.fail("resource selection reached an unselected loader or real demo")


def resource_zip(path, raw=b'{"resource":"synthetic package data"}'):
    with ZipFile(path, "w") as archive:
        archive.writestr("episodes.json", raw)
    return ZipFile(path)


def select_zip(monkeypatch, archive, tmp_path):
    calls: list[str] = []

    def files(package):
        calls.append(package)
        assert package == "src.infra.taskspace_demo_data"
        return ZipPath(archive)

    monkeypatch.setattr(cli, "DEFAULT_BUNDLE", tmp_path / "absent-source.json")
    monkeypatch.setattr(cli, "files", files)
    return calls


def test_should_prefer_existing_checkout_data_without_accessing_package_resource(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.json"
    source.write_text('{"source":"checkout"}', encoding="utf8")
    monkeypatch.setattr(cli, "DEFAULT_BUNDLE", source)
    monkeypatch.setattr(cli, "files", forbidden)
    assert cli._load_default_bundle() == {"source": "checkout"}


def test_should_extract_zip_resource_only_for_the_bounded_loader_lifetime(monkeypatch, tmp_path):
    paths: list[Path] = []
    load = cli.load_bundle

    def read(path):
        assert path.is_file()
        paths.append(path)
        return load(path)

    with resource_zip(tmp_path / "package.zip") as archive:
        calls = select_zip(monkeypatch, archive, tmp_path)
        monkeypatch.setattr(cli, "load_bundle", read)
        value = cli._load_default_bundle()
    assert value == {"resource": "synthetic package data"}
    assert calls == ["src.infra.taskspace_demo_data"]
    assert len(paths) == 1 and not paths[0].exists()


@pytest.mark.parametrize(
    "raw",
    [b'{"same":1,"same":2}', b'{"value":NaN}', b" " * (512 * 1024 + 1)],
    ids=["duplicate", "nonfinite", "oversized"],
)
def test_should_keep_resource_json_bounds_and_cleanup_on_loader_refusal(monkeypatch, tmp_path, raw):
    paths: list[Path] = []
    load = cli.load_bundle

    def read(path):
        paths.append(path)
        return load(path)

    with resource_zip(tmp_path / "package.zip", raw) as archive:
        select_zip(monkeypatch, archive, tmp_path)
        monkeypatch.setattr(cli, "load_bundle", read)
        with pytest.raises(ValueError):
            cli._load_default_bundle()
    assert len(paths) == 1 and not paths[0].exists()


def test_should_refuse_missing_resource_without_attempting_demo(monkeypatch, tmp_path):
    with ZipFile(tmp_path / "package.zip", "w"):
        pass
    with ZipFile(tmp_path / "package.zip") as archive:
        select_zip(monkeypatch, archive, tmp_path)
        monkeypatch.setattr(cli, "run_demo", forbidden)
        with pytest.raises(FileNotFoundError):
            cli.main([])


def test_should_not_replace_malformed_checkout_data_with_packaged_data(monkeypatch, tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"value":Infinity}', encoding="utf8")
    monkeypatch.setattr(cli, "DEFAULT_BUNDLE", source)
    monkeypatch.setattr(cli, "files", forbidden)
    with pytest.raises(ValueError):
        cli._load_default_bundle()


@pytest.mark.parametrize("native_selected", [False, True])
def test_should_route_explicit_bundle_without_default_resource_access(
    monkeypatch, tmp_path, capsys, native_selected
):
    selected = tmp_path / "selected.json"
    output = tmp_path / "report.json"
    selected.write_text('{"selected":"explicit"}', encoding="utf8")
    calls: list[Any] = []
    monkeypatch.setattr(cli, "_load_default_bundle", forbidden)
    monkeypatch.setattr(cli, "files", forbidden)

    def offline(payload):
        calls.append(("offline", payload))
        return dict(complete=True, selected=payload)

    def fresh(payload, path):
        assert path == output
        calls.append(("native", payload))
        path.write_text("native owns this output", encoding="utf8")
        return dict(complete=True, selected=payload)

    monkeypatch.setattr(cli, "run_demo", offline)
    monkeypatch.setattr(native, "run_native_demo", fresh)
    arguments = ["--bundle", str(selected), "--output", str(output)]
    if native_selected:
        arguments.append("--native")
    assert cli.main(arguments) == 0
    assert calls == [("native" if native_selected else "offline", {"selected": "explicit"})]
    assert json.loads(capsys.readouterr().out)["selected"] == {"selected": "explicit"}
    if native_selected:
        assert output.read_text(encoding="utf8") == "native owns this output"
    else:
        assert json.loads(output.read_text(encoding="utf8"))["complete"] is True


@pytest.mark.parametrize("native_selected", [False, True])
def test_should_use_default_resource_loader_once_for_either_mode(
    monkeypatch, tmp_path, capsys, native_selected
):
    payload = {"default": "resource-owned"}
    calls: list[str] = []
    output = tmp_path / "report.json"

    def default():
        calls.append("default")
        return payload

    def offline(value):
        assert value is payload
        calls.append("offline")
        return dict(complete=True)

    def fresh(value, path):
        assert value is payload and path == output
        calls.append("native")
        path.write_text("native result", encoding="utf8")
        return dict(complete=True)

    monkeypatch.setattr(cli, "_load_default_bundle", default)
    monkeypatch.setattr(cli, "load_bundle", forbidden)
    monkeypatch.setattr(cli, "run_demo", offline)
    monkeypatch.setattr(native, "run_native_demo", fresh)
    arguments = ["--output", str(output)]
    if native_selected:
        arguments.append("--native")
    assert cli.main(arguments) == 0
    assert calls == ["default", "native" if native_selected else "offline"]
    assert json.loads(capsys.readouterr().out) == {"complete": True}


def test_should_refuse_explicit_missing_bundle_without_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_default_bundle", forbidden)
    monkeypatch.setattr(cli, "run_demo", forbidden)
    with pytest.raises(FileNotFoundError):
        cli.main(["--bundle", str(tmp_path / "missing-explicit.json")])


def test_should_refuse_existing_output_before_any_default_resource_read(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    output.write_text("preserve", encoding="utf8")
    monkeypatch.setattr(cli, "_load_default_bundle", forbidden)
    monkeypatch.setattr(cli, "load_bundle", forbidden)
    with pytest.raises(ValueError):
        cli.main(["--output", str(output)])
    assert output.read_text(encoding="utf8") == "preserve"


def test_should_require_native_output_before_any_packaged_resource_read(monkeypatch):
    monkeypatch.setattr(cli, "_load_default_bundle", forbidden)
    monkeypatch.setattr(cli, "load_bundle", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["--native"])
    assert error.value.code == 2
