"""Optional installed-game metadata and lazy CLI/errors, without models or physics."""

import json
from pathlib import Path
import sys
import tomllib

import pytest

from scripts import run_coupled_attribution as cli
from src.infra import minigrid_memory as memory

ROOT = Path(__file__).resolve().parents[1]


def test_should_keep_numpy_only_base_and_exact_existing_optional_pins():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == ["numpy>=1.26.4,<2.5"]
    original = [
        line.strip()
        for line in (ROOT / "requirements-minigrid.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert project["optional-dependencies"]["minigrid"] == original
    pins = {line.split("==")[0]: line.split("==")[1] for line in original}
    assert all(pins[name] == version for name, version in memory.VERSIONS.items())


@pytest.mark.parametrize("fault", ["distribution", "module"])
def test_should_explain_installed_extra_and_checkout_setup_without_constructing_environment(
    monkeypatch, fault
):
    def version(name):
        if fault == "distribution":
            raise memory.metadata.PackageNotFoundError(name)
        return memory.VERSIONS[name]

    def import_module(name):
        if fault == "distribution":
            pytest.fail("import before metadata check")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(memory.metadata, "version", version)
    monkeypatch.setattr(memory, "import_module", import_module)
    with pytest.raises(RuntimeError) as error:
        memory._require_minigrid()
    assert "install optional dependencies" in str(error.value)
    assert "minigrid extra" in str(error.value) and "local Circadian wheel" in str(error.value)
    assert "python -m pip install -r requirements-minigrid.txt" in str(error.value)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize("package", tuple(memory.VERSIONS))
def test_should_preserve_exact_version_refusal_before_optional_import(monkeypatch, package):
    monkeypatch.setattr(
        memory.metadata,
        "version",
        lambda name: "0.0.0" if name == package else memory.VERSIONS[name],
    )
    monkeypatch.setattr(memory, "import_module", lambda *a: pytest.fail("optional import"))
    with pytest.raises(ValueError, match="must be pinned"):
        memory._require_minigrid()


def test_should_import_only_original_game_packages_after_pins_pass(monkeypatch):
    gym = object()
    calls = []

    def import_module(name):
        calls.append(name)
        return gym if name == "gymnasium" else object()

    monkeypatch.setattr(memory.metadata, "version", memory.VERSIONS.__getitem__)
    monkeypatch.setattr(memory, "import_module", import_module)
    assert memory._require_minigrid() is gym
    assert calls == ["gymnasium", "minigrid"]


@pytest.mark.parametrize(
    "flag,marker",
    [
        (None, "public_history"),
        ("--invariant-memory", "invariant_history"),
        ("--invariant-fifo-memory", "invariant_fifo36_history"),
    ],
)
def test_should_describe_original_count_modes_without_collection(monkeypatch, capsys, flag, marker):
    args = ["run_coupled_attribution", "--scenario", "minigrid-action-memory", "--describe"]
    if flag is not None:
        args.append(flag)
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert marker in report["protocol_id"]
    assert report["planned_work"]["storage_updates"] == 128
    assert report["planned_work"]["training_rows"] == 1152
