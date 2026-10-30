"""Zero-model checks for optional dependency declaration and actionable errors."""

from pathlib import Path
import tomllib

import pytest

from src.shared import torch_runtime


def test_should_keep_torch_optional_and_reuse_existing_constraint():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    project = config["project"]
    assert project["dependencies"] == ["numpy>=1.26.4,<2.5"]
    assert project["optional-dependencies"]["native"] == ["torch>=2.2,<3.0"]
    assert "torch>=2.2,<3.0" in (root / "requirements-resnet.txt").read_text().splitlines()
    assert "include docs/native-ticket-setup.md" in (root / "MANIFEST.in").read_text()
    assert config["tool"]["setuptools"]["data-files"] == {
        "share/doc/circadian-predictive-coding": ["docs/native-ticket-setup.md"]
    }


def test_should_explain_installed_native_extra_without_loading_models(monkeypatch):
    requests = []

    def missing(name):
        requests.append(name)
        raise ModuleNotFoundError("torch unavailable")

    monkeypatch.setattr(torch_runtime.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"circadian-predictive-coding\[native\]") as error:
        torch_runtime.require_torch()
    assert "requirements-resnet.txt" in str(error.value)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert requests == ["torch"]


def test_should_leave_an_available_runtime_unchanged(monkeypatch):
    runtime = object()
    monkeypatch.setattr(torch_runtime.importlib, "import_module", lambda name: runtime)
    assert torch_runtime.require_torch() is runtime
