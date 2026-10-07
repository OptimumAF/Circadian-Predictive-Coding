"""Isolated numerical fixture pins for real factor CLI/worker boundary tests.

Inputs are a test-created reference directory and unchanged production adapters.
Outputs are test-local exact pins and a bootstrap command. This helper grants no
historic source/result admission and never modifies production pins on disk.
"""

from hashlib import sha256
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

_BOOTSTRAP = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests'))
from factor_cli_value_fixtures import run_factor_cli_entry
run_factor_cli_entry()
"""


def select_factor_reference(
    monkeypatch: pytest.MonkeyPatch, adapter: ModuleType, directory: Path
) -> None:
    preflight = importlib.import_module(adapter.__name__.replace("development", "preflight"))
    paths = preflight.artifact_paths(directory)
    monkeypatch.setattr(adapter, "REFERENCE_DIR", directory)
    monkeypatch.setattr(
        adapter, "REFERENCE_SHA256", sha256(paths["result"].read_bytes()).hexdigest()
    )
    development = importlib.import_module(
        "src.app.continual_" + adapter.__name__.removeprefix("scripts.run_p63_")
    )
    monkeypatch.setattr(development, "REFERENCE_SHA256", adapter.REFERENCE_SHA256)


def factor_cli_command(module_name: str, directory: Path, *arguments: str) -> list[str]:
    return [sys.executable, "-c", _BOOTSTRAP, module_name, str(directory), *arguments]


def run_factor_cli_entry() -> None:
    module_name, directory = sys.argv[1:3]
    arguments = sys.argv[3:]
    adapter = importlib.import_module(module_name)
    preflight = importlib.import_module(module_name.replace("development", "preflight"))
    setattr(adapter, "REFERENCE_DIR", Path(directory))
    paths = preflight.artifact_paths(Path(directory))
    reference_sha = sha256(paths["result"].read_bytes()).hexdigest()
    setattr(adapter, "REFERENCE_SHA256", reference_sha)
    development = importlib.import_module(
        "src.app.continual_" + module_name.removeprefix("scripts.run_p63_")
    )
    setattr(development, "REFERENCE_SHA256", reference_sha)
    original_run = subprocess.run

    def run_worker(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Why this: the real worker must inherit only these isolated fixture pins.
        if command[1:] == ["-m", module_name, "--worker"]:
            command = factor_cli_command(module_name, Path(directory), "--worker")
        return original_run(command, **kwargs)

    adapter.subprocess.run = run_worker
    sys.argv = [module_name, *arguments]
    adapter.main()
