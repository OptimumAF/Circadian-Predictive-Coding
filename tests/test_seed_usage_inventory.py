"""Whole physical membership, late corruption and no scientific construction."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.infra.seed_usage_inventory import (
    OWNERSHIP_BYTES,
    freeze_seed_metadata_files,
    read_seed_usage_inventory,
)


def corpus(root: Path) -> None:
    (root / "src/config").mkdir(parents=True)
    (root / "src/config/first.json").write_text('{"seeds":[41,43,59]}', encoding="utf-8")
    (root / "last.json").write_text('{"seed":101}', encoding="utf-8")


def test_should_repeat_every_full_file_and_keep_failed_metadata_without_role_authority(
    tmp_path: Path,
) -> None:
    corpus(tmp_path)
    (tmp_path / "failed.json").write_bytes(b'{"seed":1,"seed":2}')
    files = freeze_seed_metadata_files(tmp_path)
    first = read_seed_usage_inventory(tmp_path, files)
    second = read_seed_usage_inventory(tmp_path, files)
    assert first == second
    assert first["file_count"] == 3 and len(first["files"]) == 3
    assert first["declared_seed_values"] == [41, 43, 59, 101]
    assert first["unparsed_file_count"] == 1
    assert first["physical_byte_count"] == sum(f.byte_count for f in files)
    assert first["fresh_roles_authorized"] is first["complete_prior_usage_acceptance"] is False
    assert first["original_p67_acceptance_complete"] is False


@pytest.mark.parametrize(
    "change", ["missing", "extra", "reorder", "late_bytes", "late_identity", "wrong_type"]
)
def test_should_refuse_narrowed_membership_or_late_corruption_before_claiming_inventory(
    tmp_path: Path, change: str
) -> None:
    corpus(tmp_path)
    files: Any = freeze_seed_metadata_files(tmp_path)
    if change == "missing":
        files = files[:-1]
    if change == "extra":
        (tmp_path / "new.json").write_text('{"seed":777}', encoding="utf-8")
    if change == "reorder":
        files = tuple(reversed(files))
    if change == "late_bytes":
        (tmp_path / "src/config/first.json").write_text('{"seed":999}', encoding="utf-8")
    if change == "late_identity":
        files = files[:-1] + (replace(files[-1], sha256="0" * 64),)
    if change == "wrong_type":
        files = list(files)
    with pytest.raises(ValueError):
        read_seed_usage_inventory(tmp_path, files)


def test_should_reject_a_file_changed_during_the_complete_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus(tmp_path)
    files = freeze_seed_metadata_files(tmp_path)
    original = Path.read_bytes
    changed = False

    def read(path: Path) -> bytes:
        nonlocal changed
        raw = original(path)
        if path.name == "first.json" and not changed:
            changed = True
            path.write_bytes(b'{"seeds":[41,43,999]}')
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="changed after"):
        read_seed_usage_inventory(tmp_path, files)


def test_should_ignore_only_runtime_caches_and_the_owned_new_output_directory(
    tmp_path: Path,
) -> None:
    corpus(tmp_path)
    for name in (".venv", "artifacts/runs/p67-untouched-seed-usage"):
        parent = tmp_path / name
        parent.mkdir(parents=True)
        (parent / "output.json").write_text('{"seed":999}', encoding="utf-8")
        if "p67-untouched" in name:
            (parent / "ownership.txt").write_bytes(OWNERSHIP_BYTES)
    runtime = tmp_path / "optional-runtime-snapshot"
    runtime.mkdir()
    (runtime / "pyvenv.cfg").write_text("home = recorded-runtime", encoding="utf-8")
    (runtime / "library.json").write_text('{"seed":995}', encoding="utf-8")
    (tmp_path / "artifacts/runs/older").mkdir()
    (tmp_path / "artifacts/runs/older/failure.JSON").write_text('{"seed":997}', encoding="utf-8")
    result = read_seed_usage_inventory(tmp_path, freeze_seed_metadata_files(tmp_path))
    assert result["file_count"] == 3
    assert 997 in result["declared_seed_values"] and 999 not in result["declared_seed_values"]
    assert 995 not in result["declared_seed_values"]


def test_should_refuse_to_hide_a_preexisting_unowned_output_namespace(tmp_path: Path) -> None:
    corpus(tmp_path)
    output = tmp_path / "artifacts/runs/p67-untouched-seed-usage"
    output.mkdir(parents=True)
    (output / "older.json").write_text('{"seed":999}', encoding="utf-8")
    with pytest.raises(ValueError, match="unowned"):
        freeze_seed_metadata_files(tmp_path)


@pytest.mark.parametrize("part", ["directory", "marker"])
def test_should_refuse_symbolic_owned_paths_before_excluding_their_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, part: str
) -> None:
    corpus(tmp_path)
    output = tmp_path / "artifacts/runs/p67-untouched-seed-usage"
    output.mkdir(parents=True)
    marker = output / "ownership.txt"
    marker.write_bytes(OWNERSHIP_BYTES)
    (output / "older.json").write_text('{"seed":999}', encoding="utf-8")
    original = Path.is_symlink
    target = output if part == "directory" else marker
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    with pytest.raises(ValueError, match="symbolic owned"):
        freeze_seed_metadata_files(tmp_path)


def test_should_inventory_full_corpus_without_source_model_training_scoring_or_final_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus(tmp_path)
    files = freeze_seed_metadata_files(tmp_path)
    from src.app import continual_arrived_benchmark as arrived
    from src.core.backprop_mlp import BackpropMLP
    from src.core.predictive_coding import PredictiveCodingNetwork
    from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork

    def prohibited(*args: Any, **kwargs: Any) -> None:
        pytest.fail("seed metadata inventory accessed science")

    for model in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        for name in ("__init__", "train_epoch", "predict_proba", "compute_accuracy"):
            monkeypatch.setattr(model, name, prohibited)
    for name in ("_build_phase_a_roles", "_build_phase_b_roles", "release_final_test"):
        monkeypatch.setattr(arrived, name, prohibited)
    assert read_seed_usage_inventory(tmp_path, files)["declaration_count"] == 4
