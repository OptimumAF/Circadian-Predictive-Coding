"""Regression checks for versioned hardest-mode dynamics evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib import import_module
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.datasets import LabeledData, make_role_separated_dataset

script = import_module("scripts.generate_hardest_mode_dynamics")


def tiny_config() -> Any:
    return script.HardestModeConfig(
        sample_count_phase_a=40, sample_count_phase_b=40,
        hidden_dim=4, hidden_dims=(4,), phase_a_epochs=2, phase_b_epochs=2,
        phase_b_train_fraction=0.5, snapshot_interval=1,
        decision_grid_size=8, latency_repeats=1,
        sleep_interval_phase_a=0, sleep_interval_phase_b=0,
    )


def test_validation_dynamics_uses_disjoint_holdout_and_versions_payload(
    tmp_path: Path,
) -> None:
    config = tiny_config()
    run = script.collect_hardest_mode_snapshots(config)
    phase_a, phase_b = script.build_validation_datasets(config)

    assert run.protocol_id == script.VALIDATION_PROTOCOL
    assert run.evaluation_split == "validation"
    assert len(run.snapshots) == 4
    assert np.array_equal(run.phase_b_evaluation.input, phase_b.validation.input)
    assert run.split_hashes["phase_a_train"] == phase_a.split_hashes["train"]
    assert run.split_hashes["phase_b_validation"] == phase_b.split_hashes["validation"]
    assert all(0 <= value <= 1 for value in run.final_test_accuracy.values())

    payload = script.build_interactive_payload(
        snapshots=run.snapshots,
        phase_b_evaluation_input=run.phase_b_evaluation.input,
        phase_b_evaluation_target=run.phase_b_evaluation.target,
        x_bounds=run.x_bounds, y_bounds=run.y_bounds,
        evaluation_split=run.evaluation_split, protocol_id=run.protocol_id,
        split_hashes=run.split_hashes, final_test_accuracy=run.final_test_accuracy,
    )
    assert payload["protocol_id"] == script.VALIDATION_PROTOCOL
    assert payload["evaluation_split"] == "validation"
    assert payload["training_metric_ids"]["Predictive"] == "numpy_pc_bce_plus_half_mean_all_hidden_error_sq_v1"
    assert payload["training_metric_ids"]["Circadian"] == "numpy_circadian_bce_plus_half_mean_final_hidden_error_sq_v1"
    assert "phase_b_test_labels" not in payload
    assert len(payload["phase_b_evaluation_labels"]) == phase_b.validation.input.shape[0]

    gif_path = tmp_path / "dynamics.gif"
    html_path = tmp_path / "dynamics.html"
    script.render_hardest_mode_gif(
        snapshots=run.snapshots,
        phase_b_evaluation_input=run.phase_b_evaluation.input,
        phase_b_evaluation_target=run.phase_b_evaluation.target,
        x_bounds=run.x_bounds, y_bounds=run.y_bounds,
        output_path=gif_path, frame_duration_ms=20,
        evaluation_split=run.evaluation_split,
    )
    script.write_interactive_hardest_mode_html(
        snapshots=run.snapshots,
        phase_b_evaluation_input=run.phase_b_evaluation.input,
        phase_b_evaluation_target=run.phase_b_evaluation.target,
        x_bounds=run.x_bounds, y_bounds=run.y_bounds,
        output_path=html_path, evaluation_split=run.evaluation_split,
        protocol_id=run.protocol_id, split_hashes=run.split_hashes,
        final_test_accuracy=run.final_test_accuracy,
    )
    assert gif_path.stat().st_size > 0
    html = html_path.read_text(encoding="utf-8")
    assert "Intermediate accuracy uses Phase-B validation data" in html
    assert "payload.phase_b_evaluation_labels" in html
    assert '"protocol_id": "validation_dynamics_v1"' in html
    assert '"training_metric_ids"' in html


def test_final_test_scoring_waits_for_all_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config()
    test_targets: list[Any] = []
    original_builder = script.build_validation_datasets

    def capture_split(config_arg: Any) -> Any:
        phase_a, phase_b = original_builder(config_arg)
        test_targets.append(phase_b.test.target)
        return phase_a, phase_b

    monkeypatch.setattr(script, "build_validation_datasets", capture_split)
    classes = (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork)
    counts = {model_class.__name__: 0 for model_class in classes}
    test_scores = {model_class.__name__: 0 for model_class in classes}
    for model_class in classes:
        train_epoch = model_class.train_epoch
        compute_accuracy = model_class.compute_accuracy
        name = model_class.__name__

        def tracked_train(self: Any, *args: Any, _original: Any = train_epoch,
                          _name: str = name, **kwargs: Any) -> Any:
            counts[_name] += 1
            return _original(self, *args, **kwargs)

        def tracked_score(self: Any, inputs: Any, targets: Any,
                          _original: Any = compute_accuracy, _name: str = name) -> float:
            if targets is test_targets[0]:
                assert all(count == 4 for count in counts.values())
                test_scores[_name] += 1
            return _original(self, inputs, targets)

        monkeypatch.setattr(model_class, "train_epoch", tracked_train)
        monkeypatch.setattr(model_class, "compute_accuracy", tracked_score)

    script.collect_hardest_mode_snapshots(config)
    assert counts == {name: 4 for name in counts}
    assert test_scores == {name: 1 for name in test_scores}


def test_corrected_dynamics_training_boundary_cannot_open_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_builder = script.build_validation_datasets
    original_train = script._train_dynamics_models
    training_complete = False
    final_reads = 0

    class SealedFinalTest:
        def __init__(self, actual: Any) -> None:
            self.actual = actual

        @property
        def input(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("Dynamics training opened final-test inputs")
            final_reads += 1
            return self.actual.input

        @property
        def target(self) -> Any:
            nonlocal final_reads
            if not training_complete:
                raise AssertionError("Dynamics training opened final-test labels")
            final_reads += 1
            return self.actual.target

    def sealed_builder(config: Any) -> Any:
        phase_a, phase_b = original_builder(config)
        sealed_b = SimpleNamespace(
            train=phase_b.train, validation=phase_b.validation,
            test=SealedFinalTest(phase_b.test), split_hashes=phase_b.split_hashes,
        )
        return phase_a, sealed_b

    def train_without_test(**kwargs: Any) -> Any:
        nonlocal training_complete
        assert set(kwargs) == {
            "config", "phase_a_train", "phase_b_train", "phase_b_evaluation",
            "x_bounds", "y_bounds",
        }
        outcome = original_train(**kwargs)
        training_complete = True
        return outcome

    monkeypatch.setattr(script, "build_validation_datasets", sealed_builder)
    monkeypatch.setattr(script, "_train_dynamics_models", train_without_test)

    result = script.collect_hardest_mode_snapshots(tiny_config())

    assert training_complete
    assert final_reads == 6
    assert len(result.snapshots) == 4


def test_explicit_legacy_protocol_preserves_test_informed_snapshot_route() -> None:
    config = replace(tiny_config(), protocol_id=script.LEGACY_PROTOCOL)
    run = script.collect_hardest_mode_snapshots(config)
    _, legacy_phase_b = script.build_datasets(config)

    assert run.protocol_id == script.LEGACY_PROTOCOL
    assert run.evaluation_split.startswith("test (legacy")
    assert run.split_hashes == {}
    assert np.array_equal(run.phase_b_evaluation.input, legacy_phase_b.test_input)
    assert len(run.snapshots) == 4


def test_changed_final_test_labels_leave_training_and_validation_dynamics_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config()
    original_builder = script.build_validation_datasets
    model_classes = {
        "BackpropMLP": BackpropMLP,
        "PredictiveCodingNetwork": PredictiveCodingNetwork,
        "CircadianPredictiveCodingNetwork": CircadianPredictiveCodingNetwork,
    }

    def run_with_shift(shift: bool) -> tuple[Any, dict[str, bytes]]:
        created: dict[str, Any] = {}
        for name, model_class in model_classes.items():
            def capture(*args: Any, _class: Any = model_class,
                        _name: str = name, **kwargs: Any) -> Any:
                model = _class(*args, **kwargs)
                created[_name] = model
                return model
            monkeypatch.setattr(script, name, capture)

        def builder(config_arg: Any) -> Any:
            phase_a, phase_b = original_builder(config_arg)
            if not shift:
                return phase_a, phase_b
            changed = make_role_separated_dataset(
                train=phase_b.train, validation=phase_b.validation,
                test=LabeledData(phase_b.test.input, 1.0 - phase_b.test.target),
            )
            return phase_a, changed

        monkeypatch.setattr(script, "build_validation_datasets", builder)
        run = script.collect_hardest_mode_snapshots(config)
        return run, {key: pickle.dumps(model) for key, model in created.items()}

    original_run, original_models = run_with_shift(False)
    shifted_run, shifted_models = run_with_shift(True)
    assert original_models == shifted_models
    assert original_run.split_hashes["phase_b_train"] == shifted_run.split_hashes["phase_b_train"]
    assert original_run.split_hashes["phase_b_validation"] == shifted_run.split_hashes["phase_b_validation"]
    assert original_run.split_hashes["phase_b_test"] != shifted_run.split_hashes["phase_b_test"]
    for original, shifted in zip(original_run.snapshots, shifted_run.snapshots):
        assert original.backprop_phase_b_accuracy == shifted.backprop_phase_b_accuracy
        assert original.predictive_phase_b_accuracy == shifted.predictive_phase_b_accuracy
        assert original.circadian_phase_b_accuracy == shifted.circadian_phase_b_accuracy
        assert np.array_equal(original.decision_map, shifted.decision_map)


def test_dynamics_command_refuses_to_overwrite_historical_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    old_gif = tmp_path / "historical.gif"
    old_gif.write_bytes(b"historical")
    monkeypatch.setattr(script, "parse_args", lambda: argparse.Namespace(
        gif_output_path=str(old_gif),
        interactive_output_path=str(tmp_path / "new.html"),
        protocol_id=script.VALIDATION_PROTOCOL,
        seed=7, snapshot_interval=1, gif_duration_ms=20,
    ))
    monkeypatch.setattr(
        script, "collect_hardest_mode_snapshots",
        lambda config: pytest.fail("Training must not start before checking outputs"),
    )

    with pytest.raises(FileExistsError, match="Dynamics output already exists"):
        script.main()
    assert old_gif.read_bytes() == b"historical"
