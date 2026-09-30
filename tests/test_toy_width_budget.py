"""An opt-in toy cap includes transient circadian split width."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.app import experiment_runner
from src.app.experiment_runner import ExperimentConfig, run_experiment
from src.app.toy_execution_budget import ToyExecutionBudget, ToyExecutionStopped
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    HiddenWidthLimitExceeded,
)
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal
from src.core.sleep_clocks import SleepEpochProgress
from src.infra.circadian_checkpoint_files import TrustedLocalToyCheckpointStore


def _width_config(epochs: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        sample_count=80,
        hidden_dim=4,
        epoch_count=epochs,
        pc_inference_steps=2,
        circadian_inference_steps=2,
        circadian_sleep_interval=1,
        random_seed=29,
        circadian_config=CircadianConfig(
            sleep_mode="components",
            split_threshold=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            replay_steps=0,
        ),
    )


def _assert_same_scores_and_width(actual: Any, expected: Any) -> None:
    assert actual.split_hashes == expected.split_hashes
    for name in ("backprop", "predictive_coding", "circadian_predictive_coding"):
        left, right = getattr(actual, name), getattr(expected, name)
        assert left.loss_history == right.loss_history
        assert left.validation_accuracy == right.validation_accuracy
        assert left.test_accuracy == right.test_accuracy
    assert actual.circadian_sleep.hidden_dim_end == expected.circadian_sleep.hidden_dim_end
    assert [event.changes for event in actual.circadian_sleep.events] == [
        event.changes for event in expected.circadian_sleep.events
    ]


def test_core_should_reject_transient_width_even_when_final_width_would_fit() -> None:
    config = CircadianConfig(
        sleep_mode="components",
        split_threshold=0.5,
        prune_threshold=0.1,
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        prune_decay_steps=1,
        replay_steps=0,
    )
    model = CircadianPredictiveCodingNetwork(
        2, 5, seed=37, circadian_config=config, min_hidden_dim=4
    )
    model.set_chemical_state(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    before_weights = model.weight_input_hidden.copy()
    before_lineage = model.get_neuron_lineage()
    before_clocks = model.get_sleep_clocks()
    progress = SleepEpochProgress(7, 10)

    with pytest.raises(HiddenWidthLimitExceeded) as caught:
        model.sleep_event(epoch_progress=progress, max_hidden_width=5)

    assert (caught.value.proposed_width, caught.value.max_hidden_width) == (6, 5)
    assert model.hidden_dim == 5
    assert model.get_neuron_lineage() == before_lineage
    assert model.get_sleep_clocks() == before_clocks
    np.testing.assert_array_equal(model.weight_input_hidden, before_weights)
    applied = model.sleep_event(epoch_progress=progress, max_hidden_width=6)
    assert (applied.old_hidden_dim, applied.new_hidden_dim) == (5, 5)
    assert len(applied.split_indices) == len(applied.pruned_indices) == 1


def test_core_should_bound_final_external_proposal_before_mutation() -> None:
    model = CircadianPredictiveCodingNetwork(2, 4, seed=37)
    proposal = [NeuronChangeProposal(layer_name="hidden", add_count=1)]

    with pytest.raises(HiddenWidthLimitExceeded) as caught:
        model.apply_neuron_proposals(proposal, max_hidden_width=4)

    assert caught.value.proposed_width == 5
    assert model.hidden_dim == 4
    model.apply_neuron_proposals(proposal, max_hidden_width=5)
    assert model.hidden_dim == 5


def test_should_stop_before_sleep_without_opening_final_role_then_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _width_config()
    store = TrustedLocalToyCheckpointStore(tmp_path / "width.checkpoint")
    original_split = experiment_runner.split_training_validation
    final_reads = 0

    class SealedFinal:
        @property
        def input(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("width stop opened final input")

        @property
        def target(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("width stop opened final labels")

    def sealed_split(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        return SimpleNamespace(
            train=roles.train,
            validation=roles.validation,
            test=SealedFinal(),
            split_hashes=roles.split_hashes,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(experiment_runner, "split_training_validation", sealed_split)
        with pytest.raises(ToyExecutionStopped) as stopped:
            run_experiment(
                config,
                checkpoint_store=store,
                execution_budget=ToyExecutionBudget(max_hidden_width=4),
            )

    stop = stopped.value.stop
    assert (stop.reason, stop.updates_completed) == ("max_hidden_width", 3)
    assert (
        stop.hidden_width_observed,
        stop.peak_hidden_width_observed,
        stop.proposed_hidden_width,
    ) == (4, 4, 5)
    assert stop.checkpoint_position == store.load().combined.position
    assert stop.checkpoint_position is not None
    assert stop.checkpoint_position.stage == "before_sleep"
    assert final_reads == 0
    _assert_same_scores_and_width(
        run_experiment(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
            execution_budget=ToyExecutionBudget(max_hidden_width=5),
        ),
        run_experiment(config),
    )


def test_should_complete_at_exact_width_and_restore_peak() -> None:
    config = _width_config()
    actual = run_experiment(config, execution_budget=ToyExecutionBudget(max_hidden_width=5))
    expected = run_experiment(config)

    _assert_same_scores_and_width(actual, expected)
    assert actual.circadian_sleep.hidden_dim_end == 5


def test_should_refuse_initial_and_resumed_width_over_cap(tmp_path: Path) -> None:
    config = _width_config()
    with pytest.raises(ToyExecutionStopped) as initial:
        run_experiment(config, execution_budget=ToyExecutionBudget(max_hidden_width=3))
    assert (initial.value.stop.updates_completed, initial.value.stop.hidden_width_observed) == (
        0,
        4,
    )
    assert initial.value.stop.checkpoint_position is None

    store = TrustedLocalToyCheckpointStore(tmp_path / "width.checkpoint")
    with pytest.raises(ToyExecutionStopped) as first:
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_hidden_width=5, max_training_updates=3),
        )
    assert first.value.stop.reason == "max_training_updates"
    assert first.value.stop.hidden_width_observed == 5
    assert first.value.stop.peak_hidden_width_observed == 5
    before = store.path.read_bytes()

    with pytest.raises(ToyExecutionStopped) as resumed:
        run_experiment(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
            execution_budget=ToyExecutionBudget(max_hidden_width=4, max_training_updates=6),
        )

    assert (resumed.value.stop.reason, resumed.value.stop.hidden_width_observed) == (
        "max_hidden_width",
        5,
    )
    assert resumed.value.stop.checkpoint_position == store.load().combined.position
    assert store.path.read_bytes() == before


def test_should_refuse_lower_cap_below_saved_transient_peak(tmp_path: Path) -> None:
    config = ExperimentConfig(
        sample_count=80,
        epoch_count=2,
        circadian_sleep_interval=1,
        circadian_config=CircadianConfig(split_threshold=0.0),
    )
    store = TrustedLocalToyCheckpointStore(tmp_path / "transient.checkpoint")
    with pytest.raises(ToyExecutionStopped) as first:
        run_experiment(
            config,
            checkpoint_store=store,
            execution_budget=ToyExecutionBudget(max_hidden_width=14, max_training_updates=3),
        )
    assert first.value.stop.reason == "max_training_updates"
    assert (
        first.value.stop.hidden_width_observed,
        first.value.stop.peak_hidden_width_observed,
    ) == (12, 14)
    before = store.path.read_bytes()

    with pytest.raises(ToyExecutionStopped) as resumed:
        run_experiment(
            config,
            checkpoint_store=store,
            resume_from_checkpoint=True,
            execution_budget=ToyExecutionBudget(max_hidden_width=12, max_training_updates=6),
        )

    assert (resumed.value.stop.reason, resumed.value.stop.proposed_hidden_width) == (
        "max_hidden_width",
        14,
    )
    assert resumed.value.stop.hidden_width_observed == 12
    assert resumed.value.stop.peak_hidden_width_observed == 14
    assert store.path.read_bytes() == before


def test_should_allow_no_growth_at_initial_width_cap() -> None:
    config = _width_config(epochs=1)
    circadian = config.circadian_config
    assert circadian is not None
    config = replace(config, circadian_config=replace(circadian, max_split_per_sleep=0))
    _assert_same_scores_and_width(
        run_experiment(config, execution_budget=ToyExecutionBudget(max_hidden_width=4)),
        run_experiment(config),
    )


def test_should_bound_final_external_proposal_before_final_scoring() -> None:
    config = replace(_width_config(epochs=1), circadian_sleep_interval=0)

    class AddCircadianOnly:
        def propose(self, traffic: list[LayerTraffic]) -> list[NeuronChangeProposal]:
            if any(layer.layer_name == "chemical" for layer in traffic):
                return [NeuronChangeProposal(layer_name="hidden", add_count=1)]
            return []

    with pytest.raises(ToyExecutionStopped) as stopped:
        run_experiment(
            config,
            adaptation_policy=AddCircadianOnly(),
            execution_budget=ToyExecutionBudget(max_hidden_width=4),
        )

    assert (stopped.value.stop.reason, stopped.value.stop.proposed_hidden_width) == (
        "max_hidden_width",
        5,
    )
    assert stopped.value.stop.checkpoint_position is None
    completed = run_experiment(
        config,
        adaptation_policy=AddCircadianOnly(),
        execution_budget=ToyExecutionBudget(max_hidden_width=5),
    )
    assert completed.circadian_sleep.hidden_dim_end == 5


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, float("nan")])
def test_should_reject_invalid_width_caps(invalid: object) -> None:
    with pytest.raises(ValueError, match="max_hidden_width"):
        ToyExecutionBudget(max_hidden_width=cast(Any, invalid))
