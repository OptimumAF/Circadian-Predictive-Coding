"""Opt-in recurring representation selection and resource-free frozen previews."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_action_memory_pilot as pilot
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("describe", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_should_forward_only_explicit_invariant_memory_selection(
    monkeypatch, capsys, describe, enabled
):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_action_memory_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(
        cli, "run_minigrid_action_memory_pilot", forbidden if describe else selected
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--scenario",
            "minigrid-action-memory",
            *(["--invariant-memory"] if enabled else []),
            *(["--describe"] if describe else []),
        ],
    )
    cli.main()
    assert calls == [dict(invariant_memory=True) if enabled else {}]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "scenario", ["single", "torch-minigrid-cue-key-door", "torch-minigrid-history-learnability"]
)
def test_should_refuse_representation_flag_on_other_scenarios_before_dispatch(
    monkeypatch, scenario
):
    for name in (
        "describe_minigrid_action_memory_protocol",
        "run_minigrid_action_memory_pilot",
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", scenario, "--invariant-memory", "--describe"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_retain_negative_result_and_exit_two(monkeypatch, capsys):
    failed = dict(status="incomplete", reason="fixed count table refused work", observations=[])
    monkeypatch.setattr(cli, "run_minigrid_action_memory_pilot", lambda **options: failed)
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", "minigrid-action-memory", "--invariant-memory"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out) == failed


def test_should_preserve_raw_default_descriptor_exactly():
    assert pilot.describe_protocol() == pilot.describe_protocol(invariant_memory=False)
    assert pilot.describe_protocol()["methods"] == [pilot.METHOD]


def test_should_describe_bound_refusal_memory_and_original_recurring_dose_without_ports(
    monkeypatch,
):
    for name in ("ProcessRssSampler", "ActionMemoryLearner", "_invariant_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    raw = pilot.describe_protocol()
    bound = pilot.describe_protocol(invariant_memory=True)
    assert raw != bound and bound["methods"] == [pilot.INVARIANT_METHOD]
    assert bound["policy"]["encoding_id"] == INVARIANT_HISTORY_ENCODING_ID
    assert bound["policy"]["native_policy"] == raw["policy"]
    assert bound["task_sequence"] == raw["task_sequence"] == ["A", "B", "C_train", "A"]
    assert bound["checkpoint_updates"] == raw["checkpoint_updates"] == [0, 32, 64, 96, 128]
    assert bound["planned_work"]["storage_updates"] == 128
    assert bound["planned_work"]["training_rows"] == 1152
    assert bound["planned_work"]["raw_score_forwards"] == 30
    assert bound["planned_work"]["raw_score_rows"] == 270
    assert bound["planned_work"]["live_episodes"] == 30
    assert bound["limits"] == raw["limits"] and bound["trainable_parameters"] is None
