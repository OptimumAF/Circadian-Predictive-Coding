"""Perceptron reuses the existing TRAIN stream; no Torch or environment calls."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_constructive_fit as fit
from src.adapters.invariant_perceptron_learner import (
    InvariantPerceptronLearner,
    InvariantPerceptronSnapshot,
    copy_invariant_perceptron_snapshot,
)
from src.adapters.perceptron_learner import PerceptronLearner
from src.core.categorical_perceptron import PerceptronPolicy, create_perceptron_snapshot
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from test_minigrid_train_fit_bounds import histories, raw_sources
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    for name in (
        "InvariantTorchFeatureLearner",
        "CueRelativeTorchFeatureLearner",
        "_same_torch_state",
    ):
        monkeypatch.setattr(fit, name, forbidden)


def test_should_describe_separate_fixed_recipe_without_changing_defaults(monkeypatch):
    monkeypatch.setattr(fit, "AttributionStream", forbidden)
    monkeypatch.setattr(fit, "ProcessRssSampler", forbidden)
    assert fit.describe_protocol() == fit.describe_protocol(perceptron=False)
    selected = fit.describe_protocol(perceptron=True, updates=3, max_wall_seconds=10)
    assert selected["definition"] == fit.PERCEPTRON_DEFINITION
    assert selected["method"] == "public_invariant_perceptron"
    assert selected["step_size"] == 1 and selected["training_rows"] == 162
    assert (
        selected["training_decision_rows"] == 162
        and selected["coefficient_array_equivalent_bytes"] == 16632
    )
    assert selected["checkpoint_updates"] == [0, 3] and selected["score_rows"] == 108
    assert all(key not in selected for key in ("hidden_dim", "learning_rate", "momentum", "seed"))
    assert selected["probe_updates"] == selected["final_episodes"] == selected["replay_rows"] == 0


@pytest.mark.parametrize(
    "options",
    [
        dict(perceptron=1),
        dict(perceptron=None),
        dict(perceptron=True, cue_relative=True),
        dict(perceptron=True, cue_turn_gradients=True),
    ],
)
def test_should_refuse_invalid_family_before_construction(monkeypatch, options):
    monkeypatch.setattr(fit, "_learner", forbidden)
    monkeypatch.setattr(fit, "ProcessRssSampler", forbidden)
    for operation in (
        fit.describe_protocol,
        lambda **kw: fit.diagnose_train_fit(histories(), **kw),
    ):
        with pytest.raises(ValueError):
            operation(**options)


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_reserved_roles_before_any_learner(monkeypatch, role):
    sources = histories()
    changed = (replace(sources[0], role=role), *sources[1:])
    monkeypatch.setattr(fit, "_learner", forbidden)
    monkeypatch.setattr(fit, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(changed, perceptron=True, updates=3)


def test_should_project_once_and_preserve_complete_bound_state():
    source = InvariantPerceptronLearner()
    plain = PerceptronLearner(PerceptronPolicy(296, 7))
    features, targets = episode_arrays(histories()[0])
    canonical = np.array(
        [canonicalize_public_history(tuple(float(x) for x in row)) for row in features]
    )
    original = source.snapshot_state()
    assert source.train_batch(features, targets) == plain.train_batch(canonical, targets)
    state = source.snapshot_state()
    assert state.native_state == plain.snapshot_state()
    assert np.array_equal(source.predict(features), plain.predict(canonical))
    child = source.fork()
    child.restore_state(original)
    assert source.snapshot_state() == state and child.snapshot_state() == original
    changed = deepcopy(state)
    object.__setattr__(changed, "encoding_id", "foreign")
    with pytest.raises(ValueError):
        source.restore_state(changed)
    assert source.snapshot_state() == state
    with pytest.raises(ValueError):
        source.restore_state(plain.snapshot_state())
    assert source.snapshot_state() == state


@pytest.mark.parametrize("fault", ["later_invalid_frame", "rogue_snapshot", "different_policy"])
def test_should_refuse_invalid_wrapper_atomically(fault):
    source = InvariantPerceptronLearner()
    original = source.snapshot_state()
    if fault == "later_invalid_frame":
        features, targets = episode_arrays(histories()[0])
        features[-1, -1] = 0.123
        with pytest.raises(ValueError):
            source.train_batch(features, targets)
    else:
        changed = deepcopy(original)
        if fault == "rogue_snapshot":
            object.__setattr__(changed, "rogue", True)
        else:
            changed = InvariantPerceptronSnapshot(
                PerceptronPolicy(2, 7), create_perceptron_snapshot(PerceptronPolicy(2, 7))
            )
        with pytest.raises(ValueError):
            copy_invariant_perceptron_snapshot(changed)
    assert source.snapshot_state() == original


@pytest.mark.parametrize(
    "options",
    [
        ["--perceptron", "--constructive-bp"],
        ["--perceptron", "--bounds-only"],
        ["--perceptron", "--cue-relative"],
        ["--perceptron", "--cue-turn-gradients"],
    ],
)
def test_should_refuse_incompatible_cli_before_reading_sources(monkeypatch, options):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", *options, "--describe"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_preview_public_perceptron_without_reading_sources(monkeypatch, capsys):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(fit, "InvariantPerceptronLearner", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", "--perceptron", "--describe", "--fit-updates", "3"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == fit.describe_protocol(perceptron=True, updates=3)


def test_should_run_public_three_pass_train_stream_with_separate_source_and_costs(
    monkeypatch,
    tmp_path,
    capsys,
):
    source_path, output_path = tmp_path / "sources.json", tmp_path / "fit.json"
    source_path.write_text(
        json.dumps(
            dict(
                training_sources=[asdict(e) for e in raw_sources()],
                final_test="unavailable",
                probe_data="unavailable",
            )
        )
    )
    saved = source_path.read_bytes()
    original_rng = pickle.dumps((random.getstate(), np.random.get_state()))
    sources, initial = [], []
    factory = fit._learner

    def learner(**options):
        assert options == dict(perceptron=True)
        result = factory(**options)
        sources.append(result)
        initial.append(result.snapshot_state())
        return result

    monkeypatch.setattr(fit, "_learner", learner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--perceptron",
            "--fit-updates",
            "3",
            "--fit-seconds",
            "10",
            "--source",
            str(source_path),
            "--output",
            str(output_path),
        ],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    stage = Path("artifacts/runs/perceptron-train-fit-20261018")
    label = os.environ.get("PERCEPTRON_CHECK_LABEL")
    report_path = (
        stage / (label + "-synthetic-public-report.json")
        if label
        else tmp_path / "public-report.json"
    )
    with report_path.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
    assert result["status"] == "complete", result["reason"]
    assert (
        result["charged_updates"]
        == result["committed_updates"]
        == result["native_completed_updates"]
        == 3
    )
    assert result["native_training_rows"] == 162
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert [p["updates"] for p in result["readouts"]] == [0, 3]
    assert result["work"]["score_forward_calls_completed"] == 12
    assert result["work"]["score_rows_completed"] == 108
    costs = result["work"]["perceptron"]
    assert costs["completed_pass_calls"] == 3
    assert costs["completed_training_rows"] == costs["completed_online_score_rows"] == 162
    assert 0 <= costs["completed_mistake_row_updates"] <= 162
    assert costs["backward_calls"] == costs["optimizer_steps"] == 0
    assert len(sources) == 1 and sources[0].snapshot_state() == initial[0]
    assert pickle.dumps((random.getstate(), np.random.get_state())) == original_rng
    assert source_path.read_bytes() == saved
    assert json.loads(output_path.read_text()) == result
    assert result["protocol"]["final_episodes"] == result["protocol"]["probe_updates"] == 0
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_retain_native_paid_prefix_when_completed_return_is_lost(monkeypatch):
    train = PerceptronLearner.train_batch

    def lost(owner, *args):
        train(owner, *args)
        raise RuntimeError("injected lost completed perceptron return")

    monkeypatch.setattr(PerceptronLearner, "train_batch", lost)
    result = fit.diagnose_train_fit(histories(), perceptron=True, updates=3, max_wall_seconds=10)
    assert (
        result["status"] == "incomplete" and "lost completed perceptron return" in result["reason"]
    )
    assert result["native_completed_updates"] == 1 and result["native_training_rows"] == 54
    assert result["charged_updates"] == result["committed_updates"] == 0
    assert result["work"]["training_work_may_be_unobserved"] is True
    assert result["work"]["perceptron"]["completed_training_rows"] == 54
