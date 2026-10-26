"""KernelPerceptron reuses the existing TRAIN stream; no Torch or environment calls."""

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
from src.adapters.invariant_kernel_perceptron_learner import (
    InvariantKernelPerceptronLearner,
    InvariantKernelPerceptronSnapshot,
    copy_invariant_kernel_perceptron_snapshot,
)
from src.adapters.kernel_perceptron_learner import KernelPerceptronLearner
from src.core.categorical_kernel_perceptron import (
    KernelPerceptronPolicy,
    create_kernel_perceptron_snapshot,
)
from src.core.categorical_kernel_perceptron import KernelPerceptronCapacityExceeded
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
        "InvariantPerceptronLearner",
    ):
        monkeypatch.setattr(fit, name, forbidden)


def test_should_describe_separate_fixed_recipe_without_changing_defaults(monkeypatch):
    monkeypatch.setattr(fit, "AttributionStream", forbidden)
    monkeypatch.setattr(fit, "ProcessRssSampler", forbidden)
    assert fit.describe_protocol() == fit.describe_protocol(kernel_perceptron=False)
    selected = fit.describe_protocol(kernel_perceptron=True, updates=3, max_wall_seconds=10)
    assert selected["definition"] == fit.KERNEL_PERCEPTRON_DEFINITION
    assert selected["method"] == "public_invariant_polynomial_kernel_perceptron"
    assert selected["step_size"] == 1 and selected["training_rows"] == 162
    assert selected["training_decision_rows"] == 162
    assert selected["maximum_prototypes"] == 128
    assert selected["maximum_dictionary_array_equivalent_bytes"] == 310272
    assert selected["kernel"] == "(1 + dot(x,y)/296) ** 2"
    assert selected["checkpoint_updates"] == [0, 3] and selected["score_rows"] == 108
    assert all(key not in selected for key in ("hidden_dim", "learning_rate", "momentum", "seed"))
    assert selected["probe_updates"] == selected["final_episodes"] == selected["replay_rows"] == 0


@pytest.mark.parametrize(
    "options",
    [
        dict(kernel_perceptron=1),
        dict(kernel_perceptron=None),
        dict(kernel_perceptron=True, perceptron=True),
        dict(kernel_perceptron=True, cue_relative=True),
        dict(kernel_perceptron=True, cue_turn_gradients=True),
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
        fit.diagnose_train_fit(changed, kernel_perceptron=True, updates=3)


def test_should_project_once_and_preserve_complete_bound_state():
    source = InvariantKernelPerceptronLearner()
    plain = KernelPerceptronLearner(KernelPerceptronPolicy(296, 7))
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
    source = InvariantKernelPerceptronLearner()
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
            changed = InvariantKernelPerceptronSnapshot(
                KernelPerceptronPolicy(2, 7),
                create_kernel_perceptron_snapshot(KernelPerceptronPolicy(2, 7)),
            )
        with pytest.raises(ValueError):
            copy_invariant_kernel_perceptron_snapshot(changed)
    assert source.snapshot_state() == original


@pytest.mark.parametrize(
    "options",
    [
        ["--kernel-perceptron", "--constructive-bp"],
        ["--kernel-perceptron", "--perceptron"],
        ["--kernel-perceptron", "--bounds-only"],
        ["--kernel-perceptron", "--cue-relative"],
        ["--kernel-perceptron", "--cue-turn-gradients"],
    ],
)
def test_should_refuse_incompatible_cli_before_reading_sources(monkeypatch, options):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", *options, "--describe"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_preview_public_kernel_perceptron_without_reading_sources(monkeypatch, capsys):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(fit, "InvariantKernelPerceptronLearner", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["fit", "--kernel-perceptron", "--describe", "--fit-updates", "3"]
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == fit.describe_protocol(kernel_perceptron=True, updates=3)


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
        assert options == dict(kernel_perceptron=True)
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
            "--kernel-perceptron",
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
    stage = Path("artifacts/runs/kernel-perceptron-fit-20261018")
    label = os.environ.get("KERNEL_CHECK_LABEL")
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
    costs = result["work"]["kernel_perceptron"]
    assert costs["completed_pass_calls"] == 3
    assert costs["completed_training_rows"] == 162
    assert 0 <= costs["completed_mistake_row_updates"] <= 162
    assert costs["backward_calls"] == costs["optimizer_steps"] == 0
    assert 0 < costs["prototype_count"] <= 128
    assert costs["prototype_array_equivalent_bytes"] == costs["prototype_count"] * 296 * 8
    assert costs["coefficient_array_equivalent_bytes"] == costs["prototype_count"] * 7 * 8
    assert 0 < costs["completed_training_kernel_evaluations"] <= 162 * 128
    assert (
        result["work"]["kernel_prediction_pairs_derived_attempted"] == 54 * costs["prototype_count"]
    )
    assert (
        result["work"]["kernel_prediction_pairs_derived_returned"] == 54 * costs["prototype_count"]
    )
    assert result["readouts"][0]["kernel_prototype_count"] == 0
    assert result["readouts"][1]["kernel_prototype_count"] == costs["prototype_count"]
    assert len(sources) == 1 and sources[0].snapshot_state() == initial[0]
    assert pickle.dumps((random.getstate(), np.random.get_state())) == original_rng
    assert source_path.read_bytes() == saved
    assert json.loads(output_path.read_text()) == result
    assert result["protocol"]["final_episodes"] == result["protocol"]["probe_updates"] == 0
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_retain_native_paid_prefix_when_completed_return_is_lost(monkeypatch):
    train = KernelPerceptronLearner.train_batch

    def lost(owner, *args):
        train(owner, *args)
        raise RuntimeError("injected lost completed kernel_perceptron return")

    monkeypatch.setattr(KernelPerceptronLearner, "train_batch", lost)
    result = fit.diagnose_train_fit(
        histories(), kernel_perceptron=True, updates=3, max_wall_seconds=10
    )
    assert (
        result["status"] == "incomplete"
        and "lost completed kernel_perceptron return" in result["reason"]
    )
    assert result["native_completed_updates"] == 1 and result["native_training_rows"] == 54
    assert result["charged_updates"] == result["committed_updates"] == 0
    assert result["work"]["training_work_may_be_unobserved"] is True
    assert result["work"]["kernel_perceptron"]["completed_training_rows"] == 54


def test_should_preserve_typed_capacity_failure_work_without_committing(monkeypatch):
    failure = KernelPerceptronCapacityExceeded(
        attempted_training_rows=54,
        training_rows_evaluated=2,
        kernel_evaluations_completed=1,
        mistake_updates_uncommitted=1,
        prototype_count_before=0,
        prototype_count_proposed=129,
    )

    def refuse(owner, *args):
        raise failure

    monkeypatch.setattr(KernelPerceptronLearner, "train_batch", refuse)
    result = fit.diagnose_train_fit(
        histories(), kernel_perceptron=True, updates=3, max_wall_seconds=10
    )
    assert result["status"] == "incomplete" and "prototype capacity" in result["reason"]
    assert result["native_failure_work"] == failure.work
    assert result["native_completed_updates"] == result["native_training_rows"] == 0
    assert result["charged_updates"] == result["committed_updates"] == 0
    assert result["work"]["training_work_may_be_unobserved"] is True
    assert result["work"]["kernel_perceptron"]["prototype_count"] == 0
