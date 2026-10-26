"""Selected original TRAIN-row gradients and constructive runtime composition.

Independent NumPy derivatives and finite differences check the diagnostic. Real
gradient reads make zero SGD updates; constructive integration uses the existing
fake learner and ordinary stream, with scripted diagnostic ports.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import random
import sys
from typing import Any, cast

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_constructive_fit as fit
from src.adapters import torch_feature_gradient_diagnostics as diagnostics
from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch
from test_minigrid_constructive_fit import fake_runtime as fake_runtime
from test_minigrid_learning_pilot import Learner
from test_minigrid_train_fit_bounds import histories, raw_sources
from test_torch_feature_gradient_diagnostics import (
    METHODS,
    NAMES,
    expected_geometry,
    originals as originals,
    pair,
)
from test_torch_feature_learner import assert_tree
from test_torch_feature_rehearsal_pilot import forbidden


def turn_pair():
    sources = histories()[:2]
    return tuple(
        replace(
            episode,
            transitions=tuple(
                replace(step, action=index) if step.step == 2 else step
                for step in episode.transitions
            ),
        )
        for index, episode in enumerate(sources)
    )


def metadata_snapshot():
    return InvariantTorchFeatureSnapshot(
        InvariantTorchFeaturePolicy(native.TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7)),
        cast(Any, None),
    )


@pytest.fixture
def gradient_budget(monkeypatch, record_property):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    monkeypatch.setattr(native.TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    work = dict(forwards=0, forward_rows=0, backwards=0)
    logits, grad = diagnostics._prior_logits, torch.autograd.grad

    def forward(parameters, values, backend):
        work["forwards"] += 1
        work["forward_rows"] += len(values)
        return logits(parameters, values, backend)

    def backward(*args, **kwargs):
        work["backwards"] += 1
        return grad(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "_prior_logits", forward)
    monkeypatch.setattr(torch.autograd, "grad", backward)
    yield work
    for name, value in work.items():
        record_property(f"cue_turn_{name}", value)
    record_property("cue_turn_native_updates", 0)
    record_property("cue_turn_optimizer_steps", 0)


def numpy_selected(parameters, episode, indices):
    x = np.array([canonicalize_public_history(episode.transitions[i].observation) for i in indices])
    labels = np.array([episode.transitions[i].action for i in indices])
    hidden = np.tanh(x @ parameters[NAMES[0]] + parameters[NAMES[1]])
    logits = hidden @ parameters[NAMES[2]] + parameters[NAMES[3]]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    normalizer = exponentials.sum(axis=1, keepdims=True)
    loss = float(-(shifted - np.log(normalizer))[np.arange(len(labels)), labels].mean())
    residual = exponentials / normalizer
    residual[np.arange(len(labels)), labels] -= 1
    residual /= len(labels)
    hidden_error = (residual @ parameters[NAMES[2]].T) * (1 - hidden * hidden)
    return loss, dict(
        zip(
            NAMES,
            (
                x.T @ hidden_error,
                hidden_error.sum(axis=0, keepdims=True),
                hidden.T @ residual,
                residual.sum(axis=0, keepdims=True),
            ),
            strict=True,
        )
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_match_selected_row_derivatives_finite_differences_and_preserve_state(
    originals,
    method,
    monkeypatch,
    gradient_budget,
):
    torch = require_torch()
    snapshot, episodes = deepcopy(originals[method]), turn_pair()
    head = snapshot.native_state.head_state
    if method == "bp":
        for name in NAMES:
            snapshot.native_state.gradients[name] = torch.full_like(head[name], 0.125)
    if method in ("neutral_cpc", "gated_cpc"):
        head["reward_error_ema"], head["last_reward_scale"] = 0.25, 0.75
    head[NAMES[0]].requires_grad_(True)
    head[NAMES[0]].grad = torch.full_like(head[NAMES[0]], -0.25)
    before, saved_grad = deepcopy(snapshot), head[NAMES[0]].grad.clone()
    inputs = pickle.dumps(episodes)
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    captured = []
    original = torch.autograd.grad

    def gradients(*args, **kwargs):
        values = original(*args, **kwargs)
        captured.append(tuple(value.detach().numpy().copy() for value in values))
        return values

    monkeypatch.setattr(torch.autograd, "grad", gradients)
    with torch.no_grad():
        result = diagnostics.diagnose_prior_gradients(snapshot, episodes, row_indices=(2,))
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert result["definition"] == diagnostics.ROW_DEFINITION != diagnostics.DEFINITION
    assert result["row_indices"] == [2] and result["status"] == "complete"
    assert [loss["episode_id"] for loss in result["losses"]] == [e.episode_id for e in episodes]
    assert all(loss["context"] == "A" and loss["rows"] == 1 for loss in result["losses"])
    parameters = {name: head[name].detach().numpy().astype(np.float64) for name in NAMES}
    expected = []
    for episode, actual, loss in zip(episodes, captured, result["losses"], strict=True):
        objective, values = numpy_selected(parameters, episode, (2,))
        assert loss["mean_ce"] == pytest.approx(objective, rel=2e-6, abs=2e-7)
        expected.append(values)
        for position, name in enumerate(NAMES):
            np.testing.assert_allclose(actual[position], values[name], rtol=2e-5, atol=2e-7)
            direction = np.cos(np.arange(parameters[name].size)).reshape(parameters[name].shape)
            direction /= np.linalg.norm(direction)
            epsilon = 1e-5
            plus = dict(parameters, **{name: parameters[name] + epsilon * direction})
            minus = dict(parameters, **{name: parameters[name] - epsilon * direction})
            difference = (
                numpy_selected(plus, episode, (2,))[0] - numpy_selected(minus, episode, (2,))[0]
            ) / (2 * epsilon)
            assert difference == pytest.approx(
                float(np.sum(actual[position] * direction)), rel=5e-4, abs=2e-7
            )
    for names, actual in [
        (NAMES, result["joint_geometry"]),
        *(([name], result["per_parameter_geometry"][name]) for name in NAMES),
    ]:
        geometry = expected_geometry(
            [expected[0][n] for n in names], [expected[1][n] for n in names]
        )
        for key, value in geometry.items():
            assert actual[key] == pytest.approx(value, rel=2e-5, abs=2e-7)
    work = result["work"]
    for kind in ("projection", "forward", "backward"):
        assert work[f"{kind}_calls_attempted"] == work[f"{kind}_calls_completed"] == 2
        assert work[f"{kind}_rows_attempted"] == work[f"{kind}_rows_completed"] == 2
    assert work["projection_array_bytes_returned"] == 2 * 296 * 8
    assert work["admission_projection_rows"] == 36
    assert work["admission_projection_array_equivalent_bytes"] == 36 * 296 * 8
    assert work["optimizer_steps"] == work["published_updates"] == 0
    assert gradient_budget == dict(forwards=2, forward_rows=2, backwards=2)
    assert_tree(torch, snapshot, before)
    assert head[NAMES[0]].requires_grad and torch.equal(head[NAMES[0]].grad, saved_grad)
    assert pickle.dumps(episodes) == inputs
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    json.dumps(result, allow_nan=False)


def test_should_preserve_original_full_episode_wire_and_work_for_explicit_none(
    originals, gradient_budget
):
    snapshot, episodes = originals["bp"], pair()
    original = diagnostics.diagnose_prior_gradients(snapshot, episodes)
    explicit = diagnostics.diagnose_prior_gradients(snapshot, episodes, row_indices=None)
    assert original == explicit and original["definition"] == diagnostics.DEFINITION
    assert "row_indices" not in original
    assert original["work"]["forward_rows_completed"] == 4
    assert gradient_budget == dict(forwards=4, forward_rows=8, backwards=4)


def test_should_reduce_only_the_requested_multiple_rows_without_rewriting_episodes(
    originals, gradient_budget
):
    episodes = turn_pair()
    result = diagnostics.diagnose_prior_gradients(originals["bp"], episodes, row_indices=(0, 2, 8))
    assert result["row_indices"] == [0, 2, 8]
    assert [value["rows"] for value in result["losses"]] == [3, 3]
    assert (
        result["work"]["forward_rows_completed"] == result["work"]["backward_rows_completed"] == 6
    )
    assert all(len(episode.transitions) == 9 for episode in episodes)
    assert gradient_budget == dict(forwards=2, forward_rows=6, backwards=2)


@pytest.mark.parametrize(
    "indices",
    [
        (),
        [],
        [2],
        (True,),
        (False,),
        (np.int64(2),),
        (2.0,),
        ("2",),
        (-1,),
        (9,),
        (2, 2),
        (3, 2),
        2,
        np.array([2]),
    ],
)
def test_should_refuse_bad_indices_before_optional_backend_access(monkeypatch, indices):
    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_prior_gradients(metadata_snapshot(), turn_pair(), row_indices=indices)


def test_should_validate_selection_against_both_complete_episode_lengths(monkeypatch):
    short, long = pair()[0], turn_pair()[1]
    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_prior_gradients(metadata_snapshot(), (long, short), row_indices=(2,))


@pytest.mark.parametrize(
    "fault", ["probe_train", "probe_test", "final_test", "unselected_next", "unselected_rogue"]
)
def test_should_validate_the_whole_original_episode_before_selecting_rows(monkeypatch, fault):
    episodes = turn_pair()
    if fault in ("probe_train", "probe_test", "final_test"):
        episodes = (replace(episodes[0], role=fault), episodes[1])
    elif fault == "unselected_next":
        step = episodes[0].transitions[-1]
        object.__setattr__(step, "next_observation", (float("nan"), *step.next_observation[1:]))
    else:
        object.__setattr__(episodes[0].transitions[-1], "foreign", 1)
    monkeypatch.setattr(diagnostics, "require_torch", forbidden)
    with pytest.raises(ValueError):
        diagnostics.diagnose_prior_gradients(metadata_snapshot(), episodes, row_indices=(2,))


def test_should_refuse_inference_mode_before_native_validation(originals, monkeypatch):
    torch = require_torch()
    monkeypatch.setattr(diagnostics, "copy_invariant_torch_feature_snapshot", forbidden)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        diagnostics.diagnose_prior_gradients(originals["bp"], turn_pair(), row_indices=(2,))


@pytest.mark.parametrize("fault", ["second_backward", "foreign_episode"])
def test_should_keep_selected_paid_prefix_and_refuse_late_changes(
    originals, monkeypatch, gradient_budget, fault
):
    torch = require_torch()
    snapshot, episodes = deepcopy(originals["bp"]), turn_pair()
    before = deepcopy(snapshot)
    owner, name = (
        (torch.autograd, "grad") if fault == "second_backward" else (diagnostics, "_prior_logits")
    )
    original, calls = getattr(owner, name), []

    def callback(*args, **kwargs):
        calls.append(1)
        if fault == "second_backward" and len(calls) == 2:
            raise RuntimeError("injected selected-row second backward failure")
        result = original(*args, **kwargs)
        if fault == "foreign_episode" and len(calls) == 2:
            object.__setattr__(episodes[0], "context", "foreign")
        return result

    monkeypatch.setattr(owner, name, callback)
    with pytest.raises(diagnostics.GradientDiagnosticFailed) as caught:
        diagnostics.diagnose_prior_gradients(snapshot, episodes, row_indices=(2,))
    result, work = caught.value.report, caught.value.report["work"]
    assert result["definition"] == diagnostics.ROW_DEFINITION and result["row_indices"] == [2]
    assert result["status"] == "incomplete" and len(result["losses"]) == 2
    assert work["forward_calls_completed"] == 2
    assert work["backward_calls_completed"] == (1 if fault == "second_backward" else 2)
    assert work["failure_work_may_be_unobserved"] is True
    assert result["joint_geometry"] is result["per_parameter_geometry"] is None
    assert work["optimizer_steps"] == work["published_updates"] == 0
    assert_tree(torch, snapshot, before)
    if fault == "foreign_episode":
        assert episodes[0].context == "foreign"


def scripted_diagnostic(episodes, *, status="complete"):
    result = diagnostics._report()
    result.update(
        status=status,
        definition=diagnostics.ROW_DEFINITION,
        row_indices=[2],
        losses=[
            dict(episode_id=e.episode_id, context=e.context, rows=1, mean_ce=1.0) for e in episodes
        ],
    )
    for kind in ("projection", "forward", "backward"):
        for suffix in ("calls_attempted", "calls_completed", "rows_attempted", "rows_completed"):
            result["work"][f"{kind}_{suffix}"] = 2
    return result


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_bool_public_flags_before_resources(monkeypatch, flag):
    monkeypatch.setattr(fit, "InvariantTorchFeatureLearner", forbidden)
    monkeypatch.setattr(fit, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        fit.describe_protocol(cue_turn_gradients=flag)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), updates=2, cue_turn_gradients=flag)


def test_should_declare_separate_fixed_rows_and_costs_without_changing_fit_recipe(monkeypatch):
    monkeypatch.setattr(fit, "InvariantTorchFeatureLearner", forbidden)
    monkeypatch.setattr(fit, "ProcessRssSampler", forbidden)
    original = fit.describe_protocol()
    assert original == fit.describe_protocol(cue_turn_gradients=False)
    selected = fit.describe_protocol(cue_turn_gradients=True)
    declaration = selected.pop("cue_turn_gradients")
    assert selected == original
    assert declaration["row_indices"] == [2] and declaration["checkpoint_updates"] == [0, 128, 512]
    assert declaration["role"] == "train" and declaration["context"] == "A"
    assert declaration["source_episode_ids"] == ["minigrid-source-A-blue", "minigrid-source-A-red"]
    assert declaration["pair_calls"] == 3
    assert declaration["forward_calls"] == declaration["forward_rows"] == 6
    assert declaration["backward_calls"] == declaration["backward_rows"] == 6
    assert declaration["optimizer_steps"] == declaration["published_updates"] == 0


def test_should_run_fixed_gradient_reads_only_after_all_training_and_raw_scores(
    monkeypatch, fake_runtime
):
    captured, _ = fake_runtime
    episodes = histories()
    inputs = pickle.dumps(episodes)
    returned, calls, events = [], [], []
    original_score = fit._score_point

    def score(*args):
        result = original_score(*args)
        events.append(("score", args[2]))
        return result

    def diagnostic(snapshot, sources, *, row_indices):
        assert captured["stream"]._finished is True
        assert captured["stream"].completed_updates == 2
        assert events[:2] == [("score", 0), ("score", 2)]
        assert len(Learner.work["predicts"]) == 12
        assert row_indices == (2,) and len(sources) == 2
        assert [source.episode_id for source in sources] == [e.episode_id for e in episodes[:2]]
        assert all(source.role == "train" and len(source.transitions) == 9 for source in sources)
        calls.append(snapshot.completed_updates)
        events.append(("gradient", snapshot.completed_updates))
        result = scripted_diagnostic(sources)
        returned.append(result)
        return result

    monkeypatch.setattr(fit, "_score_point", score)
    monkeypatch.setattr(fit, "diagnose_prior_gradients", diagnostic)
    result = fit.diagnose_train_fit(
        episodes, updates=2, max_wall_seconds=10, cue_turn_gradients=True
    )
    assert result["status"] == "complete", result["reason"]
    assert calls == [0, 2] and events == [
        ("score", 0),
        ("score", 2),
        ("gradient", 0),
        ("gradient", 2),
    ]
    assert [record["updates"] for record in result["cue_turn_gradients"]] == [0, 2]
    assert all(
        record["elapsed_seconds"] == 0 and record["diagnostic"]["status"] == "complete"
        for record in result["cue_turn_gradients"]
    )
    assert result["charged_updates"] == result["committed_updates"] == 2
    assert len(result["applied_updates"]) == 2 and len(result["readouts"]) == 2
    assert result["work"]["score_forward_calls_completed"] == 12
    returned[0]["work"]["backward_rows_completed"] = 999
    returned[0]["losses"].clear()
    assert result["cue_turn_gradients"][0]["diagnostic"]["work"]["backward_rows_completed"] == 2
    assert len(result["cue_turn_gradients"][0]["diagnostic"]["losses"]) == 2
    assert pickle.dumps(episodes) == inputs
    assert pickle.dumps(captured["source"].snapshot_state()) == pickle.dumps(captured["initial"])


def test_should_leave_default_runtime_wire_and_work_unchanged(monkeypatch, fake_runtime):
    monkeypatch.setattr(fit, "diagnose_prior_gradients", forbidden)
    result = fit.diagnose_train_fit(
        histories(), updates=2, max_wall_seconds=10, cue_turn_gradients=False
    )
    assert result["status"] == "complete", result["reason"]
    assert "cue_turn_gradients" not in result and "cue_turn_gradients" not in result["protocol"]
    assert result["charged_updates"] == 2 and result["work"]["score_forward_calls_completed"] == 12


@pytest.mark.parametrize(
    "fault",
    [
        "diagnostic_failure",
        "source_episode",
        "source_model",
        "candidate",
        "checkpoint",
        "wall",
        "row_report",
    ],
)
def test_should_refuse_diagnostic_or_closing_corruption_and_preserve_paid_fit(
    monkeypatch,
    fake_runtime,
    fault,
):
    captured, clock = fake_runtime
    episodes, calls, returned = histories(), [], []

    def diagnostic(snapshot, sources, *, row_indices):
        calls.append(snapshot.completed_updates)
        result = scripted_diagnostic(sources)
        returned.append(result)
        if fault == "diagnostic_failure":
            if len(calls) == 2:
                result.update(status="incomplete", reason="injected paid selected-row failure")
                result["work"]["backward_calls_completed"] = 1
                result["work"]["backward_rows_completed"] = 1
                result["work"]["failure_work_may_be_unobserved"] = True
                raise diagnostics.GradientDiagnosticFailed(result)
        elif fault == "source_episode":
            object.__setattr__(episodes[0], "context", "foreign")
        elif fault == "source_model":
            captured["source"].state.parameters["weight"][0, 0] = 99
        elif fault == "candidate":
            captured["stream"]._runtime._candidate.state.parameters["weight"][0, 0] = 88
        elif fault == "checkpoint":
            snapshot.parameters["weight"][0, 0] = 77
        elif fault == "row_report":
            result["definition"] = diagnostics.DEFINITION
            result["row_indices"] = [0, 1, 2]
        else:
            clock[0] = 1000.0
        return result

    monkeypatch.setattr(fit, "diagnose_prior_gradients", diagnostic)
    result = fit.diagnose_train_fit(
        episodes, updates=2, max_wall_seconds=10, cue_turn_gradients=True
    )
    assert result["status"] == "incomplete" and result["reason"]
    assert result["charged_updates"] == result["committed_updates"] == 2
    assert result["native_training_rows"] == 108 and len(result["applied_updates"]) == 2
    assert result["stream_result"]["evaluation_scope"] == "training_only"
    assert all(
        record["status"] == "complete" and len(record["scores"]) == 6
        for record in result["readouts"]
    )
    assert result["work"]["score_forward_calls_completed"] == 12
    assert result["fit_witness"] is False and "outcomes" not in result
    assert captured["options"]["budget"].started_at == 1.0
    if fault == "diagnostic_failure":
        assert calls == [0, 2]
        failed = result["cue_turn_gradients"][-1]["diagnostic"]
        assert failed["work"]["backward_calls_completed"] == 1
        assert failed["work"]["failure_work_may_be_unobserved"] is True
        returned[-1]["work"]["backward_calls_completed"] = 999
        assert failed["work"]["backward_calls_completed"] == 1
    else:
        assert calls == [0] and len(result["cue_turn_gradients"]) == 1
    if fault == "source_episode":
        assert episodes[0].context == "foreign"
    if fault == "candidate":
        assert captured["stream"]._runtime._candidate.state.parameters["weight"][0, 0] == 88


def test_should_describe_selected_cli_without_source_or_model_access(monkeypatch, capsys):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["fit", "--constructive-bp", "--cue-turn-gradients", "--describe"]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == fit.describe_protocol(cue_turn_gradients=True)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--cue-turn-gradients", "--describe"],
        ["--bounds-only", "--cue-turn-gradients", "--describe"],
    ],
)
def test_should_refuse_wrong_cli_route_before_data_access(monkeypatch, arguments):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_forward_selected_cli_option_only_to_constructive_fit(tmp_path, monkeypatch, capsys):
    source = tmp_path / "sources.json"
    source.write_text(json.dumps(dict(training_sources=[asdict(e) for e in raw_sources()])))
    calls = []

    def diagnose(episodes, **options):
        calls.append((episodes, options))
        return dict(status="complete", proof="scripted selected-row dispatch")

    monkeypatch.setattr(cli, "diagnose_train_fit", diagnose)
    monkeypatch.setattr(cli, "diagnose_train_linear_fit", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--constructive-bp",
            "--cue-turn-gradients",
            "--fit-updates",
            "2",
            "--fit-seconds",
            "10",
            "--source",
            str(source),
        ],
    )
    cli.main()
    assert len(calls) == 1 and calls[0][1] == dict(
        updates=2, max_wall_seconds=10, cue_turn_gradients=True
    )
    assert all(episode.role == "train" for episode in calls[0][0])
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
