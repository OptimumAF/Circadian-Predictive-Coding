"""Disposable TRAIN-fit admission, ordinary stream isolation and paid failures.

One case performs two native BP updates and twelve nine-row predictions. Other
runtime cases use the established owned fake learner; no game or teacher runs.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import random
import sys
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_constructive_fit as fit
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.app import toy_execution_budget as budgets
from src.core.categorical_probability_scores import probability_scores
from src.core.minigrid_public_history import encode_public_history_episode
from src.shared.torch_runtime import require_torch
from test_minigrid_learning_pilot import Learner
from test_minigrid_train_fit_bounds import histories, raw_sources
from test_torch_feature_learner import assert_tree
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


@pytest.fixture(autouse=True)
def observed_native_work(monkeypatch, record_property):
    work = dict(updates=0, training_rows=0, predictions=0, prediction_rows=0)
    train, predict = TorchFeatureLearner.train_batch, TorchFeatureLearner.predict

    def counted_train(owner, features, targets):
        work["updates"] += 1
        work["training_rows"] += len(features)
        assert work["updates"] <= 2 and work["training_rows"] <= 108
        return train(owner, features, targets)

    def counted_predict(owner, features):
        work["predictions"] += 1
        work["prediction_rows"] += len(features)
        assert work["predictions"] <= 12 and work["prediction_rows"] <= 108
        return predict(owner, features)

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", counted_train)
    monkeypatch.setattr(TorchFeatureLearner, "predict", counted_predict)
    yield work
    for name, value in work.items():
        record_property(f"constructive_native_{name}", value)


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("InvariantTorchFeatureLearner", "AttributionStream", "ProcessRssSampler"):
        monkeypatch.setattr(fit, name, forbidden)


@pytest.mark.parametrize("updates", [True, False, 0, -1, 513, 2.0, np.int64(2), "2"])
def test_should_refuse_invalid_update_limits_before_any_resource(no_resources, updates):
    with pytest.raises(ValueError):
        fit.describe_protocol(updates=updates)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), updates=updates)


@pytest.mark.parametrize(
    "seconds", [True, False, 0, -1, 90.001, float("nan"), float("inf"), np.float64(1), "1"]
)
def test_should_refuse_invalid_wall_limits_before_any_resource(no_resources, seconds):
    with pytest.raises(ValueError):
        fit.describe_protocol(max_wall_seconds=seconds)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), max_wall_seconds=seconds)


def test_should_describe_fixed_default_recipe_without_constructing_resources(no_resources):
    first = fit.describe_protocol()
    assert first == fit.describe_protocol(updates=512, max_wall_seconds=90)
    encoded = json.dumps(first, sort_keys=True, allow_nan=False)
    assert "512" in encoded and "train" in encoded
    assert first["method"] == "public_invariant_bp" and first["hidden_dim"] == 8
    assert first["seed"] == 17 and first["learning_rate"] == 0.03 and first["momentum"] == 0.9
    assert first["checkpoint_updates"] == [0, 128, 512]
    assert first["training_rows"] == 512 * 54
    assert first["stream_input_array_bytes"] == 512 * 54 * 297 * 8
    assert first["training_only"] is True and first["heldout"] == []
    assert first["label_delay"] == 1
    assert first["score_forward_calls"] == 18 and first["score_rows"] == 162
    assert first["probe_updates"] == first["final_episodes"] == first["replay_rows"] == 0
    first["foreign"] = True
    assert "foreign" not in fit.describe_protocol()


@pytest.mark.parametrize(
    "updates,points",
    [(1, [0, 1]), (2, [0, 2]), (128, [0, 128]), (129, [0, 128, 129]), (512, [0, 128, 512])],
)
def test_should_freeze_unique_ordered_points_before_runtime(no_resources, updates, points):
    protocol = fit.describe_protocol(updates=updates, max_wall_seconds=1)
    assert protocol["checkpoint_updates"] == points
    assert protocol["score_forward_calls"] == 6 * len(points)


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "missing",
        "order",
        "identity",
        "context",
        "probe_train",
        "probe_test",
        "final_test",
        "raw148",
        "next",
        "rogue",
    ],
)
def test_should_admit_only_original_complete_train_sources_before_models(no_resources, fault):
    sources: Any = histories()
    if fault == "list":
        sources = list(sources)
    elif fault == "missing":
        sources = sources[:-1]
    elif fault == "order":
        sources = (sources[1], sources[0], *sources[2:])
    elif fault == "raw148":
        sources = raw_sources()
    elif fault == "identity":
        sources = (replace(sources[0], episode_id="foreign"), *sources[1:])
    elif fault == "context":
        sources = (replace(sources[0], context="foreign"), *sources[1:])
    elif fault in ("probe_train", "probe_test", "final_test"):
        sources = (replace(sources[0], role=fault), *sources[1:])
    elif fault == "rogue":
        object.__setattr__(sources[0], "unobserved", True)
    else:
        last = sources[0].transitions[-1]
        object.__setattr__(last, "next_observation", (float("nan"), *last.next_observation[1:]))
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(sources, updates=2, max_wall_seconds=10)


def test_should_describe_constructive_cli_without_reading_data_or_running_models(
    monkeypatch, capsys
):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_linear_fit", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit", "--constructive-bp", "--describe", "--fit-updates", "2", "--fit-seconds", "10"],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == fit.describe_protocol(
        updates=2, max_wall_seconds=10
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--bounds-only", "--constructive-bp", "--describe"],
        ["--describe", "--fit-updates", "2"],
        ["--bounds-only", "--describe", "--fit-seconds", "10"],
        ["--constructive-bp", "--describe", "--source", "unused"],
    ],
)
def test_should_refuse_incompatible_cli_options_before_source_access(monkeypatch, options):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbidden)
    monkeypatch.setattr(sys, "argv", ["fit", *options])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_dispatch_saved_sources_to_only_the_selected_constructive_route(
    tmp_path, monkeypatch, capsys
):
    source, output = tmp_path / "sources.json", tmp_path / "result.json"
    source.write_text(
        json.dumps(
            dict(
                training_sources=[asdict(e) for e in raw_sources()],
                final_test="unavailable",
                probe_data="unavailable",
            )
        )
    )
    before = source.read_bytes()
    calls = []

    def constructive(episodes, **options):
        calls.append((episodes, options))
        return dict(status="complete", proof="scripted route only")

    monkeypatch.setattr(cli, "diagnose_train_linear_fit", forbidden)
    monkeypatch.setattr(cli, "summarize_train_fit_bounds", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", constructive)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--constructive-bp",
            "--fit-updates",
            "2",
            "--fit-seconds",
            "10",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
    )
    cli.main()
    assert len(calls) == 1 and calls[0][1] == dict(updates=2, max_wall_seconds=10)
    expected = tuple(encode_public_history_episode(e) for e in raw_sources())
    assert calls[0][0] == expected
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())
    assert source.read_bytes() == before
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_save_incomplete_cli_results_with_their_original_paid_work(
    tmp_path, monkeypatch, capsys
):
    source, output = tmp_path / "sources.json", tmp_path / "incomplete.json"
    source.write_text(json.dumps(dict(training_sources=[asdict(e) for e in raw_sources()])))
    paid = dict(
        status="incomplete",
        reason="injected paid failure",
        charged_updates=1,
        applied_updates=[],
        work=dict(training_work_may_be_unobserved=True),
    )
    monkeypatch.setattr(cli, "diagnose_train_fit", lambda *a, **k: deepcopy(paid))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--constructive-bp",
            "--fit-updates",
            "2",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    result = json.loads(output.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["charged_updates"] == 1 and result["applied_updates"] == []
    assert result["work"] == paid["work"] and result["reason"] == paid["reason"]


class FitSampler(Sampler):
    peak_bytes = 100


def capture_stream(monkeypatch):
    original = fit.AttributionStream
    captured: dict[str, Any] = {}

    def create(source, training, heldout, **options):
        captured.update(
            source=source,
            initial=deepcopy(source.snapshot_state()),
            training=training,
            heldout=heldout,
            options=options,
        )
        stream = original(source, training, heldout, **options)
        captured["stream"] = stream
        return stream

    monkeypatch.setattr(fit, "AttributionStream", create)
    monkeypatch.setattr(fit, "ProcessRssSampler", FitSampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", FitSampler)
    return captured


def assert_raw_scores(result):
    """Recompute the original metrics from every stored probability and target."""
    for record, outcome in zip(result["readouts"], result["outcomes"], strict=True):
        assert record["status"] == "complete" and len(record["scores"]) == 6
        assert record["updates"] == outcome["updates"]
        for score in record["scores"]:
            predictions = np.array(score["probabilities"], dtype=np.float64)
            targets = np.array(score["target_actions"], dtype=np.float64).reshape(-1, 1)
            expected = probability_scores(predictions, targets, 7)
            assert predictions.shape == (9, 7)
            assert score["ce"] == expected["ce"] and score["accuracy"] == expected["accuracy"]
            assert score["predicted_actions"] == np.argmax(predictions, axis=1).tolist()
            assert score["exact_sequence_accuracy"] == float(expected["accuracy"] == 1)
        for context in ("A", "B", "C_train"):
            scores = [value for value in record["scores"] if value["context"] == context]
            assert len(scores) == 2
            for metric in ("ce", "accuracy", "exact_sequence_accuracy"):
                assert outcome[context][metric] == sum(value[metric] for value in scores) / 2
    assert result["fit_witness"] is all(
        score["exact_sequence_accuracy"] == 1 for score in result["readouts"][-1]["scores"]
    )


def test_should_execute_two_native_updates_then_score_isolated_fixed_points(
    monkeypatch,
    observed_native_work,
):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    captured = capture_stream(monkeypatch)
    sources = histories()
    inputs = pickle.dumps(sources)
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        pickle.dumps(np.random.get_state()),
        torch.get_rng_state().clone(),
    )
    original_score = fit._score_point
    scoring_states = []

    def score(source, state, point, episodes, report, guard):
        stream = captured["stream"]
        assert stream.completed_updates == 2 and stream._finished is True
        assert report["stream_result"]["evaluation_scope"] == "training_only"
        scoring_states.append(deepcopy(state))
        return original_score(source, state, point, episodes, report, guard)

    monkeypatch.setattr(fit, "_score_point", score)
    result = fit.diagnose_train_fit(sources, updates=2, max_wall_seconds=10)
    assert result["status"] == "complete", result["reason"]
    assert observed_native_work == dict(
        updates=2, training_rows=108, predictions=12, prediction_rows=108
    )
    assert (
        result["charged_updates"]
        == result["committed_updates"]
        == result["native_completed_updates"]
        == 2
    )
    assert result["native_training_rows"] == 108
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert [point["updates"] for point in result["readouts"]] == [0, 2]
    assert (
        result["work"]["score_forward_calls_attempted"]
        == result["work"]["score_forward_calls_completed"]
        == 12
    )
    assert result["work"]["score_rows_completed"] == 108
    assert result["work"]["score_work_may_be_unobserved"] is False
    assert result["work"]["training_work_may_be_unobserved"] is False
    training = captured["training"]
    assert len(training) == 2 and captured["heldout"] == ()
    assert captured["options"]["training_only"] is True
    assert captured["options"]["label_delay"] == 1
    assert captured["options"]["checkpoint_updates"] == (0, 2)
    budget = captured["options"]["budget"]
    assert budget.budget.max_training_updates == 2 and budget.updates_completed == 2
    assert 0 < budget.budget.max_wall_seconds <= 10
    assert result["budget"]["limits"] == asdict(budget.budget)
    assert result["budget"]["started_at"] == budget.started_at
    assert training[0].features is training[1].features
    assert set(training[0].row_ids).isdisjoint(training[1].row_ids)
    stream = captured["stream"]
    assert not np.shares_memory(stream._training[0].features, training[0].features)
    assert not np.shares_memory(stream._training[0].features, stream._training[1].features)
    assert all(packet.role == "train" for packet in stream._training)
    receipts = result["applied_updates"]
    assert len(receipts) == 2
    assert [r["update_number"] for r in receipts] == [1, 2]
    assert all(
        r["arrived_at"] == r["observed_at"] + 1 and r["applied_at"] >= r["arrived_at"]
        for r in receipts
    )
    initial = captured["initial"]
    assert_tree(torch, captured["source"].snapshot_state(), initial)
    assert_tree(torch, stream._runtime.actor.snapshot_state().state, initial)
    assert [state.completed_updates for state in scoring_states] == [0, 2]
    assert_tree(torch, scoring_states[0], initial)
    assert_raw_scores(result)
    assert pickle.dumps(sources) == inputs
    assert random.getstate() == python_rng and pickle.dumps(np.random.get_state()) == numpy_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    result["readouts"][0]["scores"][0]["probabilities"][0][0] = 999
    assert_tree(torch, captured["source"].snapshot_state(), initial)
    assert_tree(torch, stream._runtime.actor.snapshot_state().state, initial)


@pytest.fixture
def fake_runtime(monkeypatch):
    captured = capture_stream(monkeypatch)
    monkeypatch.setattr(Learner, "work", {})
    monkeypatch.setattr(fit, "InvariantTorchFeatureLearner", lambda policy: Learner(policy))
    monkeypatch.setattr(fit, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    clock = [1.0]
    monkeypatch.setattr(fit, "monotonic", lambda: clock[0])
    return captured, clock


@pytest.mark.parametrize(
    "fault", ["train_error", "source_input", "wall", "source_model", "actor_model"]
)
def test_should_retain_permanent_charges_when_training_or_a_closing_guard_fails(
    monkeypatch,
    fake_runtime,
    observed_native_work,
    fault,
):
    captured, clock = fake_runtime
    sources = histories()

    def after_train(owner):
        if fault == "train_error":
            raise RuntimeError("injected post-native return failure")
        if fault == "source_input":
            object.__setattr__(sources[0], "context", "foreign")
        elif fault == "wall":
            clock[0] = 1000.0
        elif fault == "source_model":
            captured["source"].state.parameters["weight"][0, 0] = 99
        else:
            captured["stream"]._runtime.actor._learner.state.parameters["weight"][0, 0] = 99

    Learner.work["on_train"] = after_train
    result = fit.diagnose_train_fit(sources, updates=2, max_wall_seconds=10)
    assert result["status"] == "incomplete" and result["reason"]
    # The stream charges completed wake returns; the failed fake call still
    # exposes native work. Preserve both facts instead of inventing a receipt.
    assert result["charged_updates"] == (0 if fault == "train_error" else 1)
    assert result["native_completed_updates"] == 1
    assert result["native_training_rows"] == 54
    assert result["readouts"] == [] and result["fit_witness"] is False
    assert result["work"]["training_work_may_be_unobserved"] is True
    assert captured["options"]["budget"].started_at == 1.0
    assert len(Learner.work["trains"]) == 1
    assert observed_native_work["updates"] == observed_native_work["predictions"] == 0
    if fault == "train_error":
        assert result["applied_updates"] == []
        assert captured["stream"]._runtime._inbox.stopped is True
    else:
        assert len(result["applied_updates"]) == 1
    if fault == "source_input":
        assert sources[0].context == "foreign"
    if fault == "source_model":
        assert captured["source"].state.parameters["weight"][0, 0] == 99
    if fault == "actor_model":
        assert captured["stream"]._runtime.actor._learner.state.parameters["weight"][0, 0] == 99


def test_should_refuse_a_refunded_budget_before_any_checkpoint_readout(monkeypatch, fake_runtime):
    captured, _ = fake_runtime
    score = fit._score_point

    def refund(*args):
        captured["options"]["budget"].updates_completed = 0
        return score(*args)

    monkeypatch.setattr(fit, "_score_point", refund)
    result = fit.diagnose_train_fit(histories(), updates=2, max_wall_seconds=10)
    assert result["status"] == "incomplete" and result["reason"]
    assert result["committed_updates"] == result["native_completed_updates"] == 2
    assert len(result["applied_updates"]) == 2
    assert result["readouts"] == [] and result["fit_witness"] is False
    assert "outcomes" not in result


@pytest.mark.parametrize(
    "fault", ["exception", "input_mutation", "reader_mutation", "invalid_probabilities"]
)
def test_should_preserve_paid_training_and_returned_raw_scores_on_readout_failure(
    monkeypatch,
    fake_runtime,
    observed_native_work,
    fault,
):
    captured, _ = fake_runtime
    sources = histories()
    before = pickle.dumps(sources)
    original = Learner.predict
    calls = []

    def predict(owner, features):
        calls.append(1)
        result = original(owner, features)
        if len(calls) == 2:
            if fault == "exception":
                raise RuntimeError("injected second TRAIN prediction failure")
            if fault == "input_mutation":
                features[0, 0] = 0.123
            elif fault == "reader_mutation":
                owner.state.parameters["weight"][0, 0] = 77
            else:
                result[0, 0] = float("nan")
        return result

    monkeypatch.setattr(Learner, "predict", predict)
    result = fit.diagnose_train_fit(sources, updates=2, max_wall_seconds=10)
    assert result["status"] == "incomplete" and result["reason"]
    assert result["charged_updates"] == result["committed_updates"] == 2
    assert result["native_completed_updates"] == 2 and result["native_training_rows"] == 108
    assert (
        len(result["applied_updates"]) == 2
        and result["stream_result"]["evaluation_scope"] == "training_only"
    )
    assert len(result["readouts"]) == 1 and result["readouts"][0]["status"] != "complete"
    assert len(result["readouts"][0]["scores"]) == (1 if fault == "exception" else 2)
    assert "ce" in result["readouts"][0]["scores"][0]
    assert result["work"]["score_forward_calls_attempted"] == 2
    assert result["work"]["score_forward_calls_completed"] == (1 if fault == "exception" else 2)
    assert result["work"]["score_work_may_be_unobserved"] is (fault == "exception")
    assert result["work"]["training_work_may_be_unobserved"] is False
    assert len(calls) == 2 and pickle.dumps(sources) == before
    assert pickle.dumps(captured["source"].snapshot_state()) == pickle.dumps(captured["initial"])
    assert observed_native_work["updates"] == observed_native_work["predictions"] == 0


def test_should_refuse_identity_reader_before_restoring_a_checkpoint_into_the_source(
    monkeypatch,
    fake_runtime,
    observed_native_work,
):
    captured, _ = fake_runtime
    original = fit._score_point

    def score(*args):
        Learner.work["identity_fork"] = True
        return original(*args)

    monkeypatch.setattr(fit, "_score_point", score)
    result = fit.diagnose_train_fit(histories(), updates=2, max_wall_seconds=10)
    assert result["status"] == "incomplete" and result["reason"]
    assert result["charged_updates"] == result["committed_updates"] == 2
    assert len(result["applied_updates"]) == 2 and result["readouts"] == []
    assert result["work"]["score_forward_calls_attempted"] == 0
    assert pickle.dumps(captured["source"].snapshot_state()) == pickle.dumps(captured["initial"])
    assert observed_native_work["updates"] == observed_native_work["predictions"] == 0
