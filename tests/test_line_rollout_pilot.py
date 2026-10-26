"""Prediction import, honest recorded/live origins, and existing CLI isolation."""

from copy import deepcopy
import io
import json
from math import log
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.frozen_torch_predictor as frozen
import src.adapters.line_rollout_pilot as pilot
import src.app.toy_execution_budget as budgets
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("undeclared native or unselected work")


def parameters():
    return {
        name: [[0.0] * columns for _ in range(rows)]
        for name, (rows, columns) in frozen.SHAPES.items()
    }


def source():
    runs = []
    for enabled in (False, True):
        for method in pilot.METHODS:
            runs.append(
                dict(
                    method=method,
                    replay_enabled=enabled,
                    status="complete",
                    observations=[
                        dict(
                            wake_updates=n,
                            native_updates=j * (3 if enabled else 2),
                            native_training_examples=j * (6 if enabled else 4),
                            parameters=parameters(),
                        )
                        for j, n in enumerate(pilot.POINTS)
                    ],
                    stream_result=dict(
                        checkpoints=[
                            dict(
                                updates=n,
                                training_context=(None, "A", "B", "A")[j],
                                scores={
                                    context: dict(bce=log(2), accuracy=float(context == "A"))
                                    for context in ("A", "B")
                                },
                            )
                            for j, n in enumerate(pilot.POINTS)
                        ]
                    ),
                )
            )
    return dict(
        status="complete",
        protocol=dict(
            protocol_id=pilot.SOURCE_PROTOCOL,
            seed=17,
            feature_dim=2,
            hidden_dim=3,
            phase_contexts=["A", "B", "A"],
            phase_lengths=[2, 2, 2],
            checkpoint_updates=[0, 2, 4, 6],
        ),
        runs=runs,
    )


@pytest.mark.parametrize("fault", ["method", "seed", "keys", "shape", "bool", "nan", "float32"])
def test_should_refuse_invalid_frozen_parameters_before_native_construction(monkeypatch, fault):
    monkeypatch.setattr(frozen, "TorchFeatureLearner", forbidden)
    method, seed, values = "pc", 17, parameters()
    if fault == "method":
        method = "other"
    elif fault == "seed":
        seed = True
    elif fault == "keys":
        values.pop("bias_output")
    elif fault == "shape":
        values["bias_output"] = [[0.0]]
    else:
        values["bias_output"][0][0] = {"bool": True, "nan": float("nan"), "float32": 0.1}[fault]
    with pytest.raises(ValueError):
        frozen.FrozenTorchPredictor(method, values, seed=seed)


class NativeStub:
    def __init__(self, policy):
        self.state = SimpleNamespace(
            head_state={name: np.zeros(shape, np.float32) for name, shape in frozen.SHAPES.items()},
            clock=0,
        )
        self.fail = False
        self.mutate = False

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, *args):
        forbidden()

    def predict(self, features):
        if self.fail:
            raise RuntimeError("native prediction fault")
        if self.mutate:
            self.state.clock += 1
        return np.full((len(features), 1), 0.5, np.float64)


def native_stub(monkeypatch):
    monkeypatch.setattr(frozen, "TorchFeatureLearner", NativeStub)
    monkeypatch.setattr(
        frozen,
        "require_torch",
        lambda: SimpleNamespace(tensor=lambda data, dtype: np.array(data, dtype=dtype)),
    )
    monkeypatch.setattr(frozen, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))


def test_should_import_owned_weights_without_claiming_historical_native_state(monkeypatch):
    native_stub(monkeypatch)
    values = parameters()
    values["bias_output"][0][0] = 1.0
    predictor = frozen.FrozenTorchPredictor("pc", values)
    values["bias_output"][0][0] = 99.0
    native = cast(NativeStub, predictor._learner)
    assert native.state.head_state["bias_output"][0, 0] == 1.0
    assert native.state.clock == 0
    assert predictor.predict(np.zeros((1, 2), np.float64))[0, 0] == 0.5
    predictor.assert_unchanged()


@pytest.mark.parametrize("fault", ["before", "during", "prediction"])
def test_should_stop_prediction_after_mutation_or_fault_without_reset(monkeypatch, fault):
    native_stub(monkeypatch)
    predictor = frozen.FrozenTorchPredictor("pc", parameters())
    native = cast(NativeStub, predictor._learner)
    if fault == "before":
        native.state.clock = 1
    if fault == "during":
        native.mutate = True
    if fault == "prediction":
        native.fail = True
    with pytest.raises((ValueError, RuntimeError)):
        predictor.predict(np.zeros((1, 2), np.float64))
    with pytest.raises(ValueError, match="stopped"):
        predictor.predict(np.zeros((1, 2), np.float64))


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "order",
        "clock",
        "parameter",
        "scores",
        "phase_contexts",
        "phase_lengths",
        "checkpoint_updates",
        "training_context",
    ],
)
def test_should_reject_corrupted_or_replaced_source_metadata_before_native_work(monkeypatch, fault):
    record = source()
    if fault == "incomplete":
        record["status"] = "incomplete"
    elif fault == "order":
        record["runs"].reverse()
    elif fault == "clock":
        record["runs"][0]["observations"][1]["native_updates"] = 7
    elif fault == "parameter":
        record["runs"][0]["observations"][0]["parameters"] = {}
    elif fault == "scores":
        record["runs"][0]["stream_result"]["checkpoints"][0]["scores"] = {}
    elif fault == "training_context":
        record["runs"][0]["stream_result"]["checkpoints"][1]["training_context"] = "B"
    elif fault == "phase_contexts":
        record["protocol"][fault] = ["A", "A", "B"]
    elif fault == "phase_lengths":
        record["protocol"][fault] = [3, 1, 2]
    else:
        record["protocol"][fault].reverse()
    monkeypatch.setattr(pilot, "FrozenTorchPredictor", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.validate_source(record)


def test_should_bound_source_read_before_json_or_model_construction(monkeypatch):
    monkeypatch.setattr(Path, "open", lambda *a, **k: io.BytesIO(b" " * (pilot.MAX_BYTES + 1)))
    monkeypatch.setattr(pilot, "FrozenTorchPredictor", forbidden)
    with pytest.raises(ValueError, match="allowance"):
        pilot.read_source("oversized.json")


class Sampler:
    start_bytes = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 1

    def snapshot(self):
        return ProcessRssSegment(1, 1, 1, 1, 0.005)


class TorchStub:
    @staticmethod
    def set_num_threads(value):
        assert value == 1

    @staticmethod
    def get_num_interop_threads():
        return 1


def setup(monkeypatch, tmp_path):
    location = tmp_path / "source.json"
    location.write_text(json.dumps(source()), encoding="utf8")
    events = []

    class Predictor:
        def __init__(self, method, parameters, *, seed):
            events.append(("construct", method))

        def predict(self, features):
            events.append(("forward", len(features)))
            return np.full((len(features), 1), 0.5, np.float64)

    monkeypatch.setattr(pilot, "FrozenTorchPredictor", Predictor)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    return location, events, Predictor


def test_should_qualify_all_imports_before_live_episodes_and_keep_expected_negatives(
    monkeypatch, tmp_path
):
    location, events, _ = setup(monkeypatch, tmp_path)
    original = location.read_bytes()
    report = pilot.run_pilot(location)
    assert report["status"] == "complete", report["global_errors"]
    assert report["training_updates"] == 0 and report["historical_training_state_restored"] is False
    assert report["work"] == dict(
        predictors_created=32, forwards_attempted=192, forwards_returned=192
    )
    forwards = [rows for kind, rows in events if kind == "forward"]
    assert forwards[:64] == [2] * 64 and forwards[64:] == [1] * 128
    assert sum(episode["success"] for episode in report["episodes"]) == 32
    assert sum(episode["outcome"] == "illegal_action" for episode in report["episodes"]) == 32
    assert len(report["episodes"]) == 64 and location.read_bytes() == original
    assert all(
        episode["learning_active"] is False and episode["execution_origin"] == "live_environment"
        for episode in report["episodes"]
    )


def test_should_keep_all_live_slots_unreached_on_import_score_mismatch(monkeypatch, tmp_path):
    location, _, _ = setup(monkeypatch, tmp_path)
    record = json.loads(location.read_text())
    record["runs"][0]["stream_result"]["checkpoints"][0]["scores"]["A"]["bce"] += 0.01
    location.write_text(json.dumps(record), encoding="utf8")
    report = pilot.run_pilot(location)
    assert report["status"] == "incomplete" and report["checkpoints"][0]["status"] == "failed"
    assert report["work"]["predictors_created"] == 1
    assert all(episode["status"] == "not_run" for episode in report["episodes"])


def test_should_preserve_partial_live_fault_and_stop_all_future_episodes(monkeypatch, tmp_path):
    location, events, Predictor = setup(monkeypatch, tmp_path)
    original = Predictor.predict

    def predict(self, features):
        if len(features) == 1:
            raise RuntimeError("declared live forward failure")
        return original(self, features)

    monkeypatch.setattr(Predictor, "predict", predict)
    report = pilot.run_pilot(location)
    assert report["status"] == "incomplete" and len(events) == 96
    assert report["episodes"][0]["status"] == "failed"
    assert report["episodes"][0]["prediction_calls_attempted"] == 1
    assert all(episode["status"] == "not_run" for episode in report["episodes"][1:])
    assert report["work"]["forwards_attempted"] == 65 and report["work"]["forwards_returned"] == 64


@pytest.mark.parametrize("describe", [False, True])
def test_should_route_only_selected_live_evaluation_callback(monkeypatch, capsys, describe):
    calls = []

    def describe_selected():
        calls.append("describe")
        return dict(status="complete")

    def run_selected(path):
        calls.append(path)
        return dict(status="complete")

    monkeypatch.setattr(
        cli,
        "describe_rollout_protocol",
        describe_selected,
    )
    monkeypatch.setattr(cli, "run_rollout_pilot", run_selected)
    monkeypatch.setattr(cli, "run_trajectory_pilot", forbidden)
    args = ["pilot", "--scenario", "torch-trajectory-rollouts"]
    args += ["--describe"] if describe else ["--source-artifact", "source.json"]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert calls == (["describe"] if describe else ["source.json"])
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "args",
    [
        ["--scenario", "torch-trajectory-rollouts"],
        ["--scenario", "original", "--source-artifact", "source.json"],
        [
            "--scenario",
            "torch-trajectory-rollouts",
            "--describe",
            "--source-artifact",
            "source.json",
        ],
    ],
)
def test_should_refuse_missing_or_unselected_source_arguments_before_execution(monkeypatch, args):
    monkeypatch.setattr(cli, "run_rollout_pilot", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
