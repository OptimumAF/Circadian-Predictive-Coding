"""Old TRAIN fit is measured on the disposable probe, never trained again.

These deterministic native-port fakes test routing and accounting, not learning
or scientific benefit. No environment or neural implementation is involved.
"""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch
import src.app.fresh_task_probe as probe_module
from src.app.fresh_task_probe import run_fresh_task_probe
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.learner_ports import TrainingDiagnostic
from test_fresh_task_probe import session


class ReadoutLearner:
    def __init__(self, *, trace=None, hooks=None):
        self.trace = [] if trace is None else trace
        self.hooks = {} if hooks is None else hooks
        self.value = np.array([99.0])
        self.child: Any = None

    def fork(self):
        self.trace.append(("fork",))
        if "fork" in self.hooks:
            self.hooks["fork"]()
        self.child = ReadoutLearner(trace=self.trace, hooks=self.hooks)
        return self.child

    def restore_state(self, state):
        self.value = state.copy()
        self.trace.append(("restore", state.copy()))

    def snapshot_state(self):
        return self.value.copy()

    def train_batch(self, features, targets):
        self.trace.append(("train", int(self.value[0]), features.copy(), targets.copy()))
        self.value += 1
        return TrainingDiagnostic("engineering_readout_fake", float(self.value[0]))

    def predict(self, features):
        old_task = bool(features[0, 0] < 0)
        self.trace.append(("A" if old_task else "F", int(self.value[0]), features.copy()))
        probability = (0.8 - 0.03 * self.value[0]) if old_task else (0.2 + 0.03 * self.value[0])
        result = np.full((len(features), 3), (1 - probability) / 2)
        result[:, 1 if old_task else 0] = probability
        if "predict" in self.hooks:
            self.hooks["predict"](self, features, result)
        return result


def packet(role, context, identity, *, width=2, rows=9, episode=False):
    marker = -0.75 if role == "train" else 0.25
    features = np.full((rows, width), marker, dtype=np.float64)
    targets = np.full((rows, 1), 1 if role == "train" else 0, dtype=np.float64)
    metadata = None
    if episode:
        observation = tuple(float(value) for value in features[0])
        metadata = DemonstrationEpisode(
            identity,
            context,
            role,
            tuple(
                DemonstrationTransition(
                    index,
                    observation,
                    int(targets[index, 0]),
                    0.0,
                    observation,
                    index == rows - 1,
                    3,
                )
                for index in range(rows)
            ),
            3,
        )
    return AttributionBatch(
        role,
        context,
        tuple(f"{identity}:step:{index}" for index in range(rows)),
        features,
        targets,
        metadata,
        3,
    )


def cohorts(*, width=2, episode=False):
    training = tuple(
        packet("probe_train", f"F-{index % 2}", f"F-train-{index}", width=width, episode=episode)
        for index in range(16)
    )
    evaluation = tuple(
        packet("probe_test", f"F-{index}", f"F-test-{index}", width=width, episode=episode)
        for index in range(2)
    )
    monitors = tuple(
        packet("train", "A", f"A-cue-{index}", width=width, episode=episode) for index in range(2)
    )
    return training, evaluation, monitors


def run(source, *, data=None, budget=None, **kwargs):
    training, evaluation, monitors = cohorts() if data is None else data
    return run_fresh_task_probe(
        source,
        np.array([0.0]),
        training,
        evaluation,
        budget=session(16) if budget is None else budget,
        checkpoint_updates=(0, 4, 8, 16),
        timer=kwargs.pop("timer", lambda: 2.5),
        training_readouts=monitors,
        **kwargs,
    )


@pytest.mark.parametrize("episode", [False, True])
def test_should_measure_old_fit_on_the_same_probe_without_old_task_updates(episode):
    source = ReadoutLearner()
    data = cohorts(episode=episode)
    before = pickle.dumps((data, source.value))
    budget = session(16)
    report = run(source, data=data, budget=budget, decision_diagnostics=True)
    assert report["status"] == "complete", report["reason"]
    assert report["completed_updates"] == budget.updates_completed == 16
    assert report["completed_training_examples"] == 144
    assert report["score_forward_calls_completed"] == 8
    old = report["training_readouts"]
    assert old["status"] == "complete" and old["reason"] is None
    assert old["score_forward_calls_attempted"] == old["score_forward_calls_completed"] == 8
    assert old["attempted_rows"] == old["completed_rows"] == 72
    assert old["reserved_data_bytes"] == 2 * 9 * 3 * 8
    assert [point["updates"] for point in old["checkpoints"]] == [0, 4, 8, 16]
    names = {"A-cue-0", "A-cue-1"} if episode else {"batch-0", "batch-1"}
    for point in old["checkpoints"]:
        assert point["status"] == "complete"
        assert set(point["scores"]) == set(point["decision_diagnostics"]) == names
        for score in point["scores"].values():
            assert score["ce"] == pytest.approx(-np.log(0.8 - 0.03 * point["updates"]))
    assert all(
        event[2][0, 0] > 0 and np.all(event[3] == 0)
        for event in source.trace
        if event[0] == "train"
    )
    assert [(event[0], event[1]) for event in source.trace if event[0] in {"A", "F"}] == [
        (kind, point) for point in (0, 4, 8, 16) for kind in ("F", "F", "A", "A")
    ]
    assert source.child is not source and source.child.value[0] == 16
    assert pickle.dumps((data, source.value)) == before
    assert (
        next(iter(old["checkpoints"][0]["scores"].values()))["ce"]
        < next(iter(old["checkpoints"][-1]["scores"].values()))["ce"]
    )
    baseline = run(ReadoutLearner(), data=(*data[:2], ()), decision_diagnostics=True)
    assert {key: value for key, value in report.items() if key != "training_readouts"} == baseline


def test_should_preserve_default_probe_report_when_empty_readouts_are_explicit():
    training, evaluation, _ = cohorts()
    options = dict(budget=session(16), checkpoint_updates=(0, 4, 8, 16), timer=lambda: 2.5)
    ordinary = run_fresh_task_probe(
        ReadoutLearner(), np.array([0.0]), training, evaluation, **options
    )
    explicit = run(ReadoutLearner(), data=(training, evaluation, ()))
    assert ordinary == explicit
    assert "training_readouts" not in ordinary


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "too_many",
        "final",
        "probe_test",
        "probe_train",
        "width",
        "classes",
        "row_ids",
        "shared_features",
        "shared_targets",
        "mixed_episodes",
        "late_nonfinite",
    ],
)
def test_should_validate_the_complete_readout_envelope_before_any_callback(monkeypatch, fault):
    training, evaluation, monitors = cohorts()
    values: Any = list(monitors)
    if fault == "list":
        pass
    elif fault == "too_many":
        values = (*monitors, packet("train", "A", "extra"))
    elif fault in {"final", "probe_test", "probe_train"}:
        values[1] = replace(values[1], role={"final": "final_test"}.get(fault, fault))
    elif fault == "width":
        values[1] = packet("train", "A", "wide", width=3)
    elif fault == "classes":
        values[1] = replace(values[1], num_classes=4)
    elif fault == "row_ids":
        values[1] = replace(values[1], row_ids=training[0].row_ids)
    elif fault == "shared_features":
        values[1] = replace(values[1], features=training[0].features)
    elif fault == "shared_targets":
        values[1] = replace(values[1], targets=training[0].targets)
    elif fault == "mixed_episodes":
        values[1] = packet("train", "A", "complete", episode=True)
    else:
        values[1].features[-1, -1] = np.nan
    monitors = values if fault == "list" else tuple(values)
    source = ReadoutLearner()
    calls = []

    def no_copy(batch):
        pytest.fail("invalid readouts reached owned batch allocation")

    def timer():
        calls.append("timer")
        return 2.5

    monkeypatch.setattr(probe_module, "_copy_batch", no_copy)
    with pytest.raises(ValueError):
        run(
            source,
            data=(training, evaluation, monitors),
            timer=timer,
            check_boundary=lambda: calls.append("guard"),
        )
    assert source.trace == calls == []


def test_should_apply_the_original_entry_limit_to_probe_and_readout_arrays_together():
    training, evaluation, _ = cohorts(width=296)
    monitors = tuple(
        packet("train", "A", f"large-{index}", width=296, rows=30) for index in range(2)
    )
    assert (
        sum(batch.features.size + batch.targets.size for batch in (*training, *evaluation)) < 65536
    )
    source = ReadoutLearner()
    with pytest.raises(ValueError):
        run(source, data=(training, evaluation, monitors))
    assert source.trace == []


def test_should_own_readouts_before_a_source_fork_can_change_caller_inputs():
    data = cohorts()
    expected = [batch.features.copy() for batch in data[2]]

    def mutate_callers():
        for batch in data[2]:
            batch.features.fill(0.99)
            batch.targets.fill(2)

    source = ReadoutLearner(hooks={"fork": mutate_callers})
    report = run(source, data=data)
    assert report["status"] == "complete", report["reason"]
    for index, event in enumerate(item for item in source.trace if item[0] == "A"):
        np.testing.assert_array_equal(event[2], expected[index % 2])


@pytest.mark.parametrize("failure", ["predict", "post_return_guard"])
def test_should_keep_the_paid_f_checkpoint_and_monitor_costs_when_a_later_readout_fails(failure):
    failed = [False]

    def predict(learner, features, probabilities):
        if features[0, 0] < 0 and learner.value[0] == 4:
            if failure == "predict":
                raise RuntimeError("old TRAIN readout failed")
            failed[0] = True

    def guard():
        if failed[0]:
            raise ValueError("paid readout boundary changed")

    source = ReadoutLearner(hooks={"predict": predict})
    budget = session(16)
    report = run(source, budget=budget, check_boundary=guard)
    assert report["status"] == "failed" and report["reason"]
    assert report["completed_updates"] == budget.updates_completed == 4
    assert [point["updates"] for point in report["checkpoints"]] == [0, 4]
    assert report["score_forward_calls_completed"] == 4
    old = report["training_readouts"]
    assert old["status"] == "failed" and old["score_forward_calls_attempted"] == 3
    returned = 2 if failure == "predict" else 3
    assert (
        old["score_forward_calls_completed"] == returned and old["completed_rows"] == returned * 9
    )
    assert old["attempted_rows"] == 27
    assert [point["updates"] for point in old["checkpoints"]] == [0, 4]
    assert (
        old["checkpoints"][0]["status"] == "complete"
        and old["checkpoints"][1]["status"] == "failed"
    )
    assert sum(event[0] == "train" for event in source.trace) == 4
