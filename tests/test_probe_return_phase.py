"""A fixed old-task return uses the already F-adapted disposable learner.

The deterministic fake policy makes fit gains and losses visible without native
learning or environment execution. Its trace is the independent routing oracle.
"""

from dataclasses import replace
from copy import deepcopy
import pickle
from typing import Any

import numpy as np
import pytest

import src.app.fresh_task_probe as probe
from test_fresh_task_probe import session
from test_probe_training_readouts import ReadoutLearner, cohorts, packet


class ReturnLearner(ReadoutLearner):
    def fork(self):
        self.trace.append(("fork",))
        if "fork" in self.hooks:
            self.hooks["fork"]()
        self.child = ReturnLearner(trace=self.trace, hooks=self.hooks)
        return self.child

    def train_batch(self, features, targets):
        result = super().train_batch(features, targets)
        if "train" in self.hooks:
            self.hooks["train"](self, features, targets)
        return result

    def predict(self, features):
        old_task = bool(features[0, 0] < 0)
        count = int(self.value[0])
        self.trace.append(("A" if old_task else "F", count, features.copy()))
        if count <= 16:
            probability = (0.8 - 0.03 * count) if old_task else (0.2 + 0.03 * count)
        else:
            probability = (0.32 + 0.03 * (count - 16)) if old_task else (0.68 - 0.02 * (count - 16))
        result = np.full((len(features), 3), (1 - probability) / 2)
        result[:, 1 if old_task else 0] = probability
        if "predict" in self.hooks:
            self.hooks["predict"](self, features, result)
        return result


def return_templates(monitors):
    result = []
    for index, original in enumerate(monitors):
        identity = f"return-A-{index}"
        episode = (
            None if original.episode is None else replace(original.episode, episode_id=identity)
        )
        result.append(
            replace(
                original,
                row_ids=tuple(f"{identity}:step:{row}" for row in range(len(original.features))),
                features=original.features.copy(),
                targets=original.targets.copy(),
                episode=episode,
            )
        )
    return tuple(result)


def run(source, *, data=None, returns=None, budget=None, return_budget=None, **options):
    training, evaluation, monitors = cohorts() if data is None else data
    return probe.run_fresh_task_probe(
        source,
        np.array([0.0]),
        training,
        evaluation,
        budget=session(16) if budget is None else budget,
        checkpoint_updates=(0, 4, 8, 16),
        timer=options.pop("timer", lambda: 2.5),
        decision_diagnostics=True,
        training_readouts=monitors,
        return_training=return_templates(monitors) if returns is None else returns,
        return_budget=session(16) if return_budget is None else return_budget,
        **options,
    )


@pytest.mark.parametrize("episode", [False, True])
def test_should_return_to_a_on_the_same_f_adapted_fork_and_keep_the_original_f_report(episode):
    source = ReturnLearner()
    data = cohorts(episode=episode)
    second = data[2][1]
    different = np.full_like(second.features, -0.5)
    metadata = (
        None
        if second.episode is None
        else replace(
            second.episode,
            transitions=tuple(
                replace(transition, observation=(-0.5, -0.5), next_observation=(-0.5, -0.5))
                for transition in second.episode.transitions
            ),
        )
    )
    data = (*data[:2], (data[2][0], replace(second, features=different, episode=metadata)))
    templates = return_templates(data[2])
    before = pickle.dumps((source.value, data, templates))
    first_budget, second_budget = session(16), session(16)
    report = run(
        source, data=data, returns=templates, budget=first_budget, return_budget=second_budget
    )
    baseline = probe.run_fresh_task_probe(
        ReturnLearner(),
        np.array([0.0]),
        *data[:2],
        budget=session(16),
        checkpoint_updates=(0, 4, 8, 16),
        timer=lambda: 2.5,
        decision_diagnostics=True,
        training_readouts=data[2],
    )
    assert report["status"] == "complete", report["reason"]
    assert {key: value for key, value in report.items() if key != "return_phase"} == baseline
    returned = report["return_phase"]
    assert returned["status"] == "complete", returned["reason"]
    assert (
        returned["completed_updates"]
        == returned["attempted_updates"]
        == second_budget.updates_completed
        == 16
    )
    assert report["completed_updates"] == first_budget.updates_completed == 16
    assert returned["completed_training_examples"] == returned["attempted_training_examples"] == 144
    assert (
        returned["score_forward_calls_completed"] == returned["score_forward_calls_attempted"] == 8
    )
    assert returned["reserved_data_bytes"] == 2 * 9 * 3 * 8
    assert returned["training_readouts"]["score_forward_calls_completed"] == 8
    assert returned["training_readouts"]["completed_rows"] == 72
    assert [point["updates"] for point in returned["checkpoints"]] == [0, 4, 8, 16]
    assert returned["checkpoints"][0]["scores"] == report["checkpoints"][-1]["scores"]
    assert (
        returned["training_readouts"]["checkpoints"][0]["scores"]
        == report["training_readouts"]["checkpoints"][-1]["scores"]
    )
    trained = [event for event in source.trace if event[0] == "train"]
    assert len(trained) == 32 and [event[1] for event in trained] == list(range(32))
    assert all(event[2][0, 0] > 0 and np.all(event[3] == 0) for event in trained[:16])
    assert all(event[2][0, 0] < 0 and np.all(event[3] == 1) for event in trained[16:])
    for index, event in enumerate(trained[16:]):
        np.testing.assert_array_equal(event[2], templates[index % 2].features)
    assert (
        sum(event[0] == "fork" for event in source.trace)
        == sum(event[0] == "restore" for event in source.trace)
        == 1
    )
    assert source.child.value[0] == 32 and pickle.dumps((source.value, data, templates)) == before
    assert (
        next(iter(returned["checkpoints"][-1]["scores"].values()))["ce"]
        > next(iter(returned["checkpoints"][0]["scores"].values()))["ce"]
    )
    assert (
        next(iter(returned["training_readouts"]["checkpoints"][-1]["scores"].values()))["ce"]
        < next(iter(returned["training_readouts"]["checkpoints"][0]["scores"].values()))["ce"]
    )


def test_should_preserve_the_default_report_with_an_explicit_disabled_return():
    training, evaluation, monitors = cohorts()
    options = dict(checkpoint_updates=(0, 4, 8, 16), timer=lambda: 2.5, training_readouts=monitors)
    ordinary = probe.run_fresh_task_probe(
        ReturnLearner(), np.array([0.0]), training, evaluation, budget=session(16), **options
    )
    disabled = probe.run_fresh_task_probe(
        ReturnLearner(),
        np.array([0.0]),
        training,
        evaluation,
        budget=session(16),
        return_training=(),
        return_budget=None,
        **options,
    )
    assert ordinary == disabled and "return_phase" not in ordinary


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "one",
        "three",
        "final",
        "probe_test",
        "values",
        "targets",
        "context",
        "width",
        "classes",
        "old_ids",
        "shared_array",
        "swapped",
        "no_monitors",
        "missing_budget",
        "same_budget",
        "wrong_limit",
        "spent_budget",
        "unused_budget",
    ],
)
def test_should_admit_the_whole_fixed_return_contract_before_copies_or_callbacks(
    monkeypatch, fault
):
    training, evaluation, monitors = cohorts()
    templates: Any = list(return_templates(monitors))
    first_budget, second_budget = session(16), session(16)
    if fault == "list":
        pass
    elif fault == "one":
        templates = templates[:1]
    elif fault == "three":
        templates.append(packet("train", "A", "third"))
    elif fault in {"final", "probe_test"}:
        templates[1] = replace(templates[1], role="final_test" if fault == "final" else fault)
    elif fault == "values":
        templates[1].features[-1, -1] = -0.5
    elif fault == "targets":
        templates[1].targets[-1, 0] = 2
    elif fault == "context":
        templates[1] = replace(templates[1], context="B")
    elif fault == "width":
        templates[1] = packet("train", "A", "wide", width=3)
    elif fault == "classes":
        templates[1] = replace(templates[1], num_classes=4)
    elif fault == "old_ids":
        templates[1] = replace(templates[1], row_ids=monitors[1].row_ids)
    elif fault == "shared_array":
        templates[1] = replace(templates[1], features=monitors[1].features)
    elif fault == "swapped":
        # Give the second cue different values while keeping its update copy
        # identical, then reverse only the fixed return correspondence.
        monitors[1].features.fill(-0.5)
        templates = list(return_templates(monitors))[::-1]
    elif fault == "no_monitors":
        monitors = ()
    elif fault == "missing_budget":
        second_budget = None
    elif fault == "same_budget":
        second_budget = first_budget
    elif fault == "wrong_limit":
        second_budget = session(15)
    elif fault == "spent_budget":
        second_budget.record_update()
    else:
        templates = []
    templates = templates if fault == "list" else tuple(templates)
    source = ReturnLearner()
    calls = []

    def no_copy(batch):
        pytest.fail("invalid return reached owned copies")

    def timer():
        calls.append("timer")
        return 2.5

    monkeypatch.setattr(probe, "_copy_batch", no_copy)
    with pytest.raises(ValueError):
        probe.run_fresh_task_probe(
            source,
            np.array([0.0]),
            training,
            evaluation,
            budget=first_budget,
            checkpoint_updates=(0, 4, 8, 16),
            timer=timer,
            training_readouts=monitors,
            return_training=templates,
            return_budget=second_budget,
            check_boundary=lambda: calls.append("guard"),
        )
    assert source.trace == calls == []


def test_should_keep_the_original_combined_array_cap_when_adding_a_return():
    training, evaluation, _ = cohorts(width=296)
    monitors = tuple(
        packet("train", "A", f"A-big-{index}", width=296, rows=20) for index in range(2)
    )
    templates = return_templates(monitors)
    assert (
        sum(
            batch.features.size + batch.targets.size
            for batch in (*training, *evaluation, *monitors)
        )
        < 65536
    )
    source = ReturnLearner()
    with pytest.raises(ValueError):
        run(source, data=(training, evaluation, monitors), returns=templates)
    assert source.trace == []


def test_should_detach_both_return_templates_before_the_original_fork():
    data = cohorts()
    templates = return_templates(data[2])
    expected = [batch.features.copy() for batch in templates]

    def change_inputs():
        for batch in templates:
            batch.features.fill(0.99)
            batch.targets.fill(2)

    source = ReturnLearner(hooks={"fork": change_inputs})
    report = run(source, data=data, returns=templates)
    assert report["return_phase"]["status"] == "complete"
    actual = [event for event in source.trace if event[0] == "train"][16:]
    for index, event in enumerate(actual):
        np.testing.assert_array_equal(event[2], expected[index % 2])
        np.testing.assert_array_equal(event[3], np.ones((9, 1)))


@pytest.mark.parametrize("phase", ["F", "A_monitor"])
def test_should_not_start_return_work_when_the_original_probe_or_monitor_fails(phase):
    def fail(learner, features, probabilities):
        if (features[0, 0] < 0) == (phase == "A_monitor"):
            raise RuntimeError("original phase failed")

    source = ReturnLearner(hooks={"predict": fail})
    second_budget = session(16)
    report = run(source, return_budget=second_budget)
    assert report["status"] == "failed" and report["return_phase"]["status"] == "not_run"
    assert second_budget.updates_completed == 0
    assert not any(event[0] == "train" for event in source.trace)


def test_should_preserve_a_paid_return_update_when_the_post_update_guard_fails():
    dirty = [False]

    def on_train(learner, features, targets):
        if learner.value[0] == 17:
            dirty[0] = True

    def guard():
        if dirty[0]:
            raise ValueError("return boundary changed after commit")

    source = ReturnLearner(hooks={"train": on_train})
    first_budget, second_budget = session(16), session(16)
    report = run(source, budget=first_budget, return_budget=second_budget, check_boundary=guard)
    assert (
        report["status"] == "complete"
        and report["completed_updates"] == first_budget.updates_completed == 16
    )
    returned = report["return_phase"]
    assert returned["status"] == "failed" and returned["reason"]
    assert (
        returned["attempted_updates"]
        == returned["completed_updates"]
        == second_budget.updates_completed
        == 1
    )
    assert returned["completed_training_examples"] == 9
    assert [point["updates"] for point in returned["checkpoints"]] == [0]
    assert returned["training_readouts"]["completed_rows"] == 18
    assert len([event for event in source.trace if event[0] == "train"]) == 17


@pytest.mark.parametrize("fault", ["score_value", "equal_f_checkpoints", "equal_a_checkpoints"])
def test_should_refuse_changes_to_original_paid_evidence_during_the_return(monkeypatch, fault):
    actual_guard = probe.paid_report_guard
    owners = []
    originals = []
    changed = [False]

    def capture_paid_report(report):
        owners.append(report)
        originals.append(deepcopy(report))
        return actual_guard(report)

    def corrupt(learner, features, probabilities):
        if owners and not changed[0]:
            assert learner.value[0] == 16 and features[0, 0] > 0
            owner = owners[0]
            if fault == "score_value":
                owner["checkpoints"][-1]["scores"].clear()
            elif fault == "equal_f_checkpoints":
                owner["checkpoints"] = deepcopy(owner["checkpoints"])
            else:
                old = owner["training_readouts"]
                old["checkpoints"] = deepcopy(old["checkpoints"])
            changed[0] = True

    monkeypatch.setattr(probe, "paid_report_guard", capture_paid_report)
    source = ReturnLearner(hooks={"predict": corrupt})
    original_budget, returned_budget = session(16), session(16)
    report = run(source, budget=original_budget, return_budget=returned_budget)
    assert changed[0] and len(originals) == 1
    assert {key: value for key, value in report.items() if key != "return_phase"} == originals[0]
    assert report["status"] == "complete" and original_budget.updates_completed == 16
    returned = report["return_phase"]
    assert returned["status"] == "failed" and returned["reason"]
    assert (
        returned["completed_updates"]
        == returned["attempted_updates"]
        == returned_budget.updates_completed
        == 0
    )
    assert (
        returned["score_forward_calls_attempted"] == returned["score_forward_calls_completed"] == 1
    )
    assert returned["checkpoints"] == []
    assert sum(event[0] == "train" for event in source.trace) == 16
    assert sum(event[0] in {"F", "A"} for event in source.trace) == 17


@pytest.mark.parametrize("fault", ["return_clock", "return_started_at", "original_f_clock"])
def test_should_bind_both_original_budgets_during_return_without_an_external_guard(fault):
    original_budget, returned_budget = session(16), session(16)
    changed = [False]

    def corrupt_binding_after_native_update(learner, features, targets):
        if learner.value[0] == 17:
            assert features[0, 0] < 0 and np.all(targets == 1)
            if fault == "return_clock":
                returned_budget.clock = lambda: 0.0
            elif fault == "return_started_at":
                returned_budget.started_at += 0.25
            else:
                original_budget.clock = lambda: 0.0
            changed[0] = True

    source = ReturnLearner(hooks={"train": corrupt_binding_after_native_update})
    # Deliberately supply no check_boundary: the standalone use-case owns its
    # original resource bindings rather than relying on an adapter callback.
    report = run(source, budget=original_budget, return_budget=returned_budget)

    assert changed[0]
    assert report["status"] == "complete" and report["reason"] is None
    assert report["completed_updates"] == original_budget.updates_completed == 16
    assert report["completed_training_examples"] == 144
    returned = report["return_phase"]
    assert returned["status"] == "failed" and returned["reason"]
    assert (
        returned["attempted_updates"]
        == returned["completed_updates"]
        == returned_budget.updates_completed
        == 1
    )
    assert returned["attempted_training_examples"] == returned["completed_training_examples"] == 9
    assert [point["updates"] for point in returned["checkpoints"]] == [0]
    assert returned["score_forward_calls_completed"] == 2
    assert returned["training_readouts"]["score_forward_calls_completed"] == 2
    assert sum(event[0] == "train" for event in source.trace) == 17
    assert sum(event[0] in {"F", "A"} for event in source.trace) == 20
    np.testing.assert_array_equal(source.value, [99.0])
