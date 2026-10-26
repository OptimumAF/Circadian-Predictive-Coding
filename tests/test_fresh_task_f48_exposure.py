"""Explicit paid F48 uses the ordinary disposable loop through scripted ports.

These fakes test routing, authority and immutable evidence. They execute no neural
updates, native predictions or environment steps and establish no learning benefit.
"""

from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from src.app import fresh_task_probe as probe
from test_fresh_task_probe import session
from test_probe_return_phase import ReturnLearner, return_templates
from test_probe_training_readouts import cohorts, packet


class ScriptedLearner(ReturnLearner):
    def fork(self):
        self.trace.append(("fork",))
        if "fork" in self.hooks:
            self.hooks["fork"]()
        self.child = ScriptedLearner(trace=self.trace, hooks=self.hooks)
        return self.child

    def predict(self, features):
        old_task = bool(features[0, 0] < 0)
        self.trace.append(("A" if old_task else "F", int(self.value[0]), features.copy()))
        # Why: constant probabilities stay valid over all64 scripted updates;
        # changing accuracy is not needed to prove routing and paid accounting.
        result = np.full((len(features), 3), 0.2)
        result[:, 1 if old_task else 0] = 0.6
        if "predict" in self.hooks:
            self.hooks["predict"](self, features, result)
        return result


def run(source=None, *, data=None, returns=None, wake=None, returning=None, **options):
    source = ScriptedLearner() if source is None else source
    training, evaluation, monitors = cohorts() if data is None else data
    return probe.run_fresh_task_probe(
        source,
        np.array([0.0]),
        training,
        evaluation,
        budget=session(48) if wake is None else wake,
        checkpoint_updates=options.pop("checkpoint_updates", (0, 12, 24, 48)),
        timer=options.pop("timer", lambda: 2.5),
        training_readouts=monitors,
        return_training=return_templates(monitors) if returns is None else returns,
        return_budget=session(16) if returning is None else returning,
        extra_f_exposure=options.pop("extra_f_exposure", True),
        **options,
    )


@pytest.mark.parametrize("episode", [False, True])
def test_should_run_three_owned_f_schedules_then_return_on_the_same_candidate(episode):
    source, data = ScriptedLearner(), cohorts(episode=episode)
    templates = return_templates(data[2])
    before = pickle.dumps((source.value, data, templates))
    wake, returning = session(48), session(16)
    authority = (
        wake.budget,
        returning.budget,
        wake.clock,
        returning.clock,
        wake.started_at,
        returning.started_at,
    )
    endpoints = []

    def observe(learner, phase, boundary):
        boundary(
            lambda: endpoints.append(
                (
                    phase,
                    learner,
                    int(learner.value[0]),
                    wake.updates_completed,
                    returning.updates_completed,
                )
            )
        )

    report = run(
        source,
        data=data,
        returns=templates,
        wake=wake,
        returning=returning,
        endpoint_observer=observe,
    )
    assert report["status"] == "complete", report["reason"]
    assert report["return_phase"]["status"] == "complete", report["return_phase"]["reason"]
    assert (
        report["attempted_updates"] == report["completed_updates"] == wake.updates_completed == 48
    )
    assert report["attempted_training_examples"] == report["completed_training_examples"] == 432
    returned = report["return_phase"]
    assert (
        returned["completed_updates"]
        == returned["attempted_updates"]
        == returning.updates_completed
        == 16
    )
    assert returned["completed_training_examples"] == 144
    assert wake.replay_examples_completed == returning.replay_examples_completed == 0
    assert [point["updates"] for point in report["checkpoints"]] == [0, 12, 24, 48]
    assert [point["updates"] for point in returned["checkpoints"]] == [0, 4, 8, 16]
    for owner in (report, returned):
        assert owner["score_forward_calls_completed"] == owner["score_forward_calls_attempted"] == 8
        assert owner["training_readouts"]["score_forward_calls_completed"] == 8
    assert returned["checkpoints"][0]["scores"] == report["checkpoints"][-1]["scores"]
    assert (
        returned["training_readouts"]["checkpoints"][0]["scores"]
        == report["training_readouts"]["checkpoints"][-1]["scores"]
    )
    trains = [event for event in source.trace if event[0] == "train"]
    assert [event[1] for event in trains] == list(range(64))
    for index, event in enumerate(trains[:48]):
        np.testing.assert_array_equal(event[2], data[0][index % 16].features)
        np.testing.assert_array_equal(event[3], data[0][index % 16].targets)
    for index, event in enumerate(trains[48:]):
        np.testing.assert_array_equal(event[2], templates[index % 2].features)
    assert [(phase, count, f, a) for phase, _, count, f, a in endpoints] == [
        ("after_probe", 48, 48, 0),
        ("after_return", 64, 48, 16),
    ]
    assert endpoints[0][1] is endpoints[1][1] is source.child
    assert (
        sum(event[0] == "fork" for event in source.trace)
        == sum(event[0] == "restore" for event in source.trace)
        == 1
    )
    assert authority == (
        wake.budget,
        returning.budget,
        wake.clock,
        returning.clock,
        wake.started_at,
        returning.started_at,
    )
    assert pickle.dumps((source.value, data, templates)) == before
    assert "rehearsal" not in report and "probe_stream" not in report
    exposure = report["extra_f_exposure"]
    assert exposure["input_training_batches"] == 16
    assert exposure["schedule_repetitions"] == 3
    assert exposure["native_training_updates"] == 48
    assert exposure["planned_training_rows"] == 432
    assert exposure["checkpoint_mapping"] == {
        "original_f16_updates": [0, 4, 8, 16],
        "native_updates": [0, 12, 24, 48],
    }
    assert exposure["replay_native_calls"] == exposure["replay_training_rows"] == 0


def test_should_keep_original_input_storage_bound_without_tripling_reserved_arrays():
    data = cohorts(width=296)
    source = ScriptedLearner()
    report = run(source, data=data)
    assert report["status"] == report["return_phase"]["status"] == "complete"
    expected = sum(batch.features.nbytes + batch.targets.nbytes for batch in (*data[0], *data[1]))
    assert report["reserved_data_bytes"] == expected
    monitors = tuple(
        packet("train", "A", f"A-big-{index}", width=296, rows=20) for index in range(2)
    )
    source = ScriptedLearner()
    with pytest.raises(ValueError, match="65536"):
        run(source, data=(*data[:2], monitors))
    assert source.trace == []


def test_should_repeat_owned_original_inputs_after_the_external_fork_changes_callers():
    data = cohorts()
    templates = return_templates(data[2])
    original_f = [batch.features.copy() for batch in data[0]]
    original_a = [batch.features.copy() for batch in templates]

    def mutate_callers():
        for batch in (*data[0], *data[1], *data[2], *templates):
            batch.features.fill(0.99)
            batch.targets.fill(2)

    source = ScriptedLearner(hooks={"fork": mutate_callers})
    report = run(source, data=data, returns=templates)
    assert report["status"] == report["return_phase"]["status"] == "complete"
    assert report["extra_f_exposure"]["planned_training_rows"] == 432
    trains = [event for event in source.trace if event[0] == "train"]
    for index, event in enumerate(trains[:48]):
        np.testing.assert_array_equal(event[2], original_f[index % 16])
        np.testing.assert_array_equal(event[3], np.zeros((9, 1)))
    for index, event in enumerate(trains[48:]):
        np.testing.assert_array_equal(event[2], original_a[index % 2])
        np.testing.assert_array_equal(event[3], np.ones((9, 1)))


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_exposure_bool_before_copy_or_callback(monkeypatch, flag):
    source = ScriptedLearner()
    monkeypatch.setattr(
        probe, "_copy_batch", lambda *args: pytest.fail("invalid input reached copies")
    )
    with pytest.raises(ValueError, match="extra_f_exposure"):
        run(source, extra_f_exposure=flag)
    assert source.trace == []


@pytest.mark.parametrize(
    "fault",
    [
        "training",
        "evaluation",
        "monitors",
        "returns",
        "budget16",
        "budget49",
        "spent",
        "spent_replay",
        "shared",
        "return48",
        "points16",
        "points_other",
        "rehearsal",
    ],
)
def test_should_refuse_incomplete_f48_authority_before_copy_fork_or_clock(monkeypatch, fault):
    source, data = ScriptedLearner(), cohorts()
    templates, wake, returning = return_templates(data[2]), session(48), session(16)
    options = {}
    if fault == "training":
        data = (data[0][:-1], *data[1:])
    elif fault == "evaluation":
        data = (data[0], data[1][:1], data[2])
    elif fault == "monitors":
        data = (*data[:2], ())
    elif fault == "returns":
        templates = ()
    elif fault in ("budget16", "budget49"):
        wake = session(16 if fault == "budget16" else 49)
    elif fault == "spent":
        wake.record_update()
    elif fault == "spent_replay":
        wake.replay_examples_completed = 1
    elif fault == "shared":
        returning = wake
    elif fault == "return48":
        returning = session(48)
    elif fault in ("points16", "points_other"):
        options["checkpoint_updates"] = (0, 4, 8, 16) if fault == "points16" else (0, 16, 32, 48)
    else:
        options["rehearsal_factory"] = lambda *args: pytest.fail("invalid replay reached factory")
    monkeypatch.setattr(
        probe, "_copy_batch", lambda *args: pytest.fail("invalid input reached copies")
    )
    with pytest.raises(ValueError):
        run(source, data=data, returns=templates, wake=wake, returning=returning, **options)
    assert source.trace == []


@pytest.mark.parametrize("failed_count", [17, 49])
def test_should_keep_paid_failure_suffix_without_retry_or_budget_renewal(failed_count):
    def fail(learner, features, targets):
        if int(learner.value[0]) == failed_count:
            raise RuntimeError("scripted paid update failed")

    source, wake, returning = ScriptedLearner(hooks={"train": fail}), session(48), session(16)
    report = run(source, wake=wake, returning=returning)
    owner = report if failed_count == 17 else report["return_phase"]
    assert owner["status"] == "failed" and "scripted paid update failed" in owner["reason"]
    assert owner["attempted_updates"] == (17 if failed_count == 17 else 1)
    assert owner["completed_updates"] == (16 if failed_count == 17 else 0)
    assert owner["attempted_training_examples"] == 9 * owner["attempted_updates"]
    assert owner["completed_training_examples"] == 9 * owner["completed_updates"]
    assert wake.updates_completed == (16 if failed_count == 17 else 48)
    assert returning.updates_completed == 0
    assert len([event for event in source.trace if event[0] == "train"]) == failed_count
    assert report["return_phase"]["status"] == ("not_run" if failed_count == 17 else "failed")
    np.testing.assert_array_equal(source.value, [99.0])


@pytest.mark.parametrize("fault", ["value", "equal_f", "equal_a"])
def test_should_refuse_corrupted_or_replaced_paid_f48_evidence_before_return_updates(
    monkeypatch, fault
):
    original_guard = probe.paid_report_guard
    owners, trusted = [], []

    def capture(report):
        owners.append(report)
        trusted.append(deepcopy(report))
        return original_guard(report)

    changed = []

    def corrupt(learner, features, probabilities):
        if owners and not changed:
            assert learner.value[0] == 48
            if fault == "value":
                owners[0]["checkpoints"][-1]["scores"].clear()
            elif fault == "equal_f":
                owners[0]["checkpoints"] = deepcopy(owners[0]["checkpoints"])
            else:
                old = owners[0]["training_readouts"]
                old["checkpoints"] = deepcopy(old["checkpoints"])
            changed.append(True)

    monkeypatch.setattr(probe, "paid_report_guard", capture)
    source, wake, returning = ScriptedLearner(hooks={"predict": corrupt}), session(48), session(16)
    report = run(source, wake=wake, returning=returning)
    assert changed == [True]
    assert {key: value for key, value in report.items() if key != "return_phase"} == trusted[0]
    assert report["status"] == "complete" and wake.updates_completed == 48
    assert report["return_phase"]["status"] == "failed"
    assert report["return_phase"]["attempted_updates"] == returning.updates_completed == 0
    assert len([event for event in source.trace if event[0] == "train"]) == 48


@pytest.mark.parametrize("fault", ["counter", "limits", "clock", "unobserved_clock"])
def test_should_refuse_endpoint_budget_drift_without_accepting_arbitrary_f_progress(fault):
    wake, returning = session(48), session(16)

    def observe(learner, phase, boundary):
        assert phase == "after_probe"
        if fault == "counter":
            wake.updates_completed = 49
        elif fault == "limits":
            wake.budget = replace(wake.budget, max_training_updates=49)
        elif fault == "clock":
            wake.clock = lambda: 0.0
        else:
            wake.last_clock += 0.1
        boundary()

    report = run(wake=wake, returning=returning, endpoint_observer=observe)
    assert report["status"] == "failed" and "endpoint observer" in report["reason"]
    assert report["completed_updates"] == 48
    assert report["return_phase"]["status"] == "not_run" and returning.updates_completed == 0


def test_should_preserve_disabled_exposure_report_and_original_f16_route():
    from test_probe_return_phase import run as ordinary

    assert ordinary(ReturnLearner()) == ordinary(ReturnLearner(), extra_f_exposure=False)
