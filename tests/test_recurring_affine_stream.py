"""Ordinary bounded TRAIN recurrence in the existing runner; no native games."""

from dataclasses import asdict, replace
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_learner as port
from src.app import coupled_attribution
from test_arrived_cue_catalogue import episode


def sources():
    return tuple(
        episode(f"minigrid-source-{context}-{name}", cue, context=context, mark=mark, action=action)
        for context, mark, action in (
            ("A", 1, 2),
            ("B", 3, 5),
            ("C_train", 5, 1),
            ("G_train", 7, 4),
        )
        for name, cue in (("blue", 2), ("red", 0))
    )


def run(packets, **options):
    return cli.run_context_linear_stream(
        packets, retain_arrived=True, recurring_catalogue=True, **options
    )


def test_should_execute_ten_delayed_fits_then_score_all_original_train_monitors(monkeypatch):
    packets = sources()
    before = pickle.dumps(packets), pickle.dumps(np.random.get_state())
    fit, predict = port.ContextLinearLearner.train_batch, port.ContextLinearLearner.predict
    fits, scores, boundaries = [], [], []
    endpoints: list[port.ContextLinearTrainingSnapshot] = []

    def train(self, x, y):
        fits.append((x.copy(), y.copy()))
        return fit(self, x, y)

    def read(self, x):
        assert len(fits) == 10  # TRAIN scores cannot affect schedule/selection.
        scores.append((self.completed_updates, self.training_examples, len(x)))
        return predict(self, x)

    monkeypatch.setattr(port.ContextLinearLearner, "train_batch", train)
    monkeypatch.setattr(port.ContextLinearLearner, "predict", read)
    result = run(
        packets, endpoint_observer=endpoints.append, check_boundary=lambda: boundaries.append(1)
    )
    assert result["status"] == "complete", result["reason"]
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert len(endpoints) == 1 and boundaries
    state = endpoints[0]
    assert (state.completed_updates, state.training_examples) == (10, 252)
    assert [None if head is None else head.fit_rows for head in state.heads] == [36, 36]
    assert result["endpoints"][-1]["state"] == asdict(state)
    assert [len(x) for x, y in fits] == [9, 9, 18, 18, 27, 27, 36, 36, 36, 36]
    sequence = (*range(8), 0, 1)
    for tick, (x, y) in enumerate(fits):
        old = tuple(range(tick % 2, tick, 2))[-3:]
        expected = tuple(packets[sequence[i]] for i in (*old, tick))
        assert y[:, 0].tolist() == [t.action for e in expected for t in e.transitions]
        assert x.tolist() == [list(t.observation) for e in expected for t in e.transitions]
    receipts = result["committed_receipts"]
    assert len(receipts) == result["budget"]["updates_completed"] == 10
    for i, receipt in enumerate(receipts):
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
            i,
            i + 1,
            i + 1,
        )
        assert receipt["update_number"] == i + 1
    assert result["observed_checkpoints"] == [0, 2, 4, 6, 8, 10]
    assert scores == [
        (n, rows, 9)
        for n, rows in ((0, 0), (2, 18), (4, 54), (6, 108), (8, 180), (10, 252))
        for _ in packets
    ]
    assert result["score_calls_attempted"] == result["score_calls_completed"] == 48
    assert result["score_rows_attempted"] == result["score_rows_completed"] == 432
    assert all(
        tuple(row["scores"]) == tuple(e.episode_id for e in packets) for row in result["endpoints"]
    )
    assert result["stream"]["evaluation_scope"] == "training_only"
    assert result["stream"]["training_examples"] == 252
    assert result["stream"]["score_forward_calls"] == 0
    costs = result["catalogue"]
    assert costs["committed_rows"] == dict(arrivals=90, replay=162, native=252)
    assert costs["peak_bank_rows"] == 54 and costs["input_array_bytes"] == 598752
    assert costs["peak_joint_template_array_bytes"] == 85536
    assert costs["peak_bank_observation_target_float64_equivalent_bytes"] == 256176
    assert costs["failure_work_may_be_unobserved"] is False
    assert result["budget"]["initial_limits"] == result["budget"]["limits"]
    assert result["budget"]["limits"]["max_training_updates"] == 10
    assert result["budget"]["replay_examples_completed"] == 0  # Joint replay is counted above.
    assert (pickle.dumps(packets), pickle.dumps(np.random.get_state())) == before


def test_should_keep_eight_paid_updates_and_failed_return_attempt_without_scoring(monkeypatch):
    fit = port.fit_categorical_linear
    calls = []
    delivered: list[port.ContextLinearTrainingSnapshot] = []

    def fail_ninth(x, y, classes):
        calls.append(len(x))
        if len(calls) == 9:
            raise RuntimeError("deliberate return fit failure")
        return fit(x, y, classes)

    monkeypatch.setattr(port, "fit_categorical_linear", fail_ninth)
    result = run(sources(), endpoint_observer=delivered.append)
    assert result["status"] == "failed" and delivered == []
    assert calls == [9, 9, 18, 18, 27, 27, 36, 36, 36]
    assert len(result["committed_receipts"]) == result["budget"]["updates_completed"] == 8
    assert result["budget"]["initial_limits"] == result["budget"]["limits"]
    assert result["observed_checkpoints"] == [0, 2, 4, 6, 8]
    assert result["unscored_checkpoints"][-1]["state"]["training_examples"] == 180
    assert result["catalogue"]["committed_rows"] == dict(arrivals=72, replay=108, native=180)
    assert result["catalogue"]["failed_attempt_replay_rows_upper_bound"] == 27
    assert result["catalogue"]["failure_work_may_be_unobserved"] is True
    assert result["failure_work"]["work"]["fit_rows_attempted"] == 36
    assert result["failure_work"]["work"]["failure_work_may_be_unobserved"] is True
    assert result["score_calls_attempted"] == result["score_rows_attempted"] == 0
    assert result["source_unchanged"] is result["actor_unchanged"] is None


@pytest.mark.parametrize(
    "options",
    [
        dict(recurring_catalogue=1),
        dict(recurring_catalogue=None),
        dict(recurring_catalogue=True),
        dict(recurring_catalogue=True, retain_arrived=True, abc_catalogue=True),
        dict(recurring_catalogue=True, retain_arrived=True, check_boundary=1),
    ],
)
def test_should_refuse_ambiguous_recurring_options_before_sources_or_owners(monkeypatch, options):
    monkeypatch.setattr(port, "ContextLinearLearner", lambda: pytest.fail("owner reached"))
    with pytest.raises(ValueError):
        cli.run_context_linear_stream((), **options)


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "missing",
        "order",
        "short",
        "probe_train",
        "probe_test",
        "final_test",
        "controller_validation",
        "cue",
    ],
)
def test_should_refuse_invalid_original_train_pairs_before_owners(monkeypatch, fault):
    packets = sources()
    if fault == "list":
        packets = list(packets)
    elif fault == "missing":
        packets = packets[:-2]
    elif fault == "order":
        packets = (*packets[2:], *packets[:2])
    elif fault == "short":
        packets = (
            *packets[:-1],
            episode(packets[-1].episode_id, cue=0, context="G_train", mark=7, rows=8),
        )
    elif fault == "controller_validation":
        # Constructor already refuses this role; test corrupted transport admission.
        object.__setattr__(packets[-1], "role", fault)
    elif fault == "cue":
        packets = (*packets[:-1], episode(packets[-1].episode_id, cue=2, context="G_train", mark=7))
    else:
        packets = (*packets[:-1], replace(packets[-1], role=fault))
    monkeypatch.setattr(port, "ContextLinearLearner", lambda: pytest.fail("owner reached"))
    with pytest.raises(ValueError):
        run(packets)


@pytest.mark.parametrize("fault", ["updates", "rows"])
def test_should_refuse_checkpoint_counter_drift_before_accepting_success(monkeypatch, fault):
    class BrokenStream:
        def __init__(self, *args, **options):
            options["observe_checkpoint"](
                8,
                SimpleNamespace(
                    completed_updates=9 if fault == "updates" else 8,
                    training_examples=181 if fault == "rows" else 180,
                ),
                None,
            )
            pytest.fail("incompatible checkpoint accepted")

    monkeypatch.setattr(coupled_attribution, "AttributionStream", BrokenStream)
    result = run(sources())
    assert result["status"] == "failed" and "counters disagree" in result["reason"]
    assert result["observed_checkpoints"] == [] and result["committed_receipts"] is None
    assert result["catalogue"]["committed_rows"] is None
    assert result["score_calls_attempted"] == 0


@pytest.mark.parametrize("fault", ["label", "equal_observation"])
def test_should_refuse_changed_future_original_source_at_outer_boundary(monkeypatch, fault):
    packets, checks = sources(), []

    def boundary():
        checks.append(1)
        if len(checks) == 2:
            row = packets[-1].transitions[-1]
            if fault == "label":
                object.__setattr__(row, "action", 6)
            else:
                object.__setattr__(row, "observation", tuple(list(row.observation)))

    monkeypatch.setattr(
        port.ContextLinearLearner, "train_batch", lambda *a: pytest.fail("fit reached")
    )
    result = run(packets, check_boundary=boundary)
    assert result["status"] == "failed" and "changed" in result["reason"]
    assert result["score_calls_attempted"] == result["budget"]["updates_completed"] == 0
