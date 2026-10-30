"""Two public-context LS heads: isolation, paid failures and actual recipe ports."""

from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from src.adapters import linear_fit_learner as linear
from src.adapters.context_linear_fit_learner import (
    ContextLinearFitLearner,
    CONTEXT_LINEAR_RESIDUAL_ID,
)
from src.adapters.context_linear_fit_decision import ContextLinearFitDecisionPredictor
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot
from src.core.context_linear_state import ContextLinearFitPolicy, ContextLinearFitSnapshot
from src.core.decision_features import DecisionFeatureSpec
from src.core.learner_ports import TrainingUpdateFailed
from src.core.ticket_workflow import CATALOG
from src.core.typed_decision import DecisionRequest


def policy():
    return ContextLinearFitPolicy(5, 3, (3, 4))


def batch(index=0):
    tags = np.zeros((3, 2))
    tags[:, index] = 1
    return np.column_stack((np.eye(3), tags)), np.arange(3, dtype=np.float64).reshape(3, 1)


def test_should_fit_one_head_and_preserve_inactive_head_without_row_bank():
    owner = ContextLinearFitLearner(policy())
    a, y = batch(0)
    b, _ = batch(1)
    seed = owner.snapshot_state()
    assert all(h.model is None for h in seed.heads)
    assert np.array_equal(owner.predict(np.vstack((a, b))), np.full((6, 3), 1 / 3))
    owner.train_batch(a, y)
    first = owner.snapshot_state()
    assert first.heads[1] == seed.heads[1] and first.completed_updates == 1
    before = owner.predict(a)
    child = owner.fork()
    d = child.train_batch(b, (y + 1) % 3)
    state = child.snapshot_state()
    assert d.definition == CONTEXT_LINEAR_RESIDUAL_ID
    assert (
        state.heads[0] == first.heads[0]
        and state.completed_updates == 2
        and state.training_examples == 6
    )
    assert np.array_equal(child.predict(a), before)
    assert np.argmax(child.predict(b), axis=1).tolist() == [1, 2, 0]
    assert owner.snapshot_state() == first
    assert set(state.__dataclass_fields__) == {
        "policy",
        "heads",
        "completed_updates",
        "training_examples",
    }
    assert all(h.model is not None and h.model.fit_rows == 3 for h in state.heads)
    object.__setattr__(state.heads[0].model, "squared_error", 999.0)
    assert child.snapshot_state() != state
    owner.restore_state(seed)
    assert owner.snapshot_state() == seed


@pytest.mark.parametrize("fault", ["missing", "both", "fraction", "mixed"])
def test_should_refuse_ambiguous_or_mixed_train_before_solver(fault, monkeypatch):
    owner = ContextLinearFitLearner(policy())
    x, y = batch()
    if fault == "missing":
        x[:, 3:] = 0
    elif fault == "both":
        x[:, 3:] = 1
    elif fault == "fraction":
        x[:, 3:] = 0.5
    else:
        x[1, 3:] = [0, 1]
    monkeypatch.setattr(linear, "fit_categorical_linear", lambda *a: pytest.fail("solver"))
    before = owner.snapshot_state()
    with pytest.raises(ValueError):
        owner.train_batch(x, y)
    assert owner.snapshot_state() == before


def test_should_reject_wrong_policy_and_inconsistent_head_counters():
    owner = ContextLinearFitLearner(policy())
    before = owner.snapshot_state()
    with pytest.raises(ValueError):
        owner.restore_state(replace(before, completed_updates=1))
    other = ContextLinearFitLearner(ContextLinearFitPolicy(5, 3, (2, 4)))
    with pytest.raises(ValueError):
        owner.restore_state(other.snapshot_state())
    assert owner.snapshot_state() == before


@pytest.mark.parametrize("fault", ["lost", "raw", "donor", "policy", "inactive", "rng"])
def test_should_preserve_donor_and_bounded_work_on_paid_failures(fault, monkeypatch):
    owner = ContextLinearFitLearner(policy())
    x, y = batch()
    before = owner.snapshot_state()
    original = linear.fit_categorical_linear
    changed = None
    rng = random.getstate()

    def fit(*args):
        nonlocal changed
        result = original(*args)
        if fault == "lost":
            raise RuntimeError("lost context solve")
        if fault == "raw":
            x[:, 3:] = [0, 1]
        elif fault == "donor":
            owner._state = deepcopy(owner._state)
        elif fault == "policy":
            owner._policy = replace(owner.policy)
        elif fault == "inactive":
            model = CategoricalLinearFit(3, 3, 1, ((0.0, 0.0, 0.0),) * 4, 0, (0.0,), 0.0)
            head = LinearFitSnapshot(LinearFitPolicy(3, 3), model, 1, 1)
            owner.restore_state(
                ContextLinearFitSnapshot(owner.policy, (before.heads[0], head), 1, 1)
            )
            changed = owner.snapshot_state()
        elif fault == "rng":
            random.seed(93)
        return result

    monkeypatch.setattr(linear, "fit_categorical_linear", fit)
    try:
        with pytest.raises(TrainingUpdateFailed) as failure:
            owner.train_batch(x, y)
        work = failure.value.work
        assert work["fit_attempts"] == 1 and work["fit_rows_attempted"] == 3
        assert work["candidate_commits"] == 0 and work["context_index"] == 0
        assert work["unreturned_work_unknown"] == (fault == "lost")
        assert owner.snapshot_state() == (changed if fault == "inactive" else before)
        if fault == "raw":
            assert np.array_equal(x[:, 3:], np.tile([0, 1], (3, 1)))
        if fault == "rng":
            assert random.getstate() != rng
    finally:
        random.setstate(rng)


def literal_reader():
    spec = DecisionFeatureSpec(entry.CONTEXT_FEATURES, CATALOG)
    columns = (spec.vocabulary.index("policy_v1"), spec.vocabulary.index("policy_v2"))
    p = ContextLinearFitPolicy(9, 6, columns)
    heads = []
    for index in (0, 1):
        bias = tuple(1.0 if c == index else 0.0 for c in range(6))
        model = CategoricalLinearFit(7, 6, 1, ((0.0,) * 6,) * 7 + (bias,), 0, (0.0,), 0.0)
        heads.append(LinearFitSnapshot(LinearFitPolicy(7, 6), model, 1, 1))
    return ContextLinearFitDecisionPredictor(
        spec, ContextLinearFitSnapshot(p, (heads[0], heads[1]), 2, 2)
    )


def test_should_route_only_public_tag_and_count_both_encodes():
    reader = literal_reader()
    for index, tag in enumerate(("policy_v1", "policy_v2")):
        request = DecisionRequest(f"r{index}", f"sealed invoice {tag}", CATALOG)
        result = reader(request)
        assert result.selected_id == CATALOG[index].candidate_id
        assert result.token_visits == 6
    work = reader.work
    assert work.decision_attempts == work.decision_returns == 2
    assert work.affine_attempts == work.routing_returns == 2 and work.nominal_affine_terms == 96
    assert work.routing_token_visits == 6 and work.token_visits == 12


@pytest.mark.parametrize(
    "query",
    ["invoice", "policy_v3 invoice", "policy_v1 policy_v2 invoice", "policy_v1 policy_v1 invoice"],
)
def test_should_refuse_unknown_or_duplicate_typed_tag_before_head_read(query):
    reader = literal_reader()
    with pytest.raises(ValueError, match="unambiguous"):
        reader(DecisionRequest("r", query, CATALOG))
    assert reader.work.decision_attempts == 1 and reader.work.affine_attempts == 0


def test_should_keep_paid_head_read_when_delegate_return_is_lost(monkeypatch):
    reader = literal_reader()
    selected = reader._original_readers[0]
    predict = selected.predict_logits

    def lost(features):
        predict(features)
        raise RuntimeError("lost head read")

    monkeypatch.setattr(selected, "predict_logits", lost)
    with pytest.raises(RuntimeError, match="lost head read"):
        reader(DecisionRequest("r", "invoice policy_v1", CATALOG))
    work = reader.work
    assert work.decision_attempts == 1 and work.decision_returns == 0
    assert work.routing_returns == work.routing_rows == 1
    assert work.affine_attempts == work.affine_returns == 1
    assert work.rows_returned == 1 and work.token_visits == 4
    reader.assert_unchanged()


@pytest.mark.parametrize("fault", ["state", "spec", "readers", "routing"])
def test_should_refuse_changed_frozen_two_head_readout(fault):
    reader = literal_reader()
    if fault == "state":
        reader.state = deepcopy(reader.state)
    elif fault == "spec":
        reader.spec = deepcopy(reader.spec)
    elif fault == "readers":
        reader._readers = deepcopy(reader._readers)
    else:
        object.__setattr__(reader._routing, "routing_rows", 99)
    with pytest.raises(ValueError, match="changed"):
        reader(DecisionRequest("r", "invoice policy_v1", CATALOG))
    assert reader.work.affine_attempts == 0


@pytest.mark.parametrize("method", ["linear_fit", "perceptron"])
def test_should_forward_separate_cli_and_keep_old_default(method, tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    path = tmp_path / "one.json"
    args = [
        "--context-return",
        "--separate-context-heads",
        "--method",
        method,
        "--output",
        str(path),
    ]
    assert entry.main(args) == 0
    assert calls == [
        dict(
            context_return=True,
            separate_context_heads=True,
            **(dict(method=method) if method == "linear_fit" else {}),
        )
    ]
    with pytest.raises(FileExistsError):
        entry.main(args)


@pytest.mark.parametrize(
    "options",
    [
        dict(replay="recent"),
        dict(private_adaptation=True),
        dict(paired_serving="invalid"),
        dict(record_train_fit=True),
        dict(class_balanced_bp=True),
    ],
)
def test_should_preserve_unsupported_separate_linear_modes_before_data(options, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data"))
    with pytest.raises(ValueError):
        entry.run_example(
            context_return=True, separate_context_heads=True, method="linear_fit", **options
        )


def test_should_execute_existing_context_stream_and_keep_inactive_checkpoint_heads(monkeypatch):
    holder, observed = {}, []
    stream_factory, fit = entry.AttributionStream, linear.fit_categorical_linear

    def capture(*args, **kwargs):
        stream = stream_factory(*args, **kwargs)
        holder["stream"] = stream
        return stream

    def arrived_fit(x, y, k):
        inbox = holder["stream"]._runtime._inbox
        ready = inbox._ready(inbox._last_time)
        assert len(ready) == 1
        source, label = inbox._experiences[ready[0]], inbox._labels[ready[0]]
        assert source.role == label.role == "train" and source.permissions.training
        assert source.observed_at + 1 == label.arrived_at <= inbox._last_time
        columns = tuple(entry.CONTEXT_FEATURES.index(tag) for tag in ("policy_v1", "policy_v2"))
        assert np.array_equal(x, np.delete(source.features, columns, axis=1))
        assert np.array_equal(y, label.targets)
        observed.append(len(x))
        return fit(x, y, k)

    monkeypatch.setattr(entry, "AttributionStream", capture)
    monkeypatch.setattr(linear, "fit_categorical_linear", arrived_fit)
    r = entry.run_example(context_return=True, separate_context_heads=True, method="linear_fit")
    assert r["complete"], r.get("error")
    assert observed == [6] * 12
    assert (
        r["training_work"]["returned_updates"] == 12 and r["training_work"]["attempted_rows"] == 72
    )
    assert (
        r["source_unchanged"]
        and r["actor_unchanged"]
        and r["numpy_rng_unchanged"]
        and r["python_rng_unchanged"]
    )
    points = r["checkpoints"]
    assert [p["coefficient_bytes"] for p in points] == [0, 384, 768, 768]
    assert points[1]["state"]["heads"][0] == points[2]["state"]["heads"][0]
    assert points[2]["state"]["heads"][1] == points[3]["state"]["heads"][1]
    assert (
        r["budget"]["limits"]["max_wall_seconds"] == 30 and r["budget"]["updates_completed"] == 12
    )
    assert r["stream"]["score_forward_calls"] == 0 and len(r["episodes"]) == 24
    assert all(e["episode"]["complete"] for e in r["episodes"])
    assert all(
        k == 0
        for k in (
            r["final_examples"],
            r["probe_updates"],
            r["replay_examples"],
            r["external_actions"],
        )
    )
    costs = r["costs"]
    work = costs["linear_readout_totals"]
    assert (
        costs["fitted_coefficient_bytes"] == 768
        and costs["retained_checkpoint_coefficient_bytes"] == 1920
    )
    assert work["routing_attempts"] == work["routing_returns"] == work["decision_returns"]
    assert work["nominal_affine_terms"] == work["rows_attempted"] * 48
    assert work["token_visits"] == 2 * work["routing_token_visits"]
    assert not costs["linear_readout_work_unknown"] and len(entry._encode_report(r)) <= 512 * 1024


def test_should_preserve_two_head_failed_fit_receipt_without_more_work(monkeypatch):
    original = linear.fit_categorical_linear
    calls = []

    def lost(*args):
        calls.append(True)
        original(*args)
        raise RuntimeError("lost two-head fit")

    monkeypatch.setattr(linear, "fit_categorical_linear", lost)
    r = entry.run_example(context_return=True, separate_context_heads=True, method="linear_fit")
    assert not r["complete"] and calls == [True]
    assert r["budget"]["updates_completed"] == 0
    assert (
        r["native_failure_work"]["candidate_commits"] == 0
        and r["native_failure_work"]["fit_rows_attempted"] == 6
    )
    assert r["native_failure_definition"] == CONTEXT_LINEAR_RESIDUAL_ID
    assert r["costs"]["physical_lstsq_calls"] is None
