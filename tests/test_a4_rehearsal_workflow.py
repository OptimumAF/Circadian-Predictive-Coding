"""Actual fresh-source/rehearsal composition, literal model and collection IO."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_a4_adaptation as app
from src.adapters import minigrid_affine_rehearsal as study
from src.adapters import minigrid_context_linear_learner as native
from src.core.reference_controller import ReferencePolicy
from src.core.learner_ports import TrainingUpdateFailed
from tests import test_a4_fresh_probe_workflow as whole
from tests import test_minigrid_affine_probe as fakes
from tests import test_short_affine_return as short
from tests import test_retained_a4_rehearsal as private
from tests.test_a4_adaptation import sources
from tests.test_retained_context_linear_games import state, budget

WORK = dict(collections=0, records=0)
NATIVE = short.NATIVE_OWNER


def train(owner, x, y):
    return short.train_bridge(owner, x, y) if len(x) <= 18 else private.literal_train(owner, x, y)


@pytest.fixture
def scripted(monkeypatch):
    box = fakes.scripted_collector(monkeypatch)
    original = study.original.reserved._collect_probe_sources

    def collect(*args, **kw):
        WORK["collections"] += 1
        try:
            return original(*args, **kw)
        finally:
            WORK["records"] += len(args[1])

    monkeypatch.setattr(study.original.reserved, "_collect_probe_sources", collect)
    monkeypatch.setattr(NATIVE, "train_batch", train)
    monkeypatch.setattr(NATIVE, "predict", short.predict_bridge)
    return box


def run(**options):
    return study.run_retained_rehearsal(sources(), state(), readout_budget=budget(), **options)


def harness(patch):
    h = whole.Harness(patch)
    patch.setattr(native, "ContextLinearLearner", NATIVE)
    patch.setattr(NATIVE, "train_batch", train)
    patch.setattr(NATIVE, "predict", short.predict_bridge)
    original = app.run_fresh_adaptation
    patch.setattr(app, "run_fresh_adaptation", lambda **kw: original(retained_rehearsal=True, **kw))
    return h


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = harness(patch)
        original_game, original_rehearsal = (
            app.evaluate_context_linear_stream_checkpoint,
            study.run_retained_rehearsal,
        )
        owners = []

        def game(*args, **kw):
            owners.append(kw["budget"])
            return original_game(*args, **kw)

        def compare(*args, **kw):
            assert (
                owners and all(o is owners[0] for o in owners) and kw["readout_budget"] is owners[0]
            )
            assert len(args[0]) == 4 and (args[1].completed_updates, args[1].training_examples) == (
                4,
                54,
            )
            return original_rehearsal(*args, **kw)

        patch.setattr(app, "evaluate_context_linear_stream_checkpoint", game)
        patch.setattr(study, "run_retained_rehearsal", compare)
        before = pickle.dumps((random.getstate(), np.random.get_state()))
        r = h.run(enabled=False)
        assert r["status"] == "complete", (r.get("reason"), r.get("capture_error"))
        assert pickle.dumps((random.getstate(), np.random.get_state())) == before
        return r


def test_should_collect_once_after_source_and_run_only_the_two_private_arms(integrated):
    r, q = integrated, integrated["rehearsal"]
    assert r["training"]["stream"]["completed_updates"] == 4
    assert r["training"]["stream"]["training_examples"] == 54
    assert r["training"]["stream"]["pending_unapplied"] == 2
    assert [g["updates"] for g in r["games"]] == [2, 4, 4]
    assert (
        r["rehearsal_work"]
        == q["comparison_work"]
        == dict(attempts=1, returned=1, unreturned_work_unknown=False)
    )
    assert len(q["collection"]) == 4 and q["source_native_state_unchanged"] is True
    assert q["probe_sources"]["released_train_packets"] == 16
    assert q["probe_sources"]["released_test_packets"] == 2
    assert [a["arm"] for a in q["comparison"]["arms"]] == list(study.RETAINED_ARMS)
    for arm in q["comparison"]["arms"]:
        assert arm["probe"]["completed_training_examples"] == 54
        assert arm["native_endpoint"]["completed_updates"] == 6
        assert arm["native_endpoint"]["training_examples"] == 108
        assert arm["source_native_state_unchanged"] and arm["source_fork_calls"] == 1
    assert "probe" not in r and "native_games" not in q


def test_should_preserve_whole_owner_starts_and_separate_reuse_and_preparation_costs(integrated):
    r, q = integrated, integrated["rehearsal"]
    b = r["readout_budget"]
    assert b["initial_limits"] == b["limits"] and b["started_at"] == b["observed_started_at"]
    assert b["updates"] == b["replay"] == 0
    study_budget = q["comparison"]["budget"]
    assert (study_budget["updates_completed"], study_budget["replay_examples_completed"]) == (4, 36)
    assert b["started_at"] <= r["training"]["budget"]["started_at"] <= study_budget["started_at"]
    assert all(
        a["budget"]["started_at"] == study_budget["started_at"] for a in q["comparison"]["arms"]
    )
    assert q["preparation_array_bytes"] == q["validation_array_bytes"] == 36 * 297 * 8
    assert r["summary"]["learning_recovery"] == [None, None]


@pytest.mark.parametrize(
    "options",
    [
        dict(retained_rehearsal=None),
        dict(retained_rehearsal=1),
        dict(retained_rehearsal=[]),
        dict(retained_rehearsal=True, retained_probe=True),
        dict(retained_rehearsal=True, return_old=True),
        dict(retained_rehearsal=True, native_games=True, retained_probe=True),
        dict(retained_rehearsal=True, controller_mode="retain_arrived"),
        dict(retained_rehearsal=True, reference_policy=ReferencePolicy("periodic")),
    ],
)
def test_should_refuse_wrong_or_conflicting_recipes_before_work(monkeypatch, options):
    monkeypatch.setattr(app, "_run_adaptation", lambda *a, **k: pytest.fail("work reached"))
    kw: dict[str, Any] = dict(
        reference_policy=ReferencePolicy("drift"),
        **{k: v for k, v in options.items() if k != "reference_policy"},
    )
    if "reference_policy" in options:
        kw["reference_policy"] = options["reference_policy"]
    with pytest.raises(ValueError, match="retained_rehearsal"):
        app.run_fresh_adaptation(**kw)


def test_should_describe_without_collection_or_model_work(monkeypatch):
    monkeypatch.setattr(
        study.original.reserved,
        "_collect_probe_sources",
        lambda *a, **k: pytest.fail("collection reached"),
    )
    r = app.describe_fresh_adaptation(
        reference_policy=ReferencePolicy("drift"), retained_rehearsal=True
    )
    assert (
        r["retained_rehearsal"]["total_new_fit_rows"] == 108 and r["historical_source_reads"] == 0
    )
    assert "no extraFwarmup" in r["rehearsal_scope"]
    assert app.describe_fresh_adaptation(
        reference_policy=ReferencePolicy("drift")
    ) == app.describe_fresh_adaptation(
        reference_policy=ReferencePolicy("drift"), retained_rehearsal=False
    )


@pytest.mark.parametrize("fault", ["source", "limit", "counter"])
def test_should_refuse_wrong_source_or_owner_before_f_collection(monkeypatch, fault):
    b, s = budget(), state()
    if fault == "source":
        s = replace(s, completed_updates=5)
    elif fault == "limit":
        object.__setattr__(b.budget, "max_wall_seconds", 151)
    else:
        b.replay_examples_completed = False
    monkeypatch.setattr(
        study.original.reserved,
        "_collect_probe_sources",
        lambda *a, **k: pytest.fail("collection reached"),
    )
    with pytest.raises(ValueError):
        study.run_retained_rehearsal(sources(), s, readout_budget=b)


@pytest.mark.parametrize("fault", ["count", "tail_role", "test_role"])
def test_should_validate_the_full_release_before_selected_packets_or_comparison(
    scripted, monkeypatch, fault
):
    original = study.original.reserved._collect_probe_sources

    def collect(*args, **kw):
        training, tests = original(*args, **kw)
        if fault == "count":
            training = training[:2]
        elif fault == "tail_role":
            training = (*training[:-1], replace(training[-1], role="final_test"))
        else:
            tests = (replace(tests[0], role="train"), tests[1])
        return training, tests

    monkeypatch.setattr(study.original.reserved, "_collect_probe_sources", collect)
    monkeypatch.setattr(
        study, "compare_private_rehearsal", lambda *a, **k: pytest.fail("comparison reached")
    )
    r = run()
    assert r["status"] == "failed" and r["comparison_work"]["attempts"] == 0
    assert len(r["collection"]) == 4


def test_should_keep_partial_closed_collection_without_starting_comparison(scripted, monkeypatch):
    def collect(guard, reports, **kw):
        WORK["collections"] += 1
        reports.append(dict(status="complete", closed=True))
        WORK["records"] += 1
        guard()
        reports.append(dict(status="failed", closed=True, primary_error="lostteacher"))
        WORK["records"] += 1
        raise RuntimeError("lost teacher return")

    monkeypatch.setattr(study.original.reserved, "_collect_probe_sources", collect)
    r = run()
    assert r["status"] == "failed" and len(r["collection"]) == 2
    assert r["collection"][0]["status"] == "complete" and r["comparison_work"]["attempts"] == 0


def test_should_preserve_unknown_comparison_work_when_its_return_is_lost(scripted, monkeypatch):
    def lost(*args, **kw):
        kw["check_boundary"]()
        raise RuntimeError("lostcomparison")

    monkeypatch.setattr(study, "compare_private_rehearsal", lost)
    r = run()
    assert r["status"] == "failed" and r["comparison"] is None
    assert r["comparison_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)


def test_should_keep_successful_first_arm_and_partial_joint_fit_costs(scripted, monkeypatch):
    calls = [0]

    def fail(owner, x, y):
        calls[0] += 1
        if calls[0] == 3:
            owner._state = replace(owner.snapshot_state(), failed_work=(1, 27, 0, 0, True))
            raise TrainingUpdateFailed("lostjointfit", "literal", dict(fit_calls_attempted=1))
        return train(owner, x, y)

    monkeypatch.setattr(NATIVE, "train_batch", fail)
    r = run()
    c = r["comparison"]
    assert r["status"] == c["status"] == "failed" and calls[0] == 3
    assert c["arms"][0]["status"] == "complete" and c["budget"]["updates_completed"] == 2
    assert c["arms"][1]["attempts"][0]["old_TRAIN_reuse_charged"] == 18
    assert r["comparison_work"] == dict(attempts=1, returned=1, unreturned_work_unknown=False)


@pytest.mark.parametrize("fault", ["value", "replacement"])
def test_should_refuse_changed_prior_collection_before_fitting(scripted, monkeypatch, fault):
    original = study.compare_private_rehearsal

    def corrupt(*args, **kw):
        reports = scripted["reports"]
        if fault == "value":
            reports[0]["status"] = "tampered"
        else:
            reports[0] = deepcopy(reports[0])
        return original(*args, **kw)

    monkeypatch.setattr(study, "compare_private_rehearsal", corrupt)
    r = run()
    assert r["status"] == "failed" and r["comparison"]["budget"]["updates_completed"] == 0
    assert r["collection"][0]["status"] == "complete"


def test_should_refuse_late_readout_clock_replacement_before_more_work(scripted):
    b = budget()
    original = b.before_final

    def observe():
        b.clock = lambda: 1.0
        return original()

    b.before_final = observe
    r = study.run_retained_rehearsal(sources(), state(), readout_budget=b)
    assert r["status"] == "failed" and r["collection"] == []


def test_should_preserve_original_source_when_outer_rehearsal_return_is_lost(monkeypatch):
    h = harness(monkeypatch)

    def lost(*args, **kw):
        kw["check_boundary"]()
        raise RuntimeError("lostrehearsal")

    monkeypatch.setattr(study, "run_retained_rehearsal", lost)
    r = h.run(enabled=False)
    assert r["status"] == "failed" and len(r["games"]) == 3 and r["rehearsal"] is None
    assert r["rehearsal_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)


def test_should_refuse_corrupted_paid_rehearsal_at_outer_capture(monkeypatch):
    h = harness(monkeypatch)
    original = app.paid_report_guard

    def corrupt(value):
        result = original(value)
        if "comparison_work" in value:
            value["status"] = "tampered"
        return result

    monkeypatch.setattr(app, "paid_report_guard", corrupt)
    r = h.run(enabled=False)
    assert r["status"] == "failed" and r["rehearsal"]["status"] == "complete"
    assert r["rehearsal_work"]["returned"] == 1


def test_should_recheck_paid_rehearsal_after_original_resource_clock(monkeypatch):
    h = harness(monkeypatch)
    original_guard, original_clock = app.paid_report_guard, app.monotonic
    box: dict[str, Any] = dict(calls=0, owner=None, changed=False)

    def seal(value):
        proof, detached = original_guard(value)
        if "comparison_work" not in value:
            return proof, detached
        box["owner"] = value

        def check():
            proof()
            box["calls"] += 1

        return check, detached

    def clock():
        if box["calls"] == 3 and not box["changed"]:
            box["changed"] = True
            box["owner"]["status"] = "lateclocktamper"
        return original_clock()

    monkeypatch.setattr(app, "paid_report_guard", seal)
    monkeypatch.setattr(app, "monotonic", clock)
    r = h.run(enabled=False)
    assert box["changed"] and r["status"] == "failed", (
        box["calls"],
        box["changed"],
        r["status"],
        r.get("reason"),
        r.get("capture_error"),
    )
    assert r["rehearsal"]["status"] == "complete"
