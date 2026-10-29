"""Existing fitter entry and six paid catalog trials then original-source application."""

import pytest

from scripts import context_catalog_learning as learning
from scripts import context_controller_learning as entry
from scripts import context_head_live as live
from src.core.reference_controller import ArrivedTrainingFeedback
from test_context_catalog_control import source as source


@pytest.mark.parametrize(
    "batch, mistakes, allowed",
    [
        ("new-v1", 1, True),
        ("old-v1", 0, True),
        ("new-v1", 0, False),
        ("old-v1", 1, False),
        ("other", 1, False),
    ],
)
def test_should_admit_only_the_declared_observed_train_context(batch, mistakes, allowed):
    feedback = ArrivedTrainingFeedback("train", batch, "qualified-12", 0, 1, mistakes, 3)
    if allowed:
        learning.require_observed_context(batch, feedback)
    else:
        with pytest.raises(ValueError, match="no substituted trial"):
            learning.require_observed_context(batch, feedback)


def test_should_fit_all_six_catalog_estimates_before_one_frozen_application(source):
    r = entry.run_study("unused", catalog=True)
    assert r["complete"], r.get("error")
    assert len(r["trials"]) == 6 and r["application"]["complete"]
    assert len(r["fitted_policy"]["estimates"]) == 6
    assert {row["request"]["context"] for row in r["trials"]} == {"mistakes", "clean"}
    assert r["fitted_policy"]["fitted_at"] == 86
    assert r["application"]["actor_unchanged"]
    assert r["application"]["catalog_control"]["record"] == r["fitted_policy"]


def test_should_stop_without_fit_or_application_after_a_failed_paid_trial(source, monkeypatch):
    calls = []

    def failed(*args, **kwargs):
        calls.append(kwargs)
        return dict(complete=False, error="known partial work")

    monkeypatch.setattr(live, "run_case", failed)
    r = learning.run_catalog_study("unused")
    assert not r["complete"] and len(calls) == len(r["trials"]) == 1
    assert "fitted_policy" not in r and r["application"] is None


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_should_refuse_non_boolean_new_entry_flag_before_io(monkeypatch, flag):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        entry.run_study("unused", catalog=flag)


def test_should_refuse_old_metric_option_with_new_frozen_catalog_protocol():
    with pytest.raises(ValueError):
        entry.run_study("unused", catalog=True, task_outcomes=True)


def test_should_restore_corrupted_paid_trial_and_refuse_following_case(source, monkeypatch):
    original, calls = learning.catalog_measures, []

    def changed(result):
        measured = original(result)
        calls.append(result)
        result["source_unchanged"] = False
        return measured

    monkeypatch.setattr(learning, "catalog_measures", changed)
    r = learning.run_catalog_study("unused")
    assert not r["complete"] and "preceding catalog trial corrupted" in r["error"]
    assert len(calls) == len(r["trials"]) == 1 and r["application"] is None
    assert r["trials"][0]["result"]["source_unchanged"] is True
    assert "fitted_policy" not in r
