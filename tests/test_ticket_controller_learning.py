"""Small native fit/application, actual forks/admission, no final source."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_controller_learning as learning
from src.core.contextual_controller import ContextualChoice, ContextualEstimate, FittedController


@pytest.fixture(scope="module")
def completed():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo.context, "BATCHES_PER_PHASE", 2)
        templates = demo.context._teacher_batches(
            demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
        )
        patch.setattr(demo.context, "_teacher_batches", lambda s: deepcopy(templates))
        patch.setattr(
            demo, "_sealed_cases", lambda: pytest.fail("opened final source during controller fit")
        )
        result = learning.run_study(
            Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json")
        )
    if os.environ.get("LEARNED_CONTROLLER_FIXTURE_REPORT"):
        Path(os.environ["LEARNED_CONTROLLER_FIXTURE_REPORT"]).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    assert result["complete"], result.get("error")
    return result


def test_native_fit_uses_four_disposable_source_bound_trials_and_closes_before_application(
    completed,
):
    r = completed
    assert r["declared_at"] == 0 and r["fitted_at"] == 10
    assert r["source_unchanged"] and r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["final_source_reads"] == r["probe_updates"] == 0
    assert (
        r["training_state"]["attempts"]
        == r["training_state"]["completed"]
        == r["training_state"]["discarded_trials"]
        == 4
    )
    assert not r["training_state"]["stopped"]
    assert [(t["request"]["context"], t["request"]["action_id"]) for t in r["trials"]] == [
        ("mistakes", "wait"),
        ("mistakes", "wake1"),
        ("clean", "wait"),
        ("clean", "wake1"),
    ]
    assert len({t["request"]["state_digest"] for t in r["trials"]}) == 1
    for trial in r["trials"]:
        c = trial["result"]
        updates = 0 if trial["request"]["action_id"] == "wait" else 2
        assert c["counterfactual_actor_unchanged"] and c["counterfactual_promotions"] == 0
        assert c["source_unchanged"] and c["complete"]
        assert "promotion_receipt" not in c and "guard_report" not in c
        assert c["training_work"] == dict(
            attempted_updates=updates,
            attempted_rows=6 * updates,
            returned_updates=updates,
            unreturned_work_unknown=False,
        )
        assert c["candidate_checkpoint"]["state"]["completed_updates"] == 96 + updates
        assert c["candidate_checkpoint"]["state"]["training_examples"] == 576 + 6 * updates
        assert c["final_examples"] == c["reserved_probe_updates"] == c["replay_examples"] == 0
        assert trial["measures"]["work_complete"] and trial["measures"]["execution_seconds"] > 0


def test_frozen_applications_follow_empirical_choices_and_use_original_causal_admission(completed):
    r = completed
    p = r["fitted_policy"]
    assert p == r["training_state"]["fitted"]
    assert len(p["estimates"]) == 4 and p["fitted_at"] == 10
    assert len(r["applications"]) == 2
    for kind, result in r["applications"].items():
        assert result["contextual_policy"] == p and not result["counterfactual"]
        assert result["stream"]["opportunities"] == 3
        assert result["controller_work"]["feedback_rows"] == 12
        assert result["controller_events"][0]["action_id"] == "wait"
        context = "mistakes" if kind == "new_contexts" else "clean"
        assert all(e["action_id"] == p["choices"][context] for e in result["controller_events"][1:])
        assert all(
            u["observed_at"] < u["arrived_at"] <= u["applied_at"] for u in result["applied_updates"]
        )
        assert (
            result["source_unchanged"]
            and result["numpy_rng_unchanged"]
            and result["torch_rng_unchanged"]
        )
        assert "guard_report" in result
    assert r["total_training_work"]["attempted_updates"] <= 8
    assert sum(t["result"]["training_work"]["attempted_updates"] for t in r["trials"]) == 4


def test_wrong_native_fitted_source_digest_is_refused_before_any_update(completed):
    p = completed["fitted_policy"]
    fitted = FittedController(
        p["learner_version"],
        "replaced-payload",
        demo.ControllerObjective(),
        p["fitted_at"],
        tuple(ContextualEstimate(**e) for e in p["estimates"]),
        ContextualChoice(**p["choices"]),
    )
    r = demo.run_case(
        Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"),
        contextual_policy=fitted,
    )
    assert not r["complete"] and "original learner version/payload" in r["error"]
    assert r["training_work"]["attempted_updates"] == 0


def test_unknown_native_trial_stops_before_fit_or_more_trial_sources(monkeypatch):
    calls = []

    def failed(*args, **kwargs):
        calls.append(kwargs["counterfactual"])
        return dict(complete=False, training_work=dict(unreturned_work_unknown=True))

    monkeypatch.setattr(demo, "run_case", failed)
    r = learning.run_study(Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json"))
    assert not r["complete"] and calls == [True]
    assert r["training_state"]["attempts"] == 1 and r["training_state"]["stopped"]
    assert "fitted_policy" not in r and r["applications"] == {}
    assert r["trials"][0]["result"]["training_work"]["unreturned_work_unknown"]


def test_cli_is_optin_exclusive_and_preserves_failed_result(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "run_study", lambda p: dict(complete=False))
    output = tmp_path / "learned.json"
    assert demo.main(["--source", "unused", "--learned-controller", "--output", str(output)]) == 1
    assert json.loads(output.read_bytes()) == dict(complete=False)
    with pytest.raises(FileExistsError):
        demo.main(["--source", "unused", "--learned-controller", "--output", str(output)])
    for option in (["--outer-study"], ["--reference", "periodic"]):
        with pytest.raises(SystemExit):
            demo.main(
                [
                    "--source",
                    "unused",
                    "--learned-controller",
                    *option,
                    "--output",
                    str(tmp_path / "invalid"),
                ]
            )
