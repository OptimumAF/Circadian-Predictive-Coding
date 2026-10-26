"""Complete candidate cohort precedes final reads; frozen policies never refit."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_controller_comparison as comparison

SOURCE = Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json")
FITTED = Path("artifacts/runs/learned-contextual-controller-20261023/actual.json")


@pytest.fixture(scope="module")
def completed():
    from src.adapters.torch_feature_learner import TorchFeatureLearner

    order: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo.context, "BATCHES_PER_PHASE", 2)
        templates = demo.context._teacher_batches(
            demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
        )
        patch.setattr(demo.context, "_teacher_batches", lambda s: deepcopy(templates))
        patch.setattr(
            comparison,
            "WITHHELD_RECIPES",
            tuple(replace(r, contexts=r.contexts[:2]) for r in comparison.WITHHELD_RECIPES),
        )
        run, final = demo.run_case, demo._sealed_readout

        def training(*args, **kwargs):
            assert not any(k == "final" for k in order)
            result = run(*args, **kwargs)
            order.append("train")
            return result

        def readonly(*args):
            assert order.count("train") == 10
            order.append("final")
            with pytest.MonkeyPatch.context() as sealed:
                sealed.setattr(
                    TorchFeatureLearner,
                    "train_batch",
                    lambda *a: pytest.fail("TRAIN after final source opens"),
                )
                sealed.setattr(
                    demo,
                    "choose_contextual",
                    lambda *a: pytest.fail("policy selection during final readout"),
                )
                sealed.setattr(
                    demo,
                    "choose_reference",
                    lambda *a: pytest.fail("reference selection during final readout"),
                )
                return final(*args)

        patch.setattr(demo, "run_case", training)
        patch.setattr(demo, "_sealed_readout", readonly)
        r = comparison.run_comparison(SOURCE, FITTED)
    if os.environ.get("CONTROLLER_COMPARISON_FIXTURE_REPORT"):
        Path(os.environ["CONTROLLER_COMPARISON_FIXTURE_REPORT"]).write_text(
            json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    assert r["complete"], r.get("error")
    assert order == ["train"] * 10 + ["final"] * 10
    return r


def test_all_candidates_finish_before_sealed_final_reads_and_keep_original_owners(completed):
    r = completed
    assert r["protocol"]["declared_at"] == 0 and r["models_frozen_at"] == 1
    assert len(r["cases"]) == len(r["final_readouts"]) == 10
    assert r["final_training_updates"] == r["promotions"] == r["new_controller_fits"] == 0
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    for case in r["cases"]:
        c = case["result"]
        assert c["complete"] and c["deferred_actor_unchanged"] and c["deferred_promotions"] == 0
        assert c["source_unchanged"] and c["numpy_rng_unchanged"] and c["torch_rng_unchanged"]
        assert c["episodes"] == [] and "candidate_episodes" not in c and "guard_report" not in c
        assert c["serving_work"] == dict(attempts=0, returns=0, readouts=0, native_forwards=0)
        updates = c["training_work"]["attempted_updates"]
        assert c["candidate_checkpoint"]["state"]["completed_updates"] == 96 + updates
        assert c["candidate_checkpoint"]["state"]["training_examples"] == 576 + 12 * updates
        assert c["training_work"]["attempted_rows"] == 12 * updates
        assert c["final_examples"] == c["reserved_probe_updates"] == c["replay_examples"] == 0
        assert c["training_stream"]["row_counts"] == [12, 12]
        assert [u["sample_id"] for u in c["applied_updates"]] == [
            f"batch-{i}" for i in range(updates)
        ]
    assert r["historical_fit_costs"]["fit_training_updates"] == 16
    assert r["historical_fit_costs"]["fit_training_rows"] == 96
    assert r["historical_fit_costs"]["all_in_utility"] is None


def test_delayed_receipts_have_fair_opportunities_and_no_unarrived_ready_work(completed):
    for recipe in completed["protocol"]["recipes"]:
        cases = [c["result"] for c in completed["cases"] if c["recipe_id"] == recipe["recipe_id"]]
        delay = recipe["label_delay"]
        assert len(cases) == 5
        for c in cases:
            assert c["stream"]["opportunities"] == 2 + delay
            assert c["stream"]["arrived_labels"] == 2
            assert (
                c["controller_work"]["feedback_rows"]
                == c["controller_work"]["prediction_rows"]
                == 24
            )
            assert c["controller_work"]["selections_returned"] == 2 + delay
            assert c["controller_work"]["prediction_records_remaining"] == 0
            for tick, event in enumerate(c["controller_events"]):
                o = event["observation"]
                if tick < delay:
                    assert o["feedback"] is None and o["ready_updates"] == 0
                    assert event["action_id"] == "wait"
                else:
                    f = o["feedback"]
                    assert f["role"] == "train" and f["observed_at"] == tick - delay
                    assert f["arrived_at"] == tick and f["examples"] == 12
                    assert f["actor_version"] == o["actor_version"] == "qualified-96"
            assert all(
                u["arrived_at"] - u["observed_at"] == delay and u["arrived_at"] <= u["applied_at"]
                for u in c["applied_updates"]
            )


def test_final_readouts_are_readonly_and_incremental_utility_never_imputes_all_in_cost(completed):
    r = completed
    assert r["clock"] == 21
    for i, final in enumerate(r["final_readouts"]):
        assert final["sealed_source_reads"] == [dict(at=2 + 2 * i, role="sealed_evaluation")]
        assert len(final["sealed_episodes"]) == 8
        assert all(
            e["role"] == "final_test" and e["episode"]["complete"] for e in final["sealed_episodes"]
        )
        assert all(
            w["decision_attempts"] == w["decision_returns"] == w["native_forward_returns"] == 12
            for w in final["sealed_readout_work"]
        )
        u = final["incremental_utility"]
        assert u["normalized_compute"] > 0 and u["normalized_memory"] > 0
        assert u["value"] == pytest.approx(
            u["normalized_new_gain"]
            - u["normalized_old_drop"]
            - u["normalized_compute"]
            - u["normalized_memory"]
            - u["normalized_violations"]
        )
        assert final["all_in_utility"] is None


def test_failed_preceding_work_stops_other_candidates_and_never_opens_final(monkeypatch):
    calls = []

    def failed(*a, **kw):
        calls.append(kw["stream_recipe"].recipe_id)
        return dict(complete=False, training_work=dict(unreturned_work_unknown=True))

    monkeypatch.setattr(demo, "run_case", failed)
    monkeypatch.setattr(
        demo, "_sealed_readout", lambda *a: pytest.fail("read final after failed work")
    )
    r = comparison.run_comparison(SOURCE, FITTED)
    assert not r["complete"] and len(calls) == len(r["cases"]) == 1
    assert r["models_frozen_at"] is None and r["final_readouts"] == []
    assert r["cases"][0]["result"]["training_work"]["unreturned_work_unknown"]


def test_replaced_fit_is_refused_before_any_native_candidate(monkeypatch, tmp_path):
    _, policy, _ = comparison._frozen_fit(FITTED)
    bad = json.loads(FITTED.read_bytes())
    bad["fitted_policy"]["choices"]["clean"] = "wake1"
    path = tmp_path / "replaced.json"
    path.write_text(json.dumps(bad), encoding="utf8")
    monkeypatch.setattr(
        demo, "run_case", lambda *a, **kw: pytest.fail("native work before valid fit")
    )
    r = comparison.run_comparison(SOURCE, path)
    assert not r["complete"] and r["cases"] == r["final_readouts"] == []
    assert policy.choices.clean == "wait"


def test_cli_requires_explicit_fit_and_protects_output_without_changing_other_modes(
    tmp_path, monkeypatch
):
    output = tmp_path / "comparison.json"
    with pytest.raises(SystemExit):
        demo.main(["--source", "unused", "--controller-comparison", "--output", str(output)])
    assert not output.exists()
    with pytest.raises(SystemExit):
        demo.main(["--source", "unused", "--fitted-report", "unused", "--output", str(output)])
    monkeypatch.setattr(comparison, "run_comparison", lambda *a: dict(complete=False))
    assert (
        demo.main(
            [
                "--source",
                "unused",
                "--controller-comparison",
                "--fitted-report",
                "unused",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    with pytest.raises(FileExistsError):
        demo.main(
            [
                "--source",
                "unused",
                "--controller-comparison",
                "--fitted-report",
                "unused",
                "--output",
                str(output),
            ]
        )
