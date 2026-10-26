"""Literal comparison glue and actual pure sandbox truth, no model construction."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_bp_ticket_serving as demo
from scripts import ticket_controller_comparison as study
from scripts.recurring_ticket_workflow_example import episode_slots
from scripts.ticket_controller_streams import POLICY_CHANGE_CHECKPOINTS
from src.app.ticket_workflow import run_workflow_episode
from src.core.ticket_workflow import ACTION_IDS, WorkflowCapabilities, WorkflowState
from test_ticket_workflow import decision

SOURCE = Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json")
FITTED = Path("artifacts/runs/learned-contextual-controller-20261023/actual.json")


def literal_trial(mode):
    """Prescribed metadata for procedure tests; not a native snapshot/learner."""
    n = dict(no_learning=0, periodic=6, drift=4, budget_aware=8, fitted=4)[mode]
    records = []
    for (tick, version), count in zip(
        POLICY_CHANGE_CHECKPOINTS, (0, min(n, 2), min(n, 4), n), strict=True
    ):
        records.append(
            dict(
                opportunity=tick,
                policy_version=version,
                applied_receipts=count,
                checkpoint=dict(
                    source_updates=count,
                    source_rows=576 + 6 * count,
                    state=dict(completed_updates=96 + count, training_examples=576 + 6 * count),
                ),
            )
        )
    return dict(
        definition="policy_change_native_reference_v1",
        complete=True,
        policy_change=True,
        deferred_evaluation=True,
        policy_capture_complete=True,
        policy_checkpoints=records,
        candidate_checkpoint=deepcopy(records[-1]["checkpoint"]),
        source_sha256=demo.context.SOURCE_SHA,
        source_unchanged=True,
        deferred_actor_unchanged=True,
        numpy_rng_unchanged=True,
        torch_rng_unchanged=True,
        deferred_promotions=0,
        episodes=[],
        final_examples=0,
        reserved_probe_updates=0,
        replay_examples=0,
        external_actions=0,
        training_work=dict(
            attempted_updates=n,
            returned_updates=n,
            attempted_rows=6 * n,
            unreturned_work_unknown=False,
        ),
        budget=dict(
            updates_completed=n,
            limits=asdict(
                demo.ToyExecutionBudget(
                    max_training_updates=12,
                    max_wall_seconds=120,
                    max_process_rss_bytes=512 * 1024**2,
                )
            ),
        ),
        applied_updates=[dict(update_number=i + 1) for i in range(n)],
        stream=dict(opportunities=13, arrived_labels=12, pending_unapplied=12 - n),
        controller_work=dict(prediction_rows=72, feedback_rows=72, selections_returned=13),
        action_snapshot=dict(
            envelope=dict(max_decisions=13, max_wake_updates=12),
            progress=dict(native_completions=n, unreturned_work_unknown=False),
        ),
        sharing_snapshot=dict(admitted_updates=n, limits=dict(max_admitted_updates=12)),
        costs=dict(sampled_process_rss=dict(peak_bytes=10000)),
        fixture_mode=mode,
    )


def literal_evaluation(mode="no_learning"):
    """Prescribed decisions run the real transition system; no learning claim."""
    counts = {
        r["opportunity"]: r["applied_receipts"] for r in literal_trial(mode)["policy_checkpoints"]
    }
    rows = []
    for point, version in POLICY_CHANGE_CHECKPOINTS:
        slots = tuple(("ordinary", c) for c in ("invoice", "parcel", "password")) + tuple(
            (s, c) for s, c in episode_slots(point) if s != "ordinary"
        )
        for scenario, concept in slots:
            correct = concept == "password" or version == "v2"
            if mode == "fitted" and point == 4 and concept == "invoice":
                correct = True
            if mode == "fitted" and point == 8 and concept == "password":
                correct = False
            target = (
                "route"
                if version == "v1" and concept != "password"
                else "defer"
                if concept == "parcel"
                else "inspect"
            )
            chosen = target if correct else "route" if target != "route" else "inspect"

            def predict(request):
                phase = request.query.split()[0]
                return decision(request, {"sealed": "open", "staged": "commit"}.get(phase, chosen))

            episode = run_workflow_episode(
                WorkflowState(f"literal-{point}-{concept}", concept, policy_version=version),
                predict,
                capabilities=WorkflowCapabilities(ACTION_IDS),
                scenario=scenario,
            )
            rows.append(
                dict(
                    opportunity=point,
                    source_updates=counts[point],
                    policy_version=version,
                    concept=concept,
                    role="outer_selection",
                    source_group="development-native-policy-workflow",
                    episode=episode,
                )
            )
    return dict(
        mode=mode,
        complete=True,
        episodes=rows,
        source_updates_by_opportunity=counts,
        execution_seconds=1.0,
        peak_rss_bytes=10000,
    )


class LiteralRng(list):
    def clone(self):
        return LiteralRng(self)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"literal-source")
    fit = tmp_path / "fit.json"
    fit.write_bytes(FITTED.read_bytes())
    state = SimpleNamespace(
        order=[], returned=[], source=source, fit=fit, train_hook=None, read_hook=None
    )
    monkeypatch.setattr(demo.context, "_source", lambda p: (p.read_bytes(), {}))
    monkeypatch.setattr(
        demo.native,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: LiteralRng([0]), equal=lambda a, b: a == b),
    )

    def run_trial(path, **kw):
        assert "read" not in state.order
        mode = "fitted" if "contextual_policy" in kw else kw["reference_mode"]
        assert kw["policy_change"] is True and kw["deferred_evaluation"] is True
        result = literal_trial(mode)
        state.order.append("train")
        state.returned.append(result)
        if state.train_hook:
            state.train_hook(result)
        return result

    def readout(spec, result, require):
        assert state.order.count("train") == 5
        state.order.append("read")
        require()
        result["policy_development_episodes"] = literal_evaluation(result["fixture_mode"])[
            "episodes"
        ]
        result["policy_readout_work"] = []
        result["policy_readout_complete"] = True
        if state.read_hook:
            state.read_hook(result)
        require()

    monkeypatch.setattr(demo, "run_case", run_trial)
    monkeypatch.setattr(demo.native, "policy_development_readout", readout)
    monkeypatch.setattr(demo, "_sealed_readout", lambda *a: pytest.fail("legacy final opened"))
    state.run = lambda: study.run_comparison(source, fit, policy_change=True)
    return state


def test_complete_cohort_freezes_before_development_and_uses_all_same_phase_outcomes(fixture):
    r = fixture.run()
    assert r["complete"], r.get("error")
    assert fixture.order == ["train"] * 5 + ["read"] * 5
    assert [c["mode"] for c in r["cases"]] == list(study.POLICY_MODES)
    assert r["models_frozen_at"] == 1 and r["clock"] == 7
    assert len(r["development_readouts"]) == 5
    assert all(c["result"].get("policy_development_episodes") is None for c in r["cases"])
    fitted = r["development_readouts"][-1]["phase_matched_measures"]
    assert (fitted["new_before"], fitted["new_after"], fitted["new_count"]) == (4, 5, 6)
    assert (fitted["old_before"], fitted["old_after"], fitted["old_count"]) == (3, 2, 3)
    u = r["development_readouts"][-1]["incremental_utility"]
    assert u["normalized_new_gain"] == pytest.approx(1 / 6) and u[
        "normalized_old_drop"
    ] == pytest.approx(1 / 3)
    assert all(
        e["all_in_utility"] is None and len(e["episodes"]) == 15 for e in r["development_readouts"]
    )
    assert r["historical_fit_costs"]["fit_training_updates"] == 16
    assert (
        r["new_controller_fits"]
        == r["promotions"]
        == r["final_examples"]
        == r["final_training_updates"]
        == 0
    )
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("source_unchanged", False),
        ("torch_rng_unchanged", None),
        ("deferred_actor_unchanged", False),
        ("policy_capture_complete", False),
        ("policy_summary", {}),
    ],
)
def test_failed_or_already_evaluated_trial_stops_before_second_candidate_and_any_readout(
    fixture, field, value
):
    fixture.train_hook = lambda r: r.update({field: value})
    r = fixture.run()
    assert not r["complete"] and len(r["cases"]) == 1 and r["development_readouts"] == []
    assert r["models_frozen_at"] is None and fixture.order == ["train"]


def test_replaced_prior_success_refuses_after_paid_second_trial_before_more_work(fixture):
    def replace_prior(result):
        if len(fixture.returned) == 2:
            fixture.returned[0]["stream"]["arrived_labels"] = 11

    fixture.train_hook = replace_prior
    r = fixture.run()
    assert not r["complete"] and len(r["cases"]) == 2 and r["development_readouts"] == []
    assert "cohort changed" in r["error"]


@pytest.mark.parametrize("owner", ["source", "fit"])
def test_changed_source_or_fit_bytes_refuse_before_another_trial(fixture, owner):
    fixture.train_hook = lambda r: getattr(fixture, owner).write_bytes(b"changed after paid return")
    r = fixture.run()
    assert not r["complete"] and len(r["cases"]) == 1 and r["development_readouts"] == []


def test_lost_first_readout_preserves_all_training_and_paid_development_prefix(fixture):
    def lose(result):
        result["policy_development_episodes"] = result["policy_development_episodes"][:1]
        result["policy_readout_complete"] = False
        raise RuntimeError("lost readonly return")

    fixture.read_hook = lose
    r = fixture.run()
    assert not r["complete"] and len(r["cases"]) == 5
    assert (
        len(r["development_readouts"]) == 1 and len(r["development_readouts"][0]["episodes"]) == 1
    )
    assert not r["development_readouts"][0]["complete"]
    assert "incremental_utility" not in r["development_readouts"][0]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "role", "version", "unknown", "coerced", "state", "clock"]
)
def test_reducer_refuses_incomplete_or_wrongly_bound_outcomes(fault):
    e = literal_evaluation()
    if fault == "missing":
        e["episodes"].pop()
    elif fault == "duplicate":
        e["episodes"][1] = deepcopy(e["episodes"][0])
    elif fault == "role":
        e["episodes"][0]["role"] = "final_test"
    elif fault == "version":
        e["episodes"][0]["policy_version"] = "v2"
    elif fault == "unknown":
        e["episodes"][0]["episode"]["unreturned_prediction_work_unknown"] = True
    elif fault == "coerced":
        e["episodes"][0]["episode"]["success"] = 0
    elif fault == "state":
        e["episodes"][0]["episode"]["success"] = not e["episodes"][0]["episode"]["success"]
    else:
        e["episodes"][0]["source_updates"] = 1
    with pytest.raises(ValueError):
        study._phase_matched_measures(e, literal_evaluation())


def test_cross_policy_improvement_in_no_learning_is_zero_matched_gain():
    e = literal_evaluation()
    assert e["episodes"][0]["episode"]["success"] is False
    assert (
        next(r for r in e["episodes"] if r["opportunity"] == 8 and r["concept"] == "invoice")[
            "episode"
        ]["success"]
        is True
    )
    m = study._phase_matched_measures(e, e)
    assert m.new_before == m.new_after == 4 and m.old_before == m.old_after == 3


def test_exact_frozen_fit_must_match_before_any_trial(fixture):
    data = json.loads(fixture.fit.read_bytes())
    data["qualification"] = "coherent metadata change"
    fixture.fit.write_text(json.dumps(data), encoding="utf8")
    r = fixture.run()
    assert not r["complete"] and r["cases"] == [] and fixture.order == []


def test_new_cli_requires_fit_preserves_default_and_exclusive_output(tmp_path, monkeypatch):
    calls = []

    def compare(source, fit, **kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_comparison", compare)
    output = tmp_path / "comparison.json"
    with pytest.raises(SystemExit):
        demo.main(
            ["--source", "x", "--output", str(output), "--policy-change-controller-comparison"]
        )
    assert not output.exists()
    args = [
        "--source",
        "source.json",
        "--fitted-report",
        "fit.json",
        "--output",
        str(output),
        "--policy-change-controller-comparison",
    ]
    assert demo.main(args) == 0 and calls == [dict(policy_change=True)]
    with pytest.raises(FileExistsError):
        demo.main(args)
    assert len(calls) == 1


def test_sole_frozen_native_policy_comparison():
    output = os.environ.get("PHASE_MATCHED_COMPARISON_REPORT")
    if not output:
        pytest.skip("separate five-case native study after gates")
    r = study.run_comparison(SOURCE, FITTED, policy_change=True)
    Path(output).write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf8")
    assert r["complete"], r.get("error")
    assert [c["mode"] for c in r["cases"]] == list(study.POLICY_MODES)
    assert len(r["development_readouts"]) == 5 and all(
        len(e["episodes"]) == 15 for e in r["development_readouts"]
    )
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert (
        r["new_controller_fits"]
        == r["promotions"]
        == r["final_examples"]
        == r["final_training_updates"]
        == 0
    )
    assert sum(c["result"]["training_work"]["attempted_updates"] for c in r["cases"]) <= 48
    assert all(e["all_in_utility"] is None for e in r["development_readouts"])
