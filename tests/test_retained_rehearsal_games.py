"""Fixed native-readout composition; literal games aren't scientific outcomes."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import pytest

from scripts import minigrid_a4_adaptation as app
from src.adapters import minigrid_affine_rehearsal as study
from src.adapters import minigrid_policy_evaluation as evaluation
from tests import test_a4_rehearsal_workflow as workflow
from tests import test_affine_private_rehearsal as old
from tests.test_retained_context_linear_games import budget, state
from test_context_linear_causal_games import PhaseEnvironment
from src.infra.minigrid_memory import _text
from src.core.minigrid_key_door import MiniGridKeyDoorLayout

WORK = dict(readouts=0, failed_steps=0, lost_readouts=0)
NATIVE = workflow.NATIVE


class FixedEnvironment(PhaseEnvironment):
    def __init__(self, *args, **kw):
        # Keep the real pre-physics boundary: uniqueness alone missed oversized IDs.
        _text(kw["episode_id"], "episode_id", 64)
        super().__init__(*args, **kw)

    @property
    def report(self):
        return dict(
            super().report,
            episode_id=self.episode_id,
            role=self.role,
            cue=self.cue,
            seed=self.seed,
            max_steps=self.max_steps,
            layout=asdict(self.layout),
        )


def endpoint():
    s = state()
    return replace(
        s,
        completed_updates=6,
        training_examples=108,
        heads=tuple(replace(h, fit_rows=27, singular_values=(0.0,) * 27) for h in s.heads),
    )


@pytest.fixture
def game_io(monkeypatch):
    monkeypatch.setattr(PhaseEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", FixedEnvironment)
    monkeypatch.setattr(NATIVE, "predict", workflow.short.predict_bridge)
    original = evaluation.evaluate_context_linear_rehearsal_checkpoint

    def read(*args, **kw):
        WORK["readouts"] += 1
        return original(*args, **kw)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", read)


@pytest.fixture
def scripted(monkeypatch, game_io):
    setup: Any = workflow.scripted
    setup.__wrapped__(monkeypatch)


def run():
    return study.run_retained_rehearsal(
        workflow.sources(), state(), readout_budget=budget(), native_games=True
    )


@pytest.mark.parametrize("arm", study.RETAINED_ARMS)
def test_should_bind_exact_retained_recipe_and_six_roles_before_shared_loop(monkeypatch, arm):
    def loop(owner, saved, **kw):
        assert (saved.completed_updates, saved.training_examples) == (6, 108)
        assert kw["return_metadata"]["source_recipe"] == "retained_a4_private_rehearsal"
        assert arm in kw["role"]
        assert [(l.context, l.branch_row, c, r) for l, c, r in kw["live_plan"]] == [
            ("F-blue", 2, "blue", "probe_test"),
            ("F-red", 2, "red", "probe_test"),
            ("A", 3, "blue", "train"),
            ("A", 3, "red", "train"),
            ("A", 4, "blue", "train"),
            ("A", 4, "red", "train"),
        ]
        kw["extra_check"]()
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", loop)
    r = evaluation.evaluate_context_linear_rehearsal_checkpoint(
        NATIVE(),
        endpoint(),
        arm=arm,
        budget=budget(),
        check_boundary=lambda: None,
        retained_a4=True,
    )
    assert r["status"] == "complete"


@pytest.mark.parametrize("fault", ["flag", "arm", "rows", "heads", "observer", "phase"])
def test_should_refuse_foreign_recipe_before_readout(monkeypatch, fault):
    s = endpoint()
    kw: dict[str, Any] = dict(arm="current_only", retained_a4=True)
    if fault == "flag":
        kw["retained_a4"] = 1
    elif fault == "arm":
        kw["arm"] = "joint_ABC_F"
    elif fault == "rows":
        s = replace(s, training_examples=109)
    elif fault == "heads":
        s = state()
    elif fault == "observer":
        kw["qualified_boundary"] = lambda *a, **k: None
    else:
        kw["short_return_phase"] = "after_probe"
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            NATIVE(), s, budget=budget(), check_boundary=lambda: None, **kw
        )


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        setup: Any = game_io
        setup.__wrapped__(patch)
        h = workflow.harness(patch)
        original = app.run_fresh_adaptation
        patch.setattr(app, "run_fresh_adaptation", lambda **kw: original(native_games=True, **kw))
        original_read = evaluation.evaluate_context_linear_rehearsal_checkpoint

        def read(*args, **kw):
            compared = old.closed_value(kw["check_boundary"], "comparison_owner")
            assert compared["status"] == "complete" and len(compared["arms"]) == 2
            assert compared["budget"]["updates_completed"] == 4
            assert all(a["probe"]["completed_updates"] == 2 for a in compared["arms"])
            return original_read(*args, **kw)

        patch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", read)
        r = h.run(enabled=False)
        assert r["status"] == "complete", (r.get("reason"), r.get("capture_error"))
        return r


def test_should_read_both_arms_only_after_all_fitting_on_the_original_owner(integrated):
    r = integrated
    q = r["rehearsal"]
    assert [x["arm"] for x in q["native_games"]] == list(study.RETAINED_ARMS)
    assert q["native_game_work"] == dict(attempts=2, returned=2, unreturned_work_unknown=False)
    ids: list[str] = []
    for row in q["native_games"]:
        g = row["games"]
        assert g["status"] == "complete" and g["state_unchanged"] is True
        assert len(g["episodes"]) == 6 and all(e["environment"]["closed"] for e in g["episodes"])
        ids.extend(e["environment"]["episode_id"] for e in g["episodes"])
    assert len(set(ids)) == 12
    assert r["readout_budget"]["updates"] == r["readout_budget"]["replay"] == 0
    assert q["comparison"]["budget"]["updates_completed"] == 4


def test_should_fit_all_twelve_distinct_retained_ids_inside_the_real_boundary():
    ids = []
    for arm in study.RETAINED_ARMS:
        for context, branch, cue in (
            ("F-blue", 2, "blue"),
            ("F-red", 2, "red"),
            ("A", 3, "blue"),
            ("A", 3, "red"),
            ("A", 4, "blue"),
            ("A", 4, "red"),
        ):
            identity = evaluation._live_episode_id(
                "arrived_public_cue_affine_fit",
                6,
                MiniGridKeyDoorLayout(context=context, branch_row=branch),
                cue,
                f"private_retained_{arm}",
            )
            _text(identity, "episode_id", 64)
            ids.append(identity)
    assert len(set(ids)) == 12
    # Accepted older phase IDs keep their original format.
    assert (
        evaluation._live_episode_id(
            "arrived_public_cue_affine_fit",
            6,
            MiniGridKeyDoorLayout(context="A", branch_row=3),
            "blue",
            "private_F_then_old_A3_A4",
        )
        == "live-arrived_public_cue_affine_fit-6-A-blue"
    )


@pytest.mark.parametrize("index", [1, 2])
def test_should_keep_closed_partial_games_and_both_paid_arms_without_retry(
    scripted, monkeypatch, index
):
    original = PhaseEnvironment.step
    start = WORK["readouts"]

    def step(owner, action):
        if WORK["readouts"] - start == index and owner.cue == "red":
            WORK["failed_steps"] += 1
            raise RuntimeError("lost literal game step")
        return original(owner, action)

    monkeypatch.setattr(PhaseEnvironment, "step", step)
    r = run()
    assert r["status"] == "failed" and r["comparison"]["budget"]["updates_completed"] == 4
    assert len(r["native_games"]) == index
    assert r["native_game_work"] == dict(
        attempts=index, returned=index - 1, unreturned_work_unknown=True
    )
    assert [e["status"] for e in r["native_games"][-1]["games"]["episodes"]] == [
        "complete",
        "failed",
    ] + ["not_run"] * 4


@pytest.mark.parametrize("fault", ["value", "replacement", "clock"])
def test_should_refuse_changed_prior_game_or_owner_before_second_read(scripted, monkeypatch, fault):
    original = evaluation.evaluate_context_linear_rehearsal_checkpoint
    records: list[dict[str, Any]] = []

    def read(*args, **kw):
        if records:
            if fault == "clock":
                kw["budget"].clock = lambda: 1.0
            elif fault == "value":
                records[0]["status"] = "tampered"
            else:
                records[0]["episodes"][0] = deepcopy(records[0]["episodes"][0])
        r = original(*args, **kw)
        records.append(r)
        return r

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", read)
    r = run()
    assert r["status"] == "failed" and r["native_games"][0]["games"]["status"] == "complete"
    assert len(records) == 1 and r["comparison"]["budget"]["updates_completed"] == 4


def test_should_not_invent_a_report_when_game_return_is_lost(scripted, monkeypatch):
    def lost(*a, **kw):
        WORK["lost_readouts"] += 1
        raise RuntimeError("lost game return")

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", lost)
    r = run()
    assert r["status"] == "failed" and r["native_games"] == []
    assert r["native_game_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)


def test_should_refuse_incompatible_paid_endpoint_before_game_actions(scripted, monkeypatch):
    original = study.compare_private_rehearsal

    def wrong(*a, **kw):
        r = original(*a, **kw)
        r["arms"][0]["native_endpoint"]["completed_updates"] = 7
        return r

    monkeypatch.setattr(study, "compare_private_rehearsal", wrong)
    r = run()
    assert r["status"] == "failed" and r["native_games"] == []
