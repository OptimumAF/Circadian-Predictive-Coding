"""Private return readouts through the existing loop; literal IO is not task evidence."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_a4_adaptation as app
from src.adapters import minigrid_affine_probe as binding
from src.adapters import minigrid_context_linear_learner as native
from src.adapters import minigrid_context_linear_policy as frozen
from src.adapters import minigrid_policy_evaluation as evaluation
from src.core.reference_controller import ReferencePolicy
from tests import test_a4_fresh_probe_workflow as whole
from tests import test_short_affine_return as returning
from test_context_linear_causal_games import PhaseEnvironment
from test_retained_context_linear_games import budget

WORK = dict(games=0, environments=0, steps=0, predicts=0, predict_rows=0)
NATIVE = returning.NATIVE_OWNER


class LiteralEnvironment(PhaseEnvironment):
    work: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        WORK["environments"] += 1
        super().__init__(*args, **kwargs)

    def step(self, action):
        WORK["steps"] += 1
        return super().step(action)


def literal_forward(head, features):
    WORK["predicts"] += 1
    WORK["predict_rows"] += len(features)
    return np.tile(np.eye(7)[2], (len(features), 1))


def endpoint(phase):
    s = returning.state()
    updates, rows, head_rows = (6, 72, 9) if phase == "after_probe" else (8, 108, 18)
    return replace(
        s,
        completed_updates=updates,
        training_examples=rows,
        heads=tuple(
            replace(h, fit_rows=head_rows, singular_values=(0.0,) * head_rows) for h in s.heads
        ),
    )


def boundary(read=None, *, after_clocks=None):
    if read is not None:
        read()
    if after_clocks is not None:
        after_clocks()


@pytest.fixture
def game_io(monkeypatch):
    monkeypatch.setattr(LiteralEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", LiteralEnvironment)
    monkeypatch.setattr(frozen, "predict_categorical_linear", literal_forward)
    monkeypatch.setattr(NATIVE, "predict", returning.predict_bridge)
    original = evaluation.evaluate_context_linear_rehearsal_checkpoint

    def counted(*args, **kwargs):
        WORK["games"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", counted)


def read(phase, **options):
    return evaluation.evaluate_context_linear_rehearsal_checkpoint(
        NATIVE(),
        endpoint(phase),
        arm=None,
        budget=budget(),
        check_boundary=lambda: None,
        short_return_phase=phase,
        qualified_boundary=boundary,
        **options,
    )


@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
def test_should_bind_exact_short_state_and_six_roles_to_original_loop(monkeypatch, phase):
    def run(source, saved, **options):
        assert saved == endpoint(phase) and saved is not endpoint(phase)
        assert source.snapshot_state().completed_updates == 0
        assert options["qualified_boundary"] is boundary
        assert options["require_initial_source"] is True
        assert [
            (layout.context, layout.branch_row, cue, role)
            for layout, cue, role in options["live_plan"]
        ] == [
            ("F-blue", 2, "blue", "probe_test"),
            ("F-red", 2, "red", "probe_test"),
            ("A", 3, "blue", "train"),
            ("A", 3, "red", "train"),
            ("A", 4, "blue", "train"),
            ("A", 4, "red", "train"),
        ]
        options["extra_check"]()
        assert options["return_metadata"]["source_recipe"] == "short_affine_return"
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    assert read(phase)["status"] == "complete"


@pytest.mark.parametrize(
    "fault", ["phase", "boolean", "arm", "boundary", "updates", "rows", "head", "failed", "source"]
)
def test_should_refuse_other_recipes_before_any_game(monkeypatch, fault):
    phase: Any
    source: Any
    authority: Any
    phase, s, source, arm, authority = (
        "after_probe",
        endpoint("after_probe"),
        NATIVE(),
        None,
        boundary,
    )
    if fault == "phase":
        phase = "after_return"
    elif fault == "boolean":
        phase = True
    elif fault == "arm":
        arm = "current_only"
    elif fault == "boundary":
        authority = None
    elif fault == "updates":
        s = replace(s, completed_updates=7)
    elif fault == "rows":
        s = replace(s, training_examples=73)
    elif fault == "head":
        s = replace(s, heads=endpoint("after_return").heads)
    elif fault == "failed":
        s = replace(s, failed_work=(1, 9, 0, 0, True))
    else:
        source = object()
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            source,
            s,
            arm=arm,
            budget=budget(),
            check_boundary=lambda: None,
            short_return_phase=phase,
            qualified_boundary=authority,
        )


@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
def test_should_execute_six_closed_literal_games_without_source_updates(game_io, phase):
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    r = read(phase)
    assert r["status"] == "complete" and r["state_unchanged"] is True
    assert len(r["episodes"]) == 6
    assert [e["source_role"] for e in r["episodes"]] == ["probe_test"] * 2 + ["train"] * 4
    assert all(e["status"] == "complete" and e["environment"]["closed"] for e in r["episodes"])
    assert r["work"]["live_prediction_calls_returned"] == 12
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


@pytest.mark.parametrize("fault", ["counter", "clock", "boolean", "limits"])
def test_should_refuse_replaced_budget_authority_before_more_work(monkeypatch, fault):
    session = budget()

    def run(*args, **options):
        if fault == "clock":
            session.clock = lambda: 1.0
        elif fault == "limits":
            object.__setattr__(session.budget, "max_wall_seconds", 151)
        else:
            session.replay_examples_completed = False if fault == "boolean" else 1
        options["extra_check"]()
        pytest.fail("authority accepted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            NATIVE(),
            endpoint("after_probe"),
            arm=None,
            budget=session,
            check_boundary=lambda: None,
            short_return_phase="after_probe",
            qualified_boundary=boundary,
        )


@pytest.fixture
def private(returning_scripted, game_io):
    return lambda **kw: returning.run(native_games=True, readout_budget=budget(), **kw)


@pytest.fixture
def returning_scripted(monkeypatch):
    setup: Any = returning.scripted
    setup.__wrapped__(monkeypatch)


def test_should_read_two_same_run_endpoints_and_preserve_all_learning_charges(private):
    r = private()
    assert r["status"] == "complete", r.get("reason")
    assert [p["phase"] for p in r["native_games"]] == ["after_probe", "after_return"]
    assert [p["games"]["source_fit_rows"] for p in r["native_games"]] == [72, 108]
    assert r["native_game_work"] == dict(attempts=2, returned=2, unreturned_work_unknown=False)
    assert r["probe"]["completed_training_examples"] == 18
    assert r["probe"]["return_phase"]["completed_training_examples"] == 36
    assert r["source_native_state_unchanged"] and r["source_fork_calls"] == 1
    assert len(r["native_games"]) == 2


def test_should_compose_the_whole_fresh_workflow_with_the_original_owner(monkeypatch, game_io):
    h = whole.Harness(monkeypatch)
    monkeypatch.setattr(native, "ContextLinearLearner", NATIVE)
    monkeypatch.setattr(NATIVE, "train_batch", returning.train_bridge)
    monkeypatch.setattr(NATIVE, "predict", returning.predict_bridge)
    owners = []
    original_game = app.evaluate_context_linear_stream_checkpoint

    def source_game(*args, **kw):
        owners.append(kw["budget"])
        return original_game(*args, **kw)

    monkeypatch.setattr(app, "evaluate_context_linear_stream_checkpoint", source_game)
    original_probe = binding.run_affine_fresh_probe

    def probe(*args, **kw):
        assert owners and all(owner is owners[0] for owner in owners)
        assert kw["readout_budget"] is owners[0]
        assert owners[0].updates_completed == owners[0].replay_examples_completed == 0
        return original_probe(*args, **kw)

    monkeypatch.setattr(binding, "run_affine_fresh_probe", probe)
    original = app.run_fresh_adaptation
    monkeypatch.setattr(
        app, "run_fresh_adaptation", lambda **kw: original(return_old=True, native_games=True, **kw)
    )
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    r = h.run()
    assert r["status"] == "complete", (r.get("reason"), r.get("capture_error"))
    assert r["readout_budget"]["started_at"] == r["readout_budget"]["observed_started_at"]
    assert r["readout_budget"]["updates"] == r["readout_budget"]["replay"] == 0
    assert r["training"]["stream"]["completed_updates"] == 4
    assert r["probe"]["native_game_work"]["returned"] == 2
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
def test_should_keep_paid_prefix_and_stop_when_a_game_step_is_lost(private, monkeypatch, phase):
    original = LiteralEnvironment.step

    def step(owner, action):
        if (
            WORK["games"] - start == (1 if phase == "after_probe" else 2)
            and owner.layout.branch_row == 4
        ):
            WORK["steps"] += 1
            raise RuntimeError("lost old-A4 step")
        return original(owner, action)

    start = WORK["games"]
    monkeypatch.setattr(LiteralEnvironment, "step", step)
    r = private()
    assert r["status"] == "failed"
    assert len(r["native_games"]) == (1 if phase == "after_probe" else 2)
    g = r["native_games"][-1]["games"]
    assert [e["status"] for e in g["episodes"]] == ["complete"] * 4 + ["failed", "not_run"], (
        [e["status"] for e in g["episodes"]],
        g.get("reason"),
        r.get("reason"),
    )
    assert all(e["environment"]["closed"] for e in g["episodes"][:5])
    assert r["native_game_work"]["unreturned_work_unknown"] is True
    assert r["return_budget"]["updates_completed"] == (0 if phase == "after_probe" else 2)


@pytest.mark.parametrize("fault", ["value", "replacement", "work"])
def test_should_refuse_changed_paid_first_phase_before_return_fits(private, monkeypatch, fault):
    original, saved = evaluation.evaluate_context_linear_rehearsal_checkpoint, []

    def capture(*args, **kw):
        r = original(*args, **kw)
        saved.append(r)
        return r

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", capture)
    original_train = NATIVE.train_batch

    def train(owner, x, y):
        if owner.snapshot_state().completed_updates == 6:
            saved[0]["status"] = "tampered"
        return original_train(owner, x, y)

    # A source port may run callbacks. The post-fit boundary must refuse the
    # changed preceding report and keep its trusted copy and charged work.
    if fault == "replacement":

        def train(owner, x, y):
            if owner.snapshot_state().completed_updates == 6:
                saved[0]["episodes"][0] = deepcopy(saved[0]["episodes"][0])
            return original_train(owner, x, y)
    elif fault == "work":

        def train(owner, x, y):
            if owner.snapshot_state().completed_updates == 6:
                saved[0]["work"]["live_prediction_calls_returned"] += 1
            return original_train(owner, x, y)

    monkeypatch.setattr(NATIVE, "train_batch", train)
    r = private()
    assert r["status"] == "failed" and len(saved) == 1
    assert r["native_games"][0]["games"]["status"] == "complete"
    assert r["return_budget"]["updates_completed"] <= 1


def test_should_count_lost_call_without_inventing_a_game_report(private, monkeypatch):
    def lost(*args, **kwargs):
        WORK["games"] += 1
        raise RuntimeError("lost game return")

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", lost)
    r = private()
    assert r["status"] == "failed" and r["native_games"] == []
    assert r["native_game_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)
    assert r["return_budget"]["updates_completed"] == 0


@pytest.mark.parametrize("flag,returned", [(None, True), (1, True), ([], True), (True, False)])
def test_should_refuse_wrong_opt_in_before_work(monkeypatch, flag, returned):
    monkeypatch.setattr(app, "_run_adaptation", lambda *a, **k: pytest.fail("workflow reached"))
    with pytest.raises(ValueError, match="native_games"):
        app.run_fresh_adaptation(
            reference_policy=ReferencePolicy("drift"),
            retained_probe=True,
            return_old=returned,
            native_games=flag,
        )


def test_should_accept_complete_negative_goals_without_changing_learning(private, monkeypatch):
    def negative(*args, **kwargs):
        WORK["games"] += 1
        return dict(status="complete", state_unchanged=True, episodes=[dict(success=False)] * 6)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", negative)
    r = private()
    assert r["status"] == "complete", (r.get("reason"), r.get("native_capture_error"))
    assert r["probe"]["return_phase"]["completed_training_examples"] == 36


def test_should_preserve_no_game_default(returning_scripted, monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "evaluate_context_linear_rehearsal_checkpoint",
        lambda *a, **k: pytest.fail("game reached"),
    )
    r = returning.run()
    assert r["status"] == "complete"
    assert "native_games" not in r and "native_game_work" not in r


@pytest.mark.parametrize("fault", ["clock", "endpoint"])
def test_should_refuse_changed_native_endpoint_or_owner_before_first_game(
    private, monkeypatch, fault
):
    children = []
    original_fork = binding._AffineProbeSource.fork

    def fork(owner):
        child = original_fork(owner)
        children.append(child)
        return child

    monkeypatch.setattr(binding._AffineProbeSource, "fork", fork)
    original = evaluation.evaluate_context_linear_rehearsal_checkpoint

    def changed(*args, **kw):
        if fault == "clock":
            kw["budget"].clock = lambda: 1.0
        else:
            child = children[0]
            child._state = replace(child._state, training_examples=73)
        return original(*args, **kw)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", changed)
    before = WORK["environments"]
    r = private()
    assert r["status"] == "failed" and WORK["environments"] == before
    assert r["return_budget"]["updates_completed"] == 0


def test_should_refuse_paid_game_corruption_from_final_capture(private, monkeypatch):
    reports, armed = [], [False]
    original = evaluation.evaluate_context_linear_rehearsal_checkpoint

    def capture(*args, **kw):
        r = original(*args, **kw)
        reports.append(r)
        return r

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", capture)
    original_asdict = binding.asdict

    def capture_state(value):
        result = original_asdict(value)
        # This conversion is the adapter's final disposable_state capture,
        # after the protected return observer and both successful game calls.
        if getattr(value, "completed_updates", None) == 8 and len(reports) == 2:
            armed[0] = True
            reports[0]["status"] = "late corruption"
        return result

    monkeypatch.setattr(binding, "asdict", capture_state)
    r = private()
    assert armed[0] and r["status"] == "failed"
    assert "paid" in r["native_capture_error"]
    assert r["native_games"][0]["games"]["status"] == "complete"


def test_should_recheck_paid_games_after_the_final_guard_snapshot_port(private, monkeypatch):
    reports = []
    snapshot_calls: list[int | None] = [None]
    original = evaluation.evaluate_context_linear_rehearsal_checkpoint

    def capture(*args, **kw):
        result = original(*args, **kw)
        reports.append(result)
        return result

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", capture)
    original_asdict = binding.asdict

    def arm(value):
        result = original_asdict(value)
        if getattr(value, "completed_updates", None) == 8 and len(reports) == 2:
            snapshot_calls[0] = 0
        return result

    monkeypatch.setattr(binding, "asdict", arm)
    original_snapshot = binding._AffineProbeSource.snapshot_state

    def snapshot(owner):
        result = original_snapshot(owner)
        if snapshot_calls[0] is not None:
            snapshot_calls[0] += 1
            # First is final source capture; second is the final guard port,
            # after its initial paid-prefix proof and before its last proof.
            if snapshot_calls[0] == 2:
                reports[0]["status"] = "late source-port corruption"
        return result

    monkeypatch.setattr(binding._AffineProbeSource, "snapshot_state", snapshot)
    r = private()
    assert snapshot_calls[0] is not None and snapshot_calls[0] >= 2
    assert r["status"] == "failed"
    assert "paid" in r["native_capture_error"]
    assert r["native_games"][0]["games"]["status"] == "complete"


@pytest.mark.parametrize("fault", ["missing", "limits", "updates", "disabled"])
def test_should_refuse_wrong_original_owner_before_source_copy(monkeypatch, fault):
    session = budget()
    options: dict[str, Any] = dict(native_games=True, readout_budget=session)
    if fault == "missing":
        options.pop("readout_budget")
    elif fault == "limits":
        object.__setattr__(session.budget, "max_wall_seconds", 151)
    elif fault == "updates":
        session.updates_completed = 1
    else:
        options["native_games"] = False
    monkeypatch.setattr(
        binding, "copy_context_linear_training_snapshot", lambda *a: pytest.fail("copy reached")
    )
    with pytest.raises(ValueError):
        binding.run_affine_fresh_probe((), None, retained_a4=True, return_old=True, **options)
