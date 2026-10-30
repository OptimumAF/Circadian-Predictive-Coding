"""Existing joint TRAIN rehearsal on private transfer forks; scoped software evidence."""

from copy import deepcopy
from dataclasses import replace

import pytest

from scripts import context_head_rehearsal as rehearsal
from scripts import context_retained_probe as composed
from scripts import context_head_live as live
from scripts import probe_ticket_policy_learning as probe
from scripts import recurring_ticket_workflow_example as entry
from scripts.ticket_linear_probe import bind_source
from scripts.context_training_streams import WITHHELD_CONTEXT_RECIPES
from src.app.toy_execution_budget import ToyBudgetSession
from src.shared.process_memory import ProcessRssSampler
from test_context_linear_transfer_probe import donor
from test_context_retained_probe import scope_fixture
from test_context_retained_publication import qualified_literal
from test_context_catalog_replay import spec


@pytest.fixture(scope="module")
def prepared():
    return rehearsal.prepare_cohorts(spec())


@pytest.mark.parametrize("control", [True, "unknown", 0, "old_train_replay"])
def test_should_refuse_invalid_joint_control_before_source_work(control, monkeypatch):
    monkeypatch.setattr(entry, "training_cases", lambda **_: pytest.fail("source work"))
    with pytest.raises(ValueError, match="probe rehearsal"):
        entry.run_example(probe_rehearsal=control)
    with pytest.raises(ValueError, match="joint rehearsal"):
        probe.run_arm(
            None, None, (), None, lambda: None, None, "continued", joint_rehearsal=control
        )


@pytest.mark.parametrize(
    "change", [{}, {"paired_serving": "retained_joint"}, {"method": "perceptron"}]
)
def test_should_refuse_joint_option_outside_retained_probe_before_source(change, monkeypatch):
    args = dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="retained_joint_probe",
        probe_rehearsal="joint",
    )
    args.update(change or {"context_return": False})
    monkeypatch.setattr(entry, "training_cases", lambda **_: pytest.fail("source work"))
    with pytest.raises(ValueError, match="probe rehearsal"):
        entry.run_example(**args)
    with pytest.raises(ValueError, match="probe rehearsal"):
        live.run_case(None, _probe_rehearsal="joint")


def test_should_admit_authentic_episode_cohorts_without_weakening_default(prepared):
    cohorts, _ = prepared
    source = donor()
    kwargs = dict(head_tasks=probe.TRANSFER_HEAD_TASKS)
    with pytest.raises(ValueError, match="cohorts differ"):
        bind_source(source, source.snapshot_state(), cohorts, spec(), lambda: None, **kwargs)
    guard = bind_source(
        source,
        source.snapshot_state(),
        cohorts,
        spec(),
        lambda: None,
        episode_cohorts=True,
        **kwargs,
    )
    guard()
    changed = deepcopy(cohorts)
    object.__setattr__(changed[0][0].episode, "role", "train")
    with pytest.raises(ValueError, match="metadata"):
        bind_source(
            source,
            source.snapshot_state(),
            changed,
            spec(),
            lambda: None,
            episode_cohorts=True,
            **kwargs,
        )


@pytest.mark.parametrize("fault", ["role", "context", "labels", "replacement"])
def test_should_refuse_foreign_bank_or_changed_bank_before_fork(prepared, fault):
    cohorts, original = prepared
    bank = deepcopy(original)
    if fault == "replacement":
        bank = bank[:3]
    elif fault == "labels":
        transition = replace(bank[0].transitions[0], action=5)
        bank = (replace(bank[0], transitions=(transition, *bank[0].transitions[1:])), *bank[1:])
    else:
        bank = (
            replace(bank[0], **{fault: "probe_train" if fault == "role" else "foreign"}),
            *bank[1:],
        )
    replay = ToyBudgetSession(rehearsal._limits(replay=144), lambda: 0.0)
    with pytest.raises(ValueError):
        rehearsal.joint_probe_factory(cohorts, bank, "joint", replay, lambda: None)
    assert replay.replay_examples_completed == 0


def test_should_keep_original_bank_pinned_after_construction(prepared):
    cohorts, bank = deepcopy(prepared)
    replay = ToyBudgetSession(rehearsal._limits(replay=144), lambda: 0.0)
    guard, factory = rehearsal.joint_probe_factory(cohorts, bank, "joint", replay, lambda: None)
    guard()
    object.__setattr__(bank[0], "role", "probe_train")
    with pytest.raises(ValueError, match="bank changed"):
        factory(None, None, 16)
    assert replay.replay_examples_completed == 0


def test_should_refuse_equal_valued_episode_replacement(prepared):
    cohorts, _ = deepcopy(prepared)
    source = donor()
    guard = bind_source(
        source,
        source.snapshot_state(),
        cohorts,
        spec(),
        lambda: None,
        head_tasks=probe.TRANSFER_HEAD_TASKS,
        episode_cohorts=True,
    )
    object.__setattr__(cohorts[0][0], "episode", replace(cohorts[0][0].episode))
    with pytest.raises(ValueError, match="inputs"):
        guard()


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_should_dispatch_public_option_and_account_private_fits(tmp_path, monkeypatch, control):
    from scripts import context_catalog_learning as writer

    calls = []
    observations = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(
            complete=True,
            training_work=dict(attempted_updates=12, attempted_rows=72),
            paired_serving=dict(
                training_work=dict(attempted_updates=11, attempted_rows=72),
                isolated_transfer_probe=dict(
                    private_work=dict(attempted_updates=48, attempted_training_examples=288)
                ),
            ),
        )

    def write(path, report, **kwargs):
        observations.append(kwargs["observed_work"]["counts"])
        return True

    monkeypatch.setattr(entry, "run_example", run)
    monkeypatch.setattr(writer, "write_catalog_report", write)
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--paired-serving",
        "retained_joint_probe",
        "--probe-rehearsal",
        control,
        "--output",
        str(tmp_path / "joint.json"),
    ]
    assert entry.main(flags) == 0
    assert calls[0]["probe_rehearsal"] == control
    assert observations == [dict(train_passes=71, training_rows=432)]


def test_should_keep_lost_replay_cost_unknown_and_original_charge(monkeypatch):
    runtime, helper, report, record = scope_fixture()

    def lose(seed, state, cohorts, spec, guard, memory, mode, *, row, **options):
        assert options["joint_rehearsal"] == "joint" and len(options["training_bank"]) == 4
        guard()
        row["probe"] = dict(rehearsal=dict(charged_exposures=9))
        raise RuntimeError("lost joint observation")

    monkeypatch.setattr(composed, "run_arm", lose)
    with pytest.raises(RuntimeError, match="lost joint"):
        composed.run_probe(
            runtime, spec(), helper, lambda: None, None, report, record, rehearsal_control="joint"
        )
    assert record["joint_replay_charged_exposures"] == 9
    assert record["private_work"]["attempted_updates"] is None
    assert record["private_work"]["unreturned_work_unknown"]
    assert record["donor_unchanged"] and record["retained_scope_verified"] is None


def require_task_counts(counts, control):
    """Initial selected ability is measured, not borrowed from another source."""
    assert counts[0][0] == 4 and 0 <= counts[0][1] <= 2
    assert counts[2] == (4, 0)
    assert counts[1] == ((4, 2) if control == "joint" else (2, 2))


def test_should_accept_measured_initial_ability_without_relabeling_native_result():
    for initial in (0, 1, 2):
        require_task_counts([(4, initial), (4, 2), (4, 0)], "joint")
    with pytest.raises(AssertionError):
        require_task_counts([(4, 1), (2, 2), (4, 0)], "joint")
    with pytest.raises(AssertionError):
        require_task_counts([(4, 1), (4, 2), (2, 0)], "joint")


def test_should_finish_actual_retained_stream_then_pay_both_private_controls(
    monkeypatch, record_property
):
    """One retained fixture owner; two private arms, not a scientific comparison."""
    original = composed.run_probe
    records = {}

    def both(runtime, spec, helper, boundary, memory, report, record, **options):
        assert options == {"rehearsal_control": "joint"}
        for control in ("joint", "current_only"):
            current = record if control == "joint" else {}
            report["isolated_transfer_probe"] = current
            try:
                original(
                    runtime,
                    spec,
                    helper,
                    boundary,
                    memory,
                    report,
                    current,
                    rehearsal_control=control,
                )
            finally:
                records[control] = deepcopy(current)

    monkeypatch.setattr(composed, "run_probe", both)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        result = live.run_case(
            None,
            _linear_fit=True,
            _owned_source=(qualified_literal(), lambda: None),
            _memory=memory,
            _retained_joint=True,
            _probe_retained=True,
            _probe_rehearsal="joint",
            stream_recipe=WITHHELD_CONTEXT_RECIPES[0],
        )
    # Retain partial terminal observations BEFORE asserting success; a failed
    # endpoint cannot erase paid fits, clocks, phase reason or missing task slots.
    for control, current in records.items():
        arm = current.get("arm", {})
        phases = arm.get("probe", {})
        summary_counts = []
        for phase in ("before_probe", "after_probe", "after_return"):
            rows = [e["episode"] for e in arm.get("episodes", []) if e["phase"] == phase]
            summary_counts.append(
                dict(
                    phase=phase,
                    observed_slots=len(rows),
                    old_success=sum(e["success"] for e in rows[:4]) if len(rows) >= 4 else None,
                    selected_success=sum(e["success"] for e in rows[4:])
                    if len(rows) == 6
                    else None,
                )
            )
        record_property(
            control,
            dict(
                endpoints=summary_counts,
                private_work=current.get("private_work"),
                donor_unchanged=current.get("donor_unchanged"),
                retained_scope_verified=current.get("retained_scope_verified"),
                native_readouts=[
                    (r["native_updates"], r["native_rows"]) for r in arm.get("readout_work", [])
                ],
                sessions=arm.get("sessions"),
                replay_session=arm.get("joint_replay_session"),
                replay_events=len(phases.get("rehearsal", {}).get("events", [])),
                phase_status={
                    k: {field: v.get(field) for field in ("status", "reason")}
                    for k, v in (("probe", phases), ("return", phases.get("return_phase", {})))
                },
                outer_error=result.get("error"),
                source_unchanged=result.get("source_unchanged"),
                actor_unchanged=result.get("actor_unchanged"),
            ),
        )
    assert result["complete"], result.get("error")
    assert result["source_unchanged"] and result["actor_unchanged"]
    assert result["training_work"] == dict(
        attempted_updates=11, attempted_rows=72, returned_updates=11
    )
    assert result["candidate_state"]["completed_updates"] == 23
    for control, record in records.items():
        assert record["donor_unchanged"] and record["retained_scope_verified"]
        assert record["private_work"] == dict(
            attempted_updates=48,
            completed_updates=48,
            attempted_training_examples=288,
            completed_training_examples=288,
            unreturned_work_unknown=False,
        )
        arm = record["arm"]
        assert [(r["native_updates"], r["native_rows"]) for r in arm["readout_work"]] == [
            (23, 144),
            (55, 336),
            (71, 432),
        ]
        rehearsal.require_joint_probe(arm, control)
        counts = []
        for phase in ("before_probe", "after_probe", "after_return"):
            episodes = [e["episode"] for e in arm["episodes"] if e["phase"] == phase]
            assert all(
                e["complete"] and e["training_updates"] == e["external_actions"] == 0
                for e in episodes
            )
            counts.append(
                (sum(e["success"] for e in episodes[:4]), sum(e["success"] for e in episodes[4:]))
            )
        require_task_counts(counts, control)
        damaged = deepcopy(arm)
        damaged["joint_replay_session"]["replay"] = 0
        with pytest.raises(ValueError, match="history"):
            rehearsal.require_joint_probe(damaged, control)
        # All original F/return sessions remain paid and born once.
        assert all(
            s["limits"]["max_wall_seconds"] == 30 and s["updates"] == 16 for s in arm["sessions"]
        )
    assert (
        result["live_serving"]["closed"]
        and result["final_examples"] == result["probe_updates"] == 0
    )
