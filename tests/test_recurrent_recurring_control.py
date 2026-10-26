"""Recurring schedule/readout boundaries; scripted native ports are not learning."""

from copy import deepcopy
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.adapters import minigrid_public_cue_recurrent_pilot as pilot
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.infra.minigrid_affine_example import example_affine_training
from test_public_cue_recurrent_learner import configure_scripted_native, ScriptedNative


@pytest.fixture(scope="module")
def episodes():
    with example_affine_training() as path:
        return read_training_sources(path, raw_frames=True)


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)


def state(owner, point):
    return replace(
        owner.snapshot_state(),
        heads=tuple(
            replace(h, completed_updates=point // 2, training_examples=point * 9 // 2)
            for h in owner.snapshot_state().heads
        ),
    )


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=200,
            max_process_rss_bytes=pilot.RSS_LIMIT,
        ),
        monotonic,
    )


def readout(point):
    contexts = ("A", "B", "C_train")
    live = contexts if point == 128 else ("A",) if point == 32 else ()
    return dict(
        updates=point,
        method=port.METHOD,
        role="train",
        status="complete",
        state_unchanged=True,
        scores=[
            dict(
                context=c,
                cue=cue,
                ce=2.0 - point / 128,
                accuracy=point / 128,
                exact_sequence_accuracy=float(point == 128),
            )
            for c in contexts
            for cue in ("blue", "red")
        ],
        episodes=[
            dict(
                context=c,
                cue=cue,
                status="complete",
                state_unchanged=True,
                success=point == 128,
                environment=dict(closed=True),
                **{"return": 0.49375 if point == 128 else 0.0},
            )
            for c in live
            for cue in ("blue", "red")
        ],
    )


def test_should_build_exact_original_delayed_stream_schedule_from_raw_sources(episodes):
    sources = pilot._typed_sources(episodes)
    packets = pilot.shared._packets(sources)
    assert (
        len(packets) == 128 and len({pilot.shared._episode(p).episode_id for p in packets}) == 128
    )
    for i, packet in enumerate(packets):
        context = ("A", "B", "C_train", "A")[i // 32]
        original = sources[(0, 2, 4, 0)[i // 32] + i % 2]
        assert (
            packet.role == pilot.shared._episode(packet).role == "train"
            and packet.context == context
        )
        assert packet.features.shape == (9, 148) and packet.num_classes == 7
        assert _same_value(packet.features, original.features) and _same_value(
            packet.targets, original.targets
        )


def test_should_describe_fixed_native_subset_and_separate_recurring_protocol():
    old = pilot.describe_protocol(typed_input=True)
    new = pilot.describe_protocol(typed_input=True, recurring_training=True)
    assert old["readout_order"] == [0, 128] and old["planned_work"]["live_episodes"] == 4
    assert new["readout_order"] == [0, 32, 64, 96, 128]
    assert new["context_sequence"] == ["A", "B", "C_train", "A"]
    assert new["planned_work"]["raw_calls"] == 30 and new["planned_work"]["raw_rows"] == 270
    assert new["native_readout_points"] == {"32": ["A"], "128": ["A", "B", "C_train"]}
    assert new["protocol_id"] != old["protocol_id"]


@pytest.mark.parametrize("flag", [0, 1, None, "recurring"])
def test_should_refuse_nonboolean_mode_before_native_work(flag, monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", lambda: pytest.fail("no native work"))
    with pytest.raises(ValueError):
        pilot.run_pilot(recurring_training=flag)
    with pytest.raises(ValueError):
        pilot.describe_protocol(typed_input=True, recurring_training=flag)


def test_should_require_explicit_typed_sources_before_recurring_native_work(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", lambda: pytest.fail("no native work"))
    with pytest.raises(ValueError):
        pilot.run_pilot(recurring_training=True)
    with pytest.raises(ValueError):
        pilot.describe_protocol(recurring_training=True)


@pytest.mark.parametrize(
    "point,contexts", [(0, ()), (32, ("A",)), (64, ()), (96, ()), (128, ("A", "B", "C_train"))]
)
def test_should_admit_all_teacher_curves_and_only_fixed_native_subset(
    scripted, episodes, monkeypatch, point, contexts
):
    owner = port.PublicCueRecurrentLearner()
    sources = pilot._typed_sources(episodes)
    seen: list[Any] = []

    def kernel(*a, **k):
        seen.append((a, k))
        return dict(ok=True)

    monkeypatch.setattr(evaluation, "_run_checkpoint", kernel)
    result = evaluation.evaluate_public_cue_recurrent_checkpoint(
        owner,
        state(owner, point),
        updates=point,
        sources=sources,
        budget=budget(),
        check_boundary=lambda: None,
        recurring_training=True,
    )
    assert result == dict(ok=True)
    _, options = seen[0]
    assert options["public_history"] is False and options["require_initial_source"] is True
    assert len(options["sources"]) == 6 and options["role"] == "train"
    assert [layout.context for layout, _, _ in options["live_plan"]] == [
        c for c in contexts for _ in range(2)
    ]
    assert all(role == "train" for _, _, role in options["live_plan"])
    options["extra_check"]()


@pytest.mark.parametrize(
    "fault", ["point", "bool", "counter", "source", "order", "final", "probe", "budget", "flag"]
)
def test_should_refuse_invalid_recurring_readout_before_kernel(
    scripted, episodes, monkeypatch, fault
):
    owner = port.PublicCueRecurrentLearner()
    saved = state(owner, 64)
    packets, limits, point = pilot._typed_sources(episodes), budget(), 64
    flag: Any = True
    if fault == "point":
        point = 16
    if fault == "bool":
        point = False
    if fault == "counter":
        saved = replace(saved, heads=(replace(saved.heads[0], training_examples=1), saved.heads[1]))
    if fault == "source":
        packets = packets[:2]
    if fault == "order":
        packets = packets[::-1]
    if fault in ("final", "probe"):
        packets = (
            replace(packets[0], role="final_test" if fault == "final" else "probe_train"),
            *packets[1:],
        )
    if fault == "budget":
        limits = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), monotonic)
    if fault == "flag":
        flag = 1
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("no kernel work")
    )
    with pytest.raises(ValueError):
        evaluation.evaluate_public_cue_recurrent_checkpoint(
            owner,
            saved,
            updates=point,
            sources=packets,
            budget=limits,
            check_boundary=lambda: None,
            recurring_training=flag,
        )


def test_should_report_teacher_changes_and_native_unknown_points_separately():
    result = pilot._recurring_outcomes([readout(p) for p in pilot.shared.POINTS])
    assert len(result["TRAIN_curves"]) == 15 and len(result["phase_changes"]) == 12
    assert [(r["updates"], r["context"]) for r in result["native_outcomes"]] == [
        (32, "A"),
        (128, "A"),
        (128, "B"),
        (128, "C_train"),
    ]
    assert all(r["ce_gain"] == 0.25 for r in result["phase_changes"])
    assert result["native64_96"].startswith("unmeasured")


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "point_bool",
        "score_context",
        "nan",
        "negative",
        "extra_native",
        "cleanup",
        "source",
        "partial",
    ],
)
def test_should_refuse_incomplete_or_corrupt_recurring_claims(fault):
    records = [readout(p) for p in pilot.shared.POINTS]
    if fault == "missing":
        records.pop()
    if fault == "point_bool":
        records[0]["updates"] = False
    if fault == "score_context":
        records[1]["scores"][0]["context"] = "D"
    if fault == "nan":
        records[0]["scores"][0]["ce"] = float("nan")
    if fault == "negative":
        records[0]["scores"][0]["ce"] = -1.0
    if fault == "extra_native":
        records[0]["episodes"] = readout(32)["episodes"]
    if fault == "cleanup":
        records[1]["episodes"][0]["environment"]["closed"] = False
    if fault == "source":
        records[-1]["state_unchanged"] = False
    if fault == "partial":
        records[-1]["status"] = "incomplete"
    with pytest.raises(ValueError):
        pilot._recurring_outcomes(records)


def test_should_execute_all_four_phase_boundaries_in_existing_runtime_with_scripted_heads(
    scripted, episodes
):
    source = port.PublicCueRecurrentLearner()
    initial = source.snapshot_state()
    full = pilot.shared._packets(pilot._typed_sources(episodes))
    packets = tuple(full[i] for i in (0, 1, 32, 33, 64, 65, 96, 97))
    observed = []

    def observe(point, snapshot, batch):
        observed.append((point, deepcopy(snapshot)))

    stream = AttributionStream(
        source,
        packets,
        (),
        training_only=True,
        label_delay=1,
        checkpoint_updates=(0, 2, 4, 6, 8),
        observe_checkpoint=observe,
        budget=ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=8, max_wall_seconds=30), monotonic
        ),
    )
    stream.run_until(8)
    saved = stream.candidate_snapshot()
    result = stream.finish()
    assert [point for point, _ in observed] == [0, 2, 4, 6, 8]
    assert all(s.completed_updates == p and s.training_examples == 9 * p for p, s in observed)
    assert [s.completed_updates for s in saved.heads] == [4, 4]
    assert len(ScriptedNative.calls) == 8 and saved.training_examples == 72
    assert _same_value(source.snapshot_state(), initial)
    assert _same_value(stream._runtime.actor.snapshot_state().state, initial)
    receipts = result["applied_updates"]
    assert len(receipts) == 8
    assert [
        (r["observed_at"], r["arrived_at"], r["applied_at"], r["update_number"]) for r in receipts
    ] == [(i, i + 1, i + 1, i + 1) for i in range(8)]


@pytest.mark.parametrize("fault", ["none", "readout", "binding", "budget", "source"])
def test_should_stop_after_failure_or_replaced_paid_evidence_and_preserve_prefix(
    scripted, episodes, monkeypatch, fault
):
    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        pilot,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    captured = {}
    seen = []

    def train(row, packets, memory, deadline, check, *, learner_factory):
        source = learner_factory(port.METHOD)
        row.update(
            status="trained",
            charged_updates=128,
            native_training_examples=1152,
            source_unchanged=True,
            actor_unchanged=True,
        )
        arm = dict(
            source=source,
            states=[(p, state(source, p)) for p in pilot.shared.POINTS],
            stream=object(),
            budget=object(),
            guard=check,
        )
        captured["arm"] = arm
        return arm

    def observe(*args, **kwargs):
        point = kwargs["updates"]
        seen.append(point)
        if point == 32:
            if fault == "readout":
                raise evaluation.MiniGridEvaluationFailed(
                    dict(readout(point), status="failed", reason="lost return")
                )
            if fault == "binding":
                captured["arm"]["states"] = deepcopy(captured["arm"]["states"])
            if fault == "budget":
                kwargs["budget"].updates_completed = 1
            if fault == "source":
                kwargs["sources"][0].features[0, 0] += 0.125
        return readout(point)

    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", observe)
    result = pilot.run_pilot(episodes, recurring_training=True)
    if fault == "none":
        assert seen == list(pilot.shared.POINTS) and result["status"] == "complete"
        assert result["final_preservation"] is True
        return
    assert seen == [0, 32] and result["status"] == "incomplete"
    assert result["outcomes"] is result["final_preservation"] is None
    assert result["training"]["charged_updates"] == 128
    assert result["evaluations"][0] == readout(0)
