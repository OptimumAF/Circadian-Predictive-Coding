"""Scripted native hidden reads and real tiny literal SVDs; never game evidence."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_paired_bc_smoke as cli
from src.adapters import minigrid_public_cue_health as health
from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters import minigrid_public_cue_recurrent_pilot as pilot
from src.app.coupled_attribution import _same_value
from test_public_cue_recurrent_integration import sources as literal_a
from test_public_cue_recurrent_learner import configure_scripted_native
from test_recurrent_recurring_control import budget, readout, state


def sources():
    a = literal_a()
    return tuple(
        pilot.shared._batch(
            replace(
                a[i % 2].episode,
                context=context,
                episode_id=f"minigrid-source-{context}-{('blue', 'red')[i % 2]}",
            )
        )
        for i, context in enumerate(("A", "A", "B", "B", "C_train", "C_train"))
    )


def hidden():
    values = np.linspace(-0.8, 0.8, 72).reshape(9, 8)
    values[:, 0], values[:, 1] = 0, 1
    return values


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)
    seen = []

    def read(head, features):
        seen.append((deepcopy(head), features.copy()))
        return hidden()

    monkeypatch.setattr(health, "read_native_activations", read)
    return seen


def test_should_route_actual_public_cues_and_return_detached_bounded_metrics(scripted):
    source = public.PublicCueRecurrentLearner()
    saved = state(source, 32)
    saved.heads[0].parameters["marker"][0] = 10
    saved.heads[1].parameters["marker"][0] = 20
    before, original = source.snapshot_state(), deepcopy(saved)
    packets, limits = sources(), budget()
    started, clock, config = limits.started_at, limits.clock, limits.budget
    report = health.read_checkpoint_health(
        source, saved, updates=32, sources=packets, budget=limits, check_boundary=lambda: None
    )
    assert report["status"] == "complete" and len(report["scores"]) == 6
    assert [head.parameters["marker"][0] for head, _ in scripted] == [10, 20] * 3
    assert [row["head"] for row in report["scores"]] == [0, 1] * 3
    for row in report["scores"]:
        metrics = row["metrics"][0]
        assert metrics["dead_fraction"] == 1 / 8 and metrics["saturated_fraction"] == 1 / 8
        assert metrics["feature_effective_rank"] == pytest.approx(1)
        assert "activations" not in row
    work = report["work"]
    assert work["activation_reads_attempted"] == work["activation_reads_completed"] == 6
    assert work["forward_calls_completed"] == work["discarded_output_projections_completed"] == 54
    assert work["svd_calls_completed"] == 6 and work["activation_array_bytes_returned"] == 3456
    assert work["optimizer_steps"] == work["published_updates"] == 0
    assert _same_value(before, source.snapshot_state()) and _same_value(saved, original)
    assert limits.started_at == started and limits.clock is clock and limits.budget is config
    assert limits.updates_completed == limits.replay_examples_completed == 0


@pytest.mark.parametrize("mode", [None, "recent", "coverage"])
def test_should_admit_all_declared_exact_native_clocks_without_ignoring_history(scripted, mode):
    source = public.PublicCueRecurrentLearner()
    for point in (0, 32, 64, 96, 128):
        native = point if mode is None else point + 2 * (point // 8)
        report = health.read_checkpoint_health(
            source,
            state(source, native),
            updates=point,
            sources=sources(),
            budget=budget(),
            check_boundary=lambda: None,
            recurrent_replay=mode,
        )
        assert report["native_updates"] == native and report["status"] == "complete"


@pytest.mark.parametrize(
    "fault",
    [
        "final_test",
        "probe_train",
        "probe_test",
        "order",
        "arrays",
        "cue",
        "missing",
        "point_bool",
        "mode",
    ],
)
def test_should_refuse_whole_input_before_native_validation(scripted, monkeypatch, fault):
    source = public.PublicCueRecurrentLearner()
    saved, packets, point, mode = source.snapshot_state(), sources(), 0, None
    if fault in {"final_test", "probe_train", "probe_test"}:
        packets = (replace(packets[0], role=fault),) + packets[1:]
    elif fault == "order":
        packets = packets[::-1]
    elif fault == "arrays":
        packets[0].features[0, 0] = 0
    elif fault == "cue":
        first = packets[0].episode
        red = packets[1].episode.transitions[0].observation
        changed = replace(
            first,
            transitions=(replace(first.transitions[0], observation=red),) + first.transitions[1:],
        )
        packets = (pilot.shared._batch(changed),) + packets[1:]
    elif fault == "missing":
        packets = packets[:-1]
    elif fault == "point_bool":
        point = True
    else:
        mode = "other"
    monkeypatch.setattr(
        public, "copy_snapshot", lambda *a: pytest.fail("native validation reached")
    )
    with pytest.raises(ValueError):
        health.read_checkpoint_health(
            source,
            saved,
            updates=point,
            sources=packets,
            budget=budget(),
            check_boundary=lambda: None,
            recurrent_replay=mode,
        )
    assert not scripted


@pytest.mark.parametrize("fault", ["head", "rows", "failed", "policy"])
def test_should_keep_invalid_snapshot_work_and_refuse_activation(scripted, fault):
    source = public.PublicCueRecurrentLearner()
    saved = state(source, 32)
    if fault == "head":
        saved = replace(
            saved,
            heads=(
                replace(saved.heads[0], completed_updates=15),
                replace(saved.heads[1], completed_updates=17),
            ),
        )
    elif fault == "rows":
        saved = replace(
            saved, heads=(replace(saved.heads[0], training_examples=143), saved.heads[1])
        )
    elif fault == "failed":
        saved = replace(saved, failed_work=public.FailureWork("train", 0))
    else:
        saved = replace(
            saved,
            policy=replace(
                saved.policy, native_policy=replace(saved.policy.native_policy, seed=99)
            ),
        )
    with pytest.raises(health.ActivationHealthFailed) as stopped:
        health.read_checkpoint_health(
            source,
            saved,
            updates=32,
            sources=sources(),
            budget=budget(),
            check_boundary=lambda: None,
        )
    assert stopped.value.report["work"]["public_snapshot_copies_attempted"] == 1
    assert not scripted


@pytest.mark.parametrize(
    "fault",
    [
        "checkpoint",
        "source",
        "equal_owner",
        "features",
        "head_copy",
        "input_copy",
        "clock",
        "charge",
        "lost",
    ],
)
def test_should_preserve_paid_failure_and_not_admit_next_read(scripted, monkeypatch, fault):
    source = public.PublicCueRecurrentLearner()
    saved, packets, limits = state(source, 32), sources(), budget()
    old = health.read_native_activations

    def read(head, features):
        result = old(head, features)
        if len(scripted) == 2:
            if fault == "checkpoint":
                saved.heads[1].gradients["gradient"][0] += 1
            elif fault == "source":
                source._state.heads[0].parameters["marker"][0] += 1
            elif fault == "equal_owner":
                source._state = deepcopy(source._state)
            elif fault == "features":
                packets[0].features[0, 0] = 0
            elif fault == "head_copy":
                head.parameters["marker"][0] += 1
            elif fault == "input_copy":
                features[0, 0] = 0
            elif fault == "clock":
                limits.started_at += 1
            elif fault == "charge":
                limits.updates_completed = 1
            else:
                raise RuntimeError("lost activation return")
        return result

    monkeypatch.setattr(health, "read_native_activations", read)
    with pytest.raises(health.ActivationHealthFailed) as stopped:
        health.read_checkpoint_health(
            source, saved, updates=32, sources=packets, budget=limits, check_boundary=lambda: None
        )
    report = stopped.value.report
    assert len(scripted) == 2 and len(report["scores"]) == 1
    assert report["work"]["activation_reads_attempted"] == 2
    assert report["work"]["activation_reads_completed"] == (1 if fault == "lost" else 2)
    assert report["work"]["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "bound"])
def test_should_refuse_invalid_hidden_result_without_reducing_it(scripted, monkeypatch, fault):
    value = hidden()
    if fault == "shape":
        value = value[:-1]
    elif fault == "dtype":
        value = value.astype(np.float32)
    elif fault == "nan":
        value[0, 0] = np.nan
    else:
        value[0, 0] = 2
    monkeypatch.setattr(health, "read_native_activations", lambda *a: value)
    source = public.PublicCueRecurrentLearner()
    with pytest.raises(health.ActivationHealthFailed) as stopped:
        health.read_checkpoint_health(
            source,
            source.snapshot_state(),
            updates=0,
            sources=sources(),
            budget=budget(),
            check_boundary=lambda: None,
        )
    assert stopped.value.report["work"]["svd_calls_attempted"] == 0


def test_should_preserve_default_protocol_bytes_and_describe_extra_costs():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/public-cue-health-20261025/pilot_before.py"
    )
    spec = importlib.util.spec_from_file_location("before_public_health", path)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for options in (
        {},
        dict(typed_input=True),
        dict(typed_input=True, recurring_training=True),
        dict(typed_input=True, recurring_training=True, recurrent_replay="recent"),
        dict(typed_input=True, recurring_training=True, recurrent_replay="coverage"),
        dict(typed_input=True, recurring_training=True, class_balanced_training=True),
    ):
        assert pickle.dumps(pilot.describe_protocol(**options)) == pickle.dumps(
            old.describe_protocol(**options)
        )
    descriptor = pilot.describe_protocol(
        typed_input=True, recurring_training=True, activation_health=True
    )
    assert descriptor["planned_work"]["activation_forward_calls"] == 270
    assert descriptor["planned_work"]["activation_svd_calls"] == 30
    assert descriptor["limits"]["readout_seconds"] == 200


@pytest.mark.parametrize("flag", [0, 1, None, "health"])
def test_should_refuse_invalid_option_before_io_or_native(monkeypatch, flag):
    monkeypatch.setattr(pilot, "require_torch", lambda: pytest.fail("native reached"))
    with pytest.raises(ValueError):
        pilot.run_pilot(activation_health=flag)


def test_should_describe_cli_without_data_or_native_and_refuse_other_modes(monkeypatch, capsys):
    monkeypatch.setattr(pilot, "require_torch", lambda: pytest.fail("native reached"))
    assert (
        cli.main(["--describe", "--public-cue", "--recurring-training", "--activation-health"]) == 0
    )
    assert "activation_forward_calls" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--describe", "--activation-health"])


@pytest.fixture
def wiring(scripted, monkeypatch):
    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        pilot,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    events: list[Any] = []
    captured: dict[str, Any] = {}

    def train(row, packets, memory, deadline, check, *, learner_factory, **options):
        source = learner_factory(public.METHOD)
        events.append(("fit", 128))
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
        captured.update(arm=arm)
        return arm

    def observe(source, saved, **options):
        options["check_boundary"]()
        point = options["updates"]
        events.append(("read", point))
        result = readout(point)
        if options.get("class_balanced_training") and point in (64, 96):
            result["episodes"] = readout(32)["episodes"]
        return result

    original = health.read_checkpoint_health

    def observe_health(*a, **k):
        events.append(("health", k["updates"]))
        captured["budget"] = k["budget"]
        return original(*a, **k)

    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", observe)
    monkeypatch.setattr(health, "read_checkpoint_health", observe_health)
    return events, captured


@pytest.mark.parametrize("weighted", [False, True])
def test_should_finish_fitting_before_health_and_keep_original_budget_and_readouts(
    wiring, weighted, monkeypatch
):
    from src.adapters import torch_train_weighted_learner

    monkeypatch.setattr(torch_train_weighted_learner, "_same_state", _same_value)
    events, _ = wiring
    result = pilot.run_pilot(
        tuple(p.episode for p in sources()),
        recurring_training=True,
        class_balanced_training=weighted,
        activation_health=True,
    )
    assert result["status"] == "complete", result["reason"]
    assert events == [("fit", 128)] + [
        v for point in (0, 32, 64, 96, 128) for v in (("health", point), ("read", point))
    ]
    assert len(result["activation_health"]) == 5 and result["activation_health_failure"] is None
    assert result["readout_budget"]["updates_completed"] == 0
    assert result["numpy_rng_unchanged"] and result["torch_rng_unchanged"]


def test_should_keep_failed_health_prefix_before_any_following_game(wiring, monkeypatch):
    events, _ = wiring
    original = health.read_native_activations
    calls = []

    def fail(*a):
        calls.append(1)
        if len(calls) == 8:
            raise RuntimeError("lost second checkpoint hidden return")
        return original(*a)

    monkeypatch.setattr(health, "read_native_activations", fail)
    result = pilot.run_pilot(
        tuple(p.episode for p in sources()), recurring_training=True, activation_health=True
    )
    assert result["status"] == "incomplete"
    assert events == [("fit", 128), ("health", 0), ("read", 0), ("health", 32)]
    assert len(result["activation_health"]) == len(result["evaluations"]) == 1
    assert result["activation_health_failure"]["work"]["activation_reads_attempted"] == 2


def test_should_refuse_changed_budget_before_any_native_validation(scripted, monkeypatch):
    source = public.PublicCueRecurrentLearner()
    saved, limits = source.snapshot_state(), budget()
    object.__setattr__(limits.budget, "max_wall_seconds", 201)
    monkeypatch.setattr(public, "copy_snapshot", lambda *a: pytest.fail("native reached"))
    with pytest.raises(ValueError, match="original200"):
        health.read_checkpoint_health(
            source, saved, updates=0, sources=sources(), budget=limits, check_boundary=lambda: None
        )


@pytest.mark.parametrize("fault", ["source", "checkpoint"])
def test_should_refuse_mutation_in_last_boundary_after_snapshot(scripted, monkeypatch, fault):
    source = public.PublicCueRecurrentLearner()
    saved = source.snapshot_state()
    reduction, late = [], []
    original = health.diagnose_activations

    def reduce(values):
        result = original(values)
        reduction.append(1)
        return result

    def boundary():
        if len(reduction) == 6:
            late.append(1)
            # Two boundary calls per boundary(); after the final SVD, calls7/8
            # are the second boundary of the final check(), after its snapshot.
            if len(late) == 8:
                target = source._state if fault == "source" else saved
                target.heads[0].parameters["marker"][0] += 1

    monkeypatch.setattr(health, "diagnose_activations", reduce)
    with pytest.raises(health.ActivationHealthFailed):
        health.read_checkpoint_health(
            source, saved, updates=0, sources=sources(), budget=budget(), check_boundary=boundary
        )
    assert len(late) == 8 and len(scripted) == 6


def test_should_keep_svd_failure_cost_and_not_publish_unverified_metrics(scripted, monkeypatch):
    source = public.PublicCueRecurrentLearner()

    def fail(values):
        raise RuntimeError("lost SVD return")

    monkeypatch.setattr(health, "diagnose_activations", fail)
    with pytest.raises(health.ActivationHealthFailed) as stopped:
        health.read_checkpoint_health(
            source,
            source.snapshot_state(),
            updates=0,
            sources=sources(),
            budget=budget(),
            check_boundary=lambda: None,
        )
    work = stopped.value.report["work"]
    assert work["activation_reads_completed"] == work["svd_calls_attempted"] == 1
    assert work["svd_calls_completed"] == 0 and stopped.value.report["scores"] == []


def test_should_refuse_changed_preceding_health_instead_of_resealing_it(wiring, monkeypatch):
    events, _ = wiring
    observe = pilot.evaluate_public_cue_recurrent_checkpoint

    def corrupt(source, saved, **options):
        # The pilot's original guard owns this list; a callback cannot replace
        # the prior health evidence and then obtain the next checkpoint's work.
        guard = options["check_boundary"]
        guard_cells = dict(
            zip(guard.__code__.co_freevars, (c.cell_contents for c in guard.__closure__))
        )
        boundary = guard_cells["boundary"]
        names = dict(
            zip(boundary.__code__.co_freevars, (c.cell_contents for c in boundary.__closure__))
        )
        names["health_records"][0]["scores"][0]["head"] = 99
        return observe(source, saved, **options)

    monkeypatch.setattr(pilot, "evaluate_public_cue_recurrent_checkpoint", corrupt)
    result = pilot.run_pilot(
        tuple(p.episode for p in sources()), recurring_training=True, activation_health=True
    )
    assert result["status"] == "incomplete"
    assert events == [("fit", 128), ("health", 0)]
    assert result["activation_health"][0]["scores"][0]["head"] == 0
