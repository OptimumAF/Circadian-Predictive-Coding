"""Bounded causal row reuse, original delayed admission and honest exposure costs."""

from dataclasses import replace
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_train_catalogue as catalogue
from src.adapters import minigrid_context_linear_learner as port
from src.app import coupled_attribution
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)


def episode(identity, cue=2, *, mark=1, action=2, context="metadata", rows=9, role="train"):
    frame = [-0.8, -1.0, -1.0] * 49 + [-1.0]
    frame[:3] = [0.2, 2 * cue / 5 - 1, -1.0]
    frame[3] = 2 * mark / 10 - 1
    history = tuple(frame + frame)
    return DemonstrationEpisode(
        identity,
        context,
        role,
        tuple(
            DemonstrationTransition(
                i, history, action, 0.5 if i == rows - 1 else 0.0, history, i == rows - 1, 7
            )
            for i in range(rows)
        ),
        7,
    )


def sources():
    return tuple(
        episode(f"minigrid-source-{context}-{name}", cue, mark=mark, action=action, context=context)
        for context, mark, action in (("A", 1, 2), ("B", 3, 5))
        for name, cue in (("blue", 2), ("red", 0))
    )


def schedule():
    selected = sources()
    return tuple(
        replace(selected[index], episode_id=f"arrival-{i}")
        for i, index in enumerate((0, 1, 2, 3, 0, 1))
    )


def test_should_jointly_reuse_only_previous_arrived_same_cue_rows_with_real_costs():
    packets = schedule()
    before, rng = pickle.dumps(packets), pickle.dumps(np.random.get_state())
    plan = catalogue.compose_arrived_cue_catalogue(packets)
    assert [len(b.features) for b in plan.training] == [9, 9, 18, 18, 18, 18]
    assert (plan.arrival_row_exposures, plan.replay_row_exposures, plan.fit_row_exposures) == (
        54,
        36,
        90,
    )
    assert plan.input_array_bytes == 90 * 297 * 8 == 213840
    assert plan.peak_joint_template_array_bytes == 18 * 297 * 8 == 42768
    assert plan.peak_bank_rows == 18
    assert plan.peak_bank_observation_target_float64_equivalent_bytes == 18 * 593 * 8 == 85392
    old_indices = (None, None, 0, 1, 2, 3)
    for i, (batch, wake, old) in enumerate(
        zip(plan.training, plan.wakes, old_indices, strict=True)
    ):
        assert batch.episode is None and batch.role == "train" and batch.num_classes == 7
        assert wake.observed_at == i and wake.arrived_at == i + 1 and wake.current_rows == 9
        assert wake.current_episode_id == packets[i].episode_id
        assert wake.bank_rows_after <= 18 and len(wake.bank_episode_ids_after) <= 2
        selected = (packets[i],) if old is None else (packets[old], packets[i])
        xys = [episode_arrays(e) for e in selected]
        np.testing.assert_array_equal(batch.features, np.concatenate([x for x, y in xys]))
        np.testing.assert_array_equal(batch.targets, np.concatenate([y for x, y in xys]))
        assert [span.episode_id for span in wake.source_spans] == [e.episode_id for e in selected]
        if old is None:
            assert wake.retained_episode_id is wake.retained_arrived_at is None
        else:
            assert wake.retained_episode_id == packets[old].episode_id
            assert wake.retained_arrived_at == old + 1 <= wake.observed_at
    ids = [key for batch in plan.training for key in batch.row_ids]
    assert len(ids) == len(set(ids)) == 90
    assert pickle.dumps(packets) == before and pickle.dumps(np.random.get_state()) == rng


def test_should_own_disjoint_transport_arrays_and_preserve_original_sources():
    packets = schedule()
    before = pickle.dumps(packets)
    plan = catalogue.compose_arrived_cue_catalogue(packets)
    arrays = [a for batch in plan.training for a in (batch.features, batch.targets)]
    assert all(not np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    original_last = plan.training[-1].features.copy()
    plan.training[0].features.fill(0)
    plan.training[0].targets.fill(0)
    np.testing.assert_array_equal(plan.training[-1].features, original_last)
    assert pickle.dumps(packets) == before


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "empty",
        "seven",
        "final",
        "probe",
        "short",
        "initial",
        "fractional",
        "no_cue",
        "duplicate",
    ],
)
def test_should_refuse_bad_role_schema_or_history_before_composition(monkeypatch, fault):
    packet = episode("packet")
    packets: Any = (packet,)
    if fault == "list":
        packets = [packet]
    elif fault == "empty":
        packets = ()
    elif fault == "seven":
        packets = tuple(replace(packet, episode_id=f"packet-{i}") for i in range(7))
    elif fault in ("final", "probe"):
        packets = (replace(packet, role="final_test" if fault == "final" else "probe_train"),)
    elif fault == "short":
        packets = (episode("packet", rows=8),)
    elif fault == "duplicate":
        packets = (packet, packet)
    else:
        row = packet.transitions[0]
        values = list(row.observation)
        if fault == "initial":
            values[3] = -0.4
        elif fault == "fractional":
            values[0] = values[148] = 0.123
        else:
            values[0] = values[148] = -0.8
        packets = (
            replace(
                packet,
                transitions=tuple(
                    replace(t, observation=tuple(values), next_observation=tuple(values))
                    for t in packet.transitions
                ),
            ),
        )
    monkeypatch.setattr(
        catalogue, "compose_catalogue_training", lambda *a: pytest.fail("composition reached")
    )
    with pytest.raises(ValueError):
        catalogue.compose_arrived_cue_catalogue(packets)


def test_should_refuse_source_mutation_before_following_composition(monkeypatch):
    packets = schedule()
    original = catalogue.compose_catalogue_training
    calls = []

    def compose(*args):
        calls.append(1)
        result = original(*args)
        object.__setattr__(packets[-1].transitions[-1], "action", 1)
        return result

    monkeypatch.setattr(catalogue, "compose_catalogue_training", compose)
    with pytest.raises(ValueError, match="changed"):
        catalogue.compose_arrived_cue_catalogue(packets)
    assert calls == [1]


def test_should_route_by_public_cue_rather_than_context_or_episode_names():
    packets = (episode("red-name", context="red"), episode("blue-name", mark=3, context="blue"))
    plan = catalogue.compose_arrived_cue_catalogue(packets)
    assert [wake.cue for wake in plan.wakes] == ["blue", "blue"]
    assert plan.wakes[1].retained_episode_id == "red-name" and len(plan.training[1].features) == 18


def test_should_keep_only_one_previous_episode_for_a_cue_without_history_growth():
    packets = tuple(episode(f"packet-{i}") for i in range(6))
    plan = catalogue.compose_arrived_cue_catalogue(packets)
    assert plan.fit_row_exposures == 99 and plan.replay_row_exposures == 45
    assert plan.peak_bank_rows == 9 and max(len(b.features) for b in plan.training) == 18
    assert [w.retained_episode_id for w in plan.wakes] == [None, *[f"packet-{i}" for i in range(5)]]


def test_should_train_only_arrived_joint_rows_through_original_delayed_receipts(monkeypatch):
    packets = (episode("first", action=2), episode("second", mark=3, action=5))
    plan = catalogue.compose_arrived_cue_catalogue(packets)
    source = port.ContextLinearLearner()
    original = port.ContextLinearLearner.train_batch
    calls = []

    def train(self, x, y):
        calls.append((x.copy(), y.copy()))
        return original(self, x, y)

    monkeypatch.setattr(port.ContextLinearLearner, "train_batch", train)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    stream = coupled_attribution.AttributionStream(
        source,
        plan.training,
        (),
        checkpoint_updates=(0, 1, 2),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    assert calls == [] and source.completed_updates == 0
    stream.run_until(1)
    assert len(calls) == 1 and calls[0][1].shape == (9, 1) and np.all(calls[0][1] == 2)
    stream.run_until(2)
    final = stream.candidate_snapshot()
    result = stream.finish()
    assert (
        final.completed_updates == 2
        and final.training_examples == 27
        and final.heads[0].fit_rows == 18
    )
    assert [len(x) for x, y in calls] == [9, 18]
    assert calls[1][1][:9].tolist() == [[2.0]] * 9 and calls[1][1][9:].tolist() == [[5.0]] * 9
    assert result["training_examples"] == 27 and result["score_forward_calls"] == 0
    for i, receipt in enumerate(result["applied_updates"]):
        assert (
            receipt["observed_at"] == i and receipt["arrived_at"] == receipt["applied_at"] == i + 1
        )
    assert source.completed_updates == source.training_examples == 0
    assert stream._runtime.actor.snapshot_state().state.heads == (None, None)
    reader = source.fork()
    reader.restore_state(final)
    for packet in packets:
        x, y = episode_arrays(packet)
        assert reader.predict(x).argmax(axis=1).tolist() == y[:, 0].astype(int).tolist()


def test_should_preserve_baseline_and_build_optin_plan_without_running_models(monkeypatch):
    observed = []

    class StopBeforeFit:
        def __init__(self, source, training, heldout, **options):
            observed.append((training, heldout, options))
            raise ValueError("deliberate pre-fit stop")

    monkeypatch.setattr(coupled_attribution, "AttributionStream", StopBeforeFit)
    baseline = cli.run_context_linear_stream(sources())
    retained = cli.run_context_linear_stream(sources(), retain_arrived=True)
    assert (
        baseline["definition"] == cli.describe_context_linear_stream()["definition"]
        and "catalogue" not in baseline
    )
    assert [len(b.features) for b in observed[0][0]] == [9] * 6
    assert all(b.episode is not None for b in observed[0][0])
    assert [len(b.features) for b in observed[1][0]] == [9, 9, 18, 18, 18, 18]
    assert all(b.episode is None for b in observed[1][0])
    assert observed[1][1] == () and observed[1][2]["label_delay"] == 1
    assert retained["catalogue"]["fit_row_exposures"] == 90
    assert retained["catalogue"]["replay_row_exposures"] == 36
    assert retained["catalogue"]["committed_rows"] is None


def test_should_keep_paid_prefix_and_unknown_replay_attempt_on_third_fit_failure(monkeypatch):
    original = port.fit_categorical_linear
    calls = []

    def fit(x, y, classes):
        calls.append(len(x))
        if len(calls) == 3:
            raise RuntimeError("deliberate third joint-fit failure")
        return original(x, y, classes)

    monkeypatch.setattr(port, "fit_categorical_linear", fit)
    result = cli.run_context_linear_stream(sources(), retain_arrived=True)
    assert calls == [9, 9, 18] and result["status"] == "failed"
    assert len(result["committed_receipts"]) == result["budget"]["updates_completed"] == 2
    assert result["observed_checkpoints"] == [0, 2] and result["score_calls_attempted"] == 0
    assert result["catalogue"]["committed_rows"] == dict(arrivals=18, replay=0, native=18)
    assert result["catalogue"]["failed_attempt_replay_rows_upper_bound"] == 9
    assert result["catalogue"]["failure_work_may_be_unobserved"] is True
    assert result["failure_work"]["work"]["fit_rows_attempted"] == 18
    assert result["failure_work"]["work"]["failure_work_may_be_unobserved"] is True
    assert result["source_unchanged"] is result["actor_unchanged"] is None


@pytest.mark.parametrize(
    "args",
    [
        ["--context-linear-catalogue", "--describe"],
        ["--context-linear-catalogue", "--context-linear-stream"],
        ["--context-linear-catalogue", "--fit-updates", "1"],
    ],
)
def test_should_describe_without_io_and_refuse_conflicting_options(monkeypatch, capsys, args):
    monkeypatch.setattr("sys.argv", ["control", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("TRAIN opened"))
    if "--describe" in args:
        cli.main()
        result = json.loads(capsys.readouterr().out)
        assert result["fit_rows"] == 90 and result["catalogue_replay_row_exposures"] == 36
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_route_optin_cli_and_keep_partial_exclusive_output(monkeypatch, tmp_path):
    destination = tmp_path / "actual.json"
    marker = (object(),)
    monkeypatch.setattr(
        "sys.argv",
        [
            "control",
            "--context-linear-catalogue",
            "--source",
            "source",
            "--output",
            str(destination),
        ],
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: marker)

    def run(episodes, *, retain_arrived=False):
        assert episodes is marker and retain_arrived is True
        return dict(status="failed", catalogue=dict(committed_rows=dict(native=18)))

    monkeypatch.setattr(cli, "run_context_linear_stream", run)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert (
        json.loads(destination.read_text(encoding="utf8"))["catalogue"]["committed_rows"]["native"]
        == 18
    )
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_compose_original_train_source_rows_without_future_c_train():
    path = (
        Path(__file__).resolve().parents[1] / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    )
    if not path.is_file():
        pytest.skip("accepted local TRAIN source unavailable")
    selected = cli.read_training_sources(path)[:4]
    packets = tuple(
        replace(selected[j], episode_id=f"actual-arrival-{i}")
        for i, j in enumerate((0, 1, 2, 3, 0, 1))
    )
    plan = catalogue.compose_arrived_cue_catalogue(packets)
    assert plan.fit_row_exposures == 90 and plan.peak_bank_rows == 18
    assert [len(b.features) for b in plan.training] == [9, 9, 18, 18, 18, 18]
    for batch, wake in zip(plan.training, plan.wakes, strict=True):
        assert all(span.context in ("A", "B") for span in wake.source_spans)
        assert batch.context == "train-catalogue"


def test_should_refuse_counter_mismatch_before_accepting_checkpoint(monkeypatch):
    from types import SimpleNamespace

    class InvalidCounterStream:
        def __init__(self, *args, **options):
            options["observe_checkpoint"](
                2, SimpleNamespace(completed_updates=2, training_examples=19), None
            )
            pytest.fail("incompatible counters accepted")

    monkeypatch.setattr(coupled_attribution, "AttributionStream", InvalidCounterStream)
    result = cli.run_context_linear_stream(sources(), retain_arrived=True)
    assert result["status"] == "failed" and "counters disagree" in result["reason"]
    assert result["observed_checkpoints"] == [] and result["committed_receipts"] is None
    assert result["catalogue"]["committed_rows"] is None
    assert result["catalogue"]["failure_work_may_be_unobserved"] is True


def test_should_refuse_bad_public_cue_order_before_creating_native_owner(monkeypatch):
    bad = tuple(
        episode(
            e.episode_id,
            cue=2,
            mark=1 if e.context == "A" else 3,
            action=2 if e.context == "A" else 5,
            context=e.context,
        )
        for e in sources()
    )
    monkeypatch.setattr(port, "ContextLinearLearner", lambda: pytest.fail("native owner reached"))
    with pytest.raises(ValueError, match="alternating public"):
        cli.run_context_linear_stream(bad, retain_arrived=True)
