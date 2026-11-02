"""Scripted readout ports and fake games only; no native/model or solver work."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import recurrent_readout_example as example
from src.adapters import minigrid_readout_probe as adapter
from src.adapters.minigrid_learning_pilot import _batch
from src.adapters.minigrid_learnability import _packet
from src.adapters.minigrid_return_fit import return_training_sources
from src.app import fresh_task_probe as core
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_recurrent_readout import scripted as scripted, facade_setup, nine, fake_loop as fake_loop

ORIGINAL_GAME_KERNEL = adapter._run_checkpoint


def f_sources():
    groups = []
    for role, target in (("probe_train", 6), ("probe_test", 5)):
        packets = []
        for cue in ("blue", "red"):
            episode = nine(cue, "F-" + cue)
            episode = replace(
                episode,
                role=role,
                episode_id=f"readout-{role}-{cue}",
                transitions=tuple(replace(t, action=target) for t in episode.transitions),
            )
            packets.append(_packet(episode))
        groups.append(tuple(packets))
    return tuple(groups)


def cohorts():
    old = tuple(_batch(nine(cue)) for cue in ("blue", "red"))
    train, test = f_sources()
    return tuple(
        tuple(adapter._arrays(p) for p in group)
        for group in (train, test, old, return_training_sources(old))
    )


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_replay_examples=0, max_wall_seconds=60),
        lambda: 0.0,
    )


def core_run(source, state, data=None, **options):
    training, evaluation, monitors, returning = cohorts() if data is None else data
    return core.run_fresh_task_probe(
        source,
        state,
        training,
        evaluation,
        budget=budget(),
        checkpoint_updates=(0, 1, 2),
        timer=lambda: 0.0,
        training_readouts=monitors,
        return_training=returning,
        return_budget=budget(),
        short_readout_return=True,
        decision_diagnostics=True,
        **options,
    )


def test_short_readout_uses_only_f_train_then_explicit_a_return_on_one_private_fork(scripted):
    owner, state, _ = facade_setup()
    before = owner.snapshot_state()
    data = cohorts()
    result = core_run(owner, state, data)
    assert result["status"] == result["return_phase"]["status"] == "complete"
    for phase in (result, result["return_phase"]):
        assert phase["completed_updates"] == 2 and phase["completed_training_examples"] == 18
        assert [p["updates"] for p in phase["checkpoints"]] == [0, 1, 2]
        assert phase["score_forward_calls_completed"] == 6
        assert phase["training_readouts"]["score_forward_calls_completed"] == 6
        assert phase["training_readouts"]["completed_rows"] == 54
    assert result["return_phase"]["schedule"] == [0, 1]
    assert len(scripted.fits) == 6
    assert all(np.array_equal(y, np.full((9, 1), 6.0)) for _, y, _ in scripted.fits[2:4])
    assert all(np.array_equal(scripted.fits[4 + i][1], data[2][i].targets) for i in range(2))
    assert not any(np.all(y == 5) for _, y, _ in scripted.fits)
    assert _same_value(owner.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault",
    [
        "flag",
        "both",
        "width",
        "rows",
        "classes",
        "role",
        "context",
        "episode",
        "alias",
        "values",
        "ids",
        "missing_return",
        "extra",
        "rehearsal",
        "callback",
    ],
)
def test_short_readout_refuses_invalid_contract_before_copies_or_source(monkeypatch, fault):
    training, evaluation, monitors, returning = cohorts()
    options: dict[str, Any] = dict(short_readout_return=True)
    if fault == "flag":
        options["short_readout_return"] = 1
    elif fault == "both":
        options["short_affine_return"] = True
    elif fault in ("width", "rows", "classes", "role", "context", "episode"):
        packet = training[0]
        changes = dict(
            width=dict(features=np.zeros((9, 296))),
            rows=dict(
                features=packet.features[:-1].copy(),
                targets=packet.targets[:-1].copy(),
                row_ids=packet.row_ids[:-1],
            ),
            classes=dict(num_classes=8),
            role=dict(role="final_test"),
            context=dict(context="selected-other-task"),
            episode=dict(episode=f_sources()[0][0].episode),
        )
        training = (replace(packet, **changes[fault]), training[1])
    elif fault == "alias":
        returning = (replace(returning[0], features=monitors[0].features), returning[1])
    elif fault == "values":
        returning[0].targets[0, 0] = 6
    elif fault == "ids":
        returning = (replace(returning[0], row_ids=monitors[0].row_ids), returning[1])
    elif fault == "missing_return":
        returning = ()
    elif fault == "extra":
        options["extra_f_exposure"] = True
    elif fault == "rehearsal":
        options["rehearsal_factory"] = lambda *a: None
    else:
        options["update_callback"] = lambda *a: None
    monkeypatch.setattr(core, "_copy_batch", lambda *a: pytest.fail("copies reached"))
    with pytest.raises(ValueError):
        missing_source: Any = None
        core.run_fresh_task_probe(
            missing_source,
            None,
            training,
            evaluation,
            budget=budget(),
            checkpoint_updates=(0, 1, 2),
            training_readouts=monitors,
            return_training=returning,
            return_budget=budget(),
            **options,
        )


@pytest.fixture
def fake_ports(scripted, monkeypatch):
    seen: dict[str, list[dict[str, Any]]] = dict(games=[])

    def collect(guard, records, **options):
        assert options == dict(distinct_probe=True)
        guard()
        for role in ("probe_train", "probe_test"):
            for cue in ("blue", "red"):
                records.append(
                    dict(status="complete", closed=True, environment_steps=9, role=role, cue=cue)
                )
        guard()
        seen["collection"] = records
        training, evaluation = f_sources()
        repeated = []
        for i in range(16):
            original = training[i % 2].episode
            assert original is not None
            repeated.append(_packet(replace(original, episode_id=f"readout-train-copy-{i}")))
        return tuple(repeated), evaluation

    def games(learner, state, **options):
        assert options["sources"] == ()
        assert (
            options["budget"].budget.max_training_updates
            == options["budget"].budget.max_replay_examples
            == 0
        )
        assert [item[2] for item in options["live_plan"]] == [
            "probe_test",
            "probe_test",
            "train",
            "train",
        ]
        options["check_boundary"]()
        options["extra_check"]()
        options["qualified_boundary"]()
        row = dict(
            status="complete",
            state_unchanged=True,
            episodes=[
                dict(
                    status="complete",
                    source_role=role,
                    success=False,
                    environment=dict(closed=True),
                )
                for role in ("probe_test", "probe_test", "train", "train")
            ],
            work=dict(calls=0),
        )
        seen["games"].append(row)
        return row

    monkeypatch.setattr(adapter, "_collect_probe_sources", collect)
    monkeypatch.setattr(adapter, "_run_checkpoint", games)
    return seen


def memory():
    from src.shared.process_memory import ProcessRssSampler

    sampler = ProcessRssSampler()
    sampler.sample()
    return sampler


def test_adapter_routes_both_endpoints_and_retains_all_false_game_outcomes(scripted, fake_ports):
    owner, state, sources = facade_setup()
    result = adapter.run_readout_probe(
        owner, state, sources, check_boundary=lambda: None, memory=memory()
    )
    assert result["status"] == "complete", result["reason"]
    assert result["source_unchanged"] is True
    assert [row["updates"] for row in result["native_readouts"]] == [4, 6]
    assert len(fake_ports["games"]) == 2
    assert all(
        not game["success"]
        for row in result["native_readouts"]
        for game in row["games"]["episodes"]
    )
    assert [row["updates"] for row in result["budgets"]] == [2, 2, 0]
    assert all(row["replay"] == 0 for row in result["budgets"])
    assert owner.completed_updates == 2 and _same_value(owner.snapshot_state(), state)


@pytest.mark.parametrize(
    "fault",
    [
        "lost_fit",
        "changed_source",
        "replaced_games",
        "replaced_source_array",
        "replaced_collection",
    ],
)
def test_adapter_keeps_paid_prefix_and_refuses_more_work_after_boundary_failure(
    scripted, fake_ports, fault
):
    owner, state, sources = facade_setup()
    original_fit = scripted.fit_port

    def fit(hidden, targets, classes):
        if fault == "lost_fit" and len(scripted.fits) == 2:
            raise RuntimeError("lost F fit return")
        result = original_fit(hidden, targets, classes)
        if fault == "changed_source" and len(scripted.fits) == 3:
            owner._state.native.heads[0].parameters["weight_input"][0, 0] += 1
        if fault == "replaced_games" and len(scripted.fits) == 5:
            row = fake_ports["games"][0]
            row["work"] = dict(row["work"])
        if fault == "replaced_source_array" and len(scripted.fits) == 3:
            object.__setattr__(sources[0], "features", sources[0].features.copy())
        if fault == "replaced_collection" and len(scripted.fits) == 3:
            fake_ports["collection"][0] = dict(fake_ports["collection"][0])
        return result

    scripted.fit_override = fit
    result = adapter.run_readout_probe(
        owner, state, sources, check_boundary=lambda: None, memory=memory()
    )
    assert result["status"] == "incomplete" and result["source_unchanged"] is None
    assert (
        len(scripted.fits)
        == {
            "lost_fit": 2,
            "changed_source": 3,
            "replaced_games": 5,
            "replaced_source_array": 3,
            "replaced_collection": 3,
        }[fault]
    )
    assert len(fake_ports["games"]) == (1 if fault == "replaced_games" else 0)
    if fault == "lost_fit":
        work = result["probe"]["failed_native_work"]["work"]
        assert work["fit_calls_attempted"] == 1 and work["fit_calls_returned"] == 0
    elif fault == "replaced_games":
        assert result["probe"]["return_phase"]["completed_updates"] == 1


@pytest.mark.parametrize(
    "baseline_success,tamper",
    [(False, None), (True, None), (False, "value"), (True, "equal"), (True, "source_final")],
)
def test_firstparty_real_stream_probe_composition_preserves_source_and_checks_prior_success(
    scripted, fake_ports, monkeypatch, baseline_success, tamper
):
    monkeypatch.setattr(example, "_require_minigrid", lambda: None)
    monkeypatch.setattr(
        example,
        "_episodes",
        lambda: tuple(
            nine(cue, context) for context in ("A", "B", "C_train") for cue in ("blue", "red")
        ),
    )
    returned: list[dict[str, Any]] = []
    changed: list[bool] = []

    def compare(left, right):
        result = _same_value(left, right)
        if returned and tamper in ("value", "equal") and not changed:
            changed.append(True)
            if tamper == "value":
                for row in returned[0]["episodes"]:
                    row["success"] = True
            else:
                returned[0]["episodes"] = deepcopy(returned[0]["episodes"])
        return result

    monkeypatch.setattr(example, "_same_state", compare)

    class Rng:
        def clone(self):
            return self

    rng = Rng()
    runtime = SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b)
    monkeypatch.setattr(example, "require_torch", lambda: runtime)

    def baseline(*args, **options):
        options["check_boundary"]()
        if tamper == "source_final":
            parent_guard = next(
                c.cell_contents
                for c in options["check_boundary"].__closure__
                if callable(c.cell_contents) and c.cell_contents.__name__ == "guard"
            )
            original_source = next(
                c.cell_contents
                for c in parent_guard.__closure__
                if type(c.cell_contents) is example.RecurrentReadoutLearner
            )
            report = next(
                c.cell_contents
                for c in parent_guard.__closure__
                if type(c.cell_contents) is dict and "training" in c.cell_contents
            )

            def equal(a, b):
                if report["probe"] is not None and not changed:
                    changed.append(True)
                    original_source._state = deepcopy(original_source._state)
                return a is b

            runtime.equal = equal
        row = dict(
            status="complete",
            state_unchanged=True,
            episodes=[
                dict(status="complete", success=baseline_success, environment=dict(closed=True))
                for _ in range(2)
            ],
        )
        returned.append(row)
        return row

    monkeypatch.setattr(example, "evaluate_recurrent_readout_checkpoint", baseline)
    result = example.run_example(fresh_probe=True)
    assert result["training"]["completed_updates"] == 2
    if tamper == "source_final":
        assert changed and result["status"] == "incomplete" and result["preservation"] is None
        assert result["probe"]["status"] == "complete" and len(scripted.fits) == 6
    elif tamper is not None:
        assert changed and result["status"] == "incomplete" and result["probe"] is None
        assert len(scripted.fits) == 2 and not fake_ports["games"]
        assert all(e["success"] is baseline_success for e in result["games"]["episodes"])
    elif baseline_success:
        assert result["status"] == "complete", result["reason"]
        assert result["probe"]["status"] == "complete" and len(scripted.fits) == 6
        assert all(result["preservation"].values())
    else:
        assert result["status"] == "incomplete" and result["probe"] is None
        assert len(scripted.fits) == 2 and not fake_ports["games"]


def test_cli_defaults_and_exclusive_probe_description_do_no_work(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(example, "run_example", lambda **kw: pytest.fail("model work"))
    assert example.describe()["probe_available"] is False
    monkeypatch.setattr(sys, "argv", ["readout", "--fresh-probe", "--describe"])
    example.main()
    assert '"probe_available": true' in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["readout", "--fresh-probe", "--recurring", "--describe"])
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2


def test_old_short_affine_flag_still_refuses_raw148_before_any_copy(monkeypatch):
    training, evaluation, monitors, returning = cohorts()
    monkeypatch.setattr(core, "_copy_batch", lambda *a: pytest.fail("copies reached"))
    missing_source: Any = None
    with pytest.raises(ValueError, match="short affine"):
        core.run_fresh_task_probe(
            missing_source,
            None,
            training,
            evaluation,
            budget=budget(),
            checkpoint_updates=(0, 1, 2),
            training_readouts=monitors,
            return_training=returning,
            return_budget=budget(),
            short_affine_return=True,
        )


def test_unavailable_budget_capture_cannot_report_complete_after_paid_work(
    scripted, fake_ports, monkeypatch
):
    owner, state, sources = facade_setup()

    def unavailable(*args):
        raise RuntimeError("lost budget capture")

    monkeypatch.setattr(adapter, "asdict", unavailable)
    result = adapter.run_readout_probe(
        owner, state, sources, check_boundary=lambda: None, memory=memory()
    )
    assert result["status"] == "incomplete" and result["budgets"] is None
    assert "lost budget capture" in result["budget_capture_error"]
    assert (
        result["probe"]["completed_updates"]
        == result["probe"]["return_phase"]["completed_updates"]
        == 2
    )
    assert len(scripted.fits) == 6 and owner.completed_updates == 2


@pytest.mark.parametrize(
    "option,expected", [(None, {}), ("--fresh-probe", {"fresh_probe": True}), ("--recurring", None)]
)
def test_public_cli_dispatches_each_selected_workflow_exactly_once(
    monkeypatch, tmp_path, option, expected
):
    import sys

    seen: list[dict[str, Any] | None] = []

    def ordinary(**options):
        seen.append(options)
        return dict(status="complete", reason=None)

    def recurring():
        seen.append(None)
        return dict(status="complete", reason=None)

    monkeypatch.setattr(example, "run_example", ordinary)
    monkeypatch.setattr(example, "run_recurring_example", recurring)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["readout", "--output", str(output)] + ([] if option is None else [option])
    )
    with pytest.raises(SystemExit) as result:
        example.main()
    assert result.value.code == 0 and seen == [expected]
    assert output.is_file()


def test_actual_game_kernel_reads_both_private_endpoints_and_closes_all_fake_owners(
    scripted, fake_ports, fake_loop, monkeypatch
):
    monkeypatch.setattr(adapter, "_run_checkpoint", ORIGINAL_GAME_KERNEL)
    owner, state, sources = facade_setup()
    result = adapter.run_readout_probe(
        owner, state, sources, check_boundary=lambda: None, memory=memory()
    )
    assert result["status"] == "complete", result["reason"]
    assert [row["updates"] for row in result["native_readouts"]] == [4, 6]
    for row in result["native_readouts"]:
        games = row["games"]
        assert games["state_unchanged"] is True and len(games["episodes"]) == 4
        assert [e["source_role"] for e in games["episodes"]] == [
            "probe_test",
            "probe_test",
            "train",
            "train",
        ]
        assert all(
            e["status"] == "complete" and e["environment"]["closed"] for e in games["episodes"]
        )
    assert len(fake_loop["environments"]) == 8 and len(fake_loop["steps"]) == 16
    assert all(e.closed for e in fake_loop["environments"])
    assert owner.completed_updates == 2 and owner.training_examples == 18


def test_failed_memory_attachment_retains_the_first_original_budget_birth(scripted, fake_ports):
    from src.shared.process_memory import ProcessRssSampler

    owner, state, sources = facade_setup()
    result = adapter.run_readout_probe(
        owner, state, sources, check_boundary=lambda: None, memory=ProcessRssSampler()
    )
    assert result["status"] == "incomplete" and result["probe"] is None
    assert len(result["budgets"]) == 1
    first = result["budgets"][0]
    assert first["birth"] <= first["last_clock"] and first["updates"] == first["replay"] == 0
    assert first["limits"]["max_training_updates"] == 2
    assert len(scripted.fits) == 2 and not fake_ports["games"]


@pytest.mark.parametrize("damage", ["value", "equal_container"])
def test_returned_observation_window_is_pinned_before_callbacks_and_restored_on_refusal(
    scripted, fake_ports, damage
):
    owner, state, sources = facade_setup()
    changed: list[bool] = []

    def parent():
        if fake_ports["games"] and not changed:
            row = fake_ports["games"][0]
            changed.append(True)
            if damage == "value":
                row["episodes"][0]["success"] = True
            else:
                row["episodes"] = deepcopy(row["episodes"])

    result = adapter.run_readout_probe(
        owner, state, sources, check_boundary=parent, memory=memory()
    )
    assert changed and result["status"] == "incomplete" and result["source_unchanged"] is None
    assert len(scripted.fits) == 4 and len(result["native_readouts"]) == 1
    assert result["probe"]["completed_updates"] == 2
    assert result["probe"]["return_phase"]["completed_updates"] == 0
    assert all(e["success"] is False for e in result["native_readouts"][0]["games"]["episodes"])
    assert _same_value(owner.snapshot_state(), state)


def test_last_snapshot_source_mutation_refuses_before_any_further_snapshot_port(
    scripted, fake_ports, monkeypatch
):
    owner, state, sources = facade_setup()
    original = owner.snapshot_state
    changed: list[bool] = []
    later_snapshots: list[bool] = []

    def snapshot():
        if changed:
            later_snapshots.append(True)
        result = original()
        if len(fake_ports["games"]) == 2 and not changed:
            changed.append(True)
            owner._state.native.heads[0].parameters["weight_input"][0, 0] += 1
        return result

    monkeypatch.setattr(owner, "snapshot_state", snapshot)
    result = adapter.run_readout_probe(
        owner, state, sources, check_boundary=lambda: None, memory=memory()
    )
    assert changed and not later_snapshots
    assert result["status"] == "incomplete" and result["source_unchanged"] is None
    assert len(scripted.fits) == 6
    assert result["probe"]["return_phase"]["completed_updates"] == 2
