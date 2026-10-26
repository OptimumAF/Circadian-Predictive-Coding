"""Four matched learner families through the existing bounded TRAIN runner.

Shared phase-one runs are correctness fixtures, not comparative evidence. CLI
and admission cases stub all runners; failure cases preserve paid native work.

Full module: 41 cases, eleven tiny runner invocations, 20 constructor entries and
2816 seeded normal values. There are 25 update attempts (24 publications; one
failed initial PC energy), 55 published plus two failed training-row exposures.
Task-head conditioning has 65 calls/162 rows; residual and actor readouts each
64 calls/160 rows. PC energy entries and latent proposals are each bounded by
160. No direct training oracle, real data, simulator or extra RNG is used.
"""

from dataclasses import asdict, replace
import json
import sys
from typing import Any

import numpy as np
import pytest

from scripts import run_native_stream_demo as cli
from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_pc as pc
from src.core import continuous_bc_diagnostics as diagnostics
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
CPC_WORK = {
    "chemistry_attempts",
    "chemistry_returns",
    "gate_applications_attempted",
    "gate_applications_returned",
}


def episode(name="A", *, role="train"):
    observations = (
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        if name == "A"
        else [(10.0, 1.0), (11.0, 1.0), (12.0, 1.0), (13.0, 1.0)]
    )
    actions = [(-2.0, -1.0), (6.0, 1.0)] if name == "A" else [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0)]
    declared = ContinuousActionSpec(2, ("m", "m"), (-2.0, -1.0), (6.0, 1.0))
    return ContinuousEpisode(
        "synthetic-" + name,
        "context-" + name,
        role,
        declared,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                action,
                0.0,
                observations[index + 1],
                index == len(actions) - 1,
                False,
            )
            for index, action in enumerate(actions)
        ),
    )


def assert_arrays_equal(first, second):
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for left, right in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(left, right)


def assert_full(first, second):
    assert type(first) is type(second) and first.policy == second.policy
    assert_arrays_equal(first, second)
    first_wake, second_wake = (
        getattr(first, "wake_state", None),
        getattr(second, "wake_state", None),
    )
    if first_wake is None:
        assert second_wake is None
    else:
        assert second_wake is not None
        assert first_wake.wake_updates == second_wake.wake_updates
        for name in (
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
        ):
            assert getattr(first_wake, name) == getattr(second_wake, name)
        np.testing.assert_array_equal(first_wake.chemical_state, second_wake.chemical_state)
        np.testing.assert_array_equal(first_wake.plasticity_state, second_wake.plasticity_state)


@pytest.fixture(scope="module")
def four_arms():
    warm, new = episode(), episode("B")
    inputs = asdict(warm), asdict(new)
    results = {}
    states: dict[str, list[Any]] = {}
    owners: dict[str, list[Any]] = {}
    fits = []
    original_checkpoint, original_fit = demo._checkpoint, demo.fit_training_standardization
    with pytest.MonkeyPatch.context() as patch:

        def fit(value, **options):
            fits.append((value.episode_id, options))
            return original_fit(value, **options)

        def checkpoint(report, source, runtime, readout, *args):
            original_checkpoint(report, source, runtime, readout, *args)
            states[family].append(runtime.candidate_snapshot().state)
            owners[family].append((source.snapshot_state(), runtime.actor.snapshot_state().state))

        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(demo, "_checkpoint", checkpoint)
        for family in FAMILIES:
            states[family], owners[family] = [], []
            options: dict[str, Any] = {} if family == "bp" else dict(learner_family=family)
            results[family] = demo.run_continuous_learnability(
                warm, new, phase_updates=1, **options
            )
            assert (asdict(warm), asdict(new)) == inputs
    return results, states, owners, fits


@pytest.mark.parametrize("family", FAMILIES)
def test_should_complete_same_delayed_aba_sequence_with_one_a_fit_and_unchanged_owners(
    four_arms, family
):
    results, states, owners, fits = four_arms
    result, snapshots = results[family], states[family]
    report = result.report
    assert report["complete"] is True and result.candidate is not None
    assert fits == [("synthetic-A", {})] * 4
    assert [state.completed_updates for state in snapshots] == [0, 1, 2, 3]
    assert [state.training_examples for state in snapshots] == [0, 2, 5, 7]
    assert report["progress"] == dict(
        update_attempts=3, updates_completed=3, replay_examples_completed=0, next_tick=6
    )
    assert [event["phase"] for event in report["events"]] == ["warm", "new", "warm_return"]
    for index, event in enumerate(report["events"]):
        assert event["deferred_exact"] and event["success_validated"]
        (receipt,) = event["applied"]
        assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
            2 * index,
            2 * index + 1,
            2 * index + 1,
        )
        assert receipt["episode_id"] == ("synthetic-B" if index == 1 else "synthetic-A")
    for source, actor in owners[family]:
        assert_full(snapshots[0], source)
        assert_full(snapshots[0], actor)
    assert_full(snapshots[-1], result.candidate.snapshot_state())
    assert all(row["state_exact"] and row["owners_exact"] for row in report["checkpoints"])
    if family != "bp":
        assert type(result.candidate) is pc.NumpyContinuousPCLearner
        assert report["learner_family"] == family
        assert report["protocol_id"] != results["bp"].report["protocol_id"]
        assert len(report["native_updates"]) == 3
    else:
        assert "learner_family" not in report and "native_updates" not in report


def test_should_match_pc_and_neutral_common_states_metrics_readouts_and_native_work_exactly(
    four_arms,
):
    results, states, _, _ = four_arms
    first, second = results["pc"].report, results["neutral-cpc"].report
    assert first["metrics"] == second["metrics"]
    assert first["checkpoints"] == second["checkpoints"]
    for before, after in zip(states["pc"], states["neutral-cpc"], strict=True):
        assert replace(after.policy, circadian=None) == before.policy
        assert_arrays_equal(before, after)
        assert before.wake_state is None
        assert after.wake_state.wake_updates == after.completed_updates
        np.testing.assert_array_equal(after.wake_state.plasticity_state, np.ones(64))
        if after.completed_updates:
            assert np.any(after.wake_state.chemical_state > 0)
    for pc_update, neutral_update in zip(
        first["native_updates"], second["native_updates"], strict=True
    ):
        assert (
            set(pc_update)
            == set(neutral_update)
            == {"diagnostic", "task_loss", "inference", "work", "training_rows"}
        )
        assert pc_update["training_rows"] == neutral_update["training_rows"]
        for field in ("diagnostic", "task_loss", "inference"):
            assert pc_update[field] == neutral_update[field]
        assert pc_update["work"] == {
            key: value for key, value in neutral_update["work"].items() if key not in CPC_WORK
        }
        assert len(pc_update["work"]) == 16
        assert {key: neutral_update["work"][key] for key in CPC_WORK} == {
            key: 1 for key in CPC_WORK
        }


@pytest.mark.parametrize("family", FAMILIES[1:])
def test_should_separate_returned_pc_native_work_from_readonly_task_head_gradients(
    four_arms, family
):
    report = four_arms[0][family].report
    work, native = report["work"], report["native_updates"]
    assert [item["training_rows"] for item in native] == [2, 3, 2]
    counts = {key: sum(item["work"][key] for item in native) for key in native[0]["work"]}
    weighted = {
        key: sum(item["work"][key] * item["training_rows"] for item in native) for key in counts
    }
    assert work["native_update_counts"] == counts
    assert work["native_event_row_weights"] == weighted
    assert counts["published_updates"] == 3 and weighted["published_updates"] == 7
    assert work["optimizer_updates"] == 3 and work["training_rows"] == 7
    assert work["unique_source_rows"] == 5 and work["repeated_training_rows"] == 2
    assert (
        work["conditioning_calls"] == work["residual_calls"] == work["actor_prediction_calls"] == 8
    )
    assert work["conditioning_rows"] == work["residual_rows"] == work["actor_prediction_rows"] == 20
    assert work["conditioning_definition"] == "common_task_head_bp_total_gradients_read_only"
    assert work["readout_normalization_calls"] == 24
    assert work["readout_normalization_rows"] == 60
    assert work["readout_normalization_values"] == 120
    assert work["residual_actor_forward_passes"] == 16
    assert work["residual_actor_forward_rows"] == 40
    assert work["residual_actor_dense_multiply_accumulates"] == 40 * (2 * 64 + 64 * 2)
    contexts = [item for row in report["checkpoints"] for item in row["contexts"].values()]
    assert all(
        item["conditioning_definition"] == work["conditioning_definition"] for item in contexts
    )
    for key, total in work["conditioning_work"].items():
        assert total == sum(item["conditioning"]["conditioning"]["work"][key] for item in contexts)
    assert work["wake_numeric_bytes_per_owner"] == (0 if family == "pc" else 1024)
    assert work["parameter_values_per_owner"] == 322
    assert work["model_numeric_bytes_per_owner"] == 5152
    assert not {"forward_passes", "backward_passes", "dense_multiply_accumulates"}.intersection(
        work
    )


def test_should_preserve_physical_metrics_and_real_active_wake_without_ranking_energies(four_arms):
    results, states, _, _ = four_arms
    for family in FAMILIES:
        report = results[family].report
        values = [
            {
                context: row["contexts"][context]["residual"]["residuals"]["overall"][
                    "normalized_half_mse"
                ]
                for context in ("warm", "new")
            }
            for row in report["checkpoints"]
        ]
        assert report["metrics"] == dict(
            warm_acquisition_gain=values[0]["warm"] - values[1]["warm"],
            new_acquisition_gain=values[1]["new"] - values[2]["new"],
            warm_retention_loss=values[2]["warm"] - values[1]["warm"],
            warm_return_recovery=values[2]["warm"] - values[3]["warm"],
            warm_return_deficit=values[3]["warm"] - values[1]["warm"],
            new_retention_loss=values[3]["new"] - values[2]["new"],
        )
    for state in states["active-cpc"][1:]:
        wake, config = state.wake_state, state.policy.circadian
        assert wake.wake_updates == state.completed_updates
        np.testing.assert_array_equal(
            wake.plasticity_state,
            np.clip(
                np.exp(-config.plasticity_sensitivity * wake.chemical_state),
                config.min_plasticity,
                1,
            ),
        )
        assert np.any(wake.plasticity_state < 1)


def test_should_preserve_omitted_bp_report_and_final_state_when_bp_is_explicit(four_arms):
    default = four_arms[0]["bp"]
    explicit = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="bp"
    )
    assert explicit.report["complete"] is True
    assert explicit.candidate is not None
    assert {key: value for key, value in explicit.report.items() if key != "elapsed_seconds"} == {
        key: value for key, value in default.report.items() if key != "elapsed_seconds"
    }
    assert_full(default.candidate.snapshot_state(), explicit.candidate.snapshot_state())


@pytest.mark.parametrize("family", [None, True, "", "PC", "unknown", 1])
def test_should_refuse_invalid_family_before_source_admission_fit_or_constructor(
    monkeypatch, family
):
    monkeypatch.setattr(
        demo, "_training_episodes", lambda *_: pytest.fail("invalid family admitted source data")
    )
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("invalid family fitted observations"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(None, None, learner_family=family)


def test_should_refuse_family_subclass_and_nonbp_binary_before_fitting(monkeypatch):
    class DerivedFamily(str):
        pass

    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("invalid family/likelihood fitted observations"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(
            episode(), episode("B"), learner_family=DerivedFamily("pc")
        )
    for family in FAMILIES[1:]:
        with pytest.raises(ValueError):
            demo.run_continuous_learnability(
                episode(), episode("B"), learner_family=family, binary_gripper_axis=0
            )


@pytest.mark.parametrize("family", FAMILIES[1:])
def test_should_refuse_final_or_probe_inputs_before_any_family_fit(monkeypatch, family):
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("heldout observations reached fitting"),
    )
    with pytest.raises(ValueError, match="TRAIN"):
        demo.run_continuous_learnability(
            episode(), episode("B", role="final_test"), learner_family=family
        )


def test_should_retain_native_return_and_permanent_receipt_after_postcommit_error(monkeypatch):
    original = demo.ActorShadowRuntime

    def runtime(*args, **options):
        owner = original(*args, **options)
        ready = owner.train_ready

        def late(*args, **kwargs):
            result = ready(*args, **kwargs)
            if result:
                raise RuntimeError("after actual native publication")
            return result

        monkeypatch.setattr(owner, "train_ready", late)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc"
    )
    report = result.report
    assert result.candidate is None and report["complete"] is False
    assert "after actual native publication" in report["error"]
    assert report["progress"]["updates_completed"] == 1
    assert len(report["applied_updates"]) == len(report["native_updates"]) == 1
    assert report["native_updates"][0]["work"]["published_updates"] == 1
    assert "metrics" not in report and "work" not in report


def test_should_retain_failed_native_work_without_inventing_a_return_or_charge(monkeypatch):
    monkeypatch.setattr(
        pc.energy,
        "evaluate_continuous_pc",
        lambda *_: (_ for _ in ()).throw(FloatingPointError("initial PC energy failed")),
    )
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc"
    )
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert report["progress"]["updates_completed"] == 0
    assert report["native_updates"] == report["applied_updates"] == []
    assert report["failed_native_work"]["energy_evaluations"] == 1
    assert report["failed_native_work"]["energy_returns"] == 0
    assert report["failed_native_work"]["published_updates"] == 0
    assert "work" not in report and "metrics" not in report


def changed_wake(state):
    wake, config = state.wake_state, state.policy.circadian
    chemicals = wake.chemical_state + 0.01
    gates = np.clip(np.exp(-config.plasticity_sensitivity * chemicals), config.min_plasticity, 1)
    return replace(
        state, wake_state=replace(wake, chemical_state=chemicals, plasticity_state=gates)
    )


def test_should_refuse_coherent_wake_only_replacement_during_final_empty_poll(monkeypatch):
    original = demo.ActorShadowRuntime

    def runtime(*args, **options):
        owner = original(*args, **options)
        ready = owner.train_ready

        def mutate(*args, **kwargs):
            result = ready(*args, **kwargs)
            if not result and options["budget"].updates_completed == 3:
                owner._candidate.restore_state(changed_wake(owner.candidate_snapshot().state))
            return result

        monkeypatch.setattr(owner, "train_ready", mutate)
        return owner

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="active-cpc"
    )
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert "empty poll" in report["error"]
    assert report["progress"]["updates_completed"] == 3
    assert len(report["applied_updates"]) == len(report["native_updates"]) == 3
    assert "work" not in report and "metrics" not in report


def test_should_refuse_valid_changed_wake_from_conditioning_callback_before_more_updates(
    monkeypatch,
):
    original_checkpoint, original_conditioning = (
        demo._checkpoint,
        diagnostics.continuous_bc_conditioning,
    )
    current_readout: list[Any] = []

    def checkpoint(report, source, runtime, readout, *args):
        current_readout[:] = [readout]
        return original_checkpoint(report, source, runtime, readout, *args)

    def mutate(*args, **kwargs):
        result = original_conditioning(*args, **kwargs)
        owner = current_readout[0]
        state = owner.snapshot_state()
        if state.completed_updates:
            owner.restore_state(changed_wake(state))
        return result

    monkeypatch.setattr(demo, "_checkpoint", checkpoint)
    monkeypatch.setattr(diagnostics, "continuous_bc_conditioning", mutate)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="active-cpc"
    )
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert "conditioning" in report["error"]
    assert report["progress"]["updates_completed"] == 1
    assert len(report["applied_updates"]) == len(report["native_updates"]) == 1
    assert "work" not in report and "metrics" not in report


@pytest.mark.parametrize("boundary", ["preceding_success", "final_check"])
def test_should_revalidate_preceding_success_after_owner_callbacks_before_admitting_more_work(
    monkeypatch, boundary
):
    original = demo._require_owners
    injected: list[bool] = []
    final_calls = []

    def mutate(source, runtime, initial):
        original(source, runtime, initial)
        state = runtime.candidate_snapshot().state
        if state.completed_updates == 3:
            final_calls.append(True)
        should_change = (
            state.completed_updates == 1
            if boundary == "preceding_success"
            else len(final_calls) == 3
        )
        if should_change and not injected:
            runtime._candidate.restore_state(changed_wake(state))
            injected.append(True)

    monkeypatch.setattr(demo, "_require_owners", mutate)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="active-cpc"
    )
    report = result.report
    assert injected == [True]
    assert report["complete"] is False and result.candidate is None
    assert (
        "preceding validated success" if boundary == "preceding_success" else "final owner check"
    ) in report["error"]
    updates = 1 if boundary == "preceding_success" else 3
    assert report["progress"]["updates_completed"] == updates
    assert (
        len(report["events"])
        == len(report["native_updates"])
        == len(report["applied_updates"])
        == updates
    )
    assert "metrics" not in report and "work" not in report


@pytest.mark.parametrize("family", FAMILIES)
def test_should_route_json_wrapper_with_only_nondefault_family_keyword(monkeypatch, family):
    calls = []
    report = {"complete": True, "route": family}

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return demo.ContinuousLearnabilityResult(report, None)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    assert demo.run_continuous_learnability_demo("A", "B", learner_family=family) is report
    expected = dict(expected_standardization=None)
    if family != "bp":
        expected["learner_family"] = family
    assert calls == [(("A", "B"), expected)]


@pytest.mark.parametrize("family", [None, *FAMILIES])
def test_should_route_cli_family_with_existing_control_options_without_models(
    monkeypatch, capsys, family
):
    reads, calls = [], []

    def read(path, seed, **options):
        reads.append((str(path), seed, options))
        return "A" if seed == 17 else "B"

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"complete": True}

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    argv = [
        "demo",
        "--scenario",
        "continuous-mlp-learnability",
        "--warm-teacher",
        "A.json",
        "--new-teacher",
        "B.json",
        "--phase-updates",
        "1",
        "--a-only-control",
    ]
    if family is not None:
        argv += ["--learner-family", family]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert reads == [("A.json", 17, {}), ("B.json", 19, {})]
    options = dict(phase_updates=1, a_only_control=True)
    if family not in (None, "bp"):
        options["learner_family"] = family
    assert calls == [(("A", "B"), options)]
    assert json.loads(capsys.readouterr().out) == {"complete": True}


@pytest.mark.parametrize(
    "options",
    [
        ["--scenario", "continuous-nearest", "--learner-family", "pc"],
        ["--scenario", "stream", "--learner-family", "bp"],
        [
            "--scenario",
            "continuous-mlp-learnability",
            "--learner-family",
            "pc",
            "--binary-gripper-axis",
            "3",
        ],
        [
            "--scenario",
            "continuous-mlp-learnability",
            "--learner-family",
            "neutral-cpc",
            "--binary-gripper-axis",
            "0",
        ],
        [
            "--scenario",
            "continuous-mlp-learnability",
            "--learner-family",
            "active-cpc",
            "--binary-gripper-axis",
            "3",
        ],
        ["--scenario", "continuous-mlp-learnability", "--learner-family", "unknown"],
    ],
)
def test_should_refuse_invalid_cli_family_or_likelihood_before_teacher_reads(monkeypatch, options):
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_a, **_k: pytest.fail("invalid family CLI read data")
    )
    monkeypatch.setattr(
        sys, "argv", ["demo", *options, "--warm-teacher", "A.json", "--new-teacher", "B.json"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
