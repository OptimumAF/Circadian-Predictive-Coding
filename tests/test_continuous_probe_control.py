"""A velocity-only disposable control preserves native state and probe authority.

One shared four-family phase-one fixture: 44 updates/92 training rows, four
parameter initializations (seven PC/BC constructor entries), 1,024 normal values.
All other leaf runners are stubbed; they perform no training or prediction.
Benchmark conditioning/residual/actor each use 32 calls/80 rows. The 32 probe
scopes use 256 prediction calls/576 rows and 32 diagnostic gradients/64 rows.
PC work is bounded by 165 energy entries/132 latent proposals, with 66 returned
candidate-forward rows. No extra initializer, fit, direct training oracle or
native simulation is executed by this module.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_learnability_demo as demo
from src.adapters import continuous_probe as probe
from src.adapters import continuous_probe_control as control
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def episode(name="A"):
    if name == "A":
        positions, actions, context, role = (
            [0.0, 2.0, 4.0],
            [(-2.0, -1.0), (6.0, 1.0)],
            "context-A",
            "train",
        )
    elif name == "B":
        positions, actions, context, role = (
            [10.0, 11.0, 12.0, 13.0],
            [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0)],
            "context-B",
            "train",
        )
    elif name == "Ftrain":
        positions, actions, context, role = (
            [-1.0, 1.0, 2.0],
            [(2.0, 0.5), (1.0, -0.5)],
            "fresh-F",
            "probe_train",
        )
    else:
        positions, actions, context, role = (
            [-0.5, 0.5, 1.5],
            [(1.5, 0.25), (1.25, -0.25)],
            "fresh-F",
            "probe_test",
        )
    observations = [(value, float(name == "B")) for value in positions]
    declared = ContinuousActionSpec(2, ("m", "m"), (-2.0, -1.0), (6.0, 1.0))
    return ContinuousEpisode(
        "synthetic-" + name,
        context,
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


def plan(**options):
    return probe.ContinuousProbePlan(
        episode("Ftrain"), episode("Ftest"), diagnostics=True, **options
    )


def assert_state(first, second, *, zero_velocity=False):
    assert type(first) is type(second) and first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(first.parameters, second.parameters, strict=True):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
    for a, b in zip(first.velocity, second.velocity, strict=True):
        np.testing.assert_array_equal(np.zeros_like(a) if zero_velocity else a, b)
        assert not np.shares_memory(a, b)
    a, b = getattr(first, "wake_state", None), getattr(second, "wake_state", None)
    if a is None:
        assert b is None
    else:
        assert b is not None
        assert a.wake_updates == b.wake_updates
        for name in (
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
        ):
            assert getattr(a, name) == getattr(b, name)
        np.testing.assert_array_equal(a.chemical_state, b.chemical_state)
        np.testing.assert_array_equal(a.plasticity_state, b.plasticity_state)
        assert not np.shares_memory(a.chemical_state, b.chemical_state)


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_control_flags_before_model_or_fit(monkeypatch, flag):
    with pytest.raises(ValueError):
        plan(zero_velocity_control=flag)
    value = plan(zero_velocity_control=True)
    object.__setattr__(value, "zero_velocity_control", flag)
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_a, **_k: pytest.fail("corrupt control flag reached fitting"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(episode(), episode("B"), phase_updates=1, probe_plan=value)


def test_should_own_last_control_selector_without_changing_older_defaults():
    default = probe.ContinuousProbePlan(episode("Ftrain"), episode("Ftest"))
    explicit = probe.ContinuousProbePlan(
        episode("Ftrain"), episode("Ftest"), zero_velocity_control=False
    )
    assert asdict(default) == asdict(explicit) and default.zero_velocity_control is False
    value = plan(zero_velocity_control=True)
    owned = probe.copy_probe_plan(value, (episode(), episode("B")))
    assert owned is not value and owned.zero_velocity_control is True
    object.__setattr__(value, "zero_velocity_control", False)
    assert owned.zero_velocity_control is True


@pytest.fixture(scope="module")
def four_arms():
    results: dict[str, Any] = {}
    calls: dict[str, Any] = {}
    runtimes: dict[str, Any] = {}
    warm, new, value = episode(), episode("B"), plan(zero_velocity_control=True)
    original_inputs = asdict(warm), asdict(new), asdict(value)
    original_single, original_runtime = probe._run_single_probe, probe.ActorShadowRuntime
    with pytest.MonkeyPatch.context() as patch:

        def single(source, owned_plan, **options):
            row = dict(source=source, state=source.snapshot_state(), plan=owned_plan)
            calls[family].append(row)
            returned = original_single(source, owned_plan, **options)
            row.update(returned=deepcopy(returned), after=source.snapshot_state())
            return returned

        def runtime(*args, **options):
            owner = original_runtime(*args, **options)
            budget, clock = options["budget"], options["clock"]
            runtimes[family].append(
                dict(
                    budget=budget,
                    limits=budget.budget,
                    timer=budget.clock,
                    started=budget.started_at,
                    clock=clock,
                )
            )
            return owner

        patch.setattr(probe, "_run_single_probe", single)
        patch.setattr(probe, "ActorShadowRuntime", runtime)
        for family in FAMILIES:
            calls[family], runtimes[family] = [], []
            result = demo.run_continuous_learnability(
                warm, new, phase_updates=1, learner_family=family, probe_plan=value
            )
            assert (asdict(warm), asdict(new), asdict(value)) == original_inputs
            results[family] = result, calls[family], runtimes[family]
    return results


@pytest.mark.parametrize("family", FAMILIES)
def test_should_run_independent_paired_scopes_without_changing_benchmark_or_reserved_roles(
    four_arms, family
):
    result, calls, runtimes = four_arms[family]
    parent = result.report
    assert parent["complete"] is True and result.candidate is not None
    assert parent["progress"]["updates_completed"] == 3
    assert [point["benchmark_updates"] for point in parent["probes"]] == [0, 1, 2, 3]
    assert len(calls) == len(runtimes) == 8
    assert len({id(row["budget"]) for row in runtimes}) == 8
    assert len({id(row["clock"]) for row in runtimes}) == 8
    for checkpoint, point in enumerate(parent["probes"]):
        pair = point["report"]
        assert pair["complete"] is True and pair["source_updates"] == checkpoint
        assert pair["initial_scores_exact"] is True
        first_call, second_call = calls[2 * checkpoint : 2 * checkpoint + 2]
        assert first_call["source"] is not second_call["source"]
        assert first_call["plan"].zero_velocity_control is False
        assert second_call["plan"].zero_velocity_control is False
        assert_state(first_call["state"], second_call["state"], zero_velocity=True)
        assert pair["inherited"] == first_call["returned"]
        assert pair["zero_velocity"] == second_call["returned"]
        assert (
            pair["inherited"]["checkpoints"][0]["scores"]
            == pair["zero_velocity"]["checkpoints"][0]["scores"]
        )
        for nested, captured in zip(
            (pair["inherited"], pair["zero_velocity"]),
            runtimes[2 * checkpoint : 2 * checkpoint + 2],
            strict=True,
        ):
            assert nested["complete"] is True
            assert nested["progress"] == dict(
                updates_completed=1, replay_examples_completed=0, next_tick=2
            )
            assert nested["plan"]["updates"] == 1
            assert len(nested["native_updates"]) == len(nested["applied_updates"]) == 1
            assert nested["applied_updates"][0]["episode_id"] == "synthetic-Ftrain"
            assert nested["events"][0]["deferred_exact"] is True
            assert nested["events"][0]["success_validated"] is True
            assert captured["budget"].updates_completed == 1
            assert captured["budget"].budget is captured["limits"]
            assert captured["limits"].max_training_updates == 1
            assert captured["budget"].clock is captured["timer"]
            assert captured["budget"].started_at == captured["started"]
            assert captured["clock"].now() == 1
            for entry in nested["checkpoints"]:
                scores = entry["scores"]
                assert scores["fresh"]["role"] == "probe_test"
                assert scores["fresh_training"]["role"] == "probe_train"
                assert scores["old_0"]["role"] == scores["old_1"]["role"] == "train"
        assert_state(first_call["state"], first_call["after"])
        assert_state(second_call["state"], second_call["after"])


def test_should_keep_pc_and_neutral_common_paired_scores_receipts_and_work_equal(four_arms):
    first = four_arms["pc"][0].report
    second = four_arms["neutral-cpc"][0].report
    cpc_keys = {
        "chemistry_attempts",
        "chemistry_returns",
        "gate_applications_attempted",
        "gate_applications_returned",
    }
    for a, b in zip(first["probes"], second["probes"], strict=True):
        a, b = a["report"], b["report"]
        assert a["contrast"] == b["contrast"]
        for key in ("inherited", "zero_velocity"):
            left, right = a[key], b[key]
            assert left["checkpoints"] == right["checkpoints"]
            assert left["metrics"] == right["metrics"]
            assert left["applied_updates"] == right["applied_updates"]
            assert left["update_diagnostics"] == right["update_diagnostics"]
            work_a, work_b = left["native_updates"][0]["work"], right["native_updates"][0]["work"]
            assert work_a == {name: value for name, value in work_b.items() if name not in cpc_keys}


@pytest.mark.parametrize("family", FAMILIES)
def test_should_report_signed_control_contrasts_without_requiring_improvement(four_arms, family):
    for point in four_arms[family][0].report["probes"]:
        pair = point["report"]
        inherited, zeroed = pair["inherited"]["metrics"], pair["zero_velocity"]["metrics"]
        contrast = pair["contrast"]
        assert (
            contrast["fresh_acquisition_gain"]
            == zeroed["fresh_acquisition_gain"] - inherited["fresh_acquisition_gain"]
        )
        assert (
            contrast["fresh_training_gain"]
            == zeroed["fresh_training_gain"] - inherited["fresh_training_gain"]
        )
        assert contrast["old_retention_loss"] == {
            name: zeroed["old_retention_loss"][name] - value
            for name, value in inherited["old_retention_loss"].items()
        }


@pytest.fixture
def stub_scope(four_arms, monkeypatch):
    result = four_arms["active-cpc"][0]
    source = result.candidate.fork()
    inherited = deepcopy(result.report["probes"][-1]["report"]["inherited"])
    calls = []

    def single(owner, value, **options):
        calls.append((owner, value, options))
        return deepcopy(inherited)

    monkeypatch.setattr(probe, "_run_single_probe", single)
    return source, inherited, calls


@pytest.mark.parametrize("explicit", [False, True])
def test_should_delegate_default_false_to_unchanged_single_report_without_preparation(
    stub_scope, monkeypatch, explicit
):
    source, inherited, calls = stub_scope
    value = plan(**({"zero_velocity_control": False} if explicit else {}))
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("default probe prepared a control"))
    returned = probe.run_continuous_probe(
        source, value, training_readouts=(episode(), episode("B"))
    )
    assert returned == inherited and len(calls) == 1
    assert calls[0][0] is source and calls[0][1].zero_velocity_control is False
    assert "inherited" not in returned and "preparation_work" not in returned


def test_should_not_prepare_or_run_control_after_incomplete_inherited_scope(
    stub_scope, monkeypatch
):
    source, inherited, calls = stub_scope
    inherited.update(complete=False, error="inherited stopped after paid update")
    monkeypatch.setattr(
        source, "fork", lambda: pytest.fail("incomplete inherited work prepared control")
    )
    returned = probe.run_continuous_probe(
        source, plan(zero_velocity_control=True), training_readouts=(episode(), episode("B"))
    )
    assert returned["complete"] is False and len(calls) == 1
    assert returned["inherited"] == inherited and returned["zero_velocity"] is None
    assert returned["inherited"]["progress"]["updates_completed"] == 1
    assert len(returned["inherited"]["native_updates"]) == 1
    assert "inherited" in returned["error"].lower()


@pytest.mark.parametrize("foreign", [False, True])
def test_should_refuse_aliased_or_foreign_control_owner_before_control_runner(
    stub_scope, monkeypatch, foreign
):
    source, inherited, calls = stub_scope
    monkeypatch.setattr(source, "fork", lambda: object() if foreign else source)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and returned["inherited"] == inherited
    assert returned["zero_velocity"] is None and len(calls) == 1


@pytest.mark.parametrize("damage", ["velocity", "parameters", "wake", "counter"])
def test_should_refuse_inherited_callback_change_of_original_full_native_state(
    stub_scope, monkeypatch, damage
):
    source, inherited, _calls = stub_scope
    calls = []

    def single(owner, _value, **_options):
        calls.append(True)
        if damage == "velocity":
            owner._state.velocity[0][0, 0] += 0.01
        elif damage == "parameters":
            owner._state.parameters[0][0, 0] += 0.01
        elif damage == "counter":
            owner._state = replace(
                owner._state, training_examples=owner._state.training_examples + 1
            )
        else:
            state = owner._state
            wake, config = state.wake_state, state.policy.circadian
            chemicals = wake.chemical_state + 0.01
            gates = np.clip(
                np.exp(-config.plasticity_sensitivity * chemicals), config.min_plasticity, 1.0
            )
            owner._state = replace(
                state, wake_state=replace(wake, chemical_state=chemicals, plasticity_state=gates)
            )
        return deepcopy(inherited)

    monkeypatch.setattr(probe, "_run_single_probe", single)
    monkeypatch.setattr(source, "fork", lambda: pytest.fail("changed source prepared control"))
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 1
    assert returned["inherited"] == inherited and returned["zero_velocity"] is None


def test_should_keep_failed_paid_control_report_and_original_source(stub_scope, monkeypatch):
    source, inherited, _calls = stub_scope
    before, calls = source.snapshot_state(), []
    failed = deepcopy(inherited)
    failed.update(complete=False, error="control receipt refused after permanent charge")

    def single(owner, _value, **_options):
        calls.append(owner)
        return deepcopy(inherited if len(calls) == 1 else failed)

    monkeypatch.setattr(probe, "_run_single_probe", single)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 2
    assert returned["inherited"] == inherited and returned["zero_velocity"] == failed
    assert returned["zero_velocity"]["progress"]["updates_completed"] == 1
    assert (
        len(returned["zero_velocity"]["applied_updates"])
        == len(returned["zero_velocity"]["native_updates"])
        == 1
    )
    assert_state(before, source.snapshot_state())


def test_should_account_only_zeroed_velocity_payload_and_one_extra_detached_source(four_arms):
    for family in FAMILIES:
        for index, point in enumerate(four_arms[family][0].report["probes"]):
            pair = point["report"]
            state = four_arms[family][1][2 * index]["state"]
            work = pair["preparation_work"]
            assert work["source_fork_attempts"] == work["source_fork_returns"] == 1
            assert work["additional_source_owners"] == 1
            assert work["velocity_zero_array_attempts"] == work["velocity_zero_array_returns"] == 4
            assert work["velocity_values_zeroed"] == sum(value.size for value in state.velocity)
            assert work["velocity_numeric_bytes_zeroed"] == sum(
                value.nbytes for value in state.velocity
            )
            assert work["restore_attempts"] == work["restore_returns"] == 1
            assert (
                work["fits"]
                == work["parameter_initializations"]
                == work["seeded_normal_values"]
                == 0
            )
            assert work["parameter_momentum_numeric_bytes_per_source"] == sum(
                value.nbytes for value in state.parameters + state.velocity
            )
    assert any(
        np.any(value != 0)
        for row in four_arms["active-cpc"][1][2:]
        for value in row["state"].velocity
    )


@pytest.mark.parametrize("damage", ["paid_bool", "source_float", "receipt", "event"])
def test_should_validate_nominal_inherited_success_before_admitting_more_work(
    stub_scope, monkeypatch, damage
):
    source, inherited, calls = stub_scope
    if damage == "paid_bool":
        inherited["progress"]["updates_completed"] = True
    elif damage == "source_float":
        inherited["source_updates"] = float(inherited["source_updates"])
    elif damage == "receipt":
        inherited["applied_updates"][0]["event_id"] = "replaced-preceding-label"
    else:
        inherited["events"][0]["success_validated"] = False
    fork_calls = []

    def fork():
        fork_calls.append(True)
        pytest.fail("invalid inherited receipt admitted control source")

    monkeypatch.setattr(source, "fork", fork)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 1 and fork_calls == []
    assert returned["inherited"] == inherited and returned["zero_velocity"] is None
    assert returned["preparation_work"]["source_fork_attempts"] == 0


def test_should_preserve_failed_preparation_attempt_and_no_control_runner(stub_scope, monkeypatch):
    source, inherited, calls = stub_scope
    child = source.fork()

    def fail_restore(_state):
        raise RuntimeError("zero-velocity snapshot publication failed")

    monkeypatch.setattr(child, "restore_state", fail_restore)
    monkeypatch.setattr(source, "fork", lambda: child)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 1
    assert returned["inherited"] == inherited and returned["zero_velocity"] is None
    work = returned["preparation_work"]
    assert work["source_fork_returns"] == 1 and work["additional_source_owners"] == 1
    assert work["velocity_zero_array_returns"] == 4
    assert work["restore_attempts"] == 1 and work["restore_returns"] == 0
    assert "publication failed" in returned["error"]


def test_should_refuse_changed_initial_physical_scores_without_hiding_completed_scopes(
    stub_scope, monkeypatch
):
    source, inherited, _calls = stub_scope
    responses = [deepcopy(inherited), deepcopy(inherited)]
    responses[1]["checkpoints"][0]["scores"]["fresh"]["residual"]["overall"][
        "normalized_half_mse"
    ] += 0.01
    calls = []

    def single(*_args, **_options):
        calls.append(True)
        return responses[len(calls) - 1]

    monkeypatch.setattr(probe, "_run_single_probe", single)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 2
    assert returned["inherited"] == responses[0] and returned["zero_velocity"] == responses[1]
    assert returned["initial_scores_exact"] is False
    assert "initial" in returned["error"]


def test_should_preserve_original_inherited_evidence_when_control_callback_replaces_it(
    stub_scope, monkeypatch
):
    source, inherited, _calls = stub_scope
    responses = [deepcopy(inherited), deepcopy(inherited)]
    original = deepcopy(responses[0])
    calls = []

    def single(*_args, **_options):
        calls.append(True)
        if len(calls) == 2:
            responses[0]["metrics"]["fresh_acquisition_gain"] += 1.0
        return responses[len(calls) - 1]

    monkeypatch.setattr(probe, "_run_single_probe", single)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 2
    assert returned["zero_velocity"] == responses[1]
    assert returned["inherited"] == original
    assert "inherited" in returned["error"].lower() or "preceding" in returned["error"].lower()


def test_should_retain_partial_velocity_zeroing_cost_before_preparation_failure(
    stub_scope, monkeypatch
):
    source, inherited, calls = stub_scope
    zero_like, attempts = np.zeros_like, []

    def zero(value):
        attempts.append(value.shape)
        if len(attempts) == 3:
            raise RuntimeError("third zeroed array allocation failed")
        return zero_like(value)

    monkeypatch.setattr(control.np, "zeros_like", zero)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 1
    assert returned["inherited"] == inherited and returned["zero_velocity"] is None
    work = returned["preparation_work"]
    assert work["velocity_zero_array_attempts"] == 3
    assert work["velocity_zero_array_returns"] == 2
    assert work["velocity_values_zeroed"] == sum(
        array.size for array in source.snapshot_state().velocity[:2]
    )
    assert work["restore_attempts"] == 0


def test_should_detect_control_observer_callback_changing_another_native_owner(
    stub_scope, monkeypatch
):
    source, inherited, _calls = stub_scope
    calls = []

    def single(owner, _value, **_options):
        calls.append(owner)
        if len(calls) == 2:
            snapshot = owner.snapshot_state

            def changed_other():
                returned = snapshot()
                source._state.velocity[0][0, 0] += 0.01
                return returned

            monkeypatch.setattr(owner, "snapshot_state", changed_other)
        return deepcopy(inherited)

    monkeypatch.setattr(probe, "_run_single_probe", single)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 2
    assert returned["inherited"] == inherited and returned["zero_velocity"] == inherited


def test_should_refuse_restore_callback_mutating_expected_intervention(stub_scope, monkeypatch):
    source, inherited, calls = stub_scope
    child = source.fork()
    restore = child.restore_state

    def change_argument(state):
        state.parameters[0][0, 0] += 0.01
        restore(state)

    monkeypatch.setattr(child, "restore_state", change_argument)
    monkeypatch.setattr(source, "fork", lambda: child)
    returned = probe.run_continuous_probe(source, plan(zero_velocity_control=True))
    assert returned["complete"] is False and len(calls) == 1
    assert returned["inherited"] == inherited and returned["zero_velocity"] is None
    assert returned["preparation_work"]["restore_returns"] == 1
    assert "source" in returned["error"]


def test_should_pair_without_inventing_training_diagnostics_when_disabled(stub_scope, monkeypatch):
    source, inherited, _calls = stub_scope
    inherited["metrics"].pop("fresh_training_gain")
    inherited.pop("update_diagnostics")
    inherited["plan"].pop("diagnostics")
    for checkpoint in inherited["checkpoints"]:
        checkpoint["scores"].pop("fresh_training")
    calls = []

    def single(_owner, value, **_options):
        assert value.diagnostics is False and value.zero_velocity_control is False
        calls.append(True)
        return deepcopy(inherited)

    monkeypatch.setattr(probe, "_run_single_probe", single)
    value = probe.ContinuousProbePlan(
        episode("Ftrain"), episode("Ftest"), zero_velocity_control=True
    )
    returned = probe.run_continuous_probe(source, value)
    assert returned["complete"] is True and len(calls) == 2
    assert "fresh_training_gain" not in returned["contrast"]
    assert returned["inherited"] == returned["zero_velocity"] == inherited


def test_should_compare_live_tuple_reports_with_saved_json_without_ignoring_scalar_changes():
    from scripts.continuous_probe_control_example import _without_elapsed

    live = dict(policy=dict(units=("m", "m"), bounds=((-2.0, 6.0),)), elapsed_seconds=1.0)
    saved: dict[str, Any] = dict(
        policy=dict(units=["m", "m"], bounds=[[-2.0, 6.0]]), elapsed_seconds=9.0
    )
    assert _without_elapsed(live) == _without_elapsed(saved)
    saved["policy"]["bounds"][0][0] = -1.0
    assert _without_elapsed(live) != _without_elapsed(saved)
