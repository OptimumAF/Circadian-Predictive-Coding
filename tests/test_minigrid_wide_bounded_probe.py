"""H32 weighted F/A composition with scripted ports and zero native work.

Admission uses the real descriptor and runtime path. Scripted failure routing
proves ownership and permanent charges; numerical learner behavior is separate.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_bounded_probe as recipes
from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_pc_inference import TORCH_BOUNDED_WEIGHTED_PC_ID
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, forbidden, packets


METHODS = ("pc", "neutral_cpc")
MODES = ("fixed-weighted", "bounded-weighted")
OPTIONS = {**FLAGS, "qualification_method": "pc", "joint_catalogue": True, "wide_neural": True}


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode", MODES)
def test_should_describe_unchanged_h32_benchmark_and_separate_weighted_probe(
    no_resources, method, mode
):
    options = {**OPTIONS, "qualification_method": method}
    baseline = pilot.describe_protocol(**options)
    assert baseline == pilot.describe_protocol(**options, probe_inference=None)
    actual = pilot.describe_protocol(**options, probe_inference=mode)
    for key in (
        "seed",
        "hidden_dim",
        "methods",
        "trainable_parameters",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "checkpoint_updates",
        "joint_catalogue",
        "probe_support",
        "metric_direction",
    ):
        assert actual[key] == baseline[key], key
    assert actual["hidden_dim"] == 32 and actual["inference_steps"] == 2
    assert actual["checkpoint_updates"] == [0, 32, 64, 96, 128]
    assert actual["probe"]["source_stages"] == ["initial", "post_joint128"]
    assert actual["protocol_id"] != baseline["protocol_id"]
    for key, value in baseline["planned_work"].items():
        if key == "wide_neural":
            for field, expected in value.items():
                if field != "scope":
                    assert actual["planned_work"][key][field] == expected, field
            assert "configured maxima" in actual["planned_work"][key]["scope"]
        else:
            assert actual["planned_work"][key] == value, key
    work = actual["planned_work"]
    assert work["benchmark_native_train_calls"] == 128
    assert work["benchmark_training_rows"] == 6912
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["score_forwards"] == 94 and work["score_rows"] == 846
    assert work["final_episodes"] == work["replay_rows"] == 0
    recipe = actual["probe_inference"]
    assert recipe["mode"] == mode and recipe["max_steps"] == 2
    assert recipe["policy"] == asdict(recipes.policy(mode, 2))
    assert recipe["objective"] == TORCH_BOUNDED_WEIGHTED_PC_ID
    assert (
        actual["probe"]["objective"] == actual["return_phase"]["objective"] == recipe["objective"]
    )
    assert baseline["probe"].get("objective") != recipe["objective"]
    assert work["probe_inference"]["native_train_calls"] == 64
    assert work["probe_inference"]["training_rows"] == 576
    assert work["probe_inference"]["configured_latent_row_iterations"] == 1152


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode", MODES)
def test_should_pass_actual_runtime_admission_before_resource_boundary(
    monkeypatch, no_resources, method, mode
):
    calls = []

    def boundary():
        calls.append(method)
        raise RuntimeError("scripted weighted H32 runtime resource boundary")

    monkeypatch.setattr(pilot, "require_torch", boundary)
    result = pilot.run_pilot(**{**OPTIONS, "qualification_method": method}, probe_inference=mode)
    assert calls == [method]
    assert result["status"] == "incomplete"
    assert "scripted weighted H32 runtime resource boundary" in result["reason"]
    assert result["work"]["native_train_calls"] == 0


@pytest.mark.parametrize(
    "mode", [True, False, 0, 1, "adaptive", [], {}, np.str_("bounded-weighted")]
)
def test_should_refuse_nonprimitive_or_unknown_recipe_before_resources(no_resources, mode):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**OPTIONS, probe_inference=mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "changed",
    [
        {"qualification_method": method}
        for method in (None, "bp", "gated_cpc", "pc_relax64", "neutral_cpc_relax64", np.str_("pc"))
    ]
    + [
        {key: False}
        for key in (
            "public_history",
            "invariant_history",
            "joint_catalogue",
            "distinct_probe",
            "old_task_readouts",
            "return_to_a",
        )
    ]
    + [
        {key: True}
        for key in (
            "old_a_rehearsal",
            "extra_f_exposure",
            "probe_rollouts",
            "deferred_probe_rollouts",
            "extended_wide_probe_budget",
            "weighted_pc",
            "weighted_catalogue",
            "train_action_margins",
            "activation_health",
            "gradient_geometry",
            "prior_gradient_geometry",
            "train_fit_bounds",
            "bp_update_diagnostics",
            "pc_update_diagnostics",
            "cue_relative_history",
            "memory_control",
        )
    ],
)
def test_should_refuse_incomplete_mixed_or_other_native_h32_routes(no_resources, mode, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**OPTIONS, **changed}, probe_inference=mode)


def arguments(method):
    result = [*CLI_FLAGS, "--joint-catalogue", "--wide-neural"]
    result[result.index("--qualification-method") + 1] = method
    return result


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_explicit_weighted_h32_cli(monkeypatch, capsys, method, mode, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments(method), "--probe-inference", mode]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**OPTIONS, "qualification_method": method, "probe_inference": mode}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize("method", ["bp", "gated_cpc", "pc_relax64", "neutral_cpc_relax64"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_refuse_other_h32_cli_methods_before_dispatch(monkeypatch, method, describe):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments(method), "--probe-inference", "bounded-weighted"]
        + (["--describe"] if describe else []),
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [
        "--old-a-rehearsal",
        "--extra-f-exposure",
        "--probe-rollouts",
        "--deferred-probe-rollouts",
        "--extended-wide-probe-budget",
        "--weighted-pc",
        "--train-action-margins",
        "--pc-update-diagnostics",
        "--prior-gradient-geometry",
    ],
)
def test_should_refuse_mixed_weighted_h32_cli_before_dispatch(monkeypatch, extra):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments("pc"), "--probe-inference", "bounded-weighted", extra, "--describe"],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("point", [0, 128])
def test_should_route_owned_warm_state_and_preserve_scripted_paid_failure(
    monkeypatch, method, mode, point
):
    data = tuple(pilot._history_packets(items) for items in raw_probe_data())
    monitors = packets()[:2]
    initial = SimpleNamespace(
        policy={"method": method, "hidden_dim": 32},
        completed_updates=0,
        training_examples=0,
        weights=[1.0, 2.0],
    )
    state = deepcopy(initial)
    state.completed_updates, state.training_examples = point, point * 54
    source = SimpleNamespace(snapshot_state=lambda: deepcopy(initial))
    candidate = SimpleNamespace(snapshot_state=lambda: deepcopy(state))
    benchmark = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0)
    slot = {"method": f"public_invariant_{method}", "checkpoint_updates": point, "report": None}
    before = pickle.dumps((state, initial, data, monitors))
    observed = []

    def snapshot(value, expected, bound_method, **options):
        assert bound_method == slot["method"]
        assert options == {"catalogue_rows": 54, "wide_neural": True}
        assert value.completed_updates == expected and value.training_examples == 54 * expected
        return deepcopy(value)

    def build(value, supplied, returned, selected):
        assert value is not state and pickle.dumps(value) == pickle.dumps(state)
        assert supplied is data and selected == mode
        assert len(returned) == 2 and all(packet.role == "train" for packet in returned)
        return candidate

    def execute(owner, supplied_state, training, evaluation, **options):
        assert owner is candidate and owner is not source
        assert supplied_state is state and training is data[0] and evaluation is data[1]
        assert options["checkpoint_updates"] == (0, 4, 8, 16)
        assert options["training_readouts"] is monitors
        wake, returned = options["budget"], options["return_budget"]
        assert wake is not returned and wake is not benchmark and returned is not benchmark
        assert wake.budget.max_training_updates == returned.budget.max_training_updates == 16
        assert len(options["return_training"]) == 2
        options["check_boundary"]()
        wake.record_update()
        observed.append((wake, returned))
        raise RuntimeError("scripted failure after one permanently charged update")

    monkeypatch.setattr(pilot, "_invariant_snapshot", snapshot)
    monkeypatch.setattr(pilot, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    monkeypatch.setattr(pilot.bounded_probe, "build_source", build)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", execute)
    # Memory admission is separately checked at the real runtime boundary; this
    # fixture scripts only the app callback and its permanent budget charges.
    monkeypatch.setattr(ToyBudgetSession, "attach_memory", lambda session, sampler: None)
    with pytest.raises(RuntimeError, match="scripted failure"):
        pilot._run_probe(
            slot,
            {"source": source, "row": {"method": slot["method"]}, "budget": benchmark},
            state,
            data,
            sampler=None,
            deadline=pilot.monotonic() + 60,
            check_boundary=lambda: None,
            training_readouts=monitors,
            return_to_a=True,
            catalogue_rows=54,
            wide_neural=True,
            probe_inference=mode,
        )
    assert len(observed) == 1
    assert slot["status"] == "failed" and slot["charged_updates"] == 1
    assert slot["return_charged_updates"] == 0 and slot["work_unknown"] is True
    assert observed[0][0].updates_completed == 1 and benchmark.updates_completed == 0
    assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert pickle.dumps((state, initial, data, monitors)) == before
