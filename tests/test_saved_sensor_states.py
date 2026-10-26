"""Restore saved full sensor states without training, scoring or diagnostics."""

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledSnapshot
from src.core.matched_backprop import MatchedBackpropNetwork, MatchedBackpropSnapshot


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/r46-contrast-inversion-20261017/native-trajectories.json"
HELPER = ROOT / "artifacts/runs/r21-saved-inversion-probes-20261017/saved_states.py"
METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def forbidden(*args, **kwargs):
    pytest.fail("qualification must not train, predict or diagnose native states")


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location("saved_sensor_states", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    for kind in (
        MatchedBackpropNetwork,
        CoupledPredictiveCodingNetwork,
        CoupledCircadianPredictiveCodingNetwork,
    ):
        monkeypatch.setattr(kind, "train_epoch", forbidden)
        monkeypatch.setattr(kind, "predict_proba", forbidden)
    import src.core.coupled_energy as energy
    import src.core.coupled_predictive_coding as pc
    import src.core.coupled_circadian_predictive_coding as cpc
    import src.core.matched_backprop as bp
    import src.core.learnability_diagnostics as diagnostics

    for module in (energy, pc, cpc, bp):
        for name in ("feedforward_latents", "predict_coupled", "evaluate_coupled"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(diagnostics, "diagnose_activations", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)


def document():
    return json.loads(SOURCE.read_text(encoding="utf8"))


def selected(helper, method):
    return next(
        record
        for record in helper.select_inputs(document())["methods"]
        if record["method"] == method
    )


def wire(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return wire(asdict(value))
    if type(value) is np.ndarray:
        return value.tolist()
    if type(value) is dict:
        return {key: wire(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [wire(item) for item in value]
    return value


def arrays(state):
    parameters = state.parameters
    result = [
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    ]
    if type(state) is CoupledSnapshot and state.wake_state is not None:
        result.extend([state.wake_state.chemical_state, state.wake_state.plasticity_state])
    return result


def test_should_select_only_four_methods_initial_laws_and_fixed_full_snapshots(helper):
    source = document()
    before = deepcopy(source)
    result = helper.select_inputs(source)
    assert set(result) == {"protocol", "methods"}
    assert result["protocol"] == "saved_sensor_inversion_probes_v1"
    assert tuple(record["method"] for record in result["methods"]) == METHODS
    assert source == before
    for record in result["methods"]:
        assert set(record) == {"method", "initial_parameters", "snapshots"}
        raw = next(case for case in source["cases"] if case["key"] == record["method"])
        assert record["initial_parameters"] == raw["initial_parameters"]
        assert record["snapshots"] == [
            {"updates": n, "full_state": raw["updates"][n - 1]["full_state"]} for n in (4, 8, 12)
        ]
        assert all(set(point) == {"updates", "full_state"} for point in record["snapshots"])
    encoded = json.dumps(result)
    assert "final_probabilities" not in encoded
    assert "final_update_counts" not in encoded
    assert "probabilities" not in encoded
    assert "physical_work" not in encoded
    assert "parameter_gradients" not in encoded and '"scores"' not in encoded


def test_should_detach_selected_inputs_from_raw_training_history(helper):
    source = document()
    result = helper.select_inputs(source)
    source["cases"][0]["initial_parameters"]["output_bias"][0][0] = 999
    source["cases"][0]["updates"][3]["full_state"]["parameters"]["output_bias"][0][0] = 999
    assert result["methods"][0]["initial_parameters"]["output_bias"] != [[999]]
    assert result["methods"][0]["snapshots"][0]["full_state"]["parameters"]["output_bias"] != [
        [999]
    ]


def test_should_never_inspect_or_project_final_predictions_or_update_results(helper):
    class Poison:
        def __deepcopy__(self, memo):
            pytest.fail("excluded evaluation or diagnostic field was copied")

    source = document()
    for case in source["cases"]:
        case["final_probabilities"] = Poison()
        case["final_update_counts"] = Poison()
        case["probabilities"] = Poison()
        case["physical_work"] = Poison()
        for update in case["updates"]:
            update["result"] = Poison()
    result = helper.select_inputs(source)
    assert tuple(record["method"] for record in result["methods"]) == METHODS


@pytest.mark.parametrize("method", METHODS)
def test_should_restore_exact_native_saved_states_and_canonical_initial_auxiliary_state(
    helper, method
):
    record = selected(helper, method)
    before = deepcopy(record)
    template, states = helper.restore_case(record)
    assert type(states) is tuple and [n for n, _ in states] == [0, 4, 8, 12]
    expected_type = MatchedBackpropSnapshot if method == "bp" else CoupledSnapshot
    assert all(type(state) is expected_type for _, state in states)
    assert wire(states[0][1].parameters) == record["initial_parameters"]
    assert wire(template.snapshot_state()) == wire(states[0][1])
    for (n, state), raw in zip(states[1:], record["snapshots"], strict=True):
        assert n == raw["updates"] and wire(state) == raw["full_state"]
        fork = template.fork()
        fork.restore_state(state)
        assert wire(fork.snapshot_state()) == raw["full_state"]
    assert record == before
    if method != "bp":
        assert all(state.precisions == (1.0,) for _, state in states)
        if method == "pc":
            assert all(state.wake_state is None for _, state in states)
        else:
            initial = states[0][1].wake_state
            assert initial.wake_updates == 0
            np.testing.assert_array_equal(initial.chemical_state, np.zeros(8))
            np.testing.assert_array_equal(initial.plasticity_state, np.ones(8))
            assert [state.wake_state.wake_updates for _, state in states] == [0, 4, 8, 12]
            assert initial.min_plasticity == (1.0 if method == "neutral_cpc" else 0.2)


@pytest.mark.parametrize("method", METHODS)
def test_should_own_all_restored_arrays_independently_of_other_states_template_and_input(
    helper, method
):
    record = selected(helper, method)
    template, states = helper.restore_case(record)
    expected = [wire(state) for _, state in states]
    template_before = wire(template.snapshot_state())
    native_arrays = [array for _, state in states for array in arrays(state)]
    assert all(
        not np.shares_memory(left, right)
        for i, left in enumerate(native_arrays)
        for right in native_arrays[i + 1 :]
    )
    states[1][1].parameters.hidden_weights[0].fill(999)
    assert wire(states[0][1]) == expected[0]
    assert wire(states[2][1]) == expected[2] and wire(states[3][1]) == expected[3]
    assert wire(template.snapshot_state()) == template_before
    assert record["snapshots"][0]["full_state"] == expected[1]
    record["snapshots"][1]["full_state"]["parameters"]["output_bias"][0][0] = 888
    assert wire(states[2][1]) == expected[2]


@pytest.mark.parametrize(
    "fault",
    ["missing_method", "duplicate_method", "unexpected_method", "missing_history", "short_history"],
)
def test_should_refuse_missing_duplicate_or_unknown_source_records(helper, fault):
    source = document()
    if fault == "missing_method":
        source["cases"] = [case for case in source["cases"] if case["key"] != "pc"]
    elif fault == "duplicate_method":
        source["cases"].append(deepcopy(source["cases"][0]))
    elif fault == "unexpected_method":
        source["cases"].append(dict(key="unqualified_control", updates=[]))
    elif fault == "missing_history":
        del source["cases"][0]["updates"]
    else:
        source["cases"][0]["updates"] = source["cases"][0]["updates"][:8]
    with pytest.raises(ValueError):
        helper.select_inputs(source)


@pytest.mark.parametrize(
    "fault",
    [
        "method",
        "extra_record_field",
        "missing_parameters",
        "missing_snapshot",
        "duplicate_snapshot",
        "reordered_snapshot",
        "extra_snapshot_field",
        "unexpected_state_field",
    ],
)
def test_should_refuse_corrupt_selected_record_structure(helper, fault):
    record = selected(helper, "pc")
    if fault == "method":
        record["method"] = "scheduled"
    elif fault == "extra_record_field":
        record["final_scores"] = {"bce": 0.1}
    elif fault == "missing_parameters":
        del record["initial_parameters"]
    elif fault == "missing_snapshot":
        record["snapshots"].pop()
    elif fault == "duplicate_snapshot":
        record["snapshots"][1] = deepcopy(record["snapshots"][0])
    elif fault == "reordered_snapshot":
        record["snapshots"].reverse()
    elif fault == "extra_snapshot_field":
        record["snapshots"][0]["score"] = 1
    else:
        record["snapshots"][0]["full_state"]["unknown_state"] = 1
    with pytest.raises(ValueError):
        helper.restore_case(record)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fault", ["shape", "nan", "infinity", "version"])
def test_should_refuse_malformed_saved_native_parameters_without_repairing_them(
    helper, method, fault
):
    record = selected(helper, method)
    state = record["snapshots"][0]["full_state"]
    if fault == "shape":
        state["parameters"]["hidden_weights"][0].pop()
    elif fault == "version":
        state["format_version"] = 2
    else:
        state["parameters"]["output_bias"][0][0] = float("nan" if fault == "nan" else "inf")
    with pytest.raises(ValueError):
        helper.restore_case(record)


@pytest.mark.parametrize("method", METHODS)
def test_should_refuse_nonfinite_saved_initial_law_before_restoring_later_states(helper, method):
    record = selected(helper, method)
    record["initial_parameters"]["output_weight"][0][0] = float("nan")
    with pytest.raises(ValueError):
        helper.restore_case(record)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "gated_cpc"])
def test_should_refuse_an_unmatched_saved_precision(helper, method):
    record = selected(helper, method)
    record["snapshots"][0]["full_state"]["precisions"] = [0.5]
    with pytest.raises(ValueError):
        helper.restore_case(record)


@pytest.mark.parametrize("method", ["neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("fault", ["policy", "counter", "chemical_shape", "gate_nonfinite"])
def test_should_refuse_corrupt_cpc_policy_clocks_or_auxiliary_arrays(helper, method, fault):
    record = selected(helper, method)
    wake = record["snapshots"][0]["full_state"]["wake_state"]
    if fault == "policy":
        wake["chemical_decay"] = 0.5
    elif fault == "counter":
        wake["wake_updates"] = 5
    elif fault == "chemical_shape":
        wake["chemical_state"].pop()
    else:
        wake["plasticity_state"][0] = float("nan")
    with pytest.raises(ValueError):
        helper.restore_case(record)
