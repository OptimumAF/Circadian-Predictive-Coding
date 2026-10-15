"""Independent array/complete-wake-state selection and bounded continuation."""

from dataclasses import asdict, FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pytest

from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledParameters, evaluate_coupled, feedforward_latents
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_width_pruning import (
    CoupledPruneLineage,
    copy_coupled_model,
    prune_coupled_width,
)

POSITIONS = (((3,), 0), ((3, 4), 0), ((3, 4), 1), ((3, 4, 3, 4), 0), ((3, 4, 3, 4), 3))


def arrays(parameters):
    return (
        *parameters.hidden_weights,
        *parameters.hidden_biases,
        parameters.output_weight,
        parameters.output_bias,
    )


def model(family="pc", hidden=(3, 4)):
    precisions = tuple(0.7 + 0.2 * i for i in range(len(hidden)))
    result: CoupledPredictiveCodingNetwork
    if family == "pc":
        result = CoupledPredictiveCodingNetwork(2, hidden, seed=41, precisions=precisions)
    else:
        configuration = CoupledCircadianConfig(0.9, 0.2, 0.7, 1.0 if family == "neutral" else 0.2)
        result = CoupledCircadianPredictiveCodingNetwork(
            2, hidden, seed=41, precisions=precisions, config=configuration
        )
    parameters = result.snapshot_parameters()
    result.restore_parameters(
        replace(
            parameters,
            hidden_biases=tuple(
                np.full_like(b, 0.1 * (i + 1)) for i, b in enumerate(parameters.hidden_biases)
            ),
            output_bias=np.array([[-0.13]]),
        )
    )
    if type(result) is CoupledCircadianPredictiveCodingNetwork:
        snapshot = result.snapshot_state()
        assert snapshot.wake_state is not None
        chemistry = 0.3 + 0.2 * np.arange(hidden[-1], dtype=np.float64)
        gates = np.clip(
            np.exp(-result.config.plasticity_sensitivity * chemistry),
            result.config.min_plasticity,
            1.0,
        )
        result.restore_state(
            replace(
                snapshot,
                wake_state=replace(
                    snapshot.wake_state,
                    chemical_state=chemistry,
                    plasticity_state=gates,
                    wake_updates=7,
                ),
            )
        )
    return result


def selected_snapshot(source, layer, unit):
    snapshot = source.snapshot_state()
    p = snapshot.parameters
    weights, biases = [w.copy() for w in p.hidden_weights], [b.copy() for b in p.hidden_biases]
    weights[layer] = np.delete(weights[layer], unit, axis=1)
    biases[layer] = np.delete(biases[layer], unit, axis=1)
    outgoing = p.output_weight.copy()
    if layer + 1 < len(weights):
        weights[layer + 1] = np.delete(weights[layer + 1], unit, axis=0)
    else:
        outgoing = np.delete(outgoing, unit, axis=0)
    wake = snapshot.wake_state
    if wake is not None and layer + 1 == len(weights):
        wake = replace(
            wake,
            chemical_state=np.delete(wake.chemical_state, unit),
            plasticity_state=np.delete(wake.plasticity_state, unit),
        )
    return replace(
        snapshot,
        parameters=CoupledParameters(tuple(weights), tuple(biases), outgoing, p.output_bias.copy()),
        wake_state=wake,
    )


@pytest.mark.parametrize("family", ("pc", "cpc"))
@pytest.mark.parametrize("hidden,layer", POSITIONS)
def test_should_select_exact_parameters_biases_and_associated_complete_wake_state(
    family, hidden, layer
):
    source = model(family, hidden)
    original = source.snapshot_state()
    expected = selected_snapshot(source, layer, 1)
    change = prune_coupled_width(source, layer, 1)
    assert type(change.lineage) is CoupledPruneLineage
    assert type(change.lineage.format_version) is int and change.lineage.format_version == 1
    assert change.lineage.family == family
    assert change.lineage.input_dim == 2 and change.lineage.source_hidden_dims == hidden
    assert change.lineage.precisions == source.precisions
    assert change.lineage.layer_index == layer and change.lineage.unit_index == 1
    target_dims = tuple(width - (i == layer) for i, width in enumerate(hidden))
    assert change.lineage.target_hidden_dims == change.model.hidden_dims == target_dims
    assert type(change.model) is type(source)
    np.testing.assert_equal(asdict(change.model.snapshot_state()), asdict(expected))
    restored: CoupledPredictiveCodingNetwork
    if type(change.model) is CoupledCircadianPredictiveCodingNetwork:
        assert type(source) is CoupledCircadianPredictiveCodingNetwork
        assert change.model.config == source.config and change.model.config is not source.config
        assert change.model.wake_updates == 7
        restored = CoupledCircadianPredictiveCodingNetwork(
            2,
            target_dims,
            seed=0,
            precisions=source.precisions,
            config=replace(change.model.config),
        )
    else:
        restored = CoupledPredictiveCodingNetwork(
            2, target_dims, seed=0, precisions=source.precisions
        )
    restored.restore_state(change.model.snapshot_state())
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(expected))
    changed = change.model.snapshot_parameters()
    change.model.restore_parameters(replace(changed, output_bias=np.array([[99.0]])))
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(original))
    np.testing.assert_equal(asdict(restored.snapshot_state()), asdict(expected))
    if type(source) is CoupledCircadianPredictiveCodingNetwork:
        copied_cpc = cast(CoupledCircadianPredictiveCodingNetwork, change.model)
        old = copied_cpc.config.chemical_buildup_rate
        object.__setattr__(source.config, "chemical_buildup_rate", 99.0)
        assert copied_cpc.config.chemical_buildup_rate == old
    with pytest.raises(FrozenInstanceError):
        setattr(change.lineage, "unit_index", 0)


@pytest.mark.parametrize("family", ("pc", "cpc"))
def test_should_copy_all_state_without_shared_arrays_or_policy(family):
    source = model(family)
    original = source.snapshot_state()
    copied = copy_coupled_model(source)
    assert type(copied) is type(source) and copied is not source
    np.testing.assert_equal(asdict(copied.snapshot_state()), asdict(original))
    p = copied.snapshot_parameters()
    copied.restore_parameters(replace(p, output_bias=np.array([[5.0]])))
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(original))
    if type(source) is CoupledCircadianPredictiveCodingNetwork:
        assert type(copied) is CoupledCircadianPredictiveCodingNetwork
        assert copied.config is not source.config


@pytest.mark.parametrize("hidden", ((3,), (3, 4), (3, 4, 3, 4)))
def test_should_continue_pc_and_neutral_cpc_with_exact_parameter_parity(hidden):
    pc = prune_coupled_width(model("pc", hidden), len(hidden) - 1, 1).model
    neutral = prune_coupled_width(model("neutral", hidden), len(hidden) - 1, 1).model
    x, y = np.array([[0.3, -0.4], [0.7, 0.2]]), np.array([[0.0], [0.35]])
    assert type(neutral) is CoupledCircadianPredictiveCodingNetwork
    for _ in range(2):
        ordinary = pc.train_epoch(x, y, 0.03, 2, 0.1)
        controlled = neutral.train_epoch(x, y, 0.03, 2, 0.1)
        assert ordinary.energy == controlled.energy
        for a, b in zip(
            arrays(pc.snapshot_parameters()), arrays(neutral.snapshot_parameters()), strict=True
        ):
            np.testing.assert_array_equal(a, b)
    assert neutral.wake_updates == 9
    np.testing.assert_array_equal(neutral.get_plasticity_state(), np.ones(hidden[-1] - 1))
    assert np.all(neutral.get_chemical_state() > 0)


@pytest.mark.parametrize("layer", (0, 1))
def test_should_use_selected_active_chemistry_for_the_next_real_gated_update(layer):
    source = model("cpc", (3, 4))
    pruned = prune_coupled_width(source, layer, 1).model
    assert type(pruned) is CoupledCircadianPredictiveCodingNetwork
    p = pruned.snapshot_parameters()
    x, y = np.array([[0.3, -0.4], [0.7, 0.2]]), np.array([[0.0], [0.35]])
    latents = feedforward_latents(p, x)
    for _ in range(2):
        evaluation = evaluate_coupled(p, x, y, latents, pruned.precisions)
        latents = tuple(
            z - 0.1 * g for z, g in zip(latents, evaluation.latent_gradients, strict=True)
        )
    evaluation = evaluate_coupled(p, x, y, latents, pruned.precisions)
    chemistry = (
        pruned.config.chemical_decay * pruned.get_chemical_state()
        + pruned.config.chemical_buildup_rate * np.mean(np.abs(latents[-1]), axis=0)
    )
    gates = np.clip(
        np.exp(-pruned.config.plasticity_sensitivity * chemistry), pruned.config.min_plasticity, 1.0
    )
    g = evaluation.parameter_gradients
    expected = CoupledParameters(
        tuple(
            w - 0.03 * gradient * (gates if i == len(p.hidden_weights) - 1 else 1)
            for i, (w, gradient) in enumerate(zip(p.hidden_weights, g.hidden_weights, strict=True))
        ),
        tuple(
            b - 0.03 * gradient * (gates if i == len(p.hidden_biases) - 1 else 1)
            for i, (b, gradient) in enumerate(zip(p.hidden_biases, g.hidden_biases, strict=True))
        ),
        p.output_weight - 0.03 * g.output_weight * gates[:, None],
        p.output_bias - 0.03 * g.output_bias,
    )
    pruned.train_epoch(x, y, 0.03, 2, 0.1)
    for a, b in zip(arrays(pruned.snapshot_parameters()), arrays(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-13)
    np.testing.assert_array_equal(pruned.get_chemical_state(), chemistry)
    np.testing.assert_array_equal(pruned.get_plasticity_state(), gates)
    assert pruned.wake_updates == 8


@pytest.mark.parametrize(
    "layer,unit",
    (
        (True, 0),
        (-1, 0),
        (2, 0),
        (1.0, 0),
        (np.int64(0), 0),
        (0, True),
        (0, -1),
        (0, 3),
        (0, 1.0),
        (0, np.int64(1)),
    ),
)
def test_should_refuse_non_native_or_out_of_range_coordinates(layer: Any, unit: Any):
    with pytest.raises(ValueError):
        prune_coupled_width(model(), layer, unit)


def test_should_refuse_width_one_and_foreign_native_subclass():
    with pytest.raises(ValueError):
        prune_coupled_width(CoupledPredictiveCodingNetwork(2, (1,), seed=41), 0, 0)

    class Unsupported(CoupledPredictiveCodingNetwork):
        pass

    bad: Any
    for bad in (None, {}, Unsupported(2, (3,), seed=41)):
        with pytest.raises(ValueError):
            copy_coupled_model(bad)
        with pytest.raises(ValueError):
            prune_coupled_width(bad, 0, 0)


@pytest.mark.parametrize("bound", ("depth", "parameters"))
def test_should_check_geometry_caps_before_snapshot_or_allocation(monkeypatch, bound):
    source = CoupledPredictiveCodingNetwork(2, (2,) * 17 if bound == "depth" else (3,), seed=41)

    def forbidden_snapshot():
        raise AssertionError("oversized geometry reached snapshot")

    monkeypatch.setattr(source, "snapshot_state", forbidden_snapshot)
    if bound == "parameters":
        monkeypatch.setattr(
            CoupledPredictiveCodingNetwork, "input_dim", property(lambda self: 1_000_001)
        )
    with pytest.raises(ValueError):
        copy_coupled_model(source)
    with pytest.raises(ValueError):
        prune_coupled_width(source, 0, 0)


@pytest.mark.parametrize(
    "kind",
    (
        "format",
        "precision",
        "removed_nan",
        "dtype",
        "negative_chemistry",
        "inconsistent_gate",
        "wake_count",
    ),
)
def test_should_validate_complete_source_snapshot_before_removing_bad_coordinate(monkeypatch, kind):
    source = model(
        "cpc" if kind in ("negative_chemistry", "inconsistent_gate", "wake_count") else "pc"
    )
    snapshot = source.snapshot_state()
    if kind == "format":
        snapshot = replace(snapshot, format_version=True)
    elif kind == "precision":
        snapshot = replace(snapshot, precisions=(True, 0.9))
    elif kind == "removed_nan":
        snapshot.parameters.hidden_weights[0][0, 1] = np.nan
    elif kind == "dtype":
        snapshot = replace(
            snapshot,
            parameters=replace(
                snapshot.parameters,
                hidden_weights=(
                    snapshot.parameters.hidden_weights[0].astype(np.float32),
                    snapshot.parameters.hidden_weights[1],
                ),
            ),
        )
    else:
        assert snapshot.wake_state is not None
        wake = snapshot.wake_state
        if kind == "negative_chemistry":
            wake.chemical_state[1] = -1
        elif kind == "inconsistent_gate":
            wake.plasticity_state[1] = 0.0
        else:
            snapshot = replace(snapshot, wake_state=replace(wake, wake_updates=True))
    monkeypatch.setattr(source, "snapshot_state", lambda: snapshot)
    with pytest.raises(ValueError):
        copy_coupled_model(source)
    with pytest.raises(ValueError):
        prune_coupled_width(
            source, 1 if kind in ("negative_chemistry", "inconsistent_gate", "wake_count") else 0, 1
        )
