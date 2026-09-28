"""Built-in NumPy sleep proposals are validated on the original topology."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork


def _model(
    config: CircadianConfig, *, width: int = 4, minimum: int = 3, maximum: int = 5
) -> CircadianPredictiveCodingNetwork:
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=width,
        seed=467,
        circadian_config=config,
        min_hidden_dim=minimum,
        max_hidden_dim=maximum,
    )


def _config(**changes: Any) -> CircadianConfig:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_replay=False,
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=False,
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        split_threshold=0.5,
        prune_threshold=0.5,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
        split_noise_scale=0.1,
    )
    options.update(changes)
    return CircadianConfig(**options)


def _assert_live_state_unchanged(model: CircadianPredictiveCodingNetwork, saved: Any) -> None:
    assert model.hidden_dim == saved.state["weight_input_hidden"].shape[1]
    assert model._rng.bit_generator.state == saved.state["_rng"].bit_generator.state
    for name in (
        "weight_input_hidden",
        "weight_hidden_output",
        "bias_hidden",
        "_hidden_chemical",
        "_neuron_age",
        "_prune_marked",
        "_prune_ttl",
        "_split_cooldown",
        "_prune_cooldown",
    ):
        np.testing.assert_array_equal(getattr(model, name), saved.state[name])
    assert model._sleep_events == saved.state["_sleep_events"]
    assert model._energy_history == saved.state["_energy_history"]


def test_builtin_overlap_prunes_original_and_splits_a_different_source() -> None:
    model = _model(_config())
    model.set_chemical_state(np.full(4, 0.5))

    event = model.sleep_event(force_sleep=True)

    assert event.split_indices == (2,)
    assert event.pruned_indices == (3,)
    assert event.new_hidden_dim == 4
    model._validate_training_topology()


def test_pending_gradual_prune_reserves_minimum_width_capacity() -> None:
    model = _model(
        _config(max_split_per_sleep=0, prune_threshold=0.2, prune_decay_steps=3),
        width=5,
        minimum=4,
        maximum=6,
    )
    model.set_chemical_state(np.array([1.0, 1.0, 1.0, 0.0, 0.0]))
    model._prune_marked[4] = True
    model._prune_ttl[4] = 3

    event = model.sleep_event(force_sleep=True)

    assert event.pruned_indices == ()
    assert int(np.count_nonzero(model._prune_marked)) == 1


@pytest.mark.parametrize(
    ("selector", "indices", "expected"),
    [
        ("split", (99,), "split index"),
        ("split", (0, 1), "split budget"),
        ("prune", (99,), "prune index"),
        ("prune", (0, 1), "prune budget"),
    ],
)
def test_malformed_builtin_selection_rejects_before_noise_or_mutation(
    monkeypatch: pytest.MonkeyPatch, selector: str, indices: tuple[int, ...], expected: str
) -> None:
    model = _model(_config())
    model.set_chemical_state(np.array([0.9, 0.8, 0.2, 0.1]))
    method = "_select_split_indices" if selector == "split" else "_select_prune_indices"
    monkeypatch.setattr(model, method, lambda **_: indices)
    saved = model.snapshot_state()

    with pytest.raises(ValueError, match=expected):
        model.sleep_event(force_sleep=True)
    _assert_live_state_unchanged(model, saved)


def test_disjoint_builtin_selection_retains_existing_indices() -> None:
    model = _model(_config(split_threshold=0.8, prune_threshold=0.2))
    model.set_chemical_state(np.array([0.95, 0.6, 0.4, 0.0]))

    event = model.sleep_event(force_sleep=True)

    assert event.split_indices == (0,)
    assert event.pruned_indices == (3,)
    assert event.new_hidden_dim == 4


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate", "unique"),
        ("max_width", "maximum hidden width"),
        ("min_width", "minimum hidden width"),
        ("split_cooldown", "split index is ineligible"),
        ("prune_age", "prune index is ineligible"),
        ("prune_cooldown", "prune index is ineligible"),
        ("noninteger", "split index is out of range"),
    ],
)
def test_builtin_proposal_checks_width_eligibility_and_types_before_mutation(
    monkeypatch: pytest.MonkeyPatch, case: str, message: str
) -> None:
    config = _config(
        max_split_per_sleep=2 if case == "duplicate" else 1,
        prune_min_age_steps=2 if case == "prune_age" else 0,
    )
    model = _model(
        config,
        minimum=4 if case == "min_width" else 3,
        maximum=4 if case == "max_width" else 6,
    )
    model.set_chemical_state(np.array([0.9, 0.8, 0.2, 0.1]))
    if case == "split_cooldown":
        model._split_cooldown[0] = 2
    if case == "prune_cooldown":
        model._prune_cooldown[3] = 2
    if case in {"duplicate", "max_width", "split_cooldown", "noninteger"}:
        selected: tuple[Any, ...] = {
            "duplicate": (0, 0),
            "noninteger": (True,),
        }.get(case, (0,))
        monkeypatch.setattr(model, "_select_split_indices", lambda **_: selected)
    else:
        monkeypatch.setattr(model, "_select_split_indices", lambda **_: ())
        monkeypatch.setattr(model, "_select_prune_indices", lambda **_: (3,))
    saved = model.snapshot_state()

    with pytest.raises(ValueError, match=message):
        model.sleep_event(force_sleep=True)
    _assert_live_state_unchanged(model, saved)
