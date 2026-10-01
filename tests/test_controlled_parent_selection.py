"""Bounded core fixtures for parent choice, eligibility and full transactions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.app.continual_combined_factor_preflight import _canonical_state
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    HiddenWidthLimitExceeded,
)
from src.core.controlled_parent_selection import (
    ParentControlledCircadianNetwork,
    ParentSelectionSettings,
    SelectionMode,
)
from src.core.neuron_adaptation import LayerTraffic, NeuronChangeProposal
from src.core.sleep_clocks import SleepEpochProgress

MODES: tuple[SelectionMode, ...] = ("usage", "scheduled", "random")
FEATURES = np.array([[0.2, -0.3], [-0.1, 0.4], [0.5, 0.1], [-0.4, -0.2]])
TARGETS = np.array([[1.0], [0.0], [1.0], [0.0]])


def _config(**changes: Any) -> CircadianConfig:
    return replace(
        CircadianConfig(
            sleep_mode="components",
            sleep_enable_homeostasis=False,
            sleep_enable_chemical_reset=False,
            sleep_enable_replay=False,
            replay_memory_size=0,
            max_split_per_sleep=2,
            max_prune_per_sleep=2,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            split_noise_scale=0.02,
        ),
        **changes,
    )


def _model(
    mode: SelectionMode = "usage",
    *,
    config: CircadianConfig | None = None,
    selector_seed: int = 547,
    cursor: int = 0,
    minimum: int = 1,
    maximum: int = 12,
) -> ParentControlledCircadianNetwork:
    return ParentControlledCircadianNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=521,
        circadian_config=config or _config(),
        min_hidden_dim=minimum,
        max_hidden_dim=maximum,
        parent_selection=ParentSelectionSettings(mode, selector_seed, cursor),
    )


class _FixedPolicy:
    def __init__(self, proposal: NeuronChangeProposal) -> None:
        self.proposal = proposal

    def propose(self, traffic_by_layer: list[LayerTraffic]) -> list[NeuronChangeProposal]:
        return [self.proposal]


def _apply(model: ParentControlledCircadianNetwork, add: int = 1) -> None:
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", add_count=add)])


def _sleep(model: ParentControlledCircadianNetwork) -> None:
    model.sleep_event(adaptation_policy=_FixedPolicy(NeuronChangeProposal("hidden", add_count=1)))


def _state(model: CircadianPredictiveCodingNetwork) -> Any:
    # Every copied field participates, including replay containers and both RNGs.
    return _canonical_state(model.snapshot_state())


@pytest.mark.parametrize(
    ("mode", "seed", "cursor"),
    [
        ("unsupported", 547, 0),
        (None, 547, 0),
        ("usage", -1, 0),
        ("usage", True, 0),
        ("usage", 1.5, 0),
        ("usage", np.int64(547), 0),
        ("usage", 547, -1),
        ("usage", 547, True),
        ("usage", 547, 1.5),
        ("usage", 547, np.int64(0)),
    ],
)
def test_should_reject_settings_before_model_setup(
    mode: Any, seed: Any, cursor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_setup(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid settings reached model setup")

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "__init__", forbidden_setup)
    with pytest.raises(ValueError, match="parent selection"):
        ParentControlledCircadianNetwork(
            2, 4, 521, parent_selection=ParentSelectionSettings(mode, seed, cursor)
        )


def test_should_match_every_original_core_field_in_usage_mode() -> None:
    original = CircadianPredictiveCodingNetwork(2, 4, 521, _config(), 1, 12)
    controlled = _model()

    def assert_core_equal() -> None:
        old = original.snapshot_state().state
        new = controlled.snapshot_state().state
        assert _canonical_state(old) == _canonical_state({key: new[key] for key in old})

    assert_core_equal()
    for model in (original, controlled):
        model.set_chemical_state(np.array([0.95, 0.9, 0.2, 0.1]))
        model.sleep_event(
            adaptation_policy=_FixedPolicy(
                NeuronChangeProposal("hidden", add_count=1, remove_indices=(3,))
            )
        )
    assert controlled.get_parent_selection_state().last_decision is not None
    assert_core_equal()
    for _ in range(3):
        for model in (original, controlled):
            model.train_epoch(FEATURES, TARGETS, 0.03, 2, 0.2)
        assert_core_equal()
    for model in (original, controlled):
        model.apply_neuron_proposals(
            [NeuronChangeProposal("hidden", add_count=1, remove_indices=(0,))]
        )
    assert_core_equal()


def test_should_follow_stable_ids_after_prune_and_wrap_cyclic_cursor() -> None:
    model = _model("scheduled")
    model._neuron_ids = np.array([20, 3, 11, 7], dtype=np.int64)
    model._next_neuron_id = 21
    _apply(model)
    first = model.get_parent_selection_state()
    assert first.last_decision is not None
    assert first.last_decision.selected_parent_ids == (3,)
    assert first.cursor_id == 4
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", remove_indices=(1,))])
    assert model.get_parent_selection_state() == first
    assert model.get_neuron_lineage().parent_ids[-1] == 3
    _apply(model)
    second = model.get_parent_selection_state()
    assert second.last_decision is not None
    assert second.last_decision.selected_parent_ids == (7,)
    assert second.cursor_id == 8
    assert model.get_neuron_lineage().parent_ids[-1] == 7
    wrapped = _model("scheduled", cursor=100)
    _apply(wrapped)
    decision = wrapped.get_parent_selection_state().last_decision
    assert decision is not None and decision.selected_parent_ids == (0,)
    assert wrapped.get_parent_selection_state().cursor_id == 1


@pytest.mark.parametrize("mode", MODES)
def test_should_fill_count_from_preferred_then_eligible_fallback(mode: SelectionMode) -> None:
    model = _model(mode)
    model.set_chemical_state(np.array([0.2, 0.3, 0.4, 0.9]))
    _apply(model, 2)
    state = model.get_parent_selection_state()
    decision = state.last_decision
    assert decision is not None
    assert decision.eligible_parent_ids == (0, 1, 2, 3)
    assert decision.preferred_parent_ids == (3,)
    assert decision.selected_parent_ids[0] == 3
    assert decision.selected_parent_ids[1] in (0, 1, 2)
    assert len(set(decision.selected_parent_ids)) == 2
    assert model.get_neuron_lineage().parent_ids[-2:] == decision.selected_parent_ids
    if mode == "usage":
        assert decision.selected_parent_ids == (3, 2)
    elif mode == "scheduled":
        assert decision.selected_parent_ids == (3, 0)


@pytest.mark.parametrize("mode", MODES)
def test_should_preserve_predictions_lineage_and_core_noise_draw_count(mode: SelectionMode) -> None:
    model, reference = _model(mode), _model("usage")
    before = model.predict_proba(FEATURES)
    _apply(model, 2)
    _apply(reference, 2)
    np.testing.assert_allclose(model.predict_proba(FEATURES), before, rtol=0, atol=1e-15)
    assert model._rng.bit_generator.state == reference._rng.bit_generator.state
    decision = model.get_parent_selection_state().last_decision
    assert decision is not None
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 2, 3, 4, 5)
    assert model.get_neuron_lineage().parent_ids[-2:] == decision.selected_parent_ids


def test_should_repeat_random_choices_and_cover_allowed_parents_under_fixed_unit_seeds() -> None:
    first, second = _model("random"), _model("random")
    _apply(first, 2)
    _apply(second, 2)
    assert _state(first) == _state(second)
    decision = first.get_parent_selection_state().last_decision
    assert decision is not None and decision.rng_sha256_before != decision.rng_sha256_after
    parents: set[int] = set()
    for seed in range(32):
        model = _model("random", selector_seed=seed)
        _apply(model)
        chosen = model.get_parent_selection_state().last_decision
        assert chosen is not None
        assert set(chosen.selected_parent_ids).issubset({0, 1, 2, 3})
        parents.update(chosen.selected_parent_ids)
    assert parents == {0, 1, 2, 3}


@pytest.mark.parametrize("mode", MODES)
def test_should_exclude_pruned_pending_and_cooling_parents(mode: SelectionMode) -> None:
    model = _model(mode, config=_config(prune_decay_steps=3))
    model._split_cooldown[0] = 2
    model._prune_marked[1] = True
    model._prune_ttl[1] = 3
    model.apply_neuron_proposals([NeuronChangeProposal("hidden", add_count=1, remove_indices=(3,))])
    decision = model.get_parent_selection_state().last_decision
    assert decision is not None and decision.eligible_parent_ids == (2,)
    assert decision.selected_parent_ids == (2,)
    assert model.get_pending_prune_ids() == (1, 3)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "case",
    [
        "count",
        "split_cap",
        "prune_cap",
        "maximum",
        "minimum",
        "fraction",
        "cooldown",
        "age",
        "index",
    ],
)
def test_should_reject_original_constraints_before_selection(
    mode: SelectionMode, case: str
) -> None:
    config = _config()
    if case == "split_cap":
        config = _config(max_split_per_sleep=1)
    elif case == "prune_cap":
        config = _config(max_prune_per_sleep=0)
    elif case == "fraction":
        config = _config(sleep_max_change_fraction=0.2, sleep_min_change_count=0)
    elif case == "age":
        config = _config(prune_min_age_steps=2)
    model = _model(
        mode,
        config=config,
        maximum=4 if case == "maximum" else 12,
        minimum=4 if case == "minimum" else 1,
    )
    if case == "cooldown":
        model._split_cooldown[:] = 2
    proposals = {
        "count": NeuronChangeProposal("hidden", add_count=-1),
        "split_cap": NeuronChangeProposal("hidden", add_count=2),
        "prune_cap": NeuronChangeProposal("hidden", remove_indices=(0,)),
        "maximum": NeuronChangeProposal("hidden", add_count=1),
        "minimum": NeuronChangeProposal("hidden", remove_indices=(0,)),
        "fraction": NeuronChangeProposal("hidden", add_count=1),
        "cooldown": NeuronChangeProposal("hidden", add_count=1),
        "age": NeuronChangeProposal("hidden", remove_indices=(0,)),
        "index": NeuronChangeProposal("hidden", remove_indices=(4,)),
    }
    before = _state(model)
    with pytest.raises(ValueError, match="proposal"):
        model.apply_neuron_proposals([proposals[case]])
    assert _state(model) == before


@pytest.mark.parametrize("mode", MODES)
def test_should_respect_phase_caps_noops_and_explicit_policy_requirement(
    mode: SelectionMode,
) -> None:
    model = _model(mode)
    before = _state(model)
    model.apply_neuron_proposals([NeuronChangeProposal("hidden")])
    assert _state(model) == before
    with pytest.raises(ValueError, match="explicit adaptation policy"):
        model.sleep_event()
    assert _state(model) == before
    with pytest.raises(ValueError, match="budget"):
        model.sleep_event(
            adaptation_policy=_FixedPolicy(NeuronChangeProposal("hidden", add_count=1)),
            epoch_progress=SleepEpochProgress(9, 10),
        )
    assert _state(model) == before
    selector = model.get_parent_selection_state()
    model.sleep_event(adaptation_policy=_FixedPolicy(NeuronChangeProposal("hidden")))
    assert model.get_parent_selection_state() == selector
    disabled = _model(mode, config=_config(sleep_mode="disabled"))
    before_disabled = _state(disabled)
    assert not disabled.sleep_event().performed
    assert _state(disabled) == before_disabled
    prune_only = _model(mode, config=_config(max_split_per_sleep=0))
    selector = prune_only.get_parent_selection_state()
    prune_only.sleep_event()
    assert prune_only.get_parent_selection_state() == selector


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sleep", [False, True])
def test_should_restore_selector_and_future_retry_after_transient_width_failure(
    mode: SelectionMode, sleep: bool
) -> None:
    model, reference = _model(mode), _model(mode)
    before = _state(model)
    with pytest.raises(HiddenWidthLimitExceeded):
        if sleep:
            model.sleep_event(
                adaptation_policy=_FixedPolicy(NeuronChangeProposal("hidden", add_count=1)),
                max_hidden_width=4,
            )
        else:
            model.apply_neuron_proposals(
                [NeuronChangeProposal("hidden", add_count=1)], max_hidden_width=4
            )
    assert _state(model) == before
    action = _sleep if sleep else _apply
    action(model)
    action(reference)
    assert _state(model) == _state(reference)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sleep", [False, True])
def test_should_restore_every_field_after_failure_following_split_mutation(
    mode: SelectionMode, sleep: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, reference = _model(mode), _model(mode)
    before = _state(model)

    def fail_after_split(self: ParentControlledCircadianNetwork, indices: tuple[int, ...]) -> None:
        assert self.hidden_dim == 5
        assert self.get_parent_selection_state().selection_calls == 1
        raise RuntimeError("injected after split")

    action = _sleep if sleep else _apply
    with monkeypatch.context() as patch:
        patch.setattr(ParentControlledCircadianNetwork, "_schedule_or_prune", fail_after_split)
        with pytest.raises(RuntimeError, match="injected after split"):
            action(model)
    assert _state(model) == before
    action(model)
    action(reference)
    assert _state(model) == _state(reference)


@pytest.mark.parametrize("mode", MODES)
def test_should_restore_guard_rejection_and_retain_separate_proposed_ids(
    mode: SelectionMode,
) -> None:
    model, reference = _model(mode), _model(mode)
    saved = model.snapshot_state()
    event = model.sleep_event(
        adaptation_policy=_FixedPolicy(NeuronChangeProposal("hidden", add_count=1))
    )
    decision = model.get_parent_selection_state().last_decision
    assert decision is not None and event.telemetry is not None
    proposed = event.telemetry.changes.proposed_split_pairs
    assert proposed == ((decision.selected_parent_ids[0], 4),)
    model.restore_state(saved)
    assert _state(model) == _state(reference)
    assert proposed == ((decision.selected_parent_ids[0], 4),)
    _sleep(model)
    _sleep(reference)
    assert model.get_parent_selection_state().last_decision == decision
    assert _state(model) == _state(reference)


@pytest.mark.parametrize("mode", MODES)
def test_should_replay_snapshot_through_wake_and_next_split(mode: SelectionMode) -> None:
    model = _model(mode)
    _apply(model)
    saved = model.snapshot_state()
    resumed = _model(mode)
    resumed.restore_state(saved)
    for network in (model, resumed):
        network.apply_neuron_proposals([NeuronChangeProposal("hidden", remove_indices=(0,))])
        network.train_epoch(FEATURES, TARGETS, 0.03, 2, 0.2)
        _sleep(network)
    assert _state(model) == _state(resumed)
    assert _canonical_state(saved) != _state(model)


@pytest.mark.parametrize(
    "settings",
    [
        ParentSelectionSettings("usage", 547),
        ParentSelectionSettings("random", 548),
        ParentSelectionSettings("random", 547, 1),
    ],
)
def test_should_refuse_incompatible_snapshot_settings_without_mutation(
    settings: ParentSelectionSettings,
) -> None:
    source = _model("random")
    target = _model(settings.mode, selector_seed=settings.seed, cursor=settings.initial_cursor_id)
    before = _state(target)
    with pytest.raises(ValueError, match="incompatible"):
        target.restore_state(source.snapshot_state())
    assert _state(target) == before


@pytest.mark.parametrize(
    "case",
    [
        "rng",
        "cursor",
        "calls",
        "decision",
        "duplicate",
        "selected",
        "before_hash",
        "after_hash",
        "after_cursor",
    ],
)
def test_should_refuse_malformed_selector_snapshot_without_mutation(case: str) -> None:
    model = _model("random")
    _apply(model)
    saved = model.snapshot_state()
    before = _state(model)
    decision = model.get_parent_selection_state().last_decision
    assert decision is not None
    invalid_hash: Any = None
    changes: dict[str, dict[str, Any]] = {
        "rng": {"_parent_selection_rng": None},
        "cursor": {"_parent_selection_cursor": True},
        "calls": {"_parent_selection_calls": -1},
        "decision": {"_last_parent_selection": None},
        "duplicate": {"_last_parent_selection": replace(decision, eligible_parent_ids=(0, 0))},
        "selected": {"_last_parent_selection": replace(decision, selected_parent_ids=(99,))},
        "before_hash": {
            "_last_parent_selection": replace(decision, rng_sha256_before=invalid_hash)
        },
        "after_hash": {"_last_parent_selection": replace(decision, rng_sha256_after="0" * 64)},
        "after_cursor": {"_last_parent_selection": replace(decision, cursor_after=True)},
    }
    malformed = replace(saved, state={**saved.state, **changes[case]})
    with pytest.raises(ValueError, match="parent selection"):
        model.restore_state(malformed)
    assert _state(model) == before


def test_should_reject_changed_initial_random_stream_without_mutation() -> None:
    model = _model("random")
    saved = model.snapshot_state()
    malformed = replace(
        saved,
        state={**saved.state, "_parent_selection_rng": np.random.Generator(np.random.PCG64(548))},
    )
    before = _state(model)
    with pytest.raises(ValueError, match="RNG changed"):
        model.restore_state(malformed)
    assert _state(model) == before
