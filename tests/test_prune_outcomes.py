"""Prune results distinguish requests, delayed marks, and removals by ID."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.neuron_adaptation import NeuronChangeProposal
from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead


def _numpy_model(**changes: Any) -> CircadianPredictiveCodingNetwork:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=False,
        sleep_enable_replay=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=1,
        prune_threshold=0.2,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
    )
    options.update(changes)
    return CircadianPredictiveCodingNetwork(
        input_dim=2,
        hidden_dim=4,
        seed=829,
        circadian_config=CircadianConfig(**options),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )


def _torch_head(**changes: Any) -> CircadianPredictiveCodingHead:
    torch = pytest.importorskip("torch")
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=False,
        max_split_per_sleep=0,
        max_prune_per_sleep=1,
        prune_threshold=0.2,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
    )
    options.update(changes)
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=831,
        config=CircadianHeadConfig(**options),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )


def _wake(model: CircadianPredictiveCodingNetwork) -> Any:
    return model.train_epoch(
        np.array([[0.2, -0.3], [-0.1, 0.4]]),
        np.array([[1.0], [0.0]]),
        0.03,
        2,
        0.2,
    )


def test_numpy_immediate_sleep_reports_proposed_and_removed_ids() -> None:
    model = _numpy_model()
    model.set_chemical_state(np.array([0.5, 0.5, 0.0, 0.5]))

    event = model.sleep_event(force_sleep=True)

    assert event.pruned_indices == (2,)
    assert event.prune_outcome is not None
    assert event.prune_outcome.proposed_neuron_ids == (2,)
    assert event.prune_outcome.scheduled_neuron_ids == ()
    assert event.prune_outcome.removed_neuron_ids == (2,)
    assert model.get_pending_prune_ids() == ()
    with pytest.raises(FrozenInstanceError):
        setattr(event.prune_outcome, "removed_neuron_ids", ())


def test_numpy_gradual_sleep_and_wake_report_distinct_outcomes_with_restore() -> None:
    model = _numpy_model(prune_decay_steps=2, prune_decay_factor=0.5)
    model.set_chemical_state(np.array([0.5, 0.5, 0.0, 0.5]))
    event = model.sleep_event(force_sleep=True)

    assert event.pruned_indices == (2,)
    assert event.prune_outcome is not None
    assert event.prune_outcome.proposed_neuron_ids == (2,)
    assert event.prune_outcome.scheduled_neuron_ids == (2,)
    assert event.prune_outcome.removed_neuron_ids == ()
    assert model.get_pending_prune_ids() == (2,)
    saved = model.snapshot_state()

    first = _wake(model)
    assert first.prune_outcome.removed_neuron_ids == ()
    assert model.get_pending_prune_ids() == (2,)
    second = _wake(model)
    assert second.prune_outcome.proposed_neuron_ids == ()
    assert second.prune_outcome.scheduled_neuron_ids == ()
    assert second.prune_outcome.removed_neuron_ids == (2,)
    assert model.get_pending_prune_ids() == ()
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 3)

    model.restore_state(saved)
    assert model.get_pending_prune_ids() == (2,)
    _wake(model)
    repeated = _wake(model)
    assert repeated.prune_outcome == second.prune_outcome
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 3)


def test_numpy_sleep_replay_reports_old_pending_removal_without_new_request() -> None:
    model = _numpy_model(
        sleep_enable_replay=True,
        replay_steps=1,
        replay_memory_size=1,
        prune_decay_steps=2,
        prune_decay_factor=0.5,
    )
    _wake(model)
    model.set_chemical_state(np.array([0.5, 0.5, 0.0, 0.5]))
    scheduled = model.sleep_event(force_sleep=True)
    assert scheduled.prune_outcome is not None
    assert scheduled.prune_outcome.scheduled_neuron_ids == (2,)
    assert scheduled.prune_outcome.removed_neuron_ids == ()
    assert model.get_pending_prune_ids() == (2,)

    finalized = model.sleep_event(force_sleep=True)

    assert finalized.pruned_indices == ()
    assert finalized.prune_outcome is not None
    assert finalized.prune_outcome.proposed_neuron_ids == ()
    assert finalized.prune_outcome.scheduled_neuron_ids == ()
    assert finalized.prune_outcome.removed_neuron_ids == (2,)
    assert model.get_pending_prune_ids() == ()


@pytest.mark.parametrize("decay_steps", [1, 2])
def test_numpy_external_proposal_reports_immediate_or_scheduled_id(decay_steps: int) -> None:
    model = _numpy_model(prune_decay_steps=decay_steps)

    outcome = model.apply_neuron_proposals([NeuronChangeProposal("hidden", remove_indices=(1,))])

    assert outcome.proposed_neuron_ids == (1,)
    assert outcome.scheduled_neuron_ids == ((1,) if decay_steps > 1 else ())
    assert outcome.removed_neuron_ids == ((1,) if decay_steps == 1 else ())
    assert model.get_pending_prune_ids() == ((1,) if decay_steps > 1 else ())


def test_torch_sleep_reports_proposed_and_removed_ids() -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head()
    head._chemical = torch.tensor([0.5, 0.5, 0.0, 0.5])

    event = head.sleep_event(force_sleep=True)

    assert event.pruned_indices == (2,)
    assert event.prune_outcome is not None
    assert event.prune_outcome.proposed_neuron_ids == (2,)
    assert event.prune_outcome.scheduled_neuron_ids == ()
    assert event.prune_outcome.removed_neuron_ids == (2,)
    assert head.get_pending_prune_ids() == ()


def test_torch_post_split_child_removal_reports_birth_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    head = _torch_head(
        max_split_per_sleep=1,
        prune_threshold=0.6,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
    )
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])
    monkeypatch.setattr(head, "_select_prune_indices", lambda **_: (4,))

    event = head.sleep_event(force_sleep=True)

    assert event.split_indices == (0,)
    assert event.pruned_indices == (4,)
    assert event.prune_outcome is not None
    assert event.prune_outcome.proposed_neuron_ids == (4,)
    assert event.prune_outcome.scheduled_neuron_ids == ()
    assert event.prune_outcome.removed_neuron_ids == (4,)
    assert event.lineage_after is not None
    assert 4 not in event.lineage_after.neuron_ids


def test_rejected_numpy_proposal_and_torch_sleep_do_not_emit_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _numpy_model()
    before = model.get_neuron_lineage()
    with pytest.raises(ValueError, match="remove index"):
        model.apply_neuron_proposals([NeuronChangeProposal("hidden", remove_indices=(99,))])
    assert model.get_neuron_lineage() == before
    assert model.get_pending_prune_ids() == ()

    torch = pytest.importorskip("torch")
    head = _torch_head()
    head._chemical = torch.tensor([0.5, 0.5, 0.0, 0.5])
    before_head = head.get_neuron_lineage()
    monkeypatch.setattr(head, "_select_prune_indices", lambda **_: (99,))
    with pytest.raises(ValueError, match="prune index"):
        head.sleep_event(force_sleep=True)
    assert head.get_neuron_lineage() == before_head
    assert head.get_pending_prune_ids() == ()


def test_skipped_sleep_has_no_prune_outcome_but_executed_noop_has_empty_outcome() -> None:
    disabled = _numpy_model(sleep_mode="disabled")
    skipped = disabled.sleep_event(force_sleep=True)
    assert not skipped.performed
    assert skipped.prune_outcome is None

    no_op = _numpy_model(max_prune_per_sleep=0)
    event = no_op.sleep_event(force_sleep=True)
    assert event.performed
    assert event.prune_outcome is not None
    assert event.prune_outcome.proposed_neuron_ids == ()
    assert event.prune_outcome.scheduled_neuron_ids == ()
    assert event.prune_outcome.removed_neuron_ids == ()


@pytest.mark.parametrize(
    "broken_mask",
    [np.array([True]), np.array([False, False, True, False], dtype=np.int32)],
)
def test_pending_view_rejects_misaligned_or_nonboolean_mask(broken_mask: np.ndarray) -> None:
    model = _numpy_model()
    model._prune_marked = broken_mask

    with pytest.raises(ValueError, match="pending prune mask"):
        model.get_pending_prune_ids()
    with pytest.raises(ValueError, match="pending prune mask"):
        _wake(model)


def test_numpy_snapshot_rejects_pending_mark_without_ttl() -> None:
    model = _numpy_model(prune_decay_steps=2)
    saved = model.snapshot_state()
    broken = replace(
        saved,
        state={
            **saved.state,
            "_prune_marked": np.array([False, False, True, False]),
        },
    )

    with pytest.raises(ValueError, match="pending prune"):
        model.restore_state(broken)
    assert model.get_pending_prune_ids() == ()
    assert model.get_neuron_lineage().neuron_ids == (0, 1, 2, 3)
