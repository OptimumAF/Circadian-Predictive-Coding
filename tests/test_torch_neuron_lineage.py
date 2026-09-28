"""Torch adaptive-neuron identity survives staged sleep and restoration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.app.resnet50_benchmark import _hash_trained_model  # noqa: E402
from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
)


def _head(**changes: Any) -> CircadianPredictiveCodingHead:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=False,
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
        split_noise_scale=0.1,
    )
    options.update(changes)
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=479,
        config=CircadianHeadConfig(**options),
        min_hidden_dim=3,
        max_hidden_dim=7,
    )


def test_torch_lineage_tracks_repeated_sleep_changes_and_restore() -> None:
    head = _head()
    initial = head.get_neuron_lineage()
    assert initial.neuron_ids == (0, 1, 2, 3)
    assert initial.parent_ids == (None, None, None, None)
    assert initial.next_neuron_id == 4

    head._chemical = torch.tensor([1.0, 0.5, 0.5, 0.5])
    assert head.sleep_event(force_sleep=True).split_indices == (0,)
    assert head.get_neuron_lineage().parent_ids == (None, None, None, None, 0)

    head._chemical = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])
    assert head.sleep_event(force_sleep=True).split_indices == (4,)
    second = head.get_neuron_lineage()
    assert second.neuron_ids == (0, 1, 2, 3, 4, 5)
    assert second.parent_ids[-1] == 4

    head._chemical = torch.tensor([0.5, 0.0, 0.5, 0.5, 0.5, 0.5])
    assert head.sleep_event(force_sleep=True).pruned_indices == (1,)
    assert head.get_neuron_lineage().neuron_ids == (0, 2, 3, 4, 5)
    saved = head.snapshot_state()

    def continue_after_saved() -> tuple[Any, dict[str, Any]]:
        head._chemical = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5])
        assert head.sleep_event(force_sleep=True).pruned_indices == (3,)
        after = head.get_neuron_lineage()
        assert after.neuron_ids == (0, 2, 3, 5)
        assert after.parent_ids == (None, None, None, 4)
        assert after.next_neuron_id == 6
        head._chemical = torch.tensor([1.0, 0.5, 0.5, 0.5])
        assert head.sleep_event(force_sleep=True).split_indices == (0,)
        assert head.get_neuron_lineage().neuron_ids == (0, 2, 3, 5, 6)
        assert head.get_neuron_lineage().parent_ids[-1] == 0
        return head.get_neuron_lineage(), head.snapshot_state()

    continued, final_state = continue_after_saved()
    head.restore_state(saved)
    assert head.get_neuron_lineage().neuron_ids == (0, 2, 3, 4, 5)
    replayed, replayed_state = continue_after_saved()
    assert replayed == continued
    assert final_state.keys() == replayed_state.keys()
    for name, value in final_state.items():
        if torch.is_tensor(value):
            assert torch.equal(value, replayed_state[name]), name
        else:
            assert value == replayed_state[name], name


@pytest.mark.parametrize(
    ("removed_index", "expected_ids", "expected_parents"),
    [
        (0, (1, 2, 3, 4), (None, None, None, 0)),
        (4, (0, 1, 2, 3), (None, None, None, None)),
    ],
)
def test_torch_post_split_prune_keeps_lineage(
    monkeypatch: pytest.MonkeyPatch,
    removed_index: int,
    expected_ids: tuple[int, ...],
    expected_parents: tuple[int | None, ...],
) -> None:
    head = _head(prune_threshold=0.6)
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])
    monkeypatch.setattr(head, "_select_prune_indices", lambda **_: (removed_index,))

    event = head.sleep_event(force_sleep=True)

    assert event.split_indices == (0,)
    assert event.pruned_indices == (removed_index,)
    lineage = head.get_neuron_lineage()
    assert lineage.neuron_ids == expected_ids
    assert lineage.parent_ids == expected_parents
    assert lineage.next_neuron_id == 5


def test_rejected_torch_staged_proposal_does_not_allocate_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _head(prune_threshold=0.6)
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])
    before = head.snapshot_state()
    monkeypatch.setattr(head, "_select_prune_indices", lambda **_: (99,))

    with pytest.raises(ValueError, match="prune index"):
        head.sleep_event(force_sleep=True)

    assert head.get_neuron_lineage().next_neuron_id == 4
    assert torch.equal(head._split_generator.get_state(), before["split_generator_state"])
    assert torch.equal(head.snapshot_state()["neuron_ids"], before["neuron_ids"])


@pytest.mark.parametrize("corruption", ["duplicate", "parent", "next", "old_version"])
def test_torch_snapshot_rejects_invalid_lineage_without_mutation(corruption: str) -> None:
    head = _head()
    saved = head.snapshot_state()
    broken = dict(saved)
    if corruption == "duplicate":
        broken["neuron_ids"] = torch.tensor([0, 0, 2, 3], dtype=torch.int64)
    elif corruption == "parent":
        broken["parent_ids"] = torch.tensor([-1, -1, -1, 4], dtype=torch.int64)
    elif corruption == "next":
        broken["next_neuron_id"] = 3
    else:
        broken["format_version"] = 1

    with pytest.raises(ValueError, match="lineage|format_version"):
        head.restore_state(broken)
    assert head.get_neuron_lineage().neuron_ids == (0, 1, 2, 3)


def test_trained_state_hash_includes_torch_lineage() -> None:
    head = _head()
    model = SimpleNamespace(backbone=torch.nn.Identity(), head=head)
    before = _hash_trained_model(model)
    head._next_neuron_id += 1
    assert _hash_trained_model(model) != before
