"""Declared TRAIN shifts preserve roles, clean source truth and ownership."""

from dataclasses import replace

import numpy as np
import pytest

from scripts import run_bp_ticket_serving as demo
from scripts.ticket_controller_streams import TicketStreamRecipe, WITHHELD_RECIPES, build_stream


@pytest.fixture(scope="module")
def prototypes():
    _, _, batches = demo.context._teacher_batches(
        demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
    )
    return batches[0], batches[demo.context.BATCHES_PER_PHASE]


def test_diluted_return_has_declared_recurrence_and_owned_twelve_row_packets(prototypes):
    new, old = prototypes
    new_before, old_before = new.targets.copy(), old.targets.copy()
    recipe = WITHHELD_RECIPES[0]
    batches = build_stream(recipe, new, old)
    assert len(batches) == 8 and recipe.label_delay == 1
    assert recipe.contexts == (
        "old_return",
        "new_contexts",
        "old_return",
        "old_return",
        "new_contexts",
        "new_contexts",
        "old_return",
        "new_contexts",
    )
    ids = [key for b in batches for key in b.row_ids]
    assert len(ids) == len(set(ids)) == 96
    for kind, batch in zip(recipe.contexts, batches, strict=True):
        selected = new if kind == "new_contexts" else old
        assert batch.role == "train" and batch.num_classes == 6 and batch.episode is None
        assert np.array_equal(batch.features[:6], selected.features)
        assert np.array_equal(batch.features[6:], old.features)
        assert np.array_equal(batch.targets[:6], selected.targets)
        assert np.array_equal(batch.targets[6:], old.targets)
        assert not np.shares_memory(batch.features, selected.features)
        assert not np.shares_memory(batch.targets, old.targets)
    batches[0].targets[0, 0] = 99
    assert np.array_equal(new.targets, new_before) and np.array_equal(old.targets, old_before)
    assert batches[1].targets[6, 0] != 99


def test_noisy_return_marks_exact_ready_invoice_corruption_without_changing_clean_truth(prototypes):
    new, old = prototypes
    recipe = WITHHELD_RECIPES[1]
    batches = build_stream(recipe, new, old)
    clean = np.vstack((old.targets, old.targets))
    assert recipe.label_delay == 2 and recipe.noisy_invoice
    for batch in batches:
        changed = np.flatnonzero(batch.targets[:, 0] != clean[:, 0])
        assert changed.tolist() == [1]
        assert batch.targets[1, 0] == demo.ACTION_IDS.index("defer")
        assert clean[1, 0] == old.targets[1, 0] == demo.ACTION_IDS.index("inspect")
        assert batch.role == "train" and batch.targets.shape == (12, 1)


@pytest.mark.parametrize(
    "changes",
    [
        dict(contexts=[]),
        dict(contexts=("old_return",) * 9),
        dict(contexts=("final_test",)),
        dict(label_delay=0),
        dict(label_delay=3),
        dict(label_delay=True),
        dict(noisy_invoice=1),
        dict(noisy_invoice=True, contexts=("new_contexts",)),
        dict(recipe_id="x" * 65),
    ],
)
def test_invalid_recipe_is_refused_before_packet_or_native_access(changes):
    with pytest.raises(ValueError):
        replace(WITHHELD_RECIPES[0], **changes)


@pytest.mark.parametrize("role", ["probe_train", "inner_guard", "outer_selection", "final_test"])
def test_non_train_source_is_never_relabelled_as_training(role, prototypes):
    new, old = prototypes
    with pytest.raises(ValueError, match="role must be train"):
        build_stream(WITHHELD_RECIPES[0], replace(new, role=role), old)


def test_incompatible_or_replaced_teacher_source_is_refused(prototypes):
    new, old = prototypes
    with pytest.raises(ValueError, match="original"):
        build_stream(
            WITHHELD_RECIPES[0],
            replace(new, features=new.features[:, :6]),
            replace(old, features=old.features[:, :6]),
        )
    bad = old.targets.copy()
    bad[1, 0] = demo.ACTION_IDS.index("route")
    with pytest.raises(ValueError, match="original inspect"):
        build_stream(WITHHELD_RECIPES[1], new, replace(old, targets=bad))
    recipe = TicketStreamRecipe("replaced", ("old_return",), 1)
    object.__setattr__(recipe, "label_delay", False)
    with pytest.raises(ValueError, match="delay"):
        build_stream(recipe, new, old)
