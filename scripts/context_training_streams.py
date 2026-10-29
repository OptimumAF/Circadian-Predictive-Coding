"""Explicit D9 contextual TRAIN recipes; native D7 recipes remain separate.

Inputs are detached ordinary TRAIN prototypes. Outputs are owned packets and
named synthetic-noise provenance. No learner, task outcomes or final/probe IO.
"""

from dataclasses import dataclass

import numpy as np

from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.app.coupled_attribution import AttributionBatch, _validated_batches
from src.core.experience import require_identifier
from src.core.ticket_workflow import ACTION_IDS
from src.core.temporal_decision_sequence import policy_for_phase

PAIRED_NEW_TASKS = (("v1", "parcel"), ("v2", "password"))
_PAIRED_CONTEXTS = {
    "context_paired_current9_return": ("current9-v1", "current9-v2") * 2 + ("old-v1", "old-v2"),
    "context_paired_joint9_return": ("joint-v1", "joint-v2") * 2 + ("old-v1", "old-v2"),
}


@dataclass(frozen=True)
class ContextTrainingRecipe:
    recipe_id: str
    contexts: tuple[str, ...]
    label_delay: int = 2
    noisy_old_invoice: bool = False

    def __post_init__(self):
        require_identifier(self.recipe_id, "contextual recipe ID")
        if len(self.recipe_id) > 64:
            raise ValueError("contextual recipe ID exceeds bounded namespace")
        if (
            type(self.contexts) is not tuple
            or not 1 <= len(self.contexts) <= 8
            or any(
                type(c) is not str
                or c
                not in (
                    "new-v1",
                    "old-v1",
                    "current9-v1",
                    "joint-v1",
                    "new-v2",
                    "old-v2",
                    "current9-v2",
                    "joint-v2",
                )
                for c in self.contexts
            )
        ):
            raise ValueError("contextual recipe requires1..8declared bounded TRAIN packets")
        if type(self.label_delay) is not int or self.label_delay not in (1, 2):
            raise ValueError("contextual label delay must be1or2")
        if (
            type(self.noisy_old_invoice) is not bool
            or self.noisy_old_invoice
            and "old-v1" not in self.contexts
            or self.noisy_old_invoice
            and any(c in ("current9-v1", "joint-v1") or c.endswith("-v2") for c in self.contexts)
        ):
            raise ValueError(
                "contextual noise requires separate old TRAIN packets without nine-row rehearsal"
            )
        if self.recipe_id in _PAIRED_CONTEXTS or any(c.endswith("-v2") for c in self.contexts):
            if (
                self.contexts != _PAIRED_CONTEXTS.get(self.recipe_id)
                or self.label_delay != 1
                or self.noisy_old_invoice
            ):
                raise ValueError(
                    "paired TRAIN requires its exact named contexts/order/delay without noise"
                )


CONTEXT_TRAINING_RECIPES = tuple(
    ContextTrainingRecipe(name, ("new-v1", "old-v1") * 3, 2, noisy)
    for name, noisy in (("context_recurring_clean", False), ("context_recurring_noisy", True))
)

# Complementary balanced blocks vary order and delay jointly, not task truth.
# Keep the original recurring protocol unchanged for its saved evidence.
WITHHELD_CONTEXT_RECIPES = (
    ContextTrainingRecipe(
        "context_withheld_blocks_a", ("old-v1", "new-v1", "new-v1", "old-v1", "old-v1", "new-v1"), 1
    ),
    ContextTrainingRecipe(
        "context_withheld_blocks_b", ("new-v1", "old-v1", "old-v1", "new-v1", "new-v1", "old-v1"), 2
    ),
)

# Why: equal nine-row exposure distinguishes old TRAIN reuse from simply
# spending more optimization on the current task. No score selects either arm.
CONTEXT_REHEARSAL_RECIPES = tuple(
    ContextTrainingRecipe(name, (context, context, "old-v1"), 1)
    for name, context in (
        ("context_current9_return", "current9-v1"),
        ("context_joint9_return", "joint-v1"),
    )
)

# Keep scientific comparison cohorts unchanged; this is the explicit CLI menu.
CONTEXT_PAIRED_REHEARSAL_RECIPES = tuple(
    ContextTrainingRecipe(name, contexts, 1) for name, contexts in _PAIRED_CONTEXTS.items()
)

CONTEXT_STREAM_RECIPES = (
    *CONTEXT_TRAINING_RECIPES,
    *CONTEXT_REHEARSAL_RECIPES,
    *CONTEXT_PAIRED_REHEARSAL_RECIPES,
)


def _rehearsal_prototype(context, prototypes):
    kind, _, version = context.rpartition("-")
    parts = (f"new-{version}",) * 3 if kind == "current9" else (f"new-{version}", f"old-{version}")
    names = tuple(
        f"exposure{index}:{name}" for index, key in enumerate(parts) for name in prototypes[key][1]
    )
    return AttributionBatch(
        "train",
        context,
        names,
        np.concatenate(tuple(prototypes[key][0].features for key in parts)),
        np.concatenate(tuple(prototypes[key][0].targets for key in parts)),
        num_classes=6,
    ), names


def _named_rows(batch, expected_concepts, *, version="v1", paired=False):
    """Validate semantic identities and clean truth without relying on row order."""
    names = []
    prototype = 0 if batch.context == "new-v1" else 1
    trace_names = tuple(
        f"{version}:{c}:{s}" for c in expected_concepts for s in ("sealed", "ready", "staged")
    )
    identities = {
        f"train:{name}" if paired else f"update-{prototype}-{i}": name
        for i, name in enumerate(trace_names)
    }
    for row_id, x, target in zip(batch.row_ids, batch.features, batch.targets[:, 0], strict=True):
        name = identities.get(row_id)
        if name is None:
            raise ValueError("contextual prototype requires original TRAIN trace row identities")
        _, concept, stage = name.split(":")
        words = {word for word, value in zip(CONTEXT_FEATURES, x, strict=True) if value == 1}
        concepts = words & {"invoice", "password", "legacy", "parcel"}
        stages = words & {"sealed", "ready", "staged"}
        if (
            not ((x == 0) | (x == 1)).all()
            or concepts != ({concept} if stage == "ready" else set())
            or stages != {stage}
            or f"policy_{version}" not in words
            or f"policy_{'v2' if version == 'v1' else 'v1'}" in words
        ):
            raise ValueError(
                "contextual TRAIN row requires exact policy/concept/workflow-state features"
            )
        # The public encoder omits concept outside ready. Original TRAIN trace
        # IDs bind those rows; noise selects the named ready row after validation.
        action = (
            "open"
            if stage == "sealed"
            else "commit"
            if stage == "staged"
            else policy_for_phase("after_A" if version == "v1" else "after_return")["rules"].get(
                concept
            )
        )
        if action is None or target != ACTION_IDS.index(action):
            raise ValueError("contextual prototype differs from original clean TRAIN truth")
        names.append(name)
    expected = {
        f"{version}:{c}:{s}" for c in expected_concepts for s in ("sealed", "ready", "staged")
    }
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("contextual TRAIN prototype requires each declared named row exactly once")
    return tuple(names)


def build_context_stream(recipe, new, old, *, second_head=None):
    if type(recipe) is not ContextTrainingRecipe:
        raise ValueError("contextual stream requires exact declared recipe")
    recipe.__post_init__()
    if second_head is not None:
        if (
            type(second_head) is not tuple
            or len(second_head) != 2
            or recipe not in CONTEXT_PAIRED_REHEARSAL_RECIPES
        ):
            raise ValueError("paired TRAIN requires its exact named recipe and second-head tuple")
        packets, _ = _validated_batches((new, old, *second_head), (), training_only=True)
        definitions = (
            ("new-v1", "v1", ("parcel",)),
            ("old-v1", "v1", ("invoice", "password")),
            ("new-v2", "v2", ("password",)),
            ("old-v2", "v2", ("invoice", "parcel")),
        )
        prototypes = {}
        for batch, (context, version, concepts) in zip(packets, definitions, strict=True):
            if (
                batch.context != context
                or batch.num_classes != 6
                or batch.episode is not None
                or batch.features.shape != (3 * len(concepts), 9)
            ):
                raise ValueError("paired TRAIN prototype geometry/context differs")
            prototypes[context] = (
                batch,
                _named_rows(batch, concepts, version=version, paired=True),
            )
        return _compose_stream(recipe, prototypes)
    if recipe in CONTEXT_PAIRED_REHEARSAL_RECIPES or any(
        c.endswith("-v2") for c in recipe.contexts
    ):
        raise ValueError("paired recipe requires explicit second-head ordinary TRAIN prototypes")
    (new, old), _ = _validated_batches((new, old), (), training_only=True)
    if (
        new.context != "new-v1"
        or old.context != "old-v1"
        or any(
            b.num_classes != 6 or b.features.shape != shape
            for b, shape in ((new, (3, 9)), (old, (6, 9)))
        )
    ):
        raise ValueError("contextual stream requires original3/6row D9 six-class TRAIN prototypes")
    prototypes = {
        "new-v1": (new, _named_rows(new, ("legacy",))),
        "old-v1": (old, _named_rows(old, ("invoice", "password"))),
    }
    return _compose_stream(recipe, prototypes)


def _compose_stream(recipe, prototypes):
    batches, noise = [], []
    for index, context in enumerate(recipe.contexts):
        base, names = (
            _rehearsal_prototype(context, prototypes)
            if context.startswith(("current9-", "joint-"))
            else prototypes[context]
        )
        targets = base.targets.copy()
        ids = tuple(f"{recipe.recipe_id}:batch{index}:{name}" for name in names)
        if recipe.noisy_old_invoice and context == "old-v1":
            position = names.index("v1:invoice:ready")
            targets[position, 0] = ACTION_IDS.index("defer")
            noise.append(
                dict(
                    role="train",
                    kind="synthetic_label_corruption",
                    batch=index,
                    semantic_row="v1:invoice:ready",
                    source_row_id=base.row_ids[position],
                    row_id=ids[position],
                    clean_action="route",
                    corrupted_action="defer",
                )
            )
        batches.append(
            AttributionBatch(
                "train",
                f"{recipe.recipe_id}:{context}",
                ids,
                base.features.copy(),
                targets,
                num_classes=6,
            )
        )
    result, _ = _validated_batches(tuple(batches), (), training_only=True)
    return result, noise
