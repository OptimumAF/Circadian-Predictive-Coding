"""Explicit D9 contextual TRAIN recipes; native D7 recipes remain separate.

Inputs are detached ordinary TRAIN prototypes. Outputs are owned packets and
named synthetic-noise provenance. No learner, task outcomes or final/probe IO.
"""

from dataclasses import dataclass

from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.app.coupled_attribution import AttributionBatch, _validated_batches
from src.core.experience import require_identifier
from src.core.ticket_workflow import ACTION_IDS


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
            or any(type(c) is not str or c not in ("new-v1", "old-v1") for c in self.contexts)
        ):
            raise ValueError("contextual recipe requires1..8declared new/old TRAIN packets")
        if type(self.label_delay) is not int or self.label_delay not in (1, 2):
            raise ValueError("contextual label delay must be1or2")
        if (
            type(self.noisy_old_invoice) is not bool
            or self.noisy_old_invoice
            and "old-v1" not in self.contexts
        ):
            raise ValueError("contextual noise requires declared old TRAIN packets")


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


def _named_rows(batch, expected_concepts):
    """Validate semantic identities and clean truth without relying on row order."""
    names = []
    prototype = 0 if batch.context == "new-v1" else 1
    trace_names = tuple(
        f"v1:{c}:{s}" for c in expected_concepts for s in ("sealed", "ready", "staged")
    )
    identities = {f"update-{prototype}-{i}": name for i, name in enumerate(trace_names)}
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
            or "policy_v1" not in words
            or "policy_v2" in words
        ):
            raise ValueError(
                "contextual TRAIN row requires exact v1/concept/workflow-state features"
            )
        # The public encoder omits concept outside ready. Original TRAIN trace
        # IDs bind those rows; noise selects the named ready row after validation.
        action = (
            "open"
            if stage == "sealed"
            else "commit"
            if stage == "staged"
            else {
                "invoice": "route",
                "password": "inspect",
                "legacy": "retire",
            }.get(concept)
        )
        if action is None or target != ACTION_IDS.index(action):
            raise ValueError("contextual prototype differs from original clean TRAIN truth")
        names.append(name)
    expected = {f"v1:{c}:{s}" for c in expected_concepts for s in ("sealed", "ready", "staged")}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("contextual TRAIN prototype requires each declared named row exactly once")
    return tuple(names)


def build_context_stream(recipe, new, old):
    if type(recipe) is not ContextTrainingRecipe:
        raise ValueError("contextual stream requires exact declared recipe")
    recipe.__post_init__()
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
    batches, noise = [], []
    for index, context in enumerate(recipe.contexts):
        base, names = prototypes[context]
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
