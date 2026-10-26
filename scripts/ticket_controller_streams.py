"""Bounded declared TRAIN receipt shifts built from existing teacher packets.

No learner, policy choice or final source access. Label corruption is explicit
experimental noise, never a claim of a correct teacher/episode target.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.app.coupled_attribution import AttributionBatch, _validated_batches
from src.core.experience import require_identifier
from src.core.ticket_workflow import ACTION_IDS

# Domain A returns under the current v2 rules; obsolete v1 labels never return.
POLICY_CHANGE_PHASES = (
    ("A_v1", "v1", ("invoice", "password")),
    ("B_v2", "v2", ("parcel", "password")),
    ("A_return_v2", "v2", ("invoice", "password")),
)
POLICY_CHANGE_CHECKPOINTS = ((0, "v1"), (4, "v1"), (8, "v2"), (12, "v2"))

# True context recurrence restores both the public context and the task truth.
CONTEXT_RETURN_PHASES = (
    ("A_v1", "v1", ("invoice", "password")),
    ("B_v2", "v2", ("invoice", "parcel")),
    ("A_return_v1", "v1", ("invoice", "password")),
)


def build_context_return_stream(spec):
    """Build twelve owned D9 TRAIN packets; no task outcomes select the data."""
    from scripts.run_ticket_workflow import (
        CONTEXT_FEATURES,
        CONTEXT_TASKS,
        build_joint_context_batch,
        training_cases,
    )
    from src.adapters.perceptron_decision import build_decision_training_batch
    from src.core.decision_features import DecisionFeatureSpec
    from src.core.ticket_workflow import CATALOG

    if type(spec) is not DecisionFeatureSpec or spec != DecisionFeatureSpec(
        CONTEXT_FEATURES, CATALOG
    ):
        raise ValueError("context return requires the fixed D9 feature/catalog schema")
    teachers = {
        v: training_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")
    }
    # Reuse the accepted role, public-feature alignment and conflicting-label checks.
    selected, _ = build_joint_context_batch(spec, teachers)
    batches = []
    for phase, version, concepts in CONTEXT_RETURN_PHASES:
        rows = tuple(
            selected[3 * CONTEXT_TASKS.index((version, concept)) + step]
            for concept in concepts
            for step in range(3)
        )
        if len(rows) != 6:
            raise ValueError("context return requires exactly two three-step TRAIN traces")
        for repeat in range(4):
            owned = tuple(
                replace(c, request=replace(c.request, request_id=f"return-{phase}-{repeat}-{i}"))
                for i, c in enumerate(rows)
            )
            batches.append(build_decision_training_batch(owned, spec, phase, allow_subsets=True))
    result, _ = _validated_batches(tuple(batches), (), training_only=True)
    return result


def build_policy_change_stream(spec):
    """Fixed owned TRAIN packets for the existing chronological runner.

    Inputs are the original feature schema. Outputs are12six-row TRAIN batches;
    no learner, outcome/guard/final/probe read, schedule choice or IO occurs here.
    """
    from scripts.run_ticket_workflow import FEATURES, training_cases
    from src.adapters.perceptron_decision import build_decision_training_batch
    from src.core.decision_features import DecisionFeatureSpec
    from src.core.temporal_decision_sequence import VOCABULARY
    from src.core.ticket_workflow import CATALOG

    if type(spec) is not DecisionFeatureSpec or spec != DecisionFeatureSpec(FEATURES, CATALOG):
        raise ValueError("policy stream requires the original fixed feature/catalog schema")
    teachers = {version: training_cases(policy_version=version) for version in ("v1", "v2")}
    batches = []
    for phase, version, concepts in POLICY_CHANGE_PHASES:
        teacher = teachers[version]
        if len(teacher) != 12 or any(c.role != "train" for c in teacher):
            raise ValueError("policy stream requires complete ordinary TRAIN teachers")
        selected = tuple(
            teacher[3 * VOCABULARY.index(concept) + step]
            for concept in concepts
            for step in range(3)
        )
        for repeat in range(4):
            rows = tuple(
                replace(
                    c,
                    request=replace(c.request, request_id=f"policy-{phase}-{repeat}-{index}"),
                    source_group=f"train-policy-{phase}",
                )
                for index, c in enumerate(selected)
            )
            batches.append(build_decision_training_batch(rows, spec, phase, allow_subsets=True))
    result, _ = _validated_batches(tuple(batches), (), training_only=True)
    return result


@dataclass(frozen=True)
class TicketStreamRecipe:
    recipe_id: str
    contexts: tuple[str, ...]
    label_delay: int
    noisy_invoice: bool = False

    def __post_init__(self):
        require_identifier(self.recipe_id, "recipe_id")
        if len(self.recipe_id) > 64:
            raise ValueError("recipe ID exceeds bounded namespace")
        if (
            type(self.contexts) is not tuple
            or not 1 <= len(self.contexts) <= 8
            or any(
                type(c) is not str or c not in ("new_contexts", "old_return") for c in self.contexts
            )
        ):
            raise ValueError("recipe requires1..8declared new/old TRAIN contexts")
        if type(self.label_delay) is not int or self.label_delay not in (1, 2):
            raise ValueError("recipe label delay must be1or2")
        if type(self.noisy_invoice) is not bool or (
            self.noisy_invoice and any(c != "old_return" for c in self.contexts)
        ):
            raise ValueError("declared invoice noise requires old-return TRAIN packets")


WITHHELD_RECIPES = (
    TicketStreamRecipe(
        "diluted_return",
        (
            "old_return",
            "new_contexts",
            "old_return",
            "old_return",
            "new_contexts",
            "new_contexts",
            "old_return",
            "new_contexts",
        ),
        1,
    ),
    TicketStreamRecipe("noisy_return", ("old_return",) * 8, 2, True),
)


def build_stream(recipe: TicketStreamRecipe, new: AttributionBatch, old: AttributionBatch):
    if type(recipe) is not TicketStreamRecipe:
        raise ValueError("stream requires exact declared recipe")
    recipe.__post_init__()
    (new, old), _ = _validated_batches((new, old), (), training_only=True)
    if any(b.num_classes != 6 or b.features.shape != (6, 7) for b in (new, old)):
        raise ValueError("recipe requires original six-row/seven-feature/six-class teacher packets")
    if recipe.noisy_invoice and old.targets[1, 0] != ACTION_IDS.index("inspect"):
        raise ValueError("declared invoice noise requires original inspect teacher target")
    batches = []
    for index, kind in enumerate(recipe.contexts):
        selected = new if kind == "new_contexts" else old
        features = np.vstack((selected.features, old.features))
        targets = np.vstack((selected.targets, old.targets))
        if recipe.noisy_invoice:
            targets[1, 0] = ACTION_IDS.index("defer")
        batches.append(
            AttributionBatch(
                "train",
                f"{recipe.recipe_id}:{kind}",
                tuple(f"{recipe.recipe_id}:batch{index}:row{row}" for row in range(12)),
                features,
                targets,
                num_classes=6,
            )
        )
    result, _ = _validated_batches(tuple(batches), (), training_only=True)
    return result


@dataclass(frozen=True)
class LiveTrainingRecipe:
    """Compose explicit live noise without changing frozen comparative recipes."""

    recipe: TicketStreamRecipe
    retained_invoice_noise: bool = False

    def __post_init__(self):
        if type(self.recipe) is not TicketStreamRecipe:
            raise ValueError("live TRAIN recipe requires an exact original packet recipe")
        self.recipe.__post_init__()
        if type(self.retained_invoice_noise) is not bool:
            raise ValueError("retained invoice noise must be an exact boolean")
        if self.retained_invoice_noise and self.recipe.noisy_invoice:
            raise ValueError("live TRAIN recipe cannot combine two label corruptions")


LIVE_TRAINING_RECIPES = (
    LiveTrainingRecipe(WITHHELD_RECIPES[1]),
    LiveTrainingRecipe(
        TicketStreamRecipe("live_new_retained_noise", ("new_contexts",) * 8, 2), True
    ),
)

# Same identifiers/features/order/delay in both arms; only marked row7 differs.
FOLLOWUP_TRAINING_RECIPES = tuple(
    (
        name,
        LiveTrainingRecipe(TicketStreamRecipe("followup_mixed", ("new_contexts",) * 8, 2), noisy),
    )
    for name, noisy in (("clean", False), ("noisy", True))
)


def require_followup_recipe(recipe):
    if type(recipe) is not LiveTrainingRecipe:
        raise ValueError("followup recipe requires an exact declared live TRAIN recipe")
    recipe.__post_init__()
    if recipe.recipe != TicketStreamRecipe("followup_mixed", ("new_contexts",) * 8, 2):
        raise ValueError("followup recipe differs from the fixed second-window contract")


def build_followup_stream(recipe, original):
    """Preserve original first-window data and replace only the declared second."""
    require_followup_recipe(recipe)
    original, _ = _validated_batches(original, (), training_only=True)
    if len(original) != 16:
        raise ValueError("followup recipe requires the original sixteen TRAIN packets")
    second = build_live_stream(recipe, original[0], original[8])
    combined, _ = _validated_batches((*original[:8], *second), (), training_only=True)
    return combined


def build_live_stream(recipe: LiveTrainingRecipe, new: AttributionBatch, old: AttributionBatch):
    if type(recipe) is not LiveTrainingRecipe:
        raise ValueError("live stream requires an exact live TRAIN recipe")
    recipe.__post_init__()
    batches = build_stream(recipe.recipe, new, old)
    if recipe.retained_invoice_noise:
        # Why: row1 belongs to the selected new half. Row7 is always the retained
        # old invoice, so the original new teacher and frozen row1 noise stay intact.
        if any(batch.targets[7, 0] != ACTION_IDS.index("inspect") for batch in batches):
            raise ValueError("live noise requires original retained inspect teacher target")
        for batch in batches:
            batch.targets[7, 0] = ACTION_IDS.index("defer")
    return batches
