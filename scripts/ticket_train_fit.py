"""Shared original TRAIN teacher and fit calculations for ticket scripts.

Inputs are an original feature spec/teacher factory or already-produced logits
and TRAIN cases. Outputs are detached TRAIN data or scalar fit summaries.
No learner construction, fitting, task rollout, file IO, final data or selection.
Why: keep native readouts and qualification/diagnosis entrypoints acyclic.
"""

from dataclasses import replace

import numpy as np
from scripts.run_ticket_workflow import CONTEXT_FEATURES, CONTEXT_TASKS, build_joint_context_batch
from src.adapters.perceptron_decision import build_decision_training_batch
from src.core.categorical_probability_margins import probability_margins
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.temporal_decision_sequence import VOCABULARY
from src.core.ticket_workflow import CATALOG

JOINT_TRAIN_TASKS = (*CONTEXT_TASKS, ("v1", "parcel"), ("v2", "password"))


def context_teacher(spec, build_cases):
    """Admit only aligned known TRAIN context templates before model construction."""
    if spec != DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG):
        raise ValueError("context qualification requires the exact D9 schema")
    teachers = {v: build_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")}
    return build_joint_context_batch(spec, teachers)


def joint_old_fresh_teacher(spec, build_cases):
    """Fixed six-task TRAIN cohort for diagnostics; no probe or final selection.

    Why: the original four-task teacher must stay unchanged for existing source
    recipes. Fresh tasks here are original TRAIN templates, not probe outcomes.
    """
    if spec != DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG):
        raise ValueError("joint old/fresh TRAIN requires the exact D9 schema")
    teachers = {v: build_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")}
    # Reuse original role/group/schema admission for ALL supplied teacher rows.
    build_joint_context_batch(spec, teachers)
    cases = tuple(
        teachers[version][3 * VOCABULARY.index(concept) + step]
        for version, concept in JOINT_TRAIN_TASKS
        for step in range(3)
    )
    batch = build_decision_training_batch(cases, spec, "joint-old-fresh-train", allow_subsets=True)
    for i, case in enumerate(cases):
        public = replace(case.request, query=case.request.query.removesuffix(" train") + " please")
        if not np.array_equal(batch.features[i : i + 1], encode_decision(public, spec)[0]):
            raise ValueError("joint old/fresh TRAIN/public alignment differs")
        for j in range(i):
            if (
                np.array_equal(batch.features[i], batch.features[j])
                and batch.targets[i, 0] != batch.targets[j, 0]
            ):
                raise ValueError("joint old/fresh TRAIN labels conflict on identical inputs")
    return cases, batch


def summarize_logits(logits, targets):
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    labels = targets[:, 0].astype(np.int64)
    ce = np.log(weights.sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    competitors = logits.copy()
    competitors[np.arange(len(labels)), labels] = -np.inf
    margins = logits[np.arange(len(labels)), labels] - competitors.max(axis=1)
    return probabilities, ce, margins, probability_margins(probabilities, targets, logits.shape[1])


def teacher_fit(logits, cases, batch, *, tasks=None):
    _, ce, margins, summary = summarize_logits(logits, batch.targets)
    ids = tuple(c.candidate_id for c in CATALOG)
    rows = []
    for i, case in enumerate(cases):
        allowed = sorted(c.candidate_id for c in case.request.allowed_candidates)
        winner = min(allowed, key=lambda key: (-float(logits[i, ids.index(key)]), key))
        target_value = float(logits[i, ids.index(case.target_id)])
        alternative = max(
            float(logits[i, ids.index(key)]) for key in allowed if key != case.target_id
        )
        rows.append(
            dict(
                concept=("invoice", "parcel")[i // 3] if tasks is None else tasks[i // 3][1],
                phase=("sealed", "ready", "staged")[i % 3],
                target_id=case.target_id,
                selected_id=winner,
                allowed_correct=winner == case.target_id,
                allowed_raw_margin=target_value - alternative,
                global_raw_margin=float(margins[i]),
            )
        )
        if tasks is not None:
            rows[-1]["policy_version"] = tasks[i // 3][0]
    return dict(
        global_mean_ce=float(np.mean(ce)),
        global_probability_margins=summary,
        allowed_correct_rows=sum(r["allowed_correct"] for r in rows),
        rows=rows,
    )
