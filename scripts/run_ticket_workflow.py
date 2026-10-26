"""Learn all local ticket steps, then execute fixed bounded sandbox episodes.

Uses existing perceptron/AttributionStream and typed decision ports. No native
model, RNG, final/probe labels, tuning, promotion or external tool action.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.perceptron_decision import (
    PerceptronDecisionPredictor,
    build_decision_training_batch,
)
from src.adapters.perceptron_learner import PerceptronLearner
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.decision_reference_evaluation import DecisionCase
from src.app.ticket_workflow import build_public_workflow_request, run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_perceptron import PerceptronPolicy
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.temporal_decision_sequence import VOCABULARY, policy_for_phase
from src.core.ticket_workflow import (
    ACTION_IDS,
    CATALOG,
    POLICY_CONTEXT_TOKENS,
    WorkflowCapabilities,
    WorkflowState,
    dispatch_workflow,
    workflow_request,
    workflow_success,
)
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.shared.process_memory import ProcessRssSampler

PASSES = 8
FEATURES = tuple(sorted((*VOCABULARY, "ready", "sealed", "staged")))
CONTEXT_FEATURES = tuple(sorted((*FEATURES, *POLICY_CONTEXT_TOKENS)))
CONTEXT_TASKS = (("v1", "invoice"), ("v1", "password"), ("v2", "invoice"), ("v2", "parcel"))


def training_cases(
    *, policy_version="v2", include_policy_context=False, public_observation_train=False
):
    """Four complete TRAIN traces, never actions from evaluation episodes."""
    if type(include_policy_context) is not bool:
        raise ValueError("include_policy_context must be an exact bool")
    if type(public_observation_train) is not bool:
        raise ValueError("public_observation_train must be an exact bool")
    context_options = {"include_policy_context": True} if include_policy_context else {}
    cases = []
    capabilities = WorkflowCapabilities(ACTION_IDS)
    # Validate through the actual sandbox before generating any teacher payload.
    WorkflowState("teacher-policy-check", "invoice", policy_version=policy_version)
    policy = policy_for_phase("after_A" if policy_version == "v1" else "after_return")
    prefix = "train" if policy_version == "v2" else "train-v1"
    group = "train-workflow-traces" if policy_version == "v2" else "train-workflow-traces-v1"
    if include_policy_context:
        group += "-context"
    if public_observation_train:
        group += "-public-observation-v1"
    request_builder = (
        build_public_workflow_request if public_observation_train else workflow_request
    )
    for index, concept in enumerate(VOCABULARY):
        state = WorkflowState(f"{prefix}-ticket-{index}", concept, policy_version=policy_version)
        last = "initial"
        target = policy["rules"][concept]
        for step, action in enumerate(("open", target, "commit")):
            request = request_builder(
                state, capabilities, f"{prefix}-{index}-{step}", last, **context_options
            )
            if not public_observation_train:
                request = replace(request, query=request.query + " train")
            cases.append(DecisionCase(request, action, group, "train"))
            transition = dispatch_workflow(state, action, state.revision, capabilities)
            if transition.status != "applied":
                raise ValueError("TRAIN teacher target was not applied by the sandbox")
            state, last = transition.state, transition.status
        if not workflow_success(state):
            raise ValueError("TRAIN teacher trace did not solve the declared policy")
    return tuple(cases)


def build_joint_context_batch(spec, teachers):
    """Prepare the fixed joint TRAIN packet; no learner, fitting or task selection."""
    if spec != DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG):
        raise ValueError("context qualification requires the exact D9 schema")
    if type(teachers) is not dict or set(teachers) != {"v1", "v2"}:
        raise ValueError("joint context requires both original TRAIN teachers")
    for version, cases in teachers.items():
        group = "train-workflow-traces" + ("-v1" if version == "v1" else "") + "-context"
        if (
            type(cases) is not tuple
            or len(cases) != 12
            or any(
                type(c) is not DecisionCase or c.role != "train" or c.source_group != group
                for c in cases
            )
        ):
            raise ValueError("context qualification requires original tagged TRAIN teacher")
    selected = tuple(
        teachers[v][3 * VOCABULARY.index(c) + step] for v, c in CONTEXT_TASKS for step in range(3)
    )
    batch = build_decision_training_batch(selected, spec, "context-joint-train", allow_subsets=True)
    for i, case in enumerate(selected):
        public = replace(case.request, query=case.request.query.removesuffix(" train") + " please")
        if not np.array_equal(batch.features[i : i + 1], encode_decision(public, spec)[0]):
            raise ValueError("context TRAIN/public alignment differs")
        for j in range(i):
            if (
                np.array_equal(batch.features[i], batch.features[j])
                and batch.targets[i, 0] != batch.targets[j, 0]
            ):
                raise ValueError("context TRAIN labels conflict on identical inputs")
    return selected, batch


def _context_mode(include_policy_context, joint_policy_context):
    for name, value in (
        ("include_policy_context", include_policy_context),
        ("joint_policy_context", joint_policy_context),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be an exact bool")
    if include_policy_context and joint_policy_context:
        raise ValueError("select either single-policy or joint-policy context")
    return include_policy_context or joint_policy_context


def fit_reference(report, budget, *, include_policy_context=False, joint_policy_context=False):
    """Fit one fixed trajectory through the existing delayed TRAIN runner."""
    contextual = _context_mode(include_policy_context, joint_policy_context)
    spec = DecisionFeatureSpec(CONTEXT_FEATURES if contextual else FEATURES, CATALOG)
    if joint_policy_context:
        teachers = {
            v: training_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")
        }
        cases, packet = build_joint_context_batch(spec, teachers)
        batches = tuple(
            replace(
                packet,
                context=f"workflow-pass{p}",
                row_ids=tuple(f"pass{p}-{c.request.request_id}" for c in cases),
            )
            for p in range(PASSES)
        )
    else:
        cases = training_cases(include_policy_context=True) if contextual else training_cases()
        batches = tuple(
            build_decision_training_batch(
                tuple(
                    replace(
                        c, request=replace(c.request, request_id=f"pass{p}-{c.request.request_id}")
                    )
                    for c in cases
                ),
                spec,
                f"workflow-pass{p}",
                allow_subsets=True,
            )
            for p in range(PASSES)
        )
    report["training_cases"] = [asdict(c) for c in cases]
    report["feature_spec"] = asdict(spec)
    report["training_case_generation"] = dict(
        demonstration_transitions=12,
        candidate_constructions=12,
        catalog_entries_considered=72,
        candidates_returned=60,
    )
    if joint_policy_context:
        report["training_case_generation"] = dict(
            demonstration_transitions=24,
            candidate_constructions=24,
            catalog_entries_considered=144,
            candidates_returned=120,
        )
        report["joint_context_tasks"] = [
            dict(policy_version=v, concept=c) for v, c in CONTEXT_TASKS
        ]
    data_pin = (
        deepcopy((report["training_cases"], report["feature_spec"], batches))
        if joint_policy_context
        else None
    )
    states = {}

    def capture(update, state, batch):
        if joint_policy_context and not _same_value(
            (report["training_cases"], report["feature_spec"], batches), data_pin
        ):
            raise ValueError("joint TRAIN data binding changed")
        states[update] = state

    learner = PerceptronLearner(PerceptronPolicy(len(spec.vocabulary), len(CATALOG)))
    stream = AttributionStream(
        learner,
        batches,
        (),
        checkpoint_updates=(0, PASSES),
        label_delay=1,
        budget=budget,
        observe_checkpoint=capture,
        training_only=True,
    )
    try:
        stream.run_until(PASSES)
        report["stream"] = stream.finish()
        if joint_policy_context:
            unchanged = (
                learner.snapshot_state() == states[0]
                and stream._runtime.actor.snapshot_state().state == states[0]
            )
            report["source_seed_actor_unchanged"] = unchanged
            if not unchanged:
                raise ValueError("joint fit changed original seed or deployed actor")
        return spec, states
    finally:
        report["training_work"] = dict(
            committed_updates=budget.updates_completed,
            committed_rows=12 * budget.updates_completed,
            latest_observed_mistakes=None
            if PASSES not in states
            else states[PASSES].mistake_updates,
            unobserved_attempted_work_unknown="stream" not in report,
        )
        report["snapshots"] = [dict(update=n, state=asdict(s)) for n, s in states.items()]


def _read_joint_teacher(report, budget, predict, control, source):
    rows: list[dict[str, Any]] = []
    report["teacher_fit"].append(dict(control=control, source_updates=source, rows=rows))
    for i, case in enumerate(report["training_cases"]):
        budget.before_final()
        r = case["request"]
        request = DecisionRequest(
            r["request_id"],
            r["query"],
            tuple(DecisionCandidate(**c) for c in r["allowed_candidates"]),
        )
        report["teacher_read_work"]["attempted"] += 1
        result = predict(request)
        report["teacher_read_work"]["returned"] += 1
        raw = {s.candidate_id: s.raw_score for s in result.scores}
        target = case["target_id"]
        rows.append(
            dict(
                policy_version=CONTEXT_TASKS[i // 3][0],
                concept=CONTEXT_TASKS[i // 3][1],
                phase=("sealed", "ready", "staged")[i % 3],
                target_id=target,
                selected_id=result.selected_id,
                correct=result.selected_id == target,
                raw_margin=raw[target] - max(v for k, v in raw.items() if k != target),
            )
        )


def _read_joint_reference(report, budget, spec, states, rng_before):
    """Read two fixed states; reject changed source/owners/paid prefixes."""
    source_pin = deepcopy((spec, states, report["snapshots"], report["training_cases"]))
    paid_owners: list[dict[str, Any]] = []
    paid_values: list[dict[str, Any]] = []
    episode_owner = report["episodes"]
    limits = deepcopy(budget.budget)
    binding = (
        id(budget.budget),
        id(budget.clock),
        budget.started_at,
        id(budget.process_rss_sampler),
        id(budget.progress),
    )
    for control, source in (("unchanged", 0), ("learned", PASSES)):
        predict = PerceptronDecisionPredictor(spec, states[source])
        readout_pin = deepcopy(predict)

        def guard():
            if not _same_value(np.random.get_state(), rng_before):
                raise ValueError("joint readout changed caller RNG")
            if not _same_value(
                (spec, states, report["snapshots"], report["training_cases"]), source_pin
            ) or not _same_value(predict, readout_pin):
                raise ValueError("joint readout changed original source or frozen predictor")
            observed_binding = (
                id(budget.budget),
                id(budget.clock),
                budget.started_at,
                id(budget.process_rss_sampler),
                id(budget.progress),
            )
            if (
                observed_binding != binding
                or budget.budget != limits
                or budget.updates_completed != PASSES
            ):
                raise ValueError("joint readout changed original budget")
            if (
                report["episodes"] is not episode_owner
                or len(report["episodes"]) != len(paid_owners)
                or any(a is not b for a, b in zip(report["episodes"], paid_owners))
                or report["episodes"] != paid_values
            ):
                report["episodes"] = episode_owner
                episode_owner[:] = paid_owners
                for row, saved in zip(paid_owners, paid_values):
                    row.clear()
                    row.update(deepcopy(saved))
                raise ValueError("joint readout preceding paid evidence changed")

        def guarded_predict(request):
            guard()
            try:
                result = predict(request)
            finally:
                guard()
            return result

        guard()
        _read_joint_teacher(report, budget, guarded_predict, control, source)
        for version, concept in CONTEXT_TASKS:
            for scenario in ("ordinary", "stale_once", "revoke_commit"):
                budget.before_final()
                guard()
                episode = run_workflow_episode(
                    WorkflowState("joint-development", concept, policy_version=version),
                    guarded_predict,
                    capabilities=WorkflowCapabilities(ACTION_IDS),
                    max_steps=6,
                    scenario=scenario,
                    include_policy_context=True,
                )
                row = dict(
                    control=control,
                    source_updates=source,
                    policy_version=version,
                    concept=concept,
                    role="outer_selection",
                    source_group="development-joint-context-workflow",
                    episode=episode,
                )
                report["episodes"].append(row)
                paid_owners.append(row)
                paid_values.append(deepcopy(row))
                budget.before_final()
                guard()
                if not episode["complete"]:
                    raise RuntimeError(
                        "joint workflow episode has an unvalidated or unreturned paid prefix"
                    )


def _summary(episodes):
    result = {}
    for control in ("unchanged", "learned"):
        rows = [row["episode"] for row in episodes if row["control"] == control]
        result[control] = dict(
            episodes=len(rows),
            successes=sum(r["success"] for r in rows),
            recoveries=sum(r["error_recovery_demonstrated"] for r in rows),
            by_scenario={
                scenario: dict(
                    episodes=sum(r["scenario"] == scenario for r in rows),
                    successes=sum(r["success"] for r in rows if r["scenario"] == scenario),
                    permission_denials=sum(
                        r["stopped_reason"] == "permission_denied"
                        for r in rows
                        if r["scenario"] == scenario
                    ),
                )
                for scenario in ("ordinary", "stale_once", "revoke_commit")
            },
        )
    result["learned_minus_unchanged_successes"] = (
        result["learned"]["successes"] - result["unchanged"]["successes"]
    )
    return result


def run_example(*, include_policy_context=False, joint_policy_context=False) -> dict[str, Any]:
    contextual = _context_mode(include_policy_context, joint_policy_context)
    augmented_width = 1 + len(CONTEXT_FEATURES if contextual else FEATURES)
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="local_learned_ticket_workflow_v1",
        complete=False,
        episodes=[],
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        external_actions=0,
        qualification="fixed synthetic overlapping-concept task; all six-action TRAIN before development; no modern-model/CPC/generalization/OS-containment claim",
    )
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=PASSES, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
        ),
        monotonic,
    )
    sampler = ProcessRssSampler()
    context_options = {"include_policy_context": True} if contextual else {}
    fit_options = {"joint_policy_context": True} if joint_policy_context else context_options
    rng_before = deepcopy(np.random.get_state()) if joint_policy_context else None
    if contextual:
        report["definition"] = "local_learned_ticket_workflow_context_v1"
        report["policy_context_in_query"] = True
    if joint_policy_context:
        report.update(
            definition="local_joint_context_perceptron_v1",
            teacher_fit=[],
            teacher_read_work=dict(attempted=0, returned=0),
            interpretation="fixed joint TRAIN templates, not chronological retention/return or generalization",
        )
    try:
        sampler.sample()
        budget.attach_memory(sampler)
        spec, states = fit_reference(report, budget, **fit_options)
        if joint_policy_context:
            _read_joint_reference(report, budget, spec, states, rng_before)
        else:
            cases = tuple(
                (scenario, concept)
                for scenario in ("ordinary", "stale_once")
                for concept in VOCABULARY
            ) + (("revoke_commit", "invoice"),)
            for control, source in (("unchanged", 0), ("learned", PASSES)):
                predict = PerceptronDecisionPredictor(spec, states[source])
                for index, (scenario, concept) in enumerate(cases):
                    budget.before_final()
                    episode = run_workflow_episode(
                        WorkflowState(f"development-ticket-{index}", concept),
                        predict,
                        capabilities=WorkflowCapabilities(ACTION_IDS),
                        max_steps=6,
                        scenario=scenario,
                        **context_options,
                    )
                    report["episodes"].append(
                        dict(
                            control=control,
                            source_updates=source,
                            role="outer_selection",
                            source_group="development-workflow-traces",
                            episode=episode,
                        )
                    )
                    budget.before_final()
                    if not episode["complete"]:
                        raise RuntimeError(
                            "workflow episode contains an unvalidated or unreturned paid prefix"
                        )
        report["summary"] = _summary(report["episodes"])
        report["complete"] = True
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)[:512]
    finally:
        if joint_policy_context:
            report["numpy_rng_unchanged"] = _same_value(np.random.get_state(), rng_before)
            report["budget"] = dict(
                limits=asdict(budget.budget),
                updates_completed=budget.updates_completed,
                started_at=budget.started_at,
                last_clock=budget.last_clock,
            )
            if not report["numpy_rng_unchanged"]:
                report.update(complete=False, error="joint example changed caller RNG")
        works = [row["episode"]["work"] for row in report["episodes"]]
        report["episode_work"] = (
            {key: sum(w[key] for w in works) for key in works[0]} if works else {}
        )
        count = report["episode_work"].get("prediction_calls_validated", 0)
        report["costs"] = dict(
            elapsed_seconds=monotonic() - started,
            model_random_draws=0,
            committed_training_affine_terms=budget.updates_completed * 12 * augmented_width * 6,
            validated_readout_affine_terms=count * augmented_width * 5,
            retained_snapshot_coefficient_bytes=6
            * augmented_width
            * 8
            * len(report.get("snapshots", [])),
            committed_coefficient_addition_terms=None
            if report.get("training_work", {}).get("latest_observed_mistakes") is None
            else 2 * augmented_width * report["training_work"]["latest_observed_mistakes"],
            qualification="nominal two script states; actor/array/copies/softmax/validation/Python additional; unobserved attempted work unknown on failure",
        )
        if sampler.start_bytes is not None:
            sampler.sample()
            report["costs"]["process_rss_observation"] = asdict(sampler.snapshot())
        if joint_policy_context:
            report["costs"]["attempted_teacher_readout_affine_terms"] = (
                report["teacher_read_work"]["attempted"] * augmented_width * 5
            )
    return report


def _encode_report(report):
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 512 * 1024:
        # Whitespace must not discard a paid, otherwise bounded episode prefix.
        encoded = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(encoded) > 512 * 1024:
        raise ValueError("workflow report exceeds512KiB")
    return encoded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--policy-context",
        action="store_true",
        help="include public policy-version tokens in the D9 query features",
    )
    modes.add_argument(
        "--joint-policy-context",
        action="store_true",
        help="fit the fixed v1/v2 joint D9 TRAIN tasks",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("output must be fresh")
    report = (
        run_example(joint_policy_context=True)
        if args.joint_policy_context
        else run_example(include_policy_context=True)
        if args.policy_context
        else run_example()
    )
    encoded = _encode_report(report)
    with args.output.open("xb") as stream:
        stream.write(encoded)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
