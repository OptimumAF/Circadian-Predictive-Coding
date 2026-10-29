"""Frozen controller comparison behind the existing native serving CLI.

Existing streams own all TRAIN work. Every candidate finishes before readonly
final source access. This recipe neither fits/selects policies nor promotes.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import run_bp_ticket_serving as demo
from scripts.ticket_controller_streams import WITHHELD_RECIPES
from src.core.contextual_controller import ContextualChoice, ContextualEstimate, FittedController
from src.core.controller_objective import (
    ControllerMeasures,
    ControllerObjective,
    controller_utility,
)
from scripts.ticket_catalog_control import CatalogTicketControl, case_work, deferred_case_work
from src.app.probe_return_phase import paid_report_guard
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.catalog_contextual_controller import (
    ContextualArmCatalog,
    CatalogEstimate,
    FittedCatalogController,
    CatalogTrialWork,
)
from src.core.reference_controller import ArrivedTrainingFeedback

POLICY_MODES = ("no_learning", "periodic", "drift", "budget_aware", "fitted")
POLICY_FITTED_SHA = "02e929f14a3dce174b4871e5aebdaaaaf3ce9225bda391bcf26a3e154a479415"


def _frozen_fit(path: Path, *, catalog_fit=False):
    if type(catalog_fit) is not bool:
        raise ValueError("catalog fit mode must be an exact boolean")
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("fitted source exceeds original learned report bound")
    raw = path.read_bytes()
    record = json.loads(raw)
    if catalog_fit:
        return raw, *_catalog_fit_record(record)
    p = record["fitted_policy"]
    if (
        record["complete"] is not True
        or record["definition"] != "learned_contextual_controller_v1"
        or record["source_sha256"] != demo.context.SOURCE_SHA
        or record["training_state"]["fitted"] != p
        or record["training_state"]["completed"] != 4
        or len(record["trials"]) != 4
    ):
        raise ValueError("original completed frozen contextual fit required")
    policy = FittedController(
        p["learner_version"],
        p["state_digest"],
        ControllerObjective(**p["objective"]),
        p["fitted_at"],
        tuple(ContextualEstimate(**e) for e in p["estimates"]),
        ContextualChoice(**p["choices"]),
    )
    if policy.objective != ControllerObjective():
        raise ValueError("comparison requires unchanged prospective utility")
    works = [t["result"]["training_work"] for t in record["trials"]]
    for work in works:
        if (
            work["unreturned_work_unknown"]
            or type(work["attempted_updates"]) is not int
            or not 0 <= work["attempted_updates"] <= 8
            or work["attempted_updates"] != work["returned_updates"]
            or work["attempted_rows"] != 6 * work["attempted_updates"]
        ):
            raise ValueError("fitted source training work is incomplete/unknown")
    historical = dict(
        fit_training_updates=sum(w["attempted_updates"] for w in works),
        fit_training_rows=sum(w["attempted_rows"] for w in works),
        whole_source_study_training=record["total_training_work"],
        whole_source_study_costs=record["costs"],
        exact_fit_overhead_separable=False,
        all_in_utility=None,
        qualification="fit native work known; source study also contains applications; do not impute missing fit overhead or claim net/amortized benefit",
    )
    return raw, policy, historical


def _catalog_fit_record(record):
    """Read scalar fit/cost records only; never restore a discarded model."""
    p, state = record["fitted_policy"], record["training_state"]
    c = p["catalog"]["envelope"]
    declaration = dict(c)
    declaration["actions"] = tuple(AdaptationAction(**a) for a in c["actions"])
    declaration["gate_settings"] = tuple(c["gate_settings"])
    envelope = AdaptationEnvelope(**declaration)
    catalog = ContextualArmCatalog(envelope)
    policy = FittedCatalogController(
        p["learner_version"],
        p["state_digest"],
        ControllerObjective(**p["objective"]),
        p["fitted_at"],
        tuple(CatalogEstimate(**e) for e in p["estimates"]),
        (p["choices"][0], p["choices"][1]),
        catalog=catalog,
    )
    CatalogTicketControl(
        None,
        policy,
        training_context="new_contexts",
        base_snapshot=None,
        unsupported=False,
        batches=demo.context.BATCHES_PER_PHASE,
    )
    if (
        record.get("complete") is not True
        or record.get("definition") != "catalog_ticket_controller_learning_v1"
        or record.get("source_sha256") != demo.context.SOURCE_SHA
        or state.get("fitted") != p
        or state.get("completed") != 6
        or state.get("attempts") != 6
        or state.get("discarded_trials") != 6
        or state.get("stopped") is not False
        or state.get("failed_work") is not None
        or len(record["trials"]) != 6
        or record.get("catalog") != p["catalog"]
        or state.get("catalog") != p["catalog"]
        or state.get("rows") != p["estimates"]
        or record.get("objective") != p["objective"]
        or len(p["choices"]) != 2
        or policy.objective != ControllerObjective()
        or policy.learner_version != "qualified-96"
        or record.get("final_source_reads") != 0
        or record.get("probe_updates") != 0
        or record.get("source_unchanged") is not True
        or record.get("numpy_rng_unchanged") is not True
        or record.get("torch_rng_unchanged") is not True
        or list(record["applications"]) != ["new_contexts", "old_return"]
    ):
        raise ValueError("original complete six-trial frozen catalog fit required")
    estimates = {(e.context, e.action_id): e.mean_utility for e in policy.estimates}
    monitoring = record["context_monitoring"]
    if (
        len(monitoring) != 2
        or record["fitted_at"] != policy.fitted_at
        or state["clock"] != policy.fitted_at
    ):
        raise ValueError("catalog fit lost its original causal monitoring/freeze clock")
    for index, kind in enumerate(("new_contexts", "old_return")):
        feedback = ArrivedTrainingFeedback(**monitoring[index])
        if (
            feedback.actor_version != policy.learner_version
            or feedback.examples != 6
            or feedback.sample_id != "context-" + kind
            or feedback.arrived_at <= feedback.observed_at
            or record["trials"][index * 3]["request"]["started_at"] < feedback.arrived_at
            or (feedback.mistakes == 0) != (kind == "old_return")
        ):
            raise ValueError("catalog fit requires its original arrived TRAIN contexts")
    expected = [(ctx, arm) for ctx in ("mistakes", "clean") for arm in catalog.arms]
    works, prepared = [], []
    previous = -1
    for row, cell in zip(record["trials"], expected, strict=True):
        request, result = row["request"], row["result"]
        if (
            (request["context"], request["action_id"]) != cell
            or request.get("role") != "controller_train"
            or request.get("catalog") != p["catalog"]
            or request.get("learner_version") != policy.learner_version
            or request.get("state_digest") != policy.state_digest
            or type(request["started_at"]) is not int
            or not previous < request["started_at"] < policy.fitted_at
            or result["catalog_control"]["record"] != request
            or result["catalog_control"]["phase"] != "trial"
        ):
            raise ValueError("catalog fit requires every original causal context/arm receipt")
        previous = request["started_at"]
        work = _catalog_source_case(result)
        work.require_complete(catalog, request["action_id"])
        if work != CatalogTrialWork(**row["native_work"]):
            raise ValueError("catalog fit native costs changed")
        measures = ControllerMeasures(**row["measures"])
        if measures.new_count != 2 or measures.old_count != 2:
            raise ValueError("catalog fit must preserve the original task count metrics")
        if controller_utility(policy.objective, measures).value != estimates[cell]:
            raise ValueError("catalog fit utility differs from its original paid measures")
        if any(
            getattr(measures, k) != result["reference_outcomes"][k]
            for k in ("new_before", "new_after", "old_before", "old_after")
        ):
            raise ValueError("catalog fit measures differ from original task outcomes")
        works.append(work)
        prepared.append(result["catalog_control"]["preparation"])
    fit_works = tuple(works)
    for kind, result in record["applications"].items():
        if (
            result["training_context"] != kind
            or result["catalog_control"]["record"] != p
            or result["catalog_control"]["phase"] != "private_application"
        ):
            raise ValueError("source application changed frozen fit")
        works.append(_catalog_source_case(result))
        prepared.append(result["catalog_control"]["preparation"])
    fields = (
        "attempted_updates",
        "returned_updates",
        "attempted_rows",
        "returned_rows",
        "replay_examples_charged",
        "prediction_calls",
        "prediction_rows",
    )
    totals = {k: sum(getattr(w, k) for w in works) for k in fields}
    CatalogTrialWork(**record["total_catalog_work"])
    if (
        record["total_catalog_work"] != totals
        or record["total_replay_examples"] != totals["replay_examples_charged"]
        or record["total_training_work"]
        != {k: totals[k] for k in ("attempted_updates", "attempted_rows", "returned_updates")}
        or record["prior_training_work"]
        != dict(
            attempted_updates=96,
            attempted_rows=576,
            returned_updates=96,
            unreturned_work_unknown=False,
        )
        or any(
            type(b.get("attempts")) is not int
            or b.get("attempts") != 1
            or b.get("returned") is not True
            or type(b.get("prepared_rows")) is not int
            or type(b.get("prepared_episodes")) is not int
            or not 1 <= b["prepared_episodes"] <= 12
            or b["prepared_rows"] != 3 * b["prepared_episodes"]
            or b.get("retained_rows") != 3
            for b in prepared
        )
    ):
        raise ValueError("catalog source study costs are incomplete or changed")
    return policy, dict(
        fit_training_updates=sum(w.attempted_updates for w in fit_works),
        fit_training_rows=sum(w.attempted_rows for w in fit_works),
        fit_replay_examples=sum(w.replay_examples_charged for w in fit_works),
        fit_prediction_calls=sum(w.prediction_calls for w in fit_works),
        fit_prediction_rows=sum(w.prediction_rows for w in fit_works),
        fit_prepared_teacher_rows=sum(b["prepared_rows"] for b in prepared[:6]),
        fit_prepared_teacher_episodes=sum(b["prepared_episodes"] for b in prepared[:6]),
        whole_source_prepared_teacher_rows=sum(b["prepared_rows"] for b in prepared),
        whole_source_prepared_teacher_episodes=sum(b["prepared_episodes"] for b in prepared),
        whole_source_context_monitoring=record["context_monitoring"],
        whole_source_catalog_work=record["total_catalog_work"],
        whole_source_study_training=record["total_training_work"],
        whole_source_study_costs=record["costs"],
        source_pretraining_work=record["prior_training_work"],
        source_pretraining_costs=record["trials"][0]["result"]["prior_costs"],
        exact_fit_overhead_separable=False,
        all_in_utility=None,
        qualification="all discarded native/row/replay/prediction/teacher preparation work carried; source study includes applications; no imputed net/amortized utility",
    )


def _catalog_source_case(result):
    if result.get("definition") != "catalog_bp_ticket_private_v1" or any(
        type(result.get(k)) is not int or result[k] != 0
        for k in (
            "catalog_promotions",
            "final_examples",
            "reserved_probe_updates",
            "external_actions",
        )
    ):
        raise ValueError("catalog source requires original private TRAIN trial/application")
    return case_work(result)


def run_comparison(
    input_path: Path, fitted_path: Path, *, policy_change=False, catalog_fit=False
) -> dict[str, Any]:
    if (
        type(policy_change) is not bool
        or type(catalog_fit) is not bool
        or policy_change
        and catalog_fit
    ):
        raise ValueError("comparison mode must be an exact boolean")
    if policy_change:
        return _run_policy_comparison(input_path, fitted_path)
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="withheld_controller_comparison_v1",
        complete=False,
        cases=[],
        final_readouts=[],
        models_frozen_at=None,
        final_training_updates=0,
        promotions=0,
        new_controller_fits=0,
    )
    if catalog_fit:
        report["definition"] = "catalog_controller_comparison_v1"
    torch = demo.native.require_torch()
    rng, torch_rng = np.random.get_state(), torch.get_rng_state().clone()
    assert type(rng) is tuple
    with demo.ProcessRssSampler(interval_seconds=0.02) as sampler:
        budget = demo.ToyBudgetSession(
            demo.ToyExecutionBudget(max_wall_seconds=180, max_process_rss_bytes=512 * 1024**2),
            monotonic,
        )
        budget.attach_memory(sampler)
        try:
            fit_raw, fitted, historical = (
                _frozen_fit(fitted_path, catalog_fit=True)
                if catalog_fit
                else _frozen_fit(fitted_path)
            )
            fit_pin = deepcopy(fitted)
            recipes = deepcopy(WITHHELD_RECIPES)
            source_raw = demo.context._source(input_path)[0] if catalog_fit else None
            horizons = (
                {
                    r.recipe_id: min(
                        len(r.contexts) + r.label_delay, fitted.catalog.envelope.max_decisions
                    )
                    for r in recipes
                }
                if catalog_fit
                else {}
            )
            modes = (*demo.REFERENCE_MODES, "fitted")
            report["historical_fit_costs"] = historical
            report["protocol"] = dict(
                declared_at=0,
                objective=asdict(fitted.objective),
                recipes=[asdict(r) for r in recipes],
                references=[asdict(demo.ReferencePolicy(m)) for m in demo.REFERENCE_MODES],
                fitted=asdict(fitted),
                fitted_report_sha256=sha256(fit_raw).hexdigest(),
                source_sha256=demo.context.SOURCE_SHA,
                case_order=[(r.recipe_id, m) for r in recipes for m in modes],
                utility_scope="incremental native trial callback + readonly task callback wall time and max sampled RSS; shared outer overhead separately recorded; historical fit work carried but exact all-in utility unavailable",
                final_scope="readonly shared-grammar task episodes after all candidates finish; schedule/receipt shift, not novel task/features or independent statistical confirmation",
            )
            if catalog_fit:
                report["protocol"].update(catalog_fit=True, common_opportunities=horizons)
            protocol_pin = deepcopy(report["protocol"])
            paid = []

            def remember(getter, value):
                guard, _ = paid_report_guard(value)
                paid.append((getter, value, guard))

            def boundary():
                budget.before_final()
                if catalog_fit:
                    if input_path.read_bytes() != source_raw:
                        raise ValueError("original catalog learner source changed")
                    for current, original, guard in paid:
                        if current() is not original:
                            raise ValueError(
                                "preceding paid catalog comparison evidence was replaced"
                            )
                        guard()
                if (
                    fitted_path.read_bytes() != fit_raw
                    or fitted != fit_pin
                    or report["protocol"] != protocol_pin
                ):
                    raise ValueError("original fit/protocol changed; further work refused")

            for recipe in recipes:
                for mode in modes:
                    boundary()
                    case = dict(recipe_id=recipe.recipe_id, mode=mode, result=None)
                    report["cases"].append(case)
                    then = monotonic()
                    result = (
                        demo.run_case(
                            input_path,
                            stream_recipe=recipe,
                            deferred_evaluation=True,
                            **(
                                {
                                    "catalog_policy": fitted,
                                    "decision_limit": horizons[recipe.recipe_id],
                                }
                                if catalog_fit
                                else {"contextual_policy": fitted}
                            ),
                        )
                        if mode == "fitted"
                        else demo.run_case(
                            input_path,
                            stream_recipe=recipe,
                            deferred_evaluation=True,
                            reference_mode=mode,
                            **(
                                {"decision_limit": horizons[recipe.recipe_id]}
                                if catalog_fit
                                else {}
                            ),
                        )
                    )
                    case.update(result=result, execution_seconds=monotonic() - then)
                    boundary()
                    if (
                        not result["complete"]
                        or result["training_work"]["unreturned_work_unknown"]
                        or not result["source_unchanged"]
                        or not result["deferred_actor_unchanged"]
                        or not result["numpy_rng_unchanged"]
                        or not result["torch_rng_unchanged"]
                        or result["deferred_promotions"] != 0
                        or result["episodes"]
                    ):
                        raise ValueError(
                            "preceding native candidate incomplete/unknown; evaluation closed"
                        )
                    if catalog_fit:
                        work = _catalog_comparison_case(
                            result, mode, recipe, horizons[recipe.recipe_id]
                        )
                        if work is not None:
                            case["native_work"] = asdict(work)
                        index = len(report["cases"]) - 1
                        remember(lambda i=index: report["cases"][i], case)
            boundary()
            frozen_cases = deepcopy(report["cases"])
            if catalog_fit:
                remember(lambda: frozen_cases, frozen_cases)
            report["models_frozen_at"] = 1
            clock = demo.LogicalClock(1)
            for case in frozen_cases:
                boundary()
                if report["cases"] != frozen_cases:
                    raise ValueError("frozen candidate evidence changed before final readout")
                final = dict(recipe_id=case["recipe_id"], mode=case["mode"], sealed_source_reads=[])
                report["final_readouts"].append(final)
                measures = demo._sealed_readout(case["result"], final, clock)
                if catalog_fit:
                    boundary()
                combined = replace(
                    measures,
                    execution_seconds=case["execution_seconds"] + measures.execution_seconds,
                    peak_rss_bytes=max(
                        measures.peak_rss_bytes,
                        case["result"]["costs"]["sampled_process_rss"]["peak_bytes"],
                    ),
                )
                final["incremental_measures"] = asdict(combined)
                final["incremental_utility"] = asdict(
                    controller_utility(fitted.objective, combined)
                )
                final["all_in_utility"] = None
                if catalog_fit:
                    index = len(report["final_readouts"]) - 1
                    remember(lambda i=index: report["final_readouts"][i], final)
                boundary()
            report["clock"] = clock.now()
            if catalog_fit:
                report["total_training_work"] = {
                    k: sum(c["result"]["training_work"][k] for c in report["cases"])
                    for k in ("attempted_updates", "attempted_rows", "returned_updates")
                }
                report["total_replay_examples"] = sum(
                    c["result"]["replay_examples"] for c in report["cases"]
                )
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            current = np.random.get_state()
            assert type(current) is tuple
            report["numpy_rng_unchanged"] = bool(
                rng[0] == current[0]
                and np.array_equal(rng[1], current[1])
                and rng[2:] == current[2:]
            )
            report["torch_rng_unchanged"] = bool(torch.equal(torch_rng, torch.get_rng_state()))
            if not report["numpy_rng_unchanged"] or not report["torch_rng_unchanged"]:
                report.update(complete=False, error="caller RNG changed")
            report["costs"] = dict(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
            )
    return report


def _catalog_comparison_case(result, mode, recipe, horizon):
    """Admit complete paid TRAIN work before another candidate or readonly final."""
    n = len(recipe.contexts)
    stream, work = result["stream"], result["training_work"]
    progress, monitoring = result["action_snapshot"]["progress"], result["controller_work"]
    wakes = result["budget"]["updates_completed"]
    state = result["candidate_checkpoint"]["state"]
    counts = (
        wakes,
        *(work[k] for k in ("attempted_updates", "attempted_rows", "returned_updates")),
    )
    if (
        result.get("deferred_evaluation") is not True
        or any(type(v) is not int or v < 0 for v in counts)
        or any(
            k in result
            for k in (
                "candidate_episodes",
                "candidate_readout_work",
                "reference_outcomes",
                "promotion_receipt",
            )
        )
        or not 0 <= wakes <= n
        or any(
            result.get(k) != 0
            for k in ("final_examples", "reserved_probe_updates", "external_actions")
        )
        or stream["opportunities"] != horizon
        or stream["registered_batches"] != n
        or stream["arrived_labels"] != horizon - recipe.label_delay
        or stream["unarrived_labels"] != n - stream["arrived_labels"]
        or stream["pending_unapplied"] != n - wakes
        or monitoring["prediction_attempts"] != n
        or monitoring["prediction_returns"] != n
        or monitoring["selections_attempted"] != horizon
        or monitoring["selections_returned"] != horizon
        or progress["native_attempts"] != work["attempted_updates"]
        or progress["native_completions"] != work["returned_updates"]
        or progress["unreturned_work_unknown"] is not False
        or state["completed_updates"] != 96 + work["returned_updates"]
        or state["training_examples"] != 576 + work["attempted_rows"]
        or result["budget"]["limits"]
        != asdict(
            demo.ToyExecutionBudget(
                max_training_updates=n, max_wall_seconds=120, max_process_rss_bytes=512 * 1024**2
            )
        )
        or len(result["applied_updates"]) != wakes
    ):
        raise ValueError("catalog comparison lost matched opportunity/complete native history")
    if mode == "fitted":
        return deferred_case_work(result)
    if (
        result["replay_examples"] != 0
        or work["attempted_updates"] != wakes
        or work["attempted_rows"] != 12 * wakes
    ):
        raise ValueError("reference native work differs from original wake-only stream")
    return None


def _require_policy_trial(result, mode):
    """Check original paid completion before another candidate or any readout."""
    if (
        result.get("complete") is not True
        or result.get("definition") != "policy_change_native_reference_v1"
        or result.get("policy_change") is not True
        or result.get("deferred_evaluation") is not True
        or result.get("policy_capture_complete") is not True
        or any(
            result.get(k) is not True
            for k in (
                "source_unchanged",
                "deferred_actor_unchanged",
                "numpy_rng_unchanged",
                "torch_rng_unchanged",
            )
        )
        or type(result.get("deferred_promotions")) is not int
        or result.get("deferred_promotions") != 0
        or result.get("episodes") != []
        or any(
            k in result
            for k in ("policy_development_episodes", "policy_readout_work", "policy_summary")
        )
        or any(
            type(result.get(k)) is not int or result.get(k) != 0
            for k in (
                "final_examples",
                "reserved_probe_updates",
                "replay_examples",
                "external_actions",
            )
        )
        or result.get("source_sha256") != demo.context.SOURCE_SHA
    ):
        raise ValueError(
            "preceding policy candidate incomplete/unknown or evaluated; further work closed"
        )
    work = result["training_work"]
    n = work["attempted_updates"]
    integer_counts = (
        work["returned_updates"],
        work["attempted_rows"],
        result["budget"]["updates_completed"],
        result["stream"]["opportunities"],
        result["stream"]["arrived_labels"],
        result["stream"]["pending_unapplied"],
    )
    if (
        type(n) is not int
        or any(type(value) is not int for value in integer_counts)
        or not 0 <= n <= 12
        or (mode == "no_learning" and n != 0)
        or work["unreturned_work_unknown"] is not False
        or work["returned_updates"] != n
        or work["attempted_rows"] != 6 * n
        or result["budget"]["updates_completed"] != n
        or result["budget"]["limits"]
        != asdict(
            demo.ToyExecutionBudget(
                max_training_updates=12, max_wall_seconds=120, max_process_rss_bytes=512 * 1024**2
            )
        )
        or len(result["applied_updates"]) != n
        or result["stream"]["opportunities"] != 13
        or result["stream"]["arrived_labels"] != 12
        or result["stream"]["pending_unapplied"] != 12 - n
        or result["controller_work"]["prediction_rows"] != 72
        or result["controller_work"]["feedback_rows"] != 72
        or result["controller_work"]["selections_returned"] != 13
        or result["action_snapshot"]["envelope"]["max_decisions"] != 13
        or result["action_snapshot"]["envelope"]["max_wake_updates"] != 12
        or result["sharing_snapshot"]["admitted_updates"] != n
        or result["sharing_snapshot"]["limits"]["max_admitted_updates"] != 12
        or result["action_snapshot"]["progress"]["native_completions"] != n
        or result["action_snapshot"]["progress"]["unreturned_work_unknown"] is not False
    ):
        raise ValueError("original policy candidate work/clock/envelope differs")
    demo.native._policy_phase_records(result)


def _policy_outcomes(evaluation):
    """Admit complete declared sandbox outcomes, never a favorable subset."""
    from scripts.recurring_ticket_workflow_example import episode_slots
    from scripts.ticket_controller_streams import POLICY_CHANGE_CHECKPOINTS
    from src.core.ticket_workflow import WorkflowState, workflow_success

    if evaluation.get("complete") is not True:
        raise ValueError("unknown policy readout has no score")
    rows = evaluation["episodes"]
    expected = [
        (tick, version, concept, scenario)
        for tick, version in POLICY_CHANGE_CHECKPOINTS
        for scenario, concept in (
            *(("ordinary", c) for c in ("invoice", "parcel", "password")),
            *((s, c) for s, c in episode_slots(tick) if s != "ordinary"),
        )
    ]
    if type(rows) is not list or len(rows) != len(expected):
        raise ValueError("policy scoring requires all15 declared episodes")
    ordinary = {}
    for row, identity in zip(rows, expected, strict=True):
        ep = row["episode"]
        actual = (row["opportunity"], row["policy_version"], row["concept"], ep["scenario"])
        if (
            actual != identity
            or type(row["opportunity"]) is not int
            or row["role"] != "outer_selection"
            or row["source_group"] != "development-native-policy-workflow"
            or type(row["source_updates"]) is not int
            or row["source_updates"] != evaluation["source_updates_by_opportunity"][identity[0]]
            or ep["complete"] is not True
            or type(ep["success"]) is not bool
            or type(ep["training_updates"]) is not int
            or ep["training_updates"] != 0
            or type(ep["external_actions"]) is not int
            or ep["external_actions"] != 0
            or ep["unreturned_prediction_work_unknown"] is not False
            or ep["unvalidated_return_work_unknown"] is not False
            or ep["initial_state"]["policy_version"] != identity[1]
            or ep["initial_state"]["concept"] != identity[2]
            or ep["final_state"]["policy_version"] != identity[1]
            or ep["final_state"]["concept"] != identity[2]
        ):
            raise ValueError(
                "policy scoring requires unique known policy-bound development outcomes"
            )
        if workflow_success(WorkflowState(**ep["final_state"])) is not ep["success"]:
            raise ValueError("policy success differs from actual sandbox final state")
        if ep["scenario"] == "ordinary":
            ordinary[row["opportunity"], row["concept"]] = int(ep["success"])
    return ordinary


def _phase_matched_measures(evaluation, baseline):
    current, unchanged = _policy_outcomes(evaluation), _policy_outcomes(baseline)
    count = lambda values, concepts: sum(values[p, c] for p in (4, 8, 12) for c in concepts)
    return ControllerMeasures(
        count(unchanged, ("invoice", "parcel")),
        count(current, ("invoice", "parcel")),
        6,
        count(unchanged, ("password",)),
        count(current, ("password",)),
        3,
        evaluation["execution_seconds"],
        evaluation["peak_rss_bytes"],
        demo._violation_count([r["episode"] for r in evaluation["episodes"]]),
        True,
    )


def _run_policy_comparison(input_path, fitted_path):
    """One frozen five-controller development observation; no fitting or final IO."""
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="phase_matched_policy_controllers_v1",
        complete=False,
        cases=[],
        development_readouts=[],
        models_frozen_at=None,
        new_controller_fits=0,
        final_training_updates=0,
        final_examples=0,
        promotions=0,
    )
    torch = demo.native.require_torch()
    rng, torch_rng = np.random.get_state(), torch.get_rng_state().clone()
    assert type(rng) is tuple
    with demo.ProcessRssSampler(interval_seconds=0.02) as sampler:
        budget = demo.ToyBudgetSession(
            demo.ToyExecutionBudget(max_wall_seconds=180, max_process_rss_bytes=512 * 1024**2),
            monotonic,
        )
        budget.attach_memory(sampler)
        try:
            source_raw, _ = demo.context._source(input_path)
            fit_raw, fitted, historical = _frozen_fit(fitted_path)
            if sha256(fit_raw).hexdigest() != POLICY_FITTED_SHA:
                raise ValueError("policy comparison requires the original exact frozen fit")
            fit_pin = deepcopy(fitted)
            report["historical_fit_costs"] = historical
            report["protocol"] = dict(
                declared_at=0,
                case_order=POLICY_MODES,
                objective=asdict(fitted.objective),
                fitted=asdict(fitted),
                fitted_report_sha256=POLICY_FITTED_SHA,
                source_sha256=demo.context.SOURCE_SHA,
                references=[asdict(demo.ReferencePolicy(m)) for m in demo.REFERENCE_MODES],
                recipe="A_v1(invoice,password)x4:B_v2(parcel,password)x4:A_return_CURRENTv2(invoice,password)x4",
                row_counts=[6] * 12,
                label_delay=1,
                opportunities=13,
                max_updates=12,
                scoring=dict(
                    phase_ends=[4, 8, 12],
                    current_tasks=["invoice", "parcel"],
                    current_count=6,
                    unchanged_tasks=["password"],
                    unchanged_count=3,
                    baseline="same-phase no_learning",
                    direction="higher utility/gain, lower drop/cost/violations",
                ),
                utility_scope="same-phase task gain/drop; own native trial callback+readout wall time/max sampled RSS; original unitweights; not old benchmark grouping or matched computation",
                evaluation_scope="development only after all5models frozen; all0/4/8/12 retained; no final source",
                all_in_utility=None,
            )
            protocol_pin = deepcopy(report["protocol"])
            frozen_cases = None
            accepted_cases = []

            def boundary():
                budget.before_final()
                if (
                    input_path.read_bytes() != source_raw
                    or fitted_path.read_bytes() != fit_raw
                    or fitted != fit_pin
                    or report["protocol"] != protocol_pin
                    or report["cases"][: len(accepted_cases)] != accepted_cases
                    or (frozen_cases is not None and report["cases"] != frozen_cases)
                ):
                    raise ValueError("original policy source/fit/protocol/cohort changed")

            for mode in POLICY_MODES:
                boundary()
                case = dict(mode=mode, result=None)
                report["cases"].append(case)
                then = monotonic()
                result = demo.run_case(
                    input_path,
                    policy_change=True,
                    deferred_evaluation=True,
                    **(
                        {"contextual_policy": fitted}
                        if mode == "fitted"
                        else {"reference_mode": mode}
                    ),
                )
                case.update(result=result, execution_seconds=monotonic() - then)
                boundary()
                _require_policy_trial(result, mode)
                accepted_cases.append(deepcopy(case))
            frozen_cases = deepcopy(report["cases"])
            report["models_frozen_at"] = 1
            spec = demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
            for index, case in enumerate(frozen_cases):
                boundary()
                result = deepcopy(case["result"])
                evaluation = dict(mode=case["mode"], opened_at=index + 2, complete=False)
                evaluation["source_updates_by_opportunity"] = {
                    row["opportunity"]: row["applied_receipts"]
                    for row in result["policy_checkpoints"]
                }
                report["development_readouts"].append(evaluation)
                then = monotonic()
                with demo.ProcessRssSampler(interval_seconds=0.02) as readout_sampler:
                    try:
                        demo.native.policy_development_readout(spec, result, boundary)
                        evaluation["complete"] = result["policy_readout_complete"]
                    finally:
                        evaluation.update(
                            episodes=result.get("policy_development_episodes", []),
                            readout_work=result.get("policy_readout_work", []),
                            readout_seconds=monotonic() - then,
                            readout_peak_rss_bytes=readout_sampler.snapshot().peak_bytes,
                        )
                boundary()
                if evaluation["complete"] is not True:
                    raise ValueError("unknown policy readout has no utility")
                evaluation.update(
                    execution_seconds=case["execution_seconds"] + evaluation["readout_seconds"],
                    peak_rss_bytes=max(
                        case["result"]["costs"]["sampled_process_rss"]["peak_bytes"],
                        evaluation["readout_peak_rss_bytes"],
                    ),
                )
                _policy_outcomes(evaluation)
            baseline = report["development_readouts"][0]
            for evaluation in report["development_readouts"]:
                boundary()
                measures = _phase_matched_measures(evaluation, baseline)
                evaluation.update(
                    phase_matched_measures=asdict(measures),
                    incremental_utility=asdict(controller_utility(fitted.objective, measures)),
                    all_in_utility=None,
                )
            report.update(complete=True, clock=7)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            current = np.random.get_state()
            assert type(current) is tuple
            report["numpy_rng_unchanged"] = bool(
                rng[0] == current[0]
                and np.array_equal(rng[1], current[1])
                and rng[2:] == current[2:]
            )
            report["torch_rng_unchanged"] = bool(torch.equal(torch_rng, torch.get_rng_state()))
            if not report["numpy_rng_unchanged"] or not report["torch_rng_unchanged"]:
                report.update(complete=False, error="policy comparison caller RNG changed")
            report["costs"] = dict(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
            )
    return report
