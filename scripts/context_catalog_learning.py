"""Full contextual catalog via the existing live recipe and empirical fitter.

Six isolated TRAIN trials then one frozen development application; no alternate
runner, optimizer, final/probe labels, promotion or comparative study selection.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from time import monotonic
from typing import Any

import numpy as np

from scripts import context_head_live as live
from scripts.context_catalog_control import context_catalog
from scripts.context_controller_learning import _task_counts
from scripts.context_head_serving import prepare_data
from scripts.run_bp_ticket_serving import _violation_count, _reported_work_fields
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.context_serving import snapshot_digest
from src.app.actor_shadow import StableActor
from src.app.counterfactual_controller import CounterfactualController, CatalogCounterfactualReturn
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard
from src.core.catalog_contextual_controller import CatalogTrialWork
from src.core.catalog_contextual_controller import (
    CatalogEstimate,
    ContextualArmCatalog,
    FittedCatalogController,
)
from src.core.controller_objective import ControllerMeasures, ControllerObjective
from src.core.decision_features import DecisionFeatureSpec
from src.core.experience import LogicalClock
from src.core.reference_controller import ArrivedTrainingFeedback, ReferenceObservation
from src.core.ticket_workflow import CATALOG
from src.shared.process_memory import ProcessRssSampler


def controller_record(snapshot):
    """Project the existing controller's bounded scalar dataclasses, no tensors."""
    if type(snapshot) is not dict or set(snapshot) != {
        "attempts",
        "completed",
        "discarded_trials",
        "stopped",
        "rows",
        "fitted",
        "clock",
        "catalog",
        "failed_work",
    }:
        raise ValueError("catalog controller record schema differs")
    result = dict(snapshot)
    if (
        type(result["rows"]) is not tuple
        or len(result["rows"]) > 6
        or any(type(row) is not CatalogEstimate for row in result["rows"])
    ):
        raise ValueError("catalog controller record requires at most six exact estimates")
    result["rows"] = [asdict(row) for row in result["rows"]]
    for key, kind in (
        ("fitted", FittedCatalogController),
        ("catalog", ContextualArmCatalog),
        ("failed_work", CatalogTrialWork),
    ):
        value = result[key]
        if value is not None:
            if type(value) is not kind:
                raise ValueError("unknown catalog controller record field: " + key)
            result[key] = asdict(value)
    return result


def _failed_output(report, error, observed_work):
    """Bounded reported costs only; never reconstruct lost tasks/fit/receipts."""
    report = report if type(report) is dict else {}
    observed_work = observed_work if type(observed_work) is dict else {}
    cases: list[Any] = []
    rows = report.get("trials")
    if type(rows) is list:
        cases.extend(row.get("result") if type(row) is dict else None for row in rows[:6])
    if report.get("application") is not None:
        cases.append(report["application"])
    costs = []
    for case in cases:
        case = case if type(case) is dict else {}
        costs.append(
            dict(
                training_work=_reported_work_fields(
                    case.get("training_work"),
                    ("attempted_updates", "attempted_rows", "returned_updates"),
                ),
                budget=_reported_work_fields(
                    case.get("budget"), ("updates", "started_at", "last_clock")
                ),
                replay=_reported_work_fields(case, ("replay_examples",)),
            )
        )
    return dict(
        definition="failed_context_catalog_output_v1",
        complete=False,
        reason=f"{type(error).__name__}: {error}"[:512],
        producer_complete=report.get("complete") if type(report.get("complete")) is bool else None,
        detailed_outcomes_available=False,
        fitted_policy_available=False,
        validated_receipts_available=False,
        reported_case_costs=costs,
        observer_reported=_reported_work_fields(
            dict(
                elapsed_seconds=observed_work.get("seconds"),
                peak_rss_bytes=observed_work.get("peak_rss_bytes"),
            ),
            ("elapsed_seconds", "peak_rss_bytes"),
        ),
        observer_counts=_reported_work_fields(
            observed_work.get("counts"),
            (
                "train_passes",
                "training_rows",
                "learner_constructors",
                "zero_heads",
                "prediction_rows",
                "typed_requests",
                "sandbox_steps",
                "source_reads",
                "source_returned_bytes",
            ),
        ),
        qualification="scalar producer/observer reports, not validated native receipts or recovered outcomes; null means unavailable",
    )


def write_catalog_report(path, report, *, observed_work=None, max_bytes=24 * 1024**2):
    """Exclusive bounded JSON output; serialization failure retains scalar costs."""
    if type(max_bytes) is not int or not 32 * 1024 <= max_bytes <= 24 * 1024**2:
        raise ValueError("catalog output requires32KiB..24MiB declared bound")
    complete = report.get("complete") is True if type(report) is dict else False
    try:
        if type(report) is not dict:
            raise ValueError("catalog output requires exact report dictionary")
        payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
        if len(payload) > max_bytes:
            raise ValueError("catalog report exceeds original declared byte limit")
    except (TypeError, ValueError, OverflowError) as error:
        payload = (
            json.dumps(_failed_output(report, error, observed_work), indent=2, allow_nan=False)
            + "\n"
        ).encode()
        if len(payload) > 32 * 1024:
            raise ValueError("catalog failed scalar output exceeds32KiB") from error
        complete = False
    with path.open("xb") as handle:
        handle.write(payload)
    return complete


def require_observed_context(batch_context, feedback):
    """Do not execute a different prototype when the expected source signal fails."""
    if type(feedback) is not ArrivedTrainingFeedback:
        raise ValueError("catalog monitoring requires original arrived TRAIN feedback")
    feedback.__post_init__()
    expected = {"new-v1": "mistakes", "old-v1": "clean"}.get(batch_context)
    observed = "clean" if feedback.mistakes == 0 else "mistakes"
    if expected is None or observed != expected:
        raise ValueError("declared catalog TRAIN context was not observed; no substituted trial")


def catalog_measures(result):
    """Read complete ordinary task truth only after observed native/replay work."""
    if (
        result.get("complete") is not True
        or result.get("definition") != "live_context_catalog_v1"
        or result.get("source_unchanged") is not True
        or result.get("actor_unchanged") is not True
        or result.get("final_examples") != 0
        or result.get("probe_updates") != 0
        or "promotion_receipt" in result
    ):
        raise ValueError("incomplete or changed contextual catalog case has no utility")
    control = result["catalog_control"]
    native, replay = control["native_history"], control["replay"]
    work = result["training_work"]
    if (
        any(
            type(v) is not int or v < 0
            for v in (
                *native.values(),
                *work.values(),
                result["budget"]["updates"],
                result["replay_examples"],
            )
        )
        or result["stream"]["opportunities"] != 14
        or len(result["live_serving"]["records"]) != 14
        or any(r.get("complete") is not True for r in result["live_serving"]["records"])
    ):
        raise ValueError("catalog requires exact known counters and complete original live window")
    calls = native["wake_updates"] + native["replay_updates"]
    rows = native["wake_rows"] + native["replay_rows"]
    if (
        work != dict(attempted_updates=calls, returned_updates=calls, attempted_rows=rows)
        or result["budget"]["updates"] != native["wake_updates"]
        or result["replay_examples"] != native["replay_rows"]
        or len(result["native_calls"]) != calls
        or any(c.get("complete") is not True for c in result["native_calls"])
        or replay["stopped"]
        or replay["publication_unknown"]
        or replay["native_attempts"] != replay["native_completions"]
        or replay["committed_events"] != native["replay_updates"]
    ):
        raise ValueError("unknown or incomplete catalog physical work")
    before = _task_counts(result["task_outcomes"]["before"], "before", 12)
    after = _task_counts(result["task_outcomes"]["after"], "after", native["completed_updates"])
    if before != result["before"] or after != result["after"]:
        raise ValueError("catalog task counts differ from original complete episodes")
    task_calls = 0
    for phase in ("before", "after"):
        count = sum(len(r["reads"]) for r in result["task_outcomes"][phase])
        if any(
            result["task_readout_work"][phase][k] != count
            for k in ("attempts", "returns", "readouts", "typed_attempts")
        ):
            raise ValueError("unknown catalog task prediction work")
        task_calls += count
    typed, observed = result["typed_work"], result["controller_work"]
    if (
        typed["attempts"] != typed["returns"]
        or observed["prediction_attempts"] != observed["prediction_returns"]
    ):
        raise ValueError("unknown serving/observation prediction work")
    predictions = task_calls + typed["attempts"] + observed["prediction_attempts"]
    prediction_rows = task_calls + typed["attempts"] + observed["prediction_rows"]
    episodes = [
        r["episode"] for phase in ("before", "after") for r in result["task_outcomes"][phase]
    ]
    return (
        ControllerMeasures(
            before[0],
            after[0],
            2,
            before[1],
            after[1],
            4,
            result["elapsed_seconds"],
            result["memory"]["peak_bytes"],
            _violation_count(episodes),
            True,
        ),
        CatalogTrialWork(
            calls, calls, rows, rows, native["replay_rows"], predictions, prediction_rows
        ),
    )


def run_catalog_study(input_path):
    started = monotonic()
    objective, catalog = ControllerObjective(), context_catalog()
    report: dict[str, Any] = dict(
        definition="context_catalog_learning_v1",
        complete=False,
        source_sha256=live.HEAD_SOURCE_SHA,
        prior_training=dict(updates=12, rows=72, mistakes=16),
        objective=asdict(objective),
        catalog=asdict(catalog),
        monitoring=[],
        trials=[],
        application=None,
        final_examples=0,
        probe_updates=0,
        qualification="six known TRAIN context/arm trials then one original-source frozen development application; unequal added replay work, no heldout/generalization/CPC/efficiency claim",
    )
    trainer = None
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        try:
            raw = live._source_bytes(input_path)
            if sha256(raw).hexdigest() != live.HEAD_SOURCE_SHA:
                raise ValueError("original qualified contextual source replaced")
            initial = live.head_states(json.loads(raw))[0]
            source = ContextPerceptronLearner(initial.policy)
            source.restore_state(initial)
            actor = StableActor(source, version="qualified-12")
            data = prepare_data(DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG))
            data_pin, rng = deepcopy(data), deepcopy(np.random.get_state())
            clock = LogicalClock()
            trainer = CounterfactualController(
                actor.snapshot_state, snapshot_digest, objective, clock, catalog=catalog
            )
            rows = report["trials"]
            paid: list[Any] = []

            def check():
                if monotonic() - started > 240 or memory.snapshot().peak_bytes > 512 * 1024**2:
                    raise ValueError("context catalog original whole-case allowance exhausted")
                if (
                    live._source_bytes(input_path) != raw
                    or source.snapshot_state() != initial
                    or actor.snapshot_state().state != initial
                    or not _same_value(data, data_pin)
                    or not _same_value(np.random.get_state(), rng)
                ):
                    raise ValueError("context catalog original source/TRAIN/RNG changed")
                trainer._require_current()
                if report["trials"] is not rows or len(rows) != len(paid):
                    report["trials"] = rows
                    rows[:] = [r for r, _, _ in paid]
                    raise ValueError("preceding catalog trial container replaced")
                for i, (row, expected, guard) in enumerate(paid):
                    if rows[i] is not row:
                        rows[:] = [r for r, _, _ in paid]
                        raise ValueError("preceding catalog trial replaced")
                    try:
                        guard()
                    except ValueError:
                        row.clear()
                        row.update(deepcopy(expected))
                        raise ValueError("preceding catalog trial corrupted") from None

            for batch in data[0]:
                check()
                observed = clock.now()
                prediction = actor.predict(batch.features)
                clock.advance_to(observed + 1)
                feedback = ArrivedTrainingFeedback(
                    "train",
                    batch.context,
                    prediction.actor_version,
                    observed,
                    clock.now(),
                    int(np.sum(np.argmax(prediction.prediction, axis=1) != batch.targets[:, 0])),
                    len(batch.features),
                )
                report["monitoring"].append(asdict(feedback))
                require_observed_context(batch.context, feedback)
                for _ in catalog.arms:
                    check()
                    observation = ReferenceObservation(
                        clock.now(), 1, 1, 1, actor.version, feedback
                    )

                    def operation(request, state):
                        check()
                        if state != initial or snapshot_digest(state) != request.state_digest:
                            raise ValueError(
                                "catalog trial requires original complete contextual source"
                            )
                        result = live.run_case(
                            input_path, catalog_trial=request, catalog_base=state
                        )
                        row = dict(request=asdict(request), result=result)
                        rows.append(row)
                        guard, expected = paid_report_guard(row)
                        paid.append((row, expected, guard))
                        if result.get("complete") is not True:
                            raise ValueError("paid catalog trial incomplete; later work closed")
                        if result.get("role") != "controller_train" or not _same_value(
                            result["catalog_control"]["record"], asdict(request)
                        ):
                            raise ValueError("catalog trial result lost original TRAIN request")
                        measures, work = catalog_measures(result)
                        work.require_complete(catalog, request.action_id)
                        # Translate this case's original local tick1 arrival and
                        # fourteen-opportunity window into the fit owner's clock.
                        clock.advance_to(request.started_at + 14)
                        return CatalogCounterfactualReturn(
                            request, request.started_at + 1, clock.now(), measures, work
                        )

                    trainer.run(observation, operation)
            check()
            fitted = trainer.freeze()
            report["fitted_policy"] = asdict(fitted)
            check()
            result = live.run_case(input_path, catalog_policy=fitted)
            report["application"] = result
            if result.get("complete") is not True:
                raise ValueError("frozen catalog application incomplete")
            if result.get("role") != "frozen_application_development" or not _same_value(
                result["catalog_control"]["record"], asdict(fitted)
            ):
                raise ValueError("catalog application lost original frozen fit")
            measures, work = catalog_measures(result)
            report["application_measures"], report["application_work"] = (
                asdict(measures),
                asdict(work),
            )
            check()
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            if trainer is not None:
                try:
                    report["controller"] = controller_record(trainer.snapshot())
                except Exception as error:
                    report.update(
                        complete=False,
                        controller=None,
                        controller_output_error=f"{type(error).__name__}: {error}"[:512],
                    )
            report.update(elapsed_seconds=monotonic() - started, memory=asdict(memory.snapshot()))
    return report
