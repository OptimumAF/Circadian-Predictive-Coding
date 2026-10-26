"""Declared A6 -> empty AdamW -> B6 intervention through existing inbox helpers.

Reuses accepted initial/A development readouts after exact state binding. Owns
one bounded run; opt-in rehearsal uses only prospectively authorized old TRAIN
cases. Does not tune, probe, save weights or test recovery.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts.run_open_jev_b_control import (
    _validate_readouts,
    load_reference,
    validate_initial_weights,
)
from scripts.run_open_jev_native_smoke import TEMPERATURE, _load
from scripts.run_open_jev_native_update import _weights, _weight_summary, record_readout
from scripts.run_open_jev_sequence import _deliver_phase, _difference, _group_metrics
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import LogicalClock
from src.core.native_choice_sequence import (
    INSTRUCTIONS,
    sequence_examples,
    validate_sequence_examples,
)

MAX_UPDATES = 12
RESET_REFERENCE_SHA256 = "04c2b8d5b387c5d0617a68e37516246a5560961b5f04a1bea7a44bb99db18b45"


def load_reset_reference(path: Path) -> dict[str, Any]:
    """Reuse the accepted reset-only arm; never refit or rerun its model."""
    with path.open("rb") as stream:
        body = stream.read(1024 * 1024 + 1)
    digest = hashlib.sha256(body).hexdigest()
    if len(body) > 1024 * 1024 or digest != RESET_REFERENCE_SHA256:
        raise ValueError("reset reference differs from the accepted reset-only control")
    source = json.loads(body)
    if (
        source.get("definition") != "native_optimizer_reset_control_v1"
        or source.get("complete") is not True
        or source.get("control_complete") is not True
        or source.get("app_committed_updates") != 12
        or source.get("examples")
        != json.loads(json.dumps([asdict(e) for e in sequence_examples()]))
    ):
        raise ValueError("reset reference is not the completed fixed control")
    _validate_readouts(source["readouts"], "after_optimizer_reset_B")
    return dict(source_sha256=digest, by_context=source["summary"]["by_context"])


def _projection_options(projection_rank, reset_reference, rehearsal, reset_optimizer):
    if projection_rank is None:
        if reset_reference is not None:
            raise ValueError("reset_reference requires a projection rank")
        return
    if type(projection_rank) is not int or not 0 <= projection_rank <= 8:
        raise ValueError("projection_rank must be an integer in 0..8")
    if rehearsal or not reset_optimizer:
        raise ValueError("projection requires reset authority and excludes rehearsal")
    if type(reset_reference) is not dict or set(reset_reference.get("by_context", {})) != {
        "A",
        "B",
        "anchor",
    }:
        raise ValueError("projection requires the admitted reset-only comparison")


def summarize_projection_control(readouts, reference, reset_reference) -> dict[str, Any]:
    _validate_readouts(readouts, "after_adapter_projection_B")
    rows = deepcopy(readouts)
    for row in rows:
        row["phase"] = "after_optimizer_reset_B"
    shared = summarize_optimizer_control(rows, reference)
    groups, reset = shared["by_context"], reset_reference["by_context"]
    return dict(
        by_context=groups,
        baseline_reused=shared["baseline_reused"],
        after_A_reused=shared["after_A_reused"],
        reset_only_reused=deepcopy(reset),
        warm_after_B=shared["warm_after_B"],
        contrasts=dict(
            projection_minus_reset={c: _difference(groups[c], reset[c]) for c in groups},
            projection_minus_warm_after_B=shared["contrasts"]["reset_minus_warm_after_B"],
            projection_minus_baseline=shared["contrasts"]["reset_minus_baseline"],
            B_acquisition=shared["contrasts"]["reset_B_acquisition"],
        ),
        criteria=shared["criteria"],
        recovery="not_tested",
        comparison_scope="historical_reset_only_same_TRAIN_and_reset_additional_projection_compute",
        rank_scope="allocated_rank8_boundary_projection_cap_only_regrowth_allowed",
    )


def _consolidation_options(rank, projection_rank, reset_reference, rehearsal, reset_optimizer):
    if rank is None:
        return
    if type(rank) is not int or not 0 <= rank <= 8:
        raise ValueError("consolidation_rank must be an integer in 0..8")
    if (
        projection_rank is not None
        or reset_reference is not None
        or rehearsal
        or not reset_optimizer
    ):
        raise ValueError("consolidation requires terminal reset authority and excludes other modes")


def summarize_consolidation_control(readouts, reference) -> dict[str, Any]:
    _validate_readouts(readouts, "after_adapter_consolidation_B")
    rows = deepcopy(readouts)
    for row in rows:
        row["phase"] = "after_optimizer_reset_B"
    shared = summarize_optimizer_control(rows, reference)
    return dict(
        by_context=shared["by_context"],
        baseline_reused=shared["baseline_reused"],
        after_A_reused=shared["after_A_reused"],
        warm_after_B=shared["warm_after_B"],
        contrasts=dict(
            consolidation_minus_warm_after_B=shared["contrasts"]["reset_minus_warm_after_B"],
            consolidation_minus_baseline=shared["contrasts"]["reset_minus_baseline"],
            B_acquisition=shared["contrasts"]["reset_B_acquisition"],
        ),
        criteria=shared["criteria"],
        recovery="not_tested",
        coefficients={"reference": 0.5, "current": 0.5},
        head_policy="preserve_current",
        comparison_scope="historical_warm_same_TRAIN_terminal_consolidation_reset_additional_compute",
        rank_scope="allocated_rank8_terminal_projection_no_followup_TRAIN",
        operator_history="current_already_contains_prior_history",
    )


def validate_after_A_weights(observed_summary, reference) -> None:
    _validate_phase_weights(observed_summary, reference["after_A_weights"], "after-A")


def validate_after_B_weights(observed_summary, reference) -> None:
    _validate_phase_weights(observed_summary, reference["after_B_weights"], "after-B")


def _validate_phase_weights(observed_summary, expected, phase):
    if not expected or observed_summary.keys() != expected.keys():
        raise ValueError(f"{phase} trainable parameter keys differ from accepted phase")
    for name, row in observed_summary.items():
        origin = expected[name]
        if any(row[key] != origin[key] for key in ("elements", "before_sha256", "after_sha256")):
            raise ValueError(f"{phase} trainable weights differ from accepted phase")


def summarize_optimizer_control(readouts, reference) -> dict[str, Any]:
    _validate_readouts(readouts, "after_optimizer_reset_B")
    groups = {
        c: _group_metrics([r for r in readouts if r["context"] == c]) for c in ("A", "B", "anchor")
    }
    baseline = reference["baseline_summary"]
    after_A, warm = reference["after_A_summary"], reference["sequential_summary"]
    anchor, original = groups["anchor"], baseline["anchor"]
    return dict(
        by_context=groups,
        baseline_reused=deepcopy(baseline),
        after_A_reused=deepcopy(after_A),
        warm_after_B=deepcopy(warm),
        contrasts=dict(
            reset_minus_baseline={c: _difference(groups[c], baseline[c]) for c in groups},
            reset_minus_warm_after_B={c: _difference(groups[c], warm[c]) for c in groups},
            reset_B_acquisition=_difference(groups["B"], after_A["B"]),
            warm_B_acquisition=deepcopy(reference["sequential_B_acquisition"]),
        ),
        criteria=dict(
            B_after_phase_success=groups["B"]["accuracy"] == 1,
            A_retained=after_A["A"]["accuracy"] == groups["A"]["accuracy"] == 1,
            anchors_preserved=(
                anchor["accuracy"] == 1
                and anchor["nll_status"] == original["nll_status"] == "finite"
                and anchor["nll_mean"] - original["nll_mean"] <= 0.1
            ),
        ),
        comparison_scope="same_TRAIN_exposures_different_optimizer_history_and_total_compute",
        recovery="not_tested",
    )


def summarize_rehearsal(readouts, reference, *, reset_optimizer: bool) -> dict[str, Any]:
    _validate_readouts(readouts, "after_rehearsal_B")
    projected = deepcopy(readouts)
    for row in projected:
        row["phase"] = "after_optimizer_reset_B"
    # Why: identical endpoints/guards must use the same metric definitions.
    shared = summarize_optimizer_control(projected, reference)
    contrasts = shared["contrasts"]
    return dict(
        by_context=shared["by_context"],
        baseline_reused=shared["baseline_reused"],
        after_A_reused=shared["after_A_reused"],
        continuous_no_replay_after_B=shared["warm_after_B"],
        contrasts=dict(
            endpoint_minus_baseline=contrasts["reset_minus_baseline"],
            endpoint_minus_warm_no_replay=contrasts["reset_minus_warm_after_B"],
            B_acquisition=contrasts["reset_B_acquisition"],
        ),
        criteria=shared["criteria"],
        optimizer_reset_enabled=reset_optimizer,
        comparison_scope="historical_no_replay_has_fewer_TRAIN_exposures_and_different_compute",
        recovery="not_tested",
    )


def run_optimizer_control(
    learner,
    predict,
    reference,
    report: dict[str, Any],
    progress,
    *,
    rehearsal: bool = False,
    reset_optimizer: bool = True,
    projection_rank: int | None = None,
    reset_reference=None,
    consolidation_rank: int | None = None,
) -> None:
    if type(rehearsal) is not bool or type(reset_optimizer) is not bool:
        raise ValueError("rehearsal and optimizer reset options must be boolean")
    if not rehearsal and not reset_optimizer:
        raise ValueError("continuous optimizer comparison requires rehearsal=True")
    _projection_options(projection_rank, reset_reference, rehearsal, reset_optimizer)
    _consolidation_options(
        consolidation_rank, projection_rank, reset_reference, rehearsal, reset_optimizer
    )
    max_updates = MAX_UPDATES + (6 if rehearsal else 0)
    examples = sequence_examples()
    validate_sequence_examples(examples)
    reference = deepcopy(reference)
    _validate_readouts(reference["baseline_readouts"], "baseline")
    if (
        set(learner.work)
        != {
            "training_attempts",
            "forward_calls",
            "backward_calls",
            "optimizer_steps",
            "committed_updates",
        }
        or any(type(v) is not int or v != 0 for v in learner.work.values())
        or learner.last_update is not None
        or learner.optimizer_reset_work
        != dict(optimizer_reset_attempts=0, optimizer_resets_completed=0)
    ):
        raise ValueError("optimizer control requires fresh zero-work native learner")
    if (projection_rank is not None or consolidation_rank is not None) and (
        any(type(v) is not int or v != 0 for v in learner.adapter_projection_work.values())
        or learner.last_adapter_projection is not None
    ):
        raise ValueError("projection control requires unspent projection history")
    if consolidation_rank is not None and (
        any(type(v) is not int or v != 0 for v in learner.adapter_consolidation_work.values())
        or learner.last_adapter_reference is not None
        or learner._adapter_reference is not None
    ):
        raise ValueError("consolidation requires fresh reference and consolidation history")
    report.update(
        reference_sha256=reference["source_sha256"],
        reused_baseline=reference["baseline_readouts"],
        reused_after_A=reference["after_A_summary"],
        examples=[asdict(e) for e in examples],
        readouts=[],
        delivery=[],
        control_complete=False,
    )
    if rehearsal:
        report.update(
            rehearsal=True,
            optimizer_reset_enabled=reset_optimizer,
            rehearsal_retained_source_examples=2,
        )
    if projection_rank is not None:
        report.update(
            projection_rank=projection_rank, reset_reference_sha256=reset_reference["source_sha256"]
        )
    if consolidation_rank is not None:
        report["consolidation_rank"] = consolidation_rank
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=max_updates, max_wall_seconds=600), monotonic
    )
    inbox = ExperienceInbox(
        learner,
        clock=clock,
        budget=budget,
        learner_version="native-rank8",
        max_experiences=max_updates,
    )
    try:
        _deliver_phase(
            learner,
            inbox,
            clock,
            examples,
            "after_A",
            "A",
            report,
            progress,
            replay_permission=rehearsal,
        )
        if consolidation_rank is not None:
            progress("before_adapter_reference")
            try:
                report["reference_receipt"] = learner.capture_adapter_reference()
            finally:
                report["adapter_consolidation_work"] = learner.adapter_consolidation_work
            progress("adapter_reference_finished")
        elif projection_rank is not None:
            progress("before_adapter_projection")
            try:
                report["projection_receipt"] = learner.project_adapters(projection_rank)
            except Exception as error:
                report["projection_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                report["adapter_projection_work"] = learner.adapter_projection_work
                report["optimizer_reset_work"] = learner.optimizer_reset_work
            report["reset_receipt"] = learner.optimizer_reset_work
            progress("adapter_projection_finished")
        elif reset_optimizer:
            progress("before_optimizer_reset")
            report["reset_receipt"] = learner.reset_optimizer_state()
            progress("optimizer_reset_finished")
        else:
            progress("before_optimizer_reset")
            progress("optimizer_reset_skipped")
        retained = (
            tuple(e for e in examples if e.role == "train" and e.context == "A")
            if rehearsal
            else ()
        )
        _deliver_phase(
            learner,
            inbox,
            clock,
            examples,
            "after_B_consolidation"
            if consolidation_rank is not None
            else "after_B_projection"
            if projection_rank is not None
            else ("after_B_rehearsal" if rehearsal else "after_B_reset"),
            "B",
            report,
            progress,
            rehearsal_examples=retained,
        )
        if budget.updates_completed != max_updates:
            raise RuntimeError(f"control did not complete exactly {max_updates} TRAIN updates")
        if consolidation_rank is not None:
            progress("before_adapter_consolidation")
            try:
                report["consolidation_receipt"] = learner.consolidate_adapters(consolidation_rank)
            finally:
                report["adapter_consolidation_work"] = learner.adapter_consolidation_work
                report["adapter_projection_work"] = learner.adapter_projection_work
                report["optimizer_reset_work"] = learner.optimizer_reset_work
            report["reset_receipt"] = learner.optimizer_reset_work
            progress("adapter_consolidation_finished")
            if (
                any(
                    learner.adapter_consolidation_work[k] != 1
                    for k in (
                        "reference_attempts",
                        "references_completed",
                        "consolidation_attempts",
                        "consolidations_completed",
                    )
                )
                or learner.adapter_projection_work["projection_attempts"] != 1
                or learner.adapter_projection_work["projections_completed"] != 1
                or learner.optimizer_reset_work
                != dict(optimizer_reset_attempts=1, optimizer_resets_completed=1)
                or report["consolidation_receipt"]["effective_rank"] != consolidation_rank
                or learner._adapter_reference is not None
            ):
                raise ValueError(
                    "consolidation differs from the declared single terminal intervention"
                )
        if projection_rank is not None and (
            learner.adapter_projection_work["projection_attempts"] != 1
            or learner.adapter_projection_work["projections_completed"] != 1
            or learner.optimizer_reset_work
            != dict(optimizer_reset_attempts=1, optimizer_resets_completed=1)
            or report["projection_receipt"]["effective_rank"] != projection_rank
        ):
            raise ValueError("projection/reset work differs from the declared single intervention")
        for example in (e for e in examples if e.role == "outer_selection"):
            row = record_readout(
                example.request,
                predict,
                report["readouts"],
                progress,
                phase="after_adapter_consolidation_B"
                if consolidation_rank is not None
                else "after_adapter_projection_B"
                if projection_rank is not None
                else ("after_rehearsal_B" if rehearsal else "after_optimizer_reset_B"),
                context=example.context,
                target=example.target_id,
            )
            row.update(example_id=example.example_id, source_group=example.source_group)
        report["summary"] = (
            summarize_consolidation_control(report["readouts"], reference)
            if consolidation_rank is not None
            else summarize_projection_control(report["readouts"], reference, reset_reference)
            if projection_rank is not None
            else summarize_rehearsal(report["readouts"], reference, reset_optimizer=reset_optimizer)
            if rehearsal
            else summarize_optimizer_control(report["readouts"], reference)
        )
        report["control_complete"] = True
        progress("control_validated")
    finally:
        report["native_work"] = learner.work
        report["optimizer_reset_work"] = learner.optimizer_reset_work
        if projection_rank is not None or consolidation_rank is not None:
            report["adapter_projection_work"] = learner.adapter_projection_work
        if consolidation_rank is not None:
            report["adapter_consolidation_work"] = learner.adapter_consolidation_work
        report["app_committed_updates"] = budget.updates_completed
        report["inbox_stopped"] = inbox.stopped
        if rehearsal:
            replayed = [r for r in report["delivery"] if r.get("exposure_kind") == "replay"]
            report["replay_exposures_recorded"] = len(replayed)
            report["replay_examples"] = sum("receipt" in row for row in replayed)


def _boundary_state(learner) -> dict[str, Any]:
    """Bounded reset evidence, without copying the frozen base or retaining states."""
    gradients = hashlib.sha256()
    for name, parameter in learner._trainable.items():
        gradients.update(name.encode())
        gradients.update(b"none" if parameter.grad is None else parameter.grad.numpy().tobytes())
    return dict(
        gradient_sha256=gradients.hexdigest(),
        private_rng_sha256=hashlib.sha256(learner._rng.numpy().tobytes()).hexdigest(),
        caller_rng_sha256=hashlib.sha256(
            learner._torch.get_rng_state().numpy().tobytes()
        ).hexdigest(),
        model_training=learner.model.training,
        native_work=learner.work,
    )


def _head_gradient_hash(learner):
    digest = hashlib.sha256()
    for name, parameter in learner._trainable.items():
        if name.startswith("head."):
            digest.update(name.encode())
            digest.update(b"none" if parameter.grad is None else parameter.grad.numpy().tobytes())
    return digest.hexdigest()


def _validate_projected_state(
    learner, observed, state, boundary, bound_weights, head_gradient_hash
):
    if state.keys() != boundary.keys() or any(
        state[k] != boundary[k] for k in state if k != "gradient_sha256"
    ):
        raise ValueError("projection changed protected RNG/mode/native work")
    if _head_gradient_hash(learner) != head_gradient_hash or any(
        p.grad is not None for n, p in learner._trainable.items() if not n.startswith("head.")
    ):
        raise ValueError("projection gradient changes differ from the declared policy")
    if observed.keys() != bound_weights.keys() or any(
        observed[n] != bound_weights[n] for n in observed if n.startswith("head.")
    ):
        raise ValueError("projection changed the bound scalar head")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("reference", "source", "checkpoint", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument(
        "--rehearsal", action="store_true", help="replay two authorized old A TRAIN cases"
    )
    parser.add_argument("--optimizer-reset", type=int, choices=(0, 1), default=1)
    parser.add_argument("--projection-rank", type=int, choices=range(9))
    parser.add_argument("--consolidation-rank", type=int, choices=range(9))
    parser.add_argument("--reset-reference", type=Path)
    parser.add_argument(
        "--update-health",
        action="store_true",
        help="record TRAIN gradient/realized-step groups and diagnostic costs",
    )
    args = parser.parse_args(argv)
    if args.consolidation_rank is not None and (
        args.projection_rank is not None
        or args.reset_reference is not None
        or args.rehearsal
        or args.optimizer_reset != 1
        or args.update_health
    ):
        parser.error("consolidation requires terminal reset authority and excludes other modes")
    if (args.projection_rank is not None) != (args.reset_reference is not None):
        parser.error("--projection-rank and --reset-reference are required together")
    if args.projection_rank is not None and (
        args.rehearsal or args.optimizer_reset != 1 or args.update_health
    ):
        parser.error("projection requires one reset and excludes rehearsal/update-health")
    if args.optimizer_reset == 0 and not args.rehearsal:
        parser.error("--optimizer-reset 0 requires --rehearsal")
    if args.output.exists():
        parser.error("output exists; preserve previous evidence")
    report: dict[str, Any] = dict(
        definition="native_adapter_consolidation_control_v1"
        if args.consolidation_rank is not None
        else "native_adapter_projection_control_v1"
        if args.projection_rank is not None
        else (
            "native_choice_rehearsal_v1" if args.rehearsal else "native_optimizer_reset_control_v1"
        ),
        complete=False,
        model_load_attempts=0,
        events=[],
        seed=17,
        reused_baseline_prediction_calls=0,
        reused_after_A_prediction_calls=0,
        training_temperature=1,
        inference_temperature=TEMPERATURE,
        calibration_status="inherited_unrecalibrated",
        training_random_draws="unknown",
        checkpoint_recomputation_work="unmeasured_internal_layers",
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        checkpoint_fingerprint_scope="original_load_birth_provenance",
    )
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    owner: dict[str, Any] = {}
    if args.update_health:
        report["update_health_enabled"] = True
    started = monotonic()
    max_updates = MAX_UPDATES + (6 if args.rehearsal else 0)

    def progress(event):
        import resource

        event = dict(event=event) if type(event) is str else event
        event["outer_seconds"] = monotonic() - started
        event["outer_peak_rss_bytes"] = (
            getattr(resource, "getrusage")(getattr(resource, "RUSAGE_SELF")).ru_maxrss * 1024
        )
        report["events"].append(event)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        if event["outer_seconds"] > 600 or event["outer_peak_rss_bytes"] > 20 * 1024**3:
            raise RuntimeError("optimizer control resource allowance exceeded")
        if event["event"] in (
            "before_optimizer_reset",
            "optimizer_reset_finished",
            "before_adapter_projection",
            "adapter_projection_finished",
            "before_adapter_reference",
            "adapter_reference_finished",
            "before_adapter_consolidation",
            "adapter_consolidation_finished",
        ):
            learner = owner["learner"]
            learner._guard()
            observed = _weight_summary(learner.model, owner["before"])
            state = _boundary_state(learner)
            if event["event"] in ("before_adapter_reference", "adapter_reference_finished"):
                validate_after_A_weights(observed, owner["reference"])
                if event["event"] == "before_adapter_reference":
                    report["after_A_binding"] = dict(weights=observed, state=state)
                elif (
                    state != report["after_A_binding"]["state"]
                    or observed != report["after_A_binding"]["weights"]
                ):
                    raise ValueError("reference capture changed the bound after-A state")
                else:
                    report["reference_preservation"] = state
                return
            if event["event"] == "before_adapter_consolidation":
                validate_after_B_weights(observed, owner["reference"])
                report["after_B_binding"] = dict(weights=observed, state=state)
                owner["consolidation_head_gradient_hash"] = _head_gradient_hash(learner)
                return
            if event["event"] == "adapter_consolidation_finished":
                bound = report["after_B_binding"]
                _validate_projected_state(
                    learner,
                    observed,
                    state,
                    bound["state"],
                    bound["weights"],
                    owner["consolidation_head_gradient_hash"],
                )
                report["consolidation_preservation"] = dict(
                    state=state,
                    weights=observed,
                    head_gradient_sha256=owner["consolidation_head_gradient_hash"],
                )
                return
            if event["event"] != "adapter_projection_finished":
                validate_after_A_weights(observed, owner["reference"])
            if event["event"] in ("before_optimizer_reset", "before_adapter_projection"):
                owner["reset_boundary"] = state
                report["after_A_binding"] = dict(weights=observed, state=state)
                if args.projection_rank is not None:
                    owner["head_gradient_hash"] = _head_gradient_hash(learner)
            elif event["event"] == "adapter_projection_finished":
                _validate_projected_state(
                    learner,
                    observed,
                    state,
                    owner["reset_boundary"],
                    report["after_A_binding"]["weights"],
                    owner["head_gradient_hash"],
                )
                report["projection_preservation"] = dict(
                    state=state, weights=observed, head_gradient_sha256=owner["head_gradient_hash"]
                )
            else:
                if state != owner["reset_boundary"]:
                    raise ValueError("reset changed the bound after-A state")
                report["reset_preservation"] = state

    def observe_model(model):
        api = importlib.import_module("jev.api")
        learner = OpenJevTrainingLearner(
            model,
            api.compile_request,
            instructions=INSTRUCTIONS,
            policy=OpenJevTrainingPolicy(
                max_updates,
                8,
                args.optimizer_reset,
                record_update_health=args.update_health,
                max_adapter_projections=int(
                    args.projection_rank is not None or args.consolidation_rank is not None
                ),
                max_adapter_consolidations=int(args.consolidation_rank is not None),
            ),
        )
        owner["learner"], owner["before"] = learner, _weights(model)
        observed = _weight_summary(model, owner["before"])
        validate_initial_weights(observed, owner["reference"])
        rng = hashlib.sha256(learner._rng.numpy().tobytes()).hexdigest()
        if (
            model.training
            or learner._optimizer.state_dict()["state"]
            or rng != owner["reference"]["initial_rng_sha256"]
        ):
            raise ValueError("optimizer/mode/RNG differs from accepted initial state")
        report["initial_binding"] = dict(
            weights=observed, optimizer_empty=True, model_training=False, private_rng_sha256=rng
        )
        progress("initial_state_bound")

    try:
        owner["reference"] = load_reference(
            args.reference,
            include_after_A=True,
            **({"include_after_B": True} if args.consolidation_rank is not None else {}),
        )
        if args.projection_rank is not None:
            owner["reset_reference"] = load_reset_reference(args.reset_reference)
        validate_sequence_examples(sequence_examples())
        report["reference_sha256"] = owner["reference"]["source_sha256"]
        report["model_load_attempts"] = 1
        predict = _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=6,
            forward_limit=36 if args.rehearsal else 30,
            token_limit=18432 if args.rehearsal else 13824,
            model_observer=observe_model,
            **(
                {"work_reporter": lambda f: owner.update(observe_work=f)}
                if args.projection_rank is not None or args.consolidation_rank is not None
                else {}
            ),
        )
        run_optimizer_control(
            owner["learner"],
            predict,
            owner["reference"],
            report,
            progress,
            rehearsal=args.rehearsal,
            reset_optimizer=bool(args.optimizer_reset),
            projection_rank=args.projection_rank,
            reset_reference=owner.get("reset_reference"),
            consolidation_rank=args.consolidation_rank,
        )
        learner = owner["learner"]
        learner._guard()
        report["final_weights"] = _weight_summary(learner.model, owner["before"])
        report["model_training_after"] = learner.model.training
        if args.projection_rank is not None or args.consolidation_rank is not None:
            owner["observe_work"]()
            counts = next(
                e["observed_work"] for e in reversed(report["events"]) if "observed_work" in e
            )
            lengths = counts["candidate_sequence_lengths"]
            if (
                counts["decision_forward_calls"] != 30
                or counts["scorer_calls"] != 18
                or counts["native_prediction_calls_attempted"] != 6
                or counts["native_prediction_calls_returned"] != 6
                or len(lengths) != 54
                or sum(lengths) > 13824
                or max(lengths) > 256
                or counts["initializer_cpu_elements"] > 1
                or counts["initializer_meta_elements"] > 511147008
            ):
                raise ValueError("projection stream observed work differs from the fixed case")
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if "learner" in owner:
            report["native_work"] = owner["learner"].work
            report["optimizer_reset_work"] = owner["learner"].optimizer_reset_work
    finally:
        sys.setprofile(None)
        if (
            args.projection_rank is not None or args.consolidation_rank is not None
        ) and "learner" in owner:
            report["adapter_projection_work"] = owner["learner"].adapter_projection_work
        if args.consolidation_rank is not None and "learner" in owner:
            report["adapter_consolidation_work"] = owner["learner"].adapter_consolidation_work
        if args.update_health and "learner" in owner:
            report["update_health_work"] = owner["learner"].update_health_work
        report["outer_seconds"] = monotonic() - started
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
