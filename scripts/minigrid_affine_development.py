"""Fixed read-only A4 development observation of a qualified ordinary main actor.

Consumes actual saved parameters once, not scalar reconstruction or TRAIN data.
Uses the existing evaluator/role gates and retains source costs as provenance,
never as new training authority. No fitting, selection, private promotion or IO
framework. The original installed source is fixed before observing new outcomes.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any

from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    evaluate_context_linear_stream_checkpoint,
)
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.process_memory import ProcessRssSampler

SOURCE_BYTES = 1_809_281
SOURCE_SHA = "f9d8ff7b14b562e73318f28d4d61898970b222c0ca0edb676b0cb2c35fdf89db"


def describe_development():
    return dict(
        definition="ordinary_affine_A4_development_v1",
        source_sha256=SOURCE_SHA,
        source_bytes=SOURCE_BYTES,
        source_fit_calls=10,
        source_fit_rows=252,
        source_joint_replay_rows=162,
        new_fit_calls=0,
        new_training_rows=0,
        evaluation="A branch4 blue then red; rotation0/unswapped/grey; seed17/max16/raw K7 firstargmax",
        environment_data_role="probe_test",
        evaluation_purpose="development",
        private_probe_training=False,
        max_environment_steps=32,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
        qualification="one fixed layout transfer check of ordinary main state; not broad reliability, matched compute, CPC or release",
    )


def _admit_record(record):
    from scripts.minigrid_affine_workflow import _require_training, describe_workflow
    from scripts.run_minigrid_linear_fit_control import _context_linear_training_snapshot

    if (
        type(record) is not dict
        or record.get("definition") != "portable_affine_recurring_train_v1"
        or record.get("status") != "complete"
        or record.get("reason") is not None
        or record.get("attempted_phases") != ["collection", "training", "games"]
    ):
        raise ValueError("complete ordinary recurring TRAIN workflow required")
    if json.dumps(record.get("protocol"), sort_keys=True) != json.dumps(
        describe_workflow(recurring_training=True), sort_keys=True
    ):
        raise ValueError("original ordinary recurring protocol differs")
    training, games = record["training"], record["games"]
    if (
        training.get("definition") != "minigrid_arrived_recurring_capacity3_train_stream_v1"
        or training.get("failure_work") is not None
        or training.get("stream", {}).get("evaluation_scope") != "training_only"
        or training.get("stream", {}).get("completed_updates") != 10
        or training.get("stream", {}).get("training_examples") != 252
        or [row["updates"] for row in training.get("endpoints", ())] != [0, 2, 4, 6, 8, 10]
        or games.get("status") != "complete"
        or games.get("state_unchanged") is not True
        or games.get("role") != "knownTRAIN_GA_return"
        or len(games.get("episodes", ())) != 4
        or any(
            e.get("status") != "complete" or e.get("environment", {}).get("closed") is not True
            for e in games["episodes"]
        )
    ):
        raise ValueError("original ordinary training and known-game completion required")
    for index, receipt in enumerate(training["committed_receipts"]):
        if (
            receipt["update_number"] != index + 1
            or receipt["observed_at"] != index
            or receipt["arrived_at"] != index + 1
            or receipt["applied_at"] != index + 1
        ):
            raise ValueError("original ordinary delayed-label receipt differs")
    state = _context_linear_training_snapshot(training["endpoints"][-1]["state"])
    state = _require_training(training, [state], recurring_training=True)
    if training["catalogue"]["committed_rows"] != dict(arrivals=90, replay=162, native=252):
        raise ValueError("original recurring exposure accounting differs")
    provenance = dict(
        source_sha256=SOURCE_SHA,
        source_bytes=SOURCE_BYTES,
        training_receipts=deepcopy(training["committed_receipts"]),
        training_budget=deepcopy(training["budget"]),
        catalogue=deepcopy(training["catalogue"]),
        original_workflow_budget=deepcopy(record["workflow_budget"]),
        original_known_games=4,
        original_known_game_role="train",
        training_authority_restored=False,
    )
    return state, provenance


def run_saved_development(path, *, check_boundary=lambda: None):
    if not callable(check_boundary):
        raise ValueError("development boundary must be callable")
    protocol = describe_development()
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        status="running",
        reason=None,
        protocol=protocol,
        source=None,
        games=None,
    )
    proof, original_protocol = paid_report_guard(protocol)
    source_proof: Any = None
    source_owner: Any = None
    limits = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    budget = ToyBudgetSession(limits, monotonic)
    expected_limits = replace(limits)
    owner: Any = None
    initial: Any = None

    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget.attach_memory(memory)
        # Freeze the observed sampler attachment, as in the existing workflow;
        # capturing before attachment would reject initialization as a change.
        binding = budget_binding(budget)

        def require_authority():
            proof()
            if report["protocol"] is not protocol:
                raise ValueError("original development protocol replaced")
            if source_proof is not None:
                if report["source"] is not source_owner:
                    raise ValueError("original development source provenance replaced")
                source_proof()
            if (
                budget_binding(budget) != binding
                or budget.budget is not limits
                or asdict(limits) != asdict(expected_limits)
                or type(budget.updates_completed) is not int
                or budget.updates_completed != 0
                or type(budget.replay_examples_completed) is not int
                or budget.replay_examples_completed != 0
                or (owner is not None and owner.snapshot_state() != initial)
            ):
                raise ValueError("original zero-update development authority changed")

        def guard():
            require_authority()
            budget.before_final()
            check_boundary()
            require_authority()

        try:
            guard()
            with Path(path).open("rb") as handle:
                raw = handle.read(SOURCE_BYTES + 1)
            if len(raw) != SOURCE_BYTES or sha256(raw).hexdigest() != SOURCE_SHA:
                raise ValueError("exact qualified127 ordinary main report required")
            # Why: admission consumes immutable as-read bytes once. The detached
            # typed model is the authority thereafter, not a changing live file.
            state, provenance = _admit_record(json.loads(raw))
            report["source"] = provenance
            source_proof, original_source = paid_report_guard(provenance)
            source_owner = provenance
            owner = ContextLinearLearner()
            initial = owner.snapshot_state()
            guard()
            report["games"] = evaluate_context_linear_stream_checkpoint(
                owner,
                state,
                budget=budget,
                check_boundary=guard,
                recurring_catalogue=True,
                development_a4=True,
            )
            guard()
            if report["source"] is not source_owner:
                raise ValueError("original development source provenance replaced")
            if (
                report["games"].get("status") != "complete"
                or report["games"].get("state_unchanged") is not True
            ):
                raise ValueError("development game capture incomplete")
            report["status"] = "complete"
        except MiniGridEvaluationFailed as error:
            report.update(status="failed", reason=str(error)[:512], games=error.report)
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        finally:
            report["protocol"] = original_protocol
            if source_proof is not None:
                report["source"] = original_source
            report["budget"] = dict(
                initial_limits=asdict(expected_limits),
                limits=asdict(limits),
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                started_at=binding[2],
                observed_budget_started_at=budget.started_at,
                ended_at=monotonic(),
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report


def handle_cli(args, parser):
    """Fixed option admission before file/model IO; original command remains owner."""
    allowed = {"describe", "source", "output", "context_linear_a4_development"}
    if any(
        value is not None and value is not False
        for key, value in vars(args).items()
        if key not in allowed
    ):
        parser.error("A4 development cannot combine with other modes or fit options")
    if args.describe:
        if args.source is not None or args.output is not None:
            parser.error("--describe accepts no source or output")
        print(json.dumps(describe_development(), indent=2, allow_nan=False))
        return
    if args.source is None:
        parser.error("A4 development requires --source")
    if args.output is not None and args.output.exists():
        raise FileExistsError("development refuses to replace existing output")
    report = run_saved_development(args.source)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if len(payload.encode("utf8")) > 4 * 1024**2:
        raise ValueError("development output exceeds4MiB; work must not retry")
    if args.output is not None:
        with args.output.open("x", encoding="utf8") as out:
            out.write(payload)
    print(payload)
    if report["status"] != "complete":
        raise SystemExit(2)
