"""Compose original delayed affine learning with fresh A4 TRAIN and fixed games.

Inputs are the exact fresh170 TRAIN report and bundled A3 TRAIN. Outputs retain
original costs/receipts and three fixed native readouts. No learner math, source
relabeling, private promotion or alternative stream lives in this module.
"""

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
from src.core.demonstration_episode import DemonstrationEpisode
from src.core.minigrid_public_history import encode_public_history_episode
from src.infra.minigrid_affine_example import example_affine_training
from src.infra.minigrid_memory import _require_minigrid
from src.shared.process_memory import ProcessRssSampler

SOURCE_BYTES = 428191
SOURCE_SHA = "bce498c34c23d9ef128f9344bf7164a70edc6afa156d42e7534bb708b2e3c26f"


def describe_adaptation(*, current_only=False):
    from scripts.run_minigrid_linear_fit_control import (
        describe_context_linear_a4,
        describe_context_linear_a4_current_only,
    )

    if type(current_only) is not bool:
        raise ValueError("current_only must be an exact bool")

    return dict(
        definition="minigrid_a3_a4_a3_current_only_and_games_v1"
        if current_only
        else "minigrid_a3_a4_a3_adaptation_and_games_v1",
        training=describe_context_linear_a4_current_only()
        if current_only
        else describe_context_linear_a4(),
        source_bytes=SOURCE_BYTES,
        source_sha256=SOURCE_SHA,
        native_plan="at2/4/6: A3blue/red thenA4blue/red TRAIN; seed17/max16/K7firstargmax",
        game_budget="one cumulative original150s512MiB zero-update/replay owner across all phases",
        interpretation="known TRAIN acquisition/retention/return; recovery requires loss/regain; no CPC/generalization/matchedcompute claim",
    )


def _admit_a4_record(record):
    from scripts.run_minigrid_linear_fit_control import _transition

    if (
        record.get("status") != "complete"
        or record.get("new_fits") != 0
        or record.get("new_replay") != 0
        or record.get("history_rows") != 18
        or len(record.get("collection", ())) != 2
        or len(record.get("episodes", ())) != 2
        or record["budget"]["updates"] != 0
        or record["budget"]["replay"] != 0
        or record["budget"]["started_at"] != record["budget"]["observed_started_at"]
        or record["counts"]["forbidden"] != 0
    ):
        raise ValueError("complete from-birth170 A4 TRAIN collection required")
    packets = []
    for native, saved, cue in zip(
        record["collection"], record["episodes"], ("blue", "red"), strict=True
    ):
        identity = f"minigrid-source-A4-train-{cue}"
        if (
            (native["episode_id"], native["context"], native["role"], native["cue"])
            != (identity, "A", "train", cue)
            or native["layout"]
            != dict(context="A", rotation=0, branch_row=4, swap_keys=False, wall_color="grey")
            or native["status"] != "complete"
            or native["closed"] is not True
            or native["terminated"] is not True
            or native["truncated"] is not False
            or native["failure_work_may_be_unobserved"] is not False
            or native["environment_steps"] != 9
            or native["environment_steps_attempted"] != 9
            or len(native["transitions"]) != 9
            or native["total_return"] <= 0
        ):
            raise ValueError("exact complete closed fresh A4 TRAIN native record required")
        transitions = []
        for step, row in enumerate(native["transitions"]):
            if row["applied"] is not True:
                raise ValueError("A4 TRAIN transition must have returned")
            transitions.append(
                _transition(
                    dict(
                        step=step,
                        observation=row["observation"],
                        action=row["action"],
                        reward=row["reward"],
                        next_observation=row["next_observation"],
                        terminal=row["terminated"],
                        num_classes=7,
                    )
                )
            )
        packet = encode_public_history_episode(
            DemonstrationEpisode(identity, "A", "train", tuple(transitions), 7)
        )
        if (
            json.loads(json.dumps(asdict(packet))) != saved
            or sum(t.reward for t in packet.transitions) != native["total_return"]
        ):
            raise ValueError("exported A4 TRAIN history differs from original native evidence")
        packets.append(packet)
    provenance = dict(
        source_bytes=SOURCE_BYTES,
        source_sha256=SOURCE_SHA,
        counts=record["counts"],
        budget=record["budget"],
        seconds=record["seconds"],
        sampled_peak_rss_bytes=record["sampled_peak_rss_bytes"],
        qualification="original18teachersteps/2env; no learnerfit; sunk collection cost, not new authority",
    )
    return tuple(packets), provenance


def read_a4_training(path):
    with Path(path).open("rb") as source:
        raw = source.read(SOURCE_BYTES + 1)
    if len(raw) != SOURCE_BYTES or sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("exact qualified170 from-birth A4 TRAIN report required")
    return _admit_a4_record(json.loads(raw))


def summarize_games(phases):
    """Fixed paired outcomes; an incomplete phase is never a zero-success task."""
    if [phase["updates"] for phase in phases] != [2, 4, 6]:
        return dict(
            complete=False,
            acquisition_gain=None,
            old_retention_delta=None,
            return_gain=None,
            learning_recovery=None,
        )
    outcomes = []
    for phase in phases:
        rows = phase["games"]["episodes"]
        if len(rows) != 4 or any(
            r["status"] != "complete" or r["environment"]["closed"] is not True for r in rows
        ):
            raise ValueError("paired A4 metrics require four complete closed episodes per phase")
        outcomes.append([r["success"] for r in rows])
    rates = [dict(old_A3=sum(row[:2]) / 2, new_A4=sum(row[2:]) / 2) for row in outcomes]
    return dict(
        complete=True,
        success_by_stage=rates,
        acquisition_gain=rates[1]["new_A4"] - rates[0]["new_A4"],
        old_retention_delta=rates[1]["old_A3"] - rates[0]["old_A3"],
        return_gain=rates[2]["old_A3"] - rates[1]["old_A3"],
        learning_recovery=[
            outcomes[0][i] and not outcomes[1][i] and outcomes[2][i] for i in range(2)
        ],
        qualification="two fixed cues/known TRAIN only; retention is not recovery; no matchedcompute/CPC/generalization",
    )


def _require_phase(observed):
    from src.core.minigrid_key_door import MiniGridKeyDoorLayout

    rows = observed.get("episodes", ())
    if (
        observed.get("status") != "complete"
        or observed.get("state_unchanged") is not True
        or len(rows) != 4
    ):
        raise ValueError("A4 phase capture incomplete")
    if len({row["environment"]["episode_id"] for row in rows}) != 4:
        raise ValueError("A3/A4 native episode identities must be distinct")
    for row, (branch, cue) in zip(
        rows, ((3, "blue"), (3, "red"), (4, "blue"), (4, "red")), strict=True
    ):
        environment = row["environment"]
        if (
            row["status"] != "complete"
            or row["source_role"] != "train"
            or row["cue"] != cue
            or environment["role"] != "train"
            or environment["cue"] != cue
            or environment["layout"] != asdict(MiniGridKeyDoorLayout("A", branch_row=branch))
            or environment["seed"] != 17
            or environment["max_steps"] != 16
            or environment["closed"] is not True
            or environment["failure_work_may_be_unobserved"] is not False
            or len(row["predictions"]) != environment["environment_steps"]
        ):
            raise ValueError("complete closed A3/A4 TRAIN phase evidence required")


def run_adaptation(path, *, current_only=False):
    from scripts.run_minigrid_linear_fit_control import (
        _context_linear_training_snapshot,
        read_training_sources,
        run_context_linear_stream,
    )

    protocol = describe_adaptation(current_only=current_only)
    protocol_check, original_protocol = paid_report_guard(protocol)
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        source=None,
        training=None,
        games=[],
    )
    proofs: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    source_check = training_check = None
    detached_source = detached_training = None
    limits = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(limits, monotonic)
        budget.attach_memory(memory)
        binding, expected_limits = budget_binding(budget), replace(limits)
        games = report["games"]

        def guard():
            protocol_check()
            if report["protocol"] is not protocol or report["games"] is not games:
                raise ValueError("original A4 protocol or phase list replaced")
            if budget_binding(budget) != binding or limits != expected_limits:
                raise ValueError("original cumulative A4 readout budget changed")
            if (budget.updates_completed, budget.replay_examples_completed) != (0, 0):
                raise ValueError("A4 readout is zero update/replay")
            budget.before_final()
            if budget_binding(budget) != binding:
                raise ValueError("A4 clock or memory binding changed")
            if source_check is not None:
                if report["source"] is not provenance:
                    raise ValueError("original A4 source provenance replaced")
                source_check()
            if training_check is not None:
                if report["training"] is not training:
                    raise ValueError("original A4 training evidence replaced")
                training_check()
            if len(games) != len(proofs) or len(games) > 3:
                raise ValueError("original paid A4 phase prefix changed")
            for current, (owner, check, _) in zip(games, proofs):
                if current is not owner:
                    raise ValueError("original paid A4 phase replaced")
                check()

        try:
            guard()
            _require_minigrid()
            fresh, provenance = read_a4_training(path)
            report["source"] = provenance
            source_check, detached_source = paid_report_guard(provenance)
            guard()
            with example_affine_training() as bundled:
                original = read_training_sources(bundled)[:2]
            training = run_context_linear_stream(
                original + fresh,
                **(
                    dict(a4_current_only=True)
                    if current_only
                    else dict(retain_arrived=True, a4_training=True)
                ),
                check_boundary=guard,
            )
            report["training"] = training
            training_check, detached_training = paid_report_guard(training)
            guard()
            costs_valid = (
                "catalogue" not in training
                and (training.get("stream") or {}).get("training_examples") == 54
                if current_only
                else (training.get("catalogue") or {}).get("committed_rows")
                == dict(arrivals=54, replay=36, native=90)
            )
            if (
                training["status"] != "complete"
                or training["source_unchanged"] is not True
                or training["actor_unchanged"] is not True
                or training["receipt_capture_error"] is not None
                or len(training["committed_receipts"]) != 6
                or training["budget"]["updates_completed"] != 6
                or not costs_valid
            ):
                raise ValueError("complete actual six-update A4 recipe/cost evidence required")
            states = [
                (row["updates"], _context_linear_training_snapshot(row["state"]))
                for row in training["endpoints"]
                if row["updates"] != 0
            ]
            if [n for n, _ in states] != [2, 4, 6]:
                raise ValueError("all original typed A4 stage parameters required")
            owner = ContextLinearLearner()
            for n, state in states:
                guard()
                try:
                    observed = evaluate_context_linear_stream_checkpoint(
                        owner,
                        state,
                        budget=budget,
                        check_boundary=guard,
                        **(dict(a4_current_only=True) if current_only else dict(a4_training=True)),
                    )
                except MiniGridEvaluationFailed as error:
                    report["failed_phase"] = dict(updates=n, games=error.report)
                    raise
                guard()
                try:
                    _require_phase(observed)
                except Exception:
                    report["failed_phase"] = dict(updates=n, games=observed)
                    raise
                phase = dict(updates=n, games=observed)
                check, detached = paid_report_guard(phase)
                games.append(phase)
                proofs.append((phase, check, detached))
            guard()
            report["status"] = "complete"
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        finally:
            try:
                guard()
            except Exception as error:
                report.update(
                    status="failed", capture_error=f"{type(error).__name__}: {error}"[:512]
                )
            report["protocol"] = original_protocol
            report["source"], report["training"] = detached_source, detached_training
            report["games"] = [detached for _, _, detached in proofs]
            report["summary"] = summarize_games(
                report["games"] if report["status"] == "complete" else ()
            )
            report["readout_budget"] = dict(
                initial_limits=asdict(expected_limits),
                limits=asdict(limits),
                started_at=binding[2],
                observed_started_at=budget.started_at,
                updates=budget.updates_completed,
                replay=budget.replay_examples_completed,
                ended_at=monotonic(),
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report


def handle_cli(args, parser):
    current_only = getattr(args, "context_linear_a4_current_only", False)
    allowed = {
        "describe",
        "source",
        "output",
        "context_linear_a4_adaptation",
        "context_linear_a4_current_only",
    }
    if any(v is not None and v is not False for k, v in vars(args).items() if k not in allowed):
        parser.error("A4 adaptation accepts no other modes or fit options")
    if args.describe:
        if args.source is not None or args.output is not None:
            parser.error("describe requires no paths")
        print(json.dumps(describe_adaptation(current_only=current_only), indent=2, allow_nan=False))
        return
    if args.source is None:
        parser.error("A4 adaptation requires fresh170 TRAIN --source")
    if args.output is not None and args.output.exists():
        raise FileExistsError("A4 adaptation output already exists")
    result = (
        run_adaptation(args.source, current_only=True)
        if current_only
        else run_adaptation(args.source)
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if len(payload.encode("utf8")) > 8 * 1024**2:
        raise ValueError("A4 adaptation exceeds8MiB; work must not retry")
    if args.output is not None:
        with args.output.open("x", encoding="utf8") as out:
            out.write(payload)
    print(payload)
    if result["status"] != "complete":
        raise SystemExit(2)
