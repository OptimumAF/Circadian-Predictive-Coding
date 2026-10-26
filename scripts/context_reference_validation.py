"""Fixed full execution bindings through existing train/validation chronology.

Accepted development is reused as known metadata. One new joint stress stream
validates fixed references; no fitting, final callback/source or policy update.
Inner learners still receive only arrived ordinary TRAIN. The outer controller
is fixed. This trusted local composition is not a hostile-code sandbox.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts import context_head_live as live
from scripts.context_fifo_reference_study import case_outcome, HEAD_SOURCE_SHA, _source_bytes
from scripts.context_stream_comparison import _same
from scripts.context_training_streams import ContextTrainingRecipe
from src.app.controller_chronology import ControllerChronology
from src.app.probe_return_phase import paid_report_guard
from src.core.controller_objective import (
    ControllerMeasures,
    ControllerObjective,
    ControllerProtocol,
    ControllerStream,
    ControllerTrial,
    ControllerTrialMetadata,
)
from src.core.experience import LogicalClock
from src.core.reference_controller import ReferencePolicy
from src.shared.process_memory import ProcessRssSampler

MODES = ("no_learning", "periodic", "candidate_ack")
DEVELOPMENT_SHA = "3af3fa597f73b3f7ded8c40d08f291b9be47e622668a56592d34b913006dc994"
VALIDATION_RECIPE = ContextTrainingRecipe(
    "validation_d1_poison_a2b3a3",
    ("old-v1",) * 2 + ("new-v1",) * 3 + ("old-v1",) * 3,
    1,
    True,
)


@dataclass(frozen=True)
class ReferenceExecution:
    mode: str

    def __post_init__(self):
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError("execution requires one of the three fixed references")

    @property
    def policy(self):
        return ReferencePolicy("drift" if self.mode == "candidate_ack" else self.mode)

    def record(self):
        self.__post_init__()
        return dict(
            mode=self.mode,
            policy=asdict(self.policy),
            candidate_feedback=self.mode == "candidate_ack",
            acknowledge_correct=self.mode == "candidate_ack",
        )

    def kwargs(self):
        self.__post_init__()
        options = dict(reference_mode=self.policy.mode, stream_recipe=deepcopy(VALIDATION_RECIPE))
        if self.mode == "candidate_ack":
            options.update(candidate_feedback=True, acknowledge_correct=True)
        return options


def protocol(execution):
    return ControllerProtocol(
        f"reference_validation_{execution.mode}",
        ControllerObjective(),
        execution.policy,
        tuple(
            ControllerStream(f"{execution.mode}_{name}", role, group)
            for name, role, group in (
                ("development", "controller_train", "known_fifo_development_102"),
                ("stress", "controller_validation", "new_joint_stress_validation_v1"),
                ("reserved", "sealed_evaluation", "reserved_unavailable_final_v1"),
            )
        ),
    )


def _development_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2 or sha256(raw).hexdigest() != DEVELOPMENT_SHA:
        raise ValueError("accepted development102 replaced or exceeds bound")
    return raw


def _noise(case):
    expected = [i for i, c in enumerate(VALIDATION_RECIPE.contexts) if c == "old-v1"]
    stream = case["training_stream"]
    if (
        stream.get("role") != "train"
        or stream.get("label_delay") != 1
        or stream.get("contexts")
        != [f"{VALIDATION_RECIPE.recipe_id}:{c}" for c in VALIDATION_RECIPE.contexts]
        or stream.get("row_counts")
        != [3 if c == "new-v1" else 6 for c in VALIDATION_RECIPE.contexts]
    ):
        raise ValueError("validation TRAIN source or arrival schedule changed")
    rows = case["synthetic_label_noise"]
    if len(rows) != len(expected):
        raise ValueError("declared validation corruption count differs")
    for row, i in zip(rows, expected, strict=True):
        if (
            row.get("role") != "train"
            or row.get("kind") != "synthetic_label_corruption"
            or row.get("batch") != i
            or row.get("semantic_row") != "v1:invoice:ready"
            or row.get("clean_action") != "route"
            or row.get("corrupted_action") != "defer"
            or row.get("row_id") != f"{VALIDATION_RECIPE.recipe_id}:batch{i}:v1:invoice:ready"
        ):
            raise ValueError("declared named validation corruption changed")


def _measures(case, outcome, base):
    from scripts.run_bp_ticket_serving import _violation_count

    episodes = [r["episode"] for r in case["live_serving"]["records"]] + [
        r["episode"] for r in case["episodes"]
    ]
    return ControllerMeasures(
        base["new"],
        outcome["new"],
        2,
        base["old"],
        outcome["old"],
        4,
        case["elapsed_seconds"],
        case["memory"]["peak_bytes"],
        _violation_count(episodes),
        True,
    )


def _snapshot(chronology):
    s = chronology.snapshot()
    return dict(
        protocol=asdict(s["protocol"]),
        attempts=s["attempts"],
        rows=[asdict(row) for row in s["rows"]],
        frozen_at=s["frozen_at"],
        stopped=s["stopped"],
        clock=s["clock"],
    )


def run_study(input_path, development_path):
    started = monotonic()
    executions = tuple(ReferenceExecution(m) for m in MODES)
    declared = [e.record() for e in executions]
    recipe_pin = deepcopy(VALIDATION_RECIPE)
    owners = []
    for execution in executions:
        clock = LogicalClock()
        p = protocol(execution)
        owners.append((execution, clock, p, ControllerChronology(p, clock)))
    report: dict[str, Any] = dict(
        definition="fixed_reference_validation_v1",
        complete=False,
        executions=deepcopy(declared),
        validation_recipe=asdict(VALIDATION_RECIPE),
        objective=asdict(ControllerObjective()),
        development_sha256=DEVELOPMENT_SHA,
        development=[],
        validation_cases={},
        validation_scores=[],
        pending_modes=list(MODES),
        chronologies={},
        sealed_sources_opened=0,
        fitting_callbacks=0,
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        clock_scope="outer aggregate return events; original inner clocks remain in each result",
        qualification="one unseen joint delay/order/poisoning stream of known templates; no semantic-task generalization",
    )
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        try:
            raw = _source_bytes(input_path)
            if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
                raise ValueError("qualified learner source replaced")
            dev_raw = _development_bytes(development_path)
            development = json.loads(dev_raw)
            if development.get("complete") is not True or development.get("objective") != asdict(
                ControllerObjective()
            ):
                raise ValueError("complete original development objective required")
            cases = report["validation_cases"]
            guards: dict[str, Any] = {}

            def check():
                rss = memory.sample()
                if monotonic() - started > 600 or rss is None or rss > 512 * 1024**2:
                    raise ValueError("original validation study time/RSS budget exhausted")
                if (
                    [e.record() for e in executions] != declared
                    or VALIDATION_RECIPE != recipe_pin
                    or report["executions"] != declared
                    or not _same(report["validation_recipe"], asdict(recipe_pin))
                ):
                    raise ValueError("frozen full execution/recipe binding changed")
                if (
                    _source_bytes(input_path) != raw
                    or _development_bytes(development_path) != dev_raw
                ):
                    raise ValueError("qualified source or known development evidence changed")
                if report["validation_cases"] is not cases:
                    report["validation_cases"] = cases
                    raise ValueError("paid validation case container replaced")
                for mode, (owner, guard, expected) in guards.items():
                    if cases.get(mode) is not owner:
                        cases[mode] = owner
                        raise ValueError("preceding paid validation case replaced")
                    try:
                        guard()
                    except ValueError:
                        owner.clear()
                        owner.update(deepcopy(expected))
                        raise ValueError("preceding paid validation case corrupted") from None

            score_by_mode = {row["mode"]: row for row in development["scores"]}
            # These outcomes were known before declaration: reuse metadata at
            # outer0, never pretend they are newly arrived validation labels.
            for execution, clock, p, chronology in owners:
                check()
                case_outcome(development["cases"][execution.mode], execution.mode)

                def reuse(policy):
                    if policy != execution.policy:
                        raise ValueError("development policy differs")
                    clock.advance_to(1)
                    return ControllerTrial(
                        ControllerTrialMetadata(p.streams[0], policy, 0, 0, 1),
                        ControllerMeasures(**score_by_mode[execution.mode]["measures"]),
                    )

                row = chronology.run_development(p.streams[0].stream_id, reuse)
                report["development"].append(
                    dict(
                        mode=execution.mode,
                        reused=True,
                        known_before_declaration=True,
                        metadata=asdict(row.metadata),
                        utility=asdict(row.utility),
                    )
                )
                chronology.training_rows()  # Only the known train row is available here.

            baseline = None
            scores = []
            for execution, clock, p, chronology in owners:
                check()

                def validate(policy):
                    nonlocal baseline
                    if policy != execution.policy:
                        raise ValueError("validation policy differs")
                    check()
                    case = live.run_case(input_path, **execution.kwargs())
                    cases[execution.mode] = case  # Paid failed prefixes are retained.
                    check()
                    outcome = case_outcome(case, execution.mode, recipe=VALIDATION_RECIPE)
                    _noise(case)
                    guard, expected = paid_report_guard(case)
                    guards[execution.mode] = (case, guard, expected)
                    if baseline is None:
                        baseline = outcome
                    measures = _measures(case, outcome, baseline)
                    clock.advance_to(2)  # Aggregate labels/returns have now arrived.
                    return ControllerTrial(
                        ControllerTrialMetadata(p.streams[1], policy, 1, 2, 2), measures
                    )

                row = chronology.run_development(p.streams[1].stream_id, validate)
                receipt = chronology.freeze()  # Held locally; no final callback is supplied.
                report["chronologies"][execution.mode] = _snapshot(chronology)
                scores.append(
                    dict(
                        mode=execution.mode,
                        metadata=asdict(row.metadata),
                        utility=asdict(row.utility),
                        outcomes=case_outcome(
                            cases[execution.mode], execution.mode, recipe=VALIDATION_RECIPE
                        ),
                        freeze_record=asdict(receipt),
                        sealed_released=False,
                    )
                )
                report["pending_modes"].remove(execution.mode)
                check()
            report["validation_scores"] = scores
            report["complete"] = True
        except Exception as error:
            report["validation_scores"] = []
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            report["chronologies"] = {e.mode: _snapshot(c) for e, _, _, c in owners}
            report["elapsed_seconds"] = monotonic() - started
            report["study_memory"] = asdict(memory.snapshot())
    return report
