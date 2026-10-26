"""Fixed observed C task on a disposable native warm or cold fork.

Reuses existing native owner, probe-role inbox, clocks and scalar metric helpers.
No final data, promotion, replay, reset, tuning, early stop or model saving.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from math import log
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from scripts.run_open_jev_b_control import load_reference, validate_initial_weights
from scripts.run_open_jev_native_smoke import _load
from scripts.run_open_jev_native_update import _weights, _weight_summary
from scripts.run_open_jev_optimizer_control import validate_after_A_weights
from scripts.run_open_jev_sequence import _deliver_phase, _difference, _group_metrics
from src.adapters.open_jev_owned_fork import _admit_buffers
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.native_choice_probe import PROBE_UPDATES, probe_examples, validate_probe_examples
from src.core.native_choice_sequence import INSTRUCTIONS, sequence_examples
from src.core.typed_decision import ProbabilityDecisionResult, validate_probability_decision_result


WARM_PROBE_SHA256 = "e1a701ed307d32ef4f6b3cfade9e96e357b937af9ab03d585a66243186e0e3ac"


def _source_updates(source_history):
    if type(source_history) is not str or source_history not in ("after_A6", "cold"):
        raise ValueError("source_history must be exactly after_A6 or cold")
    return 6 if source_history == "after_A6" else 0


def load_warm_probe(path: Path):
    """Reuse one independently accepted observed-task result, never rerun it."""
    with path.open("rb") as stream:
        body = stream.read(1024 * 1024 + 1)
    digest = hashlib.sha256(body).hexdigest()
    if len(body) > 1024 * 1024 or digest != WARM_PROBE_SHA256:
        raise ValueError("warm probe differs from the accepted fixed C study")
    observed = json.loads(body)
    if (
        observed["definition"] != "native_reserved_text_probe_v1"
        or not all(observed[k] is True for k in ("complete", "probe_complete", "source_preserved"))
        or observed["prediction_interface"] != "native_owner_candidate_batched_probability_v1"
        or observed["examples"] != json.loads(json.dumps([asdict(e) for e in probe_examples()]))
    ):
        raise ValueError("warm probe does not bind the accepted corpus/interface")
    return dict(source_sha256=digest, summary=observed["summary"])


def compare_probe_summaries(cold, warm):
    """Fixed cold-minus-warm endpoints; prior history and total compute differ."""
    return dict(
        direction="cold_minus_warm",
        by_phase={
            phase: {
                context: _difference(
                    cold["by_phase"][phase][context], warm["by_phase"][phase][context]
                )
                for context in ("C", "A", "anchor")
            }
            for phase in ("before", "after")
        },
        qualification="same_task_dose_and_readout_path_different_history_and_total_compute",
        A_interpretation="cold_untrained_transfer_vs_warm_retention",
        causal_scope="history_association_not_weights_vs_optimizer_causality",
    )


def summarize_probe(readouts, *, source_history="after_A6"):
    """Score fixed endpoints after work; zero probability has infinite NLL."""
    source_updates = _source_updates(source_history)
    expected = {e.example_id: e for e in probe_examples() if e.role != "probe_train"}
    if type(readouts) is not list or len(readouts) != 12:
        raise ValueError("reserved probe requires twelve declared readouts")
    by_phase = {}
    for phase in ("before", "after"):
        rows = [r for r in readouts if r["phase"] == phase]
        if len(rows) != 6 or {r["example_id"] for r in rows} != expected.keys():
            raise ValueError("reserved probe requires each readout exactly once per phase")
        for row in rows:
            example = expected[row["example_id"]]
            if any(row[k] != getattr(example, k) for k in ("context", "role", "source_group")):
                raise ValueError("reserved readout role/context/group changed")
            if row["request"] != asdict(example.request):
                raise ValueError("reserved readout request changed")
            result = row["result"]
            # In-memory rows retain tuples, not a lossy serialized reconstruction.
            result = ProbabilityDecisionResult(**result)
            validate_probability_decision_result(example.request, result)
            probability = dict(result.probabilities)[example.target_id]
            row.update(
                target=example.target_id,
                correct=result.selected_id == example.target_id,
                nll=None if probability == 0 else -log(probability),
                nll_status="infinite" if probability == 0 else "finite",
            )
        by_phase[phase] = {
            c: _group_metrics([r for r in rows if r["context"] == c]) for c in ("C", "A", "anchor")
        }
    before, after = by_phase["before"], by_phase["after"]
    anchor_finite = before["anchor"]["nll_status"] == after["anchor"]["nll_status"] == "finite"
    summary: dict[str, Any] = dict(
        by_phase=by_phase,
        contrasts={c: _difference(after[c], before[c]) for c in before},
        criteria=dict(
            C_endpoint_success=after["C"]["accuracy"] == 1,
            C_acquired=before["C"]["accuracy"] < 1 and after["C"]["accuracy"] == 1,
            A_retained=before["A"]["accuracy"] == after["A"]["accuracy"] == 1,
            anchors_preserved=before["anchor"]["accuracy"] == after["anchor"]["accuracy"] == 1
            and anchor_finite
            and after["anchor"]["nll_mean"] - before["anchor"]["nll_mean"] <= 0.1,
        ),
        recovery="not_tested",
        scope="one_fixed_warm_native_reserved_context",
    )
    if source_updates == 0:
        summary["criteria"].update(
            A_retained=None, A_transfer_endpoint_success=after["A"]["accuracy"] == 1
        )
        summary["scope"] = "one_fixed_cold_native_observed_context"
    return summary


def _readout(child, example, phase, report, progress):
    row = dict(
        phase=phase,
        example_id=example.example_id,
        context=example.context,
        role=example.role,
        source_group=example.source_group,
        request=asdict(example.request),
    )
    report["readouts"].append(row)
    report["prediction_work"]["attempted"] += 1
    progress("probe_prediction_attempt")
    result = child.predict(example.request)  # No target/label is sent to the owner.
    report["prediction_work"]["returned"] += 1
    row["result"] = asdict(result)
    progress("probe_prediction_return")
    validate_probability_decision_result(example.request, result)
    report["prediction_work"]["validated"] += 1
    progress("probe_prediction_validated")


def _probe_exposure(child, inbox, clock, example, number, report, progress):
    identity = f"probe-{number}-{example.example_id}"
    observed = clock.now()
    inbox.record_experience(
        Experience(
            identity,
            "native-fresh-probe",
            observed,
            "native-probe-copy",
            example.request,
            "probe_train",
            ExperiencePermissions(training=True),
        )
    )
    if inbox.drain(max_updates=1):
        raise RuntimeError("unlabeled probe unexpectedly trained")
    row = dict(
        exposure_id=identity,
        source_example_id=example.example_id,
        source_group=example.source_group,
        role="probe_train",
        observed_at=observed,
        prelabel_updates=0,
        permissions=asdict(ExperiencePermissions(training=True)),
    )
    report["probe_delivery"].append(row)
    clock.advance_to(observed + 1)
    inbox.record_label(
        LabelArrival(
            f"label-{identity}",
            identity,
            "native-fresh-probe",
            clock.now(),
            "native-probe-copy",
            example.target_id,
            role="probe_train",
        )
    )
    progress("probe_training_attempt")
    try:
        applied = inbox.drain(max_updates=1)
        if len(applied) != 1:
            raise RuntimeError("probe requires exactly one delayed-label update")
        row.update(receipt=asdict(applied[0]), training=deepcopy(child.last_update))
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["child_native_work"] = child.work
        report["probe_inbox_stopped"] = inbox.stopped
        progress("probe_training_finished")
    clock.advance_to(clock.now() + 1)


def run_probe_updates(child, report, progress, *, source_history="after_A6"):
    source_updates = _source_updates(source_history)
    examples = probe_examples()
    validate_probe_examples(examples)
    if set(child.work) != {
        "training_attempts",
        "forward_calls",
        "backward_calls",
        "optimizer_steps",
        "committed_updates",
    } or any(type(v) is not int or v != source_updates for v in child.work.values()):
        raise ValueError("reserved child must inherit exactly the declared source baseline")
    report.update(
        probe_delivery=[],
        readouts=[],
        probe_complete=False,
        prediction_work=dict(attempted=0, returned=0, validated=0),
    )
    birth = child.work
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(PROBE_UPDATES, 600), monotonic)
    inbox = ExperienceInbox(
        child,
        clock=clock,
        budget=budget,
        learner_version="native-probe-copy",
        max_experiences=PROBE_UPDATES,
        training_role="probe_train",
    )
    readouts = tuple(e for e in examples if e.role != "probe_train")
    training = tuple(e for e in examples if e.role == "probe_train")
    try:
        for example in readouts:
            _readout(child, example, "before", report, progress)
        for number in range(PROBE_UPDATES):
            _probe_exposure(child, inbox, clock, training[number % 2], number, report, progress)
        for example in readouts:
            _readout(child, example, "after", report, progress)
        report["summary"] = summarize_probe(report["readouts"], source_history=source_history)
        if budget.updates_completed != 6 or report["prediction_work"] != dict(
            attempted=12, returned=12, validated=12
        ):
            raise ValueError("probe work differs from the fixed protocol")
        report["probe_complete"] = True
    finally:
        report["child_native_work"] = child.work
        report["child_incremental_work"] = {k: v - birth[k] for k, v in child.work.items()}
        report["probe_app_committed_updates"] = budget.updates_completed
        report["probe_inbox_stopped"] = inbox.stopped


def _capture_source(source, report):
    source._guard()
    buffers = _admit_buffers(source)
    elements = sum(
        p.numel() + (0 if p.grad is None else p.grad.numel()) for p in source._trainable.values()
    )
    elements += sum(v.numel() for s in source._optimizer.state.values() for v in s.values())
    elements += sum(b.numel() for b in buffers.values()) + 2 * source._rng.numel()
    # The fork copies the same registered state plus one further RNG read.
    # Refuse before either snapshot allocation or the paid fork attempt.
    if elements + 2 * source._rng.numel() > 20_000_000:
        raise ValueError("source preservation/fork copy allowance exceeded")
    report["fork_copy_elements_prospective"] = elements + source._rng.numel()
    report["source_preservation_copy_elements_attempted"] = elements
    return dict(
        snapshot=source.snapshot_state(),
        buffers={n: b.detach().clone() for n, b in buffers.items()},
        modes=tuple((n, m.training) for n, m in source.model.named_modules()),
        work=source.work,
        reset=source.optimizer_reset_work,
        health=source.update_health_work,
        last_update=deepcopy(source.last_update),
        caller_rng=source._torch.get_rng_state(),
    )


def _same_tree(torch, current, expected, report):
    if torch.is_tensor(expected):
        report["source_preservation_comparison_elements_attempted"] += expected.numel()
        if report["source_preservation_comparison_elements_attempted"] > 40_000_000:
            raise ValueError("source preservation comparison allowance exceeded")
        return (
            torch.is_tensor(current)
            and current.dtype == expected.dtype
            and current.device == expected.device
            and current.shape == expected.shape
            and torch.equal(current, expected)
        )
    if type(current) is not type(expected):
        return False
    if type(expected) is dict:
        return current.keys() == expected.keys() and all(
            _same_tree(torch, current[k], v, report) for k, v in expected.items()
        )
    if type(expected) in (tuple, list):
        return len(current) == len(expected) and all(
            _same_tree(torch, a, b, report) for a, b in zip(current, expected)
        )
    return current == expected


def _assert_source_unchanged(source, birth, report):
    source._guard()
    saved = birth["snapshot"]
    report["source_preservation_copy_elements_attempted"] += source._rng.numel()
    if report["source_preservation_copy_elements_attempted"] > 20_000_000:
        raise ValueError("source preservation RNG-copy allowance exceeded")
    current = dict(
        weights={n: p for n, p in source._trainable.items()},
        gradients={n: p.grad for n, p in source._trainable.items()},
        optimizer=source._optimizer.state_dict(),
        rng=source._rng,
        buffers=dict(source.model.named_buffers()),
        caller_rng=source._torch.get_rng_state(),
    )
    expected = dict(
        weights=saved.weights,
        gradients=saved.gradients,
        optimizer=saved.optimizer,
        rng=saved.rng,
        buffers=birth["buffers"],
        caller_rng=birth["caller_rng"],
    )
    if (
        not _same_tree(source._torch, current, expected, report)
        or source.work != birth["work"]
        or source.optimizer_reset_work != birth["reset"]
        or source.update_health_work != birth["health"]
        or source.last_update != birth["last_update"]
        or tuple((n, m.training) for n, m in source.model.named_modules()) != birth["modes"]
    ):
        raise ValueError("reserved probe changed source state/work/RNG/mode")


def run_fresh_probe(source, report, progress, *, source_history="after_A6"):
    source_updates = _source_updates(source_history)
    report.update(
        source_preserved=False,
        probe_complete=False,
        source_preservation_comparison_elements_attempted=0,
    )
    source._guard()
    if (
        set(source.work.values()) != {source_updates}
        or any(type(v) is not int for v in source.work.values())
        or source.model.training
        or source.policy.max_updates != source_updates + PROBE_UPDATES
        or source.policy.max_forks != 1
        or any(source.fork_work.values())
        or any(source.optimizer_reset_work.values())
        or any(source.update_health_work.values())
        or (source_updates == 0 and (source._optimizer.state or source.last_update is not None))
    ):
        raise ValueError("reserved probe requires the admitted source history and native allowance")
    report["source_history"] = source_history
    birth = _capture_source(source, report)
    child = None
    try:
        child = source.fork()
        report["fork_work"] = source.fork_work
        report["child_birth_work"] = child.work
        if (
            sum(v for k, v in source.fork_work.items() if k.endswith("elements_attempted"))
            != report["fork_copy_elements_prospective"]
        ):
            raise ValueError("native probe fork-copy work differs from its admitted bound")
        _assert_source_unchanged(source, birth, report)
        progress("probe_fork_created")
        run_probe_updates(child, report, progress, source_history=source_history)
    except Exception as error:
        report["probe_error"] = f"{type(error).__name__}: {error}"
        report["probe_complete"] = False
        raise
    finally:
        report["source_native_work"] = source.work
        report["fork_work"] = source.fork_work
        if child is not None:
            report["child_native_work"] = child.work
            report["child_incremental_work"] = {
                k: v - report["child_birth_work"][k] for k, v in child.work.items()
            }
            report["child_optimizer_reset_work"] = child.optimizer_reset_work
            report["child_health_work"] = child.update_health_work
        try:
            _assert_source_unchanged(source, birth, report)
            report["source_preserved"] = True
        except Exception as error:
            report["source_preservation_error"] = f"{type(error).__name__}: {error}"
            report["probe_complete"] = False
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("reference", "source", "checkpoint", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--cold-control", action="store_true")
    parser.add_argument("--warm-reference", type=Path)
    args = parser.parse_args(argv)
    if args.cold_control != (args.warm_reference is not None):
        parser.error("--cold-control requires --warm-reference; warm mode forbids it")
    source_history = "cold" if args.cold_control else "after_A6"
    source_updates = _source_updates(source_history)
    forward_limit = source_updates + PROBE_UPDATES + 12
    if args.output.exists():
        parser.error("output exists; preserve previous evidence")
    with args.output.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    report: dict[str, Any] = dict(
        definition="native_cold_c_probe_v1"
        if args.cold_control
        else "native_reserved_text_probe_v1",
        complete=False,
        events=[],
        seed=17,
        model_load_attempts=0,
        final_examples=0,
        replay_examples=0,
        optimizer_resets=0,
        prediction_interface="native_owner_candidate_batched_probability_v1",
        calibration_status="inherited_unrecalibrated",
        training_random_draws="unknown",
        checkpoint_recomputation_work="unmeasured_internal_layers",
    )
    owner: dict[str, Any] = {}
    started = monotonic()

    def progress(event):
        import resource

        event = dict(event=event) if type(event) is str else event
        event.update(
            outer_seconds=monotonic() - started,
            outer_peak_rss_bytes=getattr(resource, "getrusage")(
                getattr(resource, "RUSAGE_SELF")
            ).ru_maxrss
            * 1024,
        )
        report["events"].append(event)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        if event["outer_seconds"] > 600 or event["outer_peak_rss_bytes"] > 20 * 1024**3:
            raise RuntimeError("reserved probe resource allowance exceeded")

    def observe_model(model):
        api = importlib.import_module("jev.api")
        source = OpenJevTrainingLearner(
            model,
            api.compile_request,
            instructions=INSTRUCTIONS,
            policy=OpenJevTrainingPolicy(source_updates + PROBE_UPDATES, 8, max_forks=1),
        )
        owner.update(learner=source, before=_weights(model))
        weights = _weight_summary(model, owner["before"])
        validate_initial_weights(weights, owner["reference"])
        rng = hashlib.sha256(source._rng.numpy().tobytes()).hexdigest()
        if (
            model.training
            or source._optimizer.state_dict()["state"]
            or rng != owner["reference"]["initial_rng_sha256"]
        ):
            raise ValueError("reserved probe initial optimizer/mode/RNG differs")
        report["initial_binding"] = dict(
            weights=weights, private_rng_sha256=rng, optimizer_empty=True, model_training=False
        )
        progress("initial_state_bound")

    try:
        if args.cold_control:
            owner["warm_reference"] = load_warm_probe(args.warm_reference)
            report["warm_reference_sha256"] = owner["warm_reference"]["source_sha256"]
        reference = load_reference(args.reference, include_after_A=not args.cold_control)
        owner["reference"] = {
            k: reference[k]
            for k in ("baseline_weights", "after_A_weights", "initial_rng_sha256")
            if k in reference
        }
        report["reference_sha256"] = reference["source_sha256"]
        report["examples"] = [asdict(e) for e in probe_examples()]
        report["model_load_attempts"] = 1
        _load(
            args.source,
            args.checkpoint,
            progress,
            prediction_limit=1,
            forward_limit=forward_limit,
            token_limit=forward_limit * 3 * 256,
            model_observer=observe_model,
            work_reporter=lambda f: owner.update(observe_work=f),
        )
        source = owner["learner"]
        report["delivery"] = []
        if source_updates:
            clock = LogicalClock()
            budget = ToyBudgetSession(ToyExecutionBudget(6, 600), monotonic)
            inbox = ExperienceInbox(
                source,
                clock=clock,
                budget=budget,
                learner_version="native-rank8",
                max_experiences=6,
            )
            _deliver_phase(
                source, inbox, clock, sequence_examples(), "after_A", "A", report, progress
            )
            weights = _weight_summary(source.model, owner["before"])
            validate_after_A_weights(weights, owner["reference"])
            if set(source.work.values()) != {6} or source.model.training:
                raise ValueError("reserved probe requires exactly six A TRAIN updates")
            report["after_A_binding"] = dict(weights=weights, native_work=source.work)
            progress("after_A_bound")
        run_fresh_probe(source, report, progress, source_history=source_history)
        if args.cold_control:
            report["warm_comparison"] = compare_probe_summaries(
                report["summary"], owner["warm_reference"]["summary"]
            )
        owner["observe_work"]()
        counts = next(
            e["observed_work"] for e in reversed(report["events"]) if "observed_work" in e
        )
        lengths = counts["candidate_sequence_lengths"]
        if (
            counts["decision_forward_calls"] != forward_limit
            or counts["scorer_calls"]
            or counts["native_prediction_calls_attempted"]
            or counts["native_prediction_calls_returned"]
            or counts["initializer_cpu_elements"] > 1
            or counts["initializer_meta_elements"] > 511147008
            or len(lengths) != forward_limit * 3
            or sum(lengths) > forward_limit * 3 * 256
            or any(n > 256 for n in lengths)
            or report["child_incremental_work"]
            != dict(
                training_attempts=6,
                forward_calls=18,
                backward_calls=6,
                optimizer_steps=6,
                committed_updates=6,
            )
        ):
            raise ValueError("reserved probe observed work differs from protocol")
        report["complete"] = report["probe_complete"] and report["source_preserved"]
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if "learner" in owner:
            report["source_native_work"] = owner["learner"].work
            report["fork_work"] = owner["learner"].fork_work
        if "observe_work" in owner and not report["complete"]:
            try:
                owner["observe_work"]()
            except Exception as error:
                report["final_observation_error"] = str(error)
        sys.setprofile(None)
        report["outer_seconds"] = monotonic() - started
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
