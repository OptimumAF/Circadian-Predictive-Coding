"""Fixed two-update native engineering example; no utility/benefit evaluation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import run_bp_ticket_serving as demo
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.experience import Experience, ExperiencePermissions, LabelArrival


def run_example(input_path: Path) -> dict[str, Any]:
    started = monotonic()
    torch = demo.native.require_torch()
    rng, torch_rng = np.random.get_state(), torch.get_rng_state().clone()
    assert type(rng) is tuple
    work = dict(attempted_updates=0, attempted_rows=0, returned_updates=0)
    report: dict[str, Any] = dict(
        complete=False,
        events=[],
        selected_multipliers=[],
        final_reads=0,
        probe_updates=0,
        promotions=0,
    )
    shared = None
    with demo.ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            _, prior = demo.context._source(input_path)
            initial = demo.context.bp_snapshot_from_record(prior["checkpoints"][-1]["state"])
            spec = demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
            _, _, batches = demo.context._teacher_batches(spec)
            budget = demo.ToyBudgetSession(
                demo.ToyExecutionBudget(
                    max_training_updates=2, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
                ),
                monotonic,
            )
            budget.attach_memory(sampler)

            def before(features):
                work["attempted_updates"] += 1
                work["attempted_rows"] += len(features)

            def after():
                work["returned_updates"] += 1

            source = demo.native.counted_learner("bp", 7, 6, before, after)
            source.restore_state(initial)
            clock = demo.LogicalClock()
            runtime = demo.ActorShadowRuntime(
                source,
                actor_version="qualified-96",
                candidate_version="rate-candidate",
                clock=clock,
                budget=budget,
                max_experiences=2,
                max_consolidations=0,
            )
            gate = demo.ServingPriorityGate(
                demo.SharingLimits(1, 2, 1), resource_available=lambda: True
            )
            shared = demo.ResourceSharedRuntime(
                runtime,
                gate,
                action_envelope=AdaptationEnvelope(
                    (
                        AdaptationAction("wait"),
                        AdaptationAction("half", 1, learning_rate_multiplier=0.5),
                        AdaptationAction("wake", 1),
                    ),
                    8,
                    2,
                ),
            )

            def select(action):
                result = shared.train_action(action)
                report["events"].append(asdict(result))
                return result

            batch = batches[0]
            for index, action in enumerate(("half", "wake")):
                clock.advance_to(2 * index)
                runtime.record_experience(
                    Experience(
                        f"rate-{index}",
                        "example",
                        2 * index,
                        "qualified-96",
                        batch.features,
                        "train",
                        ExperiencePermissions(training=True),
                    )
                )
                runtime.record_label(
                    LabelArrival(
                        f"label-{index}",
                        f"rate-{index}",
                        "example",
                        2 * index + 1,
                        "qualified-96",
                        batch.targets,
                    )
                )
                assert select(action).training.updates == ()  # Labels are still in the future.
                clock.advance_to(2 * index + 1)
                if index == 0:
                    gate.pause()
                    assert select(action).training.deferred_reason == "paused"
                    gate.resume()
                    with gate.serving():
                        assert select(action).training.deferred_reason == "serving_active"
                poll = select(action)
                assert len(poll.training.updates) == 1
                report["selected_multipliers"].append(poll.requested_learning_rate_multiplier)
            try:
                select("half")
            except ValueError as error:
                report["quota_refusal"] = str(error)
            else:
                raise ValueError("original wake envelope failed")
            select("wait")
            state = runtime.candidate_snapshot().state
            assert (state.completed_updates, state.training_examples) == (98, 588)
            base_rate_preserved = (
                state.policy == initial.policy
                and state.optimizer_state["param_groups"][0]["lr"] == initial.policy.learning_rate
            )
            source_unchanged = demo.native.same_state(source.snapshot_state(), initial)
            actor_unchanged = demo.native.same_state(runtime.actor.snapshot_state().state, initial)
            assert base_rate_preserved and source_unchanged and actor_unchanged
            report.update(
                complete=True,
                candidate=demo.native.checkpoint(2, state),
                base_rate_preserved=base_rate_preserved,
                source_unchanged=source_unchanged,
                actor_unchanged=actor_unchanged,
                wake_updates=budget.updates_completed,
                sharing=asdict(gate.snapshot()),
            )
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
        finally:
            report["native_work"] = work | dict(
                unreturned_work_unknown=work["attempted_updates"] != work["returned_updates"]
            )
            if shared is not None:
                report["action_snapshot"] = asdict(shared.adaptation_snapshot())
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.output.open("xb") as target:
        target.write(b"{}\n")
    report = run_example(args.source)
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 512 * 1024:
        raise ValueError("native rate report exceeds512KiB")
    args.output.write_bytes(raw)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
