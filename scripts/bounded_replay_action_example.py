"""Small native engineering example of existing wake/replay admission ports.

No task-score comparison, controller fitting, promotion, probe/final data or new
runner. Exact qualified source reused; one authentic old TRAIN episode replayed.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts import run_bp_ticket_serving as demo
from scripts.recurring_ticket_workflow_example import prepare_training
from scripts.ticket_workflow_rehearsal import trajectory_batches
from src.adapters.torch_feature_learner import TORCH_CE_ID
from src.app.categorical_replay_action import CategoricalReplayAction
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.experience import Experience, ExperiencePermissions, LabelArrival


def run_example(input_path: Path) -> dict[str, Any]:
    started = monotonic()
    work = dict(attempted_updates=0, attempted_rows=0, returned_updates=0)
    report: dict[str, Any] = dict(
        complete=False,
        definition="bounded_categorical_replay_action_v1",
        events=[],
        final_reads=0,
        probe_updates=0,
        promotions=0,
    )
    torch = demo.native.require_torch()
    rng, torch_rng = np.random.get_state(), torch.get_rng_state().clone()
    assert type(rng) is tuple
    replay = shared = source = runtime = None
    with demo.ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            _, prior = demo.context._source(input_path)
            initial = demo.context.bp_snapshot_from_record(prior["checkpoints"][-1]["state"])
            spec, training = prepare_training()
            episode = trajectory_batches(spec, training)[0].episode
            _, _, batches = demo.context._teacher_batches(spec)
            wake = demo.ToyBudgetSession(
                demo.ToyExecutionBudget(
                    max_training_updates=2, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
                ),
                monotonic,
            )
            replay_budget = demo.ToyBudgetSession(
                demo.ToyExecutionBudget(
                    max_training_updates=0,
                    max_replay_examples=3,
                    max_wall_seconds=90,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                monotonic,
            )
            wake.attach_memory(sampler)
            replay_budget.attach_memory(sampler)

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
                candidate_version="replay-candidate",
                clock=clock,
                budget=wake,
                max_experiences=2,
                max_consolidations=1,
            )
            gate = demo.ServingPriorityGate(
                demo.SharingLimits(1, 3, 1), resource_available=lambda: True
            )
            replay = CategoricalReplayAction(
                runtime,
                source,
                replay_budget,
                (episode,),
                ReplayRetentionBudget(1, 2048),
                states_equal=demo.native.same_state,
                definition=TORCH_CE_ID,
            )
            shared = demo.ResourceSharedRuntime(
                runtime,
                gate,
                action_envelope=AdaptationEnvelope(
                    (
                        AdaptationAction("wait"),
                        AdaptationAction("wake", 1),
                        AdaptationAction("replay", replay_examples=3),
                    ),
                    8,
                    2,
                    3,
                ),
                replay_action=replay,
            )

            def select(action):
                result = shared.train_action(action)
                report["events"].append(asdict(result))
                return result

            gate.pause()
            assert select("replay").training.deferred_reason == "paused"
            gate.resume()
            with gate.serving():
                assert select("replay").training.deferred_reason == "serving_active"
            for index, tick in enumerate((1, 3)):
                batch = batches[index]
                clock.advance_to(tick - 1)
                runtime.record_experience(
                    Experience(
                        f"native-{index}",
                        "example",
                        tick - 1,
                        "qualified-96",
                        batch.features,
                        "train",
                        ExperiencePermissions(training=True),
                    )
                )
                runtime.record_label(
                    LabelArrival(
                        f"label-{index}",
                        f"native-{index}",
                        "example",
                        tick,
                        "qualified-96",
                        batch.targets,
                    )
                )
                clock.advance_to(tick)
                assert len(select("wake").training.updates) == 1
                if index == 0:
                    assert select("replay").replay_receipt is not None
            try:
                select("replay")
            except ValueError as error:
                report["quota_refusal"] = str(error)
            else:
                raise ValueError("replay envelope did not refuse an extra event")
            select("wait")
            state = runtime.candidate_snapshot().state
            if (state.completed_updates, state.training_examples) != (99, 591):
                raise ValueError("native wake/replay model clocks differ")
            report.update(
                candidate=demo.native.checkpoint(3, state),
                source_unchanged=demo.native.same_state(source.snapshot_state(), initial),
                actor_unchanged=demo.native.same_state(
                    runtime.actor.snapshot_state().state, initial
                ),
                wake_updates=wake.updates_completed,
                replay_examples=replay_budget.replay_examples_completed,
                sharing=asdict(gate.snapshot()),
                complete=True,
            )
            if not report["source_unchanged"] or not report["actor_unchanged"]:
                raise ValueError("original native source or actor changed")
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
        finally:
            report["native_work"] = work | dict(
                unreturned_work_unknown=work["attempted_updates"] != work["returned_updates"]
            )
            if replay is not None:
                report["replay"] = replay.report()
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
        raise ValueError("replay example report exceeds512KiB")
    args.output.write_bytes(raw)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
