"""Fixed, local R2.7 TRAIN-only study through existing stream/observer ports.

Three known-template conditions; no selection, tuning, publication or final/probe
data. All training finishes before fixed private task readouts. This is a small
descriptive experiment, not a deployment controller or comparative benchmark.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts.context_head_probe import HEAD_SOURCE_SHA, _source_bytes, head_states
from scripts.context_head_serving import _episodes, prepare_data
from scripts.run_ticket_workflow import CONTEXT_FEATURES
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.context_serving import ContextServingPredictor
from src.app.arrived_shift_observer import ArrivedShiftObserver
from src.app.coupled_attribution import AttributionBatch, AttributionStream, _same_value
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.context_perceptron import predict_context_perceptron
from src.core.decision_features import DecisionFeatureSpec
from src.core.serving_ports import ServingConfiguration
from src.core.shift_signals import ShiftPolicy
from src.core.ticket_workflow import CATALOG
from src.infra.context_head_example import example_context_source
from src.shared.process_memory import ProcessRssSampler

CONDITIONS = ("stationary", "stationary_label_noise", "recurring")
ENDPOINTS = (0, 4, 8, 12)


def build_conditions(training):
    """Equal six-row exposures; noise is a fixed corruption from the first row.

    Why this: a fixed label error throughout is stationary. It must not be
    described as an independently sampled random-noise experiment.
    """
    new, old = training
    if (
        new.features.shape != (3, 9)
        or old.features.shape != (6, 9)
        or new.targets.shape != (3, 1)
        or old.targets.shape != (6, 1)
        or new.num_classes != 6
        or old.num_classes != 6
    ):
        raise ValueError("study requires original three-row legacy and six-row old packets")
    result = {}
    for name in CONDITIONS:
        packets = []
        for i in range(12):
            use_new = name == "recurring" and 4 <= i < 8
            base = new if use_new else old
            x = np.tile(base.features, (2, 1)) if use_new else base.features.copy()
            y = np.tile(base.targets, (2, 1)) if use_new else base.targets.copy()
            if name == "stationary_label_noise":
                y[0, 0] = (y[0, 0] + 1) % 6
            packets.append(
                AttributionBatch(
                    "train",
                    "legacy-B" if use_new else "old-A",
                    tuple(f"{name}-{i + 1}-{r}" for r in range(6)),
                    x,
                    y,
                    num_classes=6,
                )
            )
        result[name] = tuple(packets)
    return result


def train_condition(initial, packets, sampler, require, row):
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=12, max_wall_seconds=45, max_process_rss_bytes=512 * 1024**2
        ),
        monotonic,
    )
    budget.attach_memory(sampler)
    source = ContextPerceptronLearner(initial.policy)
    source.restore_state(initial)
    observer = ArrivedShiftObserver(
        predict_context_perceptron,
        num_classes=6,
        max_updates=12,
        incidents=True,
    )
    saved = {}

    def observe(k, state, batch):
        require()
        observer(k, state, batch)
        if k in ENDPOINTS:
            saved[k] = deepcopy(state)

    stream = None
    try:
        stream = AttributionStream(
            source,
            packets,
            (),
            checkpoint_updates=tuple(range(13)),
            label_delay=1,
            budget=budget,
            training_only=True,
            observe_checkpoint=observe,
        )
        stream.run_until(12)
        row["stream"] = stream.finish()
        if tuple(saved) != ENDPOINTS or source.snapshot_state() != initial:
            raise ValueError("missing endpoint or source mutated")
        row["states"] = {str(k): asdict(v) for k, v in saved.items()}
        row["training_complete"] = True
        return saved
    finally:
        row["observations"] = observer.report()
        row["budget"] = dict(
            limits=asdict(budget.budget),
            started_at=budget.started_at,
            updates_completed=budget.updates_completed,
        )
        row["source_unchanged"] = source.snapshot_state() == initial
        if stream is not None:
            row["applied_updates"] = [asdict(v) for v in stream.applied_updates]


def read_tasks(saved, spec, require, row):
    for k in ENDPOINTS:
        require()
        state = saved[k]
        owner = ContextPerceptronLearner(state.policy)
        owner.restore_state(state)
        actor = PromotableActor(
            owner,
            version=f"private-{k}",
            configuration=ServingConfiguration(3, 2),
            feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
            metadata={"source": HEAD_SOURCE_SHA, "scope": "private_fixed_train_readout"},
        )
        predictor = ContextServingPredictor(actor, spec)
        try:
            _episodes(predictor, f"update-{k}", require, row["episodes"], role="controller_train")
            if owner.snapshot_state() != state or actor.serving_snapshot().model_state != state:
                raise ValueError("private task readout mutated its owner")
        finally:
            row["typed_work"].append(dict(endpoint=k, **predictor.work))


def run_conditions(initial, data, spec, sampler, require, report):
    """The phase barrier prevents any task readout from selecting training."""
    saved = {}
    for name in CONDITIONS:
        row: dict[str, Any] = dict(
            condition=name, training_complete=False, episodes=[], typed_work=[]
        )
        report["cases"].append(row)
        saved[name] = train_condition(initial, data[name], sampler, require, row)
        require()
    report["all_training_complete_before_task_readouts"] = True
    for row in report["cases"]:
        read_tasks(saved[row["condition"]], spec, require, row)
        require()


def run_case(input_path):
    started = monotonic()
    report: dict[str, Any] = dict(
        schema="context_shift_study_v1",
        complete=False,
        cases=[],
        source_sha=HEAD_SOURCE_SHA,
        prior_training=dict(updates=12, rows=72, mistakes=16),
        shift_policy=asdict(ShiftPolicy()),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        feedback=False,
        scope="fixed known TRAIN templates; no comparative or deployment claim",
    )
    rng = deepcopy(np.random.get_state())
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            raw = _source_bytes(input_path)
            if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
                raise ValueError("qualified contextual source replaced")
            initial = head_states(json.loads(raw))[0]
            spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
            data = build_conditions(prepare_data(spec)[0])
            pin = deepcopy(data)

            def require():
                if monotonic() - started > 120:
                    raise ValueError("original whole-study deadline exhausted")
                if _source_bytes(input_path) != raw or not _same_value(data, pin):
                    raise ValueError("qualified source or fixed TRAIN data changed")
                if not _same_value(np.random.get_state(), rng):
                    raise ValueError("caller NumPy RNG changed")
                memory = sampler.sample()
                if memory is None or memory > 512 * 1024**2:
                    raise ValueError("study RSS unavailable or exhausted")

            require()
            report["input_bytes"] = sum(
                b.features.nbytes + b.targets.nbytes for packets in data.values() for b in packets
            )
            run_conditions(initial, data, spec, sampler, require, report)
            require()
            report["source_and_data_unchanged"] = True
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            report["numpy_rng_unchanged"] = _same_value(np.random.get_state(), rng)
            report["elapsed_seconds"] = monotonic() - started
            report["memory"] = asdict(sampler.snapshot())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve exclusively before any teacher/model work; failures remain spent.
    with args.output.open("x", encoding="utf8") as output:
        with example_context_source() as source:
            result = run_case(source)
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if len(encoded.encode("utf8")) > 6 * 1024**2:
            raise ValueError("study output exceeds six MiB; run remains spent")
        output.write(encoded)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
