"""Known-source/data preparation for the existing isolated ticket probe recipe.

Uses original generic probe/return/endpoint functions passed by its CLI owner.
No benchmark training, new runtime, final data, replay or context discovery.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts.run_ticket_workflow import CONTEXT_FEATURES, build_joint_context_batch, training_cases
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.perceptron_decision import build_decision_training_batch
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.probe_return_phase import paid_report_guard
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.context_perceptron import (
    ContextPerceptronPolicy,
    ContextPerceptronSnapshot,
    copy_context_snapshot,
)
from src.core.decision_features import DecisionFeatureSpec
from src.core.temporal_decision_sequence import VOCABULARY
from src.core.ticket_workflow import CATALOG
from src.shared.process_memory import ProcessRssSampler

HEAD_SOURCE_SHA = "addf37403d32682b680b74ea7a1bfc16ced2895cffe09b8cc5fdf701179be943"


def _source_bytes(path):
    with Path(path).open("rb") as source:
        data = source.read(512 * 1024 + 1)
    if len(data) > 512 * 1024:
        raise ValueError("qualified contextual source exceeds bounded record size")
    return data


def _snapshot(record):
    keys = {"policy", "coefficients", "completed_updates", "training_examples", "mistake_updates"}
    if type(record) is not dict or set(record) != (keys - {"coefficients"}) | {"heads"}:
        raise ValueError("complete contextual source snapshot required")
    policy = record["policy"]
    if type(policy) is not dict or set(policy) != {"feature_dim", "num_classes", "context_columns"}:
        raise ValueError("complete contextual source policy required")
    owned_policy = ContextPerceptronPolicy(
        policy["feature_dim"], policy["num_classes"], tuple(policy["context_columns"])
    )
    if type(record["heads"]) is not list or len(record["heads"]) != 2:
        raise ValueError("two complete source heads required")
    heads = []
    for h in record["heads"]:
        if type(h) is not dict or set(h) != keys:
            raise ValueError("complete original head fields required")
        heads.append(
            PerceptronSnapshot(
                PerceptronPolicy(**h["policy"]),
                tuple(tuple(row) for row in h["coefficients"]),
                h["completed_updates"],
                h["training_examples"],
                h["mistake_updates"],
            )
        )
    return copy_context_snapshot(
        ContextPerceptronSnapshot(
            owned_policy,
            (heads[0], heads[1]),
            record["completed_updates"],
            record["training_examples"],
            record["mistake_updates"],
        )
    )


def head_states(source):
    if (
        source.get("complete") is not True
        or source.get("definition") != "separate_context_head_ticket_workflow_v1"
    ):
        raise ValueError("complete qualified separate-head source required")
    points = source["checkpoints"]
    if len(points) != 4 or [c["source_updates"] for c in points] != [0, 4, 8, 12]:
        raise ValueError("original qualified source points0/4/8/12 required")
    for c in points:
        n, s = c["source_updates"], c["state"]
        if (
            type(n) is not int
            or type(c["source_rows"]) is not int
            or c["source_rows"] != 6 * n
            or type(s["completed_updates"]) is not int
            or s["completed_updates"] != n
            or type(s["training_examples"]) is not int
            or s["training_examples"] != 6 * n
        ):
            raise ValueError("original qualified contextual clocks required")
    states = [_snapshot(points[i]["state"]) for i in (3, 0)]
    if any(s.policy != ContextPerceptronPolicy(9, 6, (4, 5)) for s in states):
        raise ValueError("source must preserve the fixed D9/six-class public context schema")
    if states[0].mistake_updates != 16:
        raise ValueError("original qualified source mistake history required")
    return states


def build_head_cohorts(spec):
    """Project explicitly admitted TRAIN only; legacy selection is outcome-free."""
    if type(spec) is not DecisionFeatureSpec or spec != DecisionFeatureSpec(
        CONTEXT_FEATURES, CATALOG
    ):
        raise ValueError("head probe requires fixed contextual feature/catalog schema")
    teachers = {
        v: training_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")
    }
    _, old = build_joint_context_batch(spec, teachers)
    start = 3 * VOCABULARY.index("legacy")
    fresh = [
        build_decision_training_batch(
            teachers[v][start : start + 3], spec, f"legacy-{v}", allow_subsets=True
        )
        for v in ("v1", "v2")
    ]

    def batch(base, start, rows, role, context, identifier):
        return AttributionBatch(
            role,
            context,
            tuple(f"{identifier}-{i}" for i in range(rows)),
            base.features[start : start + rows].copy(),
            base.targets[start : start + rows].copy(),
            num_classes=6,
        )

    training = tuple(
        batch(fresh[i % 2], 0, 3, "probe_train", f"legacy-v{i % 2 + 1}", f"probe-{i}")
        for i in range(16)
    )
    evaluation = tuple(
        batch(fresh[i], 0, 3, "probe_test", f"legacy-v{i + 1}", f"eval-{i}") for i in range(2)
    )
    monitors = tuple(
        batch(old, i * 6, 6, "train", f"old-v{i + 1}", f"monitor-{i}") for i in range(2)
    )
    returning = tuple(
        batch(old, i * 6, 6, "train", f"old-v{i + 1}", f"return-{i}") for i in range(2)
    )
    return training, evaluation, monitors, returning


def run_context_case(input_path, *, run_arm, require_arm):
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="isolated_context_head_legacy_probe_v1",
        complete=False,
        source_sha256=HEAD_SOURCE_SHA,
        cases=[],
        training_work=dict(attempted_updates=0, attempted_rows=0, returned_updates=0),
        final_examples=0,
        replay_examples=0,
        promotions=0,
        source_prior_training=dict(updates=12, rows=72, mistakes=16),
        scope="unused known TRAIN legacy templates on private warm/fresh96-coefficient forks; no benchmark/final/deployed updates, unseen-task/CPC/plasticity-cause or equal-all-in-compute claim",
    )
    memory = None
    try:
        data = _source_bytes(input_path)
        if hashlib.sha256(data).hexdigest() != HEAD_SOURCE_SHA:
            raise ValueError("qualified contextual source replaced")
        source = json.loads(data)
        states = head_states(source)
        state_pin = deepcopy(states)
        numpy_before = deepcopy(np.random.get_state())
        spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
        cohorts = build_head_cohorts(spec)
        cohort_pin = deepcopy(cohorts)
        seeds: list[tuple[Any, Any]] = []
        paid: list[Any] = []
        cases_owner = report["cases"]
        work = report["training_work"]

        def guard():
            if monotonic() - started > 120:
                raise ValueError("original contextual probe case deadline exhausted")
            if (
                _source_bytes(input_path) != data
                or not _same_value(cohorts, cohort_pin)
                or not _same_value(states, state_pin)
            ):
                raise ValueError("context probe source or original input data changed")
            if not _same_value(np.random.get_state(), numpy_before):
                raise ValueError("context probe changed global NumPy RNG")
            if any(not _same_value(seed.snapshot_state(), original) for seed, original in seeds):
                raise ValueError("context probe changed original source owner")
            if report["cases"] is not cases_owner:
                report["cases"] = cases_owner
                raise ValueError("preceding paid probe evidence replaced")
            for i, (row, pin, check_paid) in enumerate(paid):
                changed = cases_owner[i] is not row
                try:
                    check_paid()
                except ValueError:
                    changed = True
                if changed:
                    row.clear()
                    row.update(deepcopy(pin))
                    cases_owner[i] = row
                    raise ValueError("preceding paid probe evidence changed")

        class CountedLearner(ContextPerceptronLearner):
            def fork(self):
                child = CountedLearner(self.policy)
                child.restore_state(self.snapshot_state())
                return child

            def train_batch(self, features, targets):
                guard()
                work["attempted_updates"] += 1
                work["attempted_rows"] += len(features)
                if work["attempted_updates"] > 64 or work["attempted_rows"] > 288:
                    raise ValueError("original aggregate contextual probe allowance exhausted")
                result = super().train_batch(features, targets)
                work["returned_updates"] += 1
                return result

        with ProcessRssSampler(interval_seconds=0.02) as memory:
            for name, state in zip(("warm", "fresh"), states, strict=True):
                guard()
                seed = CountedLearner(state.policy)
                seed.restore_state(state)
                seeds.append((seed, deepcopy(state)))
                guard()
                row: dict[str, Any] = {}
                cases_owner.append(row)
                run_arm(
                    seed,
                    state,
                    cohorts,
                    spec,
                    guard,
                    memory,
                    name,
                    row=row,
                    separate_context_heads=True,
                )
                require_arm(row, separate_context_heads=True)
                guard()
                check_paid, pin = paid_report_guard(row)
                paid.append((row, pin, check_paid))
            guard()
        report.update(
            complete=True,
            source_owners_unchanged=True,
            source_bytes_unchanged=True,
            numpy_rng_unchanged=True,
            model_coefficients=96,
            model_coefficient_bytes=768,
            input_array_bytes=sum(
                a.nbytes for group in cohorts for b in group for a in (b.features, b.targets)
            ),
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"[:1024]
    finally:
        if memory is not None:
            report["sampled_process_rss"] = asdict(memory.snapshot())
        report["training_work"]["unreturned_work_unknown"] = (
            report["training_work"]["attempted_updates"]
            != report["training_work"]["returned_updates"]
        )
        report["elapsed_seconds"] = monotonic() - started
    return report
