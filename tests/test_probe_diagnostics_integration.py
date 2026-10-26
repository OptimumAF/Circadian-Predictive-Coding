"""Real reserved probes add decision reports without extra native work or state changes."""

from copy import deepcopy
from dataclasses import asdict
import json
import pickle
from time import perf_counter
from typing import Any

import numpy as np

from src.adapters import torch_recurrent_learner as native
from src.adapters.key_door_pilot import _collect
from src.app.actor_shadow import StableActor
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.key_door_navigation import KeyDoorLayout, KeyDoorNavigation
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch
from test_torch_feature_learner import assert_tree

ENGINEERING_COUNTS: dict[str, int] = {}


def _counters(monkeypatch):
    caps = dict(sgd=4, train_rows=64, unroll=256, predict=128, physics=64)
    # Keep permanent attempted costs visible to the ordinary pytest controller,
    # including assertion failures before the successful result is written.
    ENGINEERING_COUNTS.clear()
    ENGINEERING_COUNTS.update(dict.fromkeys(caps, 0))
    counts = ENGINEERING_COUNTS

    def charge(key, value=1):
        if counts[key] + value > caps[key]:
            raise AssertionError(f"{key} engineering cap before entry")
        counts[key] += value

    def measured(owner, name, key):
        original = getattr(owner, name)

        def call(*args, **kwargs):
            charge(key)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, call)

    measured(KeyDoorNavigation, "step", "physics")
    return caps, counts, charge, measured


def test_should_add_native_probe_reports_with_identical_scores_updates_and_frozen_source(
    tmp_path, monkeypatch
):
    caps, counts, charge, measured = _counters(monkeypatch)
    started = perf_counter()
    with ProcessRssSampler() as memory:
        torch = require_torch()
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        measured(torch.optim.SGD, "step", "sgd")
        measured(native.TorchRecurrentLearner, "predict", "predict")
        forward = native._forward

        def observed(parameters, features, hidden):
            charge("unroll", len(features))
            if torch.is_grad_enabled():
                charge("train_rows", len(features))
            return forward(parameters, features, hidden)

        monkeypatch.setattr(native, "_forward", observed)
        source = native.TorchRecurrentLearner(native.TorchRecurrentPolicy(8, 4, 5, 17))
        state = source.snapshot_state()
        actor = StableActor(source, version="before-diagnostic-probes")
        actor_before = actor.snapshot_state()
        layout = KeyDoorLayout("E", 3, 3, True, 0.75)
        training = (_collect("diag-reserved-train", layout, 1, "probe_train"),)
        evaluation = tuple(
            _collect(f"diag-reserved-test-{cue}", layout, cue, "probe_test") for cue in (1, -1)
        )
        data_before = pickle.dumps((training, evaluation))
        torch_before = torch.get_rng_state().clone()
        numpy_before = pickle.dumps(np.random.get_state())
        children, reports, costs = [], [], []
        original_fork = source.fork

        def fork():
            owned = original_fork()
            children.append(owned)
            return owned

        monkeypatch.setattr(source, "fork", fork)
        budgets = []
        for enabled in (False, True):
            budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
            budgets.append(budget)
            before = counts.copy()
            call_started = perf_counter()
            report = run_fresh_task_probe(
                source,
                state,
                training,
                evaluation,
                budget=budget,
                checkpoint_updates=(0, 1),
                decision_diagnostics=enabled,
            )
            costs.append(
                dict(
                    enabled=enabled,
                    whole_call_seconds=perf_counter() - call_started,
                    native_counts={k: counts[k] - before[k] for k in counts},
                )
            )
            reports.append(report)
            assert report["status"] == "complete" and budget.updates_completed == 1
            assert (
                report["score_forward_calls_completed"]
                == report["score_forward_calls_attempted"]
                == 4
            )
            assert (
                report["completed_training_examples"] == 8 and report["reserved_data_bytes"] == 1728
            )
        original, enhanced = reports
        stripped = deepcopy(enhanced)
        stripped.pop("decision_diagnostics_enabled")
        stripped.pop("decision_diagnostics_scope")
        for point in stripped["checkpoints"]:
            point.pop("decision_diagnostics")
        stripped.pop("elapsed_seconds")
        default = deepcopy(original)
        default.pop("elapsed_seconds")
        assert stripped == default
        assert costs[0]["native_counts"] == costs[1]["native_counts"]
        assert len(children) == 2 and children[0] is not children[1]
        assert_tree(torch, children[0].snapshot_state(), children[1].snapshot_state())
        assert_tree(torch, source.snapshot_state(), state)
        assert_tree(torch, actor.snapshot_state(), actor_before)
        assert pickle.dumps((training, evaluation)) == data_before
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert pickle.dumps(np.random.get_state()) == numpy_before
        for point in enhanced["checkpoints"]:
            for packet in evaluation:
                assert packet.episode is not None
                detail = point["decision_diagnostics"][packet.context]
                assert detail["row_ids"] == list(packet.row_ids)
                assert detail["episode_id"] == packet.episode.episode_id
                assert detail["sequence_kind"] == "complete_episode"
                assert detail["rows"] == sum(detail["class_counts"]) == 8
                assert sum(map(sum, detail["confusion"])) == 8
                assert (
                    np.mean([q["loss"] for q in detail["decisions"]])
                    == point["scores"][packet.context]["ce"]
                )
                assert (
                    np.mean([q["correct"] for q in detail["decisions"]])
                    == point["scores"][packet.context]["accuracy"]
                )
        assert counts == dict(sgd=2, train_rows=16, unroll=80, predict=8, physics=24)
    rss = asdict(memory.snapshot())
    assert rss["peak_bytes"] <= 768 * 1024**2
    assert perf_counter() - started < 120
    result: dict[str, Any] = dict(
        classification="engineering equivalence, no scientific quality comparison",
        reports=reports,
        costs=costs,
        caps=caps,
        counts=counts,
        budget_charges=[b.updates_completed for b in budgets],
        source_actor_data_global_rng_unchanged=True,
        child_states_equal=True,
        sampled_process_rss=rss,
        total_seconds=perf_counter() - started,
    )
    (tmp_path / "probe-diagnostics-engineering.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
