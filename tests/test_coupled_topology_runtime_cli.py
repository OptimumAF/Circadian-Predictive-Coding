"""Real native serving transactions and fixed example; no quality comparison."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.adapters import coupled_topology_demo as demo
from src.app.coupled_prune_consolidation import CoupledPruneConsolidator
from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_topology import CoupledTopologyState
from src.core.learner_ports import TrainingDiagnostic


def test_should_describe_without_native_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("description must not construct or train native models")

    monkeypatch.setattr(demo, "source_model", forbidden)
    assert demo.describe_example()["train_calls"] == 9
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_topology_runtime", "--describe"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert not result.stderr
    assert json.loads(result.stdout) == demo.describe_example()


def test_should_run_native_prune_consolidation_promotion_and_exact_rollback() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_topology_runtime"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert not result.stderr
    report = json.loads(result.stdout)
    assert len(result.stdout.encode("utf-8")) <= 2 * 1024**2
    assert report["train_calls"] == 9
    assert report["guarded_infer_calls"] == 24
    assert report["coupled_energy_evaluations"] == 108
    assert all(report["pc_neutral_parameter_parity"].values())
    excluded = {tuple(key) for key in report["excluded_selection_ids"]}
    for item in report["records"]:
        initial, promoted, restored = item["initial"], item["promoted"], item["rolled_back"]
        assert initial["generation"] == 0 and len(initial["cache"]) == 1
        assert promoted["generation"] == 1 and promoted["cache"] == {}
        assert promoted["actor_version"] == "candidate-0"
        assert promoted["metadata"] == {"route": ["pruned"]}
        assert restored == {**initial, "generation": 2}
        assert item["served"]["generation"] == 1 and not item["served"]["cache_hit"]
        pruned, continued = item["pruned_candidate"], item["continued_candidate"]
        assert pruned["state"]["hidden_dims"] == continued["state"]["hidden_dims"] == [3, 2]
        assert pruned["consolidation_attempts"] == continued["consolidation_attempts"] == 3
        assert pruned["consolidations_completed"] == continued["consolidations_completed"] == 1
        assert pruned["wake_updates"] == 0 and continued["wake_updates"] == 1
        assert len(continued["state"]["prune_lineage"]) == 1
        assert promoted["model_state"] == continued["state"]
        assert [d["status"] for d in item["refusals"]] == ["rejected", "failed"]
        assert len(item["refusals"][0]["diagnostics"]) == 2
        assert item["refusals"][1]["diagnostics"] == []
        assert item["rejected_promotion_report"]["decision"]["reasons"] == ["resource"]
        report_item = item["accepted_promotion_report"]
        assert report_item["decision"]["accepted"]
        assert excluded <= {tuple(key) for key in report_item["training_ids"]}
        guard_ids = {
            tuple(key) for key in report_item["new_guard_keys"] + report_item["old_guard_keys"]
        }
        assert not guard_ids & excluded
        if item["variant"] != "pc":
            before_wake = initial["model_state"]["snapshot"]["wake_state"]
            after_wake = pruned["state"]["snapshot"]["wake_state"]
            assert before_wake["wake_updates"] == after_wake["wake_updates"] == 2
            assert after_wake["chemical_state"] == before_wake["chemical_state"][:2]
            assert after_wake["plasticity_state"] == before_wake["plasticity_state"][:2]
            assert continued["state"]["snapshot"]["wake_state"]["wake_updates"] == 3


@pytest.mark.parametrize("fault", ["build", "restore"])
def test_should_keep_serving_cache_and_candidate_on_late_preparation_fault(fault: str) -> None:
    actor, runtime, _, prototype = demo.setup_runtime("active_cpc")
    actor.serve(demo.guards()[0].features, now=1)
    runtime.consolidate("pruned", demo.consolidator(2))
    before_actor, before_candidate = actor.serving_snapshot(), runtime.candidate_snapshot()
    controller = demo.make_controller(actor, prototype)
    calls = 0

    def broken_builder(state: CoupledTopologyState):
        nonlocal calls
        calls += 1
        if calls == 3 and fault == "build":
            raise RuntimeError("injected build fault after guard measurements")
        learner = prototype.build_from_state(state)
        if calls == 3:

            def broken_restore(state: CoupledTopologyState) -> None:
                # Mutate the fresh private model before failure; published owners
                # must still retain complete old models/cache/version/lineage.
                learner._payload.learner._model._parameters.output_bias[:] = 12.0
                raise RuntimeError("injected restore fault after private mutation")

            learner.restore_state = broken_restore  # type: ignore[method-assign]
        return learner

    controller._build = controller._evaluator._build = broken_builder
    with pytest.raises(RuntimeError, match="injected"):
        controller.prepare(runtime, *demo.guards(), now=2, training_ids=demo.selection_ids())
    assert calls == 3
    assert demo.state_digest(actor.serving_snapshot()) == demo.state_digest(before_actor)
    assert demo.state_digest(runtime.candidate_snapshot()) == demo.state_digest(before_candidate)
    assert controller._pending == {} and controller._latest is None


def test_should_refuse_stale_ticket_after_same_payload_consolidation() -> None:
    actor, runtime, _, prototype = demo.setup_runtime("active_cpc")
    runtime.consolidate("pruned", demo.consolidator(2))
    actor.serve(demo.guards()[0].features, now=1)
    controller = demo.make_controller(actor, prototype)
    ticket = controller.prepare(runtime, *demo.guards(), now=2, training_ids=demo.selection_ids())
    before = actor.serving_snapshot()
    runtime.consolidate(
        "same",
        lambda state: ConsolidatedState(
            state,
            TrainingDiagnostic("same_payload_fixture_v1", 0.0),
        ),
    )
    with pytest.raises(ValueError, match="candidate"):
        controller.commit(runtime, ticket)
    assert demo.state_digest(actor.serving_snapshot()) == demo.state_digest(before)
    controller.discard(ticket)


def test_should_exclude_selection_labels_from_promotion_even_when_not_wake_trained() -> None:
    actor, runtime, _, prototype = demo.setup_runtime("pc")
    runtime.consolidate("pruned", demo.consolidator(2))
    assert runtime.applied_updates == ()
    actor.serve(demo.guards()[0].features, now=1)
    before = actor.serving_snapshot()
    new, old = demo.guards()
    selected = replace(new, sample_keys=(("selection", "validation-0"),))
    controller = demo.make_controller(actor, prototype)
    with pytest.raises(ValueError, match="disjoint"):
        controller.prepare(runtime, selected, old, now=2, training_ids=demo.selection_ids())
    assert demo.state_digest(actor.serving_snapshot()) == demo.state_digest(before)


def test_should_reject_final_role_before_copying_probe_payload() -> None:
    class Sealed:
        def __deepcopy__(self, memo: object) -> None:
            raise AssertionError("final payload must not be copied")

    probe = ScalingBatch("final_test", ("final",), Sealed(), Sealed())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="final_test"):
        CoupledPruneConsolidator(
            0,
            0,
            (probe,),
            limits=CoupledPruneLimits(0, 0, 0),
            budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0),
            inference_steps=2,
            inference_learning_rate=0.05,
        )


def test_should_refuse_shared_probe_storage_before_ownership_copy_erases_it() -> None:
    features = np.ones((2, 2))
    probe = ScalingBatch("validation", ("v0", "v1"), features, features[:, :1])
    with pytest.raises(ValueError, match="storage"):
        CoupledPruneConsolidator(
            0,
            0,
            (probe,),
            limits=CoupledPruneLimits(0, 0, 0),
            budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0),
            inference_steps=2,
            inference_learning_rate=0.05,
        )


@pytest.mark.parametrize("shape", [(1025, 2), (1, 1_000_001)])
def test_should_bound_probe_allocation_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
) -> None:
    from src.app import coupled_prune_consolidation as module

    def forbidden(value: object) -> None:
        raise AssertionError("oversized payload must not be copied")

    # Broadcast views describe the oversized input without allocating its size.
    features = np.broadcast_to(np.array(0.0), shape)
    targets = np.zeros((shape[0], 1))
    probe = ScalingBatch("validation", tuple(f"v{i}" for i in range(shape[0])), features, targets)
    monkeypatch.setattr(module, "deepcopy", forbidden)
    with pytest.raises(ValueError, match="bounds"):
        CoupledPruneConsolidator(
            0,
            0,
            (probe,),
            limits=CoupledPruneLimits(0, 0, 0),
            budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0),
            inference_steps=2,
            inference_learning_rate=0.05,
        )
