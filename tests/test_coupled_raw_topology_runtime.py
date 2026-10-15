"""Real raw topology serving transactions and faults; bounded engineering only."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.adapters import coupled_scaling_topology_demo as demo
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_scaling_topology import CoupledScalingTopologyState
from src.core.learner_ports import TrainingDiagnostic


def test_should_describe_without_model_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("description must not construct native state")

    monkeypatch.setattr(demo, "source_state", forbidden)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_raw_topology_runtime", "--describe"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert not result.stderr
    assert json.loads(result.stdout) == demo.describe_example()


def test_should_run_raw_consolidation_promotion_and_complete_bundle_rollback() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_raw_topology_runtime"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert not result.stderr
    assert len(result.stdout) <= 2 * 1024**2
    report = json.loads(result.stdout)
    assert (
        report["train_calls"],
        report["guarded_infer_calls"],
        report["coupled_energy_evaluations"],
    ) == (9, 24, 108)
    assert report["pc_neutral_parameter_parity"] == {
        "pruned_candidate": True,
        "continued_candidate": True,
    }
    excluded = {tuple(key) for key in report["excluded_selection_ids"]}
    assert len(report["records"]) == 3
    for item in report["records"]:
        initial, promoted, restored = item["initial"], item["promoted"], item["rolled_back"]
        assert initial["generation"] == 0 and len(initial["cache"]) == 1
        assert promoted["generation"] == 1 and promoted["cache"] == {}
        assert promoted["actor_version"] == "candidate-0"
        assert promoted["metadata"] == {"route": ["pruned"]}
        assert restored == {**initial, "generation": 2}
        assert item["served"]["generation"] == 1 and not item["served"]["cache_hit"]
        pruned, continued = item["pruned_candidate"], item["continued_candidate"]
        assert pruned["consolidation_attempts"] == continued["consolidation_attempts"] == 3
        assert pruned["consolidations_completed"] == continued["consolidations_completed"] == 1
        assert pruned["wake_updates"] == 0 and continued["wake_updates"] == 1
        source = initial["model_state"]["raw_state"]
        raw = pruned["state"]["raw_state"]
        assert source["config"]["width"] == 3
        assert raw["config"] == {**source["config"], "width": 2}
        lineage = continued["state"]["prune_lineage"]
        assert len(lineage) == 1
        assert lineage[0]["source_config"] == source["config"]
        assert lineage[0]["removed_units"] == [0, 1]
        assert promoted["model_state"] == continued["state"]
        assert [d["status"] for d in item["refusals"]] == ["rejected", "failed"]
        assert len(item["refusals"][0]["diagnostics"]) == 2
        assert item["refusals"][1]["diagnostics"] == []
        assert item["rejected_promotion_report"]["decision"]["reasons"] == ["resource"]
        guard_report = item["accepted_promotion_report"]
        assert guard_report["decision"]["accepted"]
        assert excluded <= {tuple(key) for key in guard_report["training_ids"]}
        guard_ids = {
            tuple(key) for key in guard_report["new_guard_keys"] + guard_report["old_guard_keys"]
        }
        assert not guard_ids & excluded
        if item["variant"] != "pc":
            before, after = source["wake_state"], raw["wake_state"]
            assert before["wake_updates"] == after["wake_updates"] == 2
            assert any(value > 0 for value in before["chemical_state"])
            assert after["chemical_state"] == [before["chemical_state"][i] for i in (0, 2)]
            assert after["plasticity_state"] == [before["plasticity_state"][i] for i in (0, 2)]
            assert continued["state"]["raw_state"]["wake_state"]["wake_updates"] == 3
    pc, neutral = report["records"][:2]
    for stage in ("pruned_candidate", "continued_candidate"):
        assert (
            pc[stage]["state"]["raw_state"]["raw_parameters"]
            == neutral[stage]["state"]["raw_state"]["raw_parameters"]
        )


@pytest.mark.parametrize("fault", ["build", "restore"])
def test_should_preserve_all_owners_on_late_preparation_fault(fault: str) -> None:
    actor, runtime, _, prototype = demo.setup_runtime("active_cpc")
    actor.serve(demo.guards()[0].features, now=1)
    runtime.consolidate("pruned", demo.consolidator())
    before_actor, before_candidate = actor.serving_snapshot(), runtime.candidate_snapshot()
    controller = demo.make_controller(actor, prototype)
    calls = 0

    def broken_builder(state: CoupledScalingTopologyState):
        nonlocal calls
        calls += 1
        if calls == 3 and fault == "build":
            raise RuntimeError("injected build fault after guard measurements")
        learner = prototype.build_from_state(state)
        if calls == 3:

            def broken_restore(state: CoupledScalingTopologyState) -> None:
                learner._payload.learner._state.raw_parameters.output_bias[:] = 12.0
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
    runtime.consolidate("pruned", demo.consolidator())
    controller = demo.make_controller(actor, prototype)
    ticket = controller.prepare(runtime, *demo.guards(), now=2, training_ids=demo.selection_ids())
    before = actor.serving_snapshot()
    runtime.consolidate(
        "same",
        lambda state: ConsolidatedState(state, TrainingDiagnostic("same_payload_fixture_v1", 0.0)),
    )
    with pytest.raises(ValueError, match="candidate"):
        controller.commit(runtime, ticket)
    assert demo.state_digest(actor.serving_snapshot()) == demo.state_digest(before)
    controller.discard(ticket)


def test_should_exclude_prune_selection_labels_from_promotion_before_wake() -> None:
    actor, runtime, _, prototype = demo.setup_runtime("pc")
    runtime.consolidate("pruned", demo.consolidator())
    assert runtime.applied_updates == ()
    before = actor.serving_snapshot()
    new, old = demo.guards()
    selected = replace(new, sample_keys=(("selection", "validation-0"),))
    with pytest.raises(ValueError, match="disjoint"):
        demo.make_controller(actor, prototype).prepare(
            runtime, selected, old, now=2, training_ids=demo.selection_ids()
        )
    assert demo.state_digest(actor.serving_snapshot()) == demo.state_digest(before)


def test_should_preserve_managed_retention_refusal_before_transform_or_attempt() -> None:
    actor, runtime, _, _ = demo.setup_runtime("pc")
    before = runtime.candidate_snapshot()

    class Managed:
        def _require_access(self):
            pass

    actor._payload_registry._lifecycle = Managed()
    invoked = []

    def forbidden(state):
        invoked.append(True)
        raise AssertionError("managed consolidation must refuse before transform")

    with pytest.raises(ValueError, match="managed retention"):
        runtime.consolidate("managed", forbidden)
    assert invoked == []
    assert demo.state_digest(runtime.candidate_snapshot()) == demo.state_digest(before)


def test_should_count_native_arrays_without_wrapper_or_heap_bytes() -> None:
    from src.core.coupled_pc_scaling import CoupledScalingConfig, initialize_coupled_scaling
    from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig

    raw = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, circadian=CoupledCircadianConfig()), 23
    )
    state = CoupledScalingTopologyState(1, raw)
    p = raw.raw_parameters
    assert raw.wake_state is not None
    arrays = (
        *p.hidden_weights,
        *p.hidden_biases,
        p.output_weight,
        p.output_bias,
        raw.wake_state.chemical_state,
        raw.wake_state.plasticity_state,
    )
    assert demo.resource_bytes(state) == sum(
        array.size * np.dtype(np.float64).itemsize for array in arrays
    )
