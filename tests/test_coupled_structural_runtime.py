"""Actual mixed-operation serving transactions, recovery and late faults."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.adapters import coupled_structural_demo as demo
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_structural_state import CoupledStructuralState
from src.core.learner_ports import TrainingDiagnostic


def _run(*args):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_structural_runtime", *args],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert not result.stderr
    assert len(result.stdout) <= 2 * 1024**2
    return json.loads(result.stdout)


def test_should_describe_without_native_work(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description must not initialize or train native state")

    monkeypatch.setattr(demo, "source_state", forbidden)
    assert _run("--describe") == demo.describe_example()


def test_should_run_mixed_history_invalid_inverse_and_complete_serving_rollback():
    report = _run()
    assert (
        report["train_calls"],
        report["guarded_infer_calls"],
        report["coupled_energy_evaluations"],
    ) == (9, 36, 144)
    assert all(report["pc_neutral_parameter_parity"].values())
    assert len(report["records"]) == 3
    excluded = {tuple(key) for key in report["excluded_selection_ids"]}
    for item in report["records"]:
        initial, promoted, restored = item["initial"], item["promoted"], item["rolled_back"]
        assert initial["generation"] == 0 and len(initial["cache"]) == 1
        assert promoted["generation"] == 1 and promoted["cache"] == {}
        assert promoted["actor_version"] == "candidate-0"
        assert promoted["metadata"] == {"route": ["mixed"]}
        assert restored == {**initial, "generation": 2}
        assert item["served"]["generation"] == 1 and not item["served"]["cache_hit"]
        original = initial["model_state"]
        assert original["transitions"] == []
        for stage, kinds, width, beta in (
            ("split_candidate", ["split"], 6, 0.5),
            ("merged_candidate", ["split", "merge"], 3, 1.0),
            ("pruned_candidate", ["split", "merge", "prune"], 2, 1.0),
            ("continued_candidate", ["split", "merge", "prune"], 2, 1.0),
        ):
            snapshot = item[stage]["state"]
            assert snapshot["origin_config"] == original["origin_config"]
            assert [t["kind"] for t in snapshot["transitions"]] == kinds
            assert snapshot["raw_state"]["config"] == {
                **original["raw_state"]["config"],
                "width": width,
                "precisions": [beta, beta],
            }
        pruned, continued = item["pruned_candidate"], item["continued_candidate"]
        assert pruned["consolidation_attempts"] == continued["consolidation_attempts"] == 5
        assert pruned["consolidations_completed"] == continued["consolidations_completed"] == 3
        assert pruned["wake_updates"] == 0 and continued["wake_updates"] == 1
        assert promoted["model_state"] == continued["state"]
        assert [a["status"] for a in item["refusals"]] == ["rejected", "failed"]
        assert item["rejected_promotion_report"]["decision"]["reasons"] == ["resource"]
        guard = item["accepted_promotion_report"]
        assert guard["decision"]["accepted"]
        assert excluded <= {tuple(key) for key in guard["training_ids"]}
        assert not excluded & {
            tuple(key) for key in guard["new_guard_keys"] + guard["old_guard_keys"]
        }
        branch = item["negative_branch"]
        a, b = branch["pruned_candidate"], branch["refused_candidate"]
        assert a["state"] == b["state"]
        assert [t["kind"] for t in b["state"]["transitions"]] == ["split", "prune"]
        assert b["state"]["raw_state"]["config"]["width"] == 5
        assert b["state"]["raw_state"]["config"]["precisions"] == [0.5, 0.5]
        assert (
            a["consolidation_attempts"],
            b["consolidation_attempts"],
            b["consolidations_completed"],
        ) == (1, 2, 1)
        assert branch["actor_before"] == branch["actor_after"]
        assert "configuration" in branch["invalid_inverse_reason"]
        if item["variant"] != "pc":
            source = original["raw_state"]["wake_state"]
            split = item["split_candidate"]["state"]["raw_state"]["wake_state"]
            merged = item["merged_candidate"]["state"]["raw_state"]["wake_state"]
            assert source["wake_updates"] == split["wake_updates"] == merged["wake_updates"] == 2
            assert any(x > 0 for x in source["chemical_state"])
            for field in ("chemical_state", "plasticity_state"):
                assert split[field] == [x for x in source[field] for _ in range(2)]
                assert merged[field] == source[field]
                assert pruned["state"]["raw_state"]["wake_state"][field] == [
                    source[field][i] for i in (0, 2)
                ]
            assert continued["state"]["raw_state"]["wake_state"]["wake_updates"] == 3
    pc, neutral = report["records"][:2]
    assert (
        pc["continued_candidate"]["state"]["raw_state"]["raw_parameters"]
        == neutral["continued_candidate"]["state"]["raw_state"]["raw_parameters"]
    )


@pytest.mark.parametrize("fault", ["build", "restore"])
def test_should_preserve_every_owner_after_late_preparation_fault(fault):
    actor, runtime, _, prototype = demo.setup_runtime("active_cpc")
    runtime.consolidate("split", CoupledScalingSplitConsolidator())
    actor.serve(demo.guards()[0].features, now=1)
    before_actor, before_candidate = actor.serving_snapshot(), runtime.candidate_snapshot()
    controller = demo.make_controller(actor, prototype)
    calls = 0

    def broken_builder(state: CoupledStructuralState):
        nonlocal calls
        calls += 1
        if calls == 3 and fault == "build":
            raise RuntimeError("injected build failure after guard measurements")
        learner = prototype.build_from_state(state)
        restore = learner.restore_state
        if calls == 3:

            def broken_restore(state: CoupledStructuralState):
                private = learner.snapshot_state()
                private.raw_state.raw_parameters.output_bias[:] = 12
                restore(private)
                raise RuntimeError("injected restore failure after private replacement")

            learner.restore_state = broken_restore  # type: ignore[method-assign]
        return learner

    controller._build = controller._evaluator._build = broken_builder
    with pytest.raises(RuntimeError, match="injected"):
        controller.prepare(runtime, *demo.guards(), now=2, training_ids=demo.selection_ids())
    assert calls == 3
    assert demo.state_digest(actor.serving_snapshot()) == demo.state_digest(before_actor)
    assert demo.state_digest(runtime.candidate_snapshot()) == demo.state_digest(before_candidate)
    assert controller._pending == {} and controller._latest is None


def test_should_refuse_stale_ticket_after_same_payload_consolidation():
    actor, runtime, _, prototype = demo.setup_runtime("active_cpc")
    runtime.consolidate("split", CoupledScalingSplitConsolidator())
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


def test_should_exclude_prune_selection_labels_before_any_wake_training():
    actor, runtime, _, prototype = demo.setup_runtime("pc")
    runtime.consolidate("prune", demo.prune_consolidator())
    assert runtime.applied_updates == ()
    before = actor.serving_snapshot()
    new, old = demo.guards()
    selected = replace(new, sample_keys=(("selection", "validation-0"),))
    with pytest.raises(ValueError, match="disjoint"):
        demo.make_controller(actor, prototype).prepare(
            runtime, selected, old, now=2, training_ids=demo.selection_ids()
        )
    assert demo.state_digest(actor.serving_snapshot()) == demo.state_digest(before)


def test_should_refuse_managed_retention_before_transform_and_attempt():
    actor, runtime, _, _ = demo.setup_runtime("pc")
    before = runtime.candidate_snapshot()

    class Managed:
        def _require_access(self):
            pass

    actor._payload_registry._lifecycle = Managed()
    with pytest.raises(ValueError, match="managed retention"):
        runtime.consolidate("split", CoupledScalingSplitConsolidator())
    assert demo.state_digest(runtime.candidate_snapshot()) == demo.state_digest(before)


def test_should_count_native_tensors_without_history_or_python_heap():
    from src.core.coupled_pc_scaling import CoupledScalingConfig, initialize_coupled_scaling
    from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig

    raw = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, circadian=CoupledCircadianConfig()), 23
    )
    state = CoupledStructuralState(1, raw.config, raw)
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
    assert demo.resource_bytes(state) == sum(a.size * np.dtype(np.float64).itemsize for a in arrays)
