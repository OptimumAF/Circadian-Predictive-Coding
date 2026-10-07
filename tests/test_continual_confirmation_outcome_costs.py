"""Complete scope fixtures fabricate scores and rekey development metadata.

Rekeyed state/hash/pointer declarations are explicit IO-free scope spies,
not reserved-source facts, actual checkpoint proofs or reader authority.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import test_continual_confirmation_matrix as matrix_tests
import test_continual_confirmation_retention_costs as retention_tests
from src.app import continual_confirmation_outcome_costs as app
from src.app.continual_confirmation_resource_fields import _work_fields
from src.app.continual_confirmation_retention_costs import _derive_retention_costs

fabricated_scored_json = matrix_tests.fabricated_scored_json
original_cost_metadata = matrix_tests.original_cost_metadata
fabricated_report = matrix_tests.fabricated_report
development_facts = retention_tests.development_facts
seal_validation = matrix_tests.report_tests.seal_validation


def _rekey(value: Any, before: str, after: str) -> Any:
    if type(value) is str:
        return value.replace(before, after)
    if type(value) is list:
        return [_rekey(item, before, after) for item in value]
    if type(value) is dict:
        return {key: _rekey(item, before, after) for key, item in value.items()}
    return value


def _fields(cost: dict[str, Any], index: int, template: dict[str, Any]) -> dict[str, Any]:
    fields = deepcopy(template["fields"])
    fields.update(_work_fields(cost, index))
    history = [
        {
            "stage": point["stage"],
            "parameters": point["parameter_count"],
            "width": point["width"],
            "status": "measured",
            "source_pointer": f"/fixture/checkpoints/{i}",
        }
        for i, point in enumerate(cost["checkpoints"])
    ]
    peak = cost["method_facts"].get("parameters_peak", max(p["parameters"] for p in history))
    history.append(
        {
            "stage": "declared_fixture_peak",
            "parameters": peak,
            "width": (peak - 1) // 4,
            "status": "derived",
            "source_pointer": "/fixture/transient_peak",
        }
    )
    fields["parameter_history"]["value"] = history
    fields["parameters_peak"]["value"] = peak
    for point in cost["checkpoints"]:
        fields[f"parameters_{point['stage']}"]["value"] = point["parameter_count"]
    method = cost["method_facts"]
    fields["sleep_attempts"]["value"] = method.get(
        "own_sleep_attempts", method.get("sleep_attempts", int(method.get("sleep") is not None))
    )
    fields["guard_prediction_calls"]["value"] = method.get(
        "guard_evaluations", (method.get("sleep") or {}).get("guard_evaluations", 0)
    )
    fields["owned_replay_array_bytes"]["value"] = method.get(
        "retained_array_bytes", (method.get("retention") or {}).get("retained_bytes")
    )
    return fields


@pytest.fixture(scope="module")
def presentation_inputs(
    fabricated_report: dict[str, Any], development_facts: tuple[Any, Any]
) -> tuple[Any, Any, Any]:
    source_rows, families, dev_inventory = retention_tests.development_inputs(development_facts)
    dev_owned = _derive_retention_costs(source_rows, families, dev_inventory)
    report = deepcopy(fabricated_report)
    inventory, retention = deepcopy(dev_inventory), deepcopy(dev_owned)
    inventory["rows"], inventory["contexts"], retention["rows"], retention["contexts"] = (
        [],
        [],
        [],
        [],
    )
    work = report["cost_reference"]["work"]
    contexts = {(row["family"], row["seed"]): i for i, row in enumerate(work["by_seed"])}
    resource_templates = {(r["family"], r["arm"]): r for r in dev_inventory["rows"]}
    owned_templates = {(r["family"], r["arm"]): r for r in dev_owned["rows"]}
    for i, joined in enumerate(report["joined_cells"]):
        cost = joined["cost"]
        key, context_index = (cost["family"], cost["arm"]), contexts[(cost["family"], cost["seed"])]
        fields = _fields(cost, i, resource_templates[key])
        resource = {
            **{name: cost[name] for name in ("family", "seed", "arm")},
            "context_index": context_index,
            "fields": fields,
        }
        template = owned_templates[key]
        old_prefix = template["roles_json_pointer"].removesuffix("/roles")
        owned = _rekey(deepcopy(template), old_prefix, f"/seed_results/{context_index}")
        owned["seed"], owned["original_inventory_field"] = (
            cost["seed"],
            deepcopy(fields["owned_replay_array_bytes"]),
        )
        for checkpoint in cost["checkpoints"]:
            for name in ("model_type", "state_sha256", "parameter_sha256"):
                owned["checkpoints"][checkpoint["stage"]][name] = checkpoint[name]
        inventory["rows"].append(resource)
        retention["rows"].append(owned)
    for context_index, source_work in enumerate(work["by_seed"]):
        family, seed = source_work["family"], source_work["seed"]
        resource = deepcopy(next(r for r in dev_inventory["contexts"] if r["family"] == family))
        owned = deepcopy(next(r for r in dev_owned["contexts"] if r["family"] == family))
        owned = _rekey(
            owned, owned["training_context_json_pointer"], f"/seed_results/{context_index}"
        )
        resource["seed"], owned["seed"] = seed, seed
        for target, source in (
            ("guard_prediction_calls", "guard_evaluations"),
            ("guarded_attempts", "guarded_attempts"),
            ("guard_examples", "guard_examples"),
        ):
            resource[target] = source_work[source]
        inventory["contexts"].append(resource)
        retention["contexts"].append(owned)
    inventory["work"], retention["work"] = deepcopy(work), deepcopy(work)
    inventory["run_resources"] = [
        {
            "kind": kind,
            "repeat": repeat,
            "scope": "fixture_whole_process_not_per_arm",
            "wall_time": {"value": 1.0, "unit": "seconds", "status": "measured"},
            "worker_time": {"value": 0.5, "unit": "seconds", "status": "measured"},
            "sampled_RSS": {
                "value": {"fixture_only": True},
                "unit": "RSS_bytes_and_seconds",
                "status": "measured",
            },
        }
        for kind, repeat in (("train", False), ("train", True), ("scored", False), ("scored", True))
    ]
    inventory["coverage"].update(cells=560, checkpoint_capacities=1680, shared_contexts=60)
    retention["coverage"].update(
        cells=560,
        checkpoints=1680,
        contexts=60,
        projection_gaps_resolved=sum(
            r["fields"]["owned_replay_array_bytes"]["value"] is None for r in inventory["rows"]
        ),
    )
    retention["stage_totals"] = {
        stage: {
            name: sum(c["stages"][stage][name] for c in retention["contexts"])
            for name in (
                "owned_array_bytes",
                "shared_fifo_array_bytes",
                "owned_plus_shared_array_bytes_before_copies",
            )
        }
        for stage in ("initial", "after_a", "after_b")
    }
    return report, inventory, retention


def test_should_preserve_every_original_declared_outcome_cost_history_and_proof_without_io(
    presentation_inputs: tuple[Any, Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    report, inventory, retention = presentation_inputs
    before = deepcopy(presentation_inputs)
    retention_tests.work_tests._seal_validation(monkeypatch)

    def forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure outcome-cost join entered IO")

    monkeypatch.setattr(Path, "read_bytes", forbid)
    monkeypatch.setattr(Path, "read_text", forbid)
    result = app._derive_outcome_costs(report, inventory, retention)
    assert presentation_inputs == before
    assert result["coverage"]["cells"] == 560 and result["coverage"]["owned_checkpoints"] == 1680
    assert result["coverage"]["cell_metrics"] == 3360
    assert (
        result["coverage"]["metric_vectors"] == 626
        and result["coverage"]["primary_contrast_statements"] == 116
    )
    for i, row in enumerate(result["rows"]):
        assert row["metrics"] == report["joined_cells"][i]["metrics"]
        assert row["original_cost"] == report["joined_cells"][i]["cost"]
        assert row["resource_fields"] == inventory["rows"][i]["fields"]
        assert row["owned_retention"] == retention["rows"][i]
        assert row["source_references"]["report"]["json_pointer"] == f"/joined_cells/{i}"
    assert result["original_report_records"]["analysis"] == report["analysis"]
    assert result["stage_storage_totals"] == retention["stage_totals"]
    assert result["work"] == report["cost_reference"]["work"]
    assert result == app._derive_outcome_costs(*presentation_inputs)
    assert result["original_P6_10_acceptance_complete"] is False


@pytest.mark.parametrize("indices", [[0], [1679], list(range(1680))])
def test_should_preserve_every_failure_raw_endpoint_null_reason_and_original_interval_rule(
    presentation_inputs: tuple[Any, Any, Any],
    fabricated_scored_json: dict[str, Any],
    original_cost_metadata: dict[str, Any],
    indices: list[int],
) -> None:
    from src.app.continual_confirmation_report import build_confirmation_report

    payload = deepcopy(fabricated_scored_json)
    matrix_tests.report_tests.scored_tests._fail_endpoints(
        payload, indices, "nonfinite_predictions"
    )
    report = build_confirmation_report((payload, deepcopy(payload)), original_cost_metadata)
    result = app._derive_outcome_costs(report, presentation_inputs[1], presentation_inputs[2])
    assert len(result["rows"]) == 560
    assert (
        result["original_report_records"]["endpoint_evaluations"] == report["endpoint_evaluations"]
    )
    assert result["original_report_records"]["analysis"] == report["analysis"]
    assert result["coverage"]["failed_cells"] == len({i // 3 for i in indices})
    for index in {i // 3 for i in indices}:
        row = result["rows"][index]
        assert row["outcome"]["failure"] == report["joined_cells"][index]["outcome"]["failure"]
        assert all(m["value"] is None and m["reason"] for m in row["metrics"].values())
        assert row["resource_fields"] == presentation_inputs[1]["rows"][index]["fields"]


@pytest.mark.parametrize("name", ["report", "inventory", "retention"])
def test_should_refuse_any_unbound_whole_public_input(
    presentation_inputs: tuple[Any, Any, Any], name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.app.continual_confirmation_report_costs import canonical_body_identity

    bodies = dict(
        zip(("report", "inventory", "retention"), deepcopy(presentation_inputs), strict=True)
    )
    # Test-only body identities enable independent first/second/third input gates;
    # these declarations grant no original file/source/reader authority.
    monkeypatch.setattr(
        app,
        "INPUT_IDENTITIES",
        {key: canonical_body_identity(value) for key, value in bodies.items()},
    )
    bodies[name]["unexpected_late_value"] = True
    with pytest.raises(ValueError, match="complete original outcome-cost " + name):
        app.build_outcome_cost_presentation(
            *(bodies[key] for key in ("report", "inventory", "retention"))
        )


@pytest.mark.parametrize(
    "change",
    [
        "report_scope",
        "report_metric",
        "report_interval",
        "report_pair",
        "inventory_scope",
        "inventory_seed",
        "inventory_context",
        "work",
        "wake",
        "latent",
        "rejected",
        "capacity",
        "history",
        "peak",
        "per_arm_time",
        "guard",
        "retention_scope",
        "retention_stage",
        "retention_state",
        "retention_parameter",
        "retention_pointer",
        "retention_bytes",
        "retention_null",
        "retention_array",
        "owned_gap",
        "shared_bytes",
        "shared_ids",
        "group_owned",
        "stage_total",
        "run_scope",
        "run_time",
        "run_count",
        "coverage",
    ],
)
def test_should_reject_whole_scope_arithmetic_source_and_storage_corruption(
    presentation_inputs: tuple[Any, Any, Any], change: str
) -> None:
    report, inventory, retention = deepcopy(presentation_inputs)
    fields, point = inventory["rows"][0]["fields"], retention["rows"][0]["checkpoints"]["after_b"]
    if change == "report_scope":
        report["joined_cells"].pop()
    elif change == "report_metric":
        report["joined_cells"][0]["metrics"]["signed_forgetting_a"]["value"] += 0.1
    elif change == "report_interval":
        report["analysis"]["families"][0]["arms"][0]["metrics"][0]["summary"]["mean"] = 0.9
    elif change == "report_pair":
        report["analysis"]["families"][0]["contrasts"].pop()
    elif change == "inventory_scope":
        inventory["rows"].pop()
    elif change == "inventory_seed":
        inventory["rows"][-1]["seed"] = -1
    elif change == "inventory_context":
        inventory["rows"][0]["context_index"] = True
    elif change == "work":
        inventory["work"]["totals"]["executed_optimizer_updates"] += 1
    elif change in {"wake", "latent", "rejected", "guard"}:
        fields[
            {
                "wake": "wake_updates",
                "latent": "optimizer_latent_iterations",
                "rejected": "rejected_replay_updates",
                "guard": "guard_prediction_calls",
            }[change]
        ]["value"] += 1
    elif change == "capacity":
        fields["parameters_initial"]["value"] += 1
    elif change == "history":
        fields["parameter_history"]["value"][0]["width"] += 1
    elif change == "peak":
        fields["parameters_peak"]["value"] += 4
    elif change == "per_arm_time":
        fields["wall_time_seconds"]["value"] = 0
    elif change == "retention_scope":
        retention["rows"].pop()
    elif change == "retention_stage":
        retention["rows"][0]["checkpoints"].pop("initial")
    elif change == "retention_state":
        point["state_sha256"] = "f" * 64
    elif change == "retention_parameter":
        point["parameter_sha256"] = "f" * 64
    elif change == "retention_pointer":
        point["checkpoint_json_pointer"] += "/wrong"
    elif change == "retention_bytes":
        point["owned_array_bytes"] = 1
    elif change == "retention_null":
        point["retention_status"] = "disabled"
    elif change == "retention_array":
        next(
            r for r in retention["rows"] if r["checkpoints"]["after_b"]["owned_array_fingerprints"]
        )["checkpoints"]["after_b"]["owned_array_fingerprints"][0]["input_array_bytes"] += 1
    elif change == "owned_gap":
        retention["rows"][0]["original_inventory_field"]["value"] = 999
    elif change in {"shared_bytes", "shared_ids", "group_owned"}:
        context = next(
            c for c in retention["contexts"] if c["stages"]["after_b"]["shared_fifo_array_bytes"]
        )
        if change == "shared_bytes":
            context["stages"]["after_b"]["shared_fifo"]["array_bytes"] += 1
        elif change == "shared_ids":
            context["stages"]["after_b"]["shared_fifo"]["sample_order_ids"].pop()
        else:
            context["stages"]["after_b"]["owned_array_bytes"] += 1
    elif change == "stage_total":
        retention["stage_totals"]["after_b"]["owned_array_bytes"] += 1
    elif change == "run_scope":
        inventory["run_resources"][0]["scope"] = "per_arm"
    elif change == "run_time":
        inventory["run_resources"][0]["wall_time"]["value"] = float("nan")
    elif change == "run_count":
        inventory["run_resources"].pop()
    else:
        retention["coverage"]["projection_gaps_resolved"] += 1
    with pytest.raises(ValueError):
        app._derive_outcome_costs(report, inventory, retention)
