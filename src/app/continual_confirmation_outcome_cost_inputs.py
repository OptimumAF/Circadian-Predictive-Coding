"""Validate the complete original outcome/resource/retention join.

Inputs are decoded complete bodies, already byte-bound by the public app.
Checks retain declared counters, capacities and owned/shared proof links.
No IO, models, datasets, scoring, profiling or current-reader authority.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from src.app.continual_confirmation_json import hash_value, integer, require, same_json
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_matrix_inputs import verify_matrix_report_input
from src.app.continual_confirmation_resource_fields import _work_fields
from src.app.continual_confirmation_retention_checkpoints import _array_proofs, STAGES


IDENTITY_FIELDS = ("family", "seed", "arm")


def cell_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in IDENTITY_FIELDS)


def _scope(report: dict[str, Any], inventory: dict[str, Any], retention: dict[str, Any]) -> None:
    families = fixed_confirmation_manifest().families
    cells = [(f.name, seed, arm) for f in families for seed in f.seeds for arm in f.arms]
    contexts = [(f.name, seed) for f in families for seed in f.seeds]
    for name, rows in (
        ("report", [cell["outcome"] for cell in report["joined_cells"]]),
        ("inventory", inventory["rows"]),
        ("retention", retention["rows"]),
    ):
        same_json([cell_identity(row) for row in rows], cells, f"outcome costs {name} scope")
    for name, rows in (
        ("inventory", inventory["contexts"]),
        ("retention", retention["contexts"]),
        ("work", inventory["work"]["by_seed"]),
    ):
        same_json(
            [(r["family"], r["seed"]) for r in rows], contexts, f"outcome costs {name} contexts"
        )
    same_json(inventory["work"], retention["work"], "outcome costs retained work")
    same_json(inventory["work"], report["cost_reference"]["work"], "outcome costs report work")
    require(inventory["coverage"]["per_arm_timing_or_RSS_measured"] is False, "per-arm scope")
    for source in (inventory, retention):
        require(
            source["original_P6_10_acceptance_complete"] is False, "parent prematurely complete"
        )
        require(
            source["new_measurement_training_or_final_access"] is False, "new scientific access"
        )
        same_json(source["coverage"]["cells"], len(cells), "original coverage cells")
    same_json(inventory["coverage"]["shared_contexts"], len(contexts), "inventory context coverage")
    same_json(
        inventory["coverage"]["checkpoint_capacities"],
        3 * len(cells),
        "inventory checkpoint coverage",
    )
    same_json(retention["coverage"]["contexts"], len(contexts), "retention context coverage")
    same_json(retention["coverage"]["checkpoints"], 3 * len(cells), "retention checkpoint coverage")
    same_json(
        retention["coverage"]["projection_gaps_resolved"],
        sum(
            row["fields"]["owned_replay_array_bytes"]["value"] is None for row in inventory["rows"]
        ),
        "original projection gap coverage",
    )


def _capacity(cost: dict[str, Any], fields: dict[str, Any]) -> None:
    history = fields["parameter_history"]["value"]
    require(type(history) is list and len(history) >= 3, "capacity history incomplete")
    same_json([p["stage"] for p in cost["checkpoints"]], list(STAGES), "cost checkpoint stages")
    for i, checkpoint in enumerate(cost["checkpoints"]):
        stage = checkpoint["stage"]
        same_json(fields[f"parameters_{stage}"]["value"], checkpoint["parameter_count"], "capacity")
        same_json(history[i]["stage"], stage, "capacity history checkpoint stage")
        same_json(history[i]["parameters"], checkpoint["parameter_count"], "capacity history value")
        same_json(history[i]["width"], checkpoint["width"], "capacity history width")
    for point in history:
        same_json(
            integer(point["parameters"], "history parameters", 1),
            4 * integer(point["width"], "history width", 1) + 1,
            "history geometry",
        )
        require(point["status"] in {"measured", "derived"}, "history status")
        require(
            type(point["source_pointer"]) is str and point["source_pointer"].startswith("/"),
            "history pointer",
        )
    peak = max(point["parameters"] for point in history)
    same_json(fields["parameters_peak"]["value"], peak, "full recorded capacity peak")
    if "parameters_peak" in cost["method_facts"]:
        same_json(peak, cost["method_facts"]["parameters_peak"], "original raw capacity peak")


def _owned_checkpoint(point: dict[str, Any], checkpoint: dict[str, Any], pointer: str) -> None:
    for name in ("model_type", "state_sha256", "parameter_sha256"):
        same_json(point[name], checkpoint[name], "owned/original checkpoint " + name)
    hash_value(point["checkpoint_identity"]["sha256"], "owned whole checkpoint")
    integer(point["checkpoint_identity"]["byte_count"], "owned whole checkpoint bytes", 1)
    same_json(point["checkpoint_json_pointer"], pointer, "owned checkpoint pointer")
    same_json(point["retention_json_pointer"], pointer + "/retention", "owned retention pointer")
    fields, arrays = point["owned_state_fields"], point["owned_array_fingerprints"]
    same_json(arrays, _array_proofs(fields), "owned ordered array proof")
    owned = sum(pair["input_array_bytes"] + pair["target_array_bytes"] for pair in arrays)
    same_json(integer(point["owned_array_bytes"], "owned bytes"), owned, "owned array sum")
    require(
        point["owned_array_bytes_status"] == "derived"
        and point["owned_array_bytes_unit"] == "array_bytes",
        "owned byte units/status",
    )
    require(
        "excludes_shared_FIFO_and_process_RSS" in point["owned_array_bytes_scope"],
        "owned byte scope",
    )
    view = point["retention"]
    if view is None:
        require(not arrays and owned == 0, "null retention has owned arrays")
        if not fields:
            same_json(point["retention_status"], "not_applicable", "baseline null status")
        else:
            same_json(
                fields["_replay_memory"]["value"], {"deque": [], "maxlen": 0}, "disabled memory"
            )
            same_json(point["retention_status"], "disabled", "disabled null status")
    else:
        same_json(view["retained_bytes"], owned, "retention view bytes")
        same_json(view["example_count"], len(arrays), "retention view count")
        require(
            len(view["sample_ids"]) == len(set(view["sample_ids"])) == len(arrays),
            "retention view IDs",
        )
        same_json(
            point["retention_status"],
            "retained" if arrays else "configured_empty",
            "configured status",
        )


def _cell(
    joined: dict[str, Any],
    resource: dict[str, Any],
    owned: dict[str, Any],
    index: int,
    context_index: int,
) -> None:
    cost, fields = joined["cost"], resource["fields"]
    same_json(cell_identity(cost), cell_identity(resource), "cost/resource identity")
    same_json(resource["context_index"], context_index, "resource context index")
    expected = _work_fields(cost, index)
    for name, field in expected.items():
        same_json(fields[name], field, "raw original work field " + name)
    _capacity(cost, fields)
    for name in ("wall_time_seconds", "peak_process_rss_bytes", "sleep_guard_duration_seconds"):
        require(
            fields[name]["value"] is None and fields[name]["status"] == "unmeasured",
            "per-arm time/RSS attribution",
        )
    for name in ("sleep_attempts", "guard_prediction_calls", "distinct_applied_replay_examples"):
        integer(fields[name]["value"], name)
    same_json(
        owned["original_inventory_field"],
        fields["owned_replay_array_bytes"],
        "original owned gap field",
    )
    same_json(set(owned["checkpoints"]), set(STAGES), "owned checkpoint stage scope")
    same_json(
        owned["roles_json_pointer"],
        f"/seed_results/{context_index}/roles",
        "owned role context pointer",
    )
    for checkpoint in cost["checkpoints"]:
        stage = checkpoint["stage"]
        _owned_checkpoint(
            owned["checkpoints"][stage],
            checkpoint,
            f"/seed_results/{context_index}/{stage}/{cost['arm']}",
        )
    if fields["owned_replay_array_bytes"]["value"] is not None:
        same_json(
            fields["owned_replay_array_bytes"]["value"],
            owned["checkpoints"]["after_b"]["owned_array_bytes"],
            "original measured owned bytes",
        )


def _stage_storage(
    values: dict[str, Any], rows: list[dict[str, Any]], stage: str
) -> dict[str, int]:
    fifo = values["shared_fifo"]
    shared = integer(fifo["array_bytes"], "shared FIFO array bytes")
    same_json(shared, 24 * integer(fifo["example_count"], "shared FIFO examples"), "FIFO geometry")
    require(
        len(fifo["sample_order_ids"])
        == len(set(fifo["sample_order_ids"]))
        == fifo["example_count"],
        "shared FIFO IDs",
    )
    owner_sum = sum(row["checkpoints"][stage]["owned_array_bytes"] for row in rows)
    expected = {
        "owned_array_bytes": owner_sum,
        "shared_fifo_array_bytes": shared,
        "owned_plus_shared_array_bytes_before_copies": owner_sum + shared,
    }
    for name, value in expected.items():
        same_json(values[name], value, "independent stage/group storage")
    for row in rows:
        view = row["checkpoints"][stage]["retention"]
        if view is not None:
            same_json(
                view["sample_ids"], sorted(fifo["sample_order_ids"]), "owned/shared FIFO samples"
            )
    return expected


def _context_work(
    resource: dict[str, Any],
    final: dict[str, Any],
    rows: list[dict[str, Any]],
    work: dict[str, Any],
) -> None:
    for field, original in (
        ("owned_array_bytes", "owned_retained_array_bytes_before_copies"),
        ("shared_fifo_array_bytes", "shared_fifo_bytes"),
        ("owned_plus_shared_array_bytes_before_copies", "retained_array_bytes_before_copies"),
    ):
        same_json(final[field], resource[original], "retention/inventory group storage")
    same_json(
        final["owned_plus_shared_array_bytes_before_copies"],
        work["retained_array_bytes_before_copies"],
        "retention/raw group work",
    )
    for field in (
        "wake_updates",
        "applied_replay_updates",
        "rejected_executed_replay_updates",
        "executed_optimizer_updates",
    ):
        name = "rejected_replay_updates" if field == "rejected_executed_replay_updates" else field
        same_json(sum(r["fields"][name]["value"] for r in rows), work[field], "per-arm/group work")
    same_json(
        sum(r["fields"]["guard_prediction_calls"]["value"] for r in rows),
        work["guard_evaluations"],
        "per-arm/group guard calls",
    )
    for field, original in (
        ("guard_prediction_calls", "guard_evaluations"),
        ("guarded_attempts", "guarded_attempts"),
        ("guard_examples", "guard_examples"),
    ):
        same_json(resource[field], work[original], "original context guard work")


def _contexts(inventory: dict[str, Any], retention: dict[str, Any]) -> None:
    totals = {
        stage: {
            field: 0
            for field in (
                "owned_array_bytes",
                "shared_fifo_array_bytes",
                "owned_plus_shared_array_bytes_before_copies",
            )
        }
        for stage in STAGES
    }
    for i, (resource, owned, work) in enumerate(
        zip(inventory["contexts"], retention["contexts"], inventory["work"]["by_seed"], strict=True)
    ):
        rows = [
            row
            for row in retention["rows"]
            if (row["family"], row["seed"]) == (owned["family"], owned["seed"])
        ]
        same_json(len(rows), resource["cells"], "context arm count")
        same_json(owned["training_context_json_pointer"], f"/seed_results/{i}", "context pointer")
        same_json(set(owned["stages"]), set(STAGES), "context stages")
        for stage in STAGES:
            for name, value in _stage_storage(owned["stages"][stage], rows, stage).items():
                totals[stage][name] += value
        _context_work(
            resource,
            owned["stages"]["after_b"],
            [r for r in inventory["rows"] if r["context_index"] == i],
            work,
        )
    same_json(totals, retention["stage_totals"], "complete stage storage totals")
    for name, value in inventory["work"]["totals"].items():
        same_json(value, sum(r[name] for r in inventory["work"]["by_seed"]), "complete work total")


def _run_scopes(inventory: dict[str, Any]) -> None:
    runs = inventory["run_resources"]
    require(type(runs) is list and len(runs) == 4, "four original process scopes required")
    same_json(
        [(r["kind"], r["repeat"]) for r in runs],
        [("train", False), ("train", True), ("scored", False), ("scored", True)],
        "original process scope order",
    )
    for run in runs:
        require(run["scope"].endswith("not_per_arm"), "historical process scope")
        for name in ("wall_time", "worker_time"):
            field = run[name]
            require(
                field["unit"] == "seconds" and field["status"] == "measured",
                "historical time unit/status",
            )
            require(
                type(field["value"]) in {int, float}
                and isfinite(field["value"])
                and field["value"] >= 0,
                "historical time value",
            )
        require(
            run["sampled_RSS"]["unit"] == "RSS_bytes_and_seconds"
            and run["sampled_RSS"]["status"] == "measured",
            "historical sampled RSS scope",
        )


def validate_outcome_cost_inputs(
    report: dict[str, Any], inventory: dict[str, Any], retention: dict[str, Any]
) -> None:
    """Validate complete joins, without claiming current IO or reader execution."""
    try:
        verify_matrix_report_input(report)
        _scope(report, inventory, retention)
        indices = {(r["family"], r["seed"]): i for i, r in enumerate(inventory["contexts"])}
        for i, (joined, resource, owned) in enumerate(
            zip(report["joined_cells"], inventory["rows"], retention["rows"], strict=True)
        ):
            _cell(joined, resource, owned, i, indices[(resource["family"], resource["seed"])])
        _contexts(inventory, retention)
        _run_scopes(inventory)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("outcome/resource/retention join is malformed or incomplete") from error
