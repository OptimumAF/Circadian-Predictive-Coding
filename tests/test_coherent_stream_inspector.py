"""Offline projection and publication of saved component-removal evidence.

Only JSON, source pointers and filesystem publication execute. No models,
learning, predictions, environments, health diagnostics or scientific recompute.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import scripts.build_key_door_inspector as cli
from src.app.coherent_stream_inspector_projection import project_coherent_stream_report
from src.app.key_door_inspector import project_key_door_report
from src.infra import key_door_inspector_files as files
from test_key_door_inspector_files import Page


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/runs/r28-removal-confirmation-20261017"
SOURCE = STAGE / "native-stdout.txt"
RESOURCE = STAGE / "native-resources.json"
ORDER = tuple(
    (policy, method)
    for policy in ("none", "rotating")
    for method in ("bp", "pc", "neutral_cpc", "gated_cpc")
)


def saved():
    return json.loads(SOURCE.read_text(encoding="utf8"))


def resources():
    return json.loads(RESOURCE.read_text(encoding="utf8"))


def section(view, identity):
    return next(item for item in view["sections"] if item["id"] == identity)


def resolve(source, pointer):
    if pointer == "":
        return source
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        source = source[int(token)] if type(source) is list else source[token]
    return source


def all_cells(view):
    result = [view[key] for key in ("protocol_id", "status", "reason")]
    for arm in view["arms"]:
        result.extend([arm["status"], arm["reason"]])
        for row in arm["curves"]:
            result.extend(row["cells"])
    for item in view["sections"]:
        for row in item["rows"]:
            assert len(row["cells"]) == len(item["columns"])
            result.extend(row["cells"])
    return result


def assert_pointers(view):
    for cell in all_cells(view):
        assert set(cell) >= {"value", "pointer", "source", "missing"}
        if cell["missing"]:
            assert cell["value"] is None and cell["pointer"] is None
        elif cell["pointer"] is not None:
            source = view["source"] if cell["source"] == "report" else view["resources"]
            assert cell["value"] == resolve(source, cell["pointer"])
        else:
            assert type(cell["value"]) is str  # Fixed UI labels, never numeric evidence.


def at(view, pointer, origin="report"):
    matches = [
        cell for cell in all_cells(view) if cell["pointer"] == pointer and cell["source"] == origin
    ]
    assert matches, pointer
    return matches[0]


def partial_report():
    report = saved()
    report["status"] = "incomplete"
    report["runs"] = report["runs"][:4]
    report["pairs"] = []
    report["group_facts"] = report["group_facts"][:1]
    for key in ("native_train_calls", "final_score_forwards"):
        report[key] = None
    return report


def test_should_route_the_saved_report_to_all_eight_arms_and_four_source_exact_pairs():
    source = saved()
    before = deepcopy(source)
    direct = project_coherent_stream_report(source, source_label=SOURCE.name)
    routed = project_key_door_report(source, source_label=SOURCE.name)
    assert direct == routed
    assert direct["schema_id"] == "coherent_stream_inspector_v1"
    assert direct["source"] == source == before
    assert direct["status"]["value"] == "complete"
    assert [(arm["policy"], arm["method"]) for arm in direct["arms"]] == list(ORDER)
    assert [arm["key"] for arm in direct["arms"]] == [
        f"{policy}:{method}" for policy, method in ORDER
    ]
    assert [item["id"] for item in direct["sections"]] == [
        "outcomes",
        "pairs",
        "work",
        "planned",
        "controls",
    ]
    assert len(section(direct, "outcomes")["rows"]) == 8
    assert len(section(direct, "pairs")["rows"]) == 4
    for index, arm in enumerate(direct["arms"]):
        assert len(arm["curves"]) == 8
        for point, row in enumerate(arm["curves"]):
            checkpoint, context = (0, 6, 12, 18)[point // 2], ("A", "B")[point % 2]
            assert row["checkpoint"] == checkpoint and row["context"] == context
            assert row["cells"][0]["value"] == checkpoint
            assert row["cells"][1]["value"] == context
            assert (
                row["cells"][2]["pointer"]
                == f"/runs/{index}/stream_result/checkpoints/{point // 2}/scores/{context}/bce"
            )
            assert (
                row["cells"][3]["pointer"]
                == f"/runs/{index}/stream_result/checkpoints/{point // 2}/scores/{context}/accuracy"
            )
    assert_pointers(direct)


def test_should_preserve_negative_losses_and_the_negative_bp_return_difference():
    view = project_coherent_stream_report(saved(), source_label=SOURCE.name)
    assert at(view, "/pairs/0/rotating_minus_none/return_gain")["value"] == -0.0037670547576639235
    assert at(view, "/pairs/0/rotating_minus_none/retention_loss")["value"] < 0
    assert at(view, "/runs/0/outcomes/retention_loss")["value"] < 0
    assert at(view, "/scientific_advantage_verified")["value"] is False
    assert at(view, "/full_R28_complete")["value"] is False
    assert_pointers(view)


def test_should_separate_measured_work_from_planned_counts_and_keep_null_facts():
    view = project_coherent_stream_report(saved(), source_label=SOURCE.name)
    assert at(view, "/wake_updates")["value"] == 144
    assert at(view, "/native_train_calls")["value"] == 192
    assert at(view, "/row_exposures_charged")["value"] == 1200
    assert at(view, "/group_facts/0/shared_replay_plans_exact")["value"] is None
    assert at(view, "/group_facts/0/shared_replay_plans_exact")["missing"] is False
    for row in section(view, "work")["rows"]:
        for cell in row["cells"]:
            assert not cell["pointer"] or not cell["pointer"].startswith("/protocol/")
    assert any(
        cell["pointer"] and cell["pointer"].startswith("/protocol/planned_work/")
        for row in section(view, "planned")["rows"]
        for cell in row["cells"]
    )


def test_should_distinguish_recorded_null_reasons_from_missing_scope_metadata():
    view = project_coherent_stream_report(saved(), source_label=SOURCE.name)
    assert view["reason"]["missing"] is True and view["reason"]["pointer"] is None
    arm_reason = view["arms"][0]["reason"]
    assert arm_reason["value"] is None and arm_reason["missing"] is False
    assert arm_reason["pointer"] == "/runs/0/reason"
    assert any(
        cell["missing"] for row in section(view, "controls")["rows"] for cell in row["cells"]
    )


def test_should_detach_source_resources_and_each_projected_cell():
    source, costs = saved(), resources()
    expected, expected_costs = deepcopy(source), deepcopy(costs)
    view = project_coherent_stream_report(
        source, source_label="source.json", resources=costs, resource_label="costs.json"
    )
    source["runs"][0]["stream_result"]["checkpoints"][0]["scores"]["A"]["bce"] = 999
    costs["status"] = "failed"
    assert view["source"] == expected and view["resources"] == expected_costs
    cell = at(view, "/runs/0/stream_result/checkpoints/0/scores/A/bce")
    value = cell["value"]
    view["source"]["runs"][0]["stream_result"]["checkpoints"][0]["scores"]["A"]["bce"] = 888
    assert cell["value"] == value
    cell["value"] = 777
    assert view["source"]["runs"][0]["stream_result"]["checkpoints"][0]["scores"]["A"]["bce"] == 888


def test_should_preserve_missing_arm_and_pair_slots_without_fabricating_scores():
    source = partial_report()
    view = project_coherent_stream_report(source, source_label="synthetic-unreached.json")
    assert len(view["arms"]) == 8 and len(section(view, "pairs")["rows"]) == 4
    for arm in view["arms"][4:]:
        assert arm["status"]["missing"] is True
        assert len(arm["curves"]) == 8
        assert all(cell["missing"] for row in arm["curves"] for cell in row["cells"][2:])
    for row in section(view, "pairs")["rows"]:
        assert all(cell["missing"] for cell in row["cells"][1:])
    null = at(view, "/native_train_calls")
    assert null["value"] is None and null["missing"] is False
    assert_pointers(view)


def test_should_leave_omitted_checkpoint_metrics_as_gaps_in_fixed_curves():
    source = saved()
    source["status"] = "incomplete"
    source["runs"][0].update(status="failed", reason="synthetic omitted score", outcomes=None)
    source["runs"][0]["stream_result"]["checkpoints"] = []
    source["pairs"] = []
    source["group_facts"] = []
    view = project_coherent_stream_report(source, source_label="synthetic-gap.json")
    arm = view["arms"][0]
    assert arm["status"]["value"] == "failed"
    assert len(arm["curves"]) == 8
    assert all(cell["missing"] for row in arm["curves"] for cell in row["cells"][2:])
    assert_pointers(view)


def test_should_retain_one_paid_replay_failure_and_unknown_unreleased_scores():
    source = saved()
    source["status"] = "incomplete"
    arm = source["runs"][4]
    arm.update(
        status="failed",
        reason="synthetic paid replay failure",
        wake_updates=3,
        replay_charged_examples=1,
        outcomes=None,
    )
    arm.pop("stream_result")
    arm["rehearsal"] = dict(
        charged_exposures=1,
        attempted_updates=1,
        returned_updates=0,
        committed_updates=0,
        stopped=True,
    )
    source["pairs"] = []
    source["group_facts"] = source["group_facts"][:1]
    source.update(
        wake_updates=129,
        replay_charged_examples=37,
        replay_attempted_updates=37,
        replay_returned_updates=36,
        replay_committed_updates=36,
        total_committed_updates=165,
        row_exposures_charged=1069,
        known_native_train_calls=166,
        known_completed_score_forwards=56,
        native_train_calls=None,
        final_score_forwards=None,
    )
    view = project_coherent_stream_report(source, source_label="synthetic-paid-failure.json")
    assert at(view, "/runs/4/replay_charged_examples")["value"] == 1
    assert at(view, "/runs/4/rehearsal/attempted_updates")["value"] == 1
    assert at(view, "/runs/4/rehearsal/returned_updates")["value"] == 0
    assert view["arms"][4]["reason"]["value"] == "synthetic paid replay failure"
    assert all(cell["missing"] for row in view["arms"][4]["curves"] for cell in row["cells"][2:])
    assert at(view, "/native_train_calls")["value"] is None
    assert_pointers(view)


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/protocol/protocol_id", "unknown_protocol"),
        ("/protocol/seeds/0", 24),
        ("/runs/0/seed", True),
        ("/runs/0/method", "invented"),
        ("/runs/0/policy", "recent_only"),
        ("/runs/0/status", None),
        ("/status", None),
        ("/runs/0/stream_result/checkpoints/0/scores/A/accuracy", 1.1),
        ("/runs/0/stream_result/checkpoints/0/scores/B/bce", -1),
        ("/runs/0/stream_result/checkpoints/0/scores/A/bce", True),
        ("/runs/0/stream_result/checkpoints/0/scores/A/bce", float("nan")),
        ("/runs/0/stream_result/checkpoints/0/scores/A/bce", float("inf")),
        ("/wake_updates", True),
        ("/native_train_calls", -1),
        ("/group_facts/0/pc_neutral_exact", False),
        ("/group_facts/0/shared_replay_plans_exact", True),
        ("/pairs/0/initial_primary_levels_exact", False),
        ("/runs/0/stream_result/completed_updates", 0),
        ("/runs/0/stream_result/score_forward_calls", 0),
        ("/runs/0/stream_result/training_examples", 0),
        ("/runs/4/rehearsal/attempted_updates", 0),
        ("/runs/4/rehearsal/returned_updates", 0),
        ("/runs/4/rehearsal/committed_updates", 0),
        ("/runs/4/rehearsal/charged_exposures", 0),
        ("/runs/0/stream_result/checkpoints/0/updates", False),
        ("/protocol/training_rng", 7301),
        ("/protocol/final_rng", 9101),
        ("/runs/0/elapsed_seconds", -1.0),
        ("/total_seconds", -1.0),
        ("/observed_process_rss/peak_bytes", -1),
        ("/protocol/resources/wall_seconds", -1.0),
        ("/native_train_calls", 0),
        ("/wake_updates", 0),
        ("/final_score_forwards", 0),
    ],
)
def test_should_refuse_malformed_or_contradictory_recorded_claims(pointer, value):
    source = saved()
    parent, token = pointer.rsplit("/", 1)
    node = resolve(source, parent)
    node[int(token) if type(node) is list else token] = value
    with pytest.raises(ValueError):
        project_coherent_stream_report(source, source_label="invalid.json")


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_arm",
        "reordered_arms",
        "duplicate_pair",
        "missing_arm",
        "missing_pair",
        "missing_checkpoint",
        "duplicate_checkpoint",
    ],
)
def test_should_refuse_complete_claims_with_missing_duplicate_or_unordered_evidence(fault):
    source = saved()
    if fault == "duplicate_arm":
        source["runs"][1] = deepcopy(source["runs"][0])
    elif fault == "reordered_arms":
        source["runs"].reverse()
    elif fault == "duplicate_pair":
        source["pairs"][1] = deepcopy(source["pairs"][0])
    elif fault == "missing_arm":
        source["runs"].pop()
    elif fault == "missing_pair":
        source["pairs"].pop()
    elif fault == "missing_checkpoint":
        source["runs"][0]["stream_result"]["checkpoints"].pop()
    else:
        source["runs"][0]["stream_result"]["checkpoints"][1] = deepcopy(
            source["runs"][0]["stream_result"]["checkpoints"][0]
        )
    with pytest.raises(ValueError):
        project_coherent_stream_report(source, source_label="false-complete.json")


def test_should_refuse_a_complete_claim_with_all_measured_global_counts_zero():
    source = saved()
    for field in (
        "wake_updates",
        "replay_charged_examples",
        "replay_attempted_updates",
        "replay_returned_updates",
        "replay_committed_updates",
        "total_committed_updates",
        "row_exposures_charged",
        "known_native_train_calls",
        "known_completed_score_forwards",
        "native_train_calls",
        "final_score_forwards",
    ):
        source[field] = 0
    with pytest.raises(ValueError):
        project_coherent_stream_report(source, source_label="zero-cost-complete.json")


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/seconds", -1.0),
        ("/physical_work/train/attempted_calls", -1),
        ("/physical_work/train/attempted_calls", True),
    ],
)
def test_should_refuse_negative_or_boolean_separate_resource_measurements(pointer, value):
    costs = resources()
    parent, key = pointer.rsplit("/", 1)
    resolve(costs, parent)[key] = value
    with pytest.raises(ValueError):
        project_coherent_stream_report(
            saved(),
            source_label=SOURCE.name,
            resources=costs,
            resource_label="invalid-resources.json",
        )


def test_should_display_separate_resource_values_with_unverified_association():
    costs = resources()
    view = project_coherent_stream_report(
        saved(), source_label=SOURCE.name, resources=costs, resource_label=RESOURCE.name
    )
    assert view["resources"] == costs
    assert "UNVERIFIED" in section(view, "work")["note"]
    physical = [
        cell
        for cell in all_cells(view)
        if cell["source"] == "resources"
        and cell["pointer"]
        and cell["pointer"].startswith("/physical_work/")
    ]
    assert physical and all(not cell["missing"] for cell in physical)
    assert_pointers(view)


def test_should_publish_through_shared_file_and_cli_boundaries_without_source_writes(
    tmp_path, monkeypatch, capsys
):
    before = SOURCE.read_bytes(), RESOURCE.read_bytes()
    output = tmp_path / "coherent.html"
    files.write_key_door_inspector(SOURCE, output, resource_artifact=RESOURCE)
    view = Page(output.read_text(encoding="utf8")).payload()
    assert view["schema_id"] == "coherent_stream_inspector_v1"
    assert view["source"] == json.loads(before[0]) and view["resources"] == json.loads(before[1])
    assert_pointers(view)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(SOURCE, output)
    assert output.read_bytes() == original
    cli_output = tmp_path / "cli.html"
    monkeypatch.setattr(
        sys, "argv", ["builder", "--source-artifact", str(SOURCE), "--output", str(cli_output)]
    )
    cli.main()
    response = json.loads(capsys.readouterr().out)
    assert response["output"] == str(cli_output.resolve())
    assert response["bytes"] == cli_output.stat().st_size
    assert Page(cli_output.read_text(encoding="utf8")).payload()["source"] == json.loads(before[0])
    assert (SOURCE.read_bytes(), RESOURCE.read_bytes()) == before


def test_should_publish_untrusted_failure_text_only_as_inert_json(tmp_path):
    source = partial_report()
    danger = '</script><img id="injected" src="https://example.invalid" onerror="alert(1)"> &'
    source["global_errors"] = [danger]
    source["runs"][0].update(status="failed", reason=danger)
    path, output = tmp_path / "untrusted.json", tmp_path / "safe.html"
    path.write_text(json.dumps(source), encoding="utf8")
    files.write_key_door_inspector(path, output)
    page = Page(output.read_text(encoding="utf8"))
    assert page.payload()["source"]["global_errors"] == [danger]
    assert "<" not in "".join(page.payload_parts)
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert all(
        not (attrs.get(key) or "").lower().startswith(("http:", "https:", "//"))
        for _, attrs in page.tags
        for key in ("src", "href")
    )


def test_should_keep_the_existing_eight_mebibyte_input_limit_before_publication(tmp_path):
    path, output = tmp_path / "oversize.json", tmp_path / "new" / "output.html"
    path.write_bytes(b" " * (files.MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="8 MiB"):
        files.write_key_door_inspector(path, output)
    assert not output.parent.exists()
