"""Literal saved diagnosis presentation; no model, diagnosis or archived inputs."""

from copy import deepcopy
import json
from typing import Any

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.app.ticket_workflow_inspector_projection import project_ticket_workflow_report
from src.infra.key_door_inspector_files import render_key_door_inspector, write_key_door_inspector
from test_taskspace_memory_inspector import assert_sources, at, section

FAMILY = "ticket_live_saved_score_diagnosis_v1"
CONCEPTS = ("invoice", "parcel", "legacy", "password")


def literal():
    report: dict[str, Any] = dict(
        definition=FAMILY,
        complete=True,
        role="saved_candidate_scores_only",
        qualification="Saved scalar scores, not calibrated confidence or noise benefit.",
        source_scope="captured_bytes_only_no_postflight_seal",
        source_sha256="a" * 64,
        work=dict(saved_decisions=8, saved_scores=16, paired_decisions=4, margin_subtractions=18),
        arms=[],
        concepts=[],
    )
    for index, name in enumerate(("clean", "retained_corruption")):
        report["arms"].append(
            dict(
                name=name,
                recipe=dict(retained_invoice_noise=bool(index)),
                corruption=dict(role="train", rows_per_batch=[7] if index else []),
                outcomes=dict(new_before=1, new_after=1 + index, old_before=2, old_after=2),
                recorded_training=dict(
                    attempted_updates=5,
                    returned_updates=5,
                    attempted_rows=60,
                    unreturned_work_unknown=False,
                ),
                traces=[
                    dict(
                        concept=c,
                        success=c != "legacy" or bool(index),
                        decisions=[
                            dict(
                                attempt=1,
                                selected_id="defer" if c == "legacy" and index else "inspect",
                                selected_raw_margin=0.01 if c == "password" and index else 0.25,
                                selected_probability_margin=0.0,
                                scores=[
                                    dict(candidate_id="inspect", raw_score=-0.2, probability=0.5),
                                    dict(candidate_id="defer", raw_score=0.0, probability=0.5),
                                ],
                            )
                        ],
                    )
                    for c in CONCEPTS
                ],
            )
        )
    for c in CONCEPTS:
        report["concepts"].append(
            dict(
                concept=c,
                paired_decisions=1,
                unpaired_steps=dict(clean=0, retained_corruption=0),
                first_divergence=None
                if c != "legacy"
                else dict(
                    attempt=1,
                    input=dict(
                        request=dict(query="same ready query"),
                        observation_state=dict(revision=1),
                        capabilities_observed={},
                    ),
                    clean_selected="inspect",
                    retained_corruption_selected="defer",
                    noise_action_minus_clean_raw=dict(clean=-0.05, retained_corruption=0.18),
                ),
            )
        )
    return report


def test_should_route_detached_observations_without_recomputing_recorded_values():
    report = literal()
    original = deepcopy(report)
    view = project_key_door_report(report, source_label="literal.json")
    assert view == project_ticket_workflow_report(report, source_label="literal.json")
    assert view["schema_id"] == "ticket_workflow_inspector_v1"
    assert view["source"] == original == report
    assert len(section(view, "margins")["rows"]) == 8
    assert len(section(view, "scores")["rows"]) == 16
    assert (
        at(view, "/concepts/2/first_divergence/noise_action_minus_clean_raw/clean")["value"]
        == -0.05
    )
    assert at(view, "/arms/1/traces/3/decisions/0/selected_raw_margin")["value"] == 0.01
    assert at(view, "/arms/0/traces/2/success")["value"] is False
    assert_sources(view)
    report["arms"][0]["traces"][0]["decisions"][0]["scores"][0]["raw_score"] = 99
    assert view["source"] == original
    at(view, "/arms/0/recipe")["value"]["retained_invoice_noise"] = True
    assert view["source"]["arms"][0]["recipe"]["retained_invoice_noise"] is False


def test_should_preserve_unknown_null_zero_false_and_partial_paid_work():
    report = dict(
        definition=FAMILY,
        complete=False,
        error="late capture failed",
        work=dict(saved_decisions=0),
        arms=[
            dict(
                name="clean",
                traces=[
                    dict(
                        concept="invoice",
                        success=False,
                        decisions=[
                            dict(
                                attempt=1, selected_raw_margin=None, selected_probability_margin=0.0
                            )
                        ],
                    )
                ],
            )
        ],
    )
    view = project_key_door_report(report, source_label="failed.json")
    assert at(view, "/arms/0/traces/0/decisions/0/selected_raw_margin")["value"] is None
    assert at(view, "/arms/0/traces/0/decisions/0/selected_probability_margin")["value"] == 0.0
    assert section(view, "differences")["rows"] == []
    outcome = section(view, "outcomes")["rows"][0]["cells"]
    assert any(c["missing"] for c in outcome)
    assert_sources(view)


@pytest.mark.parametrize(
    "fault",
    [
        "family",
        "completion",
        "role",
        "arms",
        "arm_order",
        "concepts",
        "duplicate_concept",
        "traces",
        "decisions",
        "scores",
        "nan",
        "boolean_margin",
        "probability",
        "attempt",
        "success",
        "counter",
        "training_unknown",
        "divergence",
        "input",
        "same_actions",
        "gap",
        "outcomes",
        "metadata",
        "qualification",
        "label",
    ],
)
def test_should_refuse_malformed_or_unbounded_presentation(fault):
    report = literal()
    decision = report["arms"][0]["traces"][0]["decisions"][0]
    divergence = report["concepts"][2]["first_divergence"]
    if fault == "family":
        report["definition"] = "other"
    elif fault == "completion":
        report["complete"] = 1
    elif fault == "role":
        report["role"] = "final_test"
    elif fault == "arms":
        report["arms"] *= 2
    elif fault == "arm_order":
        report["arms"].reverse()
    elif fault == "concepts":
        report["concepts"].pop()
    elif fault == "duplicate_concept":
        report["concepts"][1]["concept"] = "invoice"
    elif fault == "traces":
        report["arms"][0]["traces"] *= 2
    elif fault == "decisions":
        report["arms"][0]["traces"][0]["decisions"] *= 7
    elif fault == "scores":
        decision["scores"] *= 5
    elif fault == "nan":
        decision["selected_raw_margin"] = float("nan")
    elif fault == "boolean_margin":
        decision["selected_raw_margin"] = True
    elif fault == "probability":
        decision["scores"][0]["probability"] = 1.1
    elif fault == "attempt":
        decision["attempt"] = True
    elif fault == "success":
        report["arms"][0]["traces"][0]["success"] = 1
    elif fault == "counter":
        report["work"]["saved_scores"] = -1
    elif fault == "training_unknown":
        report["arms"][0]["recorded_training"]["unreturned_work_unknown"] = 0
    elif fault == "divergence":
        report["concepts"][2]["first_divergence"] = []
    elif fault == "input":
        divergence["input"] = "replaced"
    elif fault == "same_actions":
        divergence["retained_corruption_selected"] = "inspect"
    elif fault == "gap":
        divergence["noise_action_minus_clean_raw"]["clean"] = False
    elif fault == "outcomes":
        report["arms"][0]["outcomes"]["old_after"] = True
    elif fault == "metadata":
        report["arms"][0]["corruption"] = []
    elif fault == "qualification":
        report["qualification"] = 0
    with pytest.raises(ValueError):
        project_key_door_report(report, source_label=" " if fault == "label" else "literal")


def test_should_preserve_complete_no_difference_and_nullable_single_action_margins():
    report = literal()
    report["concepts"][2]["first_divergence"] = None
    decision = report["arms"][0]["traces"][0]["decisions"][0]
    decision.update(selected_raw_margin=None, selected_probability_margin=None)
    decision["scores"].pop()
    view = project_key_door_report(report, source_label="literal")
    assert at(view, "/concepts/2/first_divergence")["value"] is None
    assert at(view, "/arms/0/traces/0/decisions/0/selected_raw_margin")["value"] is None
    assert_sources(view)


def test_should_keep_separate_resources_unverified_and_pointer_escaped():
    resources = {"cost/~": {"bytes": 0}}
    view = project_key_door_report(
        literal(), source_label="literal", resources=resources, resource_label="separate"
    )
    assert at(view, "/cost~1~0", "resources")["value"] == {"bytes": 0}
    assert "UNVERIFIED" in section(view, "resources")["title"]
    resources["cost/~"]["bytes"] = 42
    assert view["resources"]["cost/~"]["bytes"] == 0
    assert_sources(view)
    with pytest.raises(ValueError):
        project_key_door_report(literal(), source_label="literal", resources={})


def test_should_export_inert_source_and_refuse_occupied_output(tmp_path):
    report = literal()
    report["qualification"] = "</script><script>window.pwned=1</script>&"
    source = tmp_path / "saved.json"
    source.write_text(json.dumps(report))
    output = tmp_path / "inspector.html"
    write_key_door_inspector(source, output)
    html = output.read_text(encoding="utf8")
    assert report["qualification"] not in html
    embedded = html.split('<script id="inspector-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    projection = json.loads(embedded)
    assert projection["source"] == report
    assert_sources(projection)
    assert "Saved decision comparison" in html and "calibrated confidence" in html
    assert "data.source.definition==='ticket_live_saved_score_diagnosis_v1'" in html
    assert html.encode("utf8") == render_key_door_inspector(projection)
    with pytest.raises(FileExistsError):
        write_key_door_inspector(source, output)
    assert output.read_text(encoding="utf8") == html


@pytest.mark.parametrize("failure", ["parse", "serialize", "size"])
def test_should_route_tagged_failed_new_mode_cli_summaries(tmp_path, monkeypatch, failure):
    from scripts import diagnose_ticket_training as cli
    from src.app import ticket_live_noise_diagnosis as diagnosis

    # Stub the diagnosed result; these presentation checks must not rescore.
    result = dict(complete=True, value=float("nan") if failure == "serialize" else "x" * 300000)
    monkeypatch.setattr(diagnosis, "diagnose_live_noise_pair", lambda *a, **kw: result)
    source = tmp_path / "input.json"
    source.write_text("{" if failure == "parse" else "{}")
    output = tmp_path / "failed.json"
    assert (
        cli.main(["--live-noise-comparison", "--input", str(source), "--output", str(output)]) == 1
    )
    report = json.loads(output.read_text())
    assert report["definition"] == FAMILY and report["complete"] is False
    view = project_key_door_report(report, source_label="failed")
    assert view["status"]["value"] is False and section(view, "outcomes")["rows"] == []
    assert_sources(view)
