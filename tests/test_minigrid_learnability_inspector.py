"""Saved partial F/A studies remain inspectable without running a learner."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.infra.key_door_inspector_files import (
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    write_key_door_inspector,
)
from test_key_door_inspector_files import Page
from test_minigrid_inspector_projection import pointer

STAGE = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/original-wide-probe-games-study-20261018"
)
METHODS = ("bp", "pc", "neutral-cpc", "gated-cpc")


@pytest.fixture(scope="module")
def saved():
    return {
        name: json.loads((STAGE / f"native-{name}.txt").read_text(encoding="utf8"))
        for name in METHODS
    }


@pytest.mark.parametrize(
    "method,complete,failed,unrun",
    [
        ("bp", 14, 1, 1),
        ("pc", 11, 1, 4),
        ("neutral-cpc", 3, 1, 12),
        ("gated-cpc", 3, 1, 12),
    ],
)
def test_should_display_all_planned_games_with_exact_saved_or_explicit_unrun_state(
    saved, method, complete, failed, unrun
):
    report = saved[method]
    before = deepcopy(report)
    projected = project_key_door_report(report, source_label=f"native-{method}.txt")
    assert projected["schema_id"] == "minigrid_learnability_inspector_v1"
    assert projected["source"] == before == report
    assert projected["source"] is not report
    assert report["status"] == "incomplete" and report["scientific_advantage_verified"] is False
    condition = projected["conditions"][0]
    assert condition["key"] == report["methods"][0]["method"]
    episodes = condition["episodes"]
    assert len(episodes) == 16
    assert sum(e["record"]["status"] == "complete" for e in episodes) == complete
    assert sum(e["record"]["status"] == "failed" for e in episodes) == failed
    assert sum(e["record"]["status"] == "not_run" for e in episodes) == unrun
    for e in episodes:
        assert e["context"] in ("A", "F-blue", "F-red")
        if e["pointer"] is None:
            assert e["derived"] is True and e["frames"] == []
            assert e["record"]["success"] is e["record"]["return"] is None
        else:
            assert e["record"] == pointer(projected["source"], e["pointer"])
            for frame in e["frames"]:
                assert frame["observation"] == pointer(projected["source"], frame["pointer"])
                if frame["features_pointer"]:
                    features = pointer(projected["source"], frame["features_pointer"])
                    assert features[148:] == frame["observation"]
    curves = condition["curves"]
    phases = {c["phase"] for c in curves}
    assert curves and phases >= {"benchmark", "F learning"}
    # The controls stopped before any A return. No curve may be invented for it.
    assert "A return" in phases
    assert any(c["phase"] == "A return" and c["status"] == "complete" for c in curves) == (
        method in ("bp", "pc")
    )
    for c in curves:
        if c["pointer"] is None:
            assert c["status"] == "not_run" and c["derived"] is True
            assert c["ce"] is c["accuracy"] is c["exact_sequence_accuracy"] is None
    for c in curves:
        if c["pointer"] is not None:
            assert type(pointer(projected["source"], c["pointer"])) is dict
        assert c["accuracy"] is None or 0 <= c["accuracy"] <= 1


def test_should_preserve_negative_success_and_unproven_budget_evidence(saved):
    p = project_key_door_report(saved["bp"], source_label="bp")
    condition = p["conditions"][0]
    games = [
        e
        for e in condition["episodes"]
        if e["source_updates"] == 128 and e["phase"] == "after_probe"
    ]
    assert [e["record"]["success"] for e in games] == [False, False, False, True]
    assert games[2]["record"]["return"] == -1
    assert games[3]["record"]["return"] == pytest.approx(0.49375)
    false_proof = [
        v
        for v in condition["proofs"]
        if v["pointer"] == "/methods/0/probes/1/return_budget_binding_preserved"
    ]
    assert len(false_proof) == 1 and false_proof[0]["value"] is False
    costs = condition["costs"]
    assert any(
        c["value"] is None and c["pointer"].endswith("/live_prediction_calls_returned")
        for c in costs
    )
    assert any(
        "known_returned_work_lower_bounds" in c["pointer"] and c["value"] == 224 for c in costs
    )


@pytest.mark.parametrize(
    "kind",
    ["protocol", "hidden", "final", "role", "history", "action", "pixel", "transition", "success"],
)
def test_should_refuse_relabelled_or_corrupted_evidence_before_publication(saved, tmp_path, kind):
    r = deepcopy(saved["bp"])
    e = r["methods"][0]["probes"][0]["probe_rollouts"][0]["report"]["episodes"][0]
    if kind == "protocol":
        r["protocol"]["probe"]["updates"] = 15
    elif kind == "hidden":
        r["protocol"]["hidden_dim"] = 8
    elif kind == "final":
        r["methods"][0]["final_evaluation"]["status"] = "complete"
    elif kind == "role":
        e["source_role"] = "final_test"
    elif kind == "history":
        e["predictions"][0]["features"][0] = 0.2
    elif kind == "action":
        e["predictions"][0]["action"] = (e["predictions"][0]["action"] + 1) % 7
    elif kind == "pixel":
        e["predictions"][0]["observation"][0] = 0.123
    elif kind == "transition":
        e["environment"]["transitions"][0]["action"] = 6
    else:
        e["success"] = not e["success"]
    source = tmp_path / "corrupt.json"
    source.write_text(json.dumps(r), encoding="utf8")
    destination = tmp_path / "uncreated" / "bad.html"
    with pytest.raises(ValueError):
        write_key_door_inspector(source, destination)
    assert not destination.parent.exists()


@pytest.mark.parametrize("method", METHODS)
def test_should_publish_existing_saved_report_exclusively_without_mutating_source(
    saved, tmp_path, method
):
    source = STAGE / f"native-{method}.txt"
    before = source.read_bytes()
    destination = tmp_path / "view.html"
    write_key_door_inspector(source, destination)
    assert source.read_bytes() == before
    assert destination.stat().st_size <= MAX_OUTPUT_BYTES
    page = Page(destination.read_text(encoding="utf8"))
    payload = page.payload()
    assert payload["source"] == saved[method]
    assert payload["schema_id"] == "minigrid_learnability_inspector_v1"
    for _, attributes in page.tags:
        assert not any(
            (attributes.get(k) or "").startswith(("https:", "http:", "//")) for k in ("src", "href")
        )
    if method == "bp":
        assert len(before) > MAX_INPUT_BYTES
    with pytest.raises(FileExistsError):
        write_key_door_inspector(source, destination)


def test_should_keep_original_input_limit_for_other_protocols(tmp_path):
    source = tmp_path / "too-large.json"
    source.write_text(
        json.dumps({"protocol": {"protocol_id": "other"}}) + " " * MAX_INPUT_BYTES, encoding="utf8"
    )
    with pytest.raises(ValueError, match="8 MiB"):
        write_key_door_inspector(source, tmp_path / "no.html")


def test_should_embed_saved_text_inertly_and_keep_partial_values_unknown(saved, tmp_path):
    r = deepcopy(saved["neutral-cpc"])
    dangerous = '</script><img id="injected" src="x" onerror="alert(1)"> & < >'
    r["recorded_note"] = dangerous
    source, destination = tmp_path / "partial.json", tmp_path / "safe.html"
    source.write_text(json.dumps(r), encoding="utf8")
    write_key_door_inspector(source, destination)
    page = Page(destination.read_text(encoding="utf8"))
    assert page.payload()["source"]["recorded_note"] == dangerous
    assert not any(attrs.get("id") == "injected" for _, attrs in page.tags)
    assert "<" not in "".join(page.payload_parts)
    p = page.payload()
    assert p["source"]["work"]["live_prediction_forwards_completed"] is None
    assert p["source"]["work"]["probe_rollouts"]["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize(
    "kind",
    [
        "unrun_success",
        "unrun_return",
        "unrun_attempts",
        "unrun_returns",
        "unrun_state",
        "false_complete",
        "partial_goal",
        "source_id",
        "nonfailed_tail",
    ],
)
def test_should_refuse_invented_unrun_or_partial_success_and_complete_claims(saved, kind):
    r = deepcopy(saved["bp"])
    unrun = r["methods"][0]["probes"][1]["probe_rollouts"][1]["report"]["episodes"][-1]
    if kind.startswith("unrun_"):
        key, value = {
            "unrun_success": ("success", True),
            "unrun_return": ("return", 1),
            "unrun_attempts": ("prediction_calls_attempted", 1),
            "unrun_returns": ("prediction_calls_returned", 1),
            "unrun_state": ("state_unchanged", True),
        }[kind]
        unrun[key] = value
    elif kind == "false_complete":
        r["status"] = "complete"
    elif kind == "source_id":
        r["training_sources"][0]["episode_id"] = "replaced-blue"
    elif kind == "nonfailed_tail":
        r = deepcopy(saved["pc"])
        r["methods"][0]["probes"][1]["probe_rollouts"][0]["report"]["episodes"][-1]["status"] = (
            "running"
        )
    else:
        partial = r["methods"][0]["probes"][1]["probe_rollouts"][1]["report"]["episodes"][-2]
        assert partial["status"] == "failed" and partial.get("success") is None
        partial["success"] = True
    with pytest.raises(ValueError):
        project_key_door_report(r, source_label="corrupt")


@pytest.mark.parametrize("outer", [False, True])
def test_should_refuse_complete_endpoint_with_unfinished_games(saved, outer):
    r = deepcopy(saved["bp"])
    e = r["methods"][0]["probes"][1]["probe_rollouts"][1]
    (e if outer else e["report"])["status"] = "complete"
    with pytest.raises(ValueError, match="complete endpoint"):
        project_key_door_report(r, source_label="corrupt")
