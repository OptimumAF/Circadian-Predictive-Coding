"""Synthetic paid traces only: no archived source/model/teacher/reference work."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import diagnose_ticket_training as command


def diagnose(source, **kwargs):
    from src.app.ticket_live_noise_diagnosis import diagnose_live_noise_pair

    return diagnose_live_noise_pair(source, **kwargs)


def step(index, selected, *, phase="ready", query="same observation", tie=False):
    ids = ("defer", "route")
    scores = [
        dict(
            candidate_id=c,
            raw_score=1.0 if tie or c == selected else 0.0,
            probability=0.5 if tie else (0.75 if c == selected else 0.25),
        )
        for c in ids
    ]
    request = dict(
        request_id=f"step-{index}",
        query=query,
        allowed_candidates=[dict(candidate_id=c, description=c) for c in ids],
    )
    result = dict(
        request_id=request["request_id"],
        selected_id=selected,
        scores=scores,
        token_visits=1,
        candidate_comparisons=2,
    )
    return dict(
        attempt=index + 1,
        observation_state=dict(ticket_id="paired", concept="legacy", phase=phase, revision=index),
        capabilities_observed=dict(allowed=list(ids), version=0),
        request=request,
        result=result,
        returned=dict(record_type="DecisionResult", observed=deepcopy(result), validated=False),
    )


def source(*, divergent=True):
    recipe = dict(
        recipe_id="live_new_retained_noise",
        contexts=["new_contexts"] * 8,
        label_delay=2,
        noisy_invoice=False,
    )
    arms = []
    for noisy in (False, True):
        episodes = []
        for concept in ("invoice", "parcel", "legacy", "password"):
            selected = "defer" if noisy and divergent and concept == "legacy" else "route"
            trace = [step(0, "route", phase="sealed"), step(1, selected)]
            for item in trace:
                item["observation_state"]["concept"] = concept
            episodes.append(
                dict(
                    concept=concept,
                    role="outer_selection",
                    episode=dict(
                        complete=True,
                        success=concept != "legacy" or noisy and divergent,
                        scenario="ordinary",
                        max_steps=6,
                        unreturned_prediction_work_unknown=False,
                        unvalidated_return_work_unknown=False,
                        work=dict(
                            prediction_calls_attempted=2,
                            prediction_calls_returned=2,
                            prediction_calls_validated=2,
                        ),
                        steps=trace,
                    ),
                )
            )
        declared = dict(recipe=deepcopy(recipe), retained_invoice_noise=noisy)
        case = dict(
            complete=True,
            definition="live_catalog_bp_ticket_private_v1",
            live_training_recipe=declared,
            training_corruption=dict(
                definition="explicit_experimental_label_noise_not_verified_teacher",
                role="train",
                rows_per_batch=[7] if noisy else [],
            ),
            training_work=dict(
                attempted_updates=5,
                returned_updates=5,
                attempted_rows=60,
                unreturned_work_unknown=False,
            ),
            reference_outcomes=dict(
                new_before=1,
                new_after=2 if noisy and divergent else 1,
                old_before=2,
                old_after=2,
                new_gain=1 if noisy and divergent else 0,
                old_drop=0,
            ),
            candidate_episodes=episodes,
        )
        arms.append(
            dict(
                name="retained_corruption" if noisy else "clean",
                recipe=declared,
                case=case,
                native_counts=dict(fits=5, fit_rows=60),
            )
        )
    return dict(definition="native_catalog_live_noise_pair_v1", case_complete=True, arms=arms)


def test_should_find_first_divergent_action_on_common_observation_and_keep_source_unchanged():
    data = source()
    before = deepcopy(data)
    result = diagnose(data)
    assert data == before and result["complete"]
    legacy = result["concepts"][2]
    assert legacy["concept"] == "legacy" and legacy["first_divergence"]["attempt"] == 2
    assert legacy["first_divergence"]["clean_selected"] == "route"
    assert legacy["first_divergence"]["retained_corruption_selected"] == "defer"
    assert legacy["first_divergence"]["noise_action_minus_clean_raw"] == dict(
        clean=-1.0, retained_corruption=1.0
    )
    assert result["work"]["saved_decisions"] == 16 and result["work"]["saved_scores"] == 32
    result["arms"][0]["outcomes"]["new_after"] = 0
    assert data == before


def test_should_keep_no_divergence_and_negative_outcome_without_inventing_mechanism():
    result = diagnose(source(divergent=False))
    assert all(row["first_divergence"] is None for row in result["concepts"])
    assert (
        result["arms"][0]["outcomes"]["new_after"]
        == result["arms"][1]["outcomes"]["new_after"]
        == 1
    )


def test_should_not_compare_states_after_actions_diverge():
    data = source()
    for arm in data["arms"]:
        episode = arm["case"]["candidate_episodes"][2]["episode"]
        episode["steps"].append(step(2, "route", query=arm["name"]))
        for key in episode["work"]:
            episode["work"][key] = 3
    result = diagnose(data)
    assert result["concepts"][2]["paired_decisions"] == 2
    assert len(result["arms"][0]["traces"][2]["decisions"]) == 3


def test_should_preserve_zero_margin_and_deterministic_tie():
    data = source(divergent=False)
    for arm in data["arms"]:
        row = arm["case"]["candidate_episodes"][0]["episode"]["steps"][0]
        row.update(step(0, "defer", phase="sealed", tie=True))
    result = diagnose(data)
    assert result["arms"][0]["traces"][0]["decisions"][0]["selected_raw_margin"] == 0.0


def test_should_leave_single_candidate_margin_unknown():
    data = source(divergent=False)
    for arm in data["arms"]:
        row = arm["case"]["candidate_episodes"][0]["episode"]["steps"][0]
        row["request"]["allowed_candidates"] = [dict(candidate_id="route", description="route")]
        row["result"].update(
            scores=[dict(candidate_id="route", raw_score=1.0, probability=1.0)],
            candidate_comparisons=1,
        )
        row["returned"]["observed"] = deepcopy(row["result"])
    result = diagnose(data)
    assert result["arms"][0]["traces"][0]["decisions"][0]["selected_raw_margin"] is None


def test_should_refuse_nonfinite_metadata_after_preserving_examined_work():
    data = source()
    data["arms"][0]["case"]["reference_outcomes"]["scope"] = float("nan")
    work: dict[str, int] = {}
    with pytest.raises(ValueError):
        diagnose(data, work=work)
    assert work["saved_decisions"] == 8


@pytest.mark.parametrize(
    "fault",
    [
        "family",
        "incomplete",
        "arm_order",
        "missing_concept",
        "duplicate_concept",
        "role",
        "unknown",
        "too_many_steps",
        "missing_result",
        "raw_return",
        "duplicates",
        "nan",
        "probability",
        "selection",
        "request_id",
        "counter_bool",
        "unaligned_query",
        "unaligned_state",
        "unaligned_capabilities",
        "overflow",
    ],
)
def test_should_refuse_inconsistent_or_unbounded_saved_evidence(fault):
    data = source()
    case = data["arms"][0]["case"]
    episode = case["candidate_episodes"][0]["episode"]
    row = episode["steps"][0]
    if fault == "family":
        data["definition"] = "other"
    elif fault == "incomplete":
        data["case_complete"] = False
    elif fault == "arm_order":
        data["arms"].reverse()
    elif fault == "missing_concept":
        case["candidate_episodes"].pop()
    elif fault == "duplicate_concept":
        case["candidate_episodes"][1]["concept"] = "invoice"
    elif fault == "role":
        case["candidate_episodes"][0]["role"] = "final_test"
    elif fault == "unknown":
        episode["unreturned_prediction_work_unknown"] = True
    elif fault == "too_many_steps":
        episode["steps"] *= 4
    elif fault == "missing_result":
        row.pop("result")
    elif fault == "raw_return":
        row["returned"]["observed"]["selected_id"] = "defer"
    elif fault == "duplicates":
        row["result"]["scores"][1]["candidate_id"] = "defer"
    elif fault == "nan":
        row["result"]["scores"][0]["raw_score"] = float("nan")
    elif fault == "probability":
        row["result"]["scores"][0]["probability"] = 0.9
    elif fault == "selection":
        row["result"]["selected_id"] = "defer"
    elif fault == "request_id":
        row["result"]["request_id"] = "changed"
    elif fault == "counter_bool":
        case["training_work"]["returned_updates"] = True
    elif fault == "unaligned_query":
        data["arms"][1]["case"]["candidate_episodes"][0]["episode"]["steps"][0]["request"][
            "query"
        ] = "different"
    elif fault == "unaligned_state":
        data["arms"][1]["case"]["candidate_episodes"][0]["episode"]["steps"][0][
            "observation_state"
        ]["revision"] = 99
    elif fault == "unaligned_capabilities":
        data["arms"][1]["case"]["candidate_episodes"][0]["episode"]["steps"][0][
            "capabilities_observed"
        ]["version"] = 1
    else:
        row["result"]["scores"][0]["raw_score"] = -1e308
        row["result"]["scores"][1]["raw_score"] = 1e308
        row["returned"]["observed"] = deepcopy(row["result"])
    with pytest.raises(ValueError):
        diagnose(data)


def test_should_account_saved_records_before_late_failure():
    data = source()
    data["arms"][1]["case"]["candidate_episodes"][-1]["episode"]["steps"][-1]["result"][
        "request_id"
    ] = "changed"
    work: dict[str, int] = {}
    with pytest.raises(ValueError):
        diagnose(data, work=work)
    assert work["saved_decisions"] == 16 and work["saved_scores"] == 32


def test_should_read_once_and_never_reconstruct_project_or_teach(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source()))
    for name in ("_project", "_teacher", "diagnose", "diagnose_policy_study"):
        monkeypatch.setattr(
            command, name, lambda *a, **kw: pytest.fail("model/reference/teacher route called")
        )
    original = Path.read_bytes
    reads = []

    def read(p):
        if p == path:
            reads.append(p)
        return original(p)

    monkeypatch.setattr(Path, "read_bytes", read)
    output = tmp_path / "out.json"
    assert (
        command.main(["--live-noise-comparison", "--input", str(path), "--output", str(output)])
        == 0
    )
    result = json.loads(output.read_text())
    assert reads == [path] and result["complete"] and result["work"]["projection_calls"] == 0
    assert result["source_scope"] == "captured_bytes_only_no_postflight_seal"


def test_should_preserve_failed_cli_counts_and_unknown_result(tmp_path):
    data = source()
    data["arms"][1]["case"]["candidate_episodes"][-1]["episode"]["steps"][-1]["result"][
        "request_id"
    ] = "changed"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    output = tmp_path / "failed.json"
    assert (
        command.main(["--live-noise-comparison", "--input", str(path), "--output", str(output)])
        == 1
    )
    result = json.loads(output.read_text())
    assert not result["complete"] and result["work"]["saved_decisions"] == 16
    assert "concepts" not in result


def test_should_preserve_output_bound_failure_and_all_saved_work(tmp_path):
    data = source()
    data["arms"][0]["case"]["reference_outcomes"]["scope"] = "x" * 300000
    path = tmp_path / "source.json"
    path.write_text(json.dumps(data))
    output = tmp_path / "failed.json"
    assert (
        command.main(["--live-noise-comparison", "--input", str(path), "--output", str(output)])
        == 1
    )
    result = json.loads(output.read_text())
    assert not result["complete"] and result["work"]["saved_decisions"] == 16
    assert "concepts" not in result and output.stat().st_size < 256 * 1024


def test_should_preserve_serialization_failure_instead_of_leaving_placeholder(
    tmp_path, monkeypatch
):
    from src.app import ticket_live_noise_diagnosis as module

    monkeypatch.setattr(
        module, "diagnose_live_noise_pair", lambda *a, **kw: dict(complete=True, value=float("nan"))
    )
    path = tmp_path / "source.json"
    path.write_text("{}")
    output = tmp_path / "failed.json"
    assert (
        command.main(["--live-noise-comparison", "--input", str(path), "--output", str(output)])
        == 1
    )
    result = json.loads(output.read_text())
    assert not result["complete"] and "source_sha256" in result


@pytest.mark.parametrize("fault", ["same", "occupied", "overlap", "oversize"])
def test_should_refuse_invalid_cli_boundaries(tmp_path, fault):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(source()))
    output = tmp_path / "output.json"
    argv = ["--live-noise-comparison", "--input", str(path), "--output", str(output)]
    if fault == "same":
        argv[-1] = str(path)
        with pytest.raises(ValueError):
            command.main(argv)
    elif fault == "occupied":
        output.write_text("preserve")
        with pytest.raises(FileExistsError):
            command.main(argv)
        assert output.read_text() == "preserve"
    elif fault == "overlap":
        with pytest.raises(SystemExit):
            command.main([*argv, "--policy-comparison"])
        assert not output.exists()
    else:
        path.write_bytes(b"x" * (3 * 1024**2 + 1))
        assert command.main(argv) == 1
        assert not json.loads(output.read_text())["complete"]


@pytest.mark.parametrize("policy_mode", [False, True])
def test_should_preserve_legacy_route_dispatch_and_source_reread(
    tmp_path, monkeypatch, policy_mode
):
    path = tmp_path / "literal.json"
    path.write_bytes(b"{}")
    output = tmp_path / "old.json"
    calls = []

    def original(*args, **kwargs):
        calls.append(policy_mode)
        return dict(complete=True, work=kwargs["work"])

    monkeypatch.setattr(command, "diagnose_policy_study" if policy_mode else "diagnose", original)
    if policy_mode:
        monkeypatch.setattr(command, "POLICY_STUDY_SHA", command.hashlib.sha256(b"{}").hexdigest())
    read = Path.read_bytes
    reads = []

    def counted(p):
        if p == path:
            reads.append(p)
        return read(p)

    monkeypatch.setattr(Path, "read_bytes", counted)
    args = ["--input", str(path), "--output", str(output)]
    assert command.main(args + (["--policy-comparison"] if policy_mode else [])) == 0
    assert reads == [path, path] and calls == [policy_mode]
