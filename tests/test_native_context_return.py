"""Native context-return behavior, paid faults and unchanged ordinary controls."""

import json

import pytest

from scripts import recurring_ticket_workflow_example as study
from scripts import ticket_workflow_native as native
from src.adapters.torch_feature_learner import TorchFeatureLearner


@pytest.fixture(scope="module", params=native.METHODS)
def completed(request):
    result = study.run_example(context_return=True, method=request.param)
    assert result["complete"], result.get("error")
    return result


def test_should_execute_same_contexts_roles_receipts_and_full_native_snapshots(completed):
    r = completed
    assert r["source_unchanged"] and r["actor_unchanged"]
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    assert r["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=72,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert [x["source_rows"] for x in r["checkpoints"]] == [0, 24, 48, 72]
    for row in r["checkpoints"]:
        assert row["state"]["policy"]["feature_dim"] == 9
        assert row["state"]["policy"]["seed"] == 17
        assert row["state"]["policy"]["hidden_dim"] == 3
        assert row["source_mistakes"] is None
    assert all(x["role"] == "train" and x["rows"] == 6 for x in r["training_data"])
    assert [x["policy_version"] for x in r["training_data"]] == ["v1"] * 4 + ["v2"] * 4 + ["v1"] * 4
    assert len(r["applied_updates"]) == 12
    assert all(
        x["observed_at"] + 1 == x["arrived_at"] <= x["applied_at"] for x in r["applied_updates"]
    )
    assert r["budget"]["limits"]["max_wall_seconds"] == 30
    assert r["budget"]["limits"]["max_process_rss_bytes"] == 512 * 1024**2
    assert r["stream"]["score_forward_calls"] == 0
    assert len(r["episodes"]) == 24
    for point in study.CHECKPOINTS:
        rows = [x for x in r["episodes"] if x["source_updates"] == point]
        assert [(x["policy_version"], x["concept"]) for x in rows[:4]] == list(study.CONTEXT_TASKS)
    assert all(x["role"] == "outer_selection" and x["episode"]["complete"] for x in r["episodes"])
    assert r["summary"] == study.summarize_context_return(r["episodes"], seeded_initial=True)
    assert all(
        r[k] == 0
        for k in ("final_examples", "probe_updates", "replay_examples", "external_actions")
    )


def test_should_count_actual_d9_native_costs_and_bound_output(completed):
    r = completed
    calls = sum(x["work"]["native_forward_attempts"] for x in r["native_readout_work"])
    assert calls == r["episode_work"]["prediction_calls_validated"]
    assert r["costs"]["attempted_training_nominal_affine_terms"] == 72 * 45
    assert r["costs"]["attempted_readout_nominal_affine_terms"] == calls * 45
    assert "attempted_training_affine_terms" not in r["costs"]
    assert r["costs"]["model_random_draws"] is None
    assert r["definition"] == "native_context_return_ticket_workflow_v1"
    assert "seeded initial snapshot" in r["no_learning_control"]
    assert "zero-state" not in r["summary"]["interpretation"]
    encoded = study._encode_report(r)
    assert len(encoded) <= 512 * 1024
    assert json.loads(encoded)["summary"] == json.loads(json.dumps(r["summary"]))


@pytest.mark.parametrize("method", native.METHODS)
def test_should_forward_native_cli_without_changing_default(method, tmp_path, monkeypatch):
    seen = []

    def run(**kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(study, "run_example", run)
    path = tmp_path / "native.json"
    assert study.main(["--context-return", "--method", method, "--output", str(path)]) == 0
    assert seen == [dict(context_return=True, method=method)]
    with pytest.raises(FileExistsError):
        study.main(["--context-return", "--method", method, "--output", str(path)])
    assert len(seen) == 1


@pytest.mark.parametrize(
    "kw",
    [
        dict(context_return=True, method="bp", replay="recent"),
        dict(context_return=True, method="pc", policy_change=True),
        dict(context_return=True, method="bp", separate_context_heads=True),
        dict(context_return=True, method="pc", private_adaptation=True),
        dict(context_return=True, method="gated_cpc", paired_serving="joint"),
    ],
)
def test_should_keep_unsupported_combinations_before_data(kw, monkeypatch):
    monkeypatch.setattr(study, "prepare_training", lambda **kw: pytest.fail("entered data"))
    with pytest.raises(ValueError):
        study.run_example(**kw)


def test_should_reject_final_context_labels_before_native_construction(monkeypatch):
    from scripts import run_ticket_workflow as teacher

    original = teacher.training_cases

    def corrupt(**kwargs):
        cases = original(**kwargs)
        object.__setattr__(cases[0], "role", "final_test")
        return cases

    monkeypatch.setattr(teacher, "training_cases", corrupt)
    monkeypatch.setattr(native, "counted_learner", lambda *a: pytest.fail("final reached model"))
    r = study.run_example(context_return=True, method="bp")
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0


def test_should_preserve_lost_fifth_native_return_and_original_prefix(monkeypatch):
    original = TorchFeatureLearner.train_batch
    calls = 0

    def fail(self, features, targets):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("lost native context update")
        return original(self, features, targets)

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", fail)
    r = study.run_example(context_return=True, method="bp")
    assert not r["complete"] and "lost native context" in r["error"]
    assert r["training_work"]["attempted_updates"] == 5
    assert r["training_work"]["attempted_rows"] == 30
    assert r["training_work"]["returned_updates"] == r["budget"]["updates_completed"] == 4
    assert r["training_work"]["unreturned_work_unknown"]
    assert len(r["applied_updates"]) == 4


@pytest.mark.parametrize("fault", ["binding", "source"])
def test_should_refuse_corrupted_native_readout_or_source_before_training(fault, monkeypatch):
    holder = {}
    original_stream = study.AttributionStream

    def capture(*a, **kw):
        stream = original_stream(*a, **kw)
        holder["stream"] = stream
        return stream

    monkeypatch.setattr(study, "AttributionStream", capture)
    original = native.TorchDecisionPredictor.__call__

    def corrupt(self, request):
        value = original(self, request)
        if fault == "binding":
            self._initial.head_state["bias_output"].add_(1)
            self._learner._head.bias_output.add_(1)
        else:
            getattr(holder["stream"]._runtime._candidate, "_head").bias_output.add_(1)
        return value

    monkeypatch.setattr(native.TorchDecisionPredictor, "__call__", corrupt)
    r = study.run_example(context_return=True, method="pc")
    assert not r["complete"] and r["training_work"]["attempted_updates"] == 0
    assert r["native_readout_work"][0]["work"]["native_forward_returns"] == 1
    assert r["episode_work"]["prediction_calls_validated"] == 0


def test_should_describe_seeded_initial_control_without_constructing_a_model(monkeypatch):
    def stop(**kwargs):
        raise ValueError("stop before data/model")

    monkeypatch.setattr(study, "prepare_training", stop)
    r = study.run_example(context_return=True, method="bp")
    assert "seeded initial snapshot" in r["no_learning_control"]
    assert r["training_work"]["attempted_updates"] == 0
    rows = [
        dict(
            source_updates=p,
            policy_version=v,
            concept=c,
            episode=dict(scenario="ordinary", success=False),
        )
        for p in study.CHECKPOINTS
        for v, c in study.CONTEXT_TASKS
    ]
    assert (
        "seeded initial-state"
        in study.summarize_context_return(rows, seeded_initial=True)["interpretation"]
    )
    assert "zero-state" in study.summarize_context_return(rows)["interpretation"]
