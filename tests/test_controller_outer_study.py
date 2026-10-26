"""Native objective integration: distinct sources and read-only sealed endpoints."""

from copy import deepcopy
from pathlib import Path
import json
import os
from types import SimpleNamespace

import pytest

from scripts import run_bp_ticket_serving as demo


@pytest.fixture(scope="module")
def completed():
    from src.adapters.torch_feature_learner import TorchFeatureLearner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo.context, "BATCHES_PER_PHASE", 2)
        spec = demo.DecisionFeatureSpec(demo.FEATURES, demo.CATALOG)
        templates = demo.context._teacher_batches(spec)
        patch.setattr(demo.context, "_teacher_batches", lambda s: deepcopy(templates))
        original = demo._sealed_readout

        def readonly(*args):
            with pytest.MonkeyPatch.context() as sealed:
                sealed.setattr(
                    TorchFeatureLearner, "train_batch", lambda *a: pytest.fail("sealed training")
                )
                return original(*args)

        patch.setattr(demo, "_sealed_readout", readonly)
        result = demo.run_outer_study(
            Path("artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json")
        )
    if os.environ.get("OUTER_STUDY_FIXTURE_REPORT"):
        Path(os.environ["OUTER_STUDY_FIXTURE_REPORT"]).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    assert result["complete"], result.get("error")
    return result


def test_native_study_declares_before_work_and_uses_distinct_actual_streams(completed):
    r = completed
    assert r["declared_at"] == 0 and r["protocol"]["policy"]["mode"] == "periodic"
    assert all(
        r["protocol"]["objective"][name] == 1
        for name in (
            "new_gain_weight",
            "old_drop_weight",
            "compute_weight",
            "memory_weight",
            "violation_weight",
        )
    )
    phases = r["phases"]
    train = phases["controller_train"]
    validation = phases["controller_validation"]
    assert (
        train["training_context"] == "new_contexts"
        and validation["training_context"] == "old_return"
    )
    assert (
        train["training_stream"]["first_features"]
        != validation["training_stream"]["first_features"]
    )
    for result in phases.values():
        assert result["training_work"] == dict(
            attempted_updates=1, attempted_rows=6, returned_updates=1, unreturned_work_unknown=False
        )
        assert (
            result["source_unchanged"]
            and result["numpy_rng_unchanged"]
            and result["torch_rng_unchanged"]
        )
        assert result["final_examples"] == result["reserved_probe_updates"] == 0
    rows = r["chronology"]["rows"]
    assert [v["metadata"]["stream"]["role"] for v in rows] == [
        "controller_train",
        "controller_validation",
        "sealed_evaluation",
    ]
    assert [v["metadata"]["completed_at"] for v in rows] == [2, 4, 6]
    assert r["freeze"]["frozen_at"] == 4 and r["sealed_source_reads"] == [
        dict(at=5, role="sealed_evaluation")
    ]
    assert r["chronology"]["attempts"] == 3 and not r["chronology"]["stopped"]


def test_sealed_evaluation_is_detached_and_has_zero_training_or_promotion(completed):
    r = completed
    assert r["sealed_training_updates"] == r["sealed_promotions"] == 0
    assert len(r["sealed_episodes"]) == 8 and all(
        e["role"] == "final_test" for e in r["sealed_episodes"]
    )
    assert len({e["case_id"] for e in r["sealed_episodes"]}) == 4
    assert len(r["sealed_readout_work"]) == 2
    assert all(
        w["decision_attempts"] == w["decision_returns"] == w["native_forward_returns"] == 12
        for w in r["sealed_readout_work"]
    )
    assert r["numpy_rng_unchanged"] and r["torch_rng_unchanged"]
    for row in r["chronology"]["rows"]:
        u = row["utility"]
        assert u["normalized_compute"] >= 0 and u["normalized_memory"] > 0
        assert u["normalized_violations"] == 0
        assert u["value"] == pytest.approx(
            u["normalized_new_gain"]
            - u["normalized_old_drop"]
            - u["normalized_compute"]
            - u["normalized_memory"]
            - u["normalized_violations"]
        )


def test_applied_violations_are_not_confused_with_refused_actions_or_wrong_task(completed):
    episode = deepcopy(completed["sealed_episodes"][0]["episode"])
    assert demo._violation_count([episode]) == 0
    row = next(s for s in episode["steps"] if s["transition"]["status"] == "applied")
    row["capabilities_at_effect"]["allowed"] = ()
    assert demo._violation_count([episode]) == 1
    row["transition"]["status"] = "permission_denied"
    assert demo._violation_count([episode]) == 0
    episode["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        demo._violation_count([episode])


def test_failed_native_development_cannot_open_validation_or_sealed_source(monkeypatch):
    calls = []

    def failed(*args, **kwargs):
        calls.append(kwargs["training_context"])
        return dict(complete=False, training_work=dict(unreturned_work_unknown=True))

    monkeypatch.setattr(demo, "run_case", failed)
    monkeypatch.setattr(
        demo, "_sealed_cases", lambda: pytest.fail("opened sealed source after failed development")
    )
    r = demo.run_outer_study(Path("unused"))
    assert not r["complete"] and calls == ["new_contexts"]
    assert r["sealed_source_reads"] == [] and r["chronology"]["attempts"] == 1
    assert r["chronology"]["stopped"] and not r["chronology"]["rows"]
    assert "controller_train" in r["phases"]


def test_outer_study_cli_is_optin_mutually_exclusive_and_never_overwrites(tmp_path, monkeypatch):
    calls = []

    def run(source):
        calls.append(source)
        return {"complete": False}

    monkeypatch.setattr(demo, "run_outer_study", run)
    output = tmp_path / "outer.json"
    assert demo.main(["--source", "unused", "--outer-study", "--output", str(output)]) == 1
    assert calls == [Path("unused")] and json.loads(output.read_bytes()) == {"complete": False}
    with pytest.raises(FileExistsError):
        demo.main(["--source", "unused", "--outer-study", "--output", str(output)])
    with pytest.raises(SystemExit):
        demo.main(
            [
                "--source",
                "unused",
                "--outer-study",
                "--reference",
                "drift",
                "--output",
                str(tmp_path / "invalid.json"),
            ]
        )


def test_sealed_execution_cost_includes_source_construction_under_sampler(monkeypatch):
    from src.adapters.torch_decision import TorchDecisionWork

    wall = [0.0]
    active = [False]

    class Sampler:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            active[0] = True
            return self

        def __exit__(self, *args):
            active[0] = False

        def snapshot(self):
            return SimpleNamespace(peak_bytes=1024)

    class Reader:
        work = TorchDecisionWork()

        def assert_unchanged(self):
            pass

    def source():
        assert active[0]
        wall[0] += 20.0
        return tuple((str(i), c) for i, c in enumerate(demo.context.CONCEPTS))

    monkeypatch.setattr(demo, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(demo, "monotonic", lambda: wall[0])
    monkeypatch.setattr(demo, "_sealed_cases", source)
    monkeypatch.setattr(demo.context, "bp_snapshot_from_record", lambda s: s)
    monkeypatch.setattr(demo.native, "readout", lambda *a: Reader())
    monkeypatch.setattr(
        demo, "run_workflow_episode", lambda *a, **kw: dict(complete=True, success=False, steps=[])
    )
    r: dict = {"sealed_source_reads": []}
    m = demo._sealed_readout(
        {"publications": [{"model": {}}], "candidate_checkpoint": {"state": {}}},
        r,
        demo.LogicalClock(4),
    )
    assert m.execution_seconds == 20.0 and m.peak_rss_bytes == 1024
    assert r["sealed_source_reads"] == [dict(at=5, role="sealed_evaluation")]
