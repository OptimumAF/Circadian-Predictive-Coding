"""Real CLI/live ownership transitions on existing literal native ports only."""

from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from scripts import minigrid_live_adaptation as app
from scripts import minigrid_affine_serving as published
from scripts import run_minigrid_linear_fit_control as cli
from src.app.serving_promotion import ServingPromotionController
from src.app.resource_sharing import ResourceSharedRuntime
from src.app.actor_shadow import ActorShadowRuntime
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from tests import test_live_a4_serving as literal

FLAG = "--example-affine-deploy-a4"
WORK = dict(literal.WORK)


def bind_case(patch):
    patch.setattr(literal, "WORK", WORK)
    box = literal.bind_case(patch, collection_limit=20)
    original = ServingPromotionController.start_candidate

    def start(owner, shared, receipt, **options):
        box["retired"] = shared._runtime
        prepared = original(owner, shared, receipt, **options)
        box["runtime"] = prepared
        return prepared

    patch.setattr(ServingPromotionController, "start_candidate", start)
    return box


@pytest.fixture
def case(monkeypatch):
    return bind_case(monkeypatch)


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        box = bind_case(patch)
        output = tmp_path_factory.mktemp("deployed-a4") / "result.json"
        patch.setattr(sys, "argv", ["affine", FLAG, "--output", str(output)])
        cli.main()
        return json.loads(output.read_text()), box


def test_should_run_real_command_to_two_publications_and_eight_actor_games(complete):
    result, box = complete
    assert result["status"] == "complete", result.get("reason")
    assert result["preservation"] is True and result["summary"]["complete"] is True
    assert result["definition"] == "portable_live_affine_a4_deployment_v1"
    assert box["fits"] == [(9, 1), (9, 2), (18, 3), (18, 4)]
    assert [result[name]["generation"] for name in ("publication", "publication_after")] == [1, 2]
    assert [(g["updates"], g["branch"], g["cue"]) for g in result["actor_games"]] == [
        (stage, branch, cue) for stage in (2, 4) for branch in (3, 4) for cue in ("blue", "red")
    ]
    assert result["games"] == []


def test_should_keep_original_lifetime_charges_and_retired_history(complete):
    result, box = complete
    assert result["private"]["base_version"] == "c0"
    assert (result["private"]["native_updates"], result["private"]["native_rows"]) == (4, 54)
    assert [r["update_number"] for r in result["start"]["old_receipts"]] == [1, 2]
    assert [r["update_number"] for r in result["private"]["receipts"]] == [3, 4]
    assert (
        result["start"]["remaining_capacity"]
        == result["private"]["completed_before_generation"]
        == 2
    )
    assert box["retired"]._inbox.capture_cursor().completed_updates == 2
    assert box["runtime"]._inbox._capacity == 2 and box["runtime"]._budget is box["retired"]._budget
    assert result["sharing"]["admitted_updates"] == result["budgets"]["native"]["updates"] == 4
    assert all(result["budgets"][n]["original_controls"] for n in ("native", "readout"))
    assert result["budgets"]["readout"]["updates"] == result["budgets"]["readout"]["replay"] == 0


def test_should_report_actual_actor_counts_and_explicit_repeated_exposure_without_forcing_gain(
    complete,
):
    result, _ = complete
    assert result["actor"] == dict(version="c1", generation=2, native_updates=4, native_rows=54)
    assert (
        result["catalogue"]["arrival_rows"],
        result["catalogue"]["replay_rows"],
        result["catalogue"]["native_rows"],
    ) == (36, 18, 54)
    assert result["source"]["teacher_steps"] == 18
    assert (
        result["summary"]["acquisition_gain"] == 0.0
    )  # literal predictions are deliberately constant
    assert result["summary"]["learning_recovery"] is None
    assert all(result["global_rng"].values())


def test_should_describe_without_runtime_source_or_budget_work(case, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, "--describe"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == app.describe_live_adaptation(deploy=True)
    assert result["fit_calls"] == 4 and result["fit_rows"] == 54 and result["publications"] == 2
    assert case["trace"] == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--example-affine-live-a4"],
        ["--example-affine-published"],
        ["--example-affine-a4"],
        ["--source", "foreign.json"],
        ["--fit-updates", "4"],
        ["--fit-seconds", "20"],
        ["--cue-relative"],
        ["--describe", "--output", "unused.json"],
    ],
)
def test_should_refuse_unfrozen_combinations_before_work(case, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, *extra])
    with pytest.raises(SystemExit):
        cli.main()
    assert case["trace"] == []


def test_should_refuse_existing_output_without_work(case, monkeypatch, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, "--output", str(output)])
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_text() == "keep" and case["trace"] == []


def test_should_refuse_missing_optional_runtime_before_budgets_or_models(case, monkeypatch):
    def missing():
        raise RuntimeError("missing native pins")

    monkeypatch.setattr(published, "_require_minigrid", missing)
    result = app.run_live_adaptation(deploy=True)
    assert result["status"] == "failed" and result["phase"] == "preflight"
    assert case["trace"] == [] and not result["actor_games"]


def test_should_keep_paid_baseline_and_owner_uncertainty_when_start_loses_return(case, monkeypatch):
    original = ServingPromotionController.start_candidate

    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("lost prepared owner return")

    monkeypatch.setattr(ServingPromotionController, "start_candidate", lost)
    result = app.run_live_adaptation(deploy=True)
    assert result["status"] == "failed" and result["summary"] is None
    assert len(result["actor_games"]) == 4 and not result["collection"]
    assert result["work"]["start_work_may_be_unobserved"] is True
    assert result["work"]["candidate_owner_may_be_unknown"] is True
    assert result["budgets"]["native"]["updates"] == 2
    assert result["actor"]["version"] == "c0" and result["sharing"]["paused"] is True


def test_should_keep_two_prior_receipts_and_permanent_attempt_on_failed_new_fit(case):
    case["fit_fail"] = True
    result = app.run_live_adaptation(deploy=True)
    assert result["status"] == "failed" and result["summary"] is None
    assert len(case["fits"]) == 3 and len(result["actor_games"]) == 4
    assert len(result["start"]["old_receipts"]) == 2 and result["private"]["receipts"] == []
    assert result["sharing"]["admitted_updates"] == 3
    assert result["private"]["readback_uncertain"] is True


def test_should_preserve_a_second_guard_refusal_without_deploying_or_tuning(case, monkeypatch):
    original = ContextLinearLearner.predict

    def worse(owner, x):
        result = original(owner, x)
        if (
            case["report"]["phase"] == "publication-after-A4"
            and owner._state.completed_updates == 4
        ):
            result[:] = 0
            result[:, 6] = 1
        return result

    monkeypatch.setattr(ContextLinearLearner, "predict", worse)
    result = app.run_live_adaptation(deploy=True)
    assert result["status"] == "failed" and result["summary"] is None
    assert result["guards_after"]["decision"]["accepted"] is False
    assert "publication_after" not in result and result["actor"]["version"] == "c0"
    assert len(result["actor_games"]) == 4 and len(case["fits"]) == 4


def test_should_report_actual_generation_when_second_commit_loses_receipt(case, monkeypatch):
    original = ServingPromotionController.commit

    def lost(owner, runtime, ticket):
        result = original(owner, runtime, ticket)
        if runtime._candidate_version == "c1":
            raise RuntimeError("lost second publication receipt")
        return result

    monkeypatch.setattr(ServingPromotionController, "commit", lost)
    result = app.run_live_adaptation(deploy=True)
    assert result["status"] == "failed" and result["summary"] is None
    assert result["actor"]["version"] == "c1" and result["actor"]["generation"] == 2
    assert result["work"]["publication_work_may_be_unobserved"] is True
    assert "publication_after" not in result and len(result["actor_games"]) == 4
    assert len(result["private"]["receipts"]) == 2


def test_should_keep_baseline_and_closed_partial_game_after_deployment(case, monkeypatch):
    original = ResourceSharedRuntime.predict

    def lost(owner, x):
        result = original(owner, x)
        if owner._runtime.actor.version == "c1":
            raise RuntimeError("lost deployed prediction return")
        return result

    monkeypatch.setattr(ResourceSharedRuntime, "predict", lost)
    result = app.run_live_adaptation(deploy=True)
    assert result["status"] == "failed" and result["summary"] is None
    assert len(result["actor_games"]) == 4 and result["publication_after"]["generation"] == 2
    failed = result["failed_actor_game"]
    assert failed["updates"] == 4 and failed["branch"] == 3
    assert failed["report"]["episodes"][0]["environment"]["closed"] is True
    assert result["sharing"]["active_serving_requests"] == 0


@pytest.mark.parametrize("fault", ["retired_receipt", "start_record"])
def test_should_refuse_changed_preceding_evidence_before_more_work(case, fault):
    if fault == "retired_receipt":

        def mutate():
            if len(case["fits"]) == 3:
                r = next(iter(case["retired"]._inbox._applied.values()))
                object.__setattr__(r, "sample_id", "changed")

        case["on_train"] = mutate
    else:

        def replace_record(record):
            if "old_receipts" in record:
                case["report"]["start"] = deepcopy(record)

        case["on_paid"] = replace_record
    result = app.run_live_adaptation(deploy=True)
    assert result["status"] == "failed" and result["summary"] is None
    assert len(case["fits"]) == (3 if fault == "retired_receipt" else 2)
    assert result["start"]["old_receipts"][0]["sample_id"] == "batch-0"


def test_should_record_rng_failure_before_serialization_and_preserve_paid_games(case):
    saved = np.random.get_state()
    changed = False

    def alter(owner):
        nonlocal changed
        if not changed:
            changed = True
            state = np.random.get_state()
            assert isinstance(state, tuple)
            np.random.set_state((state[0], state[1], (state[2] + 1) % 624, state[3], state[4]))

    case["on_predict"] = alter
    try:
        result = app.run_live_adaptation(deploy=True)
        assert result["status"] == "failed" and result["summary"] is None
        assert result["global_rng"]["numpy_unchanged"] is False
        assert len(result["actor_games"]) == 8 and result["publication_after"]["generation"] == 2
    finally:
        np.random.set_state(saved)


def test_should_keep_completed_game_prefix_when_final_candidate_read_fails(case, monkeypatch):
    original = ActorShadowRuntime.candidate_snapshot
    complete_reads = 0

    def read(owner):
        nonlocal complete_reads
        if case["report"]["status"] == "complete":
            complete_reads += 1
            if complete_reads == 2:
                raise ValueError("lost final candidate read")
        return original(owner)

    monkeypatch.setattr(ActorShadowRuntime, "candidate_snapshot", read)
    result = app.run_live_adaptation(deploy=True)
    assert result["status"] == "failed" and result["summary"] is None
    assert len(result["actor_games"]) == 8 and result["actor"]["generation"] == 2
    assert len(result["private"]["receipts"]) == 2 and len(result["start"]["old_receipts"]) == 2
