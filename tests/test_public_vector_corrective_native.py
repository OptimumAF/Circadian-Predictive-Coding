"""Synthetic saved-source corrective-stage gates; no fit, SGD or forward."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import sys

import pytest

from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_bc import OBJECTIVE
from src.core.pick_cube_corrections import PROVENANCE, PickCubeCorrectionBatch
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from test_maniskill_bc_smoke import assert_same_state
from test_pick_cube_corrections import labels_for
from test_public_vector_mlp_depth_native import data as data
from test_public_vector_mlp_native import model_rollout
from test_public_vector_native_case import episode_from_teacher
from test_visual_native_context_case import write_new


case = importlib.import_module(
    "artifacts.runs.public-vector-corrective-native-20261019.native_case"
)


def forbidden(*_args, **_kwargs):
    pytest.fail("invalid evidence admitted further work")


@pytest.fixture(scope="module")
def evidence(data):
    initial = replace(data["state"], training_examples=2816)
    raw = model_rollout(data["raw"]["new"], initial, success=False)
    raw.update(source_updates=96, source_training_rows=2816)
    episode = episode_from_teacher(dict(candidate_episode=raw["episode"]))
    batch = PickCubeCorrectionBatch(19, episode, labels_for(episode))
    raw["corrections"] = dict(
        provenance=PROVENANCE,
        train_seed=19,
        geometry_attempts=3,
        geometry_returns=3,
        teacher_action_attempts=3,
        teacher_actions=3,
        teacher_schedule_steps=3,
        labels=[
            dict(
                asdict(label),
                observed_at=label.observed_at,
                labels_arrived_at=label.labels_arrived_at,
            )
            for label in batch.labels
        ],
        batch=dict(train_seed=19, episode_id=episode.episode_id, rows=3, labels_arrived_at=5),
    )
    return dict(initial=initial, collection=raw, batch=batch)


def update_report(batch, initial):
    rows, arrived = len(batch.labels), batch.labels[-1].labels_arrived_at
    observed, events, receipts = batch.labels[-1].observed_at, [], []
    for index in range(8):
        receipt = dict(
            sample_id=f"corrective-batch-{index}",
            episode_id=batch.episode.episode_id,
            event_id=f"corrective-label-{index}",
            actor_version="actor-0",
            learner_version="candidate-0",
            observed_at=observed,
            arrived_at=arrived,
            applied_at=arrived,
            update_number=index + 1,
            diagnostic=dict(definition=OBJECTIVE, value=0.25),
        )
        receipts.append(receipt)
        events.append(
            dict(
                index=index,
                update_attempted=True,
                applied=[deepcopy(receipt)],
                success_validated=True,
            )
        )
    projected_work = dict(
        observation_projections=2 * rows,
        source_values_validated=2 * rows * 789,
        projected_values=2 * rows * 21,
        projected_numeric_bytes=2 * rows * 168,
        projected_label_observation_values=rows * 21,
        projected_label_target_values=rows * 4,
        projected_label_observation_numeric_bytes=rows * 168,
        projected_label_target_numeric_bytes=rows * 32,
    )
    work = dict(
        optimizer_updates=8,
        training_rows=8 * rows,
        unique_correction_rows=rows,
        repeated_training_rows=7 * rows,
        reference_replay_rows=0,
        fit_calls=0,
        forward_passes=8,
        forward_rows=8 * rows,
        backward_passes=8,
        backward_rows=8 * rows,
        normalization_calls=8,
        normalization_rows=8 * rows,
        normalization_values=8 * rows * 21,
        gradient_values=13344,
        gradient_numeric_bytes=106752,
        dense_multiply_accumulates=8 * rows * 3456,
        model_numeric_bytes_per_owner=26688,
        statistics_numeric_bytes_per_owner=336,
        named_model_owners=4,
        inbox_feature_target_numeric_bytes=8 * rows * 200,
        scope="successful fixed run; numeric payload/dense work excludes copies/validation/elementwise/process overhead",
    )
    return dict(
        complete=True,
        protocol_id="continuous_correction_8_v1",
        provenance=PROVENANCE,
        source_updates=initial.completed_updates,
        source_training_rows=initial.training_examples,
        policy=asdict(initial.policy),
        observed_at=observed,
        labels_arrived_at=arrived,
        settings=dict(
            max_training_updates=8,
            reference_replay_rows=0,
            fit_calls=0,
            promotion_calls=0,
            simulator_calls=0,
            guard_reads=0,
            probe_reads=0,
            final_reads=0,
        ),
        events=events,
        applied_updates=receipts,
        progress=dict(updates_completed=8, replay_examples_completed=0, next_tick=arrived + 1),
        input_projection=dict(
            projection=asdict(public_vector_projection()),
            source_recipe=asdict(pick_cube_rgb_recipe()),
            source_episode_id=batch.episode.episode_id,
            source_spec=asdict(batch.episode.spec),
            model_spec=asdict(initial.policy.spec),
            work=projected_work,
        ),
        work=work,
        elapsed_seconds=0.01,
    )


def test_should_reconstruct_owned_raw_labels_without_changing_student_trace(data, evidence):
    raw, initial = deepcopy(evidence["collection"]), evidence["initial"]
    original = deepcopy(raw)
    audit = {}
    batch = case.validate_collection(raw, data["episodes"]["new"], initial, audit)
    assert asdict(batch) == asdict(evidence["batch"])
    assert batch is not evidence["batch"] and batch.episode.spec.observation_dim == 789
    assert raw == original
    observations, targets = batch.training_pairs(now=5)
    assert observations == tuple(row.observation for row in batch.episode.transitions)
    assert targets == ((0.5, 0.0, 1.0, 1.0),) * 3
    assert targets != tuple(row.action for row in batch.episode.transitions)
    raw["corrections"]["labels"][0]["observation"] = (100.0,) * 789
    assert batch.training_pairs(now=5) == (observations, targets)


@pytest.mark.parametrize(
    "damage",
    [
        "role",
        "source_counter",
        "clock",
        "clock_type",
        "label_observation",
        "stage",
        "target",
        "geometry",
        "geometry_type",
        "schedule",
        "provenance",
        "batch_rows",
    ],
)
def test_should_refuse_changed_native_trace_label_pairing_clocks_or_query_work(
    data, evidence, damage
):
    raw = deepcopy(evidence["collection"])
    labels, queries = raw["corrections"]["labels"], raw["corrections"]
    if damage == "role":
        raw["episode"]["role"] = "probe_train"
    elif damage == "source_counter":
        raw["source_updates"] = 104
    elif damage == "clock":
        labels[-1]["labels_arrived_at"] = 4
    elif damage == "clock_type":
        labels[0]["observed_at"] = False
    elif damage == "label_observation":
        labels[0]["observation"] = (1000.0,) + tuple(labels[0]["observation"])[1:]
    elif damage == "stage":
        labels[0]["stage"] = "lift"
    elif damage == "target":
        labels[0]["target"] = (0.0, 0.0, 0.0, -1.0)
    elif damage == "geometry":
        queries["geometry_returns"] = 2
    elif damage == "geometry_type":
        queries["geometry_attempts"] = 3.0
    elif damage == "schedule":
        queries["teacher_schedule_steps"] = 2
    elif damage == "provenance":
        queries["provenance"] = "expert_qualified"
    else:
        queries["batch"]["rows"] = 2
    with pytest.raises((AssertionError, ValueError)):
        case.validate_collection(raw, data["episodes"]["new"], evidence["initial"], {})


def test_should_validate_exact_eight_projected_exposures_without_fitting_or_gradients(evidence):
    report = update_report(evidence["batch"], evidence["initial"])
    before, audit = deepcopy(report), {}
    case.validate_update(report, evidence["batch"], evidence["initial"], audit)
    assert report == before
    assert report["work"]["training_rows"] == 24
    assert report["work"]["dense_multiply_accumulates"] == 82944
    assert report["input_projection"]["work"]["source_values_validated"] == 4734
    assert report["progress"]["updates_completed"] == 8


@pytest.mark.parametrize(
    "damage",
    [
        "complete",
        "source",
        "policy",
        "time",
        "budget",
        "receipt_clock",
        "receipt_value",
        "receipt_objective",
        "receipt_negative",
        "receipt_id",
        "event",
        "updates",
        "rows",
        "normalization",
        "work_type",
        "projection",
        "model_spec",
    ],
)
def test_should_refuse_changed_update_receipts_source_or_projection_costs(evidence, damage):
    report = update_report(evidence["batch"], evidence["initial"])
    if damage == "complete":
        report["complete"] = 1
    elif damage == "source":
        report["source_training_rows"] = 2817
    elif damage == "policy":
        report["policy"]["momentum"] = 0.8
    elif damage == "time":
        report["labels_arrived_at"] = 6
    elif damage == "budget":
        report["progress"]["updates_completed"] = 7
    elif damage == "receipt_clock":
        report["applied_updates"][0]["arrived_at"] = 6
    elif damage == "receipt_value":
        report["applied_updates"][0]["diagnostic"]["value"] = True
    elif damage == "receipt_objective":
        report["applied_updates"][0]["diagnostic"]["definition"] = "selected_loss"
        report["events"][0]["applied"] = [deepcopy(report["applied_updates"][0])]
    elif damage == "receipt_negative":
        report["applied_updates"][0]["diagnostic"]["value"] = -0.01
        report["events"][0]["applied"] = [deepcopy(report["applied_updates"][0])]
    elif damage == "receipt_id":
        report["applied_updates"][0]["sample_id"] = "reused"
    elif damage == "event":
        report["events"][0]["success_validated"] = False
    elif damage == "updates":
        report["work"]["optimizer_updates"] = 7
    elif damage == "rows":
        report["work"]["training_rows"] = 25
    elif damage == "normalization":
        report["work"]["normalization_values"] = 505
    elif damage == "work_type":
        report["work"]["gradient_values"] = 13344.0
    elif damage == "projection":
        report["input_projection"]["work"]["projected_values"] = 125
    else:
        report["input_projection"]["model_spec"]["observation_dim"] = 789
    if damage.startswith("receipt"):
        report["events"][0]["applied"] = [deepcopy(report["applied_updates"][0])]
    with pytest.raises((AssertionError, ValueError)):
        case.validate_update(report, evidence["batch"], evidence["initial"], {})


@pytest.fixture
def scenario(data, evidence, tmp_path, monkeypatch):
    """One constructor/two restores per case; fork allocates without either call."""
    stage, baseline = tmp_path / "current", tmp_path / "uncorrected"
    stage.mkdir()
    baseline.mkdir()
    monkeypatch.setattr(case, "stage", stage)
    monkeypatch.setattr(case, "BASELINE_STAGE", baseline)
    owner = NumpyContinuousBCLearner(data["policy"])
    owner.restore_state(evidence["initial"])
    candidate = owner.fork()
    state = candidate.snapshot_state()
    parameters = tuple(value.copy() for value in state.parameters)
    parameters[3][:] += 0.1
    candidate.restore_state(
        replace(state, parameters=parameters, completed_updates=104, training_examples=2840)
    )
    candidate_initial = candidate.snapshot_state()
    for name, context in (("a-rollout.json", "warm"), ("b-rollout.json", "new")):
        raw = model_rollout(data["raw"][context], evidence["initial"], success=context == "new")
        raw.update(source_updates=96, source_training_rows=2816)
        write_new(baseline / name, raw)
    observed = dict(native=[], updates=[], options=[], failure=None)

    def load(audit):
        audit.update(source_updates=96, source_training_rows=2816)
        return owner, data["episodes"]["warm"], data["episodes"]["new"]

    def frozen(path, **options):
        collecting, seed = options.get("corrective_queries", False), options["train_seed"]
        observed["native"].append((path.name, seed, options["learner"]))
        assert options["visual_observation"] is True
        assert options["observation_projection"] == public_vector_projection()
        if collecting:
            assert options["learner"] is owner and seed == 19
            raw, batch = deepcopy(evidence["collection"]), evidence["batch"]
            raw["episode"] = asdict(batch.episode)
        else:
            assert options["learner"] is candidate
            raw = model_rollout(
                data["raw"]["warm" if seed == 17 else "new"], candidate_initial, success=seed == 17
            )
            raw.update(source_updates=104, source_training_rows=2840)
            batch = None
        if observed["failure"] == "collection_label":
            raw["corrections"]["labels"][0]["labels_arrived_at"] = 0
        if observed["failure"] == "candidate_state" and not collecting:
            state = candidate.snapshot_state()
            state.velocity[3][0] += 1.0
            candidate.restore_state(state)
        write_new(path, raw)
        if observed["failure"] == "unreturned" and collecting:
            raise RuntimeError("paid collection did not return")
        returned = deepcopy(raw)
        if observed["failure"] == "returned_trace" and collecting:
            returned["elapsed_seconds"] += 0.1
        return case.smoke.FrozenRolloutResult(returned, batch)

    def update(model, batch, *, now, **options):
        observed["updates"].append((model, batch, now, options))
        assert model is owner and now == 5
        assert asdict(batch) == asdict(evidence["batch"])
        assert options == dict(observation_projection=public_vector_projection())
        report = update_report(batch, evidence["initial"])
        if observed["failure"] == "paid_update":
            report.update(complete=False, error="paid update return failed")
            report["progress"]["updates_completed"] = 1
            report["applied_updates"] = report["applied_updates"][:1]
            report["events"] = report["events"][:1]
            report.pop("work")
            return ContinuousLearnabilityResult(report, None)
        if observed["failure"] == "corrupt_progress":
            report["progress"]["updates_completed"] = True
        return ContinuousLearnabilityResult(report, candidate)

    original_pipeline = case.smoke.run_corrective_adaptation
    original_summary = case.smoke._rollout_summary
    summary_calls = {}

    def summary(result, seed):
        value = original_summary(result, seed)
        identity = id(result.report)
        summary_calls[identity] = summary_calls.get(identity, 0) + 1
        if summary_calls[identity] == 2:
            name = "collection.json" if "corrections" in result.report else f"updated-{seed}.json"
            replacing = (
                observed["failure"] == "collection_replaced" and name == "collection.json"
            ) or (observed["failure"] == "preceding_replaced" and name == "updated-17.json")
            if replacing:
                path = stage / "adaptation" / name
                raw = json.loads(path.read_text())
                raw["elapsed_seconds"] += 0.1
                path.write_text(json.dumps(raw), encoding="utf8")
        return value

    snapshot = candidate.snapshot_state
    replaced_update = []

    def candidate_snapshot():
        value = snapshot()
        path = stage / "adaptation" / "update.json"
        if observed["failure"] == "update_replaced" and path.exists() and not replaced_update:
            report = json.loads(path.read_text())
            report["work"]["normalization_values"] += 1
            path.write_text(json.dumps(report), encoding="utf8")
            replaced_update.append(path)
        return value

    def pipeline(model, *, output_directory, visual_observation, observation_projection):
        observed["options"].append(
            (model, output_directory, visual_observation, observation_projection)
        )
        assert model is owner and visual_observation is True
        assert observation_projection == public_vector_projection()
        return original_pipeline(
            model,
            output_directory=output_directory,
            visual_observation=visual_observation,
            observation_projection=observation_projection,
        )

    monkeypatch.setattr(case.saved, "load_candidate", load)
    monkeypatch.setattr(case.smoke, "run_frozen_rollout", frozen)
    monkeypatch.setattr(case.smoke, "run_continuous_correction_update", update)
    monkeypatch.setattr(case.smoke, "run_corrective_adaptation", pipeline)
    monkeypatch.setattr(case.smoke, "_rollout_summary", summary)
    monkeypatch.setattr(candidate, "snapshot_state", candidate_snapshot)
    monkeypatch.setattr(NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", forbidden)
    return dict(
        stage=stage,
        baseline=baseline,
        owner=owner,
        candidate=candidate,
        candidate_initial=candidate_initial,
        observed=observed,
        frozen=frozen,
        update=update,
    )


def test_should_use_existing_pipeline_and_keep_all_six_signed_outcomes_without_baseline_rerun(
    data, evidence, scenario
):
    audit = dict(complete=False, optimizer_updates=0, new_label_rows=0)
    case.run_case(audit)
    assert audit["complete"] is audit["final_full_source_exact"] is True
    assert [(name, seed) for name, seed, _ in scenario["observed"]["native"]] == [
        ("collection.json", 19),
        ("updated-17.json", 17),
        ("updated-19.json", 19),
    ]
    assert len(scenario["observed"]["updates"]) == 1
    assert scenario["observed"]["options"] == [
        (scenario["owner"], scenario["stage"] / "adaptation", True, public_vector_projection())
    ]
    assert audit["reference_rows"] == 5 and audit["reference_observation_values"] == 7890
    assert audit["reference_action_values"] == 20
    assert (
        audit["optimizer_updates"] == 8
        and audit["training_rows"] == 24
        and audit["new_label_rows"] == 3
    )
    assert audit["update_validated"] is True and audit["observed_paid_receipt_count"] == 8
    assert audit["primary"] == [
        dict(seed=17, sparse_return_delta=1.0, any_success_delta=1.0, final_success_delta=1.0),
        dict(seed=19, sparse_return_delta=-1.0, any_success_delta=-1.0, final_success_delta=-1.0),
    ]
    assert all(row["qualified"] is row["returned"] is True for row in audit["rollouts"])
    assert_same_state(evidence["initial"], scenario["owner"].snapshot_state())
    assert_same_state(scenario["candidate_initial"], scenario["candidate"].snapshot_state())
    assert case.smoke.run_frozen_rollout is scenario["frozen"]
    assert case.smoke.run_continuous_correction_update is scenario["update"]
    retained = json.loads((scenario["stage"] / "adaptation" / "update.json").read_text())
    assert case.exact(retained, audit["update_observed"])


@pytest.mark.parametrize("damage", ["source", "projection", "sparse_type"])
def test_should_refuse_invalid_uncorrected_saved_trace_before_any_native_or_update(
    scenario, damage
):
    path = scenario["baseline"] / "a-rollout.json"
    raw = json.loads(path.read_text())
    if damage == "source":
        raw["source_updates"] = 104
    elif damage == "projection":
        raw["projection_work"]["source_values_validated"] -= 1
    else:
        raw["sparse_return"] = False
    path.write_text(json.dumps(raw), encoding="utf8")
    audit = dict(complete=False)
    with pytest.raises((AssertionError, ValueError)):
        case.run_case(audit)
    assert scenario["observed"]["native"] == scenario["observed"]["updates"] == []
    assert scenario["observed"]["options"] == []
    assert not (scenario["stage"] / "adaptation").exists()
    assert case.smoke.run_frozen_rollout is scenario["frozen"]
    assert case.smoke.run_continuous_correction_update is scenario["update"]


@pytest.mark.parametrize(
    "failure", ["collection_label", "returned_trace", "unreturned", "collection_replaced"]
)
def test_should_preserve_paid_collection_and_restore_callables_before_admitting_any_update(
    scenario, failure
):
    scenario["observed"]["failure"] = failure
    audit = dict(complete=False, optimizer_updates=0, new_label_rows=0)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        case.run_case(audit)
    assert [(name, seed) for name, seed, _ in scenario["observed"]["native"]] == [
        ("collection.json", 19)
    ]
    assert scenario["observed"]["updates"] == []
    assert audit["optimizer_updates"] == 0
    assert (scenario["stage"] / "adaptation" / "collection.json").exists()
    assert (scenario["stage"] / "adaptation" / "summary.json").exists()
    assert not (scenario["stage"] / "adaptation" / "update.json").exists()
    if failure != "unreturned":
        assert audit["new_label_rows"] == 3
    assert case.smoke.run_frozen_rollout is scenario["frozen"]
    assert case.smoke.run_continuous_correction_update is scenario["update"]


@pytest.mark.parametrize("failure,paid", [("paid_update", 1), ("corrupt_progress", None)])
def test_should_retain_failed_paid_update_report_or_unknown_charge_without_candidate_controls(
    scenario, failure, paid
):
    scenario["observed"]["failure"] = failure
    audit = dict(complete=False, optimizer_updates=0, new_label_rows=0)
    with pytest.raises((AssertionError, ValueError)):
        case.run_case(audit)
    assert len(scenario["observed"]["native"]) == 1 and len(scenario["observed"]["updates"]) == 1
    assert audit["update_attempted"] is True
    assert audit["optimizer_updates"] == paid
    assert audit["training_rows"] == (3 if paid == 1 else None)
    assert audit["observed_paid_receipt_count"] == (1 if paid == 1 else 8)
    assert len(audit["update_observed"]["applied_updates"]) == (1 if paid == 1 else 8)
    assert audit["new_label_rows"] == 3
    assert not (scenario["stage"] / "adaptation" / "updated-17.json").exists()
    assert case.smoke.run_frozen_rollout is scenario["frozen"]
    assert case.smoke.run_continuous_correction_update is scenario["update"]


@pytest.mark.parametrize(
    "failure,native_calls",
    [("update_replaced", 1), ("preceding_replaced", 2), ("candidate_state", 2)],
)
def test_should_revalidate_paid_update_previous_trace_and_full_candidate_before_more_controls(
    scenario, failure, native_calls
):
    scenario["observed"]["failure"] = failure
    audit = dict(complete=False, optimizer_updates=0, new_label_rows=0)
    with pytest.raises((AssertionError, ValueError)):
        case.run_case(audit)
    assert len(scenario["observed"]["native"]) == native_calls
    assert len(scenario["observed"]["updates"]) == 1
    assert audit["optimizer_updates"] == 8 and audit["training_rows"] == 24
    assert audit["update_validated"] is True
    assert len(audit["update_observed"]["applied_updates"]) == 8
    assert not (scenario["stage"] / "adaptation" / "updated-19.json").exists()
    assert case.smoke.run_frozen_rollout is scenario["frozen"]
    assert case.smoke.run_continuous_correction_update is scenario["update"]


@pytest.mark.parametrize("name", ["policy-rollout.json", "rollout-audit.json", "adaptation"])
def test_should_preserve_existing_evidence_before_loading_or_wrapping_anything(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    path = tmp_path / name
    if name == "adaptation":
        path.mkdir()
        (path / "marker").write_bytes(b"preserved")
    else:
        path.write_bytes(b"preserved")
    monkeypatch.setattr(case, "run_case", forbidden)
    with pytest.raises(FileExistsError):
        case.main()
    assert path.exists()
    if name == "adaptation":
        assert (path / "marker").read_bytes() == b"preserved"
    else:
        assert path.read_bytes() == b"preserved"


def test_should_refuse_output_alias_before_source_read_or_native_callbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "collection.json")])
    monkeypatch.setattr(case, "run_case", forbidden)
    with pytest.raises(AssertionError):
        case.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["runtime", "summary"])
def test_should_persist_failed_paid_evidence_and_refuse_complete_on_summary_failure(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])

    def run(audit):
        audit.update(
            complete=True,
            optimizer_updates=1,
            training_rows=3,
            new_label_rows=3,
            update_observed=dict(
                complete=False,
                applied_updates=[dict(update_number=1)],
                progress=dict(updates_completed=1),
            ),
        )
        if failure == "runtime":
            raise RuntimeError("paid update stage failed")

    monkeypatch.setattr(case, "run_case", run)
    original = case.prior.write_json

    def write(path, value):
        if failure == "summary" and path.name == "policy-rollout.json":
            raise OSError("primary summary write failed")
        original(path, value)

    monkeypatch.setattr(case.prior, "write_json", write)
    assert case.main() == 1
    audit = json.loads((tmp_path / "rollout-audit.json").read_text())
    assert audit["complete"] is False and audit["optimizer_updates"] == 1
    assert audit["training_rows"] == audit["new_label_rows"] == 3
    assert audit["update_observed"]["applied_updates"] == [dict(update_number=1)]
    assert ("summary_write_error" if failure == "summary" else "error") in audit
