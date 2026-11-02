"""NEW literal bounded TRAIN-only probes; no recorded input, evaluation or SDK."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from src.adapters import continuous_probe as probe
from src.adapters import continuous_nearest_probe as near
from src.adapters import continuous_stream_sequence as sequence
from src.adapters import continuous_retained_training as content
from src.adapters import numpy_retained_nearest_bc as retained
from src.adapters import numpy_nearest_continuous_bc as plain
from src.core import nearest_retention as kernel
from src.core import replay_retention as replay
from src.core.nearest_retention import RecentPriorityRetentionPolicy as Recent
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.cube_taskspace_state import cube_taskspace_spec
from test_retained_nearest_memory import _process_peak_bytes
from test_nearest_training_probe import episode

WORK: dict[str, int] = {}
PRIOR: dict[str, int] = {}
CAPS = dict(
    initializers=128,
    updates=2048,
    incoming_rows=4096,
    forks=4096,
    predictions=4096,
    query_rows=8192,
    candidates=524288,
    coordinates=6291456,
    snapshots=262144,
    validators=1048576,
    copy_bytes=2 * 1024**3,
    hash_bytes=4 * 1024**3,
    residual_rows=8192,
    endpoints=256,
    plain_owners=16,
    consistency_calls=1024,
    consistency_rows=65536,
    consistency_pairs=131072,
    consistency_conversions=262144,
    mapping_calls=512,
    mapping_rows=4096,
    content_hash_rows=65536,
    content_hash_bytes=64 * 1024**2,
    selections=512,
    id_arrivals=4096,
    id_candidates=65536,
    rank_calls=65536,
    rank_bytes=4 * 1024**2,
)
FIFO = ReplayRetentionPolicy("recent_fifo")


def charge(key, count=1):
    WORK[key] = WORK.get(key, 0) + count
    assert WORK[key] + PRIOR.get(key, 0) <= CAPS[key], key


@pytest.fixture(scope="module", autouse=True)
def bounded_literal_work():
    PRIOR.update(json.loads(os.environ.get("RETAINED_TRAINING_PRIOR", "{}")))

    def wrap(method, kind):
        def call(owner, *args, **kwargs):
            charge(kind)
            if kind == "updates":
                charge("incoming_rows", len(args[0]))
            if kind == "predictions":
                rows = len(args[0])
                candidates = rows * len(owner._state.prototype_features)
                for key, count in (
                    ("query_rows", rows),
                    ("candidates", candidates),
                    ("coordinates", candidates * 12),
                ):
                    charge(key, count)
            return method(owner, *args, **kwargs)

        return call

    validate, ids = retained.retained_snapshot, retained._ids
    residual = probe.diagnostics.continuous_bc_residuals
    select = kernel.select_retained_rows
    original_rank = replay.seeded_content_rank
    consistency = probe.diagnostics.continuous_input_target_consistency
    describe = content.describe_selection
    content_hash = content._hash_ids

    def ranked(seed, content_id):
        charge("rank_calls")
        charge("rank_bytes", 64)
        return original_rank(seed, content_id)

    def selected(prior, incoming, maximum, policy):
        charge("selections")
        charge("id_arrivals", len(incoming))
        # Upper bound before any selection/eviction: cap+1 for every arrival.
        charge("id_candidates", len(incoming) * (maximum + 1))
        return select(prior, incoming, maximum, policy)

    def validated(state, policy, rule):
        charge("validators")
        charge("copy_bytes", state.prototype_features.nbytes + state.prototype_actions.nbytes)
        return validate(state, policy, rule)

    def hashed(features, actions):
        charge("hash_bytes", features.nbytes + actions.nbytes + 24 * len(features))
        return ids(features, actions)

    def residuals(*args, **kwargs):
        charge("residual_rows", len(args[0]))
        return residual(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("literal workflow attempted model RNG")

    def consistent(features, targets, spec, **options):
        charge("consistency_calls")
        charge("consistency_rows", len(features))
        charge("consistency_pairs", len(features) * (len(features) - 1) // 2)
        charge("consistency_conversions", len(features) * 4)
        return consistency(features, targets, spec, **options)

    def described(state, binding, features, targets, indices, work, publish):
        charge("mapping_calls")
        charge("mapping_rows", len(features))
        return describe(state, binding, features, targets, indices, work, publish)

    def content_hashed(features, targets, work, publish):
        charge("content_hash_rows", len(features))
        charge("content_hash_bytes", features.nbytes + targets.nbytes + 24 * len(features))
        return content_hash(features, targets, work, publish)

    with pytest.MonkeyPatch.context() as patch:
        for name, kind in (
            ("__init__", "initializers"),
            ("train_batch", "updates"),
            ("fork", "forks"),
            ("predict_with_work", "predictions"),
            ("snapshot_state", "snapshots"),
        ):
            patch.setattr(
                retained.NumpyRetainedNearestBCLearner,
                name,
                wrap(getattr(retained.NumpyRetainedNearestBCLearner, name), kind),
            )
        for name, kind in (
            ("train_batch", "updates"),
            ("fork", "forks"),
            ("predict_with_work", "predictions"),
            ("snapshot_state", "snapshots"),
        ):
            patch.setattr(
                plain.NumpyNearestContinuousBCLearner,
                name,
                wrap(getattr(plain.NumpyNearestContinuousBCLearner, name), kind),
            )
        for module in (retained, near, sequence):
            patch.setattr(module, "retained_snapshot", validated)
        patch.setattr(retained, "select_retained_rows", selected)
        patch.setattr(kernel, "select_retained_rows", selected)
        patch.setattr(kernel, "seeded_content_rank", ranked)
        patch.setattr(replay, "seeded_content_rank", ranked)
        patch.setattr(probe.diagnostics, "continuous_input_target_consistency", consistent)
        patch.setattr(content, "describe_selection", described)
        patch.setattr(content, "_hash_ids", content_hashed)
        patch.setattr(retained, "_ids", hashed)
        patch.setattr(probe.diagnostics, "continuous_bc_residuals", residuals)
        patch.setattr(np.random, "default_rng", forbidden)
        yield
    peak = _process_peak_bytes()
    receipt = os.environ.get("RETAINED_TRAINING_RECEIPT")
    if receipt:
        with Path(receipt).open("x") as handle:
            json.dump(
                dict(
                    work=WORK,
                    aggregate={k: WORK.get(k, 0) + PRIOR.get(k, 0) for k in CAPS},
                    peak_bytes=peak,
                ),
                handle,
                indent=2,
            )
    assert peak <= 512 * 1024**2


SPEC = cube_taskspace_spec()


def pairs(xs, actions):
    x = np.zeros((len(xs), 12), dtype=np.float64)
    x[:, 0] = xs
    y = np.asarray([(a, 0.0, 0.0, 1.0) for a in actions], dtype=np.float64)
    return x, y


def source(xs=(0.0, 1.0), actions=(-1.0, -1.0), *, maximum=2, policy=FIFO):
    result = retained.NumpyRetainedNearestBCLearner(
        plain.FixedInputNearestContinuousBCPolicy(SPEC, max_prototypes=maximum), policy
    )
    if xs:
        result.train_batch(*pairs(xs, actions))
    return result


def run(owner=None, packet=None, **options):
    return probe.run_retained_nearest_training_probe(
        source() if owner is None else owner, episode() if packet is None else packet, **options
    )


def old_readout(positions=(0.0,), actions=(-1.0,), id="old"):
    return replace(episode(positions, actions, role="train"), episode_id=id, context="old")


def score(result, point=1, name="training"):
    return result["checkpoints"][point]["scores"][name]


@pytest.mark.parametrize("rule", [FIFO, ReplayRetentionPolicy("seeded_reservoir", 17), Recent(17)])
def test_private_training_only_uses_actual_rule_and_preserves_source(rule):
    owner = source(policy=rule)
    birth = owner.snapshot_state()
    pin = retained.pin_retained_native(owner)
    seen = []

    def endpoint(child, phase, guard):
        charge("endpoints")
        seen.append(phase)
        guard()

    result = run(owner, training_readouts=(old_readout(),), endpoint_observer=endpoint)
    assert result["complete"], result.get("error")
    assert result["definition"] == "continuous_retained_nearest_training_probe_v1"
    assert result["plan"]["evaluation_id"] is None and seen == ["before_probe", "after_probe"]
    assert result["work"]["score_rows_returned_by_role"] == dict(
        probe_train=2, probe_test=0, train=2
    )
    assert set(score(result)["content_coverage"]) >= {
        "query_content_rows_retained",
        "distinct_query_ids",
    }
    assert result["work"]["reserved_numeric_bytes"] == 128
    assert result["progress"] == dict(updates_completed=1, replay_examples_completed=0, next_tick=2)
    assert result["events"][0]["deferred_exact"] and result["events"][0]["success_validated"]
    assert result["metrics"]["old_retention_loss"]["old_0"] >= 0
    assert result["work"]["retained_content"]["mapping_returns"] == 4
    assert result["work"]["nearest_retrieval"]["attempts"] == 4
    costs = result["work"]["retained_content"]
    assert costs["hash_calls_attempted"] == costs["hash_calls_returned"] == 5
    assert costs["hash_numeric_materialization_bytes_attempted_upper_bound"] == 1280
    assert costs["hash_numeric_materialization_bytes_returned_upper_bound"] == 1280
    assert costs["set_input_id_occurrences_upper_bound"] == 20
    retained.require_retained_native(owner, pin)
    assert retained.same_retained_state(birth, owner.snapshot_state())


def test_actual_fit_interference_eviction_and_compacted_origin_are_observed():
    owner = source()
    incoming = episode()
    old = old_readout()
    result = run(owner, incoming, training_readouts=(old,))
    assert result["complete"] and result["metrics"]["training_fit_gain"] == 0.5
    assert result["metrics"]["old_retention_loss"] == dict(old_0=0.0)
    selected = score(result)["selected"][0]
    incoming_id = retained._ids(*pairs((10.0,), (1.0,)))[0]
    assert selected == dict(
        index=1,
        content_id=incoming_id,
        matches_source=False,
        matches_incoming=True,
        query_content_retained=True,
        selected_content_exact=True,
        selected_input_equal=True,
    )
    assert score(result, 1, "old_0")["selected"][0]["index"] == 0
    assert score(result, 1, "old_0")["selected"][0]["matches_source"]
    assert score(result, 1, "old_0")["content_coverage"]["query_content_rows_retained"] == 0
    assert result["retention"]["candidate_after_return"]["training_examples"] == 3
    assert result["retention"]["candidate_after_return"]["retained_rows"] == 2


def test_repeated_content_has_both_origins_and_occurrences_are_not_distinct_ids():
    owner = source(policy=Recent(17))
    packet = episode((0.0, 0.0), (-1.0, -1.0))
    result = run(owner, packet)
    assert result["complete"] and result["metrics"]["training_fit_gain"] == 0
    assert all(x["matches_source"] and x["matches_incoming"] for x in score(result)["selected"])
    c = score(result)["content_coverage"]
    assert c["incoming_occurrences"] == 2 and c["incoming_content_ids"] == 1
    assert (
        c["query_rows"] == 2
        and c["distinct_query_ids"] == 1
        and c["query_content_rows_retained"] == 2
    )
    assert c["shared_content_rows_retained"] == 1


def test_equal_inputs_with_conflicting_targets_keep_earliest_tie_and_explain_coverage():
    result = run(source((0.0,), (-1.0,), maximum=4), episode((0.0,), (1.0,)))
    assert result["complete"] and result["metrics"]["training_fit_gain"] == 0
    row = score(result)["selected"][0]
    c = score(result)["content_coverage"]
    assert row["index"] == 0 and row["matches_source"] and not row["matches_incoming"]
    assert (
        row["query_content_retained"]
        and not row["selected_content_exact"]
        and row["selected_input_equal"]
    )
    assert c["query_content_rows_retained"] == 1 and c["selected_content_exact_rows"] == 0
    assert result["training_consistency"][1]["exact_pairs"][0]["pair"]["raw_target_conflict"]


@pytest.mark.parametrize("role", ["train", "probe_test", "final"])
def test_wrong_training_role_refuses_before_fork(monkeypatch, role):
    owner = source()
    monkeypatch.setattr(owner, "fork", lambda: pytest.fail("invalid role forked"))
    with pytest.raises(ValueError):
        run(owner, episode(role=role))


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "source_type",
        "readout_role",
        "readout_count",
        "readout_spec",
        "readout_duplicate",
        "same_id",
        "source_spec",
        "callback",
    ],
)
def test_invalid_owners_inputs_and_readouts_refuse_without_private_append(monkeypatch, fault):
    owner = source()
    packet = episode()
    options: dict[str, Any] = {}
    if fault == "empty":
        owner = source((), ())
    elif fault == "source_type":
        owner = object()
    elif fault == "readout_role":
        options["training_readouts"] = (episode(role="probe_test"),)
    elif fault == "readout_count":
        options["training_readouts"] = tuple(old_readout(id=str(i)) for i in range(5))
    elif fault == "readout_spec":
        options["training_readouts"] = (
            replace(old_readout(), spec=replace(SPEC, action_units=("other",) * 4)),
        )
    elif fault == "readout_duplicate":
        options["training_readouts"] = (old_readout(), old_readout())
    elif fault == "same_id":
        options["training_readouts"] = (replace(old_readout(), episode_id=packet.episode_id),)
    elif fault == "source_spec":
        packet = replace(packet, spec=replace(SPEC, action_units=("other",) * 4))
    else:
        options["endpoint_observer"] = 1
    before = WORK.get("updates", 0)
    with pytest.raises((ValueError, TypeError)):
        run(owner, packet, **options)
    assert WORK.get("updates", 0) == before


def test_four_original_train_readouts_need_no_evaluation_view(monkeypatch):
    original = probe._score_view
    roles = []

    def viewed(e):
        roles.append(e.role)
        assert e.role in ("train", "probe_train")
        return original(e)

    monkeypatch.setattr(probe, "_score_view", viewed)
    old = tuple(old_readout(id=str(i)) for i in range(4))
    result = run(training_readouts=old)
    assert result["complete"] and len(roles) == 10 and "probe_test" not in roles
    assert len(result["metrics"]["old_retention_loss"]) == 4


@pytest.mark.parametrize(
    "fault", ["source_action", "source_owner", "endpoint_action", "endpoint_owner"]
)
def test_mutated_source_or_endpoint_cannot_complete(fault):
    owner = source()

    def endpoint(child, phase, guard):
        charge("endpoints")
        if phase == "before_probe":
            if fault == "source_action":
                owner._state.prototype_actions[0, 0] = 0.25
            elif fault == "source_owner":
                owner._retention = ReplayRetentionPolicy("recent_fifo")
            elif fault == "endpoint_action":
                child._state.prototype_actions[0, 0] = 0.25
            else:
                child._retention = ReplayRetentionPolicy("recent_fifo")

    result = run(owner, endpoint_observer=endpoint)
    assert not result["complete"] and result["work"]["training_attempts"] == 0


def test_lost_append_acknowledgement_preserves_permanent_attempt_and_unknown_work(monkeypatch):
    owner = source()
    original = retained.NumpyRetainedNearestBCLearner.train_batch

    def lost(*args, **kw):
        original(*args, **kw)
        raise RuntimeError("ack lost")

    monkeypatch.setattr(retained.NumpyRetainedNearestBCLearner, "train_batch", lost)
    result = run(owner)
    assert not result["complete"] and result["work"]["training_attempts"] == 1
    assert result["work"]["native_returns"] == 0 and result["work"]["unreturned_append_possible"]
    assert (
        result["work"]["append_attempts_permanently_spent"] == 1
        and result["work"]["append_slots_remaining"] == 0
    )


@pytest.mark.parametrize("fault", ["erase", "replace", "value"])
def test_changed_prior_paid_content_scores_are_refused_and_preserved(monkeypatch, fault):
    owner = source()
    paid = []
    original = probe.paid_report_guard

    def capture(report):
        if isinstance(report, dict) and "scores" in report:
            paid.append(report)
        return original(report)

    monkeypatch.setattr(probe, "paid_report_guard", capture)
    predict = retained.NumpyRetainedNearestBCLearner.predict_with_work

    def corrupted(*args, **kwargs):
        value = predict(*args, **kwargs)
        if paid and paid[0].get("complete"):
            row = paid[0]["scores"]["training"]
            if fault == "erase":
                row.clear()
            elif fault == "replace":
                paid[0]["scores"]["training"] = deepcopy(row)
            else:
                row["content_coverage"]["prototype_rows"] = 999
        return value

    monkeypatch.setattr(retained.NumpyRetainedNearestBCLearner, "predict_with_work", corrupted)
    result = run(owner)
    assert not result["complete"] and result["work"]["training_attempts"] == 1
    assert score(result, 0)["content_coverage"]["prototype_rows"] == 2


def test_mapping_hash_failure_is_paid_and_does_not_admit_private_update(monkeypatch):
    owner = source()
    original = content._hash_ids
    calls = 0

    def lost(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Delegate itself must record attempted bytes before failing.

            def failed(*a):
                raise RuntimeError("diagnostic hash lost")

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(retained, "_ids", failed)
                return original(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(content, "_hash_ids", lost)
    result = run(owner)
    assert not result["complete"] and result["work"]["training_attempts"] == 0
    c = result["work"]["retained_content"]
    assert c["hash_calls_attempted"] == 2 and c["hash_calls_returned"] == 1
    assert c["mapping_attempts"] == 1 and c.get("mapping_returns", 0) == 0
    assert c["hash_numeric_materialization_bytes_attempted_upper_bound"] == 512
    assert c["hash_numeric_materialization_bytes_returned_upper_bound"] == 256


def test_old_append_only_training_admission_is_not_widened():
    with pytest.raises(ValueError):
        probe.run_nearest_training_probe(source(), episode())


def test_old_paired_retained_route_keeps_its_schema_and_dose():
    owner = source()
    p = probe.ContinuousProbePlan(episode(), episode((10.1,), (1.0,), role="probe_test"))
    result = probe.run_retained_nearest_probe(owner, p)
    assert result["complete"] and "training_consistency" not in result
    assert "retained_content" not in result["work"]
    assert "selected" not in result["checkpoints"][0]["scores"]["fresh"]
    assert (
        result["progress"]["updates_completed"] == 1
        and "fresh_acquisition_gain" in result["metrics"]
    )


def test_real_old_task_forgetting_is_reported_even_when_private_fit_improves():
    result = run(source((0.0, 1.0), (-1.0, 1.0)), training_readouts=(old_readout(),))
    assert result["complete"]
    assert score(result, 0, "old_0")["residual"]["overall"]["normalized_half_mse"] == 0
    assert score(result, 1, "old_0")["residual"]["overall"]["normalized_half_mse"] == 0.5
    assert result["metrics"]["old_retention_loss"] == dict(old_0=0.5)
    assert score(result)["content_coverage"]["query_content_rows_retained"] == 1
    assert score(result, 1, "old_0")["content_coverage"]["query_content_rows_retained"] == 0


def test_old_append_only_training_route_retains_actual_selected_origin_schema():
    charge("plain_owners")
    owner = object.__new__(plain.NumpyNearestContinuousBCLearner)
    owner._policy = plain.FixedInputNearestContinuousBCPolicy(SPEC, max_prototypes=4)
    x, y = pairs((0.0,), (-1.0,))
    owner._state = plain._snapshot(
        plain.NearestContinuousBCSnapshot(owner._policy, x, y, 1, 1), owner._policy
    )
    result = probe.run_nearest_training_probe(owner, episode())
    assert result["complete"] and result["metrics"] == dict(training_fit_gain=0.5)
    assert score(result)["selected"] == [dict(index=1, origin="probe_train", row=0)]
    assert "retained_content" not in result["work"] and "content_coverage" not in score(result)
