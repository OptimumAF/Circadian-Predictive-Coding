"""Small literal TRAIN probes; no recorded datasets, initialization or simulator."""

from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest

from src.adapters import continuous_probe as probe
from src.adapters import continuous_nearest_probe as helper
from src.adapters import numpy_nearest_continuous_bc as near
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core import continuous_bc_diagnostics as diagnostics
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from scripts import push_cube_reserved_probe_inputs as loader

WORK = dict(
    sources=0,
    neural_owners=0,
    appends=0,
    append_rows=0,
    retrievals=0,
    queries=0,
    candidates=0,
    features=0,
    kernels=0,
    pairs=0,
    conversions=0,
    snapshots=0,
    snapshot_bytes=0,
    forbidden=0,
)
CAPS = dict(
    sources=80,
    neural_owners=8,
    appends=60,
    append_rows=240,
    retrievals=256,
    queries=1024,
    candidates=8192,
    features=98304,
    kernels=256,
    pairs=32768,
    conversions=65536,
    snapshots=65536,
    snapshot_bytes=256 * 1024**2,
    forbidden=0,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], "permanent fixture cap: " + key


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    def forbidden(*args, **kwargs):
        charge("forbidden")

    for kind in (
        near.NumpyNearestContinuousBCLearner,
        bp.NumpyContinuousBCLearner,
        pc.NumpyContinuousPCLearner,
    ):
        monkeypatch.setattr(kind, "__init__", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    original_append = near.NumpyNearestContinuousBCLearner.train_batch
    original_predict = near.NumpyNearestContinuousBCLearner.predict_with_work
    original_snapshot = near.NumpyNearestContinuousBCLearner.snapshot_state
    original_kernel = diagnostics.continuous_input_target_consistency

    def append(owner, features, targets):
        charge("appends")
        charge("append_rows", len(features))
        return original_append(owner, features, targets)

    def predict(owner, features):
        charge("retrievals")
        charge("queries", len(features))
        candidates = len(features) * owner._state.training_examples
        charge("candidates", candidates)
        charge("features", candidates * owner._policy.spec.observation_dim)
        return original_predict(owner, features)

    def snapshot(owner):
        charge("snapshots")
        result = original_snapshot(owner)
        charge("snapshot_bytes", result.prototype_features.nbytes + result.prototype_actions.nbytes)
        return result

    def kernel(features, targets, spec, **options):
        charge("kernels")
        charge("pairs", len(features) * (len(features) - 1) // 2)
        charge("conversions", len(features) * 4)
        return original_kernel(features, targets, spec, **options)

    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "train_batch", append)
    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "predict_with_work", predict)
    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "snapshot_state", snapshot)
    monkeypatch.setattr(diagnostics, "continuous_input_target_consistency", kernel)


def source(rows=1, *, fitted=False, role="train", capacity=256):
    charge("sources")
    spec = cube_taskspace_spec()
    policy: near.NearestContinuousBCPolicy = near.FixedInputNearestContinuousBCPolicy(
        spec, max_prototypes=capacity, training_role=role
    )
    if fitted:
        policy = near.NearestContinuousBCPolicy(
            spec, ObservationStandardization((0.0,) * 12, (1.0,) * 12, 1)
        )
    owner = object.__new__(near.NumpyNearestContinuousBCLearner)
    owner._policy = policy
    features = np.zeros((rows, 12))
    features[:, 0] = np.arange(rows)
    actions = np.tile((-1.0, 0.0, 0.0, 1.0), (rows, 1))
    owner._state = near._snapshot(
        near.NearestContinuousBCSnapshot(policy, features, actions, 1 if rows else 0, rows), policy
    )
    return owner


def episode(positions=(10.0,), targets=(1.0,), *, role="probe_train"):
    obs = [tuple([float(p)] + [0.0] * 11) for p in (*positions, positions[-1] + 1)]
    return ContinuousEpisode(
        "literal-" + role,
        "literal-fresh",
        role,
        cube_taskspace_spec(),
        tuple(
            ContinuousTransition(
                i, obs[i], (target, 0.0, 0.0, 1.0), 0.0, obs[i + 1], i == len(positions) - 1, False
            )
            for i, target in enumerate(targets)
        ),
    )


def score(result, checkpoint):
    return result["checkpoints"][checkpoint]["scores"]["training"]


def test_should_learn_private_packet_without_test_input_and_preserve_source():
    owner = source()
    identity = helper.training_owner_identity(owner)
    before = owner.snapshot_state()
    result = probe.run_nearest_training_probe(owner, episode())
    assert result["complete"], result
    assert score(result, 0)["residual"]["overall"]["normalized_half_mse"] == 0.5
    assert score(result, 1)["residual"]["overall"]["normalized_half_mse"] == 0.0
    assert result["metrics"] == dict(training_fit_gain=0.5)
    assert result["plan"]["evaluation_id"] is None
    assert all(p["scores"]["training"]["role"] == "probe_train" for p in result["checkpoints"])
    assert result["work"]["score_rows_returned_by_role"] == dict(
        probe_train=2, probe_test=0, train=0
    )
    assert result["events"][0]["deferred_exact"] and result["events"][0]["success_validated"]
    assert result["work"]["native_returns"] == result["progress"]["updates_completed"] == 1
    assert score(result, 1)["selected"] == [dict(index=1, origin="probe_train", row=0)]
    assert helper.same_state(before, owner.snapshot_state())
    assert all(a is b for a, b in zip(identity, helper.training_owner_identity(owner), strict=True))


def test_should_keep_probe_conflict_and_earliest_row_tie_instead_of_claiming_fit():
    result = probe.run_nearest_training_probe(source(), episode((10.0, 10.0), (1.0, -1.0)))
    assert result["complete"] and result["metrics"]["training_fit_gain"] == 0.0
    pairs = result["training_consistency"][0]["exact_pairs"]
    assert len(pairs) == 1 and pairs[0]["pair"]["raw_target_conflict"]
    assert pairs[0]["pair"]["native_target_conflict"]
    assert [r["row"] for r in score(result, 1)["selected"]] == [0, 0]


def test_should_keep_source_conflict_and_source_priority_after_append():
    result = probe.run_nearest_training_probe(source(), episode((0.0,), (1.0,)))
    assert result["complete"] and result["metrics"]["training_fit_gain"] == 0.0
    pairs = result["training_consistency"][1]["exact_pairs"]
    assert pairs[0]["other_origin"] == "source" and pairs[0]["other_row"] == 0
    assert score(result, 1)["selected"] == [dict(index=0, origin="source", row=0)]


def test_should_distinguish_raw_from_native32_target_conflict():
    owner = source()
    actions = owner._state.prototype_actions.copy()
    actions[0, 0] = 0.1
    owner._state = near._snapshot(replace(owner._state, prototype_actions=actions), owner._policy)
    result = probe.run_nearest_training_probe(
        owner, episode((0.0,), (float(np.nextafter(0.1, np.inf)),))
    )
    assert result["complete"]
    pair = result["training_consistency"][1]["exact_pairs"][0]["pair"]
    assert pair["raw_target_conflict"] and not pair["native_target_conflict"]


def test_should_cover_cross_block_source_indices_without_widening_kernel_limit():
    result = probe.run_nearest_training_probe(source(100), episode((99.0, 100.0), (1.0, 1.0)))
    assert result["complete"]
    blocks = result["training_consistency"]
    assert max(b["rows"] for b in blocks) == 100
    pairs = [p for b in blocks for p in b["exact_pairs"]]
    assert [(p["probe_row"], p["other_origin"], p["other_row"]) for p in pairs] == [
        (0, "source", 99)
    ]
    assert [s["origin"] for s in score(result, 1)["selected"]] == ["source", "probe_train"]


@pytest.mark.parametrize("role", ["train", "probe_test", "final_test"])
def test_should_refuse_non_probe_training_role_before_public_source_observation(monkeypatch, role):
    owner = source()
    monkeypatch.setattr(owner, "snapshot_state", lambda: pytest.fail("source observed"))
    with pytest.raises(ValueError, match="probe_train"):
        probe.run_nearest_training_probe(owner, episode(role=role))


@pytest.mark.parametrize("fault", ["empty", "fitted", "role", "capacity", "oversized"])
def test_should_refuse_unqualified_or_over_capacity_packet_without_append(fault):
    owner = source(
        rows=0 if fault == "empty" else 1,
        fitted=fault == "fitted",
        role="probe_train" if fault == "role" else "train",
        capacity=1 if fault == "capacity" else 256,
    )
    packet = episode(tuple(range(33)), (1.0,) * 33) if fault == "oversized" else episode()
    before = WORK["appends"]
    with pytest.raises((ValueError, RuntimeError)):
        probe.run_nearest_training_probe(owner, packet)
    assert WORK["appends"] == before


@pytest.mark.parametrize(
    "fault", ["source_values", "source_equal_policy", "prior_pairs", "prior_score"]
)
def test_should_refuse_changed_source_or_paid_evidence_before_new_work(monkeypatch, fault):
    owner = source()
    original_kernel = diagnostics.continuous_input_target_consistency
    original_predict = near.NumpyNearestContinuousBCLearner.predict_with_work
    original_seal = probe.paid_report_guard
    if fault.startswith("source"):

        def kernel(*args, **kwargs):
            result = original_kernel(*args, **kwargs)
            if fault == "source_values":
                actions = owner._state.prototype_actions
                actions.setflags(write=True)
                actions[0, 0] = 0.25
            else:
                object.__setattr__(owner._state, "policy", replace(owner._state.policy))
            return result

        monkeypatch.setattr(diagnostics, "continuous_input_target_consistency", kernel)
    else:
        captured = []

        def guard(value):
            result = original_seal(value)
            if (fault == "prior_pairs" and "exact_pairs" in value) or (
                fault == "prior_score" and "scores" in value
            ):
                captured.append(value)
            return result

        monkeypatch.setattr(probe, "paid_report_guard", guard)

        def predict(*args, **kwargs):
            result = original_predict(*args, **kwargs)
            if captured and (fault != "prior_score" or captured[0].get("complete") is True):
                captured[0].clear()
            return result

        monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "predict_with_work", predict)
    result = probe.run_nearest_training_probe(owner, episode())
    assert not result["complete"]
    assert result["work"]["training_attempts"] == (1 if fault == "prior_score" else 0)


def test_should_preserve_unknown_append_when_native_acknowledgement_is_lost(monkeypatch):
    original = near.NumpyNearestContinuousBCLearner.train_batch

    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("append acknowledgement lost")

    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "train_batch", lost)
    result = probe.run_nearest_training_probe(source(), episode())
    assert not result["complete"] and result["work"]["training_attempts"] == 1
    assert result["work"]["native_returns"] == 0 and result["work"]["unreturned_append_possible"]


def test_should_keep_existing_nearest_pair_default_separate():
    result = probe.run_continuous_probe(
        source(), probe.ContinuousProbePlan(episode(), episode((10.1,), (0.5,), role="probe_test"))
    )
    assert result["complete"]
    assert result["definition"] == "continuous_disposable_nearest_probe_v1"
    assert "training_consistency" not in result and "fresh_acquisition_gain" in result["metrics"]
    assert "selected" not in result["checkpoints"][0]["scores"]["fresh"]


@pytest.mark.parametrize("family", ["bp", "pc", "neutral-cpc", "active-cpc"])
def test_should_keep_existing_four_neural_dispatch_defaults_without_math(monkeypatch, family):
    charge("neural_owners")
    owner = object.__new__(
        bp.NumpyContinuousBCLearner if family == "bp" else pc.NumpyContinuousPCLearner
    )
    sentinel = object()

    def run(source, plan, **kwargs):
        assert source is owner and type(plan) is probe.ContinuousProbePlan
        assert kwargs == dict(training_readouts=(), check_boundary=None)
        return sentinel

    monkeypatch.setattr(probe, "_run_single_probe", run)
    assert (
        probe.run_continuous_probe(
            owner, probe.ContinuousProbePlan(episode(), episode(role="probe_test"))
        )
        is sentinel
    )


def qualified_episode(role):
    binding = pick_cube_probe_binding(role)
    action = (-1.0, 0.0, 0.0, 1.0)
    observations = [
        tuple([0.0, float(i)] + [0.0] * 5 + list((0.0,) * 4 if i == 0 else action) + [0.0])
        for i in range(19)
    ]
    return ContinuousEpisode(
        binding.episode_id,
        binding.context,
        role,
        cube_taskspace_spec(),
        tuple(
            ContinuousTransition(
                i, observations[i], action, 0.0, observations[i + 1], i == 17, False
            )
            for i in range(18)
        ),
    )


@pytest.fixture
def files(tmp_path, monkeypatch):
    paths = {}
    for role in ("probe_train", "probe_test"):
        path = tmp_path / (role + ".json")
        raw = json.dumps(asdict(qualified_episode(role))).encode()
        path.write_bytes(raw)
        paths[role] = (path, hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(loader, "PROBES", paths)
    return paths


def test_should_load_only_original_training_role_without_opening_test(files, monkeypatch):
    original = loader._bytes
    opened = []

    def read(path):
        assert path != files["probe_test"][0]
        opened.append(path)
        return original(path)

    monkeypatch.setattr(loader, "_bytes", read)
    packet, receipt = loader.load_reserved_training_input()
    assert packet.role == receipt["role"] == "probe_train"
    assert opened == [files["probe_train"][0]] * 2


def test_should_preserve_original_pair_loader_defaults(files):
    plan, receipts = loader.load_reserved_probe_inputs()
    assert plan.diagnostics is True and plan.updates == 1 and not plan.zero_velocity_control
    assert [r["role"] for r in receipts] == ["probe_train", "probe_test"]


@pytest.mark.parametrize("fault", ["digest", "changed_bytes", "decoded_values"])
def test_should_preserve_full_training_byte_and_decoded_qualification(files, monkeypatch, fault):
    if fault == "digest":
        files["probe_train"][0].write_text("{}")
    elif fault == "changed_bytes":
        original = loader.read_continuous_probe_episode

        def read(path, **kwargs):
            result = original(path, **kwargs)
            path.write_text("{}")
            return result

        monkeypatch.setattr(loader, "read_continuous_probe_episode", read)
    else:
        original = loader.read_continuous_probe_episode

        def read(path, **kwargs):
            packet, receipt = original(path, **kwargs)
            return replace(packet, episode_id=packet.episode_id + "-changed"), receipt

        monkeypatch.setattr(loader, "read_continuous_probe_episode", read)
    with pytest.raises(ValueError):
        loader.load_reserved_training_input()


def test_should_report_original_earliest_row_for_equal_nonzero_distances():
    owner = source(2)
    features = owner._state.prototype_features.copy()
    features[:, 0] = (-1.0, 1.0)
    owner._state = near._snapshot(replace(owner._state, prototype_features=features), owner._policy)
    result = probe.run_nearest_training_probe(owner, episode((0.0,), (1.0,)))
    assert result["complete"]
    assert score(result, 0)["selected"] == [dict(index=0, origin="source", row=0)]
    assert score(result, 1)["selected"] == [dict(index=2, origin="probe_train", row=0)]


def test_should_keep_consistency_work_after_its_acknowledgement_is_lost(monkeypatch):
    original = diagnostics.continuous_input_target_consistency

    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("consistency acknowledgement lost")

    monkeypatch.setattr(diagnostics, "continuous_input_target_consistency", lost)
    result = probe.run_nearest_training_probe(source(), episode((10.0, 11.0), (1.0, -1.0)))
    assert not result["complete"]
    work = result["work"]["training_consistency"]
    assert work["attempts"] == 1 and work["returns"] == 0
    assert work["pair_returns"] == 1 and work["native_conversion_returns"] == 8
    assert result["work"]["training_attempts"] == 0
