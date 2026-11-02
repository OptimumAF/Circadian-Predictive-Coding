"""NEW recent-priority behavior and ownership tests; literal inputs only."""

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from src.core import nearest_retention as kernel
from src.core import replay_retention as replay
from src.core.nearest_retention import RecentPriorityRetentionPolicy as Recent
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.circadian_predictive_coding import (
    ReplayRetentionBudget,
    CircadianPredictiveCodingNetwork,
)
from src.core.shared_replay_schedule import SharedReplayBuffer
from src.adapters import numpy_retained_nearest_bc as retained
from src.adapters import continuous_nearest_probe as near
from src.adapters import continuous_stream_sequence as sequence
from src.adapters import continuous_probe as probe
from src.adapters import taskspace_memory_demo as demo
from src.adapters.numpy_nearest_continuous_bc import FixedInputNearestContinuousBCPolicy
from scripts import run_taskspace_memory_demo as cli
from scripts import push_cube_reserved_probe_inputs as inputs
from scripts import taskspace_memory_native_adaptation as native
from test_retained_adaptation_workflow import plan
from test_taskspace_memory_demo import bundle
from test_retained_nearest_memory import _process_peak_bytes

WORK: dict[str, int] = {}
PRIOR: dict[str, int] = {}
CAPS = dict(
    initializers=160,
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
    PRIOR.update(json.loads(os.environ.get("RECENT_PRIORITY_PRIOR", "{}")))

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
        for module in (retained, near, sequence):
            patch.setattr(module, "retained_snapshot", validated)
        patch.setattr(retained, "select_retained_rows", selected)
        patch.setattr(kernel, "select_retained_rows", selected)
        patch.setattr(kernel, "seeded_content_rank", ranked)
        patch.setattr(replay, "seeded_content_rank", ranked)
        patch.setattr(retained, "_ids", hashed)
        patch.setattr(probe.diagnostics, "continuous_bc_residuals", residuals)
        patch.setattr(np.random, "default_rng", forbidden)
        yield
    peak = _process_peak_bytes()
    receipt = os.environ.get("RECENT_PRIORITY_RECEIPT")
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


def ident(index):
    return f"{index:064x}"


def independent_rank(seed, row_id):
    charge("rank_calls")
    charge("rank_bytes", 64)
    return sha256(
        b"numpy_replay_bottom_k_v1" + seed.to_bytes(8, "little") + bytes.fromhex(row_id)
    ).digest()


@pytest.mark.parametrize("seed", [0, 17, 2**64 - 1])
@pytest.mark.parametrize("row_id", ["00" * 32, "ff" * 32, "0123456789abcdef" * 4])
def test_shared_hash_keeps_original_replay_and_nearest_priority_bytes(seed, row_id):
    expected = independent_rank(seed, row_id)
    assert (
        replay.ReplayRetentionPolicy("seeded_reservoir", seed)._reservoir_rank(row_id) == expected
    )
    assert Recent(seed)._reservoir_rank(row_id) == expected


@pytest.mark.parametrize("maximum", [2, 3, 4, 8])
def test_recent_and_global_priority_membership_matches_independent_history_oracle(maximum):
    history = [ident(i) for i in (0, 1, 2, 3, 4, 5, 0, 6, 1, 7, 2, 0)]
    prior: tuple[str, ...] = ()
    seen = set()
    touched: dict[str, None] = {}
    duplicates = discarded = ranks = 0
    for count, row_id in enumerate(history, 1):
        value = kernel.select_retained_rows(prior, (row_id,), maximum, Recent(17))
        seen.add(row_id)
        touched.pop(row_id, None)
        touched[row_id] = None
        recent = set(tuple(touched)[-(maximum // 2) :])
        best = set(
            sorted(seen, key=lambda key: (independent_rank(17, key), key))[: maximum - maximum // 2]
        )
        assert recent <= set(value.ids) and best <= set(value.ids)
        assert len(value.ids) == min(maximum, len(seen)) and len(set(value.ids)) == len(value.ids)
        duplicates += value.duplicate_rows
        discarded += value.discarded_rows
        ranks += value.reservoir_rank_evaluations
        assert count == len(value.ids) + duplicates + discarded
        assert ranks == discarded * (maximum + 1 - maximum // 2)
        prior = value.ids
    batched = kernel.select_retained_rows((), tuple(history), maximum, Recent(17))
    assert batched.ids == prior
    assert (batched.duplicate_rows, batched.discarded_rows, batched.reservoir_rank_evaluations) == (
        duplicates,
        discarded,
        ranks,
    )


def test_duplicate_refreshes_only_new_mode_and_points_to_identical_incoming_content(monkeypatch):
    monkeypatch.setattr(kernel, "seeded_content_rank", lambda seed, row_id: b"same")
    a, b, c, d, e, f = map(ident, range(6))
    first = kernel.select_retained_rows((), (a, b, c, d, e), 4, Recent(17))
    assert first.ids == (a, b, d, e)
    refreshed = kernel.select_retained_rows(first.ids, (a,), 4, Recent(17))
    assert refreshed.ids == (b, d, e, a) and refreshed.indices[-1] == 4
    assert refreshed.duplicate_rows == 1 and refreshed.discarded_rows == 0
    next_value = kernel.select_retained_rows(refreshed.ids, (f,), 4, Recent(17))
    assert next_value.ids == (b, d, a, f)
    # Old FIFO still skips duplicate touches, as specified before this feature.
    old = kernel.select_retained_rows((a, b, c, d), (a, e), 4, ReplayRetentionPolicy("recent_fifo"))
    assert old.ids == (b, c, d, e) and old.duplicate_rows == 1


def test_old_reservoir_duplicate_order_and_rank_count_are_preserved(monkeypatch):
    monkeypatch.setattr(replay, "seeded_content_rank", lambda seed, row_id: b"same")
    ids = tuple(map(ident, range(5)))
    value = kernel.select_retained_rows(
        ids[:4], (ids[0], ids[4]), 4, ReplayRetentionPolicy("seeded_reservoir", 17)
    )
    assert value.ids == ids[:4] and value.duplicate_rows == 1 and value.discarded_rows == 1
    assert value.reservoir_rank_evaluations == 5


@pytest.mark.parametrize("seed", [None, True, -1, 2**64, 1.25])
def test_new_policy_refuses_non_uint64_seed(seed):
    with pytest.raises(ValueError):
        Recent(seed)


@pytest.mark.parametrize("bad", [1, True, 0, 257])
def test_invalid_new_capacity_refuses_before_episode_admission_or_model(monkeypatch, bad):
    monkeypatch.setattr(
        demo, "_admit", lambda *a: pytest.fail("invalid capacity reached input data")
    )
    with pytest.raises(ValueError):
        demo.run_demo({}, retention=Recent(17), max_prototypes=bad)
    with pytest.raises(ValueError):
        invalid_episode: Any = None
        sequence.run_sequence(
            invalid_episode, (), fixed_input=True, retention=Recent(17), max_prototypes=bad
        )


def test_policy_name_subclass_and_old_replay_names_cannot_widen_admission():
    changed = Recent(17)
    object.__setattr__(changed, "name", "seeded_reservoir")
    with pytest.raises(ValueError):
        kernel.retention_policy(changed)

    class Child(Recent):
        pass

    with pytest.raises(ValueError):
        kernel.retention_policy(Child(17))
    with pytest.raises(ValueError):
        ReplayRetentionPolicy("recent_priority", 17)


def test_new_nearest_policy_is_rejected_by_shared_and_cpc_replay_before_any_model():
    wrong: Any = Recent(17)
    budget = ReplayRetentionBudget(max_examples=2, max_bytes=48)
    with pytest.raises(TypeError):
        SharedReplayBuffer(2, budget, wrong)
    configure: Any = CircadianPredictiveCodingNetwork.configure_replay_retention
    with pytest.raises(TypeError):
        configure(None, budget, policy=wrong)


SPEC = ContinuousActionSpec(2, ("unit",), (-1.0,), (1.0,))


def learner(maximum=4):
    return retained.NumpyRetainedNearestBCLearner(
        FixedInputNearestContinuousBCPolicy(SPEC, max_prototypes=maximum), Recent(17)
    )


def pairs(*xs):
    return np.asarray([[x, 1.0] for x in xs], dtype=np.float64), np.asarray(
        [[x / 32] for x in xs], dtype=np.float64
    )


def test_real_child_preserves_recent_pairs_old_snapshots_and_original_source():
    owner = learner()
    owner.train_batch(*pairs(0, 1, 2, 3))
    published = owner.snapshot_state()
    birth = (
        published.prototype_features.tobytes(),
        published.prototype_actions.tobytes(),
        published.prototype_ids,
    )
    pin = retained.pin_retained_native(owner)
    child = owner.fork()
    child.train_batch(*pairs(10, 11))
    child.train_batch(*pairs(0))
    retained.require_retained_native(owner, pin)
    state = child.snapshot_state()
    recent = retained._ids(*pairs(11, 0))
    assert state.prototype_ids[-2:] == recent
    assert np.array_equal(child.predict(pairs(11, 0)[0]), pairs(11, 0)[1])
    assert (
        published.prototype_features.tobytes(),
        published.prototype_actions.tobytes(),
        published.prototype_ids,
    ) == birth
    assert not np.shares_memory(state.prototype_features, child._state.prototype_features)
    assert state.completed_updates == 3 and state.training_examples == 7
    assert (
        state.training_examples == state.retained_rows + state.duplicate_rows + state.discarded_rows
    )
    assert state.reservoir_rank_evaluations == state.discarded_rows * 3
    cost = retained.retained_costs(state)
    assert cost["numeric_bytes"] == 96 and cost["content_id_bytes"] == 256


def test_rank_failure_is_atomic_and_attempted_work_remains_charged(monkeypatch):
    owner = learner(2)
    owner.train_batch(*pairs(0, 1))
    before = owner._state

    def lost(*args):
        raise RuntimeError("rank failed")

    monkeypatch.setattr(Recent, "_reservoir_rank", lost)
    with pytest.raises(RuntimeError):
        owner.train_batch(*pairs(2))
    assert owner._state is before and owner.snapshot_state().training_examples == 2


@pytest.mark.parametrize("fault", ["owner", "seed", "counter"])
def test_new_policy_ownership_and_snapshot_counts_refuse_corruption(fault):
    owner = learner(2)
    owner.train_batch(*pairs(0, 1, 2))
    pin = retained.pin_retained_native(owner)
    if fault == "owner":
        owner._retention = Recent(17)
    elif fault == "seed":
        object.__setattr__(owner._retention, "seed", 18)
    else:
        owner._state = replace(owner._state, reservoir_rank_evaluations=0)
    with pytest.raises((RuntimeError, ValueError)):
        retained.require_retained_native(owner, pin)


def test_real_private_return_runs_through_existing_workflow_with_original_roles():
    phases = []

    def endpoint(child, phase, guard):
        charge("endpoints")
        phases.append(phase)
        pin = retained.pin_retained_native(child)
        guard()
        retained.require_retained_native(child, pin)

    result = demo.run_demo(
        bundle(),
        adaptation_probe=plan(),
        retention=Recent(17),
        max_prototypes=4,
        return_task="pick17",
        endpoint_observer=endpoint,
    )
    assert result["complete"], result.get("error")
    assert (
        result["retention"] == dict(seed=17, name="recent_priority")
        and result["native_outcomes"] is None
    )
    assert phases == ["before_probe", "after_probe", "after_return"]
    assert result["stream"]["model_state"]["candidate_training_examples"] == 7
    assert result["stream"]["experiment_state"] == dict(completed_updates=6, next_tick=18)
    a = result["adaptation"]
    assert a["complete"] and a["work"]["training_rows_attempted"] == 19
    assert a["return_phase"]["plan"]["training_role"] == "train"
    assert all(
        a[key]["progress"] == dict(updates_completed=1, replay_examples_completed=0, next_tick=2)
        for key in ("probe", "return_phase")
    )


@pytest.mark.parametrize("role", ["probe_test", "final"])
def test_new_memory_mode_does_not_admit_nontraining_policy_roles(role):
    with pytest.raises(ValueError):
        FixedInputNearestContinuousBCPolicy(SPEC, training_role=role)


def cli_args(output):
    return [
        "--retention",
        "recent_priority",
        "--retention-seed",
        "17",
        "--max-prototypes",
        "4",
        "--adaptation-probe",
        "train.json",
        "test.json",
        "--return-task",
        "pick17",
        "--output",
        str(output),
    ]


def test_cli_builds_narrow_policy_and_keeps_default_dispatch_metadata(
    tmp_path, monkeypatch, capsys
):
    p = plan()
    seen = []
    monkeypatch.setattr(
        inputs, "load_reserved_probe_inputs", lambda *a: (replace(p, diagnostics=True), [])
    )
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: {})

    def execute(payload, **options):
        seen.append(options)
        return dict(complete=True, native_outcomes=None)

    monkeypatch.setattr(cli, "run_demo", execute)
    assert cli.main(cli_args(tmp_path / "new.json")) == 0
    assert type(seen[0]["retention"]) is Recent and seen[0]["retention"].seed == 17
    assert seen[0]["adaptation_probe"].diagnostics is False
    assert json.loads(capsys.readouterr().out)["complete"]
    seen.clear()
    assert cli.main([]) == 0
    assert seen == [{}]  # Strict default passes no retention or probe options.


@pytest.mark.parametrize("fault", ["seed", "capacity"])
def test_cli_new_mode_refuses_invalid_options_before_input(tmp_path, monkeypatch, fault):
    args = cli_args(tmp_path / "new.json")
    if fault == "seed":
        i = args.index("--retention-seed")
        del args[i : i + 2]
    else:
        args[args.index("--max-prototypes") + 1] = "1"
    monkeypatch.setattr(
        cli, "_load_default_bundle", lambda: pytest.fail("invalid options reached data")
    )
    monkeypatch.setattr(
        inputs, "load_reserved_probe_inputs", lambda *a: pytest.fail("invalid options reached pair")
    )
    with pytest.raises(SystemExit):
        cli.main(args)


@pytest.mark.parametrize("fault", ["state", "owner"])
def test_corrupt_private_endpoint_refuses_return_and_preserves_paid_attempt(fault):
    phases = []

    def endpoint(child, phase, guard):
        charge("endpoints")
        phases.append(phase)
        if phase == "after_probe":
            if fault == "state":
                child._state.prototype_actions[0, 0] += 0.25
            else:
                child._retention = Recent(17)

    result = demo.run_demo(
        bundle(),
        adaptation_probe=plan(),
        retention=Recent(17),
        max_prototypes=4,
        return_task="pick17",
        endpoint_observer=endpoint,
    )
    assert not result["complete"]
    assert phases == ["before_probe", "after_probe"]
    assert result["adaptation"]["work"]["training_attempts"] == 1
    assert not result["unreturned_private_work_unknown"]
    assert result["stream"]["experiment_state"] == dict(completed_updates=6, next_tick=18)


def test_native_cli_passes_new_policy_to_existing_frozen_port_without_sdk(
    tmp_path, monkeypatch, capsys
):
    p = plan()
    receipts = [dict(role="probe_train"), dict(role="probe_test")]
    seen = []
    monkeypatch.setattr(inputs, "load_reserved_probe_inputs", lambda *a: (p, receipts))
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: {})
    monkeypatch.setattr(
        cli, "run_demo", lambda *a, **k: pytest.fail("native mode reached CPU dispatch")
    )

    def execute(payload, owned, output, **options):
        seen.append(options)
        assert owned.diagnostics is False and output == tmp_path / "native.json"
        return dict(complete=True)

    monkeypatch.setattr(native, "run_native_adaptation", execute)
    assert cli.main(cli_args(tmp_path / "native.json") + ["--native"]) == 0
    assert len(seen) == 1 and type(seen[0]["retention"]) is Recent
    assert seen[0]["retention"].seed == 17 and seen[0]["max_prototypes"] == 4
    assert seen[0]["input_receipts"] is receipts and seen[0]["return_task"] == "pick17"
    assert json.loads(capsys.readouterr().out)["complete"]
