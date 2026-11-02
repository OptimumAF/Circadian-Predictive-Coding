"""Once-per-packet composition, paid failures and ordinary isolated integration."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_native_probe as probe
from scripts.ticket_linear_rehearsal import SourceTrainPackets, LinearProbeWork, LIMITS, ONCE_LIMITS
from src.adapters.linear_fit_learner import LinearFitLearner
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.probe_update import ProbeUpdateFailed
from src.core.shared_linear_replay import compose_shared_packet
from test_shared_linear_callback import source, packet, scripted, SCRIPTED
from test_shared_linear_probe_entry import OPTIONS


def inputs():
    first = packet(6, "train", "a", 0.0)
    second = packet(6, "train", "b", 1.0)
    bank = tuple((b.features, b.targets) for b in (first, second))
    return bank


@pytest.mark.parametrize("control", ["joint", "current_only"])
@pytest.mark.parametrize("current_index", [0, 1])
def test_once_packet_keeps_each_context_once_and_control_rows_equal(control, current_index):
    bank = inputs()
    current = tuple(a.copy() for a in bank[current_index])
    pin = deepcopy(bank)
    x, y = compose_shared_packet(current, bank, 6, control=control, current_once=True)
    expected = (
        (
            np.concatenate((current[0], bank[1 - current_index][0])),
            np.concatenate((current[1], bank[1 - current_index][1])),
        )
        if control == "joint"
        else (np.tile(current[0], (2, 1)), np.tile(current[1], (2, 1)))
    )
    assert x.shape == (12, 9) and y.shape == (12, 1)
    assert np.array_equal(x, expected[0]) and np.array_equal(y, expected[1])
    assert all(
        np.array_equal(a, b)
        for p, q in zip(bank, pin, strict=True)
        for a, b in zip(p, q, strict=True)
    )
    assert not any(np.shares_memory(output, a) for output in (x, y) for p in bank for a in p)


@pytest.mark.parametrize("control", ["joint", "current_only"])
@pytest.mark.parametrize(
    "fault", ["nonmatch", "partial", "ambiguous", "conflict", "invalid_unused"]
)
def test_once_packet_refuses_unqualified_or_corrupted_original_information(control, fault):
    bank = inputs()
    current = tuple(a.copy() for a in bank[0])
    if fault == "nonmatch":
        current[0][0, 0] += 0.1
    if fault == "partial":
        current = tuple(a[:3] for a in current)
    if fault == "ambiguous":
        bank = (bank[0], bank[0])
    if fault == "conflict":
        bank[0][1][0, 0] = 5.0
    if fault == "invalid_unused":
        bank[1][1][0, 0] = 100.0
    with pytest.raises(ValueError):
        compose_shared_packet(current, bank, 6, control=control, current_once=True)


@pytest.mark.parametrize("flag", [None, 1, "once"])
def test_once_packet_flag_is_an_exact_bool(flag):
    bank = inputs()
    with pytest.raises(ValueError):
        compose_shared_packet(bank[0], bank, 6, control="joint", current_once=flag)


def fixture(control="joint"):
    stream, budget, memory, clock = source()
    state = stream.candidate_snapshot()
    certified = SourceTrainPackets(stream, state)
    aux = ToyBudgetSession(ONCE_LIMITS, budget.clock)
    aux.attach_memory(memory)
    row: dict = {}
    work = LinearProbeWork(
        row, lambda: budget.before_final(), certified, aux, control=control, return_once=True
    )
    child = LinearFitLearner(state.policy)
    child.restore_state(state)
    return stream, child, work, aux, row


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_real_tiny_once_callback_has_fifteen_then_twelve_rows_and_separate_paid_deltas(control):
    stream, child, work, aux, row = fixture(control)
    pin = stream.candidate_snapshot()
    for phase, current, rows, extra in (
        ("fresh", packet(3, "probe_train", "f"), 15, 12),
        ("return", work._source._bank[1], 12, 6),
    ):
        receipt = work.callback(child, current, (), phase=phase, check_boundary=lambda: None)
        assert receipt.applied and receipt.work["fit_rows"] == rows
        assert receipt.work["additional_rows_charged"] == extra
        assert receipt.work["retained_train_rows_used"] == (extra if control == "joint" else 0)
        assert receipt.work["native_sgd_updates"] == 0
    assert child.completed_updates == 14 and child.training_examples == 99
    assert aux.updates_completed == 2 and aux.replay_examples_completed == 18
    assert (
        row["replay_examples"] == 18
        and work.preserved["budget"]["limits"]["max_replay_examples"] == 288
    )
    assert stream.candidate_snapshot() == pin and stream._budget.updates_completed == 12


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_nonmatching_return_keeps_admission_and_six_rows_but_cannot_fit_or_continue(control):
    _, child, work, aux, _ = fixture(control)
    with pytest.raises(ProbeUpdateFailed) as error:
        work.callback(
            child, packet(6, "train", "foreign"), (), phase="return", check_boundary=lambda: None
        )
    receipt = error.value.receipt
    assert receipt.work["fit_attempts"] == 0 and not receipt.applied
    assert aux.updates_completed == 1 and aux.replay_examples_completed == 6
    with pytest.raises(ValueError, match="failed"):
        work.callback(child, work._source._bank[0], (), phase="return", check_boundary=lambda: None)
    assert work.preserved["additional_rows_charged"] == 6


def test_unreadable_applied_return_stays_unknown_with_permanent_six_row_charge(monkeypatch):
    stream, child, work, aux, _ = fixture()
    read = child.snapshot_state
    fitted = [False]

    def snapshot():
        if fitted[0]:
            raise RuntimeError("unreadable private post-state")
        return read()

    def fit(x, y):
        result = scripted(child, x, y)
        fitted[0] = True
        return result

    monkeypatch.setattr(child, "snapshot_state", snapshot)
    monkeypatch.setattr(child, "train_batch_minimum_change", fit)
    with pytest.raises(ProbeUpdateFailed) as error:
        work.callback(child, work._source._bank[0], (), phase="return", check_boundary=lambda: None)
    assert error.value.receipt.work["application_unknown"] == 1
    assert work.preserved["application_unknown_updates"] == 1
    assert aux.updates_completed == 1 and aux.replay_examples_completed == 6
    assert stream._budget.updates_completed == 12


@pytest.mark.parametrize("fault", ["recipe", "public_evidence", "public_counter", "bank"])
def test_once_callback_refuses_changed_prior_authority_before_admission(fault):
    _, child, work, aux, row = fixture()
    if fault == "recipe":
        work._return_once = False
    if fault == "public_evidence":
        row["linear_rehearsal_work"]["return_once"] = False
    if fault == "public_counter":
        row["replay_examples"] = 6
    if fault == "bank":
        work._source._bank[0].targets[0, 0] = 5.0
    with pytest.raises(ValueError):
        work.callback(
            child, packet(3, "probe_train", "f"), (), phase="fresh", check_boundary=lambda: None
        )
    assert aux.updates_completed == aux.replay_examples_completed == 0


def test_once_callback_cannot_reuse_the_old_384_row_authority():
    _, _, work, _, _ = fixture()
    source_budget = work._source._stream._budget
    wrong = ToyBudgetSession(LIMITS, source_budget.clock)
    with pytest.raises(ValueError, match="fresh separate"):
        LinearProbeWork({}, lambda: None, work._source, wrong, control="joint", return_once=True)


@pytest.mark.parametrize(
    "change",
    [
        dict(probe_rehearsal=None),
        dict(reserved_probe=False),
        dict(separate_context_heads=True),
        dict(replay="coverage"),
        dict(probe_update=None),
        dict(probe_return_once=1),
    ],
)
def test_once_api_refuses_unsupported_compositions_before_data(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data"))
    options = OPTIONS | dict(probe_rehearsal="joint", probe_return_once=True) | change
    with pytest.raises(ValueError):
        entry.run_example(**options)


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_once_cli_forwards_flag_and_never_overwrites_output(control, tmp_path, monkeypatch):
    calls = []

    def run(**kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--reserved-probe",
        "--probe-update",
        "minimum_change",
        "--train-update",
        "minimum_change",
        "--probe-rehearsal",
        control,
        "--probe-return-once",
        "--output",
        str(tmp_path / "out.json"),
    ]
    assert entry.main(flags) == 0
    assert calls == [OPTIONS | dict(probe_rehearsal=control, probe_return_once=True)]
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(calls) == 1


def test_once_cli_refuses_missing_shared_mode_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("runner"))
    path = tmp_path / "missing-mode.json"
    with pytest.raises(SystemExit):
        entry.main(["--probe-return-once", "--output", str(path)])
    assert not path.exists()


def install_scripted_runner(monkeypatch):
    monkeypatch.setattr(LinearFitLearner, "train_batch_minimum_change", scripted)
    monkeypatch.setattr(
        entry,
        "run_workflow_episode",
        lambda *a, **kw: dict(
            complete=True,
            success=False,
            scenario=kw.get("scenario", "ordinary"),
            training_updates=0,
            work=dict(prediction_calls_validated=0),
        ),
    )

    def reads(learner, phase, spec, require, row, **kwargs):
        require()
        row["episodes"][phase] = [
            dict(
                phase=phase,
                policy_version=v,
                concept=c,
                role="controller_train",
                complete=True,
                episode=dict(complete=True, success=False, scenario="ordinary", training_updates=0),
            )
            for v, c in probe.PAIRED_TASKS
        ]
        row["task_readout_work"][phase] = dict(decision_attempts=0, decision_returns=0)

    monkeypatch.setattr(probe, "_read_tasks", reads)


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_actual_scripted_ordinary_once_sequence_preserves_source_and_all_cost_scopes(
    control, monkeypatch
):
    install_scripted_runner(monkeypatch)
    result = entry.run_example(**OPTIONS, probe_rehearsal=control, probe_return_once=True)
    assert result["complete"], result.get("error")
    paid = result["probe_auxiliary_updates"]
    assert result["reserved_probe_return_once"] is True
    assert result["probe_updates"] == 32 and result["probe_replay_examples"] == 288
    assert paid["attempted_rows"] == 144 and paid["additional_rows_charged"] == 288
    assert paid["budget"]["limits"]["max_replay_examples"] == 288
    assert [e["work"]["fit_rows"] for e in paid["events"]] == [15] * 16 + [12] * 16
    assert [e["work"]["additional_rows_charged"] for e in paid["events"]] == [12] * 16 + [6] * 16
    assert result["budget"]["updates_completed"] == 12 and len(result["applied_updates"]) == 12
    summary = result["reserved_probe_summary"]["reported_costs"]["linear_rehearsal"]
    assert summary["applied_updates"] == 32 and summary["additional_rows_charged"] == 288
    assert all(
        result[k] is True
        for k in (
            "source_unchanged",
            "actor_unchanged",
            "numpy_rng_unchanged",
            "python_rng_unchanged",
        )
    )


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_original_once_intent_corruption_refuses_before_private_fitting(control, monkeypatch):
    install_scripted_runner(monkeypatch)
    before = SCRIPTED.copy()
    original = probe.run_owned_probe

    def corrupt(*args, **kwargs):
        args[2]["reserved_probe_return_once"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(probe, "run_owned_probe", corrupt)
    result = entry.run_example(**OPTIONS, probe_rehearsal=control, probe_return_once=True)
    assert not result["complete"] and "once-return" in result["error"]
    assert result["budget"]["updates_completed"] == 12
    # The original intent guard refuses before ANY private owner/fit is created.
    assert result["probe_updates"] == 0 and result.get("probe_auxiliary_updates") is None
    assert SCRIPTED["calls"] - before["calls"] == 12
    assert SCRIPTED["rows"] - before["rows"] == 72
