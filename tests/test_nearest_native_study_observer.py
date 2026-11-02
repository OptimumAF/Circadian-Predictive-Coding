"""Extract only the actual study's metadata procedure; no project/model imports."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def procedure(tmp_path):
    path = Path("artifacts/runs/nearest-reserved-native-study-20261101/study.py")
    tree = ast.parse(path.read_text())
    nodes: list[ast.stmt] = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("encoded", "wrap", "check_native_prefix", "observe_native")
    ]
    ns = dict(json=json, STAGE=tmp_path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    calls = []
    meter = SimpleNamespace(check=lambda: None, in_native=False)
    report: dict[str, Any] = dict(native_cases=[], native_seals=[])
    owner = object()

    def rollout(output, **options):
        calls.append(options)
        seed = options.get("probe_test_seed", options.get("train_seed"))
        output.write_text("{}")
        return SimpleNamespace(
            report=dict(
                complete=True,
                seed=seed,
                role="probe_test" if seed == 24 else "train",
                optimizer_updates=0,
                frozen_state_exact=True,
                original_owner_exact=True,
                action_history_exact=True,
                nearest_work_scope="validated_native_work",
                sparse_return=0.0,
            )
        )

    def invoke(phase, delegate=rollout):
        def check(read):
            read()

        ns["observe_native"](
            owner,
            phase,
            check,
            report=report,
            meter=meter,
            rollout=delegate,
            recipe="executed12",
            bind=lambda x: (x,),
            require=lambda x: None,
        )

    return ns, report, calls, meter, invoke, rollout


def test_should_preserve_fixed_five_episode_order_and_original_role_parameters(procedure):
    _, report, calls, meter, invoke, _ = procedure
    invoke("before_probe")
    invoke("after_probe")
    assert [c.get("probe_test_seed", c.get("train_seed")) for c in calls] == [24, 24, 17, 19, 20]
    assert [r["result"]["role"] for r in report["native_cases"]] == ["probe_test"] * 2 + [
        "train"
    ] * 3
    assert all(c["action_history_recipe"] == "executed12" for c in calls)
    assert not meter.in_native


@pytest.mark.parametrize("phase", ["unrecognized", "after_probe"])
def test_should_refuse_unexpected_phase_without_external_work(procedure, phase):
    _, _, calls, _, invoke, _ = procedure
    with pytest.raises(RuntimeError, match="phase/order"):
        invoke(phase)
    assert not calls


def test_should_capture_incomplete_return_and_stop_before_more_episodes(procedure):
    _, report, calls, _, invoke, rollout = procedure

    def incomplete(*args, **kwargs):
        result = rollout(*args, **kwargs)
        result.report["complete"] = False
        return result

    with pytest.raises(RuntimeError, match="incomplete"):
        invoke("before_probe", incomplete)
    assert len(calls) == len(report["native_cases"]) == 1
    assert report["native_cases"][0]["result"]["complete"] is False


def test_should_refuse_corrupted_preceding_success_before_more_native_work(procedure):
    _, report, calls, _, invoke, _ = procedure
    invoke("before_probe")
    report["native_cases"][0]["result"]["sparse_return"] = 1.0
    with pytest.raises(RuntimeError, match="paid evidence"):
        invoke("after_probe")
    assert len(calls) == 1


def test_should_refuse_equal_replaced_preceding_holder(procedure):
    _, report, calls, _, invoke, _ = procedure
    invoke("before_probe")
    report["native_cases"][0] = dict(report["native_cases"][0])
    with pytest.raises(RuntimeError, match="paid evidence"):
        invoke("after_probe")
    assert len(calls) == 1


def test_should_retain_returned_capture_if_work_deadline_is_crossed(procedure):
    _, report, calls, meter, invoke, rollout = procedure

    def late(*args, **kwargs):
        result = rollout(*args, **kwargs)

        def stopped():
            raise RuntimeError("permanent deadline")

        meter.check = stopped
        return result

    with pytest.raises(RuntimeError, match="deadline"):
        invoke("before_probe", late)
    assert len(calls) == len(report["native_cases"]) == 1 and not meter.in_native


def test_should_keep_cleanup_available_and_paid_after_permanent_workstop(procedure):
    ns, _, _, _, _, _ = procedure

    def stopped(*args):
        raise RuntimeError("permanent deadline")

    meter = SimpleNamespace(charge=stopped, counts=dict(closes=0, close_returns=0))
    port = SimpleNamespace(close=lambda: None)
    ns["wrap"](meter, port, "close", "closes", "close_returns", cleanup=True)
    port.close()
    assert meter.counts == dict(closes=1, close_returns=1)
