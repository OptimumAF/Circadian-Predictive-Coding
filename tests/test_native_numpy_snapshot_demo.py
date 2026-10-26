"""One actual CLI graph, with native work restricted to its original setup."""

import json
from hashlib import sha256
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_numpy_managed_capture as cli
import src.adapters.numpy_composite_capture as adapter
import src.adapters.numpy_snapshot_capture_example as example
from src.adapters.numpy_learners import CircadianLearner
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from test_numpy_retained_capture import assert_owned_graph, holder, values


def forbidden(*args, **kwargs):
    pytest.fail("native work after the sole original setup")


def identity(graph):
    runtime = graph["runtime"]
    return tuple(
        id(item)
        for item in (
            graph["owner"],
            graph["life"],
            runtime,
            graph["ledger"],
            graph["controller"],
            graph["manager"],
            graph["manager"]._copies,
            graph["token"],
            graph["pending"],
            graph["view"],
            graph["snapshot"],
            graph["view"].inbox,
            runtime._candidate,
            runtime.actor,
            runtime._budget,
            runtime._inbox._clock,
            graph["life"]._copy_budget,
        )
    )


def test_should_run_issued_snapshot_cli_with_full_data_and_original_paid_authority(
    monkeypatch, capsys, record_testsuite_property
):
    graphs: list[dict[str, Any]] = []
    captures: list[Any] = []
    passes: list[dict[int, Any]] = []
    initial: dict[str, Any] = {}
    original_setup = example._setup_snapshot_case
    original_capture = example.capture_numpy_managed_composite
    original_copy = adapter.copy_numpy_composite

    def setup():
        graph = original_setup()
        graphs.append(graph)
        runtime = graph["runtime"]
        initial.update(
            identities=identity(graph),
            native=tuple(
                pickle.dumps(vars(item._model))
                for item in (
                    runtime.actor._learner,
                    runtime._candidate,
                )
            ),
            view=pickle.dumps(graph["view"]),
            charge=graph["life"].payload_byte_snapshot(),
            accounting=graph["ledger"].accounting(),
        )
        assert sum(len(raw) for raw in initial["native"]) + len(initial["view"]) < 65536
        for kind in (CircadianLearner, CircadianPredictiveCodingNetwork, PredictiveCodingNetwork):
            for name in (
                "__init__",
                "fork",
                "train_batch",
                "train_epoch",
                "predict",
                "predict_proba",
                "snapshot_state",
                "restore_state",
                "infer_hidden_states",
                "_infer_hidden_states",
                "sleep_event",
                "_execute_sleep_event",
            ):
                if hasattr(kind, name):
                    monkeypatch.setattr(kind, name, forbidden)
        return graph

    def capture(*args, **kwargs):
        result = original_capture(*args, **kwargs)
        captures.append(result)
        assert kwargs["replay_origins"] is graphs[0]["ledger"]
        assert kwargs["replay_checkpoints"] is graphs[0]["manager"]
        return result

    def copying(state, limits, memo):
        graph = graphs[0]
        before = initial["charge"]
        assert (
            graph["life"]._copy_budget._charged
            == before.charged_bytes + before.observed_retained_bytes
        )
        passes.append(memo)
        return original_copy(state, limits, memo)

    monkeypatch.setattr(example, "_setup_snapshot_case", setup)
    monkeypatch.setattr(example, "capture_numpy_managed_composite", capture)
    monkeypatch.setattr(adapter, "copy_numpy_composite", copying)
    monkeypatch.setattr(sys, "argv", ["run_numpy_managed_capture", "--run", "--issued-snapshot"])
    assert cli.main() == 0
    output = capsys.readouterr()
    assert not output.err and len(output.out.encode("utf-8")) <= 131072
    report = json.loads(output.out)
    assert report["issued_snapshot_capture_verified"] is True
    assert report["full_managed_process_recovery_verified"] is False
    case = report["case"]
    assert case["native_before"] == case["native_after"] == case["native_captured"]
    assert case["inbox_before"] == case["inbox_after"] == case["inbox_captured"]
    assert case["issued_view_before"] == case["issued_view_after"] == case["issued_view_captured"]
    assert set(case["native_captured"]) == {"actor", "candidate", "issued"}
    assert set(case["inbox_captured"]) == {"current", "issued"}
    assert len(case["issued_view_fields"]) == 12 and len(case["issued_view_captured"]) == 10
    assert case["identity"]["original"] == case["identity"]["current"]
    for digest, raw in case["component_payloads"].items():
        assert sha256(raw.encode("utf-8")).hexdigest() == digest
        assert type(json.loads(raw)) is dict
    assert set((*case["native_captured"].values(), *case["inbox_captured"].values())) == set(
        case["component_payloads"]
    )
    assert len(graphs) == len(captures) == 1
    assert len(passes) == 2 and passes[0] is passes[1]
    graph, captured = graphs[0], captures[0]
    runtime, view = graph["runtime"], graph["view"]
    detached_controller = holder(captured, dict(life=graph["life"]), graph["controller"])
    detached_view = values(detached_controller["_pending"][0][1])["view"]
    assert_owned_graph(detached_view, view)
    original_row = view.state.state["_replay_memory"][0]
    detached_row = detached_view.state.state["_replay_memory"][0]
    source_row = runtime._candidate._model._replay_memory[0]
    assert original_row is not source_row and detached_row is not original_row
    for name in ("input_batch", "target_batch"):
        before, after = getattr(original_row, name), getattr(detached_row, name)
        assert not np.shares_memory(before, after)
        assert passes[0][id(before)] is after
    assert passes[0][id(original_row)] is detached_row
    for original_holder, field, learner in (
        (runtime, "_candidate", runtime._candidate),
        (runtime.actor, "_learner", runtime.actor._learner),
    ):
        native = values(holder(captured, dict(life=graph["life"]), original_holder)[field])[
            "_model"
        ]
        assert_owned_graph(native.state, vars(learner._model))
        assert native.config is native.state["config"]
    assert identity(graph) == initial["identities"]
    assert initial["native"] == tuple(
        pickle.dumps(vars(item._model))
        for item in (
            runtime.actor._learner,
            runtime._candidate,
        )
    )
    assert initial["view"] == pickle.dumps(view)
    assert graph["controller"]._pending[graph["token"]] is graph["pending"]
    assert graph["pending"].view is view and not graph["controller"]._models
    assert graph["controller"]._attempts == 0 and not runtime._retired
    assert runtime._budget.updates_completed == 1
    before, after = initial["charge"], graph["life"].payload_byte_snapshot()
    assert before.observed_retained_bytes == after.observed_retained_bytes == 128
    assert before.charged_bytes == 224 and after.charged_bytes == 352
    assert after.charged_bytes == before.charged_bytes + captured.state.retained_payload_bytes
    paid = graph["ledger"].accounting()
    assert paid.invocations_started == initial["accounting"].invocations_started + 2
    assert paid.records_created == initial["accounting"].records_created
    assert paid.metadata_bytes_charged == initial["accounting"].metadata_bytes_charged + 2048
    record_testsuite_property("snapshot_demo_stdout", output.out)
    record_testsuite_property(
        "snapshot_demo_observations",
        json.dumps(
            dict(
                stdout_bytes=len(output.out.encode("utf-8")),
                original_identities_preserved=17,
                native_unchanged=True,
                complete_native_fields=sorted(vars(runtime._candidate._model)),
                complete_view_unchanged=True,
                original_token_still_pending=True,
                native_updates=1,
                copy_before=before.charged_bytes,
                copy_after=after.charged_bytes,
                retained_bytes=after.observed_retained_bytes,
                capture_attempts=2,
                copy_passes=2,
                report=report,
            ),
            sort_keys=True,
        ),
    )
