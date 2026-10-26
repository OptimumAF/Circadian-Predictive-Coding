"""One actual successive-replay CLI graph; native work ends at original setup."""

from hashlib import sha256
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_numpy_managed_capture as cli
import src.adapters.numpy_composite_capture as adapter
import src.adapters.numpy_restored_capture_example as example
from src.adapters.numpy_learners import CircadianLearner
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from test_numpy_retained_capture import assert_owned_graph, holder, values


def forbidden(*args, **kwargs):
    pytest.fail("native work after the sole original successive-demo setup")


def identity(graph):
    old, new = graph["old"], graph["new"]
    return tuple(
        id(value)
        for value in (
            graph["owner"],
            graph["life"],
            old,
            new,
            graph["ledger"],
            graph["controller"],
            graph["manager"],
            graph["manager"]._copies,
            graph["token"],
            graph["pending"],
            graph["view"],
            new.actor,
            old._candidate,
            new._candidate,
            old._inbox,
            new._inbox,
            new._budget,
            new._inbox._clock,
            graph["life"]._copy_budget,
            graph["shared"],
            graph["gate"],
            graph["wall_clock"],
            graph["builder"],
        )
    )


def test_should_run_successive_cli_with_complete_data_and_original_spent_authority(
    monkeypatch, capsys, record_testsuite_property
):
    graphs: list[dict[str, Any]] = []
    captures: list[Any] = []
    attempts: list[dict[str, Any]] = []
    passes: list[dict[int, Any]] = []
    initial: dict[str, Any] = {}
    original_setup = example._setup_successive_case
    original_capture = example.capture_numpy_managed_composite
    original_copy = adapter.copy_numpy_composite

    def setup():
        graph = original_setup()
        graphs.append(graph)
        old, new = graph["old"], graph["new"]
        models = dict(
            actor=new.actor._learner._model,
            retired=old._candidate._model,
            current=new._candidate._model,
        )
        publication = graph["manager"]._publications[id(models["retired"])]
        metadata = json.dumps(
            publication.metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8")
        assert publication.charged_bytes == len(metadata) + 1024
        assert graph["manager"]._retain_capture_history is True
        assert graph["owner"]._shared._runtime is new and old._retired and not new._retired
        assert graph["runtime"] is old and not graph["controller"]._pending
        assert graph["controller"]._models[0] is new._candidate
        initial.update(
            identities=identity(graph),
            models=models,
            native={name: pickle.dumps(vars(model)) for name, model in models.items()},
            view=pickle.dumps(graph["view"]),
            charge=graph["life"].payload_byte_snapshot(),
            accounting=graph["ledger"].accounting(),
            update_binding=graph["manager"]._update_binding,
            metadata_bytes=len(metadata),
            metadata_charge=publication.charged_bytes,
        )
        assert sum(map(len, initial["native"].values())) + len(initial["view"]) < 65536
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
        attempts.append(kwargs)
        result = original_capture(*args, **kwargs)
        captures.append(result)
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

    monkeypatch.setattr(example, "_setup_successive_case", setup)
    monkeypatch.setattr(example, "capture_numpy_managed_composite", capture)
    monkeypatch.setattr(adapter, "copy_numpy_composite", copying)
    monkeypatch.setattr(sys, "argv", ["run_numpy_managed_capture", "--run", "--successive-replay"])
    assert cli.main() == 0
    output = capsys.readouterr()
    assert not output.err and len(output.out.encode("utf8")) <= 131072
    report = json.loads(output.out)
    assert report["successive_replay_capture_verified"] is True
    assert report["full_managed_process_recovery_verified"] is False
    case = report["case"]
    assert report["protocol"]["protocol_id"] == "numpy_successive_replay_capture_engineering_v1"
    assert case["native_before"] == case["native_after"] == case["native_captured"]
    assert case["inbox_before"] == case["inbox_after"] == case["inbox_captured"]
    assert set(case["native_captured"]) == {"actor", "retired", "current"}
    assert set(case["inbox_captured"]) == {"retired", "current"}
    assert case["identity"]["original"] == case["identity"]["current"]
    assert len(case["identity"]["original"]) == 23
    assert case["publication_metadata_charge"] == initial["metadata_charge"]
    assert case["publication_metadata_bytes"] == initial["metadata_bytes"]
    assert case["retired_historical_updates"] == 1 and case["current_updates"] == 3
    continuation = case["continuation"]
    assert all(
        continuation[name] is True
        for name in (
            "observed_paused_poll",
            "initial_row_expired",
            "second_row_expired",
            "previous_success_expired",
        )
    )
    assert (
        continuation["zero_work_poll_metadata_charge"]
        == continuation["zero_work_poll_metadata_bytes"] + 1024
    )
    assert case["current_update_binding_preserved"] is True
    assert case["preparation_attempts"] == 1 and case["pending_checkpoints"] == 0
    assert case["original_consumed_view_unchanged"] is case["actual_cp_current_alias"] is True
    for digest, raw in case["component_payloads"].items():
        assert sha256(raw.encode("utf8")).hexdigest() == digest
        assert type(json.loads(raw)) is dict
    assert set((*case["native_captured"].values(), *case["inbox_captured"].values())) == set(
        case["component_payloads"]
    )
    assert len(graphs) == len(captures) == 1 and len(attempts) == 2
    graph, captured = graphs[0], captures[0]
    assert attempts[0]["replay_origins"] is attempts[1]["replay_origins"] is graph["ledger"]
    assert attempts[0].get("replay_checkpoints") is None
    assert attempts[1]["replay_checkpoints"] is graph["manager"]
    assert len(passes) == 2 and passes[0] is passes[1]
    old, new = graph["old"], graph["new"]
    detached_cp = holder(captured, dict(life=graph["life"]), graph["controller"])
    detached_new = holder(captured, dict(life=graph["life"]), new)
    assert detached_cp["_models"][0] is detached_new["_candidate"]
    assert not detached_cp["_pending"]
    old_row = old._candidate._model._replay_memory[0]
    new_row = new._candidate._model._replay_memory[0]
    assert old_row is not new_row
    for name in ("input_batch", "target_batch"):
        assert not np.shares_memory(getattr(old_row, name), getattr(new_row, name))
    for runtime, label in ((old, "retired"), (new, "current")):
        detached = holder(captured, dict(life=graph["life"]), runtime)
        assert detached["_retired"] is (runtime is old)
        native = values(detached["_candidate"])["_model"]
        assert_owned_graph(native.state, vars(initial["models"][label]))
        assert native.config is native.state["config"]
        inbox = values(detached["_inbox"])
        for name in ("_experiences", "_labels", "_applied", "_erased"):
            assert_owned_graph(inbox[name], getattr(runtime._inbox, name))
        assert inbox["_historical_completed_updates"] == (1 if runtime is old else None)
        for row, copied in zip(
            runtime._candidate._model._replay_memory, native.state["_replay_memory"]
        ):
            assert passes[0][id(row)] is copied
            for name in ("input_batch", "target_batch"):
                assert passes[0][id(getattr(row, name))] is getattr(copied, name)
    actor = values(holder(captured, dict(life=graph["life"]), new.actor)["_learner"])["_model"]
    assert_owned_graph(actor.state, vars(initial["models"]["actor"]))
    assert identity(graph) == initial["identities"]
    assert initial["native"] == {
        name: pickle.dumps(vars(model)) for name, model in initial["models"].items()
    }
    assert initial["view"] == pickle.dumps(graph["view"])
    before, after = initial["charge"], graph["life"].payload_byte_snapshot()
    assert before.observed_retained_bytes == after.observed_retained_bytes == 256
    assert before.charged_bytes == 800 and after.charged_bytes == 1056
    assert after.charged_bytes == before.charged_bytes + captured.state.retained_payload_bytes
    paid = graph["ledger"].accounting()
    assert paid.invocations_started == initial["accounting"].invocations_started + 2
    assert paid.records_created == initial["accounting"].records_created
    assert paid.metadata_bytes_charged == initial["accounting"].metadata_bytes_charged + 2048
    assert graph["controller"]._attempts == 1 and len(graph["controller"]._models) == 1
    assert old._retired and not new._retired and graph["owner"]._shared._runtime is new
    assert old._inbox._historical_completed_updates == 1
    assert new._budget.updates_completed == 3
    assert len(new._candidate._model._replay_memory) == 2
    success = graph["manager"]._current_success
    metadata = json.dumps(
        success.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf8")
    assert success.completed_updates == 3
    assert case["current_success_metadata_bytes"] == len(metadata)
    assert case["current_success_metadata_charge"] == success.charged_bytes == len(metadata) + 1024
    assert graph["manager"]._update_binding is initial["update_binding"]
    assert graph["manager"]._zero_work_poll is None
    assert graph["gate"]._deferrals == {"paused": 1}
    assert graph["ledger"]._admission.limits.max_metadata_bytes == 262144
    record_testsuite_property("successive_demo_stdout", output.out)
    record_testsuite_property(
        "successive_demo_observations",
        json.dumps(
            dict(
                stdout_bytes=len(output.out.encode("utf8")),
                original_identities_preserved=23,
                native_unchanged=True,
                complete_native_fields={
                    name: sorted(vars(model)) for name, model in initial["models"].items()
                },
                consumed_view_unchanged=True,
                original_token_consumed=True,
                native_updates=3,
                copy_before=before.charged_bytes,
                copy_after=after.charged_bytes,
                retained_bytes=after.observed_retained_bytes,
                publication_metadata_charge=initial["metadata_charge"],
                capture_attempts=2,
                copy_passes=2,
                report=report,
            ),
            sort_keys=True,
        ),
    )
