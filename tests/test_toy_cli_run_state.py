"""Budgeted toy CLI runs publish truthful states and resume checked work."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.adapters import cli, toy_budget_cli
from src.app import experiment_runner
from src.core.backprop_mlp import BackpropMLP
from src.infra.circadian_checkpoint_files import TrustedLocalToyCheckpointStore
from src.infra.toy_run_state_files import ToyRunStateFile


def _arguments(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "toy",
        "--samples",
        "80",
        "--epochs",
        "1",
        "--sleep-interval",
        "0",
        "--run-state",
        str(tmp_path / "run-state.json"),
        "--checkpoint",
        str(tmp_path / "toy.checkpoint"),
        "--json-result",
        str(tmp_path / "result.json"),
        "--resolved-config",
        str(tmp_path / "config.json"),
        *extra,
    ]


def _state(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "run-state.json").read_text(encoding="utf-8"))


def test_should_stop_real_cli_without_final_access_then_resume_completed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_split = experiment_runner.split_training_validation
    final_reads = 0

    class SealedFinal:
        @property
        def input(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("incomplete run opened final input")

        @property
        def target(self) -> Any:
            nonlocal final_reads
            final_reads += 1
            raise AssertionError("incomplete run opened final labels")

    def sealed_split(*args: Any, **kwargs: Any) -> Any:
        roles = original_split(*args, **kwargs)
        return SimpleNamespace(
            train=roles.train,
            validation=roles.validation,
            test=SealedFinal(),
            split_hashes=roles.split_hashes,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(experiment_runner, "split_training_validation", sealed_split)
        scoped.setattr(sys, "argv", _arguments(tmp_path, "--max-training-updates", "2"))
        with pytest.raises(SystemExit) as stopped:
            cli.main()

    assert stopped.value.code == 3
    assert final_reads == 0
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "config.json").exists()
    checkpoint = tmp_path / "toy.checkpoint"
    state = _state(tmp_path)
    assert state["schema_id"] == "toy_cli_run_state_v1"
    assert (state["status"], state["reason"]) == ("incomplete", "max_training_updates")
    assert state["budget"] == {
        "max_training_updates": 2,
        "max_wall_seconds": None,
        "max_replay_examples": None,
        "max_hidden_width": None,
        "max_process_rss_bytes": None,
    }
    assert state["work"]["training_updates_observed"] == 2
    assert state["work"]["checkpointed_training_updates"] == 2
    assert state["checkpoint"]["sha256"] == sha256(checkpoint.read_bytes()).hexdigest()
    assert state["checkpoint"]["position"]["stage"] == "wake"
    assert state["resolved_config"]["config"]["epoch_count"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(tmp_path, "--resume", "--max-training-updates", "3"),
    )
    cli.main()

    completed = _state(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    saved_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert (completed["status"], completed["reason"]) == ("completed", "completed")
    assert completed["work"]["training_updates_observed"] == 3
    assert completed["work"]["checkpointed_training_updates"] == 3
    assert [attempt["status"] for attempt in completed["attempts"]] == [
        "incomplete",
        "completed",
    ]
    assert [attempt["budget"]["max_training_updates"] for attempt in completed["attempts"]] == [
        2,
        3,
    ]
    assert saved_config == result["resolved_config"] == completed["resolved_config"]
    assert len(result["backprop"]["loss_history"]) == 1
    assert len(result["predictive_coding"]["loss_history"]) == 1
    assert len(result["circadian_predictive_coding"]["loss_history"]) == 1
    assert isinstance(result["backprop"]["test_accuracy"], float)


def test_should_record_error_work_and_resume_from_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_train = BackpropMLP.train_epoch
    calls = 0

    def fail_second_epoch(self: BackpropMLP, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected wake failure")
        return original_train(self, *args, **kwargs)

    arguments = _arguments(tmp_path, "--epochs", "2", "--max-training-updates", "6")
    with monkeypatch.context() as scoped:
        scoped.setattr(BackpropMLP, "train_epoch", fail_second_epoch)
        scoped.setattr(sys, "argv", arguments)
        with pytest.raises(RuntimeError, match="injected wake failure"):
            cli.main()

    state = _state(tmp_path)
    assert (state["status"], state["reason"], state["error_type"]) == (
        "error",
        "exception",
        "RuntimeError",
    )
    assert state["work"]["training_updates_observed"] == 3
    assert state["work"]["checkpointed_training_updates"] == 3
    assert state["checkpoint"]["position"]["stage"] == "after_sleep"
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "config.json").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(tmp_path, "--epochs", "2", "--resume", "--max-training-updates", "6"),
    )
    cli.main()
    assert _state(tmp_path)["status"] == "completed"
    assert len(json.loads((tmp_path / "result.json").read_text())["backprop"]["loss_history"]) == 2


def test_should_distinguish_observed_from_durable_work_after_checkpoint_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_save = TrustedLocalToyCheckpointStore.save
    saves = 0

    def fail_second_save(self: TrustedLocalToyCheckpointStore, checkpoint: Any) -> None:
        nonlocal saves
        saves += 1
        if saves == 2:
            raise OSError("injected checkpoint write failure")
        original_save(self, checkpoint)

    with monkeypatch.context() as scoped:
        scoped.setattr(TrustedLocalToyCheckpointStore, "save", fail_second_save)
        scoped.setattr(sys, "argv", _arguments(tmp_path, "--max-training-updates", "3"))
        with pytest.raises(OSError, match="injected checkpoint write failure"):
            cli.main()

    state = _state(tmp_path)
    assert state["status"] == "error"
    assert state["work"]["training_updates_observed"] == 2
    assert state["work"]["checkpointed_training_updates"] == 1
    assert state["checkpoint"]["position"]["next_batch_index"] == 1
    assert not (tmp_path / "result.json").exists()

    monkeypatch.setattr(
        sys, "argv", _arguments(tmp_path, "--resume", "--max-training-updates", "3")
    )
    cli.main()
    assert _state(tmp_path)["status"] == "completed"


def test_should_record_real_zero_wall_deadline_without_completed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-wall-seconds", "0"))

    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 3
    state = _state(tmp_path)
    assert (state["status"], state["reason"]) == ("incomplete", "max_wall_seconds")
    assert state["work"]["training_updates_observed"] == 0
    assert state["checkpoint"] is None
    assert not (tmp_path / "result.json").exists()


def test_should_keep_error_state_when_checkpoint_itself_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def corrupt_save(self: TrustedLocalToyCheckpointStore, _checkpoint: Any) -> None:
        self.path.write_bytes(b"corrupt checkpoint")
        raise OSError("injected corrupt checkpoint failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(TrustedLocalToyCheckpointStore, "save", corrupt_save)
        scoped.setattr(sys, "argv", _arguments(tmp_path, "--max-training-updates", "3"))
        with pytest.raises(OSError, match="injected corrupt checkpoint failure"):
            cli.main()

    state = _state(tmp_path)
    assert (state["status"], state["reason"]) == ("error", "exception")
    assert state["work"]["training_updates_observed"] == 1
    assert state["work"]["checkpointed_training_updates"] is None
    assert state["checkpoint_error_type"] == "ValueError"
    assert state["checkpoint"] is None
    assert not (tmp_path / "result.json").exists()


def test_should_record_publication_error_without_a_completed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_result_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected result publication failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(toy_budget_cli, "write_toy_result_json", fail_result_write)
        scoped.setattr(sys, "argv", _arguments(tmp_path, "--max-training-updates", "3"))
        with pytest.raises(OSError, match="injected result publication failure"):
            cli.main()

    state = _state(tmp_path)
    assert (state["status"], state["error_type"]) == ("error", "OSError")
    assert state["work"]["training_updates_observed"] == 3
    assert not (tmp_path / "result.json").exists()
    config_path = tmp_path / "config.json"
    original_config = config_path.read_bytes()

    monkeypatch.setattr(
        sys, "argv", _arguments(tmp_path, "--resume", "--max-training-updates", "3")
    )
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="config artifact differs"):
        cli.main()
    assert _state(tmp_path) == state
    config_path.write_bytes(original_config)
    cli.main()
    assert _state(tmp_path)["status"] == "completed"
    assert (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--max-training-updates", "-1"], "max_training_updates"),
        (["--max-wall-seconds", "nan"], "max_wall_seconds"),
        (["--max-wall-seconds", "inf"], "max_wall_seconds"),
        (["--max-wall-seconds", "-0.1"], "max_wall_seconds"),
        ([], "budget"),
        (["--mode", "indepth", "--max-training-updates", "1"], "baseline"),
    ],
)
def test_should_reject_bad_budget_request_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    match: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, *extra))
    monkeypatch.setattr(cli, "run_experiment", lambda **_kwargs: pytest.fail("reached training"))
    with pytest.raises(SystemExit) as rejected:
        cli.main()
    assert rejected.value.code == 2
    assert match in capsys.readouterr().err
    assert not (tmp_path / "run-state.json").exists()
    assert not (tmp_path / "toy.checkpoint").exists()


def test_should_reject_occupied_or_colliding_artifacts_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "run_experiment", lambda **_kwargs: pytest.fail("reached training"))
    checkpoint = tmp_path / "toy.checkpoint"
    checkpoint.write_bytes(b"existing")
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-training-updates", "1"))
    with pytest.raises(FileExistsError):
        cli.main()
    assert checkpoint.read_bytes() == b"existing"
    checkpoint.unlink()

    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(
            tmp_path,
            "--max-training-updates",
            "1",
            "--checkpoint",
            str(tmp_path / "result.json"),
        ),
    )
    with pytest.raises(SystemExit) as rejected:
        cli.main()
    assert rejected.value.code == 2
    assert not (tmp_path / "run-state.json").exists()


def test_should_refuse_existing_state_and_live_lock_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "run_experiment", lambda **_kwargs: pytest.fail("reached training"))
    arguments = _arguments(tmp_path, "--max-training-updates", "1")
    state_path = tmp_path / "run-state.json"
    state_path.write_text("prior state", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(FileExistsError, match="run state"):
        cli.main()
    assert state_path.read_text(encoding="utf-8") == "prior state"

    state_path.unlink()
    lock_path = tmp_path / "run-state.json.lock"
    lock_path.write_text("other writer", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cli.main()
    assert lock_path.read_text(encoding="utf-8") == "other writer"
    assert not state_path.exists()


def test_should_prepare_new_checkpoint_parent_before_claiming_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "new-checkpoint-parent" / "nested" / "toy.bin"
    arguments = _arguments(tmp_path, "--checkpoint", str(checkpoint), "--max-training-updates", "1")
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 3
    assert _state(tmp_path)["checkpoint"]["path"] == str(checkpoint.resolve())
    assert checkpoint.is_file()


def test_should_remove_only_own_checkpoint_reservation_if_state_claim_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_claim(_self: ToyRunStateFile, _record: Any) -> None:
        raise FileExistsError("injected state claim race")

    checkpoint = tmp_path / "new-checkpoint-parent" / "toy.bin"
    monkeypatch.setattr(ToyRunStateFile, "create", fail_claim)
    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(tmp_path, "--checkpoint", str(checkpoint), "--max-training-updates", "1"),
    )

    with pytest.raises(FileExistsError, match="injected state claim race"):
        cli.main()

    assert not checkpoint.exists()
    assert not (tmp_path / "run-state.json").exists()
    assert not (tmp_path / "run-state.json.lock").exists()


def test_should_reject_tampered_checkpoint_and_changed_config_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-training-updates", "1"))
    with pytest.raises(SystemExit):
        cli.main()
    original_state = (tmp_path / "run-state.json").read_bytes()
    checkpoint = tmp_path / "toy.checkpoint"
    original_checkpoint = checkpoint.read_bytes()

    monkeypatch.setattr(cli, "run_experiment", lambda **_kwargs: pytest.fail("unsafe resume"))
    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(tmp_path, "--resume", "--epochs", "2", "--max-training-updates", "6"),
    )
    with pytest.raises(ValueError, match="config"):
        cli.main()
    assert (tmp_path / "run-state.json").read_bytes() == original_state

    checkpoint.write_bytes(original_checkpoint + b"tampered")
    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(tmp_path, "--resume", "--max-training-updates", "3"),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        cli.main()
    assert (tmp_path / "run-state.json").read_bytes() == original_state


def test_should_record_precise_nonresumability_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path, "--max-training-updates", "0")
    index = args.index("--checkpoint")
    del args[index : index + 2]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 3
    state = _state(tmp_path)
    assert state["status"] == "incomplete"
    assert state["checkpoint"] is None
    assert state["work"]["training_updates_observed"] == 0

    monkeypatch.setattr(sys, "argv", [*args, "--resume"])
    with pytest.raises(ValueError, match="non-resumable"):
        cli.main()


def test_should_record_replay_limit_and_resume_same_cli_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _arguments(
        tmp_path,
        "--epochs",
        "2",
        "--sleep-interval",
        "1",
        "--replay-steps",
        "1",
        "--replay-memory-size",
        "2",
        "--max-replay-examples",
        "0",
    )
    monkeypatch.setattr(sys, "argv", first)

    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 3
    state = _state(tmp_path)
    assert (state["status"], state["reason"]) == ("incomplete", "max_replay_examples")
    assert state["budget"]["max_replay_examples"] == 0
    assert state["work"]["replay_examples_observed"] == 0
    assert state["work"]["checkpointed_replay_examples"] == 0
    assert state["checkpoint"]["position"]["stage"] == "before_sleep"
    assert not (tmp_path / "result.json").exists()

    resumed = [*first, "--resume", "--max-replay-examples", "104"]
    monkeypatch.setattr(sys, "argv", resumed)
    cli.main()

    completed = _state(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["work"]["replay_examples_observed"] == 104
    assert completed["work"]["checkpointed_replay_examples"] == 104
    assert (
        sum(event["replay"]["applied_examples"] for event in result["circadian_sleep"]["events"])
        == 104
    )


def test_should_reject_invalid_replay_limit_before_creating_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-replay-examples", "-1"))

    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_should_resume_previous_v1_state_without_replay_work_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _arguments(tmp_path, "--max-training-updates", "2")
    monkeypatch.setattr(sys, "argv", first)
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 3
    state = _state(tmp_path)
    state["checkpoint"].pop("replay_examples")
    state["checkpoint"].pop("hidden_width")
    state["checkpoint"].pop("peak_hidden_width")
    state["work"].pop("replay_examples_observed")
    state["work"].pop("checkpointed_replay_examples")
    for key in (
        "hidden_width_observed",
        "peak_hidden_width_observed",
        "checkpointed_hidden_width",
        "checkpointed_peak_hidden_width",
        "rejected_proposed_hidden_width",
    ):
        state["work"].pop(key)
    state["budget"].pop("max_replay_examples")
    state["budget"].pop("max_hidden_width")
    for attempt in state["attempts"]:
        attempt["checkpoint"].pop("replay_examples")
        attempt["checkpoint"].pop("hidden_width")
        attempt["checkpoint"].pop("peak_hidden_width")
        attempt["work"].pop("replay_examples_observed")
        attempt["work"].pop("checkpointed_replay_examples")
        for key in (
            "hidden_width_observed",
            "peak_hidden_width_observed",
            "checkpointed_hidden_width",
            "checkpointed_peak_hidden_width",
            "rejected_proposed_hidden_width",
        ):
            attempt["work"].pop(key)
        attempt["budget"].pop("max_replay_examples")
        attempt["budget"].pop("max_hidden_width")
    (tmp_path / "run-state.json").write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", _arguments(tmp_path, "--resume", "--max-training-updates", "3")
    )
    cli.main()

    assert _state(tmp_path)["status"] == "completed"
    assert _state(tmp_path)["work"]["replay_examples_observed"] == 0


def test_should_distinguish_observed_replay_from_durable_after_sleep_save_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_save = TrustedLocalToyCheckpointStore.save

    def fail_after_sleep(self: TrustedLocalToyCheckpointStore, checkpoint: Any) -> None:
        if checkpoint.combined.position.stage == "after_sleep":
            raise OSError("injected after-sleep checkpoint failure")
        original_save(self, checkpoint)

    arguments = _arguments(
        tmp_path,
        "--sleep-interval",
        "1",
        "--replay-steps",
        "1",
        "--replay-memory-size",
        "2",
        "--max-replay-examples",
        "52",
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(TrustedLocalToyCheckpointStore, "save", fail_after_sleep)
        scoped.setattr(sys, "argv", arguments)
        with pytest.raises(OSError, match="injected after-sleep checkpoint failure"):
            cli.main()

    state = _state(tmp_path)
    assert (state["status"], state["error_type"]) == ("error", "OSError")
    assert state["work"]["replay_examples_observed"] == 52
    assert state["work"]["checkpointed_replay_examples"] == 0
    assert state["checkpoint"]["position"]["stage"] == "before_sleep"
    assert not (tmp_path / "result.json").exists()

    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    cli.main()
    completed = _state(tmp_path)
    assert completed["status"] == "completed"
    assert completed["work"]["replay_examples_observed"] == 52
    assert completed["work"]["checkpointed_replay_examples"] == 52


def test_should_record_transient_width_stop_and_checked_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _arguments(
        tmp_path,
        "--epochs",
        "2",
        "--sleep-interval",
        "1",
        "--split-threshold",
        "0",
        "--max-hidden-width",
        "12",
    )
    monkeypatch.setattr(sys, "argv", first)
    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 3
    state = _state(tmp_path)
    assert (state["status"], state["reason"]) == ("incomplete", "max_hidden_width")
    assert state["budget"]["max_hidden_width"] == 12
    assert (
        state["work"]["hidden_width_observed"],
        state["work"]["peak_hidden_width_observed"],
    ) == (
        12,
        12,
    )
    assert (
        state["work"]["checkpointed_hidden_width"],
        state["work"]["checkpointed_peak_hidden_width"],
        state["work"]["rejected_proposed_hidden_width"],
    ) == (12, 12, 14)
    assert state["checkpoint"]["position"]["stage"] == "before_sleep"
    assert not (tmp_path / "result.json").exists()

    monkeypatch.setattr(sys, "argv", [*first, "--resume", "--max-hidden-width", "14"])
    cli.main()
    completed = _state(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["work"]["peak_hidden_width_observed"] == 14
    assert completed["work"]["checkpointed_peak_hidden_width"] == 14
    assert completed["work"]["rejected_proposed_hidden_width"] is None
    first_sleep = result["circadian_sleep"]["events"][0]
    assert (first_sleep["before_width"], first_sleep["final_width"]) == (12, 12)
    assert len(first_sleep["changes"]["applied_split_pairs"]) == 2


def test_should_reject_invalid_width_flag_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-hidden-width", "0"))
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_should_record_initial_width_stop_without_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", _arguments(tmp_path, "--max-hidden-width", "11"))
    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 3
    state = _state(tmp_path)
    assert (state["status"], state["reason"]) == ("incomplete", "max_hidden_width")
    assert state["work"]["training_updates_observed"] == 0
    assert state["work"]["hidden_width_observed"] == 12
    assert state["work"]["checkpointed_hidden_width"] is None
    assert state["work"]["rejected_proposed_hidden_width"] == 12
    assert state["checkpoint"] is None
    assert not (tmp_path / "result.json").exists()


def test_should_record_transient_peak_when_after_sleep_checkpoint_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_save = TrustedLocalToyCheckpointStore.save

    def fail_after_sleep(self: TrustedLocalToyCheckpointStore, checkpoint: Any) -> None:
        if checkpoint.combined.position.stage == "after_sleep":
            raise OSError("injected width checkpoint failure")
        original_save(self, checkpoint)

    arguments = _arguments(
        tmp_path,
        "--epochs",
        "2",
        "--sleep-interval",
        "1",
        "--split-threshold",
        "0",
        "--max-hidden-width",
        "14",
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(TrustedLocalToyCheckpointStore, "save", fail_after_sleep)
        scoped.setattr(sys, "argv", arguments)
        with pytest.raises(OSError, match="injected width checkpoint failure"):
            cli.main()

    state = _state(tmp_path)
    assert (state["status"], state["error_type"]) == ("error", "OSError")
    assert (
        state["work"]["hidden_width_observed"],
        state["work"]["peak_hidden_width_observed"],
    ) == (
        12,
        14,
    )
    assert (
        state["work"]["checkpointed_hidden_width"],
        state["work"]["checkpointed_peak_hidden_width"],
    ) == (12, 12)
    assert state["checkpoint"]["position"]["stage"] == "before_sleep"
    assert not (tmp_path / "result.json").exists()

    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    cli.main()
    completed = _state(tmp_path)
    assert completed["status"] == "completed"
    assert completed["work"]["peak_hidden_width_observed"] == 14
    assert completed["work"]["checkpointed_peak_hidden_width"] == 14
