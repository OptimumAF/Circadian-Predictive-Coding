"""Actual public command on literal loader plus before-loader option refusal."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import scripts.run_clm_temporal_sequence as study
from src.adapters.clm_frozen_encoder import text_recipe
from src.core.clm_head import FrozenEncoderIdentity
from test_clm_catalog_learning import parts
from test_clm_linear_checkpoint_probes import LiteralProbeBackbone

capture = None


def test_should_execute_public_cli_with_literal_encoder_and_no_head_learning(monkeypatch, tmp_path):
    global capture
    texts, actions = study.feature_texts()
    lookup = {s: i + 1 for i, s in enumerate(texts + actions)}

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, *, add_special_tokens):
            return {"input_ids": [lookup[text]]}

    state, action, scale = parts(torch)
    loaded = SimpleNamespace(
        model=LiteralProbeBackbone().eval(),
        tokenizer=Tokenizer(),
        state_head=state,
        action_head=action,
        logit_scale=scale.detach(),
        identity=FrozenEncoderIdentity("literal", "fixed", "fixed", "last", text_recipe(2048), 2),
    )
    monkeypatch.setattr(study, "load_local_clm", lambda *a, **k: loaded)
    real_threads = torch.set_num_threads
    monkeypatch.setattr(torch, "set_num_threads", lambda n: real_threads(1))
    monkeypatch.setattr(study, "_control_readout", lambda *a: pytest.fail("unused head forward"))
    path = tmp_path / "actual.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "study",
            "--model-dir",
            "x",
            "--head",
            "x",
            "--output",
            str(path),
            "--catalog-training",
            "--train-linear-update",
            "--train-catalog-diagnostics",
        ],
    )
    study.main()
    result = json.loads(path.read_text())
    assert result["complete"] and result["control_kind"] == "untrained_uniform_union_linear"
    assert (
        result["learner_work"]["solve_returns"] == 12
        and result["learner_work"]["optimizer_steps"] == 0
    )
    assert result["encoder_work"]["forward_returns"] == 1
    capture = {"definition": "literal_cli_not_pretrained_v1", "report": result}


@pytest.mark.parametrize(
    "flag",
    [
        "--reserved-task-probes",
        "--train-state-only",
        "--typed-development-readout",
        "--compare-catalog-support",
    ],
)
def test_should_refuse_incompatible_flags_before_loader_and_output(monkeypatch, tmp_path, flag):
    monkeypatch.setattr(
        study, "load_local_clm", lambda *a, **k: pytest.fail("loaded invalid options")
    )
    path = tmp_path / "no.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "study",
            "--model-dir",
            "x",
            "--head",
            "x",
            "--output",
            str(path),
            "--catalog-training",
            "--train-linear-update",
            flag,
        ],
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not path.exists()


def test_should_require_catalog_before_loader(monkeypatch):
    monkeypatch.setattr(
        study, "load_local_clm", lambda *a, **k: pytest.fail("loaded invalid options")
    )
    with pytest.raises(ValueError, match="linear TRAIN"):
        study.run_case(Path("unused"), Path("unused"), {}, train_linear_update=True)


@pytest.mark.parametrize("fault", [None, "control_kind", "control_values"])
def test_should_validate_cached_literal_cli_without_new_model_work(monkeypatch, fault):
    # Third Windows cohort reuses only our marked software CLI output; ordinary
    # full test runs use the preceding live literal CLI test in this module.
    saved = capture
    if saved is None:
        saved = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "artifacts/runs/clm-linear-stream-20261025/windows-2-literal-cli.txt"
            ).read_text()
        )
    assert saved["definition"] == "literal_cli_not_pretrained_v1"
    report = deepcopy(saved["report"])
    if fault is None:
        assert len(study.score_train_choices(report)) == 3
        assert len(study.score_sequence(report)["checkpoints"]) == 4
    else:
        if fault == "control_kind":
            report["control_kind"] = "released_head"
        else:
            report["control_probabilities"]["v2"][0][0] = 0.5
        monkeypatch.setattr(
            study, "_score_cached", lambda *a: pytest.fail("scored changed control")
        )
        with pytest.raises(ValueError):
            study.score_sequence(report)
