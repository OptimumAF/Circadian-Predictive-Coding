"""One fixed released CLM CPU readout using the plain-text TRAIN/HF recipe.

This is not vLLM serving parity, adaptation or a comparative benchmark. All
assets must already exist at their exact pinned identities; no download occurs.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from typing import Any

from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.infra.clm_local_loader import HEAD_REVISION, load_local_clm
from src.shared.process_memory import ProcessRssSampler

SOURCE = Path("artifacts/runs/native-decision-contracts-20261022/clm")
INSTRUCTIONS = "Choose the support workflow that matches the customer's request."


def fixed_request() -> DecisionRequest:
    return DecisionRequest(
        "clm-frozen-train-0",
        "I forgot my password and cannot sign in.",
        (
            DecisionCandidate("password", "Reset account password"),
            DecisionCandidate("parcel", "Check parcel tracking"),
            DecisionCandidate("refund", "Request an invoice refund"),
        ),
    )


def _definitions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf8"))
    selected = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if len(selected) != len(names):
        raise ValueError("pinned reference definition missing")
    body: list[ast.stmt] = list(selected)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope


def run_case(model_dir: Path, head: Path, report: dict[str, Any], *, save=lambda: None) -> None:
    import math
    import torch
    from src.adapters.clm_decision import CLMDecisionPredictor, make_clm_head
    from src.adapters.clm_frozen_encoder import FrozenCLMEncoder

    torch.set_num_threads(2)
    started = monotonic()
    report.update(
        complete=False,
        events=[],
        role="train",
        encoder_work={},
        predictor_work={},
        scope="released CPU plain TRAIN/HF readout; vLLM parity and task adaptation unqualified",
    )
    save()
    caller_rng = torch.get_rng_state().clone()
    encoder = predictor = None

    def progress(event):
        report["events"].append(dict(elapsed_seconds=monotonic() - started, **event))
        save()

    with ProcessRssSampler(interval_seconds=0.02) as sampler:

        def before_step():
            if monotonic() - started > 210:
                raise ValueError("native inner wall allowance exhausted")
            memory = sampler.sample()
            if memory is None or memory > 24 * 1024**3:
                raise ValueError("native sampled RSS unavailable or allowance exceeded")

        try:
            loaded = load_local_clm(
                model_dir,
                head,
                head_factory=make_clm_head,
                before_step=before_step,
                progress=progress,
            )
            report["torch_version"] = torch.__version__
            report["identity"] = asdict(loaded.identity)
            paid: dict[str, Any] = {}

            def check_inputs(_model, args, kwargs):
                ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
                if (
                    ids.shape != mask.shape
                    or ids.ndim != 2
                    or ids.shape[0] > 4
                    or ids.shape[1] > 96
                ):
                    raise ValueError("native token/sequence allowance exceeded before forward")
                if ids.numel() > 384 or kwargs.get("use_cache") is not False:
                    raise ValueError("native padded token/cache allowance exceeded")
                paid.update(input_ids=ids.clone(), mask=mask.clone())
                progress(
                    dict(
                        event="encoder_forward_attempt",
                        sequences=int(ids.shape[0]),
                        sequence_width=int(ids.shape[1]),
                        padded_tokens=int(ids.numel()),
                    )
                )
                before_step()

            def raw_output(_model, args, output):
                paid["hidden"] = output.last_hidden_state
                progress(dict(event="encoder_forward_returned"))

            pre = loaded.model.register_forward_pre_hook(check_inputs, with_kwargs=True)
            post = loaded.model.register_forward_hook(raw_output)
            encoder = FrozenCLMEncoder(
                loaded.model,
                loaded.tokenizer,
                identity=loaded.identity,
                max_sequences=4,
                max_forward_calls=1,
            )
            predictor = CLMDecisionPredictor(
                encoder,
                loaded.state_head,
                loaded.action_head,
                loaded.logit_scale,
                instructions=INSTRUCTIONS,
                head_revision=HEAD_REVISION,
            )
            request = fixed_request()
            report["request"] = asdict(request)
            try:
                result = predictor(request)
            finally:
                pre.remove()
                post.remove()
            report["result"] = asdict(result)
            before_step()
            readout = encoder.last_readout
            reference_last = torch.stack(
                [
                    paid["hidden"][i, int(paid["mask"][i].sum()) - 1].float()
                    for i in range(len(paid["mask"]))
                ]
            )
            raw_error = float((reference_last - readout["last_token_states"]).abs().max())
            import numpy as np

            array = reference_last.detach().numpy()
            source_norm = array / (np.linalg.norm(array, axis=-1, keepdims=True) + 1e-12)
            norm_error = float(np.max(np.abs(source_norm - readout["embeddings"].numpy())))
            with torch.no_grad():
                reference_features = torch.from_numpy(source_norm)
                zs = torch.nn.functional.normalize(
                    loaded.state_head(reference_features[:1]), dim=-1
                )
                za = torch.nn.functional.normalize(
                    loaded.action_head(reference_features[1:]), dim=-1
                )
                logits = (loaded.logit_scale.exp().clamp(max=100.0) * zs @ za.t())[0].tolist()
            namespace: dict[str, Any] = dict(
                Any=Any, math=math, json=json, QUESTION_TYPES=("noul", "choice", "score")
            )
            _definitions(
                SOURCE / "src__clm__schema.py",
                {
                    "to_text",
                    "state_text",
                    "softmax",
                    "confidence",
                    "answer_from_probs",
                    "answer_from_logits",
                },
                namespace,
            )
            keys = [c.candidate_id for c in request.allowed_candidates]
            answer = namespace["answer_from_logits"]({"type": "choice"}, keys, logits)
            native_readout = predictor.last_readout
            probability_error = max(
                abs(dict(result.probabilities)[k] - answer["probabilities"][k]) for k in keys
            )
            recipe_scope = _definitions(SOURCE / "train__embed_utils.py", {"Recipe"}, {})
            recipe = object.__new__(recipe_scope["Recipe"])
            recipe.tok, recipe.cap = loaded.tokenizer, 2047
            texts = (namespace["state_text"](request.query, INSTRUCTIONS),) + tuple(
                c.description for c in request.allowed_candidates
            )
            expected_ids = [
                recipe.text_ids(t, keep=k)
                for t, k in zip(texts, ("tail", "head", "head", "head"), strict=True)
            ]
            token_match = expected_ids == [list(row) for row in readout["token_ids"]]
            oracle_tokens = sum(map(len, expected_ids))
            report["reference"] = dict(
                raw_last_abs_error=raw_error,
                normalized_abs_error=norm_error,
                probability_abs_error=probability_error,
                token_recipe_match=token_match,
                source_decoded_answer=answer,
                tokenizer_oracle_calls=4,
                oracle_token_visits=oracle_tokens,
                selected_id_match=result.selected_id == answer["choice"],
                confidence_abs_error=abs(result.reported_confidence - answer["confidence"]),
            )
            report["native_readout"] = native_readout
            report["tokens"] = {
                k: readout[k] for k in ("token_ids", "retained_lengths", "padded_tokens")
            }
            report["descriptive_expected_password"] = result.selected_id == "password"
            assert raw_error == 0 and norm_error <= 2e-7 and probability_error <= 2e-6
            assert token_match and result.selected_id == answer["choice"]
            assert abs(result.reported_confidence - answer["confidence"]) <= 2e-6
            assert encoder.work["token_ids_visited"] + oracle_tokens <= 2048
            assert encoder.work["tokenization_attempts"] + 4 <= 12
            before_step()
            report["caller_rng_unchanged"] = torch.equal(caller_rng, torch.get_rng_state())
            assert report["caller_rng_unchanged"]
            report["complete"] = True
        finally:
            if encoder is not None:
                report["encoder_work"] = encoder.work
            if predictor is not None:
                report["predictor_work"] = predictor.work
            report["elapsed_seconds"] = monotonic() - started
            report["sampled_process_rss"] = asdict(sampler.snapshot())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace existing native evidence")
    report: dict[str, Any] = {}

    def save():
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )

    try:
        run_case(args.model_dir, args.head, report, save=save)
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
