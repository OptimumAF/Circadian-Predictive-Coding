"""Local-only pinned CLM payload admission and optional CPU runtime loading.

No downloader, remote Python, training or serving endpoint lives here. The caller
supplies the native head factory and owns resource monitoring at these boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable

from src.core.clm_head import FrozenEncoderIdentity

ENCODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
HEAD_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
HEAD_FILE = "CLM_v0.1-8B.pt"
HEAD_BYTES = 75557149
HEAD_SHA256 = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
# Published at the exact encoder revision; metadata provenance in upstream docs.
ENCODER_FILES = {
    "config.json": {
        "bytes": 728,
        "kind": "git_blob_sha1",
        "digest": "d46195ac87f837ad233d02b2f80f148bf7c005e0",
    },
    "merges.txt": {
        "bytes": 1671853,
        "kind": "git_blob_sha1",
        "digest": "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",
    },
    "model-00001-of-00005.safetensors": {
        "bytes": 3996250744,
        "kind": "sha256",
        "digest": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    },
    "model-00002-of-00005.safetensors": {
        "bytes": 3993160032,
        "kind": "sha256",
        "digest": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    },
    "model-00003-of-00005.safetensors": {
        "bytes": 3959604768,
        "kind": "sha256",
        "digest": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    },
    "model-00004-of-00005.safetensors": {
        "bytes": 3187841392,
        "kind": "sha256",
        "digest": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    },
    "model-00005-of-00005.safetensors": {
        "bytes": 1244659840,
        "kind": "sha256",
        "digest": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    },
    "model.safetensors.index.json": {
        "bytes": 32878,
        "kind": "git_blob_sha1",
        "digest": "2b85c00f1b118961cd7a477e2bba0fe197a4ce1a",
    },
    "tokenizer.json": {
        "bytes": 11422654,
        "kind": "sha256",
        "digest": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
    "tokenizer_config.json": {
        "bytes": 9732,
        "kind": "git_blob_sha1",
        "digest": "417d038a63fa3de29cfde265caedae14d1a58d92",
    },
    "vocab.json": {
        "bytes": 2776833,
        "kind": "git_blob_sha1",
        "digest": "4783fe10ac3adce15ac8f358ef5462739852c569",
    },
}


def frozen_identity(max_tokens: int = 2048) -> FrozenEncoderIdentity:
    if type(max_tokens) is not int or not 2 <= max_tokens <= 2048:
        raise ValueError("max_tokens must be an integer in2..2048")
    return FrozenEncoderIdentity(
        "Qwen/Qwen3-8B",
        ENCODER_REVISION,
        ENCODER_REVISION,
        "last",
        f"clm-plain-state-tail-action-head-v1:max_len={max_tokens}:cap={max_tokens - 1}:l2-plus-1e-12",
        4096,
    )


def _verify(path: Path, specification: dict, before_step: Callable[[], None]) -> None:
    if not path.is_file() or path.stat().st_size != specification["bytes"]:
        raise ValueError(f"missing or wrong-size pinned CLM asset: {path.name}")
    size = specification["bytes"]
    digest = (
        hashlib.sha256()
        if specification["kind"] == "sha256"
        else hashlib.sha1(b"blob " + str(size).encode("ascii") + b"\0")
    )
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            before_step()
            digest.update(chunk)
    if digest.hexdigest() != specification["digest"]:
        raise ValueError(f"pinned CLM asset digest mismatch: {path.name}")


def validate_local_clm(model_dir: Path, head_path: Path, *, before_step=lambda: None) -> None:
    """Require the actual pinned local assets; never fetch or substitute weights."""
    for name, specification in ENCODER_FILES.items():
        _verify(Path(model_dir) / name, specification, before_step)
    _verify(Path(head_path), dict(bytes=HEAD_BYTES, kind="sha256", digest=HEAD_SHA256), before_step)


@dataclass(frozen=True)
class LoadedCLM:
    model: Any
    tokenizer: Any
    state_head: Any
    action_head: Any
    logit_scale: Any
    identity: FrozenEncoderIdentity
    loading_info: dict


def _plain_loading_info(info: dict) -> dict:
    """Installed Transformers5 returns sets; retain them as JSON-safe lists."""
    return {
        key: sorted(value) if isinstance(value, (set, frozenset)) else value
        for key, value in info.items()
    }


def _head_kwargs(checkpoint: dict) -> dict:
    config = checkpoint.get("cfg")
    if type(config) is not dict or len(config) > 128:
        raise ValueError("released CLM checkpoint requires bounded cfg")
    result = dict(
        width=config.get("width"),
        depth=config.get("depth"),
        proj=checkpoint.get("projection_dim", config.get("projection_dim", 512)),
        activation=config.get("activation", "gelu"),
        layernorm=config.get("layernorm", False),
        residual=config.get("residual", False),
        hidden=config.get("hidden_size", 4096),
    )
    if (
        type(result["width"]) is not int
        or not 1 <= result["width"] <= 8192
        or type(result["depth"]) is not int
        or not 2 <= result["depth"] <= 8
        or result["proj"] != 512
        or result["hidden"] != 4096
        or result["activation"] not in ("gelu", "relu", "silu")
        or any(type(result[k]) is not bool for k in ("layernorm", "residual"))
    ):
        raise ValueError("released CLM head geometry/configuration differs")
    return result


def load_local_clm(
    model_dir: Path,
    head_path: Path,
    *,
    head_factory,
    before_step=lambda: None,
    progress=lambda event: None,
) -> LoadedCLM:
    """Load once in the caller's CPU budget; return an explicitly frozen owner.

    This Transformers plain-text TRAIN recipe does not attest vLLM kernel parity.
    Safe tensor-only checkpoint loading and no remote code are required.
    """
    validate_local_clm(model_dir, head_path, before_step=before_step)
    progress(dict(event="assets_verified"))
    import torch
    from transformers import AutoModel, AutoTokenizer

    with torch.random.fork_rng(devices=[]):
        before_step()
        progress(dict(event="tokenizer_load_attempt"))
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        progress(dict(event="tokenizer_loaded"))
        before_step()
        progress(dict(event="encoder_load_attempt"))
        model, loading = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            output_loading_info=True,
        )
        loading = _plain_loading_info(loading)
        progress(dict(event="encoder_load_returned", loading_info=loading))
        # The released causal checkpoint also contains a language-model head;
        # the frozen embedding backbone deliberately does not instantiate it.
        if (
            loading.get("missing_keys")
            or loading.get("mismatched_keys")
            or set(loading.get("unexpected_keys", [])) - {"lm_head.weight"}
            or loading.get("error_msgs")
        ):
            raise ValueError(f"incomplete released encoder load: {loading}")
        model.eval().requires_grad_(False)
        if (
            model.config.model_type != "qwen3"
            or model.config.hidden_size != 4096
            or any(p.dtype != torch.bfloat16 or p.device.type != "cpu" for p in model.parameters())
        ):
            raise ValueError("released encoder is not the declared CPU BF16 backbone")
        before_step()
        progress(
            dict(
                event="encoder_loaded",
                parameters=sum(p.numel() for p in model.parameters()),
                loading_info=loading,
            )
        )
        progress(dict(event="head_checkpoint_load_attempt"))
        checkpoint = torch.load(str(head_path), map_location="cpu", weights_only=True)
        if type(checkpoint) is not dict:
            raise ValueError("released CLM checkpoint must be a native tensor dictionary")
        kwargs = _head_kwargs(checkpoint)
        progress(dict(event="head_pair_construction_attempt", device="meta"))
        with torch.device("meta"):
            state_head, action_head = head_factory(**kwargs), head_factory(**kwargs)
        if sum(p.numel() for h in (state_head, action_head) for p in h.parameters()) > 32000000:
            raise ValueError("released CLM head parameter cap exceeded")
        for head, name in ((state_head, "state_head"), (action_head, "action_head")):
            head.load_state_dict(checkpoint[name], strict=True, assign=True)
            head.float().eval().requires_grad_(False)
            if any(
                p.device.type != "cpu" or not torch.isfinite(p).all().item()
                for p in head.parameters()
            ):
                raise ValueError("released CLM head contains invalid native parameters")
        scale = torch.as_tensor(checkpoint["logit_scale"]).detach().float().clone()
        if scale.shape != () or not torch.isfinite(scale).item():
            raise ValueError("released CLM log scale must be a finite scalar")
        progress(
            dict(
                event="heads_loaded",
                head_kwargs=kwargs,
                head_parameters=sum(
                    p.numel() for h in (state_head, action_head) for p in h.parameters()
                ),
            )
        )
        before_step()
    return LoadedCLM(model, tokenizer, state_head, action_head, scale, frozen_identity(), loading)
