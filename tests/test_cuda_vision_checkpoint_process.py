"""A CUDA unmatched-vision file resumes in a second Python process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host required")


_WORKER = """
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from _pytest.monkeypatch import MonkeyPatch
import torch

sys.path.insert(0, str(Path.cwd() / "tests"))
from test_cuda_vision_checkpoint_resume import (
    _config, _install_cuda_random_backbone, _next_draws, _seed_all,
    InterruptAtStage, InterruptedAfterSave,
)
from src.app import resnet50_benchmark as vision
from src.app.resnet50_benchmark import VISION_SEEDED_UNMATCHED_PROTOCOL
from src.infra.circadian_checkpoint_files import TrustedLocalVisionCheckpointStore

mode, directory = sys.argv[1:]
torch.use_deterministic_algorithms(True)
_install_cuda_random_backbone(MonkeyPatch())
def controlled_pc_score(_torch, model, loader, _device, max_batches=None, *, on_examples_scored=None):
    if on_examples_scored is not None:
        batch_count = len(loader) if max_batches is None else min(len(loader), max_batches)
        on_examples_scored(min(len(loader.dataset), batch_count * loader.batch_size))
    return 0.8, 0.5

vision._compute_pc_metrics = controlled_pc_score
config = _config(VISION_SEEDED_UNMATCHED_PROTOCOL, reject=False)
order = ("circadian", "predictive", "backprop")
root = Path(directory)
store = TrustedLocalVisionCheckpointStore(root / "resume.ckpt")

if mode == "interrupt":
    _seed_all()
    control = vision.run_resnet50_benchmark(
        config, model_order=order,
        checkpoint_store=TrustedLocalVisionCheckpointStore(root / "control.ckpt"),
    )
    expected_draws = _next_draws()
    _seed_all()
    try:
        vision.run_resnet50_benchmark(
            config, model_order=order,
            checkpoint_store=InterruptAtStage(store, "wake"),
        )
    except InterruptedAfterSave:
        saved = store.load()
        assert saved.active_circadian is not None
        assert saved.active_circadian.stage == "wake"
        print(json.dumps({"pid": os.getpid(), "hashes": control.trained_model_hashes,
                          "draws": expected_draws, "device": saved.torch_cuda_device,
                          "cursor": saved.active_circadian.loader_state.next_batch_index}))
    else:
        raise AssertionError("CUDA vision run did not interrupt at wake")
else:
    random.seed(999)
    np.random.seed(998)
    torch.manual_seed(997)
    torch.cuda.manual_seed_all(996)
    resumed = vision.run_resnet50_benchmark(
        config, model_order=order, checkpoint_store=store, resume_from_checkpoint=True,
    )
    print(json.dumps({"pid": os.getpid(), "hashes": resumed.trained_model_hashes,
                      "draws": _next_draws()}))
"""


def _run_worker(mode: str, directory: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER, mode, str(directory)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        capture_output=True,
        text=True,
        timeout=35,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_cuda_vision_checkpoint_wake_resumes_in_new_process(tmp_path: Path) -> None:
    interrupted = _run_worker("interrupt", tmp_path)
    resumed = _run_worker("resume", tmp_path)

    assert interrupted["pid"] != resumed["pid"]
    assert interrupted["device"] == "cuda:0"
    assert interrupted["cursor"] == 1
    assert resumed["hashes"] == interrupted["hashes"]
    assert resumed["draws"] == interrupted["draws"]
