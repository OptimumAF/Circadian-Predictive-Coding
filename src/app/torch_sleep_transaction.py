"""Capture process random streams around one Torch sleep attempt.

Inputs are the selected Torch device and process RNGs; outputs are a detached
state that can be restored after an aborted attempt. This module does not
snapshot models, choose guards, emit events, or persist checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TorchSleepProcessRandomState:
    python: Any
    numpy: Any
    torch_cpu: Any
    torch_cuda: Any | None


def capture_torch_sleep_process_random(torch: Any, device: Any) -> TorchSleepProcessRandomState:
    cuda = torch.cuda.get_rng_state(device).clone() if torch.device(device).type == "cuda" else None
    return TorchSleepProcessRandomState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda,
    )


def restore_torch_sleep_process_random(
    torch: Any, device: Any, state: TorchSleepProcessRandomState
) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state(state.torch_cuda, device)
