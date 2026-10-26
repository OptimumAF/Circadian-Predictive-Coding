"""Route by one cue visible in the initial public frame, retaining spatial input.

Input is raw initial/current296 history; output is canonical296 plus two cue
bits. No labels, context names, next frames, environment or IO enters routing.
"""

from src.core.minigrid_invariant_encoding import _decode_coordinate, canonicalize_public_history

CONTEXT_HISTORY_ENCODING_ID = "minigrid_public_initial_cue_context_history_298_v1"


def context_public_history(history: tuple[float, ...]) -> tuple[float, ...]:
    """Validate both frames before selecting the initial red/blue cue head."""
    canonical = canonicalize_public_history(history)
    cues = [
        _decode_coordinate(canonical[offset + 1], 5)
        for offset in range(0, 147, 3)
        if _decode_coordinate(canonical[offset], 10) == 6
    ]
    if len(cues) != 1 or cues[0] not in (0, 2):
        raise ValueError("context history requires exactly one initial visible red or blue cue")
    # Why: the initial public cue persists when the current view loses it. Keep
    # all canonical spatial/color data; this port does not remap key semantics.
    return canonical + ((1.0, 0.0) if cues[0] == 2 else (0.0, 1.0))
