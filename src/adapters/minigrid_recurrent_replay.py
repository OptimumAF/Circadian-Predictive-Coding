"""Fixed matched episode rehearsal through existing buffer/runtime ports.

TRAIN metadata keys expose released context/cue to selection only; native inputs
and targets stay raw148/K7. Outputs are tagged copies, the existing rehearsal
owner and bounded evidence/counter checks. No optimizer, IO, model selection,
durable continuation or new replay algorithm lives here.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from typing import Any

from src.adapters.minigrid_public_cue_recurrent_learner import (
    PublicCueRecurrentPolicy,
    _cue,
    copy_snapshot,
)
from src.app.coupled_attribution import AttributionBatch, _validate_batch, _same_value
from src.app.stream_rehearsal import (
    StreamRehearsal,
    RecentStreamReplayPolicy,
    CoverageStreamReplayPolicy,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import EpisodeReplaySchema, copy_episode, episode_arrays

MODES = ("recent", "coverage")
KEYS = tuple(f"{context}-{cue}" for context in ("A", "B", "C_train") for cue in ("blue", "red"))
SCHEMA = EpisodeReplaySchema(148, 9, 7)
RETENTION = ReplayRetentionBudget(54, 256 * 1024)


def validate_mode(mode: str) -> str:
    if type(mode) is not str or mode not in MODES:
        raise ValueError("recurrent replay requires exact recent or coverage mode")
    return mode


def native_updates(wakes: int) -> int:
    if type(wakes) is not int or not 0 <= wakes <= 128:
        raise ValueError("recurrent replay wake clock must be an integer in0..128")
    return wakes + 2 * (wakes // 8)


def check_snapshot(state: Any, wakes: int) -> Any:
    owned = copy_snapshot(state)
    expected = native_updates(wakes)
    per_head = ((wakes + 1) // 2 + wakes // 8, wakes // 2 + wakes // 8)
    if (
        owned.policy != PublicCueRecurrentPolicy()
        or owned.failed_work is not None
        or (owned.completed_updates, owned.training_examples) != (expected, 9 * expected)
        or any(
            (h.completed_updates, h.training_examples) != (n, 9 * n)
            for h, n in zip(owned.heads, per_head, strict=True)
        )
    ):
        raise ValueError("replay native/per-head clocks differ from paid wake+episode schedule")
    return owned


def prepare_packets(packets: tuple[AttributionBatch, ...]) -> tuple[AttributionBatch, ...]:
    if type(packets) is not tuple or len(packets) not in (32, 128):
        raise ValueError("replay requires original128or explicit32qualification packets")
    result = []
    seen: set[str] = set()
    for i, packet in enumerate(packets):
        _validate_batch(packet, "train", seen)
        context = ("A", "B", "C_train", "A")[i // (len(packets) // 4)]
        if (
            packet.episode is None
            or packet.context != context
            or packet.features.shape != (9, 148)
            or packet.num_classes != 7
        ):
            raise ValueError("replay requires four ordered complete raw148/K7 TRAIN phases")
        episode = copy_episode(packet.episode)
        if _cue(packet.features[0]) != i % 2:
            raise ValueError("replay requires original alternating actual public cues")
        key = f"{context}-" + ("blue", "red")[i % 2]
        tagged = replace(episode, context=key)
        x, y = episode_arrays(tagged)
        result.append(replace(packet, context=key, episode=tagged, features=x, targets=y))
    return tuple(result)


def policy(mode: str):
    mode = validate_mode(mode)
    return (
        RecentStreamReplayPolicy(8, 18, RETENTION)
        if mode == "recent"
        else CoverageStreamReplayPolicy(8, 18, RETENTION, KEYS)
    )


def validate_packets(packets: tuple[AttributionBatch, ...]) -> None:
    if type(packets) is not tuple or len(packets) not in (32, 128):
        raise ValueError("recurrent replay requires128or32ordered tagged TRAIN packets")
    seen: set[str] = set()
    for i, packet in enumerate(packets):
        _validate_batch(packet, "train", seen)
        key = ("A", "B", "C_train", "A")[i // (len(packets) // 4)] + "-" + ("blue", "red")[i % 2]
        if (
            packet.context != key
            or packet.episode is None
            or packet.num_classes != 7
            or packet.features.shape != (9, 148)
            or _cue(packet.features[0]) != i % 2
        ):
            raise ValueError("recurrent replay tagged context/public-cue/shape contract differs")


def expected_episodes(packets: tuple[AttributionBatch, ...], mode: str, event: int) -> tuple:
    """Causal prefix only; no model result or future label affects selection."""
    validate_mode(mode)
    if type(event) is not int or event < 8 or event % 8 or event > len(packets):
        raise ValueError("replay event must be a declared completed eighth wake")
    if mode == "recent":
        return tuple(p.episode for p in packets[event - 2 : event])
    latest = {}
    cursor = -1
    chosen: tuple = ()
    for n in range(8, event + 1, 8):
        for packet in packets[n - 8 : n]:
            latest[packet.context] = packet.episode
        order = (*KEYS[cursor + 1 :], *KEYS[: cursor + 1])
        keys = tuple(k for k in order if k in latest)[:2]
        chosen = tuple(latest[k] for k in keys)
        cursor = KEYS.index(keys[-1])
    return chosen


class RecurrentReplayWitness:
    """One original helper and finite admitted event prefix; no success chain."""

    def __init__(self, source, wake_budget, packets, mode, sampler):
        self.mode = validate_mode(mode)
        validate_packets(packets)
        self.packets = packets
        self._expected = {
            n: expected_episodes(packets, self.mode, n) for n in range(8, len(packets) + 1, 8)
        }
        self._expected_values = {n: [asdict(e) for e in es] for n, es in self._expected.items()}
        self.source = source
        self.initial = source.snapshot_state()
        from src.adapters.minigrid_learning_pilot import _signature

        self._packet_signature = _signature(packets)
        self._seals: list[tuple[Any, bytes]] = []
        self._active: tuple[Any, tuple[Any, ...], bytes] | None = None
        self.replay_budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=9 * 2 * (len(packets) // 8),
                max_wall_seconds=wake_budget.budget.max_wall_seconds,
                max_process_rss_bytes=wake_budget.budget.max_process_rss_bytes,
            ),
            wake_budget.clock,
        )
        self.replay_budget.attach_memory(sampler)
        self.helper = StreamRehearsal(
            source,
            wake_budget,
            policy(mode),
            replay_budget=self.replay_budget,
            total_updates=len(packets),
            input_dim=148,
            episode_schema=SCHEMA,
            check_boundary=self.check_prior,
        )
        h = self.helper
        self._slot_owners = tuple(h._events)
        self._unstarted = tuple(pickle.dumps(row, protocol=5) for row in h._events)
        self._bindings = {
            k: getattr(h, k)
            for k in (
                "_learner",
                "_budget",
                "_replay_budget",
                "_policy",
                "_buffer",
                "_events",
                "_episode_schema",
                "_check_boundary",
                "_original_check_boundary",
            )
        }
        self._policy_bytes = pickle.dumps(h._policy, protocol=5)
        self._budget_bindings = [
            (
                b,
                b.budget,
                deepcopy(vars(b.budget)),
                b.clock,
                b.started_at,
                b.progress,
                b.process_rss_sampler,
            )
            for b in (wake_budget, self.replay_budget)
        ]
        self._clock_floors = [b.last_clock for b in (wake_budget, self.replay_budget)]

    def check_prior(self, *, include_inputs=True):
        h = self.helper
        if (
            any(getattr(h, k) is not v for k, v in self._bindings.items())
            or pickle.dumps(h._policy, protocol=5) != self._policy_bytes
            or len(h._events) != len(self._slot_owners)
            or any(
                row is not owner for row, owner in zip(h._events, self._slot_owners, strict=True)
            )
        ):
            raise ValueError("original replay helper/policy/input/source binding changed")
        if include_inputs:
            from src.adapters.minigrid_learning_pilot import _check_signature

            if not _check_signature(self.packets, self._packet_signature) or not _same_value(
                self.source.snapshot_state(), self.initial
            ):
                raise ValueError("original replay input/source state changed")
        for i, (b, limits, values, clock, birth, progress, sampler) in enumerate(
            self._budget_bindings
        ):
            if (
                b.budget is not limits
                or vars(limits) != values
                or b.clock is not clock
                or b.started_at != birth
                or b.progress is not progress
                or b.process_rss_sampler is not sampler
                or b.last_clock < self._clock_floors[i]
            ):
                raise ValueError("original replay budget, clock, sampler or birth changed")
            self._clock_floors[i] = b.last_clock
        if (
            self.replay_budget.updates_completed != 0
            or self.replay_budget.replay_examples_completed
            != 9 * sum(r["attempted_updates"] for r in h._events)
        ):
            raise ValueError("replay permanent charges changed outside observed native attempts")
        for i, (owner, saved) in enumerate(self._seals):
            if h._events[i] is not owner or pickle.dumps(owner, protocol=5) != saved:
                raise ValueError("preceding completed replay evidence was corrupted or replaced")
        for i, row in enumerate(h._events[len(self._seals) :], len(self._seals)):
            if not row["selected_episodes"]:
                if pickle.dumps(row, protocol=5) != self._unstarted[i]:
                    raise ValueError("unstarted replay evidence changed")
            else:
                if row["selected_episodes"] != self._expected_values[row["updates"]]:
                    raise ValueError("active replay selection changed before native work")
                diagnostics = row["diagnostics"]
                if (
                    type(row["returned_updates"]) is not int
                    or not 0 <= row["returned_updates"] <= row["attempted_updates"] <= 2
                    or len(diagnostics) != row["returned_updates"]
                ):
                    raise ValueError("active replay returned-call prefix changed")
                if self._active is not None:
                    owner, prior, payload = self._active
                    if (
                        row is not owner
                        or len(diagnostics) < len(prior)
                        or any(diagnostics[i] is not d for i, d in enumerate(prior))
                        or pickle.dumps(diagnostics[: len(prior)], protocol=5) != payload
                    ):
                        raise ValueError(
                            "preceding returned replay diagnostic was replaced or changed"
                        )
                self._active = (row, tuple(diagnostics), pickle.dumps(diagnostics, protocol=5))

    def admit(self, wakes: int, *, include_inputs=True):
        self.check_prior(include_inputs=include_inputs)

        completed = wakes // 8
        events = self.helper._events
        if (
            self.helper._observed != wakes
            or sum(r["committed_updates"] for r in events) != 2 * completed
            or sum(r["returned_updates"] for r in events) != 2 * completed
            or sum(r["attempted_updates"] for r in events) != 2 * completed
            or self.replay_budget.updates_completed != 0
            or self.replay_budget.replay_examples_completed != 18 * completed
        ):
            raise ValueError("replay completed calls/charges do not match the wake prefix")
        while len(self._seals) < completed:
            i = len(self._seals)
            row = self.helper._events[i]
            expected = self._expected[(i + 1) * 8]
            if (
                row["status"] != "complete"
                or row["updates"] != (i + 1) * 8
                or any(
                    row[k] != 2
                    for k in ("attempted_updates", "returned_updates", "committed_updates")
                )
                or row["charged_exposures"] != 18
                or row["receipt"] is None
                or row["selected_episodes"] != self._expected_values[(i + 1) * 8]
                or [_cue(episode_arrays(e)[0][0]) for e in expected] != [0, 1]
            ):
                raise ValueError(
                    "replay selection/counters differ from original balanced causal policy"
                )
            self._seals.append((row, pickle.dumps(row, protocol=5)))
            self._active = None
        self.check_prior(include_inputs=include_inputs)

    def evidence(self) -> dict[str, Any]:
        """Export retained proofs, never repair a damaged live helper."""
        try:
            report = deepcopy(self.helper.report)
        except Exception as error:
            report = dict(report_error=f"{type(error).__name__}: {error}")
        try:
            self.check_prior(include_inputs=False)
            report["prior_evidence_preserved"] = True
        except Exception as error:
            report["prior_evidence_preserved"] = None
            report["preservation_error"] = f"{type(error).__name__}: {error}"
            report["verified_completed_prefix"] = [pickle.loads(value) for _, value in self._seals]
        report["verified_completed_events"] = len(self._seals)
        return report
