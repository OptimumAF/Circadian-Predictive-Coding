"""New fixed23 provisional targets on an unchanged executed student trajectory.

Own explicit query consent/raw labels and a TS12 view using executed actions.
No privileged geometry, simulator, teacher qualification, role conversion,
model, optimizer, promotion or IO. Original reserved teacher packets stay intact.
"""

from dataclasses import dataclass

from src.core.continuous_demonstration import ContinuousEpisode, copy_continuous_episode
from src.core.experience import require_tick
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.pick_cube_readiness_corrections import ReadinessCorrectionLabel, _observed_grasp_flags
from src.core.pick_cube_action_history import _action_history_episode
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

PROVENANCE = "provisional_probe_train23_student_state_goal_consistent_v1"
STUDENT_ID = "pick-cube-probe-train-smoke-23"


@dataclass(frozen=True)
class ProbeCorrectionConsent:
    max_queries: int
    role: str = "probe_train"
    seed: int = 23

    def __post_init__(self):
        if (
            type(self.max_queries) is not int
            or not 1 <= self.max_queries <= 50
            or type(self.role) is not str
            or self.role != "probe_train"
            or type(self.seed) is not int
            or self.seed != 23
        ):
            raise ValueError("new correction consent requires probe_train23 and1..50queries")


@dataclass(frozen=True)
class ProbeCorrectionBatch:
    consent: ProbeCorrectionConsent
    episode: ContinuousEpisode
    labels: tuple[ReadinessCorrectionLabel, ...]
    observed_grasp: tuple[bool, ...]

    def __post_init__(self):
        if type(self.consent) is not ProbeCorrectionConsent:
            raise ValueError("probe corrections require explicit new query consent")
        consent = ProbeCorrectionConsent(
            self.consent.max_queries, self.consent.role, self.consent.seed
        )
        flags = _observed_grasp_flags(self.episode)
        episode = copy_continuous_episode(self.episode)
        binding = pick_cube_probe_binding("probe_train")
        if (
            episode.role != binding.role
            or episode.context != binding.context
            or episode.episode_id != STUDENT_ID
            or episode.spec.observation_dim != 42
            or episode.spec.action_units
            != (
                "normalized_root_dx",
                "normalized_root_dy",
                "normalized_root_dz",
                "normalized_gripper",
            )
            or episode.spec.lower_bounds != (-1.0,) * 4
            or episode.spec.upper_bounds != (1.0,) * 4
            or not (episode.transitions[-1].terminated or episode.transitions[-1].truncated)
        ):
            raise ValueError("probe correction batch requires complete executed raw42student23")
        if (
            type(self.labels) is not tuple
            or not 1 <= len(self.labels) <= consent.max_queries
            or len(self.labels) != len(episode.transitions)
            or type(self.observed_grasp) is not tuple
            or len(self.observed_grasp) != len(self.labels)
            or self.observed_grasp != flags
        ):
            raise ValueError("one bounded observed-grasp label per executed student transition")
        owned = []
        for i, (label, row, flag) in enumerate(
            zip(self.labels, episode.transitions, self.observed_grasp, strict=True)
        ):
            if (
                type(label) is not ReadinessCorrectionLabel
                or type(flag) is not bool
                or label.step != i
                or label.observation != row.observation
                or type(row.observation[18]) is not float
                or row.observation[18] not in (0.0, 1.0)
                or flag is not (row.observation[18] == 1.0)
            ):
                raise ValueError("probe correction label/feedback differs from executed state")
            owned.append(ReadinessCorrectionLabel(i, label.observation, label.target, label.stage))
        object.__setattr__(self, "consent", consent)
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "labels", tuple(owned))
        # Private native references are not serialized or caller-provided proof.
        # Every field below is immutable; equal replacement also refuses release.
        object.__setattr__(self, "_paid_owners", self._owners())

    def _owners(self):
        consent, episode = self.consent, self.episode
        spec = episode.spec
        values: tuple[object, ...] = (
            consent,
            consent.max_queries,
            consent.role,
            consent.seed,
            episode,
            episode.episode_id,
            episode.context,
            episode.role,
            spec,
            spec.observation_dim,
            spec.action_units,
            spec.lower_bounds,
            spec.upper_bounds,
            episode.transitions,
            self.labels,
            self.observed_grasp,
        )
        for row in episode.transitions:
            values += (
                row,
                row.step,
                row.observation,
                row.action,
                row.reward,
                row.next_observation,
                row.terminated,
                row.truncated,
            )
        for label in self.labels:
            values += (label, label.step, label.observation, label.target, label.stage)
        return values

    def _require_original(self):
        if any(
            a is not b for a, b in zip(self._owners(), getattr(self, "_paid_owners"), strict=True)
        ):
            raise ValueError("original paid probe correction evidence changed")

    def training_pairs(self, *, now: int):
        require_tick(now, "probe correction release")
        self._require_original()
        owned = ProbeCorrectionBatch(self.consent, self.episode, self.labels, self.observed_grasp)
        if any(label.labels_arrived_at > now for label in owned.labels):
            raise ValueError("probe corrective labels have not all arrived")
        # Targets never replace the recorded actions that build history.
        model = _action_history_episode(owned.episode, PickCubeTaskSpaceHistoryRecipe())
        return (
            tuple(row.observation for row in model.transitions),
            tuple(label.target for label in owned.labels),
        )

    def model_episode(self):
        self._require_original()
        owned = ProbeCorrectionBatch(self.consent, self.episode, self.labels, self.observed_grasp)
        return _action_history_episode(owned.episode, PickCubeTaskSpaceHistoryRecipe())
