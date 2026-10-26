# Public episode-step inputs

`src/core/continuous_episode_clock.py` adds one bounded public time coordinate
to continuous observations. `EpisodeClock(horizon, elapsed=True)` appends
`step / horizon`; `elapsed=False` appends zero as an equal-width control. Native
steps range from zero through the declared horizon, including a terminal next
observation. Horizons are exact integers in 1..256. The raw observation prefix,
action units and bounds are preserved; an additional coordinate must fit within
the existing 1024-feature limit.

The module is pure: inputs are owned finite native tuples and validated action
specs/episodes; outputs are owned tuples/specs/episodes. It performs no simulator,
learner, persistence or data-release work. `clock_training_episode` requires a
complete TRAIN episode before projecting current and next observations; it
preserves identity, context, labels, rewards and terminal/truncation flags.
Probe and final episodes are refused. No future reward or teacher geometry is
used to produce the feature.

## Composition with the existing learner

```python
from src.core.continuous_episode_clock import EpisodeClock, clock_training_episode
from src.adapters.numpy_continuous_bc import (
    ContinuousBCPolicy, NumpyContinuousBCLearner, fit_training_standardization,
)

recipe = EpisodeClock(50, True)
# native_episode must be a complete, released TRAIN demonstration.
derived = clock_training_episode(native_episode, recipe)
statistics = fit_training_standardization(derived)
model = NumpyContinuousBCLearner(ContinuousBCPolicy(
    derived.spec, observation_standardization=statistics,
))
diagnostic = model.train_episode(derived)
```

For an individually budgeted native TRAIN case, the existing smoke accepts
`main(learner=model, observation_clock=recipe)` with its normal `--output`
argument. Its horizon must equal the existing50-step rollout horizon. The smoke
owns and guards a copy of the recipe, encodes the current public loop step,
requires the encoded spec to match before prediction, and preserves full private
and original learner state through control and cleanup. Raw native observations
remain in the exported episode; bounded clock values, encoding attempts/returns
and the count of all encoded coordinates are separate receipt fields.

**The caller must retain and pair the exact feature recipe with the model.**
The learner policy schema is unchanged: width alone cannot distinguish zero
from elapsed inputs. There is no implicit recipe inference or new disk codec.
The optional frozen normalizer remains full learner state. Default unclocked
and zero-action smoke behavior remain available.

Why this: the current teacher stages actions by public episode step. This tests
an observable representation hypothesis with an equal-width control; uniqueness
of all29 original observations means it is not a proof of missing information.

## Verification and evidence

475 combined focused cases pass, including independent clock arithmetic,
terminal seams, role/ownership refusal and actual clock-dependent normalized
actions across a fake native boundary. Scoped lint/format and two-source type
checks pass. No real simulator execution is implied by these tests.

The prospectively fixed comparison, commands, costs, all curves and limitations
are recorded in [the local study](../artifacts/runs/continuous-bc-clock-20261019/PROTOCOL.md).
Better fitting on one TRAIN demonstration cannot establish robot success,
generalization, retention/recovery or a circadian advantage.

## Executed fixed native comparison

The separately frozen two-policy TRAIN seed17 protocol has now executed once:
zero then elapsed, each fixed final256 policy with its recorded recipe/statistics.
Both complete50steps, return0, any_success=false, final_success=false and final
truncation. Full model/config state and the identical teacher initial42D state
are exact. The runtime input is43D; the exported native trace remains42D.
No optimizer update or refit occurs. The offline improvement did not produce
sparse task success in this fixed comparison.

The stage native_case.py accepts explicit `--method zero` or `--method elapsed`
and `--output`; it restores the named final state and recipe before calling the
existing smoke. native_guard.py forwards that method and separates evidence/cache
for each case while reusing the accepted Linux resource guard. Existing evidence
is exclusive and commands targeting it refuse overwrite.59new synthetic boundary
cases plus50clock cases pass in WSL; Windows50clock cases pass with the native
module skipped before its Linux-only import. Scoped statics and independent
source/actual review pass. Prior425 cases remain accepted unchanged.

Per case50clock encodings2150coordinates,50normalizer rows,50predictions1600raw
output bytes and50controls250known physics; reset physics remains unknown.
Lifetime process-tree memory is not established by the measured sampled/final
peaks. No broader robot skill, generalization/adaptation or CPC claim follows.
[Protocol, exact commands, costs and independent acceptance](../artifacts/runs/continuous-bc-clock-20261019/NATIVE_RESULTS.md).
