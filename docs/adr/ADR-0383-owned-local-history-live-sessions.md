# ADR-0383: owned local public-history live sessions

## Context

Matched history learners/probes work but allE endpoints predictmajorityaction.
Teacherforcedfit doesnot establish navigation. Existing live runner consumesraw8
whilematched learnersrequire16; existing RNNsessions use a differentcontract.

## Decision

Add prediction-only adapter owning fullnativefork+onepublicinitialframe, bounded
12calls/counters. Existing coreencoder/probabilityvalidator, inputcurrentpublic
frame only, no context/cuemetadata/targets/oracle/mask. Modelparams/gradients/
optimizer/counters frozen. Freshsession resets beforeactions; failurestops and
native returns counted beforepostguardfailure. Check finalparentstate after
childterminalreadback, closing reproduced constructionmutationwindow.

Compose existing _live/run_key_door_episode viaoptin history andownedbudget
observationcallbacks, oneprivateTRAINepisode percall. Publishonlycurrentselected
slot, sealprior/unreached evidence; preserveoriginalbudgetbindings andonlyactual
observed clockadvance. Samefourlearners/configs/fixedknownlayouts/checkpoints,
rawargmax, no repairs. Default8/RNN/TRAINonly/E16 paths unchanged.

## Alternatives

New rolloutframework duplicates physics/errorhandling. Cuemetadata oraction
masking changes information/policy measured. Reusedmutablehistory leaks cues;
parameter-onlycopies lose fullstate provenance. LongerEprobes cannot establish
actualactionskill. Keep existing guardedstream; avoid unrelatedrefactors.

## Consequences

Liveactions/returns/costs observable, withboundedmemory butadditionalcopies/
construction/guards/encoding in time/RSS. KnownTRAINresubstitution is not unseen
generalization; noinitialskill means no usefulretention/recoveryclaim. No final/
probe/replay/promotion/CPCadvantage. Evidence: artifacts/runs/matched-history-live-20261017/.
