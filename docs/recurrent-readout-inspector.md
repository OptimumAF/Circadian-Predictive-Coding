# View sequential recurrent-readout results

From the physical checkout, export an existing saved result to a new HTML path:

```powershell
python -m scripts.build_key_door_inspector --source-artifact path/to/result.json --output path/to/new-view.html
```

This uses the shared MiniGrid inspector, not a new player. Open the exported
HTML in your own browser. It is standalone and uses recorded data only: no
model loading, training, prediction, game execution or network request.

Supported input is `protocol.definition=recurrent_readout_train_ABCA_native_v1`.
All 18 planned episode slots remain selectable across fits 2/4/6/8. The view
shows the current phase separately from the selected old or current task,
public egocentric frames, seven raw action probabilities, transitions, goals,
returns, and source JSON links. It does not invent a global world map, hidden
state or unrecorded decisions. Missing episodes remain unknown; known negative
results remain false/zero. Complete phases require complete immutable game
evidence and exact paid receipt prefixes. An interrupted phase may omit its
successful-phase receipt field; that absence remains unknown.

Current-task postfit competence, old-task retention, and A return are separate.
Recorded relearning requires acquisition, later loss and regain. In the accepted
fixed case all ten old-task checks fail, although the current tasks and A return
succeed. Repeating TRAIN A is not unseen-task transfer or retained old skill.
There is no prefit B/C game control to estimate their learning gain.

Readout fit charges and original TRAIN/readout clock lifetimes remain separate
from frozen native RNN SGD. The full fixed recipe uses 72 row exposures: 54
unique TRAIN rows and 18 repeated A rows. Partial runs do not inherit those
totals. App walltime and sampled RSS do not include outer observer costs; absent
observer counts stay unknown. No efficiency, matched-compute or CPC claim.

The new projection is in `src/app/recurrent_readout_inspector.py`, using the
existing shared dispatcher, JSON loader, public-frame validator and player.
Add another report format through a separate pure projection and explicit
dispatch; retain strict role, schema, unknown and source-pointer checks.

Focused checks: `python -m pytest -q tests/test_recurrent_readout_inspector.py`.
This is the ordinary contributor command, not authority to rerun a frozen
experiment or a spent session allowance. Tests use literal records only.

Implementation, actual export and scripted DOM behavior are verified. Genuine
browser/visual acceptance is still open under the existing file-URL access
restriction; scripted DOM checks do not satisfy it. This checkout feature is
not in the previously qualified #381 wheel. No new wheel was built.
