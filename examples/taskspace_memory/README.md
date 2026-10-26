# Generated taskspace memory example

`episodes.json` contains numeric TRAIN observations/actions generated locally by
this project's controllers in simulated PickCube/PushCube tasks. It includes all
seven released packets (220 rows) and all four recorded trajectories (70 rows)
from the accepted complete-B22 case. No images, simulator assets or model weights
are included. This example data is provided under the repository's MIT license;
retain [LICENSE](../../LICENSE). Installing a simulator separately requires its
own dependencies and terms.

The source path/hash and earlier 198-row baseline scores are embedded as
informational provenance. The source research archive is not needed to run the
example. The failed Pick19 baseline and all original packets remain represented;
the complete original research evidence remains preserved separately.

```powershell
python -m scripts.run_taskspace_memory_demo --output replay.json
```

The default rebuilds the memory from TRAIN packets, then checks every recorded
selected index and native float32 command at the saved inputs. Recorded success
is historical metadata, not a new simulator result or a generalization claim.
Replay observations/actions never become training packets. B50 corrective labels
retain their original inputs/history; they are not treated as executed actions.

For setup, exact checks and the separate native fresh-run requirement, see
[the demo guide](../../docs/taskspace-memory-demo.md).
