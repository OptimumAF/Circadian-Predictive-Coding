# Phase 6 Torch route inventory

P6.1e is a bounded functionality and artifact gate. A saved result proves its
own historical run, while this inventory names which producer can be exercised
on the current CPU host and which branches have a separate frozen CUDA gate.
It makes no method ranking. The current local runtime is Torch `2.14.0+cpu`,
torchvision `0.29.0+cpu`, no CUDA device, and no `CUBLAS_WORKSPACE_CONFIG`.

**Why these classifications:** The representative scripts preserve a verified
CUDA package/device/source/quiet-window protocol. Substituting CPU or synthetic
data under that protocol ID would change the study, so CPU coverage uses its
preparation, restore, audit, and explicit stub tests instead.

| Public route | CPU/stub or saved-output check | Execution boundary and output family |
|---|---|---|
| `resnet50_benchmark.py` synthetic unmatched reference | P6.1e1's real four-example forced/disabled CPU process pair. | Text, result JSON, resolved config; no weights or data download. |
| `src.app.matched_head_benchmark` fixed-feature/capacity/checkpoint APIs | P6.1e2's real bounded CPU calls and trusted local checkpoint. | Complete dataclass reports, typed sleep events, CPU RSS, and an unscored development-role checkpoint. CUDA allocator fields stay empty. |
| `scripts.run_repeated_confirmation_smoke` | P6.1e3's real fixed tiny CPU process and injected failure checks. | Selection, manifest, three-scope result, and stage/error failure JSON. The selection seed, three confirmation seeds, candidate grid, and budgets are unchanged. |
| `scripts.profile_cifar_representative_feasibility` `prepare` and `scripts.prepare_cifar_representative_study` | The saved probe is verified; a fresh CPU-safe request reproduced the frozen request byte for byte at SHA-256 `baa4bcaf5138946d01ad0d7c1b8babc21350ae18e53e24bc4e37a9cedf3dc773`. Focused feasibility/study tests passed. | `run`/`worker` require Torch `2.14.0+cu130`, torchvision `0.29.0+cu130`, deterministic CUDA/CUBLAS, verified 170,498,071-byte CIFAR archive and 102,540,417-byte ImageNet V2 weight, and a quiet GPU. They are explicit CPU skips, not CPU result coverage. |
| `scripts.run_cifar_representative_selection` | Six saved outer-validation trials, manifest and attempt journal restored by the public `--preflight` command with zero final-test iterations. Focused selection/restore tests passed. | New selection requires the same frozen CUDA pair, device/workspace setting, source hashes and quiet window. No new grid or seed was run. |
| `scripts.run_cifar_representative_confirmation` and `scripts.audit_cifar_representative_confirmation` | Read-only audit reconstructed nine fixed-data confirmations, three wall-time reports and three isolated-memory reports from the saved scope files. All saved scientific fields match the aggregate result. | New confirmation restores exact selection bytes before any source, then requires CUDA and a quiet device. The saved aggregate predates two empty report fields (`sleep_events`, `cuda_allocator_segments`) in each of nine wall-time head reports; the current audit adds only those 18 empty defaults. No frozen confirmation was rerun. |
| `scripts.profile_cifar_feature_setup` | P6.1e5a's five stub writer tests read finite request/result/failure and stdout using a tiny local weight and a sealed two-example feature bank; all three occupied paths refuse before weight access. E5b1 then found and corrected an earlier final-source construction gap with a raising source sentinel. The saved real profile has train/guard/validation counts 128/64/64 and zero final-test iterations. | The actual CPU route needs the verified local CIFAR archive and cached ImageNet V2 weights. The stub verifies the writer contract without loading either large source or downloading weights. Its historical saved profile was not rerun under the corrected source boundary. |
| `scripts.run_cifar_pretrained_validation`, `scripts.run_cifar_pretrained_confirmation` | E5b1–b2's public validation writer omits final-source construction and passes on verified tiny archive/weight and development-feature stubs: finite request, six complete validation trials, restorable manifest, failure ledger, stdout, and all occupied-path preflights. E5b3 checks the public confirmation writer's result/failure paths against the frozen manifest using clearly synthetic finite rows, plus tampered archive, weight, selection, manifest, and occupied-output rejection before final access. Real saved archive/weight/manifest/result identities passed read-only; the historical result has nine fixed-data, three wall-time, and three memory rows. | Original paths are occupied and the actual pretrained confirmation has a larger fixed budget. The stub checks writer/provenance behavior and makes no new model-performance claim; historical files were not rerun. |
| `scripts.run_cifar_matched_validation` and `scripts.run_cifar_matched_confirmation` | E5b1–b2's public local-CIFAR validation writer omits final-source construction and passes on a verified tiny archive and development-feature stubs: finite request, six complete validation trials, restorable manifest, new failure ledger, stdout, and four occupied-path preflights. E5b3 checks the public confirmation result/failure writer using synthetic manifest-consistent rows, plus tampered archive, selection, manifest, and occupied-output rejection before final access. Real saved archive/manifest/result identities passed read-only; the historical result has nine/three/three rows. | Original paths are occupied and actual confirmation has a fixed larger budget. No full-data study was rerun and synthetic writer rows are not science results. |
| `scripts.run_cifar_pretrained_cuda_validation`, `scripts.run_cifar_pretrained_cuda_confirmation`, `scripts.verify_cuda_vision_order` | Existing historical CUDA artifacts remain read-only. | Require the verified CUDA package pair and actual CUDA device; validation/confirmation also pin CUBLAS, archive, weights, seeds and budgets. CPU execution is an explicit skip. |

The five representative feasibility/study/selection/restore/confirmation test
modules passed 21 CPU-safe stub tests. `restore_cifar_representative_selection
--preflight` and the saved-scope audit passed separately without opening a
training or final dataset. The verified local archive and weight cache are
present, so CUDA runtime and quiet-device requirements are the current
representative execution boundary; source availability is not the skip reason.
Torch replay remains unsupported and is not treated as parity coverage.
