# Certified L4 memory model

## Contract and evidence level

[`l4_memory_budget.json`](../../l4_memory_budget.json) joins the exact
[`tensor_manifest.json`](../../tensor_manifest.json) bytes to the exact
[`l4_host_profile.json`](../../tests/fixtures/manifests/l4_host_profile.json)
bytes. It derives every weight term from tensor shape and frozen target dtype;
it does not substitute parameter-count headlines for tensor accounting.

This is a Phase 0 planning model. `measured_bytes` is deliberately `null` for
every execution phase because no converted native loader or instrumented phase
runtime exists yet. The record names that missing measurement boundary as
`pending_native_loader_and_phase_instrumentation`. Phase 1 must record actual
GGUF sizes and allocator/CUDA peaks before any fit claim becomes measured.

## Unit and ceiling definitions

The certified L4 exposes 23,034 MiB, or 24,152,899,584 bytes. CleaRx applies a
lower operational peak limit of exactly 22 GiB (23,622,320,128 bytes). The
difference—530,579,456 bytes, or 506 MiB—is an untouchable reserve between the
operational limit and the device-reported hard ceiling.

Every range maximum is at or below 22 GiB. The worst modeled phase, barge-in,
has a 21.227 GiB range maximum and therefore retains about 0.773 GiB below the
operational limit in addition to the 506 MiB device reserve.

## Target weight calculation

The builder applies these row-aware storage formulas:

- BF16: two bytes per element, identical to the source tensor byte count;
- Q8_0: 34 bytes per 32-element block, padding each final-dimension row;
- Q4_K_M planning surrogate: 144 bytes per 256-element Q4_K block, padding
  each final-dimension row;
- removed vision tensors: zero deployment bytes.

The Q4_K_M value is a conservative Phase 0 storage surrogate, not a converted
file measurement. The converter may select mixed K-quant encodings and add
GGUF metadata/alignment; P0-005B does not pre-claim those target bytes.

The full audio-only target storage estimate is 20.493 GiB across GPU and CPU:

| Module | Estimated target bytes | Primary storage |
|---|---:|---|
| Thinker | 17,174,917,120 | L4 permanent |
| AuT | 731,568,256 | L4 listening arena |
| Talker, including all experts | 3,382,015,232 | Dense subset on L4; authoritative routed experts in pinned CPU RAM |
| MTP | 283,140,608 | L4 speaking phase |
| Code2Wav | 432,033,154 | CPU permanent |
| Vision | 0 | Removed |

The 128 Talker experts require an estimated 3,208,642,560-byte authoritative
CPU copy. The L4 never holds that complete set. It reserves a fixed
1,610,612,736-byte (1.5 GiB) expert slot pool and adds only 173,372,672 bytes of
dense Talker tensors. The maximum simultaneous L4 tensor capacity, during
barge-in when AuT and speaking tensors overlap, is 19,973,611,392 bytes
(18.602 GiB).

## Phase live sets

Each expected value is the sum of six named categories: weights, KV, workspace,
staging, runtime, and fragmentation. The min/max range widens non-weight terms
for allocator and kernel uncertainty while holding tensor-derived weights
fixed.

| Phase | Expected decimal GB | Expected GiB | Range GiB | Measurement |
|---|---:|---:|---:|---|
| Listening | 19.786 | 18.427 | 18.052–18.802 | Pending native instrumentation |
| VoxZip concurrent listening | 20.121 | 18.739 | 18.239–19.239 | Pending native instrumentation |
| VoxZip sequential endpoint | 19.853 | 18.489 | 18.114–18.864 | Pending native instrumentation |
| Thinker prefill | 19.926 | 18.558 | 18.058–19.058 | Pending native instrumentation |
| Talker startup | 21.188 | 19.733 | 19.233–20.233 | Pending native instrumentation |
| Talker sustained output | 21.322 | 19.858 | 19.358–20.358 | Pending native instrumentation |
| Barge-in | 22.255 | 20.727 | 20.227–21.227 | Pending native instrumentation |
| Context rebuild | 20.195 | 18.808 | 18.308–19.308 | Pending native instrumentation |

Barge-in is the modeled peak because it retains Thinker, the Talker dense set,
the 1.5 GiB expert pool, MTP, and AuT simultaneously. Code2Wav remains on CPU
and contributes zero L4 weight bytes. VoxZip terms currently occupy workspace
and staging allowances; they do not receive unverified learned-model weights.

## Replacement rule

The budget advances from modeled to measured one phase at a time. A native run
must record phase live sets, allocator peak, process RSS, exact source commit,
manifest hash, host-profile hash, GGUF hashes, and CUDA timestamps. A measured
peak outside the current range does not get normalized into the estimate: it
updates the model, records the discrepancy, and re-runs the 22 GiB gate. If any
range maximum plus the 506 MiB reserve exceeds the device ceiling, the packet
fails rather than borrowing from the reserve.

Rebuild and validate the record with:

```bash
python tools/inventory/build_memory_budget.py l4_memory_budget.json
python tools/inventory/validate_memory_budget.py l4_memory_budget.json
```
