# Expert-cache and shared-allocator benchmark contract

## What this benchmark can establish

This benchmark measures transfer time, cache hits, copy/compute overlap, and
allocator behavior, then rejects invalid event traces. The contract itself
contains no representative measurement or cache-policy verdict. P0-006B inserts
expert shapes from `tensor_manifest.json`, runs the cases on the certified L4,
and records whether the 1,610,612,736-byte expert-slot pool is feasible.

The machine-readable contract is
[`expert_cache_contract.json`](../../tests/fixtures/manifests/expert_cache_contract.json).
Its `claim_level` remains `design_only` until P0-006B produces a result bound
to the host-profile, tensor-manifest, trace, and source-commit hashes.

## Router trace

One JSON Lines record represents one Talker layer at one generated token
position:

```json
{
  "layer": 7,
  "token_position": 41,
  "selected_experts": [4, 9, 17, 28, 66, 103],
  "routing_weights": [0.29, 0.21, 0.17, 0.14, 0.11, 0.08],
  "timestamp_ns": 1787472000000000000
}
```

The six expert IDs and six weights preserve the same ordering. Expert IDs are
distinct, weights are finite and nonnegative, timestamps increase within a
request, and every `(layer, expert)` pair resolves to tensor byte ranges in
`tensor_manifest.json`. P0-006B adds `request_id`, `phase`, `source_commit`, and
`tensor_manifest_sha256` before a trace can support a runtime claim.

## Transfer cases

The two synthetic controls move exactly 8 MiB and 64 MiB from page-locked host
memory to the L4. They establish the measurement path and do not stand in for
an expert. The three manifest cases substitute the exact byte sum of one or
two Talker experts:

1. `manifest-expert-cold` measures allocation, asynchronous H2D copy, event
   completion, and the first compute use.
2. `manifest-expert-hit` starts with the same expert resident and permits no
   H2D copy.
3. `manifest-expert-evict-load` starts with a full pool, waits for the victim's
   lease count to reach zero, evicts it, loads the target, and then permits
   compute.

Each case uses five warm-ups followed by 30 recorded repetitions. A result row
is paired by host-profile hash, tensor-manifest hash, trace hash, case ID, and
repetition index. A Spot interruption invalidates the incomplete repetition;
the runner restarts that repetition after revalidating the host profile.

## Slot lookup and policy traces

The simulator keys one slot by `(layer, expert_id)`. A lookup returns
`resident`, `loading`, or `absent` plus the current lease count. `resident`
increments the lease and proceeds. `loading` waits on that slot's transfer
event. `absent` selects an unleased victim under the candidate policy, records
the eviction, and starts one transfer.

Four deterministic router patterns isolate distinct behavior:

- `single-layer-repeat` separates cold load from repeated lookup.
- `working-set-fits` retains an active set whose manifest bytes fit the pool.
- `working-set-thrash` exceeds the pool and forces eviction.
- `layer-interleaved` confirms that equal expert IDs in different layers do
  not alias the same slot.

P0-006B may compare LRU and another explicitly named policy with the identical
trace. It cannot select a production policy from synthetic controls alone.

## Shared allocator event model

The arena event stream uses six operations:

```text
allocate -> transfer_start -> transfer_end -> compute_start -> compute_end -> release
```

An allocation without transfer uses `allocate -> compute_start`; a staging
buffer may use transfer events without compute events. Every event records
`allocation_id`, owner, device, byte size, stream, request, and monotonic
timestamp. Owners are `aut`, `whisper`, `talker_expert`, and `staging`.

The validator rejects these event sequences:

- a missing or duplicate `allocate` or `release`;
- a read before allocation or after release;
- expert compute before its `transfer_end` event;
- eviction while the victim has a nonzero lease count;
- simultaneous live bytes above the configured arena or expert-pool capacity.

P0-006B runs three lifetime schedules: AuT without Whisper, concurrent AuT and
Whisper, and Talker expert loading after endpoint release. The schedules use
manifest-derived sizes and the certified 6,313,955,328-byte memlock ceiling.

## Overlap measurements

For each repetition, CUDA events delimit the H2D interval and the dependent
compute interval on their actual streams:

- `h2d_duration_ns` is transfer end minus transfer start.
- `compute_duration_ns` is compute end minus compute start.
- `overlap_duration_ns` is the intersection of those intervals.
- `unhidden_transfer_ns` is `h2d_duration_ns - overlap_duration_ns`.

The result also stores payload bytes, hit/miss counts, evictions, and allocator
peak bytes. Host wall-clock timestamps diagnose scheduling delay but do not
replace CUDA-event durations. A row passes measurement validity only when
payload hashes match, timestamps are monotonic, and every allocation is
released.

## Raw result fields

Each P0-006B row identifies `source_commit`, `host_profile_sha256`,
`tensor_manifest_sha256`, `trace_sha256`, case, warm-up or repetition index,
all raw duration fields, cache counts, peak bytes, and the validity verdict.
Later summaries calculate aggregates from those rows. Reviewers can recompute
any feasibility or policy conclusion from the stored measurements.
