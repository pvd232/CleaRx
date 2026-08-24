# Certified CleaRx L4 host profile

## Profile identity

The canonical machine-readable record is
[`tests/fixtures/manifests/l4_host_profile.json`](../../tests/fixtures/manifests/l4_host_profile.json).
The capture ran on `mantra-g2-spot` at `2026-08-23T08:34:03.350315Z` with
[`capture_l4_host.sh`](../../tools/benchmarks/capture_l4_host.sh) from commit
`dbab49a27f6ac4716d5948762f3714ec3ffb287a`.

| Component | Captured value |
|---|---|
| GCE placement | project `mantra-477901`, zone `us-west1-a`, `g2-standard-12`, Spot with `STOP` termination action |
| GPU | one NVIDIA L4, UUID `GPU-e2d6d9a2-3e87-2a3c-73fb-589e52f39d0e` |
| Visible device memory | 23,034 MiB (24,152,899,584 bytes) |
| CPU | 12 logical x86-64 CPUs; Intel Xeon 2.20 GHz; 6 cores, 2 threads per core, one NUMA node |
| Host memory | 50,511,646,720 bytes total; 49,428,234,240 bytes available at capture |
| Pinned-memory limit | 6,313,955,328 bytes soft and hard `RLIMIT_MEMLOCK` |
| PCIe | bus `0000:00:03.0`, x16 current/max width, generation 1 current and generation 3 maximum, NUMA node 0 |
| Root storage | 390,040,887,296 filesystem bytes; 371,945,881,600 bytes available; ext4 on NVMe-backed `pd-ssd` |
| Software | Ubuntu 22.04.5; kernel 6.8.0-1066-gcp; GCC 12.3.0; CUDA toolkit 12.9; `nvcc` 12.9.41; driver 580.173.02 |
| Power/thermal state | 72 W power limit, 44 °C, P8 idle state, persistence enabled, default compute mode |

The CUDA driver reports API capability 13.0 while the installed compilation
toolkit is CUDA 12.9. Later build manifests must record both values. The
toolkit executable is `/usr/local/cuda/bin/nvcc`; the default SSH `PATH` does
not include `/usr/local/cuda/bin`.

The PCIe generation value was captured at idle in P8. P0-006B must capture the
current link generation again during pinned-host transfer measurements; this
profile treats generation 3 and x16 as the advertised host-visible maxima, not
as measured sustained bandwidth.

## Phase measurement contract

Every L4 correctness or performance run emits the following phase transitions
with monotonic timestamps:

| Phase | Entry event | Exit event | Required live-set labels |
|---|---|---|---|
| `listening` | audio ingestion starts | endpoint or cancellation | AuT weights, AuT work buffers, Thinker input queue |
| `voxzip_concurrent_listening` | concurrent Whisper path starts | endpoint or cancellation | AuT set plus Whisper weights, KV, and work buffers |
| `voxzip_sequential_endpoint` | endpoint freezes input | compacted context is committed | AuT retained state, Whisper set, alignment buffers |
| `thinker_prefill` | bounded prompt submission | first Thinker decode token | Thinker weights, KV, prompt embeddings, scheduler buffers |
| `talker_startup` | first Talker request | first primary codec token | Talker permanent weights, active experts, Talker KV, staging buffers |
| `talker_sustained_output` | first codec frame | EOS, cancellation, or barge-in | Talker set, MTP set, Code2Wav state, PCM queue |
| `barge_in` | cancellation is accepted | output buffers and expert leases are released | cancellation queue plus unreleased buffers by owner |
| `context_rebuild` | rebuild snapshot is frozen | replacement context becomes active | old context, new context, compactor buffers, swap staging |

Each transition record contains `phase`, `event`, `timestamp_ns`,
`request_id`, and `source_commit`. A phase cannot report completion until all
buffers whose lifetime ends at that boundary have emitted a release event.

## Required metrics

`phase_live_sets` is the sum of allocated bytes grouped by phase, owner,
backend, dtype, and allocation class. Allocation classes are `weights`, `kv`,
`workspace`, `staging`, `runtime`, and `reserved`. The recorder emits bytes at
phase entry, phase exit, and every intervening allocation or release.

`allocator_peak` is the maximum device allocation observed inside one phase.
The native recorder combines GGML backend-buffer sizes with device free/total
samples and records both values rather than substituting one for the other.
CUDA free-memory deltas include non-GGML runtime allocations; GGML buffer sums
retain byte totals for each recorded owner.

`process_rss` is `VmRSS` from `/proc/self/status`, sampled every 100 ms and at
every phase transition. The same sample records `VmLck` so later runs can show
how much of the 6,313,955,328-byte memlock allowance is actually pinned.

## Certification and invalidation

The profile is valid only when all of these identities match the JSON record:
instance name and zone, machine type, GPU UUID, driver, CUDA toolkit, root-disk
type, CPU count, and memlock limit. A Spot restart on the same stopped instance
does not invalidate the profile if those fields still match. Instance
recreation, disk replacement, image change, driver change, or GPU UUID change
requires a new profile ID and capture.

The first clean instance start was preempted at `2026-08-23T08:01:31Z`; the
same stopped instance and boot disk restarted at `2026-08-23T08:16:56Z` before
this capture. Result manifests must therefore tolerate Spot interruption while
preserving commit and artifact identities.
