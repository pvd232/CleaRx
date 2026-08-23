# CleaRx authorized execution checklist

This file is the execution authority for the owner-approved path through the
Phase 1 readiness checkpoint. The
[orchestrator handoff](LOCAL_ORCHESTRATOR_MASTER_PROMPT.md) owns the technical
architecture and acceptance semantics. `orchestration/plan.yaml` owns packet
identity, dependencies, and lifecycle state after Bootstrap-001 creates it.

## Terminal outcome

The authorized run ends at this observable path:

```text
validated Bootstrap package
-> complete Phase 0 evidence and frozen abi_v0
-> validated Phase 1 packet DAG and common native test skeleton
-> fixture-only ABI smoke and synthetic loader-rejection tests pass
-> owner review before production module work or parallel dispatch
```

## Checklist semantics

A checkbox closes only when the named artifact exists at the recorded Git
commit and its stated command or external inspection passes. Packet completion
also requires a schema-valid result manifest bound to the packet digest and Git
commit. Complete dependency producers before their consumers.

## Contract and deliverable coverage

| Work unit | Current state | Owning phase | Completion evidence |
|---|---|---|---|
| Local project environment | Verified locally | Environment foundation | `conda run -n clearx` resolves the declared Python and tools |
| Colab control-plane access | Verified with ADC | Environment foundation | `colab --auth=adc sessions` exits successfully |
| Clean L4 deployment host | Verified live | Infrastructure foundation | GCE instance, disk, GPU, SSH, and backup-image inspections |
| Schemas, validators, state machine, and packet DAG | Complete at `4541b7f`; first-wave and derived validator ownership corrected at `3a4b079` and `c55f3a5` | Bootstrap-001 through Bootstrap-003 | Valid results `20260823T081011Z-local`, `20260823T082530Z-local`, and `20260823T085305Z-local` |
| Pinned upstream/runtime contract | Complete at `902d871` | P0-001 | Valid result `20260823T083005Z-local` |
| Tensor inventory | Complete at `0d4ae2b` | P0-002 | Valid result `20260823T091355Z-mixed`; 28,010 source tensors and 15 shards reconcile |
| Fixture contract and generated reference fixtures | Not started | P0-003A and P0-003B | Fixture validator and packet-bound A100 evidence |
| Native ABI and ownership contract | Not started | P0-004 | Phase 0 integration gate |
| L4 host and memory contract | Host profile complete at `502d9eb`; memory budget pending | P0-005A and P0-005B | Valid P0-005A result `20260823T083617Z-l4` and later manifest-derived budget |
| Expert-cache and allocator risk contract | Benchmark contract complete at `f693c04`; manifest-derived spike pending | P0-006A and P0-006B | Valid P0-006A result `20260823T083903Z-local` and later spike artifacts |
| Evaluation and deployment acceptance contract | Evaluation contract complete at `31e5df6`; certified-host binding complete at `754f95e` | P0-007A and P0-007B | Valid results `20260823T084617Z-local` and `20260823T085543Z-local` |
| Phase 1 work decomposition | Not started | Phase 1 readiness | Validated packet DAG with disjoint write scopes |
| Common native build/test skeleton | Not started | Phase 1 readiness | Local configure, build, and test gate |

## Verified baseline

- [x] Commit `8cb51db` contains the audited handoff, TeX proposal, and rendered
  PDF on `origin/main`.
- [x] `environment.yml` declares the `clearx` environment; the local environment
  resolves Python 3.12, CMake, Ninja, NumPy, PyYAML, jsonschema, pytest,
  Hugging Face Hub, safetensors, and Google Colab CLI 0.6.0.
- [x] Colab CLI authentication through existing Application Default Credentials
  lists sessions without an authentication error.
- [x] `mantra-backup-blueprint` remains `READY` with the preserved 500 GiB MANTRA
  disk image.
- [x] `mantra-g2-spot` is a clean `g2-standard-12` Spot VM in `us-west1-a` with
  one NVIDIA L4, a 375 GiB `pd-ssd` boot disk, private ingress through IAP, and
  outbound access through `mantra-nat-us-west1`.
- [x] The stable `mantra-g2` SSH alias connects to the replacement VM.

## Phase A. Bootstrap-001 orchestration package

**Depends on:** verified baseline

**Owned contract:** [Bootstrap-001](LOCAL_ORCHESTRATOR_MASTER_PROMPT.md#bootstrap-001--create-the-orchestration-package)

- [x] Create root operating rules, decisions, and execution-environment documentation.
- [x] Create plan, packet, result, fixture, benchmark, and state schemas.
- [x] Create `plan.yaml`, immutable Bootstrap/Phase 0 packets, and initialized state.
- [x] Implement plan, result, fixture, packet-run, and remote-wrapper entry points.
- [x] Add valid and invalid orchestration fixtures and tests for every required rejection.
- [x] Run the complete Bootstrap validation command from the `clearx` environment.
- [x] Review the package against the handoff and commit the validated checkpoint.
- [x] Correct the four missing Phase 0 validator entry points through
  Bootstrap-002 without changing the immutable Phase 0 packet bytes.
- [x] Audit every derived Phase 0 acceptance command and supply the four
  remaining validators through Bootstrap-003, including success and severed-join tests.

**Acceptance gate**

```bash
conda run -n clearx python scripts/validate_bootstrap.py
```

**Commit boundary:** a complete, locally validated orchestration package with
no production runtime implementation.

## Phase B. Phase 0 first wave

**Depends on:** Phase A

- [x] Complete P0-001 and freeze exact upstream/runtime revisions.
- [x] Complete P0-002 and publish the complete tensor inventory.
- [ ] Complete P0-003A and validate the fixture contract and harness.
- [x] Complete P0-005A using the live `mantra-g2` host profile.
- [x] Complete P0-006A and validate the benchmark/result contracts.
- [x] Complete P0-007A and freeze the draft corpus, metrics, and margins.

**Acceptance gate:** every first-wave result validates against its immutable
packet and all six packet states are `complete`.

**Commit boundary:** validated first-wave contracts and evidence.

## Phase C. Phase 0 derived evidence and ABI freeze

**Depends on:** Phase B

- [ ] Complete P0-003B from a clean A100 run and download fixture evidence.
- [ ] Complete P0-005B from the tensor manifest and certified L4 profile.
- [x] Complete P0-007B and bind every later threshold to the certified host.
- [ ] Complete P0-006B from the manifest, memory budget, and benchmark design.
- [ ] Complete P0-004 last and freeze `abi_v0` after all upstream evidence validates.
- [ ] Run the complete Phase 0 integration gate.

**Acceptance gate:** P0-004 and every dependency are `complete`; packet/result
digests, artifact hashes, schemas, and the full repository test command pass.

**Commit boundary:** frozen Phase 0 evidence, evaluation contract, and `abi_v0`.

## Phase D. Authorized Phase 1 readiness checkpoint

**Depends on:** Phase C

- [ ] Materialize six Phase 1 packets for conversion/loader, audio-only AuT,
  Talker, MTP, Code2Wav, and scheduler work.
- [ ] Validate packet dependencies and disjoint write scopes.
- [ ] Create the shared CMake library/test skeleton without model operators.
- [ ] Compile and run one fixture-only executable across the frozen ABI types.
- [ ] Add and pass synthetic tests that reject missing or incompatible GGUF metadata.
- [ ] Record the proposed Phase 1 worker DAG and stop before dispatch.

**Acceptance gate**

```bash
conda run -n clearx python scripts/validate_plan.py
cmake -S . -B build/local -G Ninja
cmake --build build/local
ctest --test-dir build/local --output-on-failure
```

**Commit boundary:** validated Phase 1 DAG and common readiness skeleton.

## Owner actions

| Owner input | First consumer | Result unlocked |
|---|---|---|
| Already granted: execute through Phase 1 readiness | Phase A | Serial Bootstrap, Phase 0, and readiness work |
| Already granted: replace unchanged MANTRA disk from backup | Infrastructure foundation | Clean CleaRx L4 host |
| Next review: approve production Phase 1 work and any parallel dispatch | After Phase D | Talker, MTP, Code2Wav, AuT, loader, and scheduler implementation |
| Later: approve Phase 2 and performance work | After Phase 1 correctness gate | End-to-end streaming and L4 optimization |

## Deferred work

| Item | Concrete value | Scope basis |
|---|---|---|
| Production Phase 1 module implementations | Native model correctness | Requires owner review after the readiness checkpoint |
| Parallel worker dispatch | Shorter elapsed implementation time | Explicitly outside current authorization |
| Phase 2 waveform integration | End-to-end native conversation | Depends on the complete Phase 1 correctness gate |
| Phase 3 L4 optimization | Memory fit and real-time operation | Depends on numerically correct native execution |
| Phase 4 bounded context research | Long-session memory control | Depends on stable native streaming and instrumentation |
