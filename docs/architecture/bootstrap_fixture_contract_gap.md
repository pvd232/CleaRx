# Bootstrap-004 fixture evidence correction

## Status

The current fixture envelope and artifact hash check are inspected and
implemented. The strict descriptor, provenance joins, frozen tolerance
profiles, harness invocation, and rejection tests in this correction are
proposed until Bootstrap-004 reaches `complete`.

## Required claim

`python scripts/validate_fixtures.py --contract-only` verifies that every
contract fixture names one observable boundary, one frozen comparison profile,
the exact producer inputs and code, and output metadata identical to the bytes
declared by its envelope. When P0-003A supplies `fixture_harness.py`, the same
command also requires that harness to validate the complete boundary and
adversarial-case matrices.

## Current gap

The frozen [`fixture.schema.json`](../../orchestration/schemas/fixture.schema.json)
accepts `producer_commit`, `seed`, an unconstrained `sampling` object, and
artifact path/hash/dtype/shape. Those fields establish file identity alone. The
acceptance path drops boundary ID, numerical profile, source model identity,
and upstream/tensor/generator/schema digests before validation.
The previous [`validate_fixtures.py`](../../scripts/validate_fixtures.py) only
recomputed artifact hashes. The acceptance command could therefore pass while
all four P0-003A required checks remained unsupported.

The first missing connector was the relationship between an artifact hash and
the operation whose output those bytes represent. The old envelope established
file identity, then lost boundary identity, provenance, and comparison policy
before acceptance.

## Contract models

The existing fixture manifest remains the byte envelope. One artifact with
dtype `application/vnd.clearx.fixture-descriptor+json` identifies the strict
descriptor.

[`fixture_descriptor.schema.json`](../../orchestration/schemas/fixture_descriptor.schema.json)
defines these authorities:

- `boundary` names the module, operation, and input/output coordinate domains;
- `case` classifies a nominal or tagged adversarial execution;
- `provenance` binds `upstream.lock`, `tensor_manifest.json`, the evaluation
  contract, descriptor schema, generator path/hash/commit, and source model;
- `execution` fixes seed, deterministic mode, sampling, framework, and device;
- `comparison` selects one exact numeric profile from `PROFILE_DEFINITIONS`;
- `inputs` and `outputs` identify every consumed and produced artifact.

The five CleaRx profiles are local engineering policy:

| Profile | Comparison | Absolute tolerance | Relative tolerance | Minimum cosine |
|---|---|---:|---:|---:|
| `exact_discrete` | Exact | 0 | 0 | 1.0 |
| `fp32_strict` | All-close | $10^{-6}$ | $10^{-5}$ | 0.999999 |
| `bf16_activation` | All-close after FP32 promotion | 0.03125 | 0.03125 | 0.999 |
| `bf16_accumulation` | All-close after FP32 promotion | 0.0625 | 0.05 | 0.995 |
| `waveform_fp32` | All-close | 0.001 | 0.001 | 0.9999 |

Every profile requires zero mismatched elements after its absolute/relative
test. The same output must also meet the profile's cosine threshold. P0-003A
assigns one profile to each boundary. A later versioned profile change applies
to new fixtures; completed fixtures retain their original comparison object.

## Execution

The fixture generator writes output files and a descriptor. The envelope writer
hashes that descriptor and every output. `validate_fixture_envelope()` then
executes this order:

```text
fixture manifest
  -> validate fixture.schema.json
  -> recompute every envelope artifact hash
  -> locate exactly one strict descriptor
  -> validate fixture_descriptor.schema.json
  -> recompute repository provenance hashes
  -> compare the named profile with PROFILE_DEFINITIONS
  -> join descriptor ID, producer commit, seed, and sampling to the envelope
  -> compare descriptor input/output metadata with envelope artifact metadata
```

For `model_boundary` fixtures, the validator also reads
`generator_commit:generator_path` from Git and hashes those committed bytes.
`contract_smoke` fixtures establish only schema and join behavior.

## Persisted evidence

The envelope persists the descriptor hash and all output hashes. The descriptor
persists the boundary, case, provenance, execution, comparison, and artifact
references. P0-003B results therefore join generated arrays back to the pinned
model, tensor inventory, generator commit, and exact schema bytes through
hashes; filenames remain locators.

## Verification

Stable checks are:

- `strict_descriptor_present`: exactly one descriptor artifact exists;
- `provenance_hashes_recomputed`: every repository digest matches current bytes;
- `generator_commit_bound`: model fixtures use generator bytes stored at the
  recorded commit;
- `tolerance_profile_frozen`: every comparison object equals its named profile;
- `execution_joined`: ID, producer commit, seed, and sampling match the envelope;
- `artifact_metadata_joined`: descriptor and envelope input/output
  path/hash/dtype/shape mappings are equal;
- `p0_harness_invoked`: contract-only acceptance propagates the P0 harness exit
  status.

## Propagation

| Surface | Required change |
|---|---|
| Type | Add strict descriptor schema; retain the existing envelope schema. |
| Authoring | P0-003A defines boundary/profile assignments and adversarial coverage. |
| Runtime | P0-003B generator writes descriptors beside model-derived outputs. |
| Persistence | Envelope hashes descriptor and output bytes. |
| Verification | `fixture_contract.py` recomputes provenance and all cross-record joins. |
| Test | Success plus missing-provenance, tolerance-drift, output-join, sampling-join, and harness-failure cases. |
| Documentation | P0-003A explains each boundary, profile, and next consumer. |

## Acceptance case

The success case is `bootstrap-smoke`: its envelope hashes one strict descriptor
and one 25-byte UTF-8 payload. The descriptor binds the current upstream lock,
tensor manifest, evaluation contract, schema, source model, deterministic
execution, and `exact_discrete` profile.

The targeted rejection changes `bf16_activation.atol` while retaining the same
profile ID and rebinds the descriptor hash in the envelope. The schema still
accepts the number; `tolerance_profile_frozen` rejects the semantic drift.

## Implementation order

1. Commit Bootstrap-004 with the strict schema, validator, smoke descriptor,
   propagation tests, plan node, and running lifecycle state.
2. Run the fixture tests and complete Bootstrap validation; record and push the
   Bootstrap-004 result.
3. Resume P0-003A, implement its boundary matrix and local comparison harness,
   and require `--contract-only` to invoke that harness.
4. Generate P0-003B model fixtures only from a clean, packet-bound A100 commit.
