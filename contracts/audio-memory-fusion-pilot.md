# Audio Memory Fusion Pilot

## 1. Status

**Contract status:** Approved for one diagnostic A100 smoke test. The smoke
tests whether the pinned Thinker accepts equal-budget transcript-only and
audio--text memory positions. Its evidence covers this one execution and leaves
broader quality comparisons to a later pilot.

<!-- contract-protocol:generated:start -->
<!-- contract-protocol:generated:end -->

## 2. Claim

Given the frozen `fsdd-george-0` waveform, transcript, prompt, model revisions,
and source commit, the smoke runner produces three comparable final-position
logit vectors: full audio, transcript-only memory, and VoxZip-style addition.

For transcript token position \(j\), the compressed memory vector is

\[
m_j(\alpha) = t_j + \alpha a_j,
\]

where \(t_j\) is the Thinker text embedding and \(a_j\) is the mean of the
chronological Audio Tower vectors assigned to token \(j\). The smoke fixes
\(\alpha=0\) for transcript-only memory and \(\alpha=1\) for VoxZip addition.

## 3. Models

| Model | Role |
|---|---|
| `SmokeIdentity` | Names the exact corpus item, transcript, prompt, audio digest, coefficient set, and pinned revisions. |
| `SmokeResult` JSON object | Records the observed A100 environment, position counts, output hashes, top tokens, divergence metrics, and timings. |

The worker reads the frozen identity, downloads and hashes the waveform, and
writes the result to standard output. The local validator reloads the JSON and
rejects any identity, budget, shape, hash, or finite-number violation.

## 4. Execution

1. The bootstrap entrypoint installs the pinned Transformers revision, clones
   the requested CleaRx commit, and starts the worker from that checkout.
2. The worker loads Qwen3-Omni, disables the Talker, and prepares the frozen
   waveform and prompt.
3. The Thinker produces the full-audio teacher logits and Audio Tower vectors.
4. The worker replaces the complete audio-token span with the frozen transcript
   tokens, mean-pools the Audio Tower sequence to that token count, and runs
   \(\alpha=0\) and \(\alpha=1\).
5. The worker prints one canonical result object. The controller retains that
   object and reruns the local validator.

## 5. Persisted evidence

| Path or record | Evidence retained |
|---|---|
| `contracts/audio-memory-fusion-pilot.toml` | Requirements, verifier ownership, and the diagnostic PairBlock. |
| `plans/audio-memory-fusion-pilot/AMF-PB-01/plan.toml` | Exact implementation baseline, file actions, and local gate. |
| `runs/audio-memory-fusion-smoke/<run-id>/result.json` | Source, input, model, environment, condition, hash, and comparison observations from one A100 run. |

Large weights, logits, audio bytes, credentials, and Colab session state remain
outside Git.

## 6. Verification

| Rule | Executable condition |
|---|---|
| <nobr><code>AMF-VR-01</code></nobr> | Unit tests accept only the frozen input and revision identities. |
| <nobr><code>AMF-VR-02</code></nobr> | Unit tests establish ordered mean pooling and the two declared fusion equations at one shared memory budget. |
| <nobr><code>AMF-VR-03</code></nobr> | The local validator rejects missing provenance, unequal compressed budgets, non-finite comparisons, and malformed hashes. |

## 7. Propagation

| Surface | Required change |
|---|---|
| Research tools | Add the frozen identity, pooling, fusion, and result validation functions. |
| A100 experiment | Add separate bootstrap and worker entrypoints so dependency installation precedes third-party imports. |
| Tests | Exercise the success cases and the smallest identity, budget, and finite-number counterexamples. |
| P0-003B | Retain the existing reference-fixture scope; the exploratory smoke uses separate records. |

## 8. Acceptance case

### Success

The A100 returns all three conditions. Both compressed conditions use the same
memory-position count, every final-position logit vector is finite, and the
local validator accepts the retained result.

### Rejection

The validator rejects a result when the source commit, input digest, revision,
compressed position count, memory-position count, output hash, or comparison
metric differs from the frozen contract or is missing.

## 9. Implementation order

1. Run the PairBlock gate against an isolated candidate.
2. Publish the immutable review commit.
3. Launch the bootstrap entrypoint from that clean commit on one A100.
4. Retain and validate the result before interpreting the comparison.

## 10. Contract-owned PairBlocks

`AMF-PB-01` owns the complete diagnostic path. Its source-backed plan is
[`plan.toml`](../plans/audio-memory-fusion-pilot/AMF-PB-01/plan.toml).

## 11. ContractTarget

The plan adds complete files. The shared runner reconstructs the declared Git
baseline, copies only those additions, runs the ordered gate, and binds the
tested bytes to the review commit used by the A100.

## Future work

The coefficient sweep, multiple clips, Whisper alignment, learned gate, and
in-Thinker compression remain outside this smoke contract. A later additive
PairBlock may promote them only after this smoke result passes.
