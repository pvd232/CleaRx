# Audio Memory Fusion Pilot

## 1. Status

**Contract status:** Approved for one diagnostic A100 smoke test. The smoke
tests whether the pinned Thinker accepts equal-budget transcript-only and
audio--text memory positions. Its evidence covers this one execution and leaves
broader quality comparisons to a later pilot.

<!-- contract-protocol:generated:start -->
**In progress.** [Jump to current PairBlock](#amf-pb-01)

**Checklist:** [Audio memory pilot](../checklists/audio-memory-pilot.md)

### PairBlocks

<a id="amf-pb-01"></a>

#### <nobr><code>AMF-PB-01</code></nobr>

**Status:** review

**Requirement contribution:** Add the frozen fusion functions, local rejection tests, A100 worker, bootstrap entrypoint, and result validator required for one auditable smoke run.

**Review handoff**

**What changed**

- Add the frozen one-case pooling and fusion operations with strict result validation.
- Add separate Colab bootstrap and A100 worker entrypoints so dependency installation precedes pinned model imports.
- Add local counterexample tests for identity drift, unequal compressed budgets, invalid vector shapes, and non-finite evidence.

**Plan deviations:** The implementation contains only the approved three-condition smoke scope. Coefficient sweeps, Whisper alignment, training, and in-Thinker compression remain future work.

**Start review:** [Open tested GitHub comparison](https://github.com/pvd232/CleaRx/compare/3c63f268c1317bf071456c1c814bebcaa0f21475...da6c799431411a8c48102503b00df26145af7dc1)

**Review these files**

- [Fusion and evidence contract](../plans/audio-memory-fusion-pilot/AMF-PB-01/add/tools/research/audio_memory_fusion.py#L1)
- [A100 smoke worker](../plans/audio-memory-fusion-pilot/AMF-PB-01/add/experiments/audio_memory_smoke_worker.py#L1)
- [Contract rejection tests](../plans/audio-memory-fusion-pilot/AMF-PB-01/add/tests/experiments/test_audio_memory_fusion.py#L1)

**Evidence:** [Passing gate receipt](../evidence/audio-memory-fusion-pilot/gates/amf-pb-01-20260916t070727z.json)

**Decision:** Approve <nobr><code>AMF-PB-01</code></nobr>, or return it with findings.

<details>
<summary>Implementation details</summary>

**Plan:** [plan.toml](../plans/audio-memory-fusion-pilot/AMF-PB-01/plan.toml)

**Candidate files:** [tools/research/audio_memory_fusion.py](../plans/audio-memory-fusion-pilot/AMF-PB-01/add/tools/research/audio_memory_fusion.py) · [experiments/audio_memory_smoke_bootstrap.py](../plans/audio-memory-fusion-pilot/AMF-PB-01/add/experiments/audio_memory_smoke_bootstrap.py) · [experiments/audio_memory_smoke_worker.py](../plans/audio-memory-fusion-pilot/AMF-PB-01/add/experiments/audio_memory_smoke_worker.py) · [scripts/validate_audio_memory_smoke.py](../plans/audio-memory-fusion-pilot/AMF-PB-01/add/scripts/validate_audio_memory_smoke.py) · [tests/experiments/test_audio_memory_fusion.py](../plans/audio-memory-fusion-pilot/AMF-PB-01/add/tests/experiments/test_audio_memory_fusion.py)

**Implementation roots:** [tools/research](../tools/research) · [experiments](../experiments) · [scripts](../scripts)

**Test roots:** [tests/experiments](../tests/experiments)

**Dependencies:** None

**Gate steps:**

```bash
# typecheck
(cd . && pyright --pythonpath /Users/machina/miniconda3/envs/clearx/bin/python tools/research/audio_memory_fusion.py experiments/audio_memory_smoke_bootstrap.py experiments/audio_memory_smoke_worker.py scripts/validate_audio_memory_smoke.py tests/experiments/test_audio_memory_fusion.py)
# test
(cd . && python -m pytest -q tests/experiments/test_audio_memory_fusion.py)
# documentation
(cd . && python /Users/machina/.agents/skills/code-documentation/scripts/check-schema-descriptions.py tools/research/audio_memory_fusion.py experiments/audio_memory_smoke_bootstrap.py experiments/audio_memory_smoke_worker.py scripts/validate_audio_memory_smoke.py tests/experiments/test_audio_memory_fusion.py)
# lint
(cd . && ruff format --check tools/research/audio_memory_fusion.py experiments/audio_memory_smoke_bootstrap.py experiments/audio_memory_smoke_worker.py scripts/validate_audio_memory_smoke.py tests/experiments/test_audio_memory_fusion.py)
# lint
(cd . && ruff check tools/research/audio_memory_fusion.py experiments/audio_memory_smoke_bootstrap.py experiments/audio_memory_smoke_worker.py scripts/validate_audio_memory_smoke.py tests/experiments/test_audio_memory_fusion.py)
```

</details>


### Requirements

| Requirement | Claim | Progress | Verifiers | PairBlocks |
|---|---|---|---|---|
| <nobr><code>AMF-REQ-01</code></nobr> | Freeze one short audio item, its transcript, prompt, model revisions, and the full-audio, transcript-only, and VoxZip-addition smoke conditions. | in_progress | <nobr><code>AMF-VR-01</code></nobr> | <nobr><code>AMF-PB-01</code></nobr> |
| <nobr><code>AMF-REQ-02</code></nobr> | Construct every compressed condition from the same chronological mean-pooled Audio Tower vectors and the same transcript-token positions. | in_progress | <nobr><code>AMF-VR-02</code></nobr> | <nobr><code>AMF-PB-01</code></nobr> |
| <nobr><code>AMF-REQ-03</code></nobr> | Retain one locally revalidated A100 smoke result that binds the source commit, input hash, model revisions, condition shapes, finite logits, and comparison metrics. | in_progress | <nobr><code>AMF-VR-03</code></nobr> | <nobr><code>AMF-PB-01</code></nobr> |

### Verification rules

| Rule | Requirements | Acceptance conditions | Success case | Rejection cases |
|---|---|---|---|---|
| <nobr><code>AMF-VR-01</code></nobr> | <nobr><code>AMF-REQ-01</code></nobr> | The smoke contract selects fsdd-george-0 and rejects another item, audio hash, transcript, prompt, model revision, or coefficient set. | [test_accepts_frozen_smoke_identity](../tests/experiments/test_audio_memory_fusion.py) | [test_rejects_changed_smoke_identity](../tests/experiments/test_audio_memory_fusion.py) |
| <nobr><code>AMF-VR-02</code></nobr> | <nobr><code>AMF-REQ-02</code></nobr> | Mean pooling emits exactly one ordered audio vector per transcript token.<br>Alpha zero returns the transcript vectors and alpha one returns their element-wise sum with the pooled audio vectors. | [test_constructs_equal_budget_memory_vectors](../tests/experiments/test_audio_memory_fusion.py) | [test_rejects_invalid_fusion_inputs](../tests/experiments/test_audio_memory_fusion.py) |
| <nobr><code>AMF-VR-03</code></nobr> | <nobr><code>AMF-REQ-03</code></nobr> | The result records the source commit, audio and logits hashes, exact revisions, position counts, peak GPU allocation, and finite comparison metrics.<br>Transcript-only and VoxZip-addition conditions have identical compressed input and memory-position counts. | [test_accepts_complete_smoke_result](../tests/experiments/test_audio_memory_fusion.py) | [test_rejects_unequal_memory_budgets](../tests/experiments/test_audio_memory_fusion.py)<br>[test_rejects_nonfinite_comparison](../tests/experiments/test_audio_memory_fusion.py) |
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
