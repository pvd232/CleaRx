# Crossed Audio Retention Pilot

## 1. Status

This contract governs one exploratory A100 execution. It corrects the earlier
pilot's audio--transcript confound by evaluating every transcript condition
with both clean and degraded audio.

<!-- contract-protocol:generated:start -->
**In progress.** [Jump to current PairBlock](#car-pb-01)

**Checklist:** [Crossed audio retention pilot](../checklists/crossed-audio-retention-pilot.md)

### PairBlocks

<a id="car-pb-01"></a>

#### <nobr><code>CAR-PB-01</code></nobr>

**Status:** review

**Requirement contribution:** Add the crossed policy analysis, guarded A100 runner, focused counterexamples, and two-stage VIPER execution path.

**Review handoff**

**What changed**

- Add the complete transcript-quality by audio-quality experiment and two equal-budget learned allocation policies.
- Preserve transcript embeddings while measuring zero, two, and four chronological mean-pooled acoustic positions at alpha one.
- Run the A100 measurement and independent CPU recomputation as connected VIPER stages and retain the complete audit graph.

**Plan deviations:** The pilot allocates across independent RAVDESS items rather than among several transcript segments in one prompt. It tests the allocation mechanism without claiming publication-level generalization or end-to-end context-window savings.

**Start review:** [Open tested GitHub comparison](https://github.com/pvd232/CleaRx/compare/206f01748dc0c78700f850c66eb88ae4001e2d2e...7a3cbb6ca50903b0e6a0cc30823ce7ac30e9d174)

**Review these files**

- [Crossed retention policy and verifier](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/src/clearx/audio_memory/crossed_retention.py#L1)
- [Connected VIPER A100 execution](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/experiments/audio_memory_crossed_retention/run_viper.py#L1)
- [Contract counterexamples](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/tests/unit/test_crossed_audio_retention.py#L1)

**Evidence:** [Passing gate receipt](../evidence/crossed-audio-retention-pilot/gates/car-pb-01-20260917t015836z.json)

**Decision:** Approve <nobr><code>CAR-PB-01</code></nobr>, or return it with findings.

<details>
<summary>Implementation details</summary>

**Plan:** [plan.toml](../plans/crossed-audio-retention-pilot/CAR-PB-01/plan.toml)

**Candidate files:** [src/clearx/audio_memory/crossed_retention.py](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/src/clearx/audio_memory/crossed_retention.py) · [experiments/audio_memory_crossed_retention/__init__.py](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/experiments/audio_memory_crossed_retention/__init__.py) · [experiments/audio_memory_crossed_retention/README.md](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/experiments/audio_memory_crossed_retention/README.md) · [experiments/audio_memory_crossed_retention/bootstrap.py](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/experiments/audio_memory_crossed_retention/bootstrap.py) · [experiments/audio_memory_crossed_retention/run_viper.py](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/experiments/audio_memory_crossed_retention/run_viper.py) · [tests/unit/test_crossed_audio_retention.py](../plans/crossed-audio-retention-pilot/CAR-PB-01/add/tests/unit/test_crossed_audio_retention.py)

**Implementation roots:** [src/clearx](../src/clearx) · [experiments/audio_memory_crossed_retention](../experiments/audio_memory_crossed_retention)

**Test roots:** [tests/unit](../tests/unit)

**Dependencies:** None

**Gate steps:**

```bash
# typecheck
(cd . && pyright --pythonpath /Users/machina/miniconda3/envs/clearx/bin/python src/clearx/audio_memory/crossed_retention.py experiments/audio_memory_crossed_retention/bootstrap.py experiments/audio_memory_crossed_retention/run_viper.py tests/unit/test_crossed_audio_retention.py)
# test
(cd src && python -m pytest -q ../tests/unit/test_crossed_audio_retention.py)
# documentation
(cd . && python /Users/machina/.agents/skills/code-documentation/scripts/check-schema-descriptions.py src/clearx/audio_memory/crossed_retention.py experiments/audio_memory_crossed_retention/bootstrap.py experiments/audio_memory_crossed_retention/run_viper.py tests/unit/test_crossed_audio_retention.py)
# lint
(cd . && ruff format --check src/clearx/audio_memory/crossed_retention.py experiments/audio_memory_crossed_retention/bootstrap.py experiments/audio_memory_crossed_retention/run_viper.py tests/unit/test_crossed_audio_retention.py)
# lint
(cd . && ruff check src/clearx/audio_memory/crossed_retention.py experiments/audio_memory_crossed_retention/bootstrap.py experiments/audio_memory_crossed_retention/run_viper.py tests/unit/test_crossed_audio_retention.py)
```

</details>


### Requirements

| Requirement | Claim | Progress | Verifiers | PairBlocks |
|---|---|---|---|---|
| <nobr><code>CAR-REQ-01</code></nobr> | Freeze the complete 72-item product of six speakers, two recordings, three transcript conditions, and clean and degraded audio, with alpha 1 and acoustic-position counts 0, 2, and 4. | in_progress | <nobr><code>CAR-VR-01</code></nobr> | <nobr><code>CAR-PB-01</code></nobr> |
| <nobr><code>CAR-REQ-02</code></nobr> | Preserve every transcript embedding and append exactly the requested number of chronological mean-pooled acoustic positions. | in_progress | <nobr><code>CAR-VR-02</code></nobr> | <nobr><code>CAR-PB-01</code></nobr> |
| <nobr><code>CAR-REQ-03</code></nobr> | Compare fixed, agreement-adaptive, audio-only-adaptive, and oracle allocations on held-out speakers at the same aggregate acoustic and input-position budgets, with recording-clustered uncertainty. | in_progress | <nobr><code>CAR-VR-03</code></nobr> | <nobr><code>CAR-PB-01</code></nobr> |
| <nobr><code>CAR-REQ-04</code></nobr> | Retain one VIPER run that observes the A100 measurement stage, binds its result bytes, and recomputes both learned policies, every allocation, budget, contrast, cell summary, and bootstrap value in a separate verification stage. | in_progress | <nobr><code>CAR-VR-04</code></nobr> | <nobr><code>CAR-PB-01</code></nobr> |

### Verification rules

| Rule | Requirements | Acceptance conditions | Success case | Rejection cases |
|---|---|---|---|---|
| <nobr><code>CAR-VR-01</code></nobr> | <nobr><code>CAR-REQ-01</code></nobr> | Every speaker and recording contains all six transcript-by-audio cells, actors 01 through 04 form the training split, and actors 05 and 06 form the held-out split.<br>The model, Transformers revision, dataset archive, selected WAV digests, prompt, noise transform, alpha, and position grid equal their frozen values. | [test_frozen_protocol_is_complete_crossed_product](../tests/unit/test_crossed_audio_retention.py) | [test_payload_rejects_missing_crossed_cell_and_wrong_audio_hash](../tests/unit/test_crossed_audio_retention.py) |
| <nobr><code>CAR-VR-02</code></nobr> | <nobr><code>CAR-REQ-02</code></nobr> | The shared candidate assembler retains the transcript array byte-for-byte and appends exactly zero, two, or four acoustic rows.<br>The A100 stage asserts the same transcript slice before invoking the Thinker. | [test_candidate_assembly_preserves_transcript_embeddings](../tests/unit/test_crossed_audio_retention.py) | [test_candidate_assembly_rejects_incompatible_parts](../tests/unit/test_crossed_audio_retention.py) |
| <nobr><code>CAR-VR-03</code></nobr> | <nobr><code>CAR-REQ-03</code></nobr> | Both learned policies fit only actors 01 through 04 and never read held-out KL values during allocation.<br>The audio-only policy uses one score per recording and audio-quality group and selects complete three-transcript groups.<br>All four policies spend 48 acoustic positions and the same aggregate input positions across the 24 held-out items.<br>The result contains per-recording and per-actor contrasts plus all 256 deterministic recording-cluster bootstrap means for both primary comparisons. | [test_crossed_payload_compares_equal_budget_policies](../tests/unit/test_crossed_audio_retention.py) | [test_learned_allocations_ignore_held_out_labels](../tests/unit/test_crossed_audio_retention.py) |
| <nobr><code>CAR-VR-04</code></nobr> | <nobr><code>CAR-REQ-04</code></nobr> | The measurement result names the exact run-source commit and reports the CUDA device and compute capability observed by VIPER.<br>VIPER supplies the measurement artifact directly to the verification stage and the verification report records its SHA-256 digest.<br>The retained Git audit archive contains every run, stage, attempt, invocation, journal, log, artifact, measurement, and verification record referenced by the terminal result. | [test_result_validation_binds_commit_hardware_and_recomputed_payload](../tests/unit/test_crossed_audio_retention.py) | [test_result_validation_rejects_provenance_and_metric_tampering](../tests/unit/test_crossed_audio_retention.py) |
<!-- contract-protocol:generated:end -->

## 2. Claim

For the frozen held-out RAVDESS items, the experiment tests whether an
audio--text agreement policy allocates a fixed acoustic-position budget better
than uniform retention and an audio-only policy.

Each item combines one recording, one transcript condition, and one audio
condition. Every policy receives the same transcript embeddings. The fixed
policy appends two acoustic positions to each item; each adaptive policy
appends four positions to half the items and zero to the rest.

## 3. Models

The agreement model uses the existing eight audio--text features. The
audio-only ablation uses three features computed from Audio Tower vectors
before transcript-dependent pooling. Both standardized ridge regressors fit
actors 01--04 and allocate positions for actors 05--06 without reading their
KL targets.

## 4. Execution

1. Load the pinned Qwen3-Omni Thinker and the twelve digest-bound recordings.
2. Cross exact, incomplete, and conflicting transcripts with clean and
   degraded audio.
3. Measure transcript-only, two-position, and four-position candidates with
   fusion strength fixed at one.
4. Fit both policies on actors 01--04 and allocate equal held-out budgets.
5. Compute recording-clustered contrasts and exact bootstrap intervals.
6. Run the measurement and recomputation as connected VIPER stages.

## 5. Persisted evidence

The repository retains the contract, tested review commit, compact measurement
result, verification report, and complete Git audit copy of the VIPER record
graph. Model weights, dataset archives, credentials, and caches remain outside
Git.

## 6. Verification

Focused tests reject incomplete cells, changed inputs, transcript displacement,
held-out-label shortcuts, unequal budgets, forged provenance, and altered
summaries. VIPER binds the measurement artifact to the verification-stage
input and records the runtime observed for each stage.

## 7. Propagation

This contract adds a separate crossed experiment. It leaves the prior adaptive
retention result unchanged as historical exploratory evidence.

## 8. Acceptance case

The execution is valid when all 72 cells are present, every policy spends 48
held-out acoustic positions, and VIPER reproduces all derived values. The
result is `positive_exploratory` only when the agreement policy beats fixed and
audio-only allocation overall, for each held-out actor, and at both upper
cluster-bootstrap bounds.

## 9. Implementation order

1. Gate and publish the complete implementation candidate.
2. Run that exact review commit through the two-stage VIPER plan on an A100.
3. Retain and verify the complete run graph and compact artifacts.
4. Return the implementation and evidence for independent review.

## 10. Contract-owned PairBlocks

`CAR-PB-01` owns the crossed policy analysis, A100 execution path, tests, and
VIPER verification.

## 11. ContractTarget

The PairBlock adds complete files over the recorded clean baseline. The
contract runner materializes the candidate, runs the ordered gate, and binds
the tested bytes to the review commit.
