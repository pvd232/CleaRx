# Adaptive Audio Retention Pilot

## 1. Status

The pilot is approved for one diagnostic A100 execution. It compares allocation
policies, not fusion strength: every retained acoustic vector uses
$\alpha=1$.

<!-- contract-protocol:generated:start -->
**In progress.** [Jump to current PairBlock](#aar-pb-01)

**Checklist:** [Adaptive audio retention pilot](../checklists/adaptive-audio-retention-pilot.md)

### PairBlocks

<a id="aar-pb-01"></a>

#### <nobr><code>AAR-PB-01</code></nobr>

**Status:** review

**Requirement contribution:** Add the fixed-budget retention policy, A100 experiment, local counterexamples, and VIPER result verifier for one held-speaker pilot.

**Review handoff**

**What changed**

- Add chronological pooling and a held-speaker ridge policy that predicts the benefit of four acoustic positions.
- Add an A100 runner that preserves transcript positions and measures 0, 2, and 4 appended acoustic positions with alpha fixed at one.
- Add focused counterexample tests and a VIPER diagnostic that recomputes the retained model, allocation, budgets, and metrics.

**Plan deviations:** The pilot allocates across independent RAVDESS items rather than among several transcript segments in one prompt. It establishes the allocation signal and equal-budget comparison without claiming end-to-end context-window savings.

**Start review:** [Open tested GitHub comparison](https://github.com/pvd232/CleaRx/compare/a074c895d29bdc7ec64c5bba122246d9c1a407b9...1a91c455f7bd955fc6f91c316ddb77164bc0a182)

**Review these files**

- [Retention policy and evidence validator](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/src/clearx/audio_memory/retention.py#L1)
- [A100 equal-budget experiment](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/experiments/audio_memory_adaptive_retention/run_experiment.py#L1)
- [VIPER result verifier](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/experiments/audio_memory_adaptive_retention_verification/verify.py#L1)
- [Contract counterexamples](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/tests/unit/test_audio_memory_retention.py#L1)

**Evidence:** [Passing gate receipt](../evidence/adaptive-audio-retention-pilot/gates/aar-pb-01-20260916t193000z.json)

**Decision:** Approve <nobr><code>AAR-PB-01</code></nobr>, or return it with findings.

<details>
<summary>Implementation details</summary>

**Plan:** [plan.toml](../plans/adaptive-audio-retention-pilot/AAR-PB-01/plan.toml)

**Candidate files:** [src/clearx/audio_memory/retention.py](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/src/clearx/audio_memory/retention.py) · [tests/unit/test_audio_memory_retention.py](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/tests/unit/test_audio_memory_retention.py) · [experiments/audio_memory_adaptive_retention/README.md](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/experiments/audio_memory_adaptive_retention/README.md) · [experiments/audio_memory_adaptive_retention/bootstrap.py](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/experiments/audio_memory_adaptive_retention/bootstrap.py) · [experiments/audio_memory_adaptive_retention/run_experiment.py](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/experiments/audio_memory_adaptive_retention/run_experiment.py) · [experiments/audio_memory_adaptive_retention_verification/spec.yaml](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/experiments/audio_memory_adaptive_retention_verification/spec.yaml) · [experiments/audio_memory_adaptive_retention_verification/variants/ravdess_equal_budget.spec.yaml](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/experiments/audio_memory_adaptive_retention_verification/variants/ravdess_equal_budget.spec.yaml) · [experiments/audio_memory_adaptive_retention_verification/verify.py](../plans/adaptive-audio-retention-pilot/AAR-PB-01/add/experiments/audio_memory_adaptive_retention_verification/verify.py)

**Implementation roots:** [src/clearx](../src/clearx) · [experiments/audio_memory_adaptive_retention](../experiments/audio_memory_adaptive_retention) · [experiments/audio_memory_adaptive_retention_verification](../experiments/audio_memory_adaptive_retention_verification)

**Test roots:** [tests/unit](../tests/unit)

**Dependencies:** None

**Gate steps:**

```bash
# typecheck
(cd . && pyright --pythonpath /Users/machina/miniconda3/envs/clearx/bin/python src/clearx/audio_memory/retention.py experiments/audio_memory_adaptive_retention/bootstrap.py experiments/audio_memory_adaptive_retention/run_experiment.py experiments/audio_memory_adaptive_retention_verification/verify.py tests/unit/test_audio_memory_retention.py)
# test
(cd src && python -m pytest -q ../tests/unit/test_audio_memory_retention.py)
# documentation
(cd . && python /Users/machina/.agents/skills/code-documentation/scripts/check-schema-descriptions.py src/clearx/audio_memory/retention.py experiments/audio_memory_adaptive_retention/bootstrap.py experiments/audio_memory_adaptive_retention/run_experiment.py experiments/audio_memory_adaptive_retention_verification/verify.py tests/unit/test_audio_memory_retention.py)
# lint
(cd . && ruff format --check src/clearx/audio_memory/retention.py experiments/audio_memory_adaptive_retention/bootstrap.py experiments/audio_memory_adaptive_retention/run_experiment.py experiments/audio_memory_adaptive_retention_verification/verify.py tests/unit/test_audio_memory_retention.py)
# lint
(cd . && ruff check src/clearx/audio_memory/retention.py experiments/audio_memory_adaptive_retention/bootstrap.py experiments/audio_memory_adaptive_retention/run_experiment.py experiments/audio_memory_adaptive_retention_verification/verify.py tests/unit/test_audio_memory_retention.py)
```

</details>


### Requirements

| Requirement | Claim | Progress | Verifiers | PairBlocks |
|---|---|---|---|---|
| <nobr><code>AAR-REQ-01</code></nobr> | Freeze the RAVDESS speaker split, transcript conditions, model revisions, alpha 1 fusion strength, and acoustic-position grid 0, 2, and 4. | in_progress | <nobr><code>AAR-VR-01</code></nobr> | <nobr><code>AAR-PB-01</code></nobr> |
| <nobr><code>AAR-REQ-02</code></nobr> | Preserve every transcript-token embedding and append exactly the requested number of chronological mean-pooled acoustic positions. | in_progress | <nobr><code>AAR-VR-02</code></nobr> | <nobr><code>AAR-PB-01</code></nobr> |
| <nobr><code>AAR-REQ-03</code></nobr> | Compare a fixed two-position allocation with a learned zero-or-four-position allocation on held-out speakers at the same aggregate acoustic-position budget. | in_progress | <nobr><code>AAR-VR-03</code></nobr> | <nobr><code>AAR-PB-01</code></nobr> |
| <nobr><code>AAR-REQ-04</code></nobr> | Retain one A100 result whose source, inputs, conditions, budgets, and adaptive evaluation are independently recomputed by VIPER. | in_progress | <nobr><code>AAR-VR-04</code></nobr> | <nobr><code>AAR-PB-01</code></nobr> |

### Verification rules

| Rule | Requirements | Acceptance conditions | Success case | Rejection cases |
|---|---|---|---|---|
| <nobr><code>AAR-VR-01</code></nobr> | <nobr><code>AAR-REQ-01</code></nobr> | The experiment identity fixes alpha 1, position counts 0, 2, and 4, actors 01 through 04 for fitting, and actors 05 and 06 for evaluation. | [test_frozen_protocol_has_equal_budget_arms](../tests/unit/test_audio_memory_retention.py) | [test_payload_rejects_unequal_or_incomplete_position_grids](../tests/unit/test_audio_memory_retention.py) |
| <nobr><code>AAR-VR-02</code></nobr> | <nobr><code>AAR-REQ-02</code></nobr> | Ordered mean pooling returns exactly the requested count without changing transcript positions.<br>Zero requested acoustic positions returns an empty acoustic sequence. | [test_mean_pool_ordered_preserves_order_and_requested_count](../tests/unit/test_audio_memory_retention.py) | [test_mean_pool_ordered_rejects_invalid_inputs](../tests/unit/test_audio_memory_retention.py) |
| <nobr><code>AAR-VR-03</code></nobr> | <nobr><code>AAR-REQ-03</code></nobr> | The regressor fits only training actors and uses the declared gate features.<br>The adaptive arm assigns four positions to exactly half of held-out items and zero to the remainder, matching the fixed arm's total budget.<br>Reported fixed, adaptive, and oracle means are recomputed from measured KL values. | [test_retention_payload_fits_train_and_allocates_equal_test_budget](../tests/unit/test_audio_memory_retention.py) | [test_payload_rejects_held_out_label_shortcuts](../tests/unit/test_audio_memory_retention.py) |
| <nobr><code>AAR-VR-04</code></nobr> | <nobr><code>AAR-REQ-04</code></nobr> | The result binds the experiment commit, pinned model and Transformers revisions, RAVDESS digest, A100 device, and finite per-condition KL values.<br>VIPER reloads the retained result, recomputes the allocation payload, and rejects any budget or metric mismatch. | [test_validate_result_accepts_recomputable_evidence](../tests/unit/test_audio_memory_retention.py) | [test_validate_result_rejects_tampered_evaluation](../tests/unit/test_audio_memory_retention.py) |
<!-- contract-protocol:generated:end -->

## 2. Claim

For held-out items, a learned policy may reduce divergence from the full-audio
Thinker by concentrating the same total number of acoustic positions on the
items predicted to benefit most.

For item $i$, the Thinker receives every transcript embedding $T_i$ plus
$b_i$ chronological mean-pooled acoustic vectors:

$$
H_i = [T_i; P_{b_i}(A_i)], \qquad b_i \in \{0,2,4\}.
$$

The fixed arm assigns $b_i=2$ to every item. The adaptive arm ranks items by
a model fitted on actors 01--04, assigns $b_i=4$ to the top half of actors
05--06, and assigns $b_i=0$ to the rest. Both arms therefore spend the same
aggregate acoustic-position budget.

## 3. Models

`RetentionModel` is a standardized ridge regressor over the existing eight
audio--text agreement features. Its target is the measured benefit of four
acoustic positions over transcript only:

$$
\Delta_i = \mathrm{KL}_{i,0} - \mathrm{KL}_{i,4}.
$$

Leave-one-training-speaker-out error selects the ridge penalty. The model never
reads held-out KL values when choosing the adaptive allocation.

## 4. Execution

1. Load the pinned Qwen3-Omni Thinker and the frozen RAVDESS items.
2. Produce the full-audio teacher logits and Audio Tower vectors.
3. Preserve all transcript embeddings and evaluate 0, 2, and 4 appended pooled
   acoustic vectors with $\alpha=1$.
4. Fit the benefit model on actors 01--04.
5. Compare fixed and adaptive allocations on actors 05--06 at equal total
   acoustic positions.
6. Save the result and have VIPER recompute the model, allocation, budgets, and
   summary metrics.

## 5. Persisted evidence

The repository retains the immutable contract, PairBlock plan and gate receipt,
compact A100 result, and VIPER verification receipt. Model weights, dataset
archives, credentials, and runtime caches remain outside Git.

## 6. Verification

Local tests reject identity drift, malformed pooling, train/test leakage,
unequal budgets, non-finite measurements, and tampered summaries. VIPER
recomputes the complete allocation payload from the retained item observations.

## 7. Propagation

The pilot adds one reusable retention module, one A100 runner, one VIPER
diagnostic, and focused unit tests. It does not alter Qwen, VoxZip, or the prior
fusion experiments.

## 8. Acceptance case

The execution is valid when the fixed and adaptive test arms spend the same
number of acoustic positions and VIPER reproduces every allocation and metric.
The scientific result is positive only when adaptive mean teacher KL is lower
than fixed mean teacher KL.

## 9. Implementation order

1. Gate and publish the contract-owned implementation review commit.
2. Run that exact commit on an A100.
3. Retain the compact result.
4. Run the VIPER verifier against the retained result.

## 10. Contract-owned PairBlocks

`AAR-PB-01` owns the policy, runner, tests, and verification path.

## 11. ContractTarget

The PairBlock adds complete files. The protocol materializes them over the
recorded clean baseline, runs the ordered gate, and binds the tested bytes to
the review commit.

## Future work

A later experiment may allocate positions among several transcript segments
inside one prompt. This pilot treats each RAVDESS item as one allocation unit
and therefore establishes an item-level proxy, not a production memory policy.
