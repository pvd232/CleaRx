# Reference fixture contract

## Status and scope

P0-003A defines both PyTorch reference observations and native-runtime contract
cases. P0-003B supplies values for the PyTorch-observable pairs. P0-004 owns
synthetic scheduler and bounded-context reconstruction traces. The executable source of truth is
[`fixture_harness.py`](../../tools/reference/fixture_harness.py); this document
explains its boundary matrix, generator inputs, persisted evidence, and local
comparison rules.

Every generated fixture uses batch size one, seed `20260823`, and the
`greedy-correctness` sampling profile. Thinker, Talker, primary-codec, and MTP
selection use deterministic argmax for these fixtures. Paired end-to-end
evaluation owns the release-quality sampling path.

## Evidence model

One fixture consists of three layers:

1. The frozen
   [`fixture.schema.json`](../../orchestration/schemas/fixture.schema.json)
   envelope identifies the descriptor and every input and output artifact.
2. The strict
   [`fixture_descriptor.schema.json`](../../orchestration/schemas/fixture_descriptor.schema.json)
   descriptor names one boundary, one case, the exact comparison profile,
   execution settings, and artifact metadata.
3. The artifact files contain the observed input and output arrays. Numeric
   arrays use NumPy `.npy` files with `allow_pickle=False`. The generator widens
   BF16 observations to float32 before saving; this preserves every BF16 value
   exactly and enables local comparison using only NumPy. The
   descriptor therefore records the persisted dtype as `float32`, while the
   selected `bf16_*` profile records the reference computation class.

The envelope `fixture_id` equals the descriptor `descriptor_id`. Its
`producer_commit`, `seed`, and `sampling` object equal the corresponding
descriptor values. The envelope artifacts other than the descriptor equal the
union of `descriptor.inputs` and `descriptor.outputs` by path, SHA-256, dtype,
and shape. The validator recalculates every artifact hash.

The descriptor also binds these identities:

- [`upstream.lock`](../../upstream.lock);
- [`tensor_manifest.json`](../../tensor_manifest.json);
- [`evaluation_contract.md`](../evaluation_contract.md);
- the strict descriptor schema;
- the generator path, bytes, and Git commit; and
- the source model commit and `config.json` hash recorded in `upstream.lock`.

For a model-boundary fixture, the validator reads the generator from the
recorded Git commit and requires those bytes to equal both the working-tree
generator and `generator_sha256`. Those equality joins turn each provenance
label into an exact identity.

## Boundary matrix

Shapes use symbolic axis names because case lengths vary. P0-003B replaces
each symbolic axis with its observed nonnegative integer in the descriptor.
The fixed dimensions come from the frozen model configuration and tensor
manifest: AuT hidden size 1,280 and output size 2,048; Thinker hidden size
2,048; Talker and MTP hidden size 1,024; 128 Talker experts with six selected
per token; a 3,072-entry primary codec vocabulary; 15 residual codebooks with
2,048 entries each; and 16 Code2Wav quantizers.

| Boundary pattern | Persisted output | Shape | Profile |
| --- | --- | --- | --- |
| `input.pcm_preprocessor` | log-mel input to AuT | `[1, 128, aut_frames]` | `fp32_strict` |
| `aut.encoder_output` | final AuT hidden state | `[aut_frames, 1280]` | `bf16_accumulation` |
| `aut.output_projection` | AuT `proj2` result accepted by Thinker | `[aut_frames, 2048]` | `bf16_accumulation` |
| `thinker.accepted_hidden` | accepted Thinker text hidden state | `[1, thinker_positions, 2048]` | `bf16_accumulation` |
| `thinker.talker_conditioning` | Talker text-projection result | `[1, talker_positions, 1024]` | `bf16_accumulation` |
| `talker.layer.00..19.router_logits` | float32 router softmax over all experts | `[talker_positions, 128]` | `fp32_strict` |
| `talker.layer.00..19.selected_experts` | ordered top-six expert indices | `[talker_positions, 6]` | `exact_discrete` |
| `talker.layer.00..19.routing_weights` | normalized top-six weights | `[talker_positions, 6]` | `fp32_strict` |
| `talker.layer.00..19.expert_contributions` | six weighted routed-expert results before reduction | `[talker_positions, 6, 1024]` | `bf16_activation` |
| `talker.layer.00..19.hidden` | decoder-layer output after attention and MoE residuals | `[1, talker_positions, 1024]` | `bf16_accumulation` |
| `talker.primary_codec_logits` | primary codec head logits | `[1, codec_frames, 3072]` | `fp32_strict` |
| `talker.primary_codec_token` | primary codec argmax | `[1, codec_frames]` | `exact_discrete` |
| `mtp.residual.01..15.logits` | residual head logits at that within-frame step | `[1, codec_frames, 2048]` | `fp32_strict` |
| `mtp.residual.01..15.token` | residual-code argmax at that step | `[1, codec_frames]` | `exact_discrete` |
| `code2wav.code_input` | complete primary-plus-residual code tensor | `[1, 16, codec_frames]` | `exact_discrete` |
| `code2wav.chunk.first` | first committed waveform chunk | `[1, 1, pcm_samples]` | `waveform_fp32` |
| `code2wav.chunk.steady` | a full steady-state waveform chunk | `[1, 1, pcm_samples]` | `waveform_fp32` |
| `code2wav.chunk.final_partial` | final committed partial waveform chunk | `[1, 1, pcm_samples]` | `waveform_fp32` |
| `handoff.thinker_to_talker` | projected conditioning at transfer | `[1, talker_positions, 1024]` | `bf16_accumulation` |
| `handoff.talker_to_mtp_hidden` | last Talker hidden state at transfer | `[1, codec_frames, 1024]` | `bf16_accumulation` |
| `handoff.talker_to_mtp_token` | primary codec token at transfer | `[1, codec_frames]` | `exact_discrete` |
| `handoff.codec_to_code2wav` | complete 16-codebook frame tensor at transfer | `[1, 16, codec_frames]` | `exact_discrete` |
| `scheduler.turn_trace` | ordered state-transition records | `[events, 4]` | `exact_discrete` |

The scheduler trace columns are `sequence`, `event_code`, `turn_id`, and
`state_code`. The fixture contract freezes event codes `input_chunk=1`,
`background_interval=2`, `turn_end=3`, `thinker_ready=4`, `talker_start=5`,
`codec_frame=6`, `waveform_commit=7`, `interrupt=8`, `cancel=9`,
`context_rebuild=10`, and `listen_resume=11`. It freezes state codes `idle=0`,
`listening=1`, `thinking=2`, `speaking=3`, `cancelling=4`, and `rebuilding=5`.
P0-004 must adopt these values or supply an explicit ABI-to-fixture crosswalk;
P0-003B uses this single codebook.

The Talker router values correspond to the pinned reference sequence: a
float32 softmax over 128 logits, an ordered top-six selection, optional
top-weight normalization, and weighted expert contributions. The MTP rows
preserve the within-frame order: residual step 1 consumes the primary code,
and each later step also consumes every earlier residual code.

## Comparison profiles

These are project-local CleaRx engineering acceptance rules.
`fixture_contract.py` stores their canonical numeric objects; a
descriptor must repeat the entire selected object exactly.

| Profile | Mode | Absolute tolerance | Relative tolerance | Minimum cosine |
| --- | --- | ---: | ---: | ---: |
| `exact_discrete` | exact | 0 | 0 | 1 |
| `fp32_strict` | all-close | 0.000001 | 0.00001 | 0.999999 |
| `bf16_activation` | all-close | 0.03125 | 0.03125 | 0.999 |
| `bf16_accumulation` | all-close | 0.0625 | 0.05 | 0.995 |
| `waveform_fp32` | all-close | 0.001 | 0.001 | 0.9999 |

Every profile sets `equal_nan` to false and `max_mismatch_fraction` to zero.
The harness first requires identical shape and dtype. Exact profiles require
element equality. Floating profiles require every element to satisfy the
absolute-plus-relative rule and require the flattened vectors to meet the
cosine floor. Codec-token exactness remains an independent prerequisite for
waveform comparison.

## Generator interface

P0-003B must add `tools/reference/generate_reference_fixtures.py` with this
noninteractive interface:

```text
python tools/reference/generate_reference_fixtures.py \
  --output tests/fixtures/reference \
  --seed 20260823 \
  --sampling-profile greedy-correctness \
  --boundary all \
  --case all
```

The generator must:

1. refuse a source commit or `config.json` hash that differs from
   `upstream.lock`;
2. enable deterministic framework algorithms and record the exact framework,
   version, and A100 device string;
3. create inputs from the declared case definitions and seed;
4. observe the named values while preserving the model operation that produces
   them;
5. widen each BF16 observation directly to float32;
6. write one descriptor and envelope per pair returned by
   `required_model_fixture_pairs()`, then write a
   top-level `tests/fixtures/reference/manifest.json` index; and
7. rerun `python scripts/validate_fixtures.py` before returning success.

Hooks may read module inputs and returned values. Router expert contributions
need a dedicated observer around the selected-expert call because the reference
class normally returns only their reduction. The observer preserves selection
order, routing weights, accumulator dtype, and the returned tensor.

## Deterministic cases

`fixture_harness.py` assigns each case to the boundary where its effect first
becomes observable.

| Case | Required observation |
| --- | --- |
| `empty_background` | the PCM preprocessor receives a zero-length background interval |
| `background_noise` | seeded low-amplitude noise precedes speech |
| `variable_chunk` | unequal PCM input and Code2Wav output chunk lengths preserve order |
| `final_partial_chunk` | the final Code2Wav call contains fewer codec frames than a steady chunk |
| `turn_end` | the final codec frame is committed before the turn-end scheduler event |
| `long_turn` | the trace crosses the bounded-context rebuild threshold |
| `interruption` | new input interrupts active output; its event follows the last committed chunk |
| `cancellation` | cancellation follows the last waveform commit and discards queued codec work |
| `repeated_phase_transition` | listening and speaking repeat in the same process |
| `context_reconstruction` | rebuilt Thinker conditioning and the scheduler trace are persisted |

Every adversarial descriptor has class `adversarial` and at least one of these
tags. The nominal case carries an empty tag array. The contract-only harness fails when a tag,
case assignment, layer, residual step, handoff, profile, or coordinate domain
is missing.

The harness partitions declared pairs by evidence source. P0-003B generates 154
`model_reference` pairs. Eight `synthetic_native` pairs cover the scheduler
trace cases plus context reconstruction at `thinker.accepted_hidden` and
`handoff.thinker_to_talker`. The pinned PyTorch implementation exposes the
model tensor handoffs and codec path. The future native runtime supplies the
scheduler state machine and bounded-context rebuild operation, so its ABI tests
will generate those eight traces after P0-004 freezes their concrete inputs.

## Local validation

Run the lightweight contract gate; it loads only metadata:

```text
python scripts/validate_fixtures.py --contract-only
```

Compare one candidate NumPy artifact with its reference using the profile from
the descriptor:

```text
python tools/reference/fixture_harness.py \
  --reference tests/fixtures/reference/example.reference.npy \
  --candidate path/to/example.candidate.npy \
  --profile bf16_accumulation
```

The comparison exits nonzero for an unknown profile, shape difference, dtype
difference, elementwise tolerance failure, or cosine failure.

## Pinned implementation evidence

The boundary names follow the pinned Transformers implementation at commit
`7d9754a05193eb79b1d86aa744b622b8068008cd`: the AuT returns its post-`proj2`
hidden state; the Talker router returns softmax values, routing weights, and
indices; the Talker codec head produces the primary logits; the code predictor
produces the remaining code groups; and Code2Wav embeds all quantizers before
chunked waveform decode. See the official
[`modeling_qwen3_omni_moe.py`](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py)
and the pinned model
[`config.json`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/26291f793822fb6be9555850f06dfe95f2d7e695/config.json).
