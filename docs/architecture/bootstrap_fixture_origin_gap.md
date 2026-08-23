# Bootstrap fixture-origin gap

## Required claim

P0-003B generates deterministic reference fixtures from the pinned PyTorch
implementation. Its index coverage must represent values that the generator can
observe in that implementation.

## A100 evidence

The packet-bound probe in
[`load_probe.json`](../../runs/P0-003B/20260823T161735Z-a100/load_probe.json)
loaded the full checkpoint across a 40 GB A100 and CPU memory. An explicit map
kept Talker and Code2Wav on CUDA and placed Thinker layers 21 through 47 on CPU.
The complete Thinker-to-waveform path then produced a `[1, 1, 1365]` float32
waveform from one codec frame.

The same execution established the evidence boundary. The pinned model exposes
AuT, Thinker, Talker, MTP, Code2Wav, and their tensor handoffs. The future native
runtime owns the serving scheduler and bounded-context reconstruction operation.

## Corrected contract

The P0-003A harness retains all 162 declared boundary-case pairs and partitions
them into two exhaustive sets:

- `model_reference`: 154 pairs observable in the pinned PyTorch model and
  required in the P0-003B index;
- `synthetic_native`: eight scheduler or context-reconstruction pairs generated
  after P0-004 freezes the native ABI inputs.

The native set contains all six `scheduler.turn_trace` cases plus
`context_reconstruction` at `thinker.accepted_hidden` and
`handoff.thinker_to_talker`. Contract-only validation still requires these
cases and their tags. Full reference-index validation requires exact equality
with `required_model_fixture_pairs()`.

This split preserves the adversarial requirements and assigns each persisted
value to an implementation that can produce it. Fabricated scheduler records
labeled as model output fail the P0-003B evidence contract.

## Acceptance case

The correction passes when the harness proves that its two sets partition every
declared pair, the model-reference set contains 154 pairs, nominal Talker layer
19 remains required, and scheduler interruption plus Thinker context
reconstruction stay in the native set.
