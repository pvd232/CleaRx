# Audio-memory transcript-trust experiment

Date: 2026-09-16

## Question

Can a small gate decide how much acoustic evidence to retain when compacted
memory already contains a transcript?

This is the most favorable useful test of learned audio-text fusion for CleaRx.
Qwen3-Omni-Instruct has demonstrated speech-recognition ability, while transcript
errors create the exact failure that audio memory could repair. VoxZip uses ASR
transcripts as semantic anchors, but applies a fixed element-wise audio-text
addition and identifies learned fusion as future work. Related spoken-language
systems find that audio-text fusion mitigates error propagation from imperfect
ASR transcripts.

Sources:

- [VoxZip](https://arxiv.org/abs/2608.08569)
- [Qwen3-Omni technical report](https://arxiv.org/abs/2509.17765)
- [Multimodal Audio-textual Architecture for Robust Spoken Language Understanding](https://arxiv.org/abs/2306.06819)
- [ASR Error Detection via Audio-Transcript Entailment](https://arxiv.org/abs/2207.10849)

## Frozen test

The experiment uses the two neutral RAVDESS speech statements:

- `Kids are talking by the door.`
- `Dogs are sitting by the door.`

For each recording, compacted memory contains one of three trust conditions. The
full-audio teacher always receives the original clean recording. Only the acoustic
vectors available to compressed memory are degraded.

| Memory condition | Transcript example for statement 1 | Acoustic memory | What the gate should learn |
| --- | --- | --- | --- |
| Reliable transcript | `Kids are talking by the door.` | Deterministic white noise at −10 dB SNR | Trust the compact transcript and suppress unreliable acoustic memory. |
| Incomplete transcript | `Are talking by the door.` | Clean | Add acoustic evidence to recover the missing subject. |
| Conflicting transcript | `Dogs are sitting by the door.` | Clean | Add acoustic evidence because the transcript describes the other recording. |

Actors 01–04 form the training split. Actors 05–06 are held out until the final
evaluation. Each actor contributes both statements under all three memory
conditions, producing 24 training items and 12 held-speaker test items. The noise
seed derives from the recording bytes, so every rerun produces identical samples.

The model first runs on the original audio and the prompt:

> Repeat the spoken sentence exactly. Answer only with the sentence.

That full-audio result is the teacher. Before the experiment continues, Qwen must
transcribe both actor-01 recordings correctly. This checks the capability on the
actual files rather than assuming it from the model description.

## Fusion and gate

The Audio Tower produces clean and degraded acoustic-vector sequences. Ordered mean pooling
reduces those vectors to the same number of positions as the transcript. Each
grid condition then supplies the Thinker with:

```text
transcript embedding + alpha * pooled audio vector
```

The frozen grid is `alpha = 0.0, 0.5, 1.0`. For each item, the coefficient whose
next-token distribution has the lowest KL divergence from full-audio inference
becomes the oracle target.

The gate receives eight summaries of the paired acoustic and transcript vectors:
cosine agreement, relative norms, acoustic magnitude and variation, temporal
change, and compression ratio. It never receives the transcript-quality label,
speaker label, expected sentence, or test outcome. A logistic-link ridge
regressor learns the oracle coefficients on actors 01–04. Ridge strength is
selected by leaving out one training speaker at a time.

## Decision rule

The primary directional result is positive when the learned coefficient produces
lower mean KL divergence on actors 05–06 than the single fixed coefficient chosen
on the training set.

Two supporting checks show whether that improvement represents the intended
behavior:

1. The oracle coefficient is lower when the transcript is exact and acoustic
   memory is degraded than when transcript memory is incomplete or conflicting
   and acoustic memory is clean.
2. The learned coefficient preserves that ordering for held-out speakers.

A positive result would not establish a production compactor. It would show that
the frozen Qwen representations contain enough information for a small,
speaker-generalizing policy to vary acoustic retention according to transcript
trust. A negative result would distinguish three causes: the full-audio teacher
failed the ASR preflight, the fusion coefficient did not help corrupted memory,
or the useful coefficient varied but the frozen gate features did not predict it.

## Execution boundary

The A100 runs only the pinned Qwen and RAVDESS worker. VIPER runs afterward on the
local machine to recompute the deterministic gate fit and verify the saved
result. VIPER does not sit in the model-loading or accelerator path.
