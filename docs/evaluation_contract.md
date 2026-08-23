# CleaRx evaluation and decision contract

## Frozen inputs

[`evaluation_corpus_manifest.json`](../evaluation_corpus_manifest.json) fixes 26
audio items: 20 short spoken digits from two speakers, two public-domain long
speeches, and four deterministic PCM edge cases. Every item has a source URL or
generator, SHA-256, license, task stratum, prompt, and output limit. The FSDD
items are pinned to commit `26eb9aaf76e81b692f806f9140c2d2777410d7a1`
under CC BY-SA 4.0. The two Wikimedia Commons file records identify the source
recordings as public domain. CleaRx releases the synthetic PCM items under
CC0 1.0.

[`benchmark_thresholds.json`](../benchmark_thresholds.json) fixes the metrics,
margins, repetition rules, paired keys, confidence calculation, and failure
fields. P0-007B binds the same workload to the certified host hash; it cannot
change a metric or margin after results are visible.

## Workload and pairing

Each run uses batch size one and concurrency one. The evaluator constructs one
pair for every `(item_id, seed, sampling_id)` tuple and sends identical decoded
PCM, prompt, speaker, token limits, and sampling fields to the pinned
Transformers oracle and native candidate.

`greedy-correctness` disables Thinker and Talker sampling. It establishes exact
primary and residual codec-token custody before acoustic or performance claims.
`release-quality` keeps Thinker deterministic and uses the official Talker
settings: temperature 0.9, top-k 50, top-p 1.0, repetition penalty 1.05. Its
three seeds are 7, 19, and 43. Both paths select speaker `Ethan`, limit Thinker
output to 384 tokens, and limit Talker output to 4,096 tokens; each item may
apply a smaller declared output limit.

An evaluation is incomplete if either member of a pair is missing. A failed
member remains a failure and is not discarded from aggregate metrics. Spot
preemption may restart the incomplete repetition only after host-profile
identity is revalidated.

## Correctness gate

Under `greedy-correctness`, `codec_token_exact_match_rate` must equal 1.0 for
the primary codec stream and all 15 residual codebooks. A mismatch records the
first codec frame and codebook plus the first divergent upstream module or
boundary fixture. Waveform comparison cannot waive a token mismatch.

Module fixtures separately govern tensor tolerances. P0-003A must name those
tolerances before reference generation; this evaluation contract does not
invent one global tolerance for unrelated dtypes and operations.

## Semantic non-inferiority

The evaluator transcribes paired native and oracle waveforms with one frozen
ASR model and normalization function. `asr_word_error_rate_delta` is native WER
minus oracle WER against the item reference or expected-fact rendering. The
native system is noninferior only when the upper one-sided 95% paired-bootstrap
bound is at most 0.02 absolute.

For digit items, success also requires the normalized transcript to contain
exactly the expected digit word. Long-speech items use their named facts; an
item succeeds only when all required facts appear under the frozen semantic
scorer. Synthetic silence/noise/tone items fail when the response invents
speech content.

## Acoustic non-inferiority

The evaluator embeds native output, paired oracle output, and the frozen Ethan
enrollment clip with one pinned speaker encoder. For each pair,
`speaker_embedding_cosine_delta` is native-to-enrollment cosine similarity
minus oracle-to-enrollment similarity. The lower one-sided 95% paired-bootstrap
bound must be at least -0.02.

The speaker metric applies only after codec-token correctness passes for the
greedy profile. It detects a release-quality voice regression; it does not
establish semantic correctness or perceptual quality by itself.

## Material improvement

A bounded-context candidate declares its baseline before execution and uses
the same corpus pairs, seeds, byte budget, host, and output limits.
`paired_task_success_delta` is candidate success minus baseline success across
the declared corpus slice. “Material improvement” requires an absolute point
estimate of at least 0.03 and a lower 95% paired-bootstrap bound above zero.

The three-percentage-point threshold is an engineering adoption threshold, not
a population-effect estimate. A candidate below it remains an exploratory
result even when its confidence interval excludes zero.

## L4 performance gate

Performance runs use the certified `clearx-mantra-g2-l4-v1` identity, five
warm-ups, and 30 recorded repetitions per workload. Timing starts after input
bytes and configuration are ready in host memory.

- `audio_start_latency_ms_p95` is the 95th percentile from request acceptance
  to the first committed PCM frame and must not exceed 2,000 ms.
- `realtime_factor_p95` is the 95th percentile of generation wall time divided
  by emitted audio duration after the first frame and must not exceed 1.0.
- `underflow_count` counts output deadlines reached without one complete PCM
  frame and must equal zero.

All three conditions must pass. A mean below the limit cannot compensate for a
failed 95th percentile or any underflow.

## Confidence and repetitions

The evaluator performs 10,000 paired percentile-bootstrap resamples with seed
20,260,823. The resampling unit is one complete item-and-seed pair, preserving
both systems and the selected sampling profile inside each draw. Equivalence
and non-inferiority use the metric-specific one-sided bound. Material
improvement uses both its point estimate and lower bound.

The result stores every raw pair before calculating aggregates. Any post hoc
corpus exclusion creates a new exploratory analysis and cannot replace the
frozen acceptance result.

## Failure attribution

Every failed pair records `first_divergent_module`, `failure_class`,
`corpus_item_id`, `seed`, and `host_profile`. `failure_class` is one of input
decode, AuT, Thinker, Talker, MTP, Code2Wav, scheduler, memory, latency, or
evaluation tool. The first divergent module comes from boundary fixtures when
available; otherwise it is `unknown` and the pair remains failed.

The evaluator stores source commit, upstream-lock hash, corpus hash, thresholds
hash, model/GGUF hashes, host-profile hash, command line, driver, CUDA toolkit,
and timestamps with every run. Those identities make later comparisons joins
over exact artifacts rather than comparisons between labels.
