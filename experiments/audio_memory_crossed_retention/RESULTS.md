# Crossed audio-retention results

## Headline

At the same 48-position acoustic budget, agreement-adaptive retention reduced
mean teacher KL by **7.96%** relative to fixed two-position retention. It did
not beat the audio-only adaptive control, and the held-out effect changed sign
between the two test actors. The result is therefore **inconclusive**, not
negative: adaptive allocation has a promising point estimate, but this pilot
does not show that audio--text agreement is the reason for the gain.

Lower KL is better.

| Policy | Mean teacher KL | Difference from fixed |
|---|---:|---:|
| Oracle | 10.0088 | -1.2348 |
| Audio-only adaptive | 10.2950 | -0.9486 |
| Agreement-adaptive | 10.3481 | -0.8955 |
| Fixed retention | 11.2436 | 0.0000 |

Every policy used 48 acoustic positions and 696 aggregate input positions over
the 24 held-out items.

## Uncertainty

The agreement-policy contrast with fixed retention was **-0.8955 KL**, with a
recording-cluster bootstrap 95% interval of **[-1.9015, 0.1105]**. Its actor
effects were -1.9015 for actor 05 and +0.1105 for actor 06.

The agreement-policy contrast with the audio-only control was **+0.0531 KL**,
with a 95% interval of **[-0.2536, 0.4770]**. Agreement was slightly worse on
the aggregate point estimate, and neither held-out comparison excludes zero.

## What the crossed conditions show

Four acoustic positions provided almost no measured benefit when the transcript
was exact: 0.0025 KL for clean audio and 0.0011 for degraded audio. The benefit
was much larger when the transcript omitted or contradicted spoken content:

| Transcript condition | Clean audio | Degraded audio |
|---|---:|---:|
| Exact | 0.0025 | 0.0011 |
| Incomplete | 4.5444 | 1.7199 |
| Conflicting | 4.3288 | 0.8755 |

This supports the premise that acoustic positions are not equally valuable.
It does not yet establish that the current agreement features identify the
valuable cases better than audio-only features on unseen speakers.

## Evidence status

The A100 measurement stage completed all 72 crossed items and 216 candidate
forward passes in 1,814 seconds. The retained VIPER run then failed in its CPU
verification stage because the validator incorrectly treated JSON key order as
meaningful. The corrected validator recomputed the retained artifact with a
maximum numerical difference of $1.8 \\times 10^{-15}$, without repeating any
forward passes. The retained VIPER run is still terminally failed, so this is a
revalidated exploratory measurement rather than a fully accepted VIPER run.

- [Measured result](runs/ravdess_crossed_equal_budget/01M2PHRQCCHEAHXN4TGX3011E5/artifacts/measure_retention/result/result.json)
- [VIPER run record](runs/ravdess_crossed_equal_budget/01M2PHRQCCHEAHXN4TGX3011E5/resolved.yaml)
- Experiment source commit: `8ce7bbb7432e02cdbace99fb504a6f599d405a2a`
- VIPER run: `01M2PHRQCCHEAHXN4TGX3011E5`
- Hardware: NVIDIA A100-SXM4-40GB

## Decision

Do not pitch this as evidence that agreement-aware retention already works.
Pitch it as evidence for the research question: acoustic value changes sharply
with transcript completeness, an equal-budget learned allocator can improve on
uniform retention, and the next experiment must determine whether explicit
audio--text agreement adds predictive value beyond audio-only confidence.
