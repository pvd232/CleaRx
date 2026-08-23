# CleaRx deployment acceptance

## Certified execution identity

[`deployment_acceptance.json`](../deployment_acceptance.json) binds release
acceptance to host profile `clearx-mantra-g2-l4-v1`: the NVIDIA L4 with GPU UUID
`GPU-e2d6d9a2-3e87-2a3c-73fb-589e52f39d0e` in the `mantra-g2-spot`
`g2-standard-12` instance in `us-west1-a`. The record hashes the exact host
profile bytes. Recreated infrastructure is not the certified host until a new
profile and acceptance record name its observed GPU UUID, machine type, zone,
driver, CUDA toolkit, memory, and measurement contract.

An A100 Colab session may generate reference fixtures and exploratory
comparisons. The local M1 environment may run validators and orchestration
tests. Neither environment can satisfy the L4 release-performance gate.

## Frozen evidence join

The acceptance record contains exact SHA-256 joins to four inputs: the L4 host
profile, evaluation contract, corpus manifest, and threshold record. The
validator recomputes every digest from repository bytes. Editing any input
severs the deployment record and requires a new version; a filename match is
insufficient.

## Named workloads

`codec-correctness` runs the `greedy-correctness` sampling profile over every
corpus stratum and compares every primary and residual codec token with the
pinned Transformers oracle.

`release-quality-paired` runs native and oracle members for every declared
item, seed, and `release-quality` sampling tuple. It owns the semantic and
acoustic non-inferiority measurements.

`bounded-context-paired` compares a candidate context policy with its
predeclared baseline using identical pairs, output limits, host, and byte
budget. It owns the material-improvement measurement.

`l4-latency` runs five warm-ups and 30 recorded repetitions at batch size one
and concurrency one on the certified L4. It owns audio-start latency, real-time
factor, and underflow measurements.

## Release gates

Each gate resolves to the matching object under `thresholds` in
[`benchmark_thresholds.json`](../benchmark_thresholds.json):

- correctness: exact codec-token match under `codec-correctness`;
- semantic: paired ASR word-error-rate delta under `release-quality-paired`;
- acoustic: paired speaker-embedding cosine delta under `release-quality-paired`;
- material improvement: paired task-success delta under `bounded-context-paired`;
- latency: audio-start p95, real-time-factor p95, and zero underflows under
  `l4-latency`.

The threshold record owns all numerical margins, confidence rules,
repetitions, and paired keys. This deployment record only names the consuming
workload, sampling profile, and corpus slices. Results observed on an alternate
host remain exploratory and cannot replace a failed or missing certified-host
measurement.
