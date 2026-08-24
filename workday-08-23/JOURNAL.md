# Workday journal — August 23, 2026

## Day charter

End the workday with the PyTorch fixture boundary defined, the A100 execution path proven, and tomorrow's first implementation decision preserved for review.

## Capacity

The workday is closed. Overnight A100 and implementation work is deferred. Tomorrow's plan covers three hours, including 30 minutes for debugging, review, and short breaks; later work remains uncommitted until the generator design is visible.

## End-of-day record

The project can load the complete pinned Qwen3-Omni model on an A100 and execute Thinker, Talker, MTP, and Code2Wav. The successful probe placed Thinker layers 0–20, Talker, and Code2Wav on the A100; Thinker layers 21–47 remained on CPU. It produced one complete codec frame and a `[1, 1, 1365]` float32 waveform. The probe measured 36,511,910,400 bytes of peak GPU allocation. The stored evidence is the [A100 load probe](../runs/P0-003B/20260823T161735Z-a100/load_probe.json).

Automatic device placement failed because generation encountered a meta-device tensor while the wrapper expected CUDA. The explicit placement above is therefore the validated reference path for fixture generation. This A100 result proves that the pinned PyTorch path can execute. Native-runtime memory use and latency on the 24 GB L4 remain unmeasured.

The [fixture harness](../tools/reference/fixture_harness.py) defines 154 boundary-and-case pairs that the PyTorch model can produce. The future native runtime owns eight scheduler and bounded-context pairs because PyTorch lacks the corresponding operations. The [fixture validator test](../tests/orchestration/test_fixture_validation.py) checks that partition.

The generator and `tests/fixtures/reference/manifest.json` remain absent. The lifecycle file labels fixture generation as `running`; [the lifecycle history](../orchestration/state.json) records only the start of the work packet. Colab reports zero active sessions. In operational terms, the probe finished and fixture capture awaits implementation.

We chose `[1, 16, 601]` as the isolated Code2Wav input. The first 300 codec frames produce the first full chunk. Frames 300–599 produce a steady full chunk after Code2Wav prepends frames 275–299 as left context. Frame 600 creates a final partial chunk after Code2Wav prepends frames 575–599. The pinned Transformers implementation computes the available context, decodes the context-inclusive slice, and removes the waveform samples produced by the repeated context.

The pair session ended with the worktree unchanged. Tomorrow's implementation should preserve upstream's explicit `context_size` calculation; the shorter `max(0, frame_start - 25)` expression produces the same slice start while hiding the value used to trim the decoded waveform.

## Tomorrow's project briefing

Begin by encoding and testing the exact Code2Wav chunk calculation. That small contract will establish what the generator must capture for the first, steady, and final-partial waveform boundaries.

CleaRx aims to run Qwen3-Omni as a native waveform-to-waveform system in one C++ process on a single 24 GB NVIDIA L4. The native runtime must preserve the pinned model's behavior while bounding live memory independently of conversation length. The [implementation proposal](../docs/proposal.tex) defines that target and hardware constraint.

The current work builds the answer key for the native port. One fixed input follows this path:

```text
decoded PCM
-> audio preprocessing and encoder
-> Thinker
-> Talker
-> one primary plus 15 residual codec values per frame
-> Code2Wav
-> waveform
```

The fixture generator will save the input and output at each declared boundary. Later, when the native C++ runtime differs from PyTorch, the first mismatching fixture will identify the component and operation that introduced the error.

The project has proven the complete PyTorch path and defined the required observations. The first missing result is a frozen generator input contract. That contract must identify the exact decoded JFK PCM bytes, sample rate, sample count, PCM digest, and resulting model dimensions. It must also identify the exact 601-frame, 16-codebook Code2Wav array and its digest. The [evaluation corpus manifest](../evaluation_corpus_manifest.json) identifies the source recording as `commons-jfk-inaugural`; its decoded PCM identity still needs freezing.

Tomorrow's operations proceed in this order:

1. Encode the upstream Code2Wav bounds calculation and verify the three expected slices.
2. Decode the pinned JFK recording through one specified path and record the exact PCM identity that the generator will consume.
3. Implement one representative observation hook and verify that it captures the named input and output while preserving model execution.
4. Review that pattern before applying it to the remaining PyTorch-observable boundaries.

The reviewed input contract and representative hook will make the broader generator implementation mechanical. Once the generator and local validation pass, an A100 run can create the 154 reference fixtures using only reviewed architectural decisions.

## Tomorrow's execution plan

| Budget | Work block | Deliverable | Done condition |
| ---: | --- | --- | --- |
| 35 min | Code2Wav contract | Chunk descriptor and focused tests | The test proves slices `0:300`, `275:600`, and `575:601`, including the context size used for waveform trimming. |
| 60 min | JFK input freeze | Deterministic decode procedure and PCM identity record | The source digest, decoder settings, sample format, sample rate, sample count, PCM digest, and model input dimensions are explicit and reproducible. |
| 55 min | Representative capture | One observation hook plus its artifact-writing test | The hook records one declared boundary, preserves the returned model value, and produces shape, dtype, and digest metadata that pass the focused test. |
| 30 min | Reserved slack | Debugging, review, and short breaks | The primary deliverable retains its acceptance checks even if one earlier block overruns. |

The primary outcome is a reviewed generator foundation containing both frozen input contracts and one proven capture pattern.

## Replan rule

If repeated JFK decodes produce different PCM identities after the first 95 minutes, stop before hook implementation. Preserve the passing Code2Wav contract, record the decoder discrepancy, and make PCM reproduction the next session's first action.

## Deferred tomorrow

Tomorrow's first session excludes bulk generation of all 154 fixtures, native C++ module implementation, L4 deployment measurements, bounded-context reconstruction, and an unattended A100 run. Each item depends on the reviewed generator foundation.

## Shutdown

Completed: the reference model path runs on the A100, the fixture boundary matrix is defined, and the Code2Wav input length has a concrete rationale.

Blocked: fixture generation needs exact frozen inputs and an implemented capture path.

First action tomorrow: write the upstream-shaped `context_size` and chunk-slice calculation, add the focused 600-versus-601 tests, and review the result before proceeding to JFK PCM.

## Sources

- [Native Qwen3-Omni waveform runtime proposal](../docs/proposal.tex)
- [Reference fixture contract](../docs/architecture/fixture_contract.md)
- [Fixture harness](../tools/reference/fixture_harness.py)
- [Fixture-generation work packet](../orchestration/work_packets/P0-003B.yaml)
- [A100 load probe](../runs/P0-003B/20260823T161735Z-a100/load_probe.json)
- [Evaluation corpus manifest](../evaluation_corpus_manifest.json)
- Hugging Face Transformers, [`Qwen3OmniMoeCode2Wav.chunked_decode` at pinned commit `7d9754a`](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L3505-L3515)
