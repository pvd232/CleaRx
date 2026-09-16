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

---

## Retrospective technical reconstruction

This reconstruction records the system understanding and evidence produced during the August 23 workday. Its execution claims come only from repository artifacts saved that day.

### The dependency chain

CleaRx follows one dependency chain:

```text
freeze the reference model's behavior
-> implement native modules that reproduce it
-> connect those modules into streaming speech
-> fit and run the correct system on one L4
-> add bounded conversational memory without losing correctness
```

The August 23 work established most of the first link. Production native model implementation remained ahead.

### The target system

CleaRx aims to run Qwen3-Omni as one native C++ process that accepts speech and returns speech. One NVIDIA L4 with 24 GB of VRAM is the deployment target. Python and external text-to-speech systems remain development tools outside deployed inference. The [runtime proposal](../docs/proposal.tex) defines those constraints.

The model transforms one turn through this path:

```text
input PCM
-> audio preprocessing and AuT encoder
-> Thinker
-> Talker
-> primary codec value
-> fifteen ordered MTP residual values
-> complete sixteen-value codec frame
-> Code2Wav
-> output PCM
```

AuT converts PCM into audio representations. Its final 1,280-dimensional states pass through a 2,048-dimensional projection into the Thinker. The Thinker performs multimodal reasoning and produces the representation that conditions speech generation. The 20-layer Talker routes each position through six of 128 experts and produces the primary codec value. MTP generates 15 residual codec values in order. Code2Wav converts the complete 16-value frames into waveform samples.

A native scheduler will surround that numerical path. The scheduler must accept incoming PCM, close a turn, start generation, commit completed waveform chunks, process interruptions, cancel queued work, release buffers, and rebuild bounded conversational context.

### Why reference fixtures come first

The native baseline supplies useful loading, backend, convolution, attention, and mixture-of-experts machinery. CleaRx still owns the complete Talker, MTP, Code2Wav, conversion, and scheduling path described in the [runtime gap matrix](../docs/architecture/runtime_gap_matrix.md).

A wrong final waveform could originate in audio preprocessing, Thinker-to-Talker conditioning, expert routing, primary codec prediction, any MTP residual step, Code2Wav chunking, or scheduler ordering. End-to-end audio reveals the final failure while leaving its first faulty operation unknown.

The [fixture contract](../docs/architecture/fixture_contract.md) therefore saves reference arrays at each meaningful handoff. Each fixture identifies its input and output arrays, shapes, dtypes, comparison rule, model revision, generator revision, seed, and content hashes. A later native test can then report the first boundary that diverges.

The boundary matrix contains 162 boundary-and-case pairs. PyTorch can produce 154 of them. The future native scheduler owns the remaining eight because they describe interruption, cancellation, scheduler transitions, or bounded-context reconstruction. The [fixture validation tests](../tests/orchestration/test_fixture_validation.py) enforce that evidence-origin split.

### Each machine answers a different question

| Environment | Role | Evidence it can produce |
| --- | --- | --- |
| Local Mac | Control plane | Schema validation, orchestration, fixture comparison, small builds, and CPU scheduler tests |
| Colab A100 | PyTorch oracle | Full reference-model outputs and boundary fixtures |
| `mantra-g2` L4 | Deployment target | Measured VRAM, PCIe transfer behavior, expert residency, latency, underflow, and stability |

The A100 defines what the pinned reference model computes. The L4 must later establish whether the native implementation fits and runs in real time. The [memory model](../docs/architecture/memory_model.md) remained a planning model on August 23 because every `measured_bytes` field was still empty.

### The phased program

The phases answer their questions in dependency order:

| Phase | Question | August 23 position |
| --- | --- | --- |
| Bootstrap | Can immutable packets, artifacts, and state transitions control the work? | Complete |
| Phase 0 | What must the native runtime reproduce, load, own, measure, and reject? | Fixture generation remained open |
| Phase 1 | Can each native module reproduce its reference arrays? | Production implementation remained pending |
| Phase 2 | Can the correct modules stream speech together? | Planned |
| Phase 3 | Can the correct streaming system fit and run in real time on one L4? | Planned |
| Phase 4 | Can live conversational state stay bounded while retaining useful history? | Planned |

Phase 0 pins upstream revisions, inventories model tensors, defines fixture boundaries, freezes evaluation inputs, and establishes the native interface contract. Phase 1 implements conversion, AuT, Talker, MTP, Code2Wav, and scheduler tracks independently against stored arrays. Phase 2 connects the verified modules into short-context streaming. Phase 3 measures the native system on the L4. Phase 4 changes history retention only after the short-context runtime is correct.

### One long speech turn

The `commons-jfk-inaugural` corpus item provides the long Thinker case. The [evaluation corpus manifest](../evaluation_corpus_manifest.json) identifies the compressed recording, its SHA-256 digest, its 840.228345-second descriptive duration, its prompt, and a 384-token output cap.

The generator must convert that recording into one exact PCM input and record the PCM bytes, sample format, channels, sample rate, sample count, digest, AuT frame count, and resulting Thinker dimensions. The 384-token cap applies after input processing and leaves the full audio intact.

The reference path then records these transformations:

```text
PCM
-> log-mel input [1, 128, aut_frames]
-> AuT output [aut_frames, 1280]
-> AuT projection [aut_frames, 2048]
-> Thinker hidden state [1, thinker_positions, 2048]
-> Talker conditioning [1, talker_positions, 1024]
-> primary plus fifteen residual codec values [1, 16, codec_frames]
-> Code2Wav waveform chunks
```

Every discrete codec value requires exact agreement. A fixture with a different codec sequence fails even when its waveform similarity passes.

### Why the isolated Code2Wav input has 601 frames

The isolated Code2Wav fixture is one exact, hashed `int64` tensor with shape `[1, 16, 601]`. The pinned helper decodes 300 new frames per call and repeats as many as 25 earlier frames as left context.

The 601-frame tensor produces three decoder windows:

```text
input 0:300   -> first 300 new frames
input 275:600 -> 25 context frames plus 300 new frames
input 575:601 -> 25 context frames plus one final frame
```

Code2Wav trims the waveform produced from repeated context before concatenating each new chunk. A 301-frame tensor reaches a first chunk and a partial ending. A 600-frame tensor reaches the first and steady chunks. Frame 601 creates the smallest single input that reaches the first, steady, and final-partial states.

The model's 72-frame attention window serves a separate internal purpose. The 601-frame decision comes from the public 300-frame decode boundary and its 25-frame overlap.

These frozen lengths serve as test inputs. The eventual native interface must accept variable-length sequences while fixture descriptors retain the observed dimensions.

### Evidence completed on August 23

The tensor inventory reconciled 28,010 tensors across 15 checkpoint shards. The fixture schema, comparison profiles, provenance joins, boundary matrix, and evidence-origin split were executable. The [A100 preflight](../runs/P0-003B/20260823T155524Z-a100/preflight.json) verified the pinned model and Transformers revisions, checkpoint access, 20 Talker layers, 15 residual codebooks, and 16 Code2Wav quantizers.

Automatic placement loaded the model and then encountered a meta-device tensor during generation. The explicit map placed Thinker layers 0–20, Talker, and Code2Wav on the A100 while Thinker layers 21–47 remained in CPU memory. The [saved A100 probe](../runs/P0-003B/20260823T161735Z-a100/load_probe.json) completed Thinker, Talker, MTP, and Code2Wav. It produced one codec frame and a `[1, 1, 1365]` float32 waveform in 26.692 seconds with 36,511,910,400 bytes of peak GPU allocation.

That probe established executable reference-model access. L4 fit, PCIe cost, native latency, scheduler behavior, and bounded-context reconstruction still required their own implementations and measurements.

### Handoff at the end of the workday

The orchestration ledger marked P0-003B as `running`, while the stored evidence showed a completed probe and an absent fixture generator. The practical next sequence was:

```text
freeze exact JFK PCM and exact [1, 16, 601] codec values
-> implement the reference-fixture generator
-> run it on the A100
-> download and validate 154 model-reference fixture pairs
-> complete P0-003B
-> freeze the native module interfaces
-> begin independent native module work
```

The workday ended before generator implementation so the frozen inputs and first observation hook could receive owner review. The project had built the answer-key contract; the next workday would begin manufacturing the answer key itself.
