# Workday journal — September 15, 2026

## Day charter

Publish the first executable Code2Wav fixture-input contract, record the same frozen-input requirements in the v2 proposal, and preserve the August 23 technical reconstruction.

## Capacity

This workday contains one bounded implementation and publication cycle. It includes local validation, proposal rendering, journal updates, and Git synchronization. A100 execution and JFK audio decoding remain outside this cycle.

## Execution plan

| Budget | Work block | Deliverable | Done condition |
| ---: | --- | --- | --- |
| 45 min | Code2Wav contract | Importable generator foundation | The module defines the `[1, 16, 601]` input shape, computes upstream-shaped decode windows, rejects zero frames, and carries a runnable docstring example with expected output. |
| 35 min | Proposal update | Updated v2 source and PDF | The parity-fixture section names the exact JFK PCM fields and exact hashed Code2Wav tensor; LaTeX compilation and PDF rendering pass. |
| 30 min | Historical record | August 23 reconstruction and September 15 progress journal | Both dated journals identify completed evidence, missing work, and the next operation. |
| 20 min | Reserved slack | Validation and Git synchronization | The complete scoped diff passes its observing checks and reaches the public upstream repository. |

## Progress

The repository's GitHub remote, [`pvd232/CleaRx`](https://github.com/pvd232/CleaRx), already has public visibility.

The new [reference generator foundation](../../proposals/full_plan/8-23/tools/reference/generate_reference_fixtures.py) defines one `Code2WavDecodeWindow` for each context-inclusive decoder call. Its module docstring contains the exact verification command and the three expected records. The implementation preserves the pinned helper's explicit left-context count because the later waveform trim consumes that value.

Local validation established these behaviors:

- Ruff applied its safe fixes and reported the final source clean and formatted.
- Pyright reported zero errors and warnings.
- The documented command produced windows `0:300`, `275:600`, and `575:601` with context counts `0`, `25`, and `25`.
- A focused failure check confirmed that a zero-frame input raises `ValueError`.
- The direct-prose checker accepted every Python docstring.

The [v2 proposal source](../../proposals/full_plan/9-15/talker.proposal.v2.tex) now places the frozen inputs inside its cross-runtime parity-fixture section. The Thinker long-context fixture must freeze the JFK decode path, exact PCM bytes, sample format, channels, sample rate, sample count, SHA-256 digest, and model-input dimensions. The Code2Wav fixture must freeze every value and the digest of one `int64 [1,16,601]` tensor.

The rebuilt [v2 proposal PDF](../../proposals/full_plan/9-15/talker.proposal.v2.pdf) contains 11 letter-sized pages. LaTeX compiled successfully, text extraction found both new requirements, and a rendered review found consistent margins, readable type, intact equations, and complete content inside every page boundary.

The publication set includes the original and v2 proposal sources and PDFs. LaTeX `.aux`, `.log`, `.fls`, `.fdb_latexmk`, and `.out` files remain local build products. `.DS_Store` and the pre-existing `tmp/` directory also remain outside Git.

## Replan rule

If final diff inspection reveals generated files, secrets, or unrelated content in the publication set, remove those paths from the staging set and publish only the validated generator, journals, proposal sources, and proposal PDFs.

## Deferred today

The exact JFK PCM bytes and digest remain unfrozen. The 601-frame Code2Wav tensor still needs exact codec values and a digest. The generator still needs model loading, observation hooks, artifact serialization, the required command-line interface, and A100 execution for all 154 PyTorch-observable pairs.

## Shutdown

Completed: the Code2Wav window contract is executable and locally validated; the v2 proposal carries the same frozen-input requirements; the August 23 architecture explanation is preserved as a dated journal.

Next action: define one deterministic decoder and resampling path for `commons-jfk-inaugural`, run it twice, and freeze the identical PCM identity before adding the first model observation hook.
