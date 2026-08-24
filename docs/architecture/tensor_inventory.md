# Qwen3-Omni tensor inventory

## What the inventory measures

[`tensor_manifest.json`](../../tensor_manifest.json) inventories every tensor in
the 15 safetensors shards of
`Qwen/Qwen3-Omni-30B-A3B-Instruct` at commit
`26291f793822fb6be9555850f06dfe95f2d7e695`. The generator downloaded the
pinned checkpoint outside Git on the certified L4 disk, verified every complete
shard against its Hugging Face LFS SHA-256, and computed a second SHA-256 over
each tensor's exact safetensors `data_offsets` byte range.

The manifest records source bytes, not converted GGUF bytes. Each included
tensor has a frozen `clearx.<source_name>` GGUF name, target precision, backend,
and residency policy, but its `target_tensor_sha256` remains `null` with status
`pending_phase1_conversion`. Phase 1 must replace that status with the checksum
of the emitted target bytes. Vision tensors use `removed_not_applicable` and
have no target name.

## Complete checkpoint accounting

All source tensors are BF16. Decimal GB and binary GiB are both shown because
the memory-budget packet must not treat them as interchangeable.

| Module | Tensors | Source bytes | Decimal GB | Binary GiB | Deployment disposition |
|---|---:|---:|---:|---:|---|
| Thinker | 18,867 | 61,065,293,824 | 61.065 | 56.871 | Included; matrices Q4_K_M, one-dimensional and normalization tensors BF16 |
| AuT/audio tower | 525 | 1,295,854,336 | 1.296 | 1.207 | Included; matrices Q8_0, one-dimensional and normalization tensors BF16 |
| Vision | 351 | 1,077,262,816 | 1.077 | 1.003 | Removed from the audio-only package |
| Talker | 7,951 | 6,366,052,352 | 6.366 | 5.929 | Included; matrices Q8_0, one-dimensional and normalization tensors BF16 |
| MTP/code predictor | 86 | 283,140,608 | 0.283 | 0.264 | Included in BF16 |
| Code2Wav | 230 | 432,033,154 | 0.432 | 0.402 | Included in BF16 on CPU |
| **Total** | **28,010** | **70,519,637,090** | **70.520** | **65.677** | 27,659 included; 351 removed |

The Thinker contains 18,432 routed-expert tensors totaling 57,982,058,496
source bytes. The Talker contains 7,680 routed-expert tensors totaling exactly
6,039,797,760 bytes (5.625 GiB). The deployment keeps every Talker expert in
pinned CPU memory and loads selected experts into L4 cache slots on demand.
Dense Talker tensors, routers, norms, embeddings, projections, and heads remain
on the GPU while the model speaks.

## Released architecture facts

The pinned `config.json` and observed tensor names agree on these structural
facts:

- Thinker: 48 layers, hidden size 2,048, 128 experts, eight active experts per
  token, and vocabulary size 152,064.
- AuT: 32 encoder layers, hidden size 1,280, 128 mel bins, and output dimension
  2,048.
- Vision: 27 layers at hidden size 1,152; every `thinker.visual.*` tensor is
  retained in the inventory and marked removed.
- Talker: 20 layers, hidden size 1,024, 128 experts, six active experts per
  token, a 3,072-entry primary codec vocabulary, and 16 code groups.
- MTP/code predictor: five layers, hidden size 1,024, and 15 residual codebooks
  after the primary code group, each with vocabulary size 2,048.
- Code2Wav: eight transformer layers, 16 quantizers, one semantic quantizer, a
  4,096-entry semantic codebook, 2,048-entry residual codebooks, and sliding
  window 72.

The configuration names three speaker IDs—Chelsie 2301, Ethan 2302, and Aiden
2303—but the checkpoint has no dedicated speaker tensor prefix. The manifest
therefore records speaker selection as a configuration binding rather than
inventing a nonexistent `speaker.*` tensor family.

## Target-name and placement rules

The target name is mechanically reversible: removing the `clearx.` prefix
recovers the exact source name. That decision avoids implying compatibility
with a published llama.cpp name map that does not yet cover the native Talker,
MTP, or Code2Wav conversion path.

Target precision and placement follow one deterministic rule per module:

| Source family | Target precision | Backend | Lifetime |
|---|---|---|---|
| `thinker.model.*`, `thinker.lm_head.*` | Q4_K_M matrices; BF16 1-D/norm/bias | L4 CUDA | Permanent |
| `thinker.audio_tower.*` | Q8_0 matrices; BF16 1-D/norm/bias | L4 CUDA | Listening arena |
| `thinker.visual.*` | Removed | Absent | Absent |
| `talker.model.*`, projections, and codec head | Q8_0 matrices; BF16 1-D/norm/bias | L4 CUDA | Speaking phase; all expert bytes stay in pinned CPU RAM and selected experts occupy L4 cache slots |
| `talker.code_predictor.*` | BF16 | L4 CUDA | Speaking phase |
| `code2wav.*` | BF16 | CPU | Permanent CPU residency |

These rules are conversion inputs, not measured memory results. P0-005B must
derive estimated and measured phase budgets from the tensor shapes and target
types, while Phase 1 conversion must establish actual GGUF byte counts and
target tensor checksums.

## Reproduction and validation

With the pinned model files in a non-Git directory:

```bash
python tools/inventory/generate_tensor_manifest.py /path/to/model tensor_manifest.json
python tools/inventory/validate_tensor_manifest.py tensor_manifest.json
```

The validator rejects a missing module, tensor, shard join, shape, dtype,
stride, source checksum, target name, backend, residency policy, or
reconciliation total. It also rejects a non-null target checksum before Phase 1
conversion, preventing this Phase 0 inventory from overclaiming emitted GGUF
evidence.
