# Pinned native runtime gap matrix

## Evidence boundary

This audit freezes six upstream repositories in `upstream.lock`. “Supported”
means the pinned native source contains the loading path and execution graph for
the named Qwen3-Omni surface. A compatible loader, a related Qwen model, or a
published tensor file does not establish an execution graph by itself.

The current native baseline is `llama.cpp`
[`6657ded4faa3b8450221119fc6b4d002e35104a2`](https://github.com/ggml-org/llama.cpp/tree/6657ded4faa3b8450221119fc6b4d002e35104a2).
The PyTorch oracle is Transformers tag `v5.2.0`, commit
[`7d9754a05193eb79b1d86aa744b622b8068008cd`](https://github.com/huggingface/transformers/tree/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3_omni_moe).

## Runtime surface verdicts

| Surface | Pinned native evidence | Verdict | CleaRx consequence |
|---|---|---|---|
| Main GGUF loading | `include/llama.h` exports [`llama_model_load_from_file`](https://github.com/ggml-org/llama.cpp/blob/6657ded4faa3b8450221119fc6b4d002e35104a2/include/llama.h#L507). | Supported generic loader | Reuse the loader and add versioned CleaRx metadata rejection around output-module artifacts. |
| Qwen3-Omni audio input | `tools/mtmd/models/qwen3a.cpp` defines [`clip_graph_qwen3a::build`](https://github.com/ggml-org/llama.cpp/blob/6657ded4faa3b8450221119fc6b4d002e35104a2/tools/mtmd/models/qwen3a.cpp#L3); the Qwen3 audio preprocessor is selected in `tools/mtmd/mtmd.cpp`. | Supported for multimodal input | Treat AuT/Thinker input as the starting native baseline, subject to boundary fixtures. |
| Final partial audio chunk | The pinned audio path contains the merged padded-tail fix; [PR 22770](https://github.com/ggml-org/llama.cpp/pull/22770) records reproduction with the released Qwen3-Omni GGUF. | Supported after the fix | Retain a final-partial-chunk fixture because this boundary regressed previously. |
| Thinker text/MoE execution | The published main GGUF loads through existing Qwen3 MoE runtime paths; the official GGUF repository documents `llama-server` usage. | Supported release path; numerical equivalence still unverified | P0-003B must compare hidden/logit boundaries before CleaRx treats the path as its correctness oracle. |
| Talker | No Qwen3-Omni Talker class, tensor namespace, graph builder, or scheduler exists in the pinned native tree. The oracle defines [`Qwen3OmniMoeTalkerForConditionalGeneration`](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L3066). | Missing | Phase 1 needs a native Talker graph, router/expert execution, sampling loop, and fixtures. |
| Residual codec predictor (MTP) | The oracle calls the Talker code predictor for residual codebooks. Native `llama.cpp` contains MTP implementations for other architectures, not the Qwen3-Omni five-layer codec predictor. | Missing for Qwen3-Omni | Phase 1 needs an Omni-specific residual loop and 15 residual-codebook boundary fixtures. |
| Code2Wav | The oracle defines [`Qwen3OmniMoeCode2Wav`](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L3785). Native source has [`clip_graph_qwen3tts_gen::code2wav::decode`](https://github.com/ggml-org/llama.cpp/blob/6657ded4faa3b8450221119fc6b4d002e35104a2/tools/mtmd/models/qwen3tts-gen.cpp#L544) for Qwen3-TTS, a different model family. | Missing for Qwen3-Omni | Reuse verified GGML operators and state-buffer patterns only after tensor/layout comparison; do not treat Qwen3-TTS output as semantic compatibility. |
| Thinker–Talker coordination | The oracle method [`Qwen3OmniMoeForConditionalGeneration.generate`](https://github.com/huggingface/transformers/blob/7d9754a05193eb79b1d86aa744b622b8068008cd/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L3951) prepares Talker inputs from Thinker hidden states, runs Talker generation, and decodes codes. No corresponding native Omni coordinator exists. | Missing | The scheduler packet must own queues, phase transitions, cancellation, and the hidden-state handoff. |
| Output-module conversion | The pinned `convert_hf_to_gguf.py` contains no `Qwen3Omni` model converter. The published GGUF repository says it used `convert_hf_to_gguf.py`, but that statement does not define current output-tensor conversion behavior. | Missing/reproducibility gap | Phase 1 conversion must enumerate Talker, code predictor, speaker, and Code2Wav tensors and version their metadata. |
| Optional Whisper anchor source | `whisper.cpp` [`233fe1fc9b48a09e361d3594520838ca266537fe`](https://github.com/ggml-org/whisper.cpp/tree/233fe1fc9b48a09e361d3594520838ca266537fe) exports [`whisper_full`](https://github.com/ggml-org/whisper.cpp/blob/233fe1fc9b48a09e361d3594520838ca266537fe/include/whisper.h#L603) and token timestamps through [`whisper_full_get_token_t0`](https://github.com/ggml-org/whisper.cpp/blob/233fe1fc9b48a09e361d3594520838ca266537fe/include/whisper.h#L675). | Supported API, deferred consumer | Keep Whisper outside the native baseline; Phase 4 may consume transcript/timestamp records under matched-budget experiments. |

## Relevant native APIs and operators

| Need | Pinned symbol | What the symbol establishes |
|---|---|---|
| Device and tensor placement | `llama_model_params.devices`, `llama_model_params.tensor_split`, and `llama_split_mode` in [`include/llama.h`](https://github.com/ggml-org/llama.cpp/blob/6657ded4faa3b8450221119fc6b4d002e35104a2/include/llama.h#L307) | The loader can choose devices and split model tensors. CleaRx still needs per-output-module placement rules. |
| Quantized KV configuration | `llama_context_params.type_k` and `type_v` in [`include/llama.h`](https://github.com/ggml-org/llama.cpp/blob/6657ded4faa3b8450221119fc6b4d002e35104a2/include/llama.h#L381) | The context API accepts K/V cache types, including Q8-capable GGML types. Compatibility must be measured per model path. |
| Backend scheduling | [`ggml_backend_sched_new`](https://github.com/ggml-org/llama.cpp/blob/6657ded4faa3b8450221119fc6b4d002e35104a2/ggml/include/ggml-backend.h#L319) | GGML can assign one graph across registered backends. It does not define CleaRx phase ownership. |
| Async expert transfer | [`ggml_backend_tensor_copy_async`](https://github.com/ggml-org/llama.cpp/blob/6657ded4faa3b8450221119fc6b4d002e35104a2/ggml/include/ggml-backend.h#L116) | A backend-owned asynchronous copy primitive exists for the later expert-cache spike. |
| Memory instrumentation | `ggml_backend_dev_memory`, `ggml_backend_sched_get_buffer_size`, `llama_model_size`, and `llama_perf_context` | The runtime exposes device free/total memory, scheduler buffer bytes, model bytes, and timing counters. P0-005 defines the sampling and phase-attribution contract. |
| Routed expert matmul | `ggml_mul_mat_id`, `ggml_top_k`, and `ggml_argsort_top_k` | Required MoE selection and per-expert matrix operations exist; Qwen3-Omni Talker tensor layout remains to be bound. |
| Code2Wav building blocks | `ggml_conv_1d`, `ggml_conv_1d_dw`, `ggml_conv_transpose_1d`, RMSNorm, SiLU, RoPE, and attention matmuls | The pinned CUDA backend implements the required operator families. Exact padding, persistent state, and numerical tolerances remain fixture obligations. |

## Pinned release identities

- Official Qwen usage repository: `e4235853125589c789f06a2dd83e9f4126df5e9d`,
  [`README.md`](https://github.com/QwenLM/Qwen3-Omni/blob/e4235853125589c789f06a2dd83e9f4126df5e9d/README.md).
  Its examples call `Qwen3OmniMoeForConditionalGeneration.from_pretrained`;
  `model.disable_talker` is the documented text-only memory reduction.
- Official model repository: `26291f793822fb6be9555850f06dfe95f2d7e695`,
  [`config.json`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/26291f793822fb6be9555850f06dfe95f2d7e695/config.json).
  The frozen configuration contains `talker_config`, `code2wav_config`, and
  `enable_audio_output`.
- Published GGUF repository: `6e35a28f4a19b18730f8949b0c579c6429649ab8`,
  [`README.md`](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/blob/6e35a28f4a19b18730f8949b0c579c6429649ab8/README.md).
  The `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf` LFS object is
  `d9e2876556e7873e02c0359f832432ee2d67ab7dd0cee3efe0f77fd7a1f4dd85`
  and declares 18,557,053,952 bytes.

## Frozen toolchains

Local contract and CPU checks use Apple clang 21.0.0, CMake 3.30.5, Ninja
1.13.2, and Python 3.12.14 from the `clearx` environment. L4 CUDA work targets
the image `common-cu129-ubuntu-2204-nvidia-580-v20260818`, Ubuntu 22.04.5,
GCC 12.3.0, CUDA 12.9 (`nvcc` 12.9.41), and NVIDIA driver 580.173.02.
The toolkit executable is `/usr/local/cuda/bin/nvcc`; the base image does not
place that directory on the default SSH `PATH`.
