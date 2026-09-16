#!/usr/bin/env python3
"""Run the three-condition audio-memory smoke on one A100 and print its result."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import transformers
from tools.research.audio_memory_fusion import (
    CONTRACT_PACKAGE_SHA256,
    EXPERIMENT_ID,
    FROZEN_SMOKE_IDENTITY,
    float_array_sha256,
    fuse_memory_vectors,
    mean_pool_ordered,
    validate_smoke_result,
)
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RESULT_PREFIX = "CLEARX_AUDIO_MEMORY_SMOKE_RESULT="


def download_audio(destination: Path) -> bytes:
    """Download the frozen waveform and reject bytes with another digest."""
    with urllib.request.urlopen(
        FROZEN_SMOKE_IDENTITY.audio_url, timeout=60
    ) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != FROZEN_SMOKE_IDENTITY.audio_sha256:
        raise RuntimeError("downloaded audio digest differs from the frozen identity")
    destination.write_bytes(content)
    return content


def move_inputs_to_device(
    inputs: Any, device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    """Move processor tensors to the Thinker's first device without casting IDs."""
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def last_logits(output: Any) -> torch.Tensor:
    """Return one CPU float32 vector for the final input position."""
    logits = output.logits[0, -1].detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Thinker returned non-finite logits")
    return logits


def comparison(teacher: torch.Tensor, candidate: torch.Tensor) -> tuple[float, bool]:
    """Return teacher-to-candidate KL divergence and top-token agreement."""
    teacher_log = torch.log_softmax(teacher, dim=-1)
    candidate_log = torch.log_softmax(candidate, dim=-1)
    kl = torch.sum(torch.exp(teacher_log) * (teacher_log - candidate_log)).item()
    if not np.isfinite(kl) or kl < 0:
        raise RuntimeError("comparison produced an invalid KL divergence")
    return float(kl), bool(torch.argmax(teacher) == torch.argmax(candidate))


def condition_record(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    alpha: float | None,
    input_positions: int,
    memory_positions: int | None,
    teacher: torch.Tensor | None = None,
) -> dict[str, object]:
    """Build the persisted observation for one Thinker condition."""
    array = logits.numpy()
    top_token_id = int(np.argmax(array))
    record: dict[str, object] = {
        "alpha": alpha,
        "input_positions": input_positions,
        "memory_positions": memory_positions,
        "vocabulary_size": int(array.shape[0]),
        "top_token_id": top_token_id,
        "top_token_text": tokenizer.decode([top_token_id]),
        "logits_sha256": float_array_sha256(array),
    }
    if teacher is not None:
        kl, agreement = comparison(teacher, logits)
        record["teacher_kl"] = kl
        record["teacher_top1_agreement"] = agreement
    return record


def audio_span(
    input_ids: torch.Tensor, thinker: Any, tokenizer: Any
) -> tuple[int, int, torch.Tensor]:
    """Return the complete audio-tag span and its contiguous placeholder positions."""
    ids = input_ids[0]
    positions = torch.nonzero(
        ids == thinker.config.audio_token_id, as_tuple=False
    ).flatten()
    if positions.numel() < 1:
        raise RuntimeError("processor output contains no audio positions")
    expected = torch.arange(positions[0], positions[-1] + 1, device=positions.device)
    if not torch.equal(positions, expected):
        raise RuntimeError("audio positions must form one contiguous interval")
    bos_id = tokenizer.convert_tokens_to_ids(tokenizer.audio_bos_token)
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.audio_eos_token)
    start = int(positions[0]) - 1
    stop = int(positions[-1]) + 2
    if start < 0 or stop > ids.shape[0]:
        raise RuntimeError("audio tag span exceeds the processor sequence")
    if int(ids[start]) != bos_id or int(ids[stop - 1]) != eos_id:
        raise RuntimeError("audio positions lack their declared boundary tokens")
    return start, stop, positions


def main() -> int:
    """Execute the frozen full-audio, transcript-only, and addition conditions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if _GIT_COMMIT.fullmatch(args.git_commit) is None:
        raise ValueError("--git-commit must be a full lowercase commit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_properties = torch.cuda.get_device_properties(0)
    if "A100" not in device_properties.name:
        raise RuntimeError(f"requested A100, received {device_properties.name}")

    torch.manual_seed(FROZEN_SMOKE_IDENTITY.seed)
    torch.cuda.manual_seed_all(FROZEN_SMOKE_IDENTITY.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model_started = time.monotonic()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        FROZEN_SMOKE_IDENTITY.model_repository,
        revision=FROZEN_SMOKE_IDENTITY.model_commit,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "38GiB", "cpu": "76GiB"},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.disable_talker()
    model.eval()
    thinker = model.thinker
    model_load_seconds = time.monotonic() - model_started
    gc.collect()
    torch.cuda.empty_cache()

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        FROZEN_SMOKE_IDENTITY.model_repository,
        revision=FROZEN_SMOKE_IDENTITY.model_commit,
    )
    with tempfile.TemporaryDirectory(prefix="clearx-smoke-audio-") as directory:
        audio_path = Path(directory) / "input.wav"
        download_audio(audio_path)
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": FROZEN_SMOKE_IDENTITY.prompt},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=prompt,
            audio=[audio],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

    embedding_layer = thinker.get_input_embeddings()
    first_device = embedding_layer.weight.device
    prepared = move_inputs_to_device(inputs, first_device, model.dtype)
    with torch.inference_mode():
        full_output = thinker(
            **prepared,
            use_cache=False,
            return_dict=True,
            use_audio_in_video=False,
        )
        teacher_logits = last_logits(full_output)
        audio_output = thinker.get_audio_features(
            prepared["input_features"],
            feature_attention_mask=prepared["feature_attention_mask"],
            return_dict=True,
        )
    audio_features = (
        audio_output.last_hidden_state.detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    if audio_features.ndim != 2:
        raise RuntimeError(
            "Audio Tower output must have shape [positions, hidden_size]"
        )

    input_ids = prepared["input_ids"]
    span_start, span_stop, audio_positions = audio_span(
        input_ids,
        thinker,
        processor.tokenizer,
    )
    if audio_features.shape[0] != int(audio_positions.numel()):
        raise RuntimeError("Audio Tower vectors do not match processor audio positions")
    transcript_ids = processor.tokenizer(
        FROZEN_SMOKE_IDENTITY.transcript,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(first_device)
    transcript_length = int(transcript_ids.shape[1])
    compressed_ids = torch.cat(
        [input_ids[:, :span_start], transcript_ids, input_ids[:, span_stop:]],
        dim=1,
    )
    compressed_attention = torch.ones_like(compressed_ids)
    base_embeddings = embedding_layer(compressed_ids)
    memory_slice = slice(span_start, span_start + transcript_length)
    text_vectors = (
        base_embeddings[0, memory_slice]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    pooled_audio = mean_pool_ordered(audio_features, transcript_length)

    conditions: dict[str, dict[str, object]] = {
        "full_audio": condition_record(
            teacher_logits,
            processor.tokenizer,
            alpha=None,
            input_positions=int(input_ids.shape[1]),
            memory_positions=None,
        )
    }
    for name, alpha in (("transcript_only", 0.0), ("voxzip_addition", 1.0)):
        fused = fuse_memory_vectors(pooled_audio, text_vectors, alpha)
        variant_embeddings = base_embeddings.clone()
        variant_embeddings[0, memory_slice] = torch.from_numpy(fused).to(
            device=first_device,
            dtype=variant_embeddings.dtype,
        )
        with torch.inference_mode():
            output = thinker(
                input_ids=compressed_ids,
                inputs_embeds=variant_embeddings,
                attention_mask=compressed_attention,
                use_cache=False,
                return_dict=True,
                use_audio_in_video=False,
            )
        logits = last_logits(output)
        conditions[name] = condition_record(
            logits,
            processor.tokenizer,
            alpha=alpha,
            input_positions=int(compressed_ids.shape[1]),
            memory_positions=transcript_length,
            teacher=teacher_logits,
        )
        del output, logits, variant_embeddings
        torch.cuda.empty_cache()

    result: dict[str, object] = {
        "schema_version": "1.0.0",
        "experiment_id": EXPERIMENT_ID,
        "source": {
            "git_commit": args.git_commit,
            "contract_package_sha256": CONTRACT_PACKAGE_SHA256,
        },
        "identity": FROZEN_SMOKE_IDENTITY.to_mapping(),
        "environment": {
            "device": device_properties.name,
            "device_total_bytes": device_properties.total_memory,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "observations": {
            "audio_feature_positions": int(audio_features.shape[0]),
            "transcript_token_positions": transcript_length,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "model_load_seconds": model_load_seconds,
            "execution_seconds": time.monotonic() - started,
        },
        "conditions": conditions,
    }
    validate_smoke_result(result, args.git_commit)
    print(_RESULT_PREFIX + json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
