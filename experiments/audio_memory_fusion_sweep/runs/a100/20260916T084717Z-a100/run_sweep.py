#!/usr/bin/env python3
"""Run a ten-recording, equal-memory audio-text coefficient sweep on an A100."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import librosa
import torch
import transformers
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

MODEL_REPOSITORY = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_COMMIT = "26291f793822fb6be9555850f06dfe95f2d7e695"
TRANSFORMERS_COMMIT = "7d9754a05193eb79b1d86aa744b622b8068008cd"
DATASET_COMMIT = "26eb9aaf76e81b692f806f9140c2d2777410d7a1"
SMOKE_REVIEW_COMMIT = "da6c799431411a8c48102503b00df26145af7dc1"
PROMPT = "Repeat the spoken digit and nothing else."
SEED = 7
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ITEMS = (
    ("0-george-0", "0_george_0.wav", "zero"),
    ("1-jackson-0", "1_jackson_0.wav", "one"),
    ("2-lucas-0", "2_lucas_0.wav", "two"),
    ("3-nicolas-0", "3_nicolas_0.wav", "three"),
    ("4-theo-0", "4_theo_0.wav", "four"),
    ("5-yweweler-0", "5_yweweler_0.wav", "five"),
    ("6-george-0", "6_george_0.wav", "six"),
    ("7-jackson-0", "7_jackson_0.wav", "seven"),
    ("8-lucas-0", "8_lucas_0.wav", "eight"),
    ("9-nicolas-0", "9_nicolas_0.wav", "nine"),
)
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def emit(stage: str, **values: object) -> None:
    """Print one machine-readable progress or observation record immediately."""
    print(
        "CLEARX_AUDIO_MEMORY_SWEEP="
        + json.dumps({"stage": stage, **values}, allow_nan=False, sort_keys=True),
        flush=True,
    )


def load_smoke_helpers(checkout: Path) -> Any:
    """Import the validated smoke helpers from the exact review checkout."""
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        != SMOKE_REVIEW_COMMIT
    ):
        raise RuntimeError("smoke checkout does not match the validated review commit")
    sys.path.insert(0, str(checkout))
    from experiments import audio_memory_smoke_worker

    return audio_memory_smoke_worker


def download_recording(filename: str, destination: Path) -> tuple[str, str]:
    """Download one commit-pinned FSDD recording and return its URL and digest."""
    url = (
        "https://raw.githubusercontent.com/Jakobovski/"
        f"free-spoken-digit-dataset/{DATASET_COMMIT}/recordings/{filename}"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    destination.write_bytes(content)
    return url, hashlib.sha256(content).hexdigest()


def heartbeat(stop: threading.Event, started: float) -> None:
    """Report accelerator and host-memory use once per minute."""
    while not stop.wait(60):
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        meminfo = {
            key: value
            for key, value in (
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
                if ":" in line
            )
        }
        emit(
            "heartbeat",
            seconds=round(time.monotonic() - started, 1),
            gpu=gpu,
            mem_available_kb=int(meminfo["MemAvailable"].split()[0]),
        )


def prepare_item(
    item_id: str,
    filename: str,
    transcript: str,
    processor: Any,
    thinker: Any,
    smoke: Any,
) -> tuple[Any, ...]:
    """Build the full-audio input and equal-position compressed memory inputs."""
    with tempfile.TemporaryDirectory(prefix=f"clearx-sweep-{item_id}-") as directory:
        audio_path = Path(directory) / filename
        audio_url, audio_sha256 = download_recording(filename, audio_path)
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": PROMPT},
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
    prepared = smoke.move_inputs_to_device(inputs, first_device, thinker.dtype)
    with torch.inference_mode():
        full_output = thinker(
            **prepared,
            use_cache=False,
            return_dict=True,
            use_audio_in_video=False,
        )
        teacher_logits = smoke.last_logits(full_output)
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
    input_ids = prepared["input_ids"]
    span_start, span_stop, audio_positions = smoke.audio_span(
        input_ids,
        thinker,
        processor.tokenizer,
    )
    if audio_features.shape[0] != int(audio_positions.numel()):
        raise RuntimeError("Audio Tower vectors do not match processor positions")
    transcript_ids = processor.tokenizer(
        transcript,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(first_device)
    transcript_length = int(transcript_ids.shape[1])
    compressed_ids = torch.cat(
        [input_ids[:, :span_start], transcript_ids, input_ids[:, span_stop:]],
        dim=1,
    )
    base_embeddings = embedding_layer(compressed_ids)
    pooled_audio = smoke.mean_pool_ordered(audio_features, transcript_length)
    metadata = {
        "item_id": item_id,
        "filename": filename,
        "transcript": transcript,
        "audio_url": audio_url,
        "audio_sha256": audio_sha256,
        "audio_feature_positions": int(audio_features.shape[0]),
        "transcript_token_positions": transcript_length,
        "full_input_positions": int(input_ids.shape[1]),
        "compressed_input_positions": int(compressed_ids.shape[1]),
        "teacher": smoke.condition_record(
            teacher_logits,
            processor.tokenizer,
            alpha=None,
            input_positions=int(input_ids.shape[1]),
            memory_positions=None,
        ),
    }
    del full_output, audio_output, prepared
    return (
        metadata,
        teacher_logits,
        pooled_audio,
        base_embeddings,
        compressed_ids,
        span_start,
        transcript_length,
        str(first_device),
        str(thinker.dtype),
    )


def main() -> int:
    """Load Qwen once, stream each comparison, and print the aggregate result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-checkout", type=Path, required=True)
    parser.add_argument("--sweep-commit", required=True)
    args = parser.parse_args()
    if _GIT_COMMIT.fullmatch(args.sweep_commit) is None:
        raise ValueError("--sweep-commit must be a full lowercase commit")
    smoke = load_smoke_helpers(args.smoke_checkout.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name:
        raise RuntimeError(f"requested A100, received {properties.name}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, started), daemon=True).start()
    emit("model_load_started", sweep_commit=args.sweep_commit)
    model_started = time.monotonic()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_COMMIT,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "38GiB", "cpu": "76GiB"},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.disable_talker()
    model.eval()
    thinker = model.thinker
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_COMMIT,
    )
    model_load_seconds = time.monotonic() - model_started
    emit("model_ready", seconds=model_load_seconds)

    item_results: list[dict[str, Any]] = []
    try:
        for item_index, (item_id, filename, transcript) in enumerate(ITEMS, start=1):
            emit(
                "item_started",
                item_index=item_index,
                item_count=len(ITEMS),
                item_id=item_id,
            )
            (
                metadata,
                teacher_logits,
                pooled_audio,
                base_embeddings,
                compressed_ids,
                span_start,
                transcript_length,
                first_device,
                _thinker_dtype,
            ) = prepare_item(
                item_id,
                filename,
                transcript,
                processor,
                thinker,
                smoke,
            )
            compressed_ids_length = int(metadata["compressed_input_positions"])
            compressed_attention = torch.ones_like(compressed_ids)
            memory_slice = slice(span_start, span_start + transcript_length)
            text_vectors = (
                base_embeddings[0, memory_slice]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
            conditions: dict[str, dict[str, object]] = {}
            for alpha in ALPHAS:
                condition_started = time.monotonic()
                fused = smoke.fuse_memory_vectors(pooled_audio, text_vectors, alpha)
                variant_embeddings = base_embeddings.clone()
                variant_embeddings[0, memory_slice] = torch.from_numpy(fused).to(
                    device=torch.device(first_device),
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
                logits = smoke.last_logits(output)
                record = smoke.condition_record(
                    logits,
                    processor.tokenizer,
                    alpha=alpha,
                    input_positions=compressed_ids_length,
                    memory_positions=transcript_length,
                    teacher=teacher_logits,
                )
                record["seconds"] = time.monotonic() - condition_started
                conditions[f"alpha_{alpha:.2f}"] = record
                emit(
                    "condition_complete",
                    item_index=item_index,
                    item_count=len(ITEMS),
                    item_id=item_id,
                    alpha=alpha,
                    teacher_kl=record["teacher_kl"],
                    teacher_top1_agreement=record["teacher_top1_agreement"],
                    top_token_text=record["top_token_text"],
                    seconds=record["seconds"],
                )
                del output, logits, variant_embeddings
            metadata["conditions"] = conditions
            item_results.append(metadata)
            del teacher_logits, pooled_audio, base_embeddings
            gc.collect()
            torch.cuda.empty_cache()
            emit("item_completed", item_index=item_index, item_id=item_id)
    finally:
        stop.set()

    aggregate: dict[str, dict[str, object]] = {}
    for alpha in ALPHAS:
        key = f"alpha_{alpha:.2f}"
        divergences = [
            float(item["conditions"][key]["teacher_kl"]) for item in item_results
        ]
        agreements = [
            bool(item["conditions"][key]["teacher_top1_agreement"])
            for item in item_results
        ]
        baselines = [
            float(item["conditions"]["alpha_0.00"]["teacher_kl"])
            for item in item_results
        ]
        aggregate[key] = {
            "alpha": alpha,
            "mean_teacher_kl": statistics.fmean(divergences),
            "median_teacher_kl": statistics.median(divergences),
            "top1_agreement_count": sum(agreements),
            "item_count": len(item_results),
            "mean_relative_kl_reduction_vs_alpha_0": statistics.fmean(
                (baseline - divergence) / baseline
                for baseline, divergence in zip(baselines, divergences, strict=True)
            ),
            "items_better_than_alpha_0": sum(
                divergence < baseline
                for baseline, divergence in zip(baselines, divergences, strict=True)
            ),
        }
    result = {
        "schema_version": "exploratory-1.0.0",
        "experiment_id": "audio-memory-fusion-sweep-v1",
        "source": {
            "sweep_commit": args.sweep_commit,
            "smoke_review_commit": SMOKE_REVIEW_COMMIT,
        },
        "identity": {
            "model_repository": MODEL_REPOSITORY,
            "model_commit": MODEL_COMMIT,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "dataset_commit": DATASET_COMMIT,
            "prompt": PROMPT,
            "seed": SEED,
            "alphas": list(ALPHAS),
        },
        "environment": {
            "device": properties.name,
            "device_total_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "observations": {
            "model_load_seconds": model_load_seconds,
            "execution_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "item_count": len(item_results),
        },
        "aggregate": aggregate,
        "items": item_results,
    }
    print(
        "CLEARX_AUDIO_MEMORY_SWEEP_RESULT="
        + json.dumps(result, allow_nan=False, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
