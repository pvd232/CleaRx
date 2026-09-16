#!/usr/bin/env python3
"""Train and test a segment-level VoxZip fusion gate on emotion-bearing speech."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import librosa
import torch
import transformers
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from tools.research.audio_memory_gate import (
    gate_payload,
    oracle_alpha,
    segment_features,
)

MODEL_REPOSITORY = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_COMMIT = "26291f793822fb6be9555850f06dfe95f2d7e695"
TRANSFORMERS_COMMIT = "7d9754a05193eb79b1d86aa744b622b8068008cd"
SMOKE_REVIEW_COMMIT = "da6c799431411a8c48102503b00df26145af7dc1"
RAVDESS_URL = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)
RAVDESS_EXPECTED_MD5 = "bc696df654c87fed845eb13823edef8a"
TRANSCRIPT = "Kids are talking by the door."
PROMPT = (
    "Identify the emotion conveyed by the speaker. Answer with exactly one "
    "lowercase word: neutral, happy, sad, angry, fearful, or disgust."
)
SEED = 7
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EMOTIONS = (
    ("neutral", "01"),
    ("happy", "03"),
    ("sad", "04"),
    ("angry", "05"),
    ("fearful", "06"),
    ("disgust", "07"),
)
TRAIN_ACTORS = ("01", "02", "03", "04")
TEST_ACTORS = ("05", "06")
PREFLIGHT_MIN_CORRECT = 4
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def emit(stage: str, **values: object) -> None:
    """Print one machine-readable progress or observation record immediately."""
    print(
        "CLEARX_AUDIO_MEMORY_GATE="
        + json.dumps({"stage": stage, **values}, allow_nan=False, sort_keys=True),
        flush=True,
    )


def load_smoke_helpers(checkout: Path) -> Any:
    """Import the validated model-input helpers from the reviewed checkout."""
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if observed != SMOKE_REVIEW_COMMIT:
        raise RuntimeError("smoke checkout does not match the reviewed commit")
    sys.path.insert(0, str(checkout))
    module_path = checkout / "experiments" / "audio_memory_smoke_worker.py"
    specification = importlib.util.spec_from_file_location(
        "clearx_reviewed_audio_memory_smoke_worker",
        module_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the reviewed smoke helper module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def items() -> tuple[dict[str, str], ...]:
    """Return the frozen speaker-disjoint RAVDESS item set."""
    selected: list[dict[str, str]] = []
    for actor_id in (*TRAIN_ACTORS, *TEST_ACTORS):
        split = "train" if actor_id in TRAIN_ACTORS else "test"
        for emotion, emotion_code in EMOTIONS:
            filename = f"03-01-{emotion_code}-01-01-01-{actor_id}.wav"
            selected.append(
                {
                    "item_id": f"actor-{actor_id}-{emotion}",
                    "actor_id": actor_id,
                    "emotion": emotion,
                    "split": split,
                    "filename": filename,
                }
            )
    return tuple(selected)


def md5(path: Path) -> str:
    """Return the hexadecimal MD5 required by the published dataset record."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(cache: Path) -> dict[str, Path]:
    """Download, verify, and extract only the frozen RAVDESS recordings."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "Audio_Speech_Actors_01-24.zip"
    if not archive.exists() or md5(archive) != RAVDESS_EXPECTED_MD5:
        emit("dataset_download_started", url=RAVDESS_URL)
        with (
            urllib.request.urlopen(RAVDESS_URL, timeout=600) as response,
            archive.open("wb") as destination,
        ):
            while block := response.read(1024 * 1024):
                destination.write(block)
        emit("dataset_download_completed", bytes=archive.stat().st_size)
    observed_md5 = md5(archive)
    if observed_md5 != RAVDESS_EXPECTED_MD5:
        raise RuntimeError(
            f"RAVDESS archive MD5 {observed_md5} differs from the published value"
        )

    selected = items()
    by_name = {item["filename"]: item for item in selected}
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as source:
        members = {
            Path(name).name: name
            for name in source.namelist()
            if Path(name).name in by_name
        }
        if set(members) != set(by_name):
            missing = sorted(set(by_name) - set(members))
            raise RuntimeError(f"RAVDESS archive lacks selected recordings: {missing}")
        for filename, member in members.items():
            destination = cache / filename
            if not destination.exists():
                destination.write_bytes(source.read(member))
            extracted[filename] = destination
    return extracted


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


def restricted_label(logits: torch.Tensor, tokenizer: Any) -> str:
    """Choose the highest-scoring declared emotion label at the next token."""
    scores: dict[str, float] = {}
    for label, _code in EMOTIONS:
        token_ids: set[int] = set()
        for variant in (
            label,
            label.capitalize(),
            f" {label}",
            f" {label.capitalize()}",
        ):
            encoded = tokenizer(variant, add_special_tokens=False).input_ids
            if encoded:
                token_ids.add(int(encoded[0]))
        scores[label] = max(float(logits[token_id]) for token_id in token_ids)
    return max(scores, key=lambda label: scores[label])


def condition_record(
    logits: torch.Tensor,
    tokenizer: Any,
    smoke: Any,
    *,
    alpha: float | None,
    input_positions: int,
    memory_positions: int | None,
    teacher_logits: torch.Tensor | None,
) -> dict[str, object]:
    """Record model comparison values and the restricted emotion prediction."""
    record = smoke.condition_record(
        logits,
        tokenizer,
        alpha=alpha,
        input_positions=input_positions,
        memory_positions=memory_positions,
        teacher=teacher_logits,
    )
    record["restricted_emotion"] = restricted_label(logits, tokenizer)
    return record


def prepare_item(
    item: dict[str, str],
    audio_path: Path,
    processor: Any,
    thinker: Any,
    smoke: Any,
) -> dict[str, Any]:
    """Build full-audio and equal-position compressed inputs for one recording."""
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
        TRANSCRIPT,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(first_device)
    transcript_length = int(transcript_ids.shape[1])
    compressed_ids = torch.cat(
        [input_ids[:, :span_start], transcript_ids, input_ids[:, span_stop:]],
        dim=1,
    )
    base_embeddings = embedding_layer(compressed_ids)
    memory_slice = slice(span_start, span_start + transcript_length)
    text_vectors = (
        base_embeddings[0, memory_slice]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    pooled_audio = smoke.mean_pool_ordered(audio_features, transcript_length)
    metadata: dict[str, Any] = {
        **item,
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "audio_feature_positions": int(audio_features.shape[0]),
        "transcript_token_positions": transcript_length,
        "full_input_positions": int(input_ids.shape[1]),
        "compressed_input_positions": int(compressed_ids.shape[1]),
        "gate_features": segment_features(
            pooled_audio,
            text_vectors,
            int(audio_features.shape[0]),
        ),
        "teacher": condition_record(
            teacher_logits,
            processor.tokenizer,
            smoke,
            alpha=None,
            input_positions=int(input_ids.shape[1]),
            memory_positions=None,
            teacher_logits=None,
        ),
    }
    del full_output, audio_output, prepared
    return {
        "metadata": metadata,
        "teacher_logits": teacher_logits,
        "pooled_audio": pooled_audio,
        "base_embeddings": base_embeddings,
        "compressed_ids": compressed_ids,
        "span_start": span_start,
        "transcript_length": transcript_length,
        "first_device": str(first_device),
        "text_vectors": text_vectors,
    }


def run_condition(
    prepared: dict[str, Any],
    alpha: float,
    thinker: Any,
    processor: Any,
    smoke: Any,
) -> dict[str, object]:
    """Run one compressed fusion coefficient through the frozen Thinker."""
    started = time.monotonic()
    fused = smoke.fuse_memory_vectors(
        prepared["pooled_audio"],
        prepared["text_vectors"],
        alpha,
    )
    variant_embeddings = prepared["base_embeddings"].clone()
    start = int(prepared["span_start"])
    length = int(prepared["transcript_length"])
    variant_embeddings[0, start : start + length] = torch.from_numpy(fused).to(
        device=torch.device(prepared["first_device"]),
        dtype=variant_embeddings.dtype,
    )
    compressed_ids = prepared["compressed_ids"]
    with torch.inference_mode():
        output = thinker(
            input_ids=compressed_ids,
            inputs_embeds=variant_embeddings,
            attention_mask=torch.ones_like(compressed_ids),
            use_cache=False,
            return_dict=True,
            use_audio_in_video=False,
        )
    logits = smoke.last_logits(output)
    record = condition_record(
        logits,
        processor.tokenizer,
        smoke,
        alpha=alpha,
        input_positions=int(compressed_ids.shape[1]),
        memory_positions=length,
        teacher_logits=prepared["teacher_logits"],
    )
    record["seconds"] = time.monotonic() - started
    del output, logits, variant_embeddings
    return record


def release_prepared(prepared: dict[str, Any]) -> None:
    """Release tensors retained for one item after its final condition."""
    prepared.clear()
    gc.collect()
    torch.cuda.empty_cache()


def split_aggregate(
    item_results: list[dict[str, Any]], split: str
) -> dict[str, dict[str, object]]:
    """Aggregate grid conditions for one declared speaker split."""
    selected = [item for item in item_results if item["split"] == split]
    aggregate: dict[str, dict[str, object]] = {}
    for alpha in ALPHAS:
        key = f"alpha_{alpha:.2f}"
        divergences = [
            float(item["conditions"][key]["teacher_kl"]) for item in selected
        ]
        accuracy = sum(
            item["conditions"][key]["restricted_emotion"] == item["emotion"]
            for item in selected
        )
        aggregate[key] = {
            "alpha": alpha,
            "item_count": len(selected),
            "mean_teacher_kl": statistics.fmean(divergences),
            "median_teacher_kl": statistics.median(divergences),
            "emotion_accuracy": accuracy / len(selected),
        }
    return aggregate


def main() -> int:
    """Run the frozen preflight, grid, gate fit, and held-speaker evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-checkout", type=Path, required=True)
    parser.add_argument("--gate-commit", required=True)
    parser.add_argument("--dataset-cache", type=Path, default=Path("/content/ravdess"))
    args = parser.parse_args()
    if _GIT_COMMIT.fullmatch(args.gate_commit) is None:
        raise ValueError("--gate-commit must be a full lowercase commit")
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
    recordings = prepare_dataset(args.dataset_cache)
    emit("model_load_started", gate_commit=args.gate_commit)
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

    frozen_items = items()
    preflight_items = [item for item in frozen_items if item["actor_id"] == "01"]
    prepared_cache: dict[str, dict[str, Any]] = {}
    preflight_observations: list[dict[str, object]] = []
    try:
        for item in preflight_items:
            prepared = prepare_item(
                item,
                recordings[item["filename"]],
                processor,
                thinker,
                smoke,
            )
            prepared_cache[item["item_id"]] = prepared
            prediction = prepared["metadata"]["teacher"]["restricted_emotion"]
            observation = {
                "item_id": item["item_id"],
                "expected": item["emotion"],
                "predicted": prediction,
                "correct": prediction == item["emotion"],
            }
            preflight_observations.append(observation)
            emit("preflight_item", **observation)

        correct = sum(bool(item["correct"]) for item in preflight_observations)
        unique_predictions = len(
            {str(item["predicted"]) for item in preflight_observations}
        )
        preflight_passed = correct >= PREFLIGHT_MIN_CORRECT and unique_predictions >= 4
        preflight = {
            "passed": preflight_passed,
            "minimum_correct": PREFLIGHT_MIN_CORRECT,
            "correct": correct,
            "unique_predictions": unique_predictions,
            "items": preflight_observations,
        }
        print(
            "CLEARX_AUDIO_MEMORY_GATE_PREFLIGHT_RESULT="
            + json.dumps(preflight, allow_nan=False, sort_keys=True),
            flush=True,
        )
        if not preflight_passed:
            for prepared in prepared_cache.values():
                release_prepared(prepared)
            stopped_result = {
                "schema_version": "exploratory-1.0.0",
                "experiment_id": "audio-memory-fusion-gate-v1",
                "status": "stopped_after_preflight",
                "preflight": preflight,
                "source": {
                    "gate_commit": args.gate_commit,
                    "smoke_review_commit": SMOKE_REVIEW_COMMIT,
                },
            }
            print(
                "CLEARX_AUDIO_MEMORY_GATE_RESULT="
                + json.dumps(stopped_result, allow_nan=False, sort_keys=True),
                flush=True,
            )
            return 0

        item_results: list[dict[str, Any]] = []
        held_out_prepared: dict[str, dict[str, Any]] = {}
        for item_index, item in enumerate(frozen_items, start=1):
            emit(
                "item_started",
                item_index=item_index,
                item_count=len(frozen_items),
                item_id=item["item_id"],
                split=item["split"],
            )
            prepared = prepared_cache.pop(item["item_id"], None)
            if prepared is None:
                prepared = prepare_item(
                    item,
                    recordings[item["filename"]],
                    processor,
                    thinker,
                    smoke,
                )
            conditions: dict[str, dict[str, object]] = {}
            for alpha in ALPHAS:
                record = run_condition(prepared, alpha, thinker, processor, smoke)
                conditions[f"alpha_{alpha:.2f}"] = record
                emit(
                    "condition_complete",
                    item_index=item_index,
                    item_count=len(frozen_items),
                    item_id=item["item_id"],
                    alpha=alpha,
                    teacher_kl=record["teacher_kl"],
                    predicted_emotion=record["restricted_emotion"],
                    seconds=record["seconds"],
                )
            metadata = prepared["metadata"]
            metadata["conditions"] = conditions
            metadata["oracle_alpha"] = oracle_alpha(metadata)
            item_results.append(metadata)
            if item["split"] == "test":
                held_out_prepared[item["item_id"]] = prepared
            else:
                release_prepared(prepared)
            emit(
                "item_completed",
                item_index=item_index,
                item_id=item["item_id"],
                oracle_alpha=metadata["oracle_alpha"],
            )

        learned = gate_payload(item_results)
        for item in item_results:
            if item["split"] != "test":
                continue
            alpha = float(learned["test_predictions"][item["item_id"]])
            prepared = held_out_prepared.pop(item["item_id"])
            record = run_condition(prepared, alpha, thinker, processor, smoke)
            item["learned_condition"] = record
            release_prepared(prepared)
            emit(
                "learned_condition_complete",
                item_id=item["item_id"],
                alpha=alpha,
                teacher_kl=record["teacher_kl"],
                predicted_emotion=record["restricted_emotion"],
            )

        train_aggregate = split_aggregate(item_results, "train")
        test_aggregate = split_aggregate(item_results, "test")
        test_items = [item for item in item_results if item["split"] == "test"]
        fixed_alpha = float(learned["fixed_alpha_selected_on_train"])
        fixed_key = f"alpha_{fixed_alpha:.2f}"
        learned_divergences = [
            float(item["learned_condition"]["teacher_kl"]) for item in test_items
        ]
        fixed_divergences = [
            float(item["conditions"][fixed_key]["teacher_kl"]) for item in test_items
        ]
        transcript_divergences = [
            float(item["conditions"]["alpha_0.00"]["teacher_kl"]) for item in test_items
        ]
        oracle_divergences = [
            min(float(record["teacher_kl"]) for record in item["conditions"].values())
            for item in test_items
        ]
        learned_accuracy = statistics.fmean(
            item["learned_condition"]["restricted_emotion"] == item["emotion"]
            for item in test_items
        )
        fixed_accuracy = statistics.fmean(
            item["conditions"][fixed_key]["restricted_emotion"] == item["emotion"]
            for item in test_items
        )
        learned_evaluation = {
            "item_count": len(test_items),
            "mean_teacher_kl": statistics.fmean(learned_divergences),
            "median_teacher_kl": statistics.median(learned_divergences),
            "emotion_accuracy": learned_accuracy,
            "fixed_alpha_selected_on_train": fixed_alpha,
            "fixed_alpha_mean_teacher_kl": statistics.fmean(fixed_divergences),
            "fixed_alpha_emotion_accuracy": fixed_accuracy,
            "transcript_only_mean_teacher_kl": statistics.fmean(transcript_divergences),
            "oracle_grid_mean_teacher_kl": statistics.fmean(oracle_divergences),
            "items_better_than_fixed_alpha": sum(
                learned_value < fixed_value
                for learned_value, fixed_value in zip(
                    learned_divergences,
                    fixed_divergences,
                    strict=True,
                )
            ),
            "directional_success": statistics.fmean(learned_divergences)
            < statistics.fmean(fixed_divergences),
        }
        result = {
            "schema_version": "exploratory-1.0.0",
            "experiment_id": "audio-memory-fusion-gate-v1",
            "status": "completed",
            "source": {
                "gate_commit": args.gate_commit,
                "smoke_review_commit": SMOKE_REVIEW_COMMIT,
            },
            "identity": {
                "model_repository": MODEL_REPOSITORY,
                "model_commit": MODEL_COMMIT,
                "transformers_commit": TRANSFORMERS_COMMIT,
                "ravdess_url": RAVDESS_URL,
                "ravdess_archive_md5": RAVDESS_EXPECTED_MD5,
                "transcript": TRANSCRIPT,
                "prompt": PROMPT,
                "seed": SEED,
                "alphas": list(ALPHAS),
                "train_actors": list(TRAIN_ACTORS),
                "test_actors": list(TEST_ACTORS),
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
            "preflight": preflight,
            "train_aggregate": train_aggregate,
            "test_aggregate": test_aggregate,
            "gate": learned,
            "learned_evaluation": learned_evaluation,
            "items": item_results,
        }
        print(
            "CLEARX_AUDIO_MEMORY_GATE_RESULT="
            + json.dumps(result, allow_nan=False, sort_keys=True),
            flush=True,
        )
        return 0
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
