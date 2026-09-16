"""Run the held-speaker equal-budget acoustic-retention pilot on one A100."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import statistics
import subprocess
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import librosa
import torch
import transformers
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from clearx.audio_memory.gate import degrade_audio, segment_features
from clearx.audio_memory.retention import (
    ALPHA,
    CONCENTRATED_POSITION_COUNT,
    FIXED_POSITION_COUNT,
    MODEL_COMMIT,
    MODEL_REPOSITORY,
    POSITION_COUNTS,
    RAVDESS_ARCHIVE_MD5,
    TEST_ACTORS,
    TRAIN_ACTORS,
    TRANSFORMERS_COMMIT,
    mean_pool_ordered,
    retention_payload,
    validate_result,
)

RAVDESS_URL = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)
TRANSCRIPTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}
INCOMPLETE_TRANSCRIPTS = {
    "01": "Are talking by the door.",
    "02": "Are sitting by the door.",
}
MEMORY_QUALITIES = ("exact", "incomplete", "conflicting")
PROMPT = "Repeat the spoken sentence exactly. Answer only with the sentence."
NOISE_SNR_DB = -10.0
SEED = 7
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RESULT_PREFIX = "CLEARX_ADAPTIVE_RETENTION_RESULT="


def emit(stage: str, **values: object) -> None:
    """Print one machine-readable progress record immediately."""
    print(
        "CLEARX_ADAPTIVE_RETENTION="
        + json.dumps({"stage": stage, **values}, allow_nan=False, sort_keys=True),
        flush=True,
    )


def frozen_items() -> tuple[dict[str, str], ...]:
    """Return all speaker, statement, and transcript-memory combinations."""
    selected: list[dict[str, str]] = []
    for actor_id in (*TRAIN_ACTORS, *TEST_ACTORS):
        split = "train" if actor_id in TRAIN_ACTORS else "test"
        for statement_id, transcript in TRANSCRIPTS.items():
            filename = f"03-01-01-01-{statement_id}-01-{actor_id}.wav"
            other_statement = "02" if statement_id == "01" else "01"
            memories = {
                "exact": transcript,
                "incomplete": INCOMPLETE_TRANSCRIPTS[statement_id],
                "conflicting": TRANSCRIPTS[other_statement],
            }
            audio_qualities = {
                "exact": "degraded",
                "incomplete": "clean",
                "conflicting": "clean",
            }
            for memory_quality in MEMORY_QUALITIES:
                selected.append(
                    {
                        "item_id": (
                            f"actor-{actor_id}-statement-{statement_id}-{memory_quality}"
                        ),
                        "actor_id": actor_id,
                        "statement_id": statement_id,
                        "expected_transcript": transcript,
                        "memory_quality": memory_quality,
                        "memory_transcript": memories[memory_quality],
                        "audio_quality": audio_qualities[memory_quality],
                        "split": split,
                        "filename": filename,
                    }
                )
    return tuple(selected)


def md5(path: Path) -> str:
    """Return the hexadecimal MD5 published with the RAVDESS archive."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(cache: Path) -> dict[str, Path]:
    """Download, verify, and extract only the selected RAVDESS recordings."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "Audio_Speech_Actors_01-24.zip"
    if not archive.exists() or md5(archive) != RAVDESS_ARCHIVE_MD5:
        emit("dataset_download_started", url=RAVDESS_URL)
        with (
            urllib.request.urlopen(RAVDESS_URL, timeout=600) as response,
            archive.open("wb") as destination,
        ):
            while block := response.read(1024 * 1024):
                destination.write(block)
    if md5(archive) != RAVDESS_ARCHIVE_MD5:
        raise RuntimeError("RAVDESS archive differs from its published MD5")
    names = {item["filename"] for item in frozen_items()}
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as source:
        members = {
            Path(name).name: name
            for name in source.namelist()
            if Path(name).name in names
        }
        if set(members) != names:
            raise RuntimeError("RAVDESS archive lacks one or more selected recordings")
        for filename, member in members.items():
            destination = cache / filename
            if not destination.exists():
                destination.write_bytes(source.read(member))
            extracted[filename] = destination
    return extracted


def move_inputs(
    inputs: Any, device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    """Move processor tensors without casting integer token IDs."""
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
    """Return finite CPU float32 logits for the final input position."""
    logits = output.logits[0, -1].detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Thinker returned non-finite logits")
    return logits


def teacher_kl(teacher: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return KL from the full-audio next-token distribution to one candidate."""
    teacher_log = torch.log_softmax(teacher, dim=-1)
    candidate_log = torch.log_softmax(candidate, dim=-1)
    value = torch.sum(torch.exp(teacher_log) * (teacher_log - candidate_log)).item()
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("comparison produced invalid KL")
    return float(value)


def audio_span(
    input_ids: torch.Tensor, thinker: Any, tokenizer: Any
) -> tuple[int, int, torch.Tensor]:
    """Return the complete audio-tag span and its contiguous placeholder positions."""
    ids = input_ids[0]
    positions = torch.nonzero(ids == thinker.config.audio_token_id).flatten()
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
        raise RuntimeError("audio positions lack their boundary tokens")
    return start, stop, positions


def prepare_recording(
    path: Path,
    processor: Any,
    model: Any,
    thinker: Any,
) -> dict[str, Any]:
    """Compute one full-audio teacher and clean and degraded Audio Tower vectors."""
    audio, _ = librosa.load(path, sr=16000, mono=True)
    audio_bytes = path.read_bytes()
    degraded = degrade_audio(audio, audio_bytes, snr_db=NOISE_SNR_DB)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=prompt,
        audio=[audio],
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    degraded_inputs = processor(
        text=prompt,
        audio=[degraded],
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    embedding = thinker.get_input_embeddings()
    device = embedding.weight.device
    prepared = move_inputs(inputs, device, thinker.dtype)
    degraded_prepared = move_inputs(degraded_inputs, device, thinker.dtype)
    with torch.inference_mode():
        full_output = thinker(
            **prepared, use_cache=False, return_dict=True, use_audio_in_video=False
        )
        clean_output = thinker.get_audio_features(
            prepared["input_features"],
            feature_attention_mask=prepared["feature_attention_mask"],
            return_dict=True,
        )
        degraded_output = thinker.get_audio_features(
            degraded_prepared["input_features"],
            feature_attention_mask=degraded_prepared["feature_attention_mask"],
            return_dict=True,
        )
    clean = clean_output.last_hidden_state.detach().cpu().float().numpy()
    noisy = degraded_output.last_hidden_state.detach().cpu().float().numpy()
    start, stop, positions = audio_span(
        prepared["input_ids"], thinker, processor.tokenizer
    )
    if clean.shape[0] != int(positions.numel()) or noisy.shape != clean.shape:
        raise RuntimeError("Audio Tower outputs do not match processor positions")
    state = {
        "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "audio_features": {"clean": clean, "degraded": noisy},
        "audio_feature_positions": int(clean.shape[0]),
        "input_ids": prepared["input_ids"],
        "span_start": start,
        "span_stop": stop,
        "teacher_logits": last_logits(full_output),
        "full_input_positions": int(prepared["input_ids"].shape[1]),
        "device": device,
    }
    del prepared, degraded_prepared, full_output, clean_output, degraded_output
    return state


def prepare_item(
    item: dict[str, str], state: dict[str, Any], processor: Any, thinker: Any
) -> dict[str, Any]:
    """Create the transcript prefix and agreement features for one memory item."""
    device = state["device"]
    input_ids = state["input_ids"]
    start = int(state["span_start"])
    stop = int(state["span_stop"])
    transcript_ids = processor.tokenizer(
        item["memory_transcript"], add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    transcript_positions = int(transcript_ids.shape[1])
    transcript_only_ids = torch.cat(
        [input_ids[:, :start], transcript_ids, input_ids[:, stop:]], dim=1
    )
    text_embeddings = thinker.get_input_embeddings()(transcript_only_ids)
    text_vectors = (
        text_embeddings[0, start : start + transcript_positions]
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    selected_audio = state["audio_features"][item["audio_quality"]]
    aligned_audio = mean_pool_ordered(selected_audio, transcript_positions)
    return {
        "metadata": {
            **item,
            "audio_sha256": state["audio_sha256"],
            "audio_feature_positions": state["audio_feature_positions"],
            "transcript_token_positions": transcript_positions,
            "full_input_positions": state["full_input_positions"],
            "gate_features": segment_features(
                aligned_audio, text_vectors, state["audio_feature_positions"]
            ),
        },
        "audio_features": selected_audio,
        "input_ids": input_ids,
        "transcript_ids": transcript_ids,
        "start": start,
        "stop": stop,
        "teacher_logits": state["teacher_logits"],
        "device": device,
    }


def run_condition(
    prepared: dict[str, Any], positions: int, processor: Any, thinker: Any
) -> dict[str, object]:
    """Append the requested pooled acoustic positions and run the Thinker."""
    start = int(prepared["start"])
    stop = int(prepared["stop"])
    input_ids = prepared["input_ids"]
    transcript_ids = prepared["transcript_ids"]
    carrier = torch.full(
        (1, positions),
        int(thinker.config.audio_token_id),
        dtype=input_ids.dtype,
        device=prepared["device"],
    )
    candidate_ids = torch.cat(
        [input_ids[:, :start], transcript_ids, carrier, input_ids[:, stop:]], dim=1
    )
    candidate_embeddings = thinker.get_input_embeddings()(candidate_ids)
    if positions:
        pooled = mean_pool_ordered(prepared["audio_features"], positions)
        acoustic_start = start + int(transcript_ids.shape[1])
        candidate_embeddings[0, acoustic_start : acoustic_start + positions] = (
            torch.from_numpy(pooled).to(
                device=prepared["device"], dtype=candidate_embeddings.dtype
            )
            * ALPHA
        )
    with torch.inference_mode():
        output = thinker(
            input_ids=candidate_ids,
            inputs_embeds=candidate_embeddings,
            attention_mask=torch.ones_like(candidate_ids),
            use_cache=False,
            return_dict=True,
            use_audio_in_video=False,
        )
    logits = last_logits(output)
    divergence = teacher_kl(prepared["teacher_logits"], logits)
    record = {
        "acoustic_positions": positions,
        "alpha": ALPHA,
        "input_positions": int(candidate_ids.shape[1]),
        "transcript_positions": int(transcript_ids.shape[1]),
        "teacher_kl": divergence,
        "teacher_top1_agreement": bool(
            torch.argmax(prepared["teacher_logits"]) == torch.argmax(logits)
        ),
    }
    del output, logits, candidate_embeddings, candidate_ids
    return record


def heartbeat(stop: threading.Event, started: float) -> None:
    """Report accelerator use once per minute during the long model run."""
    while not stop.wait(60):
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        emit("heartbeat", seconds=round(time.monotonic() - started, 1), gpu=gpu)


def main() -> int:
    """Measure the position grid, fit the policy, and emit one validated result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--dataset-cache", type=Path, default=Path("/content/ravdess"))
    args = parser.parse_args()
    if _GIT_COMMIT.fullmatch(args.experiment_commit) is None:
        raise ValueError("--experiment-commit must be a full lowercase commit")
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
    try:
        recordings = prepare_dataset(args.dataset_cache)
        emit("model_load_started", experiment_commit=args.experiment_commit)
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
            MODEL_REPOSITORY, revision=MODEL_COMMIT
        )
        model_load_seconds = time.monotonic() - model_started
        emit("model_ready", seconds=model_load_seconds)

        grouped: dict[str, list[dict[str, str]]] = {}
        for item in frozen_items():
            grouped.setdefault(item["filename"], []).append(item)
        results: list[dict[str, Any]] = []
        for recording_index, (filename, item_group) in enumerate(
            grouped.items(), start=1
        ):
            emit(
                "recording_started",
                recording_index=recording_index,
                recording_count=len(grouped),
                filename=filename,
            )
            state = prepare_recording(recordings[filename], processor, model, thinker)
            for item in item_group:
                prepared = prepare_item(item, state, processor, thinker)
                conditions: dict[str, dict[str, object]] = {}
                for positions in POSITION_COUNTS:
                    condition = run_condition(prepared, positions, processor, thinker)
                    conditions[f"positions_{positions}"] = condition
                    emit(
                        "condition_completed",
                        item_id=item["item_id"],
                        positions=positions,
                        teacher_kl=condition["teacher_kl"],
                    )
                metadata = prepared["metadata"]
                metadata["conditions"] = conditions
                results.append(metadata)
                prepared.clear()
            state.clear()
            gc.collect()
            torch.cuda.empty_cache()

        retention = retention_payload(results)
        test_items = [item for item in results if item["split"] == "test"]
        quality_summary = {
            quality: {
                "item_count": len(
                    selected := [
                        item for item in test_items if item["memory_quality"] == quality
                    ]
                ),
                "mean_measured_benefit": statistics.fmean(
                    retention["test_measured_benefit"][item["item_id"]]
                    for item in selected
                ),
                "mean_predicted_benefit": statistics.fmean(
                    retention["test_predicted_benefit"][item["item_id"]]
                    for item in selected
                ),
            }
            for quality in MEMORY_QUALITIES
        }
        result: dict[str, Any] = {
            "schema_version": "exploratory-1.0.0",
            "experiment_id": "audio-memory-adaptive-retention-v1",
            "status": "completed",
            "source": {"experiment_commit": args.experiment_commit},
            "identity": {
                "model_repository": MODEL_REPOSITORY,
                "model_commit": MODEL_COMMIT,
                "transformers_commit": TRANSFORMERS_COMMIT,
                "ravdess_url": RAVDESS_URL,
                "ravdess_archive_md5": RAVDESS_ARCHIVE_MD5,
                "transcripts": TRANSCRIPTS,
                "incomplete_transcripts": INCOMPLETE_TRANSCRIPTS,
                "memory_qualities": list(MEMORY_QUALITIES),
                "noise_snr_db": NOISE_SNR_DB,
                "prompt": PROMPT,
                "seed": SEED,
                "alpha": ALPHA,
                "position_counts": list(POSITION_COUNTS),
                "fixed_position_count": FIXED_POSITION_COUNT,
                "concentrated_position_count": CONCENTRATED_POSITION_COUNT,
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
                "item_count": len(results),
            },
            "quality_summary": quality_summary,
            "retention": retention,
            "items": results,
        }
        validate_result(result)
        print(_RESULT_PREFIX + json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
