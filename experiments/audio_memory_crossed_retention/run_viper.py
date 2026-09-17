"""Measure and verify the crossed acoustic-retention pilot through VIPER."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import librosa
import torch
import transformers
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from viper import execution
from viper.authoring import experiment, input, plan, replicate, stage, variant
from viper.config import BuildConfig, DiagnosticConfig
from viper.outputs import StageOutputs, output
from viper.references import GitFileRef
from viper.repository import read_source
from viper.runtime import (
    CPUComputeSpec,
    CUDAComputeSpec,
    LocalEnvSpec,
    observe_python_env,
)
from viper.stages import StageContext, build, diagnostic

from clearx.audio_memory.crossed_retention import (
    ALPHA,
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_DEVICE_NAME,
    EXPERIMENT_ID,
    MODEL_COMMIT,
    MODEL_REPOSITORY,
    NOISE_SNR_DB,
    POSITION_COUNTS,
    PROMPT,
    RAVDESS_ARCHIVE_MD5,
    RAVDESS_AUDIO_SHA256,
    RAVDESS_URL,
    SCHEMA_VERSION,
    SEED,
    VIPER_COMMIT,
    assemble_candidate_embeddings,
    audio_only_features,
    crossed_retention_payload,
    frozen_item_specs,
    mean_pool_ordered,
    result_identity,
    validate_result,
)
from clearx.audio_memory.gate import degrade_audio, segment_features

RESULT_PREFIX = "CLEARX_CROSSED_RETENTION_RESULT="
ARCHIVE_PREFIX = "CLEARX_CROSSED_RETENTION_ARCHIVE="


class MeasurementConfig(BuildConfig):
    """Bind the source commit and dataset cache used by the A100 stage."""

    experiment_commit: str
    dataset_cache: str


class VerificationConfig(DiagnosticConfig):
    """Bind the independently supplied source commit expected by verification."""

    expected_experiment_commit: str


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object emitted by an experiment stage."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("experiment artifacts must contain JSON objects")
    return value


def emit(event: str, **values: object) -> None:
    """Print one machine-readable progress record immediately."""
    print(
        "CLEARX_CROSSED_RETENTION="
        + json.dumps({"event": event, **values}, allow_nan=False, sort_keys=True),
        flush=True,
    )


def md5(path: Path) -> str:
    """Return the hexadecimal MD5 published with the RAVDESS archive."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(cache: Path) -> dict[str, Path]:
    """Download and bind exactly the twelve selected RAVDESS WAV files."""
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
    names = set(RAVDESS_AUDIO_SHA256)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as source:
        members = {
            Path(name).name: name
            for name in source.namelist()
            if Path(name).name in names
        }
        if set(members) != names:
            raise RuntimeError("RAVDESS archive lacks a selected recording")
        for filename in sorted(names):
            raw = source.read(members[filename])
            digest = hashlib.sha256(raw).hexdigest()
            if digest != RAVDESS_AUDIO_SHA256[filename]:
                raise RuntimeError(f"RAVDESS bytes differ for {filename}")
            destination = cache / filename
            if destination.exists() and destination.read_bytes() != raw:
                raise RuntimeError(f"dataset cache contains other bytes for {filename}")
            if not destination.exists():
                destination.write_bytes(raw)
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
    """Return KL from the full clean-audio teacher to one candidate."""
    teacher_log = torch.log_softmax(teacher, dim=-1)
    candidate_log = torch.log_softmax(candidate, dim=-1)
    value = torch.sum(torch.exp(teacher_log) * (teacher_log - candidate_log)).item()
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("comparison produced invalid KL")
    return float(value)


def audio_span(
    input_ids: torch.Tensor, thinker: Any, tokenizer: Any
) -> tuple[int, int, torch.Tensor]:
    """Return the complete audio-tag span and contiguous placeholder positions."""
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


def prepare_recording(path: Path, processor: Any, thinker: Any) -> dict[str, Any]:
    """Compute one clean-audio teacher and both Audio Tower representations."""
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
    clean_inputs = processor(
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
    prepared = move_inputs(clean_inputs, device, thinker.dtype)
    degraded_prepared = move_inputs(degraded_inputs, device, thinker.dtype)
    with torch.inference_mode():
        teacher_output = thinker(
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
    degraded_vectors = degraded_output.last_hidden_state.detach().cpu().float().numpy()
    start, stop, positions = audio_span(
        prepared["input_ids"], thinker, processor.tokenizer
    )
    if (
        clean.shape[0] != int(positions.numel())
        or degraded_vectors.shape != clean.shape
    ):
        raise RuntimeError("Audio Tower outputs do not match processor positions")
    state = {
        "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "audio_features": {"clean": clean, "degraded": degraded_vectors},
        "audio_feature_positions": int(clean.shape[0]),
        "input_ids": prepared["input_ids"],
        "span_start": start,
        "span_stop": stop,
        "teacher_logits": last_logits(teacher_output),
        "full_input_positions": int(prepared["input_ids"].shape[1]),
        "device": device,
    }
    del prepared, degraded_prepared, teacher_output, clean_output, degraded_output
    return state


def prepare_item(
    item: Mapping[str, str], state: Mapping[str, Any], processor: Any, thinker: Any
) -> dict[str, Any]:
    """Create transcript embeddings and both feature sets for one crossed item."""
    device = state["device"]
    input_ids = state["input_ids"]
    start = int(state["span_start"])
    stop = int(state["span_stop"])
    transcript_ids = processor.tokenizer(
        item["memory_transcript"], add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    embedding = thinker.get_input_embeddings()
    prefix_embeddings = embedding(input_ids[:, :start])
    transcript_embeddings = embedding(transcript_ids)
    suffix_embeddings = embedding(input_ids[:, stop:])
    transcript_positions = int(transcript_ids.shape[1])
    text_vectors = transcript_embeddings[0].detach().cpu().float().numpy()
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
            "audio_only_features": audio_only_features(selected_audio),
        },
        "audio_features": selected_audio,
        "prefix_ids": input_ids[:, :start],
        "transcript_ids": transcript_ids,
        "suffix_ids": input_ids[:, stop:],
        "prefix_embeddings": prefix_embeddings,
        "transcript_embeddings": transcript_embeddings,
        "suffix_embeddings": suffix_embeddings,
        "teacher_logits": state["teacher_logits"],
        "device": device,
    }


def run_condition(
    prepared: Mapping[str, Any], positions: int, thinker: Any
) -> dict[str, object]:
    """Append the requested pooled acoustic positions and run the Thinker."""
    transcript_ids = prepared["transcript_ids"]
    carrier = torch.full(
        (1, positions),
        int(thinker.config.audio_token_id),
        dtype=transcript_ids.dtype,
        device=prepared["device"],
    )
    candidate_ids = torch.cat(
        [
            prepared["prefix_ids"],
            transcript_ids,
            carrier,
            prepared["suffix_ids"],
        ],
        dim=1,
    )
    hidden_size = int(prepared["transcript_embeddings"].shape[2])
    pooled = mean_pool_ordered(prepared["audio_features"], positions)
    acoustic_embeddings = torch.from_numpy(pooled).to(
        device=prepared["device"], dtype=prepared["transcript_embeddings"].dtype
    )
    acoustic_embeddings = acoustic_embeddings.reshape(1, positions, hidden_size) * ALPHA
    candidate_embeddings = assemble_candidate_embeddings(
        prepared["prefix_embeddings"],
        prepared["transcript_embeddings"],
        acoustic_embeddings,
        prepared["suffix_embeddings"],
        join=lambda blocks: torch.cat(blocks, dim=1),
    )
    transcript_start = int(prepared["prefix_ids"].shape[1])
    transcript_stop = transcript_start + int(transcript_ids.shape[1])
    if not torch.equal(
        candidate_embeddings[:, transcript_start:transcript_stop],
        prepared["transcript_embeddings"],
    ):
        raise RuntimeError("candidate assembly altered transcript embeddings")
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
    record = {
        "acoustic_positions": positions,
        "alpha": ALPHA,
        "input_positions": int(candidate_ids.shape[1]),
        "transcript_positions": int(transcript_ids.shape[1]),
        "teacher_kl": teacher_kl(prepared["teacher_logits"], logits),
        "teacher_top1_agreement": bool(
            torch.argmax(prepared["teacher_logits"]) == torch.argmax(logits)
        ),
    }
    del output, logits, candidate_embeddings, candidate_ids
    return record


def heartbeat(stop: threading.Event, started: float) -> None:
    """Report accelerator use once per minute during the model run."""
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


@build(config=MeasurementConfig)
def measure_retention(context: StageContext[MeasurementConfig]) -> None:
    """Measure the complete crossed grid on the selected CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    if properties.name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(
            f"expected {EXPECTED_DEVICE_NAME}, received {properties.name}"
        )
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(f"expected capability 8.0, received {capability}")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, started), daemon=True).start()
    try:
        recordings = prepare_dataset(Path(context.config.dataset_cache))
        emit("model_load_started", commit=context.config.experiment_commit)
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
        grouped: dict[str, list[Mapping[str, str]]] = {}
        for item in frozen_item_specs():
            grouped.setdefault(item["filename"], []).append(item)
        results: list[dict[str, Any]] = []
        for recording_index, filename in enumerate(sorted(grouped), start=1):
            emit(
                "recording_started",
                index=recording_index,
                count=len(grouped),
                filename=filename,
            )
            state = prepare_recording(recordings[filename], processor, thinker)
            for item in grouped[filename]:
                prepared = prepare_item(item, state, processor, thinker)
                conditions: dict[str, dict[str, object]] = {}
                for positions in POSITION_COUNTS:
                    condition = run_condition(prepared, positions, thinker)
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
        retention = crossed_retention_payload(results)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "completed",
            "source": {
                "experiment_commit": context.config.experiment_commit,
                "viper_run_id": str(context.run_id),
                "viper_attempt_id": context.attempt_id,
                "viper_stage_id": str(context.stage_id),
            },
            "identity": result_identity(),
            "environment": {
                "device": properties.name,
                "compute_capability": list(capability),
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
            "retention": retention,
            "items": results,
        }
        validate_result(
            result, expected_experiment_commit=context.config.experiment_commit
        )
        destination = context.outputs["result"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        emit(
            "measurement_completed",
            classification=retention["evaluation"]["classification"],
        )
    finally:
        stop.set()


@diagnostic(config=VerificationConfig)
def verify_retention(context: StageContext[VerificationConfig]) -> None:
    """Recompute the measurement artifact in an independent CPU stage."""
    raw = context.inputs["a100_result"].read_bytes()
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise TypeError("A100 result must contain one JSON object")
    report = validate_result(
        result,
        expected_experiment_commit=context.config.expected_experiment_commit,
    )
    report["input_result_sha256"] = hashlib.sha256(raw).hexdigest()
    report["verification_stage"] = {
        "viper_run_id": str(context.run_id),
        "viper_attempt_id": context.attempt_id,
        "viper_stage_id": str(context.stage_id),
    }
    destination = context.outputs["report"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_study(
    *,
    experiment_commit: str,
    dataset_cache: Path,
    gpu_environment: LocalEnvSpec,
    cpu_environment: LocalEnvSpec,
) -> Any:
    """Construct the two-stage crossed-retention study."""
    measurement = stage(
        measure_retention,
        stage_id="measure_retention",
        config=MeasurementConfig(
            experiment_commit=experiment_commit,
            dataset_cache=str(dataset_cache),
        ),
        outputs=StageOutputs(
            result=output(path="result.json", loader=load_json, data_role="evaluation")
        ),
        env=gpu_environment,
    )
    verification = stage(
        verify_retention,
        stage_id="verify_retention",
        config=VerificationConfig(expected_experiment_commit=experiment_commit),
        inputs=(input("a100_result", source=measurement.outputs["result"]),),
        outputs=StageOutputs(
            report=output(
                path="verification.json", loader=load_json, data_role="evaluation"
            )
        ),
        env=cpu_environment,
    )
    return experiment(
        experiment_id="audio_memory_crossed_retention",
        variants=(
            variant(
                "ravdess_crossed_equal_budget",
                stages=(measurement, verification),
                estimator=verification.outputs["report"],
            ),
        ),
        replicates=(replicate(seed=SEED),),
    )


def materialize_audit_archive(root: Path, resolved_run: Path) -> tuple[bytes, str]:
    """Collect the scattered local-store records into one Git audit tree."""
    run_root = resolved_run.parent
    experiment_root = run_root.parents[2]
    relative_experiment = experiment_root.relative_to(root)
    with tempfile.TemporaryDirectory(prefix="clearx-viper-audit-") as temporary:
        materialized = Path(temporary) / relative_experiment
        for source in experiment_root.rglob("*"):
            if source.is_file():
                destination = materialized / source.relative_to(experiment_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        store = root / ".viper" / "store"
        for source in store.glob(f"*/{relative_experiment.as_posix()}/**/*"):
            if not source.is_file():
                continue
            marker = source.parts.index(relative_experiment.parts[0])
            relative = Path(*source.parts[marker:]).relative_to(relative_experiment)
            destination = materialized / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = source.read_bytes()
            if destination.exists() and destination.read_bytes() != raw:
                raise RuntimeError(f"conflicting VIPER records at {relative}")
            destination.write_bytes(raw)
        required = (
            run_root.relative_to(experiment_root) / "spec.yaml",
            run_root.relative_to(experiment_root) / "resolved.yaml",
            run_root.relative_to(experiment_root) / "attempts/1/resolved.yaml",
            run_root.relative_to(experiment_root) / "attempts/1/journal.jsonl",
            run_root.relative_to(experiment_root)
            / "artifacts/measure_retention/result/result.json",
            run_root.relative_to(experiment_root)
            / "artifacts/verify_retention/report/verification.json",
        )
        missing = [
            path.as_posix() for path in required if not (materialized / path).is_file()
        ]
        if missing:
            raise RuntimeError(f"VIPER audit archive lacks records: {missing}")
        archive_path = Path(temporary) / "audit.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(materialized, arcname=relative_experiment.as_posix())
        raw_archive = archive_path.read_bytes()
    return raw_archive, hashlib.sha256(raw_archive).hexdigest()


def main() -> int:
    """Run the governed study and emit its complete local audit archive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dataset-cache", type=Path, default=Path("/content/ravdess"))
    args = parser.parse_args()
    source = read_source()
    if source.commit != args.expected_commit:
        raise RuntimeError("workspace source differs from --expected-commit")
    if VIPER_COMMIT != "43a939a7abb2412cf5da7a0ddd5d01f67d1d84ab":
        raise RuntimeError("runner and frozen VIPER commit differ")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable before VIPER planning")
    if torch.cuda.get_device_name(0) != EXPECTED_DEVICE_NAME:
        raise RuntimeError("VIPER host does not expose the frozen A100 model")
    lockfile = GitFileRef(
        repository=source.repository,
        commit=source.commit,
        path="environment.yml",
    )
    python_environment = observe_python_env()
    gpu_environment = LocalEnvSpec(
        lockfile=lockfile,
        python_env=python_environment,
        compute=CUDAComputeSpec(model=EXPECTED_DEVICE_NAME, count=1),
    )
    cpu_environment = LocalEnvSpec(
        lockfile=lockfile,
        python_env=python_environment,
        compute=CPUComputeSpec(),
    )
    study = build_study(
        experiment_commit=args.expected_commit,
        dataset_cache=args.dataset_cache,
        gpu_environment=gpu_environment,
        cpu_environment=cpu_environment,
    )
    draft = plan(
        experiment=study,
        source=source,
        env=cpu_environment,
        reproducibility="relaxed",
    )
    completed = execution.run(draft, repository_root=Path.cwd())
    if completed.status != "succeeded":
        raise RuntimeError(f"VIPER run ended with {completed.status}")
    raw_archive, archive_sha256 = materialize_audit_archive(Path.cwd(), completed.path)
    run_root = completed.path.parent
    result_path = run_root / "artifacts/measure_retention/result/result.json"
    if not result_path.is_file():
        candidates = list(
            (Path.cwd() / ".viper" / "store").glob(
                f"*/{result_path.relative_to(Path.cwd()).as_posix()}"
            )
        )
        if len(candidates) != 1:
            raise RuntimeError("cannot resolve the measured result artifact")
        result_path = candidates[0]
    print(
        RESULT_PREFIX + base64.b64encode(result_path.read_bytes()).decode(), flush=True
    )
    print(
        ARCHIVE_PREFIX
        + json.dumps(
            {
                "archive_base64": base64.b64encode(raw_archive).decode(),
                "archive_sha256": archive_sha256,
                "run_id": str(completed.record.run_id),
                "resolved_run": completed.path.relative_to(Path.cwd()).as_posix(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
