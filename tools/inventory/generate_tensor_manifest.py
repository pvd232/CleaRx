#!/usr/bin/env python3
"""Build the complete Qwen3-Omni tensor manifest from pinned safetensors bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


REPOSITORY = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "BOOL": 1}


def sha256_file(path: Path) -> str:
    """Hash one complete file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contiguous_strides(shape: list[int]) -> list[int]:
    """Return row-major element strides for a safetensors shape."""
    stride = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return list(reversed(result))


def classify(name: str, shape: list[int]) -> dict[str, Any]:
    """Map one source name to a module, role, target dtype, and placement."""
    if name.startswith("thinker.visual."):
        module = "vision"
    elif name.startswith("thinker.audio_tower."):
        module = "aut"
    elif name.startswith("thinker."):
        module = "thinker"
    elif name.startswith("talker.code_predictor."):
        module = "mtp"
    elif name.startswith("talker."):
        module = "talker"
    elif name.startswith("code2wav."):
        module = "code2wav"
    else:
        raise ValueError(f"unclassified tensor: {name}")

    roles: list[str] = []
    if ".mlp.experts." in name:
        roles.append("routed_expert")
    if ".shared_expert." in name:
        roles.append("shared_expert")
    if ".mlp.gate." in name or name.endswith(".gate.weight"):
        roles.append("router")
    if "norm" in name:
        roles.append("normalization")
    if "embed" in name:
        roles.append("embedding")
    if "proj" in name or "projection" in name:
        roles.append("projection")
    if "lm_head" in name or "codec_head" in name:
        roles.append("output_head")
    if name.endswith(".bias"):
        roles.append("bias")
    if not roles:
        roles.append("dense_weight")

    if module == "vision":
        return {"module": module, "roles": roles, "disposition": "removed", "target_gguf_name": None, "target_tensor_sha256": None, "target_checksum_status": "removed_not_applicable", "target_dtype": None, "quantization": "removed", "backend": "absent", "residency": "absent"}
    if module == "code2wav":
        target_dtype, quantization, backend, residency = "BF16", "none", "cpu", "cpu_permanent"
    elif module == "mtp":
        target_dtype, quantization, backend, residency = "BF16", "none", "l4_cuda", "speaking_gpu"
    elif len(shape) <= 1 or any(role in roles for role in ("normalization", "bias")):
        target_dtype, quantization, backend = "BF16", "none", "l4_cuda"
        residency = "listening_gpu_arena" if module == "aut" else ("gpu_permanent" if module == "thinker" else "speaking_gpu")
    elif module == "thinker":
        target_dtype, quantization, backend, residency = "Q4_K_M", "Q4_K_M", "l4_cuda", "gpu_permanent"
    else:
        target_dtype, quantization, backend = "Q8_0", "Q8_0", "l4_cuda"
        residency = "listening_gpu_arena" if module == "aut" else "speaking_gpu"
    if module == "talker" and "routed_expert" in roles:
        residency = "pinned_cpu_authoritative_gpu_slot_cache"
    return {
        "module": module,
        "roles": roles,
        "disposition": "included",
        "target_gguf_name": f"clearx.{name}",
        "target_tensor_sha256": None,
        "target_checksum_status": "pending_phase1_conversion",
        "target_dtype": target_dtype,
        "quantization": quantization,
        "backend": backend,
        "residency": residency,
    }


def read_header(handle: Any) -> tuple[int, dict[str, Any]]:
    """Read one safetensors header and return the first data-byte offset."""
    header_size = struct.unpack("<Q", handle.read(8))[0]
    header = json.loads(handle.read(header_size))
    return 8 + header_size, header


def tensor_digest(mapped: mmap.mmap, start: int, end: int) -> str:
    """Hash one exact tensor byte range without copying the full shard."""
    digest = hashlib.sha256()
    view = memoryview(mapped)[start:end]
    for offset in range(0, len(view), 16 * 1024 * 1024):
        digest.update(view[offset : offset + 16 * 1024 * 1024])
    view.release()
    return digest.hexdigest()


def architecture_facts(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the official configuration values required by P0-002."""
    thinker = config["thinker_config"]
    thinker_text = thinker["text_config"]
    talker = config["talker_config"]
    talker_text = talker["text_config"]
    mtp = talker["code_predictor_config"]
    code2wav = config["code2wav_config"]
    return {
        "source_dtype": config["dtype"],
        "audio_output_enabled": config["enable_audio_output"],
        "thinker": {"layers": thinker_text["num_hidden_layers"], "hidden_size": thinker_text["hidden_size"], "experts": thinker_text["num_experts"], "active_experts_per_token": thinker_text["num_experts_per_tok"], "vocabulary": thinker_text["vocab_size"]},
        "aut": {"layers": thinker["audio_config"]["encoder_layers"], "hidden_size": thinker["audio_config"]["d_model"], "mel_bins": thinker["audio_config"]["num_mel_bins"], "output_dim": thinker["audio_config"]["output_dim"]},
        "vision": {"layers": thinker["vision_config"]["depth"], "hidden_size": thinker["vision_config"]["hidden_size"], "deployment_disposition": "removed"},
        "talker": {"layers": talker_text["num_hidden_layers"], "hidden_size": talker_text["hidden_size"], "experts": talker_text["num_experts"], "active_experts_per_token": talker_text["num_experts_per_tok"], "primary_codec_vocabulary": talker_text["vocab_size"], "code_groups": talker["num_code_groups"]},
        "mtp": {"layers": mtp["num_hidden_layers"], "hidden_size": mtp["hidden_size"], "code_groups": mtp["num_code_groups"], "residual_codebooks": mtp["num_code_groups"] - 1, "residual_vocabulary": mtp["vocab_size"]},
        "code2wav": {"layers": code2wav["num_hidden_layers"], "hidden_size": code2wav["hidden_size"], "quantizers": code2wav["num_quantizers"], "semantic_quantizers": code2wav["num_semantic_quantizers"], "semantic_codebook_size": code2wav["semantic_codebook_size"], "residual_codebook_size": code2wav["codebook_size"], "sliding_window": code2wav["sliding_window"], "upsample_rates": code2wav["upsample_rates"]},
        "speaker": {"ids": talker["speaker_id"], "dedicated_tensor_prefix": None, "binding": "speaker IDs select the Talker conditioning path; the checkpoint has no dedicated speaker tensor prefix"},
    }


def generate(model_dir: Path) -> dict[str, Any]:
    """Inspect all pinned model shards and return a checksum-complete manifest."""
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings if item.rfilename.endswith(".safetensors")}
    tensors: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for shard_name in sorted(remote):
        path = model_dir / shard_name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = remote[shard_name].lfs.sha256
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha or path.stat().st_size != remote[shard_name].size:
            raise ValueError(f"shard identity mismatch: {shard_name}")
        with path.open("rb") as handle:
            data_start, header = read_header(handle)
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                count = 0
                for source_name, metadata in sorted(header.items()):
                    if source_name == "__metadata__":
                        continue
                    shape = metadata["shape"]
                    dtype = metadata["dtype"]
                    relative_start, relative_end = metadata["data_offsets"]
                    size_bytes = relative_end - relative_start
                    expected_bytes = DTYPE_BYTES[dtype]
                    for dimension in shape:
                        expected_bytes *= dimension
                    if expected_bytes != size_bytes:
                        raise ValueError(f"tensor byte count mismatch: {source_name}")
                    policy = classify(source_name, shape)
                    tensors.append({
                        "source_name": source_name,
                        "source_shard": shard_name,
                        "source_shard_sha256": actual_sha,
                        "source_tensor_sha256": tensor_digest(mapped, data_start + relative_start, data_start + relative_end),
                        "source_byte_range": [data_start + relative_start, data_start + relative_end],
                        "shape": shape,
                        "source_dtype": dtype,
                        "source_layout": "row_major_contiguous",
                        "source_strides_elements": contiguous_strides(shape),
                        "size_bytes": size_bytes,
                        **policy,
                    })
                    count += 1
        shards.append({"path": shard_name, "sha256": actual_sha, "size_bytes": path.stat().st_size, "tensor_count": count})
    tensors.sort(key=lambda item: item["source_name"])
    module_counts = Counter(item["module"] for item in tensors)
    module_bytes = Counter()
    for item in tensors:
        module_bytes[item["module"]] += item["size_bytes"]
    return {
        "schema_version": "1.0.0",
        "manifest_id": "qwen3-omni-30b-a3b-instruct-clearx-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"repository": REPOSITORY, "commit": REVISION, "config_path": "config.json", "config_sha256": sha256_file(config_path), "tensor_checksum": "SHA-256 over exact safetensors data_offsets bytes"},
        "target_policy": {"gguf_namespace": "clearx.", "thinker_matrices": "Q4_K_M", "aut_matrices": "Q8_0", "talker_matrices": "Q8_0", "one_dimensional_and_normalization": "BF16", "mtp": "BF16", "code2wav": "BF16 CPU", "vision": "removed", "conversion_status": "names and placement frozen; target bytes are produced in Phase 1"},
        "architecture": architecture_facts(config),
        "shards": shards,
        "summary": {"tensor_count": len(tensors), "source_bytes": sum(item["size_bytes"] for item in tensors), "by_module_count": dict(sorted(module_counts.items())), "by_module_bytes": dict(sorted(module_bytes.items()))},
        "tensors": tensors,
    }


def main() -> int:
    """Generate one deterministic-content manifest, excluding its timestamp."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = generate(args.model_dir)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE: {len(manifest['tensors'])} tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
