#!/usr/bin/env python3
"""Derive the Phase 0 L4 memory budget from target tensor policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


MIB = 1_048_576
GIB = 1_073_741_824
EXPERT_SLOT_POOL_BYTES = 1_610_612_736


def sha256_file(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_bytes(tensor: dict[str, Any]) -> int:
    """Calculate pre-conversion target bytes from the frozen tensor policy."""
    if tensor["disposition"] == "removed":
        return 0
    if tensor["target_dtype"] == "BF16":
        return tensor["size_bytes"]
    shape = tensor["shape"]
    row_elements = shape[-1] if shape else 1
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    if tensor["target_dtype"] == "Q8_0":
        return rows * math.ceil(row_elements / 32) * 34
    if tensor["target_dtype"] == "Q4_K_M":
        return rows * math.ceil(row_elements / 256) * 144
    raise ValueError(f"unsupported target dtype: {tensor['target_dtype']}")


def phase(weights: int, kv_mib: int, workspace_mib: int, staging_mib: int, runtime_mib: int, fragmentation_mib: int, uncertainty_mib: int) -> dict[str, Any]:
    """Create one expected breakdown and symmetric conservative range."""
    categories = {
        "weights": weights,
        "kv": kv_mib * MIB,
        "workspace": workspace_mib * MIB,
        "staging": staging_mib * MIB,
        "runtime": runtime_mib * MIB,
        "fragmentation": fragmentation_mib * MIB,
    }
    expected = sum(categories.values())
    return {
        "expected_bytes": expected,
        "range_min_bytes": max(weights, expected - uncertainty_mib * MIB),
        "range_max_bytes": expected + uncertainty_mib * MIB,
        "measured_bytes": None,
        "measurement_status": "pending_native_loader_and_phase_instrumentation",
        "categories": categories,
    }


def build(root: Path) -> dict[str, Any]:
    """Build the budget from current manifest and certified host bytes."""
    manifest_path = root / "tensor_manifest.json"
    host_path = root / "tests/fixtures/manifests/l4_host_profile.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    host = json.loads(host_path.read_text(encoding="utf-8"))
    module_storage: Counter[str] = Counter()
    residency_storage: Counter[str] = Counter()
    for tensor in manifest["tensors"]:
        size = target_bytes(tensor)
        module_storage[tensor["module"]] += size
        residency_storage[tensor["residency"]] += size
    thinker = module_storage["thinker"]
    aut = module_storage["aut"]
    talker_dense = residency_storage["speaking_gpu"] - module_storage["mtp"]
    mtp = module_storage["mtp"]
    talking_weights = thinker + talker_dense + EXPERT_SLOT_POOL_BYTES + mtp
    listening_weights = thinker + aut
    barge_in_weights = talking_weights + aut
    hard_ceiling = host["gpu"]["memory_total_mib"] * MIB
    operational_limit = 22 * GIB
    if hard_ceiling <= operational_limit:
        raise ValueError("certified host cannot preserve the 22 GiB operational limit")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {
        "schema_version": "1.0.0",
        "budget_id": "clearx-l4-memory-budget-v1",
        "source_commit": commit,
        "tensor_manifest_sha256": sha256_file(manifest_path),
        "host_profile_sha256": sha256_file(host_path),
        "units": {"bytes": 1, "decimal_gb": 1_000_000_000, "binary_gib": GIB},
        "hard_ceiling_bytes": hard_ceiling,
        "operational_peak_limit_bytes": operational_limit,
        "reserve_bytes": hard_ceiling - operational_limit,
        "tensor_bytes": barge_in_weights,
        "tensor_categories": {
            "thinker": thinker,
            "aut": aut,
            "talker": talker_dense + EXPERT_SLOT_POOL_BYTES,
            "mtp": mtp,
            "code2wav": 0,
        },
        "target_storage_categories": dict(sorted(module_storage.items())),
        "target_storage_total_bytes": sum(module_storage.values()),
        "expert_policy": {
            "talker_authoritative_cpu_bytes": residency_storage["pinned_cpu_authoritative_gpu_slot_cache"],
            "l4_slot_pool_bytes": EXPERT_SLOT_POOL_BYTES,
            "simultaneous_full_expert_residency": False,
        },
        "estimation_contract": {
            "BF16": "2 bytes per element; unchanged from source bytes",
            "Q8_0": "34 bytes per 32-element block, padded independently per final-dimension row",
            "Q4_K_M": "Phase 0 planning surrogate of 144 bytes per 256-element Q4_K block, padded independently per final-dimension row",
            "limitation": "actual GGUF sizes, allocator peaks, and CUDA workspaces replace estimates only after Phase 1 conversion and native instrumentation",
        },
        "phases": {
            "listening": phase(listening_weights, 768, 384, 128, 256, 256, 384),
            "voxzip_concurrent_listening": phase(listening_weights, 768, 512, 256, 256, 320, 512),
            "voxzip_sequential_endpoint": phase(listening_weights, 768, 384, 192, 256, 256, 384),
            "thinker_prefill": phase(thinker, 1024, 768, 256, 256, 320, 512),
            "talker_startup": phase(talking_weights, 512, 512, 256, 256, 320, 512),
            "talker_sustained_output": phase(talking_weights, 768, 384, 256, 256, 320, 512),
            "barge_in": phase(barge_in_weights, 768, 512, 256, 256, 384, 512),
            "context_rebuild": phase(thinker, 1280, 768, 256, 256, 320, 512),
        },
    }


def main() -> int:
    """Write one manifest-derived memory budget."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record = build(root)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
