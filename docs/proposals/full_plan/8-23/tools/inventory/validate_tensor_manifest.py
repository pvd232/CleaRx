#!/usr/bin/env python3
"""Validate complete source identities, tensor metadata, target policy, and totals."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.contract_validation import ContractViolation, load_json, require, require_commit, require_keys, require_sha256  # noqa: E402


MODULES = {"thinker", "aut", "vision", "talker", "mtp", "code2wav"}
INCLUDED_BACKENDS = {"l4_cuda", "cpu"}
TARGET_DTYPES = {"Q4_K_M", "Q8_0", "BF16"}


def validate_architecture(value: dict[str, Any]) -> None:
    """Require the released layer, expert, codec, and streaming facts."""
    require(value["thinker"]["layers"] == 48 and value["thinker"]["experts"] == 128 and value["thinker"]["active_experts_per_token"] == 8, "tensor manifest: Thinker architecture changed")
    require(value["aut"]["layers"] == 32 and value["aut"]["mel_bins"] == 128, "tensor manifest: AuT architecture changed")
    require(value["vision"]["layers"] == 27 and value["vision"]["deployment_disposition"] == "removed", "tensor manifest: vision disposition changed")
    require(value["talker"]["layers"] == 20 and value["talker"]["experts"] == 128 and value["talker"]["active_experts_per_token"] == 6, "tensor manifest: Talker architecture changed")
    require(value["talker"]["code_groups"] == 16 and value["mtp"]["residual_codebooks"] == 15, "tensor manifest: 16-codebook hierarchy changed")
    require(value["code2wav"]["quantizers"] == 16 and value["code2wav"]["sliding_window"] == 72, "tensor manifest: Code2Wav contract changed")
    require(value["speaker"]["ids"] == {"chelsie": 2301, "ethan": 2302, "aiden": 2303}, "tensor manifest: speaker IDs changed")


def validate_tensor_manifest(path: Path) -> dict[str, Any]:
    """Validate one complete P0-002 manifest."""
    record = load_json(path)
    require_keys(record, {"schema_version", "manifest_id", "generated_at", "source", "target_policy", "architecture", "shards", "summary", "tensors"}, "tensor manifest")
    require(record["schema_version"] == "1.0.0", "tensor manifest: unsupported schema")
    require_keys(record["source"], {"repository", "commit", "config_path", "config_sha256", "tensor_checksum"}, "tensor manifest.source")
    require(record["source"]["repository"] == "Qwen/Qwen3-Omni-30B-A3B-Instruct", "tensor manifest: source repository changed")
    require_commit(record["source"]["commit"], "tensor manifest.source.commit")
    require(record["source"]["commit"] == "26291f793822fb6be9555850f06dfe95f2d7e695", "tensor manifest: source commit changed")
    require_sha256(record["source"]["config_sha256"], "tensor manifest.source.config_sha256")
    validate_architecture(record["architecture"])
    require(isinstance(record["shards"], list) and len(record["shards"]) == 15, "tensor manifest: expected 15 source shards")
    shard_by_name: dict[str, dict[str, Any]] = {}
    for shard in record["shards"]:
        require_keys(shard, {"path", "sha256", "size_bytes", "tensor_count"}, "tensor manifest.shard")
        require_sha256(shard["sha256"], f"tensor manifest shard {shard['path']}")
        require(shard["path"] not in shard_by_name and shard["size_bytes"] > 0 and shard["tensor_count"] > 0, "tensor manifest: invalid or duplicate shard")
        shard_by_name[shard["path"]] = shard
    tensors = record["tensors"]
    require(isinstance(tensors, list) and len(tensors) >= 28000, "tensor manifest: tensor inventory is incomplete")
    names: set[str] = set()
    targets: set[str] = set()
    counts = Counter()
    sizes = Counter()
    shard_counts = Counter()
    for tensor in tensors:
        require_keys(tensor, {"source_name", "source_shard", "source_shard_sha256", "source_tensor_sha256", "source_byte_range", "shape", "source_dtype", "source_layout", "source_strides_elements", "size_bytes", "module", "roles", "disposition", "target_gguf_name", "target_tensor_sha256", "target_checksum_status", "target_dtype", "quantization", "backend", "residency"}, "tensor manifest.tensor")
        name = tensor["source_name"]
        require(isinstance(name, str) and name and name not in names, "tensor manifest: duplicate or empty source name")
        names.add(name)
        require(tensor["source_shard"] in shard_by_name and tensor["source_shard_sha256"] == shard_by_name[tensor["source_shard"]]["sha256"], f"tensor manifest: shard join failed for {name}")
        require_sha256(tensor["source_tensor_sha256"], f"tensor manifest tensor {name}")
        require(isinstance(tensor["shape"], list) and all(isinstance(value, int) and value >= 0 for value in tensor["shape"]), f"tensor manifest: invalid shape for {name}")
        require(tensor["source_layout"] == "row_major_contiguous" and len(tensor["source_strides_elements"]) == len(tensor["shape"]), f"tensor manifest: layout is absent for {name}")
        require(tensor["size_bytes"] == tensor["source_byte_range"][1] - tensor["source_byte_range"][0] and tensor["size_bytes"] > 0, f"tensor manifest: byte range mismatch for {name}")
        module = tensor["module"]
        require(module in MODULES and isinstance(tensor["roles"], list) and tensor["roles"], f"tensor manifest: module/role missing for {name}")
        if module == "vision":
            require(tensor["disposition"] == "removed" and tensor["backend"] == "absent" and tensor["target_gguf_name"] is None, f"tensor manifest: vision tensor not removed: {name}")
            require(tensor["target_tensor_sha256"] is None and tensor["target_checksum_status"] == "removed_not_applicable", f"tensor manifest: removed target checksum state changed: {name}")
        else:
            require(tensor["disposition"] == "included" and tensor["backend"] in INCLUDED_BACKENDS, f"tensor manifest: backend missing for {name}")
            require(tensor["target_dtype"] in TARGET_DTYPES and isinstance(tensor["target_gguf_name"], str) and tensor["target_gguf_name"].startswith("clearx."), f"tensor manifest: target mapping missing for {name}")
            require(tensor["target_gguf_name"] not in targets, f"tensor manifest: duplicate target name for {name}")
            require(tensor["target_tensor_sha256"] is None and tensor["target_checksum_status"] == "pending_phase1_conversion", f"tensor manifest: target checksum overclaimed for {name}")
            targets.add(tensor["target_gguf_name"])
        counts[module] += 1
        sizes[module] += tensor["size_bytes"]
        shard_counts[tensor["source_shard"]] += 1
    require(set(counts) == MODULES, "tensor manifest: not all modules are covered")
    require(dict(sorted(counts.items())) == record["summary"]["by_module_count"], "tensor manifest: module counts do not reconcile")
    require(dict(sorted(sizes.items())) == record["summary"]["by_module_bytes"], "tensor manifest: module bytes do not reconcile")
    require(sum(sizes.values()) == record["summary"]["source_bytes"] and len(tensors) == record["summary"]["tensor_count"], "tensor manifest: summary totals do not reconcile")
    require(all(shard_counts[name] == shard["tensor_count"] for name, shard in shard_by_name.items()), "tensor manifest: shard tensor counts do not reconcile")
    return record


def main() -> int:
    """Validate one tensor manifest from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        record = validate_tensor_manifest(args.path)
    except (ContractViolation, KeyError, OSError, TypeError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {record['summary']['tensor_count']} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
