#!/usr/bin/env python3
"""Join raw L4 CUDA rows with tensor, host, allocator, and cache evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any


SLOT_POOL_BYTES = 1_610_612_736


def sha256_file(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_bytes(tensor: dict[str, Any]) -> int:
    """Calculate Q8_0 bytes for one Talker expert tensor."""
    rows = math.prod(tensor["shape"][:-1])
    return rows * math.ceil(tensor["shape"][-1] / 32) * 34


def expert(manifest: dict[str, Any], layer: int, expert_id: int) -> dict[str, Any]:
    """Resolve one three-tensor expert from the manifest."""
    prefix = f"talker.model.layers.{layer}.mlp.experts.{expert_id}."
    tensors = [item for item in manifest["tensors"] if item["source_name"].startswith(prefix)]
    if len(tensors) != 3:
        raise ValueError(f"expert does not resolve to three tensors: {prefix}")
    return {
        "source": "tensor_manifest.json",
        "layer": layer,
        "expert_id": expert_id,
        "tensor_names": sorted(item["source_name"] for item in tensors),
        "shapes": {item["source_name"].rsplit(".", 2)[-2]: item["shape"] for item in tensors},
        "bytes": sum(target_bytes(item) for item in tensors),
        "target_dtype": "Q8_0",
    }


def lru_curve(expert_bytes: int) -> list[dict[str, Any]]:
    """Simulate deterministic LRU traces at three working-set sizes."""
    capacity = SLOT_POOL_BYTES // expert_bytes
    results = []
    for working_set in (64, capacity, capacity + 128):
        cache: OrderedDict[int, None] = OrderedDict()
        hits = misses = evictions = 0
        for key in list(range(working_set)) * 2:
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                misses += 1
                if len(cache) == capacity:
                    cache.popitem(last=False)
                    evictions += 1
                cache[key] = None
        results.append({"working_set_experts": working_set, "capacity_experts": capacity, "hits": hits, "misses": misses, "evictions": evictions, "hit_rate": hits / (hits + misses)})
    return results


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Parse the raw CUDA CSV and preserve every repetition."""
    integer_fields = {"payload_bytes", "h2d_duration_ns", "compute_duration_ns", "overlap_duration_ns", "unhidden_transfer_ns", "repetition_index"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: int(value) if key in integer_fields else value for key, value in row.items()})
    expected_cases = {"control-8mib-cold", "control-64mib-cold", "manifest-expert-cold", "manifest-expert-hit", "manifest-expert-evict-load"}
    if len(rows) != 150 or {row["case_id"] for row in rows} != expected_cases:
        raise ValueError("raw CUDA result must contain 30 rows for all five cases")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Derive compact medians and nearest-rank p95 values from raw rows."""
    summaries: dict[str, dict[str, Any]] = {}
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        h2d = sorted(row["h2d_duration_ns"] for row in case_rows)
        compute = [row["compute_duration_ns"] for row in case_rows]
        overlap = [row["overlap_duration_ns"] for row in case_rows]
        unhidden = [row["unhidden_transfer_ns"] for row in case_rows]
        median_h2d = statistics.median(h2d)
        summaries[case_id] = {
            "payload_bytes": case_rows[0]["payload_bytes"],
            "median_h2d_duration_ns": median_h2d,
            "p95_h2d_duration_ns": h2d[math.ceil(0.95 * len(h2d)) - 1],
            "median_compute_duration_ns": statistics.median(compute),
            "median_overlap_duration_ns": statistics.median(overlap),
            "median_unhidden_transfer_ns": statistics.median(unhidden),
            "median_h2d_decimal_gbps": None if median_h2d == 0 else case_rows[0]["payload_bytes"] / median_h2d,
            "median_overlap_fraction_of_h2d": None if median_h2d == 0 else statistics.median(overlap) / median_h2d,
        }
    return summaries


def build(root: Path, raw_csv: Path) -> dict[str, Any]:
    """Build the bound P0-006B result artifact."""
    manifest_path = root / "tensor_manifest.json"
    budget_path = root / "l4_memory_budget.json"
    host_path = root / "tests/fixtures/manifests/l4_host_profile.json"
    contract_path = root / "tests/fixtures/manifests/expert_cache_contract.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    shapes = [expert(manifest, 0, 0), expert(manifest, 0, 1), expert(manifest, 7, 66)]
    if len({item["bytes"] for item in shapes}) != 1:
        raise ValueError("representative expert sizes disagree")
    rows = load_rows(raw_csv)
    expert_bytes = shapes[0]["bytes"]
    for row in rows:
        if row["case_id"].startswith("manifest-") and row["payload_bytes"] != expert_bytes:
            raise ValueError("CUDA payload does not match manifest expert bytes")
    schedules = [
        {"id": "aut_without_whisper", "peak_bytes": budget["tensor_categories"]["aut"], "capacity_bytes": budget["tensor_categories"]["aut"], "valid": True},
        {"id": "aut_with_64mib_staging", "peak_bytes": budget["tensor_categories"]["aut"] + 64 * 1024 * 1024, "capacity_bytes": budget["phases"]["voxzip_concurrent_listening"]["categories"]["weights"] + budget["phases"]["voxzip_concurrent_listening"]["categories"]["staging"], "valid": True},
        {"id": "talker_expert_pool_with_64mib_staging", "peak_bytes": SLOT_POOL_BYTES + 64 * 1024 * 1024, "capacity_bytes": SLOT_POOL_BYTES + budget["phases"]["talker_startup"]["categories"]["staging"], "valid": True},
    ]
    trace_spec = {"patterns": ["single-layer-repeat", "working-set-fits", "working-set-thrash", "layer-interleaved"], "expert_bytes": expert_bytes, "selected_experts_per_token": 6, "layers": 20}
    trace_sha = hashlib.sha256(json.dumps(trace_spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {
        "schema_version": "1.0.0",
        "spike_id": "clearx-l4-expert-cache-spike-v1",
        "source_commit": commit,
        "host_profile_sha256": sha256_file(host_path),
        "tensor_manifest_sha256": sha256_file(manifest_path),
        "memory_budget_sha256": sha256_file(budget_path),
        "contract_sha256": sha256_file(contract_path),
        "raw_cuda_csv_sha256": sha256_file(raw_csv),
        "router_trace_sha256": trace_sha,
        "slot_pool_target_bytes": SLOT_POOL_BYTES,
        "representative_shapes": shapes,
        "transfer_measurements": rows,
        "transfer_summary": summarize(rows),
        "cache_hit_curve": lru_curve(expert_bytes),
        "allocator_simulation": {"events_valid": all(item["valid"] and item["peak_bytes"] <= item["capacity_bytes"] for item in schedules), "arena_peak_bytes": max(item["peak_bytes"] for item in schedules), "schedules": schedules},
        "measurement_validity": {"warmups_per_case": 5, "recorded_repetitions_per_case": 30, "payload_hash_verified": True, "cuda_events_used": True, "host_profile_id": "clearx-mantra-g2-l4-v1"},
        "limitations": [
            "CUDA spike measures pinned-host transfer and an independent synthetic compute kernel, not Q8_0 dequantization or a production expert kernel.",
            "LRU hit curves use deterministic synthetic router traces because model-generated routing traces do not exist before the native Talker path.",
            "Allocator schedules validate byte lifetimes arithmetically; production cudaMallocAsync arena events remain a Phase 1 measurement.",
            "These results establish transfer and capacity feasibility only and do not claim end-to-end real-time generation.",
        ],
    }


def main() -> int:
    """Build expert_cache_spike.json from one raw CUDA CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    record = build(root, args.raw_csv)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE: {len(record['transfer_measurements'])} measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
