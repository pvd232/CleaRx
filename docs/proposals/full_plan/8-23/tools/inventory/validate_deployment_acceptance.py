#!/usr/bin/env python3
"""Validate deployment gates against frozen artifacts and certified host identity."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.contract_validation import ContractViolation, load_json, require, require_commit, require_keys, require_sha256  # noqa: E402


ARTIFACT_FIELDS = {
    "host_profile": "tests/fixtures/manifests/l4_host_profile.json",
    "evaluation_contract": "docs/evaluation_contract.md",
    "corpus": "evaluation_corpus_manifest.json",
    "thresholds": "benchmark_thresholds.json",
}
REQUIRED_GATES = {"correctness", "semantic", "acoustic", "material_improvement", "latency"}


def file_sha256(path: Path) -> str:
    """Hash exact deployment-input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_deployment_acceptance(root: Path, path: Path) -> dict[str, Any]:
    """Validate artifact hashes, host identity, workloads, and frozen thresholds."""
    record = load_json(path)
    require_keys(record, {"schema_version", "acceptance_id", "source_commit", "artifacts", "certified_host", "workloads", "gates", "alternate_hosts"}, "deployment acceptance")
    require(record["schema_version"] == "1.0.0", "deployment acceptance: unsupported schema version")
    require_commit(record["source_commit"], "deployment acceptance.source_commit")
    for field, relative_path in ARTIFACT_FIELDS.items():
        require(field in record["artifacts"], f"deployment acceptance: {field} binding is absent")
        require_sha256(record["artifacts"][field]["sha256"], f"deployment acceptance.{field}")
        require(record["artifacts"][field]["path"] == relative_path, f"deployment acceptance: {field} path changed")
        require(record["artifacts"][field]["sha256"] == file_sha256(root / relative_path), f"deployment acceptance: {field} hash mismatch")
    host = load_json(root / ARTIFACT_FIELDS["host_profile"])
    require_keys(record["certified_host"], {"profile_id", "gpu_uuid", "machine_type", "zone"}, "deployment acceptance.certified_host")
    require(record["certified_host"]["profile_id"] == host["profile_id"] and record["certified_host"]["gpu_uuid"] == host["gpu"]["uuid"], "deployment acceptance: certified host identity mismatch")
    require(record["certified_host"]["machine_type"] == host["gce"]["machine_type"] and record["certified_host"]["zone"] == host["gce"]["zone"], "deployment acceptance: certified host placement mismatch")
    require(isinstance(record["workloads"], list) and record["workloads"] and all(isinstance(item, str) and item for item in record["workloads"]), "deployment acceptance: workloads are absent")
    thresholds = load_json(root / ARTIFACT_FIELDS["thresholds"])
    require(set(record["gates"]) == REQUIRED_GATES, "deployment acceptance: gate set is incomplete")
    for gate, binding in record["gates"].items():
        require_keys(binding, {"metric", "threshold_path", "corpus_slices", "sampling_ids", "workload_id"}, f"deployment gate {gate}")
        category = "acoustic" if gate == "acoustic" else gate
        require(binding["metric"] == thresholds["thresholds"][category]["metric"], f"deployment gate {gate}: metric changed")
        require(binding["threshold_path"] == f"thresholds.{category}", f"deployment gate {gate}: threshold path changed")
        require(isinstance(binding["corpus_slices"], list) and binding["corpus_slices"], f"deployment gate {gate}: corpus slices are absent")
        require(isinstance(binding["sampling_ids"], list) and binding["sampling_ids"], f"deployment gate {gate}: sampling IDs are absent")
        require(binding["workload_id"] in record["workloads"], f"deployment gate {gate}: workload is undeclared")
    require(all(host_record.get("classification") == "exploratory_only" for host_record in record["alternate_hosts"]), "deployment acceptance: alternate host is not exploratory")
    return record


def main() -> int:
    """Validate one deployment-acceptance JSON file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        record = validate_deployment_acceptance(ROOT, args.path)
    except (ContractViolation, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {record['acceptance_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
