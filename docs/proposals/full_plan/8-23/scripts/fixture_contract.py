"""Validate strict fixture descriptors and their joins to envelope manifests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from orchestration_lib import ContractError, load_record, load_schema, safe_repo_path, sha256_file, validate_schema


DESCRIPTOR_DTYPE = "application/vnd.clearx.fixture-descriptor+json"
DESCRIPTOR_SCHEMA = "orchestration/schemas/fixture_descriptor.schema.json"
PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "exact_discrete": {"profile_id": "exact_discrete", "mode": "exact", "atol": 0.0, "rtol": 0.0, "equal_nan": False, "max_mismatch_fraction": 0.0, "minimum_cosine_similarity": 1.0},
    "fp32_strict": {"profile_id": "fp32_strict", "mode": "allclose", "atol": 1e-6, "rtol": 1e-5, "equal_nan": False, "max_mismatch_fraction": 0.0, "minimum_cosine_similarity": 0.999999},
    "bf16_activation": {"profile_id": "bf16_activation", "mode": "allclose", "atol": 0.03125, "rtol": 0.03125, "equal_nan": False, "max_mismatch_fraction": 0.0, "minimum_cosine_similarity": 0.999},
    "bf16_accumulation": {"profile_id": "bf16_accumulation", "mode": "allclose", "atol": 0.0625, "rtol": 0.05, "equal_nan": False, "max_mismatch_fraction": 0.0, "minimum_cosine_similarity": 0.995},
    "waveform_fp32": {"profile_id": "waveform_fp32", "mode": "allclose", "atol": 0.001, "rtol": 0.001, "equal_nan": False, "max_mismatch_fraction": 0.0, "minimum_cosine_similarity": 0.9999},
}
PROVENANCE_FILES = {
    "upstream_lock_sha256": "upstream.lock",
    "tensor_manifest_sha256": "tensor_manifest.json",
    "evaluation_contract_sha256": "docs/evaluation_contract.md",
    "fixture_schema_sha256": DESCRIPTOR_SCHEMA,
}


def git_file_sha256(root: Path, commit: str, path: str) -> str:
    """Hash one repository file exactly as stored at a named commit."""
    safe_repo_path(root, path)
    try:
        content = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise ContractError(f"fixture generator is absent from recorded commit: {path}") from error
    import hashlib

    return hashlib.sha256(content).hexdigest()


def validate_descriptor(root: Path, descriptor_path: Path) -> dict[str, Any]:
    """Validate one descriptor's schema, local policy, and frozen identities."""
    descriptor = load_record(descriptor_path)
    validate_schema(descriptor, load_schema(root, "fixture_descriptor.schema.json"), str(descriptor_path))
    comparison = descriptor["comparison"]
    expected_profile = PROFILE_DEFINITIONS[comparison["profile_id"]]
    if comparison != expected_profile:
        raise ContractError(f"fixture tolerance profile changed: {comparison['profile_id']}")
    for field, relative_path in PROVENANCE_FILES.items():
        if descriptor["provenance"][field] != sha256_file(root / relative_path):
            raise ContractError(f"fixture provenance mismatch: {field}")
    provenance = descriptor["provenance"]
    upstream = load_record(root / "upstream.lock")
    try:
        model = next(repository for repository in upstream["repositories"] if repository["id"] == "qwen-model")
        config = next(identity for identity in upstream["file_identities"] if identity["repository"] == "qwen-model" and identity["path"] == "config.json")
    except StopIteration as error:
        raise ContractError("upstream lock omits the fixture source-model identity") from error
    if provenance["source_model_commit"] != model["commit"] or provenance["source_model_config_sha256"] != config["sha256"]:
        raise ContractError("fixture source-model identity mismatch")
    generator_path = safe_repo_path(root, provenance["generator_path"])
    if not generator_path.is_file() or sha256_file(generator_path) != provenance["generator_sha256"]:
        raise ContractError("fixture generator working-tree identity mismatch")
    if descriptor["fixture_kind"] == "model_boundary" and git_file_sha256(root, provenance["generator_commit"], provenance["generator_path"]) != provenance["generator_sha256"]:
        raise ContractError("fixture generator commit identity mismatch")
    if descriptor["case"]["class"] == "nominal" and descriptor["case"]["adversarial_tags"]:
        raise ContractError("nominal fixture declares adversarial tags")
    if descriptor["case"]["class"] == "adversarial" and not descriptor["case"]["adversarial_tags"]:
        raise ContractError("adversarial fixture omits adversarial tags")
    artifacts = descriptor["inputs"] + descriptor["outputs"]
    artifact_paths = [artifact["path"] for artifact in artifacts]
    logical_names = [artifact["logical_name"] for artifact in artifacts]
    if len(artifact_paths) != len(set(artifact_paths)) or len(logical_names) != len(set(logical_names)):
        raise ContractError("fixture descriptor has duplicate artifact identity")
    return descriptor


def validate_fixture_envelope(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate envelope bytes and join exactly one strict descriptor to outputs."""
    manifest = load_record(manifest_path)
    validate_schema(manifest, load_schema(root, "fixture.schema.json"), str(manifest_path))
    for artifact in manifest["artifacts"]:
        path = safe_repo_path(root, artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ContractError(f"fixture artifact identity mismatch: {artifact['path']}")
    descriptors = [artifact for artifact in manifest["artifacts"] if artifact["dtype"] == DESCRIPTOR_DTYPE]
    if len(descriptors) != 1:
        raise ContractError("fixture envelope must identify exactly one strict descriptor")
    descriptor_artifact = descriptors[0]
    descriptor = validate_descriptor(root, safe_repo_path(root, descriptor_artifact["path"]))
    if descriptor["descriptor_id"] != manifest["fixture_id"]:
        raise ContractError("fixture descriptor ID does not match envelope fixture_id")
    if descriptor["provenance"]["generator_commit"] != manifest["producer_commit"]:
        raise ContractError("fixture generator commit does not match envelope producer_commit")
    if descriptor["execution"]["seed"] != manifest["seed"]:
        raise ContractError("fixture seed does not match envelope seed")
    expected_sampling = {"profile_id": descriptor["execution"]["sampling_profile_id"], **descriptor["execution"]["sampling"]}
    if manifest.get("sampling") != expected_sampling:
        raise ContractError("fixture sampling does not match descriptor execution")
    envelope_artifacts = {
        artifact["path"]: {key: artifact[key] for key in ("path", "sha256", "dtype", "shape")}
        for artifact in manifest["artifacts"]
        if artifact is not descriptor_artifact
    }
    descriptor_artifacts = {
        artifact["path"]: {key: artifact[key] for key in ("path", "sha256", "dtype", "shape")}
        for artifact in descriptor["inputs"] + descriptor["outputs"]
    }
    if envelope_artifacts != descriptor_artifacts:
        raise ContractError("fixture descriptor artifacts do not match envelope artifacts")
    return descriptor
