"""Validate CleaRx orchestration records and their cross-file invariants."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "planned": {"blocked", "ready"},
    "blocked": {"ready", "cancelled"},
    "ready": {"running", "blocked", "cancelled"},
    "running": {"validating", "failed", "cancelled"},
    "validating": {"complete", "failed"},
    "failed": {"ready", "cancelled"},
    "complete": set(),
    "cancelled": set(),
}


class ContractError(ValueError):
    """Report one violated orchestration relationship."""


def repository_root(start: Path | None = None) -> Path:
    """Resolve the Git repository that owns the orchestration files."""
    cwd = start or Path.cwd()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for exact bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without normalizing its bytes."""
    return sha256_bytes(path.read_bytes())


def git_file_bytes(root: Path, commit: str, path: str) -> bytes:
    """Read one repository file exactly as stored at a recorded commit."""
    safe_repo_path(root, path)
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ContractError(f"file is absent from recorded commit: {path}") from error


def load_record(path: Path) -> Any:
    """Load a JSON or YAML record according to its suffix."""
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def load_schema(root: Path, name: str) -> dict[str, Any]:
    """Load one schema from the canonical schema directory."""
    return load_record(root / "orchestration" / "schemas" / name)


def validate_schema(instance: Any, schema: dict[str, Any], label: str) -> None:
    """Raise a compact contract error for the first JSON Schema violation."""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.path) or "<root>"
        raise ContractError(f"{label}: {location}: {error.message}")


def safe_repo_path(root: Path, value: str) -> Path:
    """Resolve a repository-relative path and reject every escape."""
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ContractError(f"path leaves repository: {value}")
    resolved = (root / pure).resolve()
    if not resolved.is_relative_to(root):
        raise ContractError(f"path leaves repository after resolution: {value}")
    return resolved


def scopes_overlap(left: str, right: str) -> bool:
    """Return whether two owned paths are identical or ancestor-related."""
    left_parts = PurePosixPath(left.rstrip("/")).parts
    right_parts = PurePosixPath(right.rstrip("/")).parts
    shared = min(len(left_parts), len(right_parts))
    return left_parts[:shared] == right_parts[:shared]


def validate_schemas(root: Path) -> None:
    """Check every canonical JSON Schema against Draft 2020-12."""
    for path in sorted((root / "orchestration" / "schemas").glob("*.schema.json")):
        Draft202012Validator.check_schema(load_record(path))


def validate_plan_and_state(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate plan, packets, state, digests, paths, DAG, and ownership."""
    plan_path = root / "orchestration" / "plan.yaml"
    state_path = root / "orchestration" / "state.json"
    plan = load_record(plan_path)
    state = load_record(state_path)
    validate_schema(plan, load_schema(root, "plan.schema.json"), "plan.yaml")
    validate_schema(state, load_schema(root, "state.schema.json"), "state.json")

    nodes = plan["nodes"]
    ids = [node["id"] for node in nodes]
    if len(ids) != len(set(ids)):
        raise ContractError("plan contains duplicate packet IDs")
    node_by_id = {node["id"]: node for node in nodes}
    packets: dict[str, Any] = {}
    packet_schema = load_schema(root, "work_packet.schema.json")

    for node in nodes:
        packet_path = safe_repo_path(root, node["packet_path"])
        if not packet_path.is_file():
            raise ContractError(f"missing packet file: {node['packet_path']}")
        actual_digest = sha256_file(packet_path)
        if actual_digest != node["packet_sha256"]:
            raise ContractError(f"packet digest mismatch: {node['id']}")
        packet = load_record(packet_path)
        validate_schema(packet, packet_schema, node["packet_path"])
        if packet["id"] != node["id"] or packet["phase"] != node["phase"]:
            raise ContractError(f"plan identity mismatch: {node['id']}")
        if packet["depends_on"] != node["depends_on"]:
            raise ContractError(f"plan dependency mismatch: {node['id']}")
        for path in packet["execution"]["write_scope"] + packet["outputs"]:
            safe_repo_path(root, path)
        result_schema = safe_repo_path(root, packet["evidence"]["result_schema"])
        if not result_schema.is_file():
            raise ContractError(f"missing result schema for {node['id']}")
        for input_record in packet["inputs"]:
            input_path = safe_repo_path(root, input_record["path"])
            if "sha256" in input_record:
                if not input_path.is_file():
                    raise ContractError(f"missing repository input for {node['id']}: {input_record['path']}")
                if sha256_file(input_path) != input_record["sha256"]:
                    raise ContractError(f"repository input digest mismatch for {node['id']}: {input_record['path']}")
            elif input_record["producer_packet"] not in packet["depends_on"]:
                raise ContractError(f"producer input is not a direct dependency of {node['id']}")
        packets[node["id"]] = packet

    for node in nodes:
        missing = set(node["depends_on"]) - set(node_by_id)
        if missing:
            raise ContractError(f"{node['id']} has missing dependencies: {sorted(missing)}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(packet_id: str) -> None:
        """Traverse one node and reject a dependency back-edge."""
        if packet_id in visiting:
            raise ContractError(f"dependency cycle includes {packet_id}")
        if packet_id in visited:
            return
        visiting.add(packet_id)
        for dependency in node_by_id[packet_id]["depends_on"]:
            visit(dependency)
        visiting.remove(packet_id)
        visited.add(packet_id)

    for packet_id in ids:
        visit(packet_id)

    by_wave: dict[str, list[str]] = {}
    for node in nodes:
        by_wave.setdefault(node["wave"], []).append(node["id"])
    for wave, wave_ids in by_wave.items():
        for index, left_id in enumerate(wave_ids):
            for right_id in wave_ids[index + 1 :]:
                for left_scope in packets[left_id]["execution"]["write_scope"]:
                    for right_scope in packets[right_id]["execution"]["write_scope"]:
                        if scopes_overlap(left_scope, right_scope):
                            raise ContractError(
                                f"wave {wave} has overlapping scopes: {left_id}:{left_scope} and {right_id}:{right_scope}"
                            )

    if sha256_file(plan_path) != state["plan_sha256"]:
        raise ContractError("state plan digest does not match plan.yaml")
    if set(state["packets"]) != set(ids):
        raise ContractError("state packet IDs do not match plan packet IDs")

    for packet_id, entry in state["packets"].items():
        node = node_by_id[packet_id]
        if entry["packet_path"] != node["packet_path"] or entry["packet_sha256"] != node["packet_sha256"]:
            raise ContractError(f"state packet binding mismatch: {packet_id}")
        previous: str | None = None
        for event in entry["history"]:
            if event["packet_sha256"] != entry["packet_sha256"]:
                raise ContractError(f"transition digest mismatch: {packet_id}")
            if event["previous_state"] != previous:
                raise ContractError(f"transition history is discontinuous: {packet_id}")
            next_state = event["next_state"]
            if previous is None:
                if next_state != "planned":
                    raise ContractError(f"first lifecycle state must be planned: {packet_id}")
            elif next_state not in ALLOWED_TRANSITIONS[previous]:
                raise ContractError(f"invalid transition for {packet_id}: {previous} -> {next_state}")
            previous = next_state
        if previous != entry["status"]:
            raise ContractError(f"history does not end at current state: {packet_id}")
        if entry["status"] in {"ready", "running", "validating", "complete"}:
            incomplete = [dep for dep in node["depends_on"] if state["packets"][dep]["status"] != "complete"]
            if incomplete:
                raise ContractError(f"{packet_id} advanced before dependencies completed: {incomplete}")

    for gate in plan["gates"]:
        missing = set(gate["requires"]) - set(node_by_id)
        if missing:
            raise ContractError(f"gate {gate['id']} references missing packets: {sorted(missing)}")
    return plan, state, packets


def validate_result_file(root: Path, result_path: Path) -> dict[str, Any]:
    """Validate one result and bind it to packet bytes at its recorded commit."""
    plan, state, packets = validate_plan_and_state(root)
    result = load_record(result_path)
    validate_schema(result, load_schema(root, "result.schema.json"), str(result_path))
    packet_id = result["packet_id"]
    node_by_id = {node["id"]: node for node in plan["nodes"]}
    if packet_id not in node_by_id:
        raise ContractError(f"result references unknown packet: {packet_id}")
    node = node_by_id[packet_id]
    packet_bytes = git_file_bytes(root, result["git_commit"], node["packet_path"])
    reconstructed_digest = sha256_bytes(packet_bytes)
    if result["packet_sha256"] != node["packet_sha256"] or result["packet_sha256"] != reconstructed_digest:
        raise ContractError(f"result packet binding mismatch: {packet_id}")
    expected_artifacts = set(packets[packet_id]["outputs"])
    actual_artifacts = {artifact["path"] for artifact in result["artifacts"]}
    if actual_artifacts != expected_artifacts:
        raise ContractError(f"artifact set mismatch for {packet_id}")
    for artifact in result["artifacts"]:
        artifact_bytes = git_file_bytes(root, result["git_commit"], artifact["path"])
        if len(artifact_bytes) != artifact["size_bytes"] or sha256_bytes(artifact_bytes) != artifact["sha256"]:
            raise ContractError(f"artifact identity mismatch: {artifact['path']}")
    if result["outcome"] == "complete":
        if any(command["exit_code"] != 0 for command in result["commands"]):
            raise ContractError("complete result contains a failed command")
        required = set(packets[packet_id]["acceptance"]["required_checks"])
        if set(result["required_checks"]) != required or not all(result["required_checks"].values()):
            raise ContractError("complete result does not pass every required check")
        incomplete = [dep for dep in node["depends_on"] if state["packets"][dep]["status"] != "complete"]
        if incomplete:
            raise ContractError(f"result dependencies are incomplete: {incomplete}")
    return result


def validate_all(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run every static orchestration validation owned by Bootstrap."""
    validate_schemas(root)
    return validate_plan_and_state(root)
