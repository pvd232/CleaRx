"""Shared strict primitives for CleaRx Phase 0 contract validators."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ContractViolation(ValueError):
    """Report one missing or incompatible Phase 0 contract field."""


def load_json(path: Path) -> dict[str, Any]:
    """Load one mapping-valued JSON contract."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path}: root must be an object")
    return value


def require(condition: bool, message: str) -> None:
    """Raise a stable contract failure when a predicate is false."""
    if not condition:
        raise ContractViolation(message)


def require_keys(record: dict[str, Any], keys: set[str], label: str) -> None:
    """Require every named key in one mapping."""
    missing = keys - set(record)
    require(not missing, f"{label}: missing fields {sorted(missing)}")


def require_sha256(value: Any, label: str) -> None:
    """Require one lowercase SHA-256 identity."""
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None, f"{label}: invalid SHA-256")


def require_commit(value: Any, label: str) -> None:
    """Require one full lowercase Git commit identity."""
    require(isinstance(value, str) and COMMIT.fullmatch(value) is not None, f"{label}: invalid commit")


def require_timestamp(value: Any, label: str) -> None:
    """Require an ISO-8601 timestamp with an explicit UTC offset."""
    require(isinstance(value, str), f"{label}: timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractViolation(f"{label}: invalid timestamp") from error
    require(parsed.tzinfo is not None, f"{label}: timestamp requires a UTC offset")


def validate_benchmark_schema(root: Path, record: dict[str, Any], label: str) -> None:
    """Validate one record against the frozen benchmark schema."""
    schema = load_json(root / "orchestration/schemas/benchmark.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(record), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.path) or "<root>"
        raise ContractViolation(f"{label}: {location}: {error.message}")
