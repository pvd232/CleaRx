"""Define the frozen inputs, fusion operation, and evidence checks for the smoke."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

EXPERIMENT_ID: Final = "audio-memory-fusion-smoke-v1"
CONTRACT_PACKAGE_SHA256: Final = (
    "f9566b7da715ae61c0cc1c6cc913554ec22d21ab892169a3b7bb8208198262f8"
)
MODEL_REPOSITORY: Final = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_COMMIT: Final = "26291f793822fb6be9555850f06dfe95f2d7e695"
TRANSFORMERS_COMMIT: Final = "7d9754a05193eb79b1d86aa744b622b8068008cd"
SMOKE_ITEM_ID: Final = "fsdd-george-0"
SMOKE_TRANSCRIPT: Final = "zero"
SMOKE_PROMPT: Final = "Repeat the spoken digit and nothing else."
SMOKE_AUDIO_URL: Final = (
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/"
    "26eb9aaf76e81b692f806f9140c2d2777410d7a1/recordings/0_george_0.wav"
)
SMOKE_AUDIO_SHA256: Final = (
    "228ab63fccdf262d2e05817b6ec918b15e7d9e4bfb6bb20183c46ae088405240"
)
SMOKE_SEED: Final = 7
SMOKE_ALPHAS: Final = (0.0, 1.0)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SmokeIdentity:
    """Identify every frozen input that distinguishes this smoke execution."""

    item_id: str = SMOKE_ITEM_ID
    transcript: str = SMOKE_TRANSCRIPT
    prompt: str = SMOKE_PROMPT
    audio_url: str = SMOKE_AUDIO_URL
    audio_sha256: str = SMOKE_AUDIO_SHA256
    seed: int = SMOKE_SEED
    alphas: tuple[float, ...] = SMOKE_ALPHAS
    model_repository: str = MODEL_REPOSITORY
    model_commit: str = MODEL_COMMIT
    transformers_commit: str = TRANSFORMERS_COMMIT

    def to_mapping(self) -> dict[str, object]:
        """Return the JSON-compatible identity stored in the result."""
        value = asdict(self)
        value["alphas"] = list(self.alphas)
        return value


FROZEN_SMOKE_IDENTITY: Final = SmokeIdentity()


def validate_smoke_identity(value: Mapping[str, object]) -> None:
    """Reject an input identity that differs from the approved smoke case."""
    if dict(value) != FROZEN_SMOKE_IDENTITY.to_mapping():
        raise ValueError("smoke identity differs from the frozen contract")


def mean_pool_ordered(sequence: np.ndarray, output_length: int) -> np.ndarray:
    """Mean-pool one chronological sequence into ordered nonempty intervals."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("sequence must have shape [positions, hidden_size]")
    if output_length < 1 or output_length > values.shape[0]:
        raise ValueError("output_length must select nonempty chronological intervals")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")
    intervals = np.array_split(values, output_length, axis=0)
    return np.stack([interval.mean(axis=0) for interval in intervals]).astype(
        np.float32,
        copy=False,
    )


def fuse_memory_vectors(
    audio_vectors: np.ndarray,
    text_vectors: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return transcript vectors plus one scaled aligned audio vector per position."""
    audio = np.asarray(audio_vectors, dtype=np.float32)
    text = np.asarray(text_vectors, dtype=np.float32)
    if audio.ndim != 2 or text.ndim != 2 or audio.shape != text.shape:
        raise ValueError("audio and text vectors must share [positions, hidden_size]")
    if audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError("memory vectors must contain at least one value")
    if not np.isfinite(audio).all() or not np.isfinite(text).all():
        raise ValueError("memory vectors must contain only finite values")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    return (text + np.float32(alpha) * audio).astype(np.float32, copy=False)


def float_array_sha256(value: np.ndarray) -> str:
    """Hash a floating array after canonical little-endian float32 conversion."""
    canonical = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def validate_smoke_result(value: Mapping[str, Any], expected_commit: str) -> None:
    """Reject a smoke result that cannot support the contract's acceptance claim."""
    if _GIT_COMMIT.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a full lowercase Git commit")
    _require_equal(value, "schema_version", "1.0.0")
    _require_equal(value, "experiment_id", EXPERIMENT_ID)

    source = _require_mapping(value, "source")
    _require_equal(source, "git_commit", expected_commit)
    _require_equal(source, "contract_package_sha256", CONTRACT_PACKAGE_SHA256)

    identity = _require_mapping(value, "identity")
    validate_smoke_identity(identity)

    environment = _require_mapping(value, "environment")
    device = _require_text(environment, "device")
    if "A100" not in device:
        raise ValueError("environment.device must identify an A100")
    _require_positive_int(environment, "device_total_bytes")
    _require_text(environment, "torch_version")
    _require_text(environment, "transformers_version")

    observations = _require_mapping(value, "observations")
    audio_positions = _require_positive_int(observations, "audio_feature_positions")
    transcript_positions = _require_positive_int(
        observations,
        "transcript_token_positions",
    )
    _require_positive_int(observations, "peak_gpu_allocated_bytes")
    _require_nonnegative_finite(observations, "model_load_seconds")
    _require_nonnegative_finite(observations, "execution_seconds")

    conditions = _require_mapping(value, "conditions")
    expected_names = {"full_audio", "transcript_only", "voxzip_addition"}
    if set(conditions) != expected_names:
        raise ValueError("conditions must contain exactly the three smoke conditions")
    full = _validate_condition(conditions, "full_audio", expected_alpha=None)
    text = _validate_condition(conditions, "transcript_only", expected_alpha=0.0)
    fused = _validate_condition(conditions, "voxzip_addition", expected_alpha=1.0)

    if text["input_positions"] != fused["input_positions"]:
        raise ValueError("compressed conditions must have equal input positions")
    if text["memory_positions"] != fused["memory_positions"]:
        raise ValueError("compressed conditions must have equal memory positions")
    if text["memory_positions"] != transcript_positions:
        raise ValueError("compressed memory positions must equal transcript tokens")
    if audio_positions < transcript_positions:
        raise ValueError(
            "audio features cannot supply every transcript memory position"
        )
    if full["input_positions"] <= text["input_positions"]:
        raise ValueError("smoke input must demonstrate position compression")


def _validate_condition(
    conditions: Mapping[str, Any],
    name: str,
    expected_alpha: float | None,
) -> dict[str, Any]:
    """Validate one condition and return its normalized comparison fields."""
    condition = _require_mapping(conditions, name)
    alpha = condition.get("alpha")
    if alpha != expected_alpha:
        raise ValueError(f"{name}.alpha differs from the frozen condition")
    input_positions = _require_positive_int(condition, "input_positions")
    vocabulary_size = _require_positive_int(condition, "vocabulary_size")
    _require_nonnegative_int(condition, "top_token_id")
    _require_text(condition, "top_token_text", allow_empty=True)
    digest = _require_text(condition, "logits_sha256")
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name}.logits_sha256 must be lowercase SHA-256")

    memory_positions = condition.get("memory_positions")
    normalized: dict[str, Any] = {
        "input_positions": input_positions,
        "memory_positions": memory_positions,
        "vocabulary_size": vocabulary_size,
    }
    if expected_alpha is None:
        if memory_positions is not None:
            raise ValueError("full_audio.memory_positions must be null")
        return normalized
    if not isinstance(memory_positions, int) or isinstance(memory_positions, bool):
        raise TypeError(f"{name}.memory_positions must be an integer")
    if memory_positions < 1:
        raise ValueError(f"{name}.memory_positions must be positive")
    normalized["memory_positions"] = memory_positions
    _require_nonnegative_finite(condition, "teacher_kl")
    if not isinstance(condition.get("teacher_top1_agreement"), bool):
        raise TypeError(f"{name}.teacher_top1_agreement must be Boolean")
    return normalized


def _require_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one mapping-valued field."""
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise TypeError(f"{key} must be an object")
    return selected


def _require_equal(value: Mapping[str, Any], key: str, expected: object) -> None:
    """Require one field to equal its frozen value."""
    if value.get(key) != expected:
        raise ValueError(f"{key} differs from the frozen value")


def _require_text(
    value: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Return one text field after enforcing its empty-value policy."""
    selected = value.get(key)
    if not isinstance(selected, str) or (not allow_empty and not selected):
        raise ValueError(f"{key} must be text")
    return selected


def _require_positive_int(value: Mapping[str, Any], key: str) -> int:
    """Return one strictly positive integer field."""
    selected = value.get(key)
    if not isinstance(selected, int) or isinstance(selected, bool) or selected < 1:
        raise ValueError(f"{key} must be a positive integer")
    return selected


def _require_nonnegative_int(value: Mapping[str, Any], key: str) -> int:
    """Return one nonnegative integer field."""
    selected = value.get(key)
    if not isinstance(selected, int) or isinstance(selected, bool) or selected < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return selected


def _require_nonnegative_finite(value: Mapping[str, Any], key: str) -> float:
    """Return one finite nonnegative numeric field."""
    selected = value.get(key)
    if not isinstance(selected, int | float) or isinstance(selected, bool):
        raise TypeError(f"{key} must be numeric")
    result = float(selected)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{key} must be finite and nonnegative")
    return result
