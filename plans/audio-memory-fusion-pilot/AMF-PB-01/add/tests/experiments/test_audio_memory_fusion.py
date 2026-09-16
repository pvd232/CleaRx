"""Verify the frozen fusion inputs, vector construction, and result envelope."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from tools.research.audio_memory_fusion import (
    CONTRACT_PACKAGE_SHA256,
    EXPERIMENT_ID,
    FROZEN_SMOKE_IDENTITY,
    fuse_memory_vectors,
    mean_pool_ordered,
    validate_smoke_identity,
    validate_smoke_result,
)

COMMIT = "a" * 40


def valid_result() -> dict[str, Any]:
    """Return the smallest result accepted by the smoke evidence contract."""
    condition_common = {
        "input_positions": 12,
        "memory_positions": 1,
        "vocabulary_size": 100,
        "top_token_id": 7,
        "top_token_text": "zero",
        "logits_sha256": "b" * 64,
        "teacher_kl": 0.25,
        "teacher_top1_agreement": True,
    }
    return {
        "schema_version": "1.0.0",
        "experiment_id": EXPERIMENT_ID,
        "source": {
            "git_commit": COMMIT,
            "contract_package_sha256": CONTRACT_PACKAGE_SHA256,
        },
        "identity": FROZEN_SMOKE_IDENTITY.to_mapping(),
        "environment": {
            "device": "NVIDIA A100-SXM4-40GB",
            "device_total_bytes": 40_000_000_000,
            "torch_version": "test",
            "transformers_version": "test",
        },
        "observations": {
            "audio_feature_positions": 8,
            "transcript_token_positions": 1,
            "peak_gpu_allocated_bytes": 1,
            "model_load_seconds": 1.0,
            "execution_seconds": 2.0,
        },
        "conditions": {
            "full_audio": {
                "alpha": None,
                "input_positions": 19,
                "memory_positions": None,
                "vocabulary_size": 100,
                "top_token_id": 7,
                "top_token_text": "zero",
                "logits_sha256": "c" * 64,
            },
            "transcript_only": {"alpha": 0.0, **condition_common},
            "voxzip_addition": {"alpha": 1.0, **condition_common},
        },
    }


def test_accepts_frozen_smoke_identity() -> None:
    """Accept the exact corpus item, prompt, transcript, and pinned revisions."""
    validate_smoke_identity(FROZEN_SMOKE_IDENTITY.to_mapping())


def test_rejects_changed_smoke_identity() -> None:
    """Reject an otherwise valid identity after its transcript changes."""
    changed = FROZEN_SMOKE_IDENTITY.to_mapping()
    changed["transcript"] = "one"
    with pytest.raises(ValueError, match="frozen contract"):
        validate_smoke_identity(changed)


def test_constructs_equal_budget_memory_vectors() -> None:
    """Pool ordered intervals and establish alpha zero and one exactly."""
    audio = np.arange(8, dtype=np.float32).reshape(4, 2)
    pooled = mean_pool_ordered(audio, 2)
    np.testing.assert_array_equal(
        pooled,
        np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32),
    )
    text = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    np.testing.assert_array_equal(fuse_memory_vectors(pooled, text, 0.0), text)
    np.testing.assert_array_equal(fuse_memory_vectors(pooled, text, 1.0), text + pooled)


def test_rejects_invalid_fusion_inputs() -> None:
    """Reject mismatched position counts and non-finite vector values."""
    with pytest.raises(ValueError, match="share"):
        fuse_memory_vectors(np.zeros((2, 3)), np.zeros((1, 3)), 1.0)
    with pytest.raises(ValueError, match="finite"):
        mean_pool_ordered(np.array([[np.inf]], dtype=np.float32), 1)


def test_accepts_complete_smoke_result() -> None:
    """Accept a complete result with equal compressed memory budgets."""
    validate_smoke_result(valid_result(), COMMIT)


def test_rejects_unequal_memory_budgets() -> None:
    """Reject compressed conditions that use different memory-position counts."""
    changed = deepcopy(valid_result())
    changed["conditions"]["voxzip_addition"]["memory_positions"] = 2
    with pytest.raises(ValueError, match="equal memory positions"):
        validate_smoke_result(changed, COMMIT)


def test_rejects_nonfinite_comparison() -> None:
    """Reject a comparison metric that cannot be serialized as finite evidence."""
    changed = deepcopy(valid_result())
    changed["conditions"]["transcript_only"]["teacher_kl"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_smoke_result(changed, COMMIT)
