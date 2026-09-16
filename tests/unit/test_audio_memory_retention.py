"""Verify equal-budget acoustic-position allocation and retained evidence."""

# pyright: reportMissingImports=false

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from clearx.audio_memory.gate import FEATURE_NAMES
from clearx.audio_memory.retention import (
    ALPHA,
    CONCENTRATED_POSITION_COUNT,
    FIXED_POSITION_COUNT,
    MODEL_COMMIT,
    MODEL_REPOSITORY,
    POSITION_COUNTS,
    RAVDESS_ARCHIVE_MD5,
    TEST_ACTORS,
    TRAIN_ACTORS,
    TRANSFORMERS_COMMIT,
    mean_pool_ordered,
    retention_payload,
    validate_result,
)


def _item(actor: str, index: int, split: str, benefit: float) -> dict[str, Any]:
    """Build one synthetic measured item with a known four-position benefit."""
    transcript_kl = 10.0 + index / 10.0
    features = {
        name: (index + position + 1.0) / (position + 2.0)
        for position, name in enumerate(FEATURE_NAMES)
    }
    return {
        "item_id": f"actor-{actor}-item-{index:02d}",
        "actor_id": actor,
        "split": split,
        "memory_quality": "unused-label",
        "gate_features": features,
        "conditions": {
            "positions_0": {
                "acoustic_positions": 0,
                "alpha": ALPHA,
                "teacher_kl": transcript_kl,
            },
            "positions_2": {
                "acoustic_positions": 2,
                "alpha": ALPHA,
                "teacher_kl": transcript_kl - benefit / 2.0,
            },
            "positions_4": {
                "acoustic_positions": 4,
                "alpha": ALPHA,
                "teacher_kl": transcript_kl - benefit,
            },
        },
    }


def _items() -> list[dict[str, Any]]:
    """Return four training speakers and two held-out speakers."""
    observations: list[dict[str, Any]] = []
    for actor_index, actor in enumerate(TRAIN_ACTORS):
        observations.extend(
            _item(actor, actor_index * 2 + offset, "train", 0.2 + actor_index + offset)
            for offset in range(2)
        )
    observations.extend(
        [
            _item("05", 20, "test", 0.1),
            _item("05", 21, "test", 2.0),
            _item("06", 22, "test", 0.2),
            _item("06", 23, "test", 3.0),
        ]
    )
    return observations


def _result() -> dict[str, Any]:
    """Build one internally consistent retained-result fixture."""
    observations = _items()
    return {
        "status": "completed",
        "source": {"experiment_commit": "a" * 40},
        "identity": {
            "model_repository": MODEL_REPOSITORY,
            "model_commit": MODEL_COMMIT,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "ravdess_archive_md5": RAVDESS_ARCHIVE_MD5,
            "alpha": ALPHA,
            "position_counts": list(POSITION_COUNTS),
            "fixed_position_count": FIXED_POSITION_COUNT,
            "concentrated_position_count": CONCENTRATED_POSITION_COUNT,
            "train_actors": list(TRAIN_ACTORS),
            "test_actors": list(TEST_ACTORS),
        },
        "environment": {"device": "NVIDIA A100-SXM4-40GB"},
        "items": observations,
        "retention": retention_payload(observations),
    }


def test_frozen_protocol_has_equal_budget_arms() -> None:
    """Two fixed positions equal four positions assigned to half the items."""
    assert POSITION_COUNTS == (0, 2, 4)
    assert ALPHA == 1.0
    assert 4 * FIXED_POSITION_COUNT == 2 * CONCENTRATED_POSITION_COUNT


def test_mean_pool_ordered_preserves_order_and_requested_count() -> None:
    """Consecutive bins produce the requested number of ordered averages."""
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)

    pooled = mean_pool_ordered(vectors, 2)

    assert pooled.shape == (2, 4)
    assert np.array_equal(pooled[0], vectors[:3].mean(axis=0))
    assert np.array_equal(pooled[1], vectors[3:].mean(axis=0))
    assert mean_pool_ordered(vectors, 0).shape == (0, 4)


def test_mean_pool_ordered_rejects_invalid_inputs() -> None:
    """Pooling rejects empty, non-finite, negative, and oversized requests."""
    with pytest.raises(ValueError):
        mean_pool_ordered(np.empty((0, 4), dtype=np.float32), 0)
    with pytest.raises(ValueError):
        mean_pool_ordered(np.asarray([[np.nan]], dtype=np.float32), 1)
    with pytest.raises(ValueError):
        mean_pool_ordered(np.ones((2, 4), dtype=np.float32), -1)
    with pytest.raises(ValueError):
        mean_pool_ordered(np.ones((2, 4), dtype=np.float32), 3)


def test_retention_payload_fits_train_and_allocates_equal_test_budget() -> None:
    """The held-out adaptive allocation uses exactly the fixed arm's budget."""
    payload = retention_payload(_items())
    evaluation = payload["evaluation"]

    assert payload["model"]["training_speakers"] == list(TRAIN_ACTORS)
    assert evaluation["fixed_total_acoustic_positions"] == 8
    assert evaluation["adaptive_total_acoustic_positions"] == 8
    assert evaluation["adaptive_selected_count"] == 2
    assert sorted(payload["adaptive_allocation"].values()) == [0, 0, 4, 4]


def test_payload_rejects_unequal_or_incomplete_position_grids() -> None:
    """Every item must contain the complete declared grid with alpha fixed at one."""
    observations = _items()
    del observations[0]["conditions"]["positions_2"]
    with pytest.raises(ValueError, match="lacks positions_2"):
        retention_payload(observations)

    observations = _items()
    observations[0]["conditions"]["positions_4"]["alpha"] = 0.5
    with pytest.raises(ValueError, match="alpha 1"):
        retention_payload(observations)


def test_payload_rejects_held_out_label_shortcuts() -> None:
    """Changing undeclared labels cannot change predictions or allocations."""
    observations = _items()
    expected = retention_payload(observations)
    for item in observations:
        item["memory_quality"] = f"changed-{item['item_id']}"

    observed = retention_payload(observations)

    assert observed == expected


def test_validate_result_accepts_recomputable_evidence() -> None:
    """Complete A100 evidence is accepted when its payload recomputes exactly."""
    report = validate_result(_result())

    assert report["verified"] is True
    assert report["fixed_total_acoustic_positions"] == 8
    assert report["adaptive_total_acoustic_positions"] == 8


def test_validate_result_rejects_tampered_evaluation() -> None:
    """Changing one retained summary metric invalidates the evidence."""
    result = copy.deepcopy(_result())
    result["retention"]["evaluation"]["adaptive_mean_teacher_kl"] += 0.1

    with pytest.raises(ValueError, match="recomputed retention payload differs"):
        validate_result(result)
