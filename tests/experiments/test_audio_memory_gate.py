"""Verify the segment features, speaker split, and deterministic gate fit."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tools.research.audio_memory_gate import (
    FEATURE_NAMES,
    fit_gate,
    gate_payload,
    load_json,
    oracle_alpha,
    predict_alpha,
    segment_features,
    select_fixed_alpha,
)


def test_load_json_requires_an_object(tmp_path: Path) -> None:
    """The isolated VIPER loader accepts objects and rejects other JSON values."""
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"verified": true}\n', encoding="utf-8")

    assert load_json(artifact) == {"verified": True}

    artifact.write_text("[]\n", encoding="utf-8")
    try:
        load_json(artifact)
    except TypeError as error:
        assert str(error) == "JSON artifact must contain an object"
    else:
        raise AssertionError("load_json accepted a non-object artifact")


def _item(actor: str, index: int, best_alpha: float, split: str) -> dict[str, object]:
    """Build one synthetic item whose features vary by speaker and observation."""
    features = {
        name: float(index + position) / (position + 2)
        for position, name in enumerate(FEATURE_NAMES)
    }
    conditions = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        conditions[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "teacher_kl": (alpha - best_alpha) ** 2 + 0.1,
        }
    return {
        "item_id": f"{actor}-{index}",
        "actor_id": actor,
        "split": split,
        "gate_features": features,
        "conditions": conditions,
    }


def test_segment_features_are_finite_and_complete() -> None:
    """Aligned finite vectors produce every frozen gate feature."""
    audio = np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    text = np.asarray([[2.0, 1.0], [4.0, 2.0]], dtype=np.float32)

    observed = segment_features(audio, text, raw_audio_positions=8)

    assert tuple(observed) == FEATURE_NAMES
    assert all(math.isfinite(value) for value in observed.values())
    assert observed["audio_positions_per_text_position"] == 4.0


def test_gate_fit_uses_training_speakers_and_predicts_held_out_items() -> None:
    """The gate fits four speakers and emits bounded predictions for test speakers."""
    items = []
    for actor_index, actor in enumerate(("01", "02", "03", "04")):
        items.extend(
            _item(actor, actor_index * 2 + offset, 0.25 * actor_index, "train")
            for offset in range(2)
        )
    items.extend([_item("05", 20, 0.5, "test"), _item("06", 21, 0.75, "test")])

    payload = gate_payload(items)

    assert payload["model"]["training_speakers"] == ["01", "02", "03", "04"]
    assert set(payload["test_predictions"]) == {"05-20", "06-21"}
    assert all(0.0 <= value <= 1.0 for value in payload["test_predictions"].values())
    assert payload == gate_payload(items)


def test_oracle_fixed_and_predicted_coefficients_reject_shortcuts() -> None:
    """Targets come from measured KL and prediction uses the declared feature order."""
    items = []
    for actor_index, actor in enumerate(("01", "02", "03")):
        items.extend(
            _item(actor, actor_index * 3 + offset, 0.5, "train") for offset in range(3)
        )
    model = fit_gate(items)

    assert oracle_alpha(items[0]) == 0.5
    assert select_fixed_alpha(items) == 0.5
    assert 0.0 <= predict_alpha(model, items[0]["gate_features"]) <= 1.0
