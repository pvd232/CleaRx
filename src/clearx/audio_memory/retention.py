"""Fit and verify the equal-budget acoustic-position retention pilot."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final

import numpy as np

from clearx.audio_memory.gate import FEATURE_NAMES

MODEL_REPOSITORY: Final = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_COMMIT: Final = "26291f793822fb6be9555850f06dfe95f2d7e695"
TRANSFORMERS_COMMIT: Final = "7d9754a05193eb79b1d86aa744b622b8068008cd"
RAVDESS_URL: Final = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)
RAVDESS_ARCHIVE_MD5: Final = "bc696df654c87fed845eb13823edef8a"
TRANSCRIPTS: Final = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}
INCOMPLETE_TRANSCRIPTS: Final = {
    "01": "Are talking by the door.",
    "02": "Are sitting by the door.",
}
MEMORY_QUALITIES: Final = ("exact", "incomplete", "conflicting")
PROMPT: Final = "Repeat the spoken sentence exactly. Answer only with the sentence."
NOISE_SNR_DB: Final = -10.0
SEED: Final = 7
TRAIN_ACTORS: Final = ("01", "02", "03", "04")
TEST_ACTORS: Final = ("05", "06")
POSITION_COUNTS: Final = (0, 2, 4)
FIXED_POSITION_COUNT: Final = 2
CONCENTRATED_POSITION_COUNT: Final = 4
ALPHA: Final = 1.0
RIDGE_VALUES: Final = (0.01, 0.1, 1.0, 10.0, 100.0)


def mean_pool_ordered(vectors: np.ndarray, output_positions: int) -> np.ndarray:
    """Average consecutive source bins into the requested ordered positions."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("pooling requires a nonempty [positions, hidden_size] array")
    if not np.isfinite(values).all():
        raise ValueError("pooling requires finite vectors")
    if output_positions < 0 or output_positions > values.shape[0]:
        raise ValueError("output positions must be between zero and source positions")
    if output_positions == 0:
        return np.empty((0, values.shape[1]), dtype=np.float32)
    boundaries = np.linspace(0, values.shape[0], output_positions + 1, dtype=int)
    pooled = np.stack(
        [values[start:stop].mean(axis=0) for start, stop in pairwise(boundaries)]
    )
    if pooled.shape != (output_positions, values.shape[1]):
        raise RuntimeError("ordered pooling emitted an unexpected shape")
    return pooled.astype(np.float32, copy=False)


def _condition(item: Mapping[str, Any], positions: int) -> Mapping[str, Any]:
    """Return one measured position condition after validating its identity."""
    conditions = item.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError("each item must contain condition measurements")
    key = f"positions_{positions}"
    condition = conditions.get(key)
    if condition is None:
        raise ValueError(f"item lacks {key}")
    if not isinstance(condition, Mapping):
        raise TypeError(f"{key} must be a mapping")
    if int(condition.get("acoustic_positions", -1)) != positions:
        raise ValueError(f"{key} records another acoustic-position count")
    alpha = float(condition.get("alpha", math.nan))
    divergence = float(condition.get("teacher_kl", math.nan))
    if alpha != ALPHA:
        raise ValueError("every retained acoustic vector must use alpha 1")
    if not math.isfinite(divergence) or divergence < 0.0:
        raise ValueError("teacher KL must be finite and nonnegative")
    return condition


def retention_benefit(item: Mapping[str, Any]) -> float:
    """Return measured KL improvement from four positions over transcript only."""
    transcript_only = float(_condition(item, 0)["teacher_kl"])
    concentrated = float(_condition(item, CONCENTRATED_POSITION_COUNT)["teacher_kl"])
    return transcript_only - concentrated


def _design(
    items: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build the declared feature matrix, benefit target, and speaker groups."""
    if not items:
        raise ValueError("retention fitting requires observations")
    rows: list[list[float]] = []
    targets: list[float] = []
    groups: list[str] = []
    for item in items:
        features = item.get("gate_features")
        if not isinstance(features, Mapping):
            raise TypeError("each item must contain gate_features")
        rows.append([float(features[name]) for name in FEATURE_NAMES])
        targets.append(retention_benefit(item))
        groups.append(str(item["actor_id"]))
    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("retention design must contain only finite values")
    return matrix, target, tuple(groups)


def _fit(matrix: np.ndarray, target: np.ndarray, ridge: float) -> dict[str, Any]:
    """Fit one standardized linear ridge regressor with an unpenalized intercept."""
    if ridge <= 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be positive and finite")
    means = matrix.mean(axis=0)
    scales = np.where(matrix.std(axis=0) < 1e-12, 1.0, matrix.std(axis=0))
    standardized = (matrix - means) / scales
    design = np.column_stack([np.ones(matrix.shape[0]), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ target,
    )
    return {
        "feature_names": list(FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge": ridge,
    }


def predict_benefit(model: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    """Predict four-position benefit from only the frozen agreement features."""
    names = tuple(str(name) for name in model["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError("retention model feature order differs from the contract")
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    values = np.asarray([float(features[name]) for name in names], dtype=np.float64)
    if means.shape != values.shape or scales.shape != values.shape:
        raise ValueError("retention normalization has the wrong shape")
    if coefficients.shape != (values.shape[0] + 1,):
        raise ValueError("retention coefficients have the wrong shape")
    prediction = coefficients[0] + np.dot(coefficients[1:], (values - means) / scales)
    if not math.isfinite(float(prediction)):
        raise ValueError("predicted benefit must be finite")
    return float(prediction)


def fit_retention_model(train_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select ridge strength by held-speaker-out error and refit all training data."""
    matrix, target, groups = _design(train_items)
    speakers = tuple(sorted(set(groups)))
    if speakers != TRAIN_ACTORS:
        raise ValueError("retention fitting requires exactly actors 01 through 04")
    validation: dict[str, float] = {}
    for ridge in RIDGE_VALUES:
        errors: list[float] = []
        for held_speaker in speakers:
            train_indices = [
                index for index, group in enumerate(groups) if group != held_speaker
            ]
            held_indices = [
                index for index, group in enumerate(groups) if group == held_speaker
            ]
            candidate = _fit(matrix[train_indices], target[train_indices], ridge)
            for index in held_indices:
                features = {
                    name: matrix[index, column]
                    for column, name in enumerate(FEATURE_NAMES)
                }
                errors.append(
                    (predict_benefit(candidate, features) - target[index]) ** 2
                )
        validation[f"{ridge:g}"] = statistics.fmean(errors)
    selected = min(RIDGE_VALUES, key=lambda value: (validation[f"{value:g}"], value))
    model = _fit(matrix, target, selected)
    model["speaker_validation_mse"] = validation
    model["training_item_count"] = len(train_items)
    model["training_speakers"] = list(speakers)
    return model


def _ranked_ids(
    items: Sequence[Mapping[str, Any]], scores: Mapping[str, float]
) -> list[str]:
    """Rank item IDs by descending score with an explicit stable tie break."""
    item_ids = [str(item["item_id"]) for item in items]
    if set(item_ids) != set(scores) or len(item_ids) != len(set(item_ids)):
        raise ValueError("allocation scores must name each item exactly once")
    return sorted(item_ids, key=lambda item_id: (-float(scores[item_id]), item_id))


def _validate_item_protocol(items: Sequence[Mapping[str, Any]]) -> None:
    """Require the exact speaker, statement, and transcript-condition product."""
    expected: dict[str, dict[str, str]] = {}
    for actor_id in (*TRAIN_ACTORS, *TEST_ACTORS):
        split = "train" if actor_id in TRAIN_ACTORS else "test"
        for statement_id, transcript in TRANSCRIPTS.items():
            other_statement = "02" if statement_id == "01" else "01"
            memories = {
                "exact": transcript,
                "incomplete": INCOMPLETE_TRANSCRIPTS[statement_id],
                "conflicting": TRANSCRIPTS[other_statement],
            }
            for quality in MEMORY_QUALITIES:
                item_id = f"actor-{actor_id}-statement-{statement_id}-{quality}"
                expected[item_id] = {
                    "actor_id": actor_id,
                    "statement_id": statement_id,
                    "expected_transcript": transcript,
                    "memory_quality": quality,
                    "memory_transcript": memories[quality],
                    "audio_quality": "degraded" if quality == "exact" else "clean",
                    "split": split,
                    "filename": f"03-01-01-01-{statement_id}-01-{actor_id}.wav",
                }
    observed = {str(item.get("item_id")): item for item in items}
    if len(observed) != len(items) or set(observed) != set(expected):
        raise ValueError("items must equal the frozen 36-item protocol product")
    audio_hashes: dict[str, str] = {}
    for item_id, frozen in expected.items():
        item = observed[item_id]
        for field, value in frozen.items():
            if item.get(field) != value:
                raise ValueError(f"{item_id} differs at {field}")
        audio_hash = str(item.get("audio_sha256", ""))
        if len(audio_hash) != 64 or any(
            character not in "0123456789abcdef" for character in audio_hash
        ):
            raise ValueError(f"{item_id} has an invalid audio SHA-256")
        filename = frozen["filename"]
        prior_hash = audio_hashes.setdefault(filename, audio_hash)
        if prior_hash != audio_hash:
            raise ValueError("transcript variants for one recording changed audio")
        audio_positions = int(item.get("audio_feature_positions", 0))
        transcript_positions = int(item.get("transcript_token_positions", 0))
        full_positions = int(item.get("full_input_positions", 0))
        if min(audio_positions, transcript_positions, full_positions) < 1:
            raise ValueError(f"{item_id} has invalid position counts")
        for positions in POSITION_COUNTS:
            condition = _condition(item, positions)
            if int(condition.get("transcript_positions", -1)) != transcript_positions:
                raise ValueError(f"{item_id} changed transcript positions")
            expected_input = (
                full_positions - audio_positions - 2 + transcript_positions + positions
            )
            if int(condition.get("input_positions", -1)) != expected_input:
                raise ValueError(f"{item_id} has inconsistent input positions")


def retention_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit on train actors and compare fixed, adaptive, and oracle test allocations."""
    _validate_item_protocol(items)
    train_items = [item for item in items if item.get("split") == "train"]
    test_items = [item for item in items if item.get("split") == "test"]
    if not train_items or not test_items or len(test_items) % 2:
        raise ValueError("retention payload requires train data and an even test count")
    if tuple(sorted({str(item["actor_id"]) for item in test_items})) != TEST_ACTORS:
        raise ValueError("retention evaluation requires exactly actors 05 and 06")
    for item in items:
        for positions in POSITION_COUNTS:
            _condition(item, positions)

    model = fit_retention_model(train_items)
    predicted = {
        str(item["item_id"]): predict_benefit(model, item["gate_features"])
        for item in test_items
    }
    measured = {str(item["item_id"]): retention_benefit(item) for item in test_items}
    selected_count = len(test_items) // 2
    adaptive_selected = _ranked_ids(test_items, predicted)[:selected_count]
    oracle_selected = _ranked_ids(test_items, measured)[:selected_count]
    by_id = {str(item["item_id"]): item for item in test_items}

    def allocated_mean(selected: set[str]) -> float:
        values = [
            float(
                _condition(
                    item,
                    CONCENTRATED_POSITION_COUNT
                    if str(item["item_id"]) in selected
                    else 0,
                )["teacher_kl"]
            )
            for item in test_items
        ]
        return statistics.fmean(values)

    fixed_values = [
        float(_condition(item, FIXED_POSITION_COUNT)["teacher_kl"])
        for item in test_items
    ]
    fixed_budget = len(test_items) * FIXED_POSITION_COUNT
    adaptive_budget = len(adaptive_selected) * CONCENTRATED_POSITION_COUNT
    if fixed_budget != adaptive_budget:
        raise ValueError("fixed and adaptive acoustic-position budgets differ")
    adaptive_mean = allocated_mean(set(adaptive_selected))
    fixed_mean = statistics.fmean(fixed_values)
    allocation = {
        item_id: CONCENTRATED_POSITION_COUNT if item_id in adaptive_selected else 0
        for item_id in sorted(by_id)
    }
    return {
        "model": model,
        "test_predicted_benefit": predicted,
        "test_measured_benefit": measured,
        "adaptive_allocation": allocation,
        "adaptive_selected_item_ids": adaptive_selected,
        "oracle_selected_item_ids": oracle_selected,
        "evaluation": {
            "test_item_count": len(test_items),
            "fixed_positions_per_item": FIXED_POSITION_COUNT,
            "adaptive_high_positions": CONCENTRATED_POSITION_COUNT,
            "adaptive_selected_count": len(adaptive_selected),
            "fixed_total_acoustic_positions": fixed_budget,
            "adaptive_total_acoustic_positions": adaptive_budget,
            "fixed_mean_teacher_kl": fixed_mean,
            "adaptive_mean_teacher_kl": adaptive_mean,
            "oracle_mean_teacher_kl": allocated_mean(set(oracle_selected)),
            "adaptive_minus_fixed_mean_teacher_kl": adaptive_mean - fixed_mean,
            "directional_success": adaptive_mean < fixed_mean,
        },
    }


def maximum_numeric_delta(expected: Any, observed: Any) -> float:
    """Return the greatest numeric difference across matching JSON structures."""
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        if set(expected) != set(observed):
            raise ValueError("result mappings contain different keys")
        return max(
            (maximum_numeric_delta(expected[key], observed[key]) for key in expected),
            default=0.0,
        )
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            raise ValueError("result sequences contain different lengths")
        return max(
            (
                maximum_numeric_delta(left, right)
                for left, right in zip(expected, observed, strict=True)
            ),
            default=0.0,
        )
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected != observed:
            raise ValueError("result values differ")
        return 0.0
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        delta = abs(float(expected) - float(observed))
        if not math.isfinite(delta):
            raise ValueError("result comparison produced a non-finite difference")
        return delta
    if expected != observed:
        raise ValueError("result values differ")
    return 0.0


def validate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate provenance and recompute the complete adaptive-retention payload."""
    if result.get("status") != "completed":
        raise ValueError("retention result must be completed")
    source = result.get("source")
    identity = result.get("identity")
    environment = result.get("environment")
    items = result.get("items")
    if not isinstance(source, Mapping) or not isinstance(identity, Mapping):
        raise TypeError("retention result must contain source and identity mappings")
    commit = str(source.get("experiment_commit", ""))
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("experiment commit must be a full lowercase Git hash")
    expected_identity = {
        "model_repository": MODEL_REPOSITORY,
        "model_commit": MODEL_COMMIT,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "ravdess_url": RAVDESS_URL,
        "ravdess_archive_md5": RAVDESS_ARCHIVE_MD5,
        "transcripts": TRANSCRIPTS,
        "incomplete_transcripts": INCOMPLETE_TRANSCRIPTS,
        "memory_qualities": list(MEMORY_QUALITIES),
        "noise_snr_db": NOISE_SNR_DB,
        "prompt": PROMPT,
        "seed": SEED,
        "alpha": ALPHA,
        "position_counts": list(POSITION_COUNTS),
        "fixed_position_count": FIXED_POSITION_COUNT,
        "concentrated_position_count": CONCENTRATED_POSITION_COUNT,
        "train_actors": list(TRAIN_ACTORS),
        "test_actors": list(TEST_ACTORS),
    }
    for key, value in expected_identity.items():
        if identity.get(key) != value:
            raise ValueError(f"retention identity differs at {key}")
    if not isinstance(environment, Mapping) or "A100" not in str(
        environment.get("device", "")
    ):
        raise ValueError("retention result must come from an A100")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError("retention result must contain item observations")
    expected = retention_payload(items)
    observed = result.get("retention")
    if not isinstance(observed, Mapping):
        raise TypeError("retention result must contain its fitted payload")
    delta = maximum_numeric_delta(expected, observed)
    if delta > 1e-8:
        raise ValueError(f"recomputed retention payload differs by {delta}")
    return {
        "verified": True,
        "experiment_commit": commit,
        "item_count": len(items),
        "maximum_numeric_delta": delta,
        **expected["evaluation"],
    }
