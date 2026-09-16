"""Fit and reproduce the segment-level audio-memory gate used by the pilot."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

FEATURE_NAMES: Final = (
    "mean_cosine",
    "std_cosine",
    "mean_log_norm_ratio",
    "std_log_norm_ratio",
    "mean_audio_log_norm",
    "std_audio_log_norm",
    "mean_temporal_delta_ratio",
    "audio_positions_per_text_position",
)
RIDGE_VALUES: Final = (0.01, 0.1, 1.0, 10.0, 100.0)


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object for the gate experiment or its VIPER verifier."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON artifact must contain an object")
    return value


def degrade_audio(
    audio: np.ndarray,
    identity: bytes,
    *,
    snr_db: float,
) -> np.ndarray:
    """Add deterministic white noise at a declared signal-to-noise ratio."""
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("audio degradation requires one finite nonempty waveform")
    if not math.isfinite(snr_db):
        raise ValueError("audio degradation requires a finite SNR")
    seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(signal.shape).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    if signal_rms == 0.0 or noise_rms == 0.0:
        raise ValueError("audio degradation requires non-silent audio")
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    mixture = signal + noise * np.float32(target_noise_rms / noise_rms)
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.99:
        mixture *= np.float32(0.99 / peak)
    if not np.isfinite(mixture).all():
        raise ValueError("audio degradation produced a non-finite sample")
    return mixture


def segment_features(
    audio_vectors: np.ndarray,
    text_vectors: np.ndarray,
    raw_audio_positions: int,
) -> dict[str, float]:
    """Summarize one aligned audio-text segment without using its target label."""
    audio = np.asarray(audio_vectors, dtype=np.float64)
    text = np.asarray(text_vectors, dtype=np.float64)
    if audio.ndim != 2 or text.ndim != 2 or audio.shape != text.shape:
        raise ValueError("audio and text vectors must share [positions, hidden_size]")
    if audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError("aligned vectors must contain at least one value")
    if raw_audio_positions < audio.shape[0]:
        raise ValueError("raw audio positions cannot be fewer than pooled positions")
    if not np.isfinite(audio).all() or not np.isfinite(text).all():
        raise ValueError("aligned vectors must contain only finite values")

    epsilon = np.finfo(np.float64).eps
    audio_norm = np.linalg.norm(audio, axis=1)
    text_norm = np.linalg.norm(text, axis=1)
    cosine = np.sum(audio * text, axis=1) / np.maximum(
        audio_norm * text_norm,
        epsilon,
    )
    log_norm_ratio = np.log(np.maximum(audio_norm, epsilon)) - np.log(
        np.maximum(text_norm, epsilon)
    )
    if audio.shape[0] == 1:
        temporal_delta_ratio = np.array([0.0], dtype=np.float64)
    else:
        deltas = np.linalg.norm(np.diff(audio, axis=0), axis=1)
        denominator = max(float(np.mean(audio_norm)), epsilon)
        temporal_delta_ratio = deltas / denominator

    values = {
        "mean_cosine": float(np.mean(cosine)),
        "std_cosine": float(np.std(cosine)),
        "mean_log_norm_ratio": float(np.mean(log_norm_ratio)),
        "std_log_norm_ratio": float(np.std(log_norm_ratio)),
        "mean_audio_log_norm": float(np.mean(np.log(np.maximum(audio_norm, epsilon)))),
        "std_audio_log_norm": float(np.std(np.log(np.maximum(audio_norm, epsilon)))),
        "mean_temporal_delta_ratio": float(np.mean(temporal_delta_ratio)),
        "audio_positions_per_text_position": raw_audio_positions / audio.shape[0],
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("segment features must be finite")
    return values


def oracle_alpha(item: Mapping[str, Any]) -> float:
    """Return the measured coefficient with the lowest teacher KL for one item."""
    conditions = item.get("conditions")
    if not isinstance(conditions, Mapping) or not conditions:
        raise ValueError("item must contain measured coefficient conditions")
    candidates: list[tuple[float, float]] = []
    for condition in conditions.values():
        if not isinstance(condition, Mapping):
            raise TypeError("each condition must be a mapping")
        alpha = float(condition["alpha"])
        divergence = float(condition["teacher_kl"])
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("condition alpha must be finite and within [0, 1]")
        if not math.isfinite(divergence) or divergence < 0.0:
            raise ValueError("condition teacher_kl must be finite and nonnegative")
        candidates.append((divergence, alpha))
    return min(candidates)[1]


def _design(
    items: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build the feature matrix, oracle targets, and speaker groups."""
    if not items:
        raise ValueError("gate fitting requires at least one item")
    rows: list[list[float]] = []
    targets: list[float] = []
    groups: list[str] = []
    for item in items:
        features = item.get("gate_features")
        if not isinstance(features, Mapping):
            raise TypeError("each item must contain gate_features")
        rows.append([float(features[name]) for name in FEATURE_NAMES])
        targets.append(oracle_alpha(item))
        groups.append(str(item["actor_id"]))
    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("gate design must contain only finite values")
    return matrix, target, tuple(groups)


def _fit(matrix: np.ndarray, target: np.ndarray, ridge: float) -> dict[str, Any]:
    """Fit one standardized logistic-link ridge regressor."""
    if ridge <= 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be positive and finite")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    standardized = (matrix - means) / scales
    design = np.column_stack([np.ones(matrix.shape[0]), standardized])
    clipped = np.clip(target, 0.025, 0.975)
    linked_target = np.log(clipped / (1.0 - clipped))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ linked_target,
    )
    return {
        "feature_names": list(FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge": ridge,
        "link": "logistic",
    }


def predict_alpha(model: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    """Predict one bounded fusion coefficient from segment summary features."""
    names = tuple(str(name) for name in model["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError("gate model feature order differs from the frozen contract")
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    values = np.asarray([float(features[name]) for name in names], dtype=np.float64)
    if means.shape != values.shape or scales.shape != values.shape:
        raise ValueError("gate model normalization has the wrong shape")
    if coefficients.shape != (values.shape[0] + 1,):
        raise ValueError("gate model coefficients have the wrong shape")
    linked = coefficients[0] + np.dot(coefficients[1:], (values - means) / scales)
    if linked >= 0:
        prediction = 1.0 / (1.0 + math.exp(-float(linked)))
    else:
        exponential = math.exp(float(linked))
        prediction = exponential / (1.0 + exponential)
    return min(1.0, max(0.0, prediction))


def fit_gate(train_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select ridge strength by held-speaker-out validation and fit the gate."""
    matrix, target, groups = _design(train_items)
    unique_groups = tuple(sorted(set(groups)))
    if len(unique_groups) < 3:
        raise ValueError("gate fitting requires at least three training speakers")

    validation: dict[str, float] = {}
    for ridge in RIDGE_VALUES:
        squared_errors: list[float] = []
        for held_group in unique_groups:
            train_indices = [
                index for index, group in enumerate(groups) if group != held_group
            ]
            held_indices = [
                index for index, group in enumerate(groups) if group == held_group
            ]
            candidate = _fit(matrix[train_indices], target[train_indices], ridge)
            for index in held_indices:
                feature_values = {
                    name: matrix[index, position]
                    for position, name in enumerate(FEATURE_NAMES)
                }
                prediction = predict_alpha(candidate, feature_values)
                squared_errors.append((prediction - target[index]) ** 2)
        validation[f"{ridge:g}"] = statistics.fmean(squared_errors)

    selected = min(RIDGE_VALUES, key=lambda value: (validation[f"{value:g}"], value))
    model = _fit(matrix, target, selected)
    model["speaker_validation_mse"] = validation
    model["training_item_count"] = len(train_items)
    model["training_speakers"] = list(unique_groups)
    return model


def select_fixed_alpha(train_items: Sequence[Mapping[str, Any]]) -> float:
    """Choose the fixed coefficient with the lowest training-set mean KL."""
    if not train_items:
        raise ValueError("fixed-alpha selection requires training items")
    condition_names = tuple(train_items[0]["conditions"])
    means: list[tuple[float, float]] = []
    for name in condition_names:
        divergences = [
            float(item["conditions"][name]["teacher_kl"]) for item in train_items
        ]
        alpha = float(train_items[0]["conditions"][name]["alpha"])
        means.append((statistics.fmean(divergences), alpha))
    return min(means)[1]


def gate_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit the frozen training split and predict coefficients for held-out items."""
    train_items = [item for item in items if item.get("split") == "train"]
    test_items = [item for item in items if item.get("split") == "test"]
    if not train_items or not test_items:
        raise ValueError("gate payload requires train and test items")
    model = fit_gate(train_items)
    predictions = {
        str(item["item_id"]): predict_alpha(model, item["gate_features"])
        for item in test_items
    }
    return {
        "model": model,
        "fixed_alpha_selected_on_train": select_fixed_alpha(train_items),
        "test_predictions": predictions,
    }
