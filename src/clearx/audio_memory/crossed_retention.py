"""Fit and verify the crossed equal-budget acoustic-retention pilot."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np

from clearx.audio_memory.gate import FEATURE_NAMES

MODEL_REPOSITORY: Final = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_COMMIT: Final = "26291f793822fb6be9555850f06dfe95f2d7e695"
TRANSFORMERS_COMMIT: Final = "7d9754a05193eb79b1d86aa744b622b8068008cd"
VIPER_COMMIT: Final = "43a939a7abb2412cf5da7a0ddd5d01f67d1d84ab"
RAVDESS_URL: Final = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)
RAVDESS_ARCHIVE_MD5: Final = "bc696df654c87fed845eb13823edef8a"
RAVDESS_AUDIO_SHA256: Final = {
    "03-01-01-01-01-01-01.wav": "c467aad2ca80b089dbd4c2ba0dd16be508e4ce08230f46d36a208b859dc7fbac",
    "03-01-01-01-01-01-02.wav": "e343477557ff71c099c7a71ef70ff65158fcab232a56ee23bd231319a0f2c845",
    "03-01-01-01-01-01-03.wav": "d61bd344a100e9ba43b2e589912e0ce4b8c4f8be14e7e9a5b618a51728faf34f",
    "03-01-01-01-01-01-04.wav": "30ed5827386c30840f93ec49e4cf5689597a6d48412d79c94a171e71495ccee2",
    "03-01-01-01-01-01-05.wav": "d6ae1eaf5f0638d0fbbb6ea9ba040e089e741c13caca902d91059541c24c093a",
    "03-01-01-01-01-01-06.wav": "fec3fbd174eac5a623f30ac63b09bbb28b947754ab5f1ba57aa3758d10186ea1",
    "03-01-01-01-02-01-01.wav": "53970b4b7c4fd2a1efabeb74409dd06914e8c113d0e5cced4d9fceb036318739",
    "03-01-01-01-02-01-02.wav": "bf8422b8ff263607a9264f15f1db6e979aba70c7b0966c283fbf0e9ad92d424e",
    "03-01-01-01-02-01-03.wav": "d2e500826753fc20c4d095fcd4f470cd3f44d523313cc34fe568d284b2912d62",
    "03-01-01-01-02-01-04.wav": "7a49df186c48e1bf40bec92085831cb7dc7e3d0edfcf2d08b23a67a15a2d8ccf",
    "03-01-01-01-02-01-05.wav": "023c14008c4810c5114b8690c3d0adeb46d97cd1cca88420a56752a5ec28fb99",
    "03-01-01-01-02-01-06.wav": "90f9e5d015c898554a488ee2ad6179b43c63645993cde40fa6e468e5254922d3",
}
TRANSCRIPTS: Final = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}
INCOMPLETE_TRANSCRIPTS: Final = {
    "01": "Are talking by the door.",
    "02": "Are sitting by the door.",
}
MEMORY_QUALITIES: Final = ("exact", "incomplete", "conflicting")
AUDIO_QUALITIES: Final = ("clean", "degraded")
AUDIO_ONLY_FEATURE_NAMES: Final = (
    "mean_audio_log_norm",
    "std_audio_log_norm",
    "mean_temporal_delta_ratio",
)
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
EXPECTED_DEVICE_NAME: Final = "NVIDIA A100-SXM4-40GB"
EXPECTED_COMPUTE_CAPABILITY: Final = (8, 0)
EXPERIMENT_ID: Final = "audio-memory-crossed-retention-v2"
SCHEMA_VERSION: Final = "exploratory-2.0.0"


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
        [
            values[start:stop].mean(axis=0)
            for start, stop in itertools.pairwise(boundaries)
        ]
    )
    if pooled.shape != (output_positions, values.shape[1]):
        raise RuntimeError("ordered pooling emitted an unexpected shape")
    return pooled.astype(np.float32, copy=False)


def audio_only_features(audio_vectors: np.ndarray) -> dict[str, float]:
    """Summarize Audio Tower vectors without transcript-dependent alignment."""
    audio = np.asarray(audio_vectors, dtype=np.float64)
    if audio.ndim != 2 or min(audio.shape) < 1 or not np.isfinite(audio).all():
        raise ValueError("audio-only features require a finite nonempty matrix")
    epsilon = np.finfo(np.float64).eps
    norms = np.linalg.norm(audio, axis=1)
    log_norms = np.log(np.maximum(norms, epsilon))
    if audio.shape[0] == 1:
        temporal_ratio = np.asarray([0.0], dtype=np.float64)
    else:
        temporal_ratio = np.linalg.norm(np.diff(audio, axis=0), axis=1) / max(
            float(np.mean(norms)), epsilon
        )
    values = {
        "mean_audio_log_norm": float(np.mean(log_norms)),
        "std_audio_log_norm": float(np.std(log_norms)),
        "mean_temporal_delta_ratio": float(np.mean(temporal_ratio)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("audio-only features must be finite")
    return values


def assemble_candidate_embeddings(
    prefix: Any,
    transcript: Any,
    acoustic: Any,
    suffix: Any,
    *,
    join: Callable[[tuple[Any, Any, Any, Any]], Any],
) -> Any:
    """Concatenate four embedding blocks while retaining transcript bytes."""
    shapes = [tuple(part.shape) for part in (prefix, transcript, acoustic, suffix)]
    if any(len(shape) != 3 for shape in shapes):
        raise ValueError(
            "candidate embedding parts must have [batch, positions, hidden] shape"
        )
    if any(shape[0] != shapes[0][0] or shape[2] != shapes[0][2] for shape in shapes):
        raise ValueError(
            "candidate embedding parts must share batch and hidden dimensions"
        )
    return join((prefix, transcript, acoustic, suffix))


def frozen_item_specs() -> tuple[dict[str, str], ...]:
    """Return the complete speaker, recording, transcript, and audio product."""
    selected: list[dict[str, str]] = []
    for actor_id in (*TRAIN_ACTORS, *TEST_ACTORS):
        split = "train" if actor_id in TRAIN_ACTORS else "test"
        for statement_id, transcript in TRANSCRIPTS.items():
            filename = f"03-01-01-01-{statement_id}-01-{actor_id}.wav"
            other_statement = "02" if statement_id == "01" else "01"
            memories = {
                "exact": transcript,
                "incomplete": INCOMPLETE_TRANSCRIPTS[statement_id],
                "conflicting": TRANSCRIPTS[other_statement],
            }
            for memory_quality in MEMORY_QUALITIES:
                for audio_quality in AUDIO_QUALITIES:
                    selected.append(
                        {
                            "item_id": (
                                f"actor-{actor_id}-statement-{statement_id}-"
                                f"{memory_quality}-{audio_quality}"
                            ),
                            "actor_id": actor_id,
                            "statement_id": statement_id,
                            "expected_transcript": transcript,
                            "memory_quality": memory_quality,
                            "memory_transcript": memories[memory_quality],
                            "audio_quality": audio_quality,
                            "split": split,
                            "filename": filename,
                        }
                    )
    return tuple(selected)


def _condition(item: Mapping[str, Any], positions: int) -> Mapping[str, Any]:
    """Return one measured position condition after validating its identity."""
    conditions = item.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError("each item must contain condition measurements")
    key = f"positions_{positions}"
    condition = conditions.get(key)
    if not isinstance(condition, Mapping):
        raise TypeError(f"item lacks valid {key}")
    if (
        type(condition.get("acoustic_positions")) is not int
        or int(condition["acoustic_positions"]) != positions
    ):
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
    return float(_condition(item, 0)["teacher_kl"]) - float(
        _condition(item, CONCENTRATED_POSITION_COUNT)["teacher_kl"]
    )


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
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge": ridge,
    }


def predict_benefit(model: Mapping[str, Any], features: Mapping[str, Any]) -> float:
    """Predict four-position benefit using the model's frozen feature order."""
    names = tuple(str(name) for name in model["feature_names"])
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


def _fit_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: tuple[str, ...],
    training_unit: str,
) -> dict[str, Any]:
    """Select ridge strength by held-speaker-out error and refit all rows."""
    if not rows:
        raise ValueError("retention fitting requires observations")
    matrix = np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    target = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    groups = tuple(str(row["actor_id"]) for row in rows)
    speakers = tuple(sorted(set(groups)))
    if speakers != TRAIN_ACTORS:
        raise ValueError("retention fitting requires exactly actors 01 through 04")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("retention design must contain only finite values")
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
            candidate["feature_names"] = list(feature_names)
            for index in held_indices:
                values = {
                    name: matrix[index, column]
                    for column, name in enumerate(feature_names)
                }
                errors.append((predict_benefit(candidate, values) - target[index]) ** 2)
        validation[f"{ridge:g}"] = statistics.fmean(errors)
    selected = min(RIDGE_VALUES, key=lambda value: (validation[f"{value:g}"], value))
    model = _fit(matrix, target, selected)
    model.update(
        {
            "feature_names": list(feature_names),
            "speaker_validation_mse": validation,
            "training_row_count": len(rows),
            "training_speakers": list(speakers),
            "training_unit": training_unit,
        }
    )
    return model


def _validate_features(item: Mapping[str, Any]) -> None:
    """Require complete finite agreement and audio-only feature mappings."""
    for field, names in (
        ("gate_features", FEATURE_NAMES),
        ("audio_only_features", AUDIO_ONLY_FEATURE_NAMES),
    ):
        features = item.get(field)
        if not isinstance(features, Mapping) or set(features) != set(names):
            raise ValueError(f"{field} differs from its frozen feature names")
        if not all(math.isfinite(float(features[name])) for name in names):
            raise ValueError(f"{field} must contain finite values")


def _validate_item_protocol(items: Sequence[Mapping[str, Any]]) -> None:
    """Require the exact crossed product and its frozen input identities."""
    expected = {item["item_id"]: item for item in frozen_item_specs()}
    observed = {str(item.get("item_id")): item for item in items}
    if len(observed) != len(items) or set(observed) != set(expected):
        raise ValueError("items must equal the frozen 72-item crossed product")
    group_features: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item_id, frozen in expected.items():
        item = observed[item_id]
        for field, value in frozen.items():
            if item.get(field) != value:
                raise ValueError(f"{item_id} differs at {field}")
        filename = frozen["filename"]
        if item.get("audio_sha256") != RAVDESS_AUDIO_SHA256[filename]:
            raise ValueError(f"{item_id} differs at audio_sha256")
        _validate_features(item)
        group_key = (filename, frozen["audio_quality"])
        prior = group_features.setdefault(group_key, item["audio_only_features"])
        if item["audio_only_features"] != prior:
            raise ValueError("audio-only features changed across transcript conditions")
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


def _item_rows(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build item-level agreement-model training rows."""
    return [
        {
            "actor_id": item["actor_id"],
            "features": item["gate_features"],
            "target": retention_benefit(item),
        }
        for item in items
    ]


def _audio_group_rows(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average targets within recording and audio-quality training groups."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in items:
        key = (str(item["filename"]), str(item["audio_quality"]))
        groups.setdefault(key, []).append(item)
    rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) != len(MEMORY_QUALITIES):
            raise ValueError("each audio-only group must contain three transcripts")
        rows.append(
            {
                "group_id": f"{key[0]}::{key[1]}",
                "actor_id": members[0]["actor_id"],
                "features": members[0]["audio_only_features"],
                "target": statistics.fmean(retention_benefit(item) for item in members),
            }
        )
    return rows


def _ranked_ids(scores: Mapping[str, float]) -> list[str]:
    """Rank identifiers by descending score with a stable tie break."""
    return sorted(scores, key=lambda identity: (-float(scores[identity]), identity))


def _allocation(item_ids: Sequence[str], selected: set[str]) -> dict[str, int]:
    """Assign four positions to selected items and zero to every other item."""
    return {
        item_id: CONCENTRATED_POSITION_COUNT if item_id in selected else 0
        for item_id in sorted(item_ids)
    }


def _policy_losses(
    items: Sequence[Mapping[str, Any]], allocation: Mapping[str, int]
) -> dict[str, float]:
    """Return the measured held-out loss selected by one allocation."""
    return {
        str(item["item_id"]): float(
            _condition(item, int(allocation[str(item["item_id"])]))["teacher_kl"]
        )
        for item in items
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return one linearly interpolated percentile of sorted finite values."""
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile requires values and a probability within [0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_contrast(
    items: Sequence[Mapping[str, Any]],
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, Any]:
    """Compute recording and actor contrasts plus an exact cluster bootstrap."""
    recordings = sorted({str(item["filename"]) for item in items})
    by_recording: dict[str, float] = {}
    for filename in recordings:
        members = [item for item in items if item["filename"] == filename]
        by_recording[filename] = statistics.fmean(
            left[str(item["item_id"])] - right[str(item["item_id"])] for item in members
        )
    by_actor: dict[str, float] = {}
    for actor in TEST_ACTORS:
        members = [item for item in items if item["actor_id"] == actor]
        by_actor[actor] = statistics.fmean(
            left[str(item["item_id"])] - right[str(item["item_id"])] for item in members
        )
    resampled_means = [
        statistics.fmean(by_recording[recordings[index]] for index in sample)
        for sample in itertools.product(range(len(recordings)), repeat=len(recordings))
    ]
    return {
        "recording_effects": by_recording,
        "actor_effects": by_actor,
        "cluster_count": len(recordings),
        "resample_count": len(resampled_means),
        "resampled_means": resampled_means,
        "percentile_95": [
            _percentile(resampled_means, 0.025),
            _percentile(resampled_means, 0.975),
        ],
    }


def crossed_retention_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit both policies and recompute every held-out allocation and summary."""
    _validate_item_protocol(items)
    train_items = [item for item in items if item["split"] == "train"]
    test_items = [item for item in items if item["split"] == "test"]
    agreement_model = _fit_model(
        _item_rows(train_items), feature_names=FEATURE_NAMES, training_unit="item"
    )
    audio_model = _fit_model(
        _audio_group_rows(train_items),
        feature_names=AUDIO_ONLY_FEATURE_NAMES,
        training_unit="recording_audio_group",
    )
    agreement_predictions = {
        str(item["item_id"]): predict_benefit(agreement_model, item["gate_features"])
        for item in test_items
    }
    test_audio_rows = _audio_group_rows(test_items)
    audio_group_predictions = {
        str(row["group_id"]): predict_benefit(audio_model, row["features"])
        for row in test_audio_rows
    }
    audio_predictions = {
        str(item["item_id"]): audio_group_predictions[
            f"{item['filename']}::{item['audio_quality']}"
        ]
        for item in test_items
    }
    measured = {str(item["item_id"]): retention_benefit(item) for item in test_items}
    item_ids = [str(item["item_id"]) for item in test_items]
    selected_count = len(test_items) // 2
    agreement_selected = set(_ranked_ids(agreement_predictions)[:selected_count])
    selected_group_count = len(test_audio_rows) // 2
    audio_selected_groups = set(
        _ranked_ids(audio_group_predictions)[:selected_group_count]
    )
    audio_selected = {
        str(item["item_id"])
        for item in test_items
        if f"{item['filename']}::{item['audio_quality']}" in audio_selected_groups
    }
    oracle_selected = set(_ranked_ids(measured)[:selected_count])
    allocations = {
        "fixed": {item_id: FIXED_POSITION_COUNT for item_id in sorted(item_ids)},
        "adaptive_agreement": _allocation(item_ids, agreement_selected),
        "adaptive_audio_only": _allocation(item_ids, audio_selected),
        "oracle": _allocation(item_ids, oracle_selected),
    }
    losses = {
        policy: _policy_losses(test_items, allocation)
        for policy, allocation in allocations.items()
    }
    budgets: dict[str, dict[str, int]] = {}
    for policy, allocation in allocations.items():
        acoustic = sum(allocation.values())
        aggregate_input = sum(
            int(_condition(item, allocation[str(item["item_id"])])["input_positions"])
            for item in test_items
        )
        budgets[policy] = {
            "acoustic_positions": acoustic,
            "aggregate_input_positions": aggregate_input,
        }
    if {value["acoustic_positions"] for value in budgets.values()} != {48}:
        raise ValueError("policy acoustic-position budgets differ")
    if len({value["aggregate_input_positions"] for value in budgets.values()}) != 1:
        raise ValueError("policy aggregate input-position budgets differ")
    means = {
        policy: statistics.fmean(policy_losses.values())
        for policy, policy_losses in losses.items()
    }
    agreement_vs_fixed = _cluster_contrast(
        test_items, losses["adaptive_agreement"], losses["fixed"]
    )
    agreement_vs_audio = _cluster_contrast(
        test_items, losses["adaptive_agreement"], losses["adaptive_audio_only"]
    )
    cell_summary: dict[str, dict[str, Any]] = {}
    for memory_quality in MEMORY_QUALITIES:
        cell_summary[memory_quality] = {}
        for audio_quality in AUDIO_QUALITIES:
            selected = [
                item
                for item in test_items
                if item["memory_quality"] == memory_quality
                and item["audio_quality"] == audio_quality
            ]
            cell_summary[memory_quality][audio_quality] = {
                "item_count": len(selected),
                "mean_measured_benefit": statistics.fmean(
                    measured[str(item["item_id"])] for item in selected
                ),
                "mean_agreement_prediction": statistics.fmean(
                    agreement_predictions[str(item["item_id"])] for item in selected
                ),
                "mean_audio_only_prediction": statistics.fmean(
                    audio_predictions[str(item["item_id"])] for item in selected
                ),
            }
    positive = (
        means["adaptive_agreement"] < means["fixed"]
        and means["adaptive_agreement"] < means["adaptive_audio_only"]
        and all(value < 0.0 for value in agreement_vs_fixed["actor_effects"].values())
        and all(value < 0.0 for value in agreement_vs_audio["actor_effects"].values())
        and agreement_vs_fixed["percentile_95"][1] < 0.0
        and agreement_vs_audio["percentile_95"][1] < 0.0
    )
    negative = (
        means["adaptive_agreement"] >= means["fixed"]
        and means["adaptive_agreement"] >= means["adaptive_audio_only"]
    )
    classification = (
        "positive_exploratory"
        if positive
        else ("negative" if negative else "inconclusive")
    )
    return {
        "models": {"agreement": agreement_model, "audio_only": audio_model},
        "predictions": {
            "agreement": agreement_predictions,
            "audio_only": audio_predictions,
            "audio_only_groups": audio_group_predictions,
            "measured_benefit": measured,
        },
        "allocations": allocations,
        "selected": {
            "adaptive_agreement": sorted(agreement_selected),
            "adaptive_audio_only": sorted(audio_selected),
            "adaptive_audio_only_groups": sorted(audio_selected_groups),
            "oracle": sorted(oracle_selected),
        },
        "evaluation": {
            "test_item_count": len(test_items),
            "test_recording_count": len({item["filename"] for item in test_items}),
            "budgets": budgets,
            "mean_teacher_kl": means,
            "agreement_minus_fixed": means["adaptive_agreement"] - means["fixed"],
            "agreement_minus_audio_only": (
                means["adaptive_agreement"] - means["adaptive_audio_only"]
            ),
            "relative_reduction_vs_fixed": (
                (means["fixed"] - means["adaptive_agreement"]) / means["fixed"]
                if means["fixed"] > 0.0
                else 0.0
            ),
            "clustered_contrasts": {
                "agreement_vs_fixed": agreement_vs_fixed,
                "agreement_vs_audio_only": agreement_vs_audio,
            },
            "cell_summary": cell_summary,
            "classification": classification,
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
        if type(expected) is not type(observed) or expected != observed:
            raise ValueError("result Boolean values differ")
        return 0.0
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        delta = abs(float(expected) - float(observed))
        if not math.isfinite(delta):
            raise ValueError("result comparison produced a non-finite difference")
        return delta
    if expected != observed:
        raise ValueError("result values differ")
    return 0.0


def result_identity() -> dict[str, Any]:
    """Return the frozen identity written by the measurement stage."""
    return {
        "model_repository": MODEL_REPOSITORY,
        "model_commit": MODEL_COMMIT,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "viper_commit": VIPER_COMMIT,
        "ravdess_url": RAVDESS_URL,
        "ravdess_archive_md5": RAVDESS_ARCHIVE_MD5,
        "ravdess_audio_sha256": RAVDESS_AUDIO_SHA256,
        "transcripts": TRANSCRIPTS,
        "incomplete_transcripts": INCOMPLETE_TRANSCRIPTS,
        "memory_qualities": list(MEMORY_QUALITIES),
        "audio_qualities": list(AUDIO_QUALITIES),
        "cluster_key": "filename",
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


def validate_result(
    result: Mapping[str, Any], *, expected_experiment_commit: str
) -> dict[str, Any]:
    """Bind provenance and recompute the crossed-retention payload."""
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("retention result has another schema version")
    if (
        result.get("experiment_id") != EXPERIMENT_ID
        or result.get("status") != "completed"
    ):
        raise ValueError("retention result has another identity or status")
    source = result.get("source")
    identity = result.get("identity")
    environment = result.get("environment")
    items = result.get("items")
    if not isinstance(source, Mapping) or not isinstance(identity, Mapping):
        raise TypeError("retention result must contain source and identity mappings")
    if source.get("experiment_commit") != expected_experiment_commit:
        raise ValueError("experiment commit differs from the independent expectation")
    if identity != result_identity():
        raise ValueError("retention identity differs from the frozen protocol")
    if not isinstance(environment, Mapping):
        raise TypeError("retention result must contain environment observations")
    if environment.get("device") != EXPECTED_DEVICE_NAME:
        raise ValueError("retention result reports another CUDA device")
    if environment.get("compute_capability") != list(EXPECTED_COMPUTE_CAPABILITY):
        raise ValueError("retention result reports another compute capability")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError("retention result must contain item observations")
    expected = crossed_retention_payload(items)
    observed = result.get("retention")
    if not isinstance(observed, Mapping):
        raise TypeError("retention result must contain its fitted payload")
    delta = maximum_numeric_delta(expected, observed)
    if delta > 1e-8:
        raise ValueError(f"recomputed retention payload differs by {delta}")
    return {
        "verified": True,
        "experiment_commit": expected_experiment_commit,
        "item_count": len(items),
        "maximum_numeric_delta": delta,
        **expected["evaluation"],
    }
