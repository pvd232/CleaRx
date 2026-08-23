#!/usr/bin/env python3
"""Define and validate CleaRx reference-fixture boundaries and comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from fixture_contract import PROFILE_DEFINITIONS  # noqa: E402


FIXTURE_SEED: Final = 20_260_823
GENERATOR_SAMPLING_PROFILE: Final = "greedy-correctness"
REQUIRED_PROVENANCE_FIELDS: Final = frozenset(
    {
        "upstream_lock_sha256",
        "tensor_manifest_sha256",
        "evaluation_contract_sha256",
        "fixture_schema_sha256",
        "generator_commit",
        "generator_path",
        "generator_sha256",
        "source_model_commit",
        "source_model_config_sha256",
    }
)
ADVERSARIAL_TAGS: Final = frozenset(
    {
        "empty_background",
        "background_noise",
        "variable_chunk",
        "final_partial_chunk",
        "turn_end",
        "long_turn",
        "interruption",
        "cancellation",
        "repeated_phase_transition",
        "context_reconstruction",
    }
)
COORDINATE_DOMAINS: Final = frozenset(
    {
        "metadata",
        "pcm_input_sample",
        "aut_frame",
        "thinker_position",
        "talker_position",
        "codec_frame",
        "codec_codebook",
        "pcm_output_sample",
        "scheduler_event",
    }
)
SCHEDULER_EVENT_CODES: Final = {
    "input_chunk": 1,
    "background_interval": 2,
    "turn_end": 3,
    "thinker_ready": 4,
    "talker_start": 5,
    "codec_frame": 6,
    "waveform_commit": 7,
    "interrupt": 8,
    "cancel": 9,
    "context_rebuild": 10,
    "listen_resume": 11,
}
SCHEDULER_STATE_CODES: Final = {
    "idle": 0,
    "listening": 1,
    "thinking": 2,
    "speaking": 3,
    "cancelling": 4,
    "rebuilding": 5,
}
SYNTHETIC_NATIVE_PAIRS: Final = frozenset(
    {
        ("thinker.accepted_hidden", "context_reconstruction"),
        ("handoff.thinker_to_talker", "context_reconstruction"),
        ("scheduler.turn_trace", "turn_end"),
        ("scheduler.turn_trace", "long_turn"),
        ("scheduler.turn_trace", "interruption"),
        ("scheduler.turn_trace", "cancellation"),
        ("scheduler.turn_trace", "repeated_phase_transition"),
        ("scheduler.turn_trace", "context_reconstruction"),
    }
)


@dataclass(frozen=True)
class CaseSpec:
    """Name one deterministic nominal or adversarial generator input."""

    case_id: str
    case_class: str
    description: str
    adversarial_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundarySpec:
    """Name one observed value and the rule used to compare it."""

    boundary_id: str
    module: str
    operation: str
    input_coordinate_domain: str
    output_coordinate_domain: str
    output_dtype: str
    output_shape: tuple[int | str, ...]
    profile_id: str
    case_ids: tuple[str, ...] = ("nominal_short",)


CASE_MATRIX: Final = (
    CaseSpec("nominal_short", "nominal", "One short deterministic speech turn."),
    CaseSpec("empty_background", "adversarial", "Zero-length background interval.", ("empty_background",)),
    CaseSpec("background_noise", "adversarial", "Seeded low-amplitude noise before speech.", ("background_noise",)),
    CaseSpec("variable_chunk", "adversarial", "PCM arrives in deterministic unequal chunk lengths.", ("variable_chunk",)),
    CaseSpec("final_partial_chunk", "adversarial", "The last codec batch is shorter than the steady chunk.", ("final_partial_chunk",)),
    CaseSpec("turn_end", "adversarial", "End-of-turn arrives immediately after a committed codec frame.", ("turn_end",)),
    CaseSpec("long_turn", "adversarial", "A turn crosses the bounded live-context rebuild threshold.", ("long_turn",)),
    CaseSpec("interruption", "adversarial", "A new input turn interrupts active audio output.", ("interruption",)),
    CaseSpec("cancellation", "adversarial", "Cancellation occurs with queued codec work.", ("cancellation",)),
    CaseSpec("repeated_phase_transition", "adversarial", "Listening and speaking transitions repeat without a process restart.", ("repeated_phase_transition",)),
    CaseSpec("context_reconstruction", "adversarial", "Bounded context is rebuilt from the retained representation.", ("context_reconstruction",)),
)


def _boundary(
    boundary_id: str,
    module: str,
    operation: str,
    input_domain: str,
    output_domain: str,
    dtype: str,
    shape: tuple[int | str, ...],
    profile: str,
    cases: tuple[str, ...] = ("nominal_short",),
) -> BoundarySpec:
    """Construct one compact immutable boundary specification."""
    return BoundarySpec(boundary_id, module, operation, input_domain, output_domain, dtype, shape, profile, cases)


def build_boundary_matrix() -> tuple[BoundarySpec, ...]:
    """Return every required observable boundary in deterministic order."""
    boundaries = [
        _boundary("input.pcm_preprocessor", "input", "pcm_to_log_mel", "pcm_input_sample", "aut_frame", "float32", (1, 128, "aut_frames"), "fp32_strict", ("nominal_short", "empty_background", "background_noise", "variable_chunk")),
        _boundary("aut.encoder_output", "aut", "encoder_final_hidden", "aut_frame", "aut_frame", "float32", ("aut_frames", 1280), "bf16_accumulation"),
        _boundary("aut.output_projection", "aut", "proj2_output", "aut_frame", "thinker_position", "float32", ("aut_frames", 2048), "bf16_accumulation"),
        _boundary("thinker.accepted_hidden", "thinker", "accepted_text_hidden", "thinker_position", "thinker_position", "float32", (1, "thinker_positions", 2048), "bf16_accumulation", ("nominal_short", "long_turn", "context_reconstruction")),
        _boundary("thinker.talker_conditioning", "thinker", "text_projection", "thinker_position", "talker_position", "float32", (1, "talker_positions", 1024), "bf16_accumulation"),
    ]
    for layer in range(20):
        prefix = f"talker.layer.{layer:02d}"
        boundaries.extend(
            [
                _boundary(f"{prefix}.router_logits", "talker", "router_softmax", "talker_position", "talker_position", "float32", ("talker_positions", 128), "fp32_strict"),
                _boundary(f"{prefix}.selected_experts", "talker", "router_topk_indices", "talker_position", "talker_position", "int64", ("talker_positions", 6), "exact_discrete"),
                _boundary(f"{prefix}.routing_weights", "talker", "normalized_topk_weights", "talker_position", "talker_position", "float32", ("talker_positions", 6), "fp32_strict"),
                _boundary(f"{prefix}.expert_contributions", "talker", "selected_expert_weighted_outputs", "talker_position", "talker_position", "float32", ("talker_positions", 6, 1024), "bf16_activation"),
                _boundary(f"{prefix}.hidden", "talker", "decoder_layer_output", "talker_position", "talker_position", "float32", (1, "talker_positions", 1024), "bf16_accumulation"),
            ]
        )
    boundaries.extend(
        [
            _boundary("talker.primary_codec_logits", "talker", "codec_head", "talker_position", "codec_frame", "float32", (1, "codec_frames", 3072), "fp32_strict"),
            _boundary("talker.primary_codec_token", "talker", "primary_codec_argmax", "codec_frame", "codec_codebook", "int64", (1, "codec_frames"), "exact_discrete"),
        ]
    )
    for residual_step in range(1, 16):
        prefix = f"mtp.residual.{residual_step:02d}"
        boundaries.extend(
            [
                _boundary(f"{prefix}.logits", "mtp", "residual_codec_head", "codec_codebook", "codec_codebook", "float32", (1, "codec_frames", 2048), "fp32_strict"),
                _boundary(f"{prefix}.token", "mtp", "residual_codec_argmax", "codec_codebook", "codec_codebook", "int64", (1, "codec_frames"), "exact_discrete"),
            ]
        )
    boundaries.extend(
        [
            _boundary("code2wav.code_input", "code2wav", "code_embedding_input", "codec_codebook", "codec_frame", "int64", (1, 16, "codec_frames"), "exact_discrete"),
            _boundary("code2wav.chunk.first", "code2wav", "chunked_decode_first", "codec_frame", "pcm_output_sample", "float32", (1, 1, "pcm_samples"), "waveform_fp32", ("nominal_short", "variable_chunk")),
            _boundary("code2wav.chunk.steady", "code2wav", "chunked_decode_steady", "codec_frame", "pcm_output_sample", "float32", (1, 1, "pcm_samples"), "waveform_fp32", ("nominal_short", "variable_chunk", "long_turn")),
            _boundary("code2wav.chunk.final_partial", "code2wav", "chunked_decode_final_partial", "codec_frame", "pcm_output_sample", "float32", (1, 1, "pcm_samples"), "waveform_fp32", ("final_partial_chunk", "turn_end")),
            _boundary("handoff.thinker_to_talker", "handoff", "projected_conditioning_transfer", "thinker_position", "talker_position", "float32", (1, "talker_positions", 1024), "bf16_accumulation", ("nominal_short", "context_reconstruction")),
            _boundary("handoff.talker_to_mtp_hidden", "handoff", "last_talker_hidden_transfer", "talker_position", "codec_frame", "float32", (1, "codec_frames", 1024), "bf16_accumulation"),
            _boundary("handoff.talker_to_mtp_token", "handoff", "primary_codec_transfer", "codec_frame", "codec_codebook", "int64", (1, "codec_frames"), "exact_discrete"),
            _boundary("handoff.codec_to_code2wav", "handoff", "sixteen_codebook_frame_transfer", "codec_codebook", "codec_frame", "int64", (1, 16, "codec_frames"), "exact_discrete", ("nominal_short", "final_partial_chunk")),
            _boundary("scheduler.turn_trace", "scheduler", "state_transition_trace", "scheduler_event", "scheduler_event", "int64", ("events", 4), "exact_discrete", ("turn_end", "long_turn", "interruption", "cancellation", "repeated_phase_transition", "context_reconstruction")),
        ]
    )
    return tuple(boundaries)


BOUNDARY_MATRIX: Final = build_boundary_matrix()


def required_model_fixture_pairs() -> set[tuple[str, str]]:
    """Return boundary-case pairs observable in the pinned PyTorch reference."""
    declared = {(boundary.boundary_id, case_id) for boundary in BOUNDARY_MATRIX for case_id in boundary.case_ids}
    return declared - SYNTHETIC_NATIVE_PAIRS


def validate_contract() -> None:
    """Reject incomplete boundary, case, profile, shape, or provenance contracts."""
    boundary_ids = [boundary.boundary_id for boundary in BOUNDARY_MATRIX]
    if len(boundary_ids) != len(set(boundary_ids)):
        raise ValueError("boundary IDs are not unique")
    case_by_id = {case.case_id: case for case in CASE_MATRIX}
    if len(case_by_id) != len(CASE_MATRIX):
        raise ValueError("case IDs are not unique")
    if {tag for case in CASE_MATRIX for tag in case.adversarial_tags} != ADVERSARIAL_TAGS:
        raise ValueError("adversarial tag coverage is incomplete")
    used_cases = {case_id for boundary in BOUNDARY_MATRIX for case_id in boundary.case_ids}
    if used_cases != set(case_by_id):
        raise ValueError("every declared case must be assigned to a boundary")
    for case in CASE_MATRIX:
        if case.case_class == "nominal" and case.adversarial_tags:
            raise ValueError(f"nominal case has adversarial tags: {case.case_id}")
        if case.case_class == "adversarial" and not case.adversarial_tags:
            raise ValueError(f"adversarial case omits tags: {case.case_id}")
    for boundary in BOUNDARY_MATRIX:
        if boundary.profile_id not in PROFILE_DEFINITIONS:
            raise ValueError(f"unknown tolerance profile: {boundary.boundary_id}")
        if boundary.input_coordinate_domain not in COORDINATE_DOMAINS or boundary.output_coordinate_domain not in COORDINATE_DOMAINS:
            raise ValueError(f"unknown coordinate domain: {boundary.boundary_id}")
        if not boundary.output_shape or any(isinstance(axis, int) and axis < 0 for axis in boundary.output_shape):
            raise ValueError(f"invalid symbolic shape: {boundary.boundary_id}")
        if set(boundary.case_ids) - set(case_by_id):
            raise ValueError(f"unknown case assignment: {boundary.boundary_id}")
        if boundary.output_dtype.startswith("int") and boundary.profile_id != "exact_discrete":
            raise ValueError(f"discrete output lacks exact comparison: {boundary.boundary_id}")
    required_modules = {"input", "aut", "thinker", "talker", "mtp", "code2wav", "handoff", "scheduler"}
    if {boundary.module for boundary in BOUNDARY_MATRIX} != required_modules:
        raise ValueError("module coverage is incomplete")
    if len(set(SCHEDULER_EVENT_CODES.values())) != len(SCHEDULER_EVENT_CODES) or len(set(SCHEDULER_STATE_CODES.values())) != len(SCHEDULER_STATE_CODES):
        raise ValueError("scheduler trace code values are not unique")
    required_ids = {
        "input.pcm_preprocessor",
        "aut.encoder_output",
        "aut.output_projection",
        "thinker.accepted_hidden",
        "thinker.talker_conditioning",
        "talker.primary_codec_logits",
        "talker.primary_codec_token",
        "code2wav.code_input",
        "code2wav.chunk.first",
        "code2wav.chunk.steady",
        "code2wav.chunk.final_partial",
        "handoff.thinker_to_talker",
        "handoff.talker_to_mtp_hidden",
        "handoff.talker_to_mtp_token",
        "handoff.codec_to_code2wav",
        "scheduler.turn_trace",
    }
    for layer in range(20):
        required_ids.update(f"talker.layer.{layer:02d}.{suffix}" for suffix in ("router_logits", "selected_experts", "routing_weights", "expert_contributions", "hidden"))
    for residual_step in range(1, 16):
        required_ids.update(f"mtp.residual.{residual_step:02d}.{suffix}" for suffix in ("logits", "token"))
    if set(boundary_ids) != required_ids:
        raise ValueError("boundary matrix differs from the complete required set")
    declared_pairs = {(boundary.boundary_id, case_id) for boundary in BOUNDARY_MATRIX for case_id in boundary.case_ids}
    if not SYNTHETIC_NATIVE_PAIRS < declared_pairs:
        raise ValueError("native-only fixture pairs must be a strict subset of declared pairs")
    if any(boundary_id != "scheduler.turn_trace" and case_id != "context_reconstruction" for boundary_id, case_id in SYNTHETIC_NATIVE_PAIRS):
        raise ValueError("native-only fixture pairs exceed scheduler or reconstruction scope")
    if required_model_fixture_pairs() | SYNTHETIC_NATIVE_PAIRS != declared_pairs:
        raise ValueError("fixture origin partition is incomplete")
    schema = json.loads((ROOT / "orchestration/schemas/fixture_descriptor.schema.json").read_text(encoding="utf-8"))
    schema_provenance = set(schema["properties"]["provenance"]["required"])
    if REQUIRED_PROVENANCE_FIELDS != schema_provenance:
        raise ValueError("provenance requirements changed")
    same = np.array([1, 2, 3], dtype=np.int64)
    changed = np.array([1, 2, 4], dtype=np.int64)
    if not compare_arrays(same, same.copy(), "exact_discrete")["passed"] or compare_arrays(same, changed, "exact_discrete")["passed"]:
        raise ValueError("exact comparator self-check failed")
    reference = np.array([1.0, -2.0], dtype=np.float32)
    near = np.array([1.0 + 1e-7, -2.0], dtype=np.float32)
    far = np.array([1.1, -2.0], dtype=np.float32)
    if not compare_arrays(reference, near, "fp32_strict")["passed"] or compare_arrays(reference, far, "fp32_strict")["passed"]:
        raise ValueError("floating comparator self-check failed")


def compare_arrays(reference: np.ndarray, candidate: np.ndarray, profile_id: str) -> dict[str, float | int | bool]:
    """Compare one candidate array with one frozen reference under a named profile."""
    if profile_id not in PROFILE_DEFINITIONS:
        raise ValueError(f"unknown comparison profile: {profile_id}")
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, candidate={candidate.shape}")
    if reference.dtype != candidate.dtype:
        raise ValueError(f"dtype mismatch: reference={reference.dtype}, candidate={candidate.dtype}")
    profile = PROFILE_DEFINITIONS[profile_id]
    if profile["mode"] == "exact":
        equal = np.equal(reference, candidate)
        mismatch_fraction = float(1.0 - np.mean(equal)) if equal.size else 0.0
        cosine = 1.0 if mismatch_fraction == 0.0 else 0.0
    else:
        close = np.isclose(reference, candidate, rtol=profile["rtol"], atol=profile["atol"], equal_nan=profile["equal_nan"])
        mismatch_fraction = float(1.0 - np.mean(close)) if close.size else 0.0
        ref = reference.astype(np.float64, copy=False).reshape(-1)
        cand = candidate.astype(np.float64, copy=False).reshape(-1)
        denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
        cosine = 1.0 if denominator == 0.0 and np.array_equal(ref, cand) else (0.0 if denominator == 0.0 else float(np.dot(ref, cand) / denominator))
    passed = mismatch_fraction <= profile["max_mismatch_fraction"] and cosine >= profile["minimum_cosine_similarity"]
    return {"passed": passed, "element_count": int(reference.size), "mismatch_fraction": mismatch_fraction, "cosine_similarity": cosine}


def main() -> int:
    """Validate the contract or compare two NumPy artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFINITIONS))
    args = parser.parse_args()
    try:
        validate_contract()
        if args.contract_only:
            if args.reference or args.candidate or args.profile:
                raise ValueError("--contract-only cannot be combined with array comparison")
            print(f"CONTRACT VALID: {len(BOUNDARY_MATRIX)} boundaries, {len(CASE_MATRIX)} cases")
            return 0
        if not (args.reference and args.candidate and args.profile):
            raise ValueError("array comparison requires --reference, --candidate, and --profile")
        result = compare_arrays(np.load(args.reference, allow_pickle=False), np.load(args.candidate, allow_pickle=False), args.profile)
        print(result)
        return 0 if result["passed"] else 1
    except (OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
