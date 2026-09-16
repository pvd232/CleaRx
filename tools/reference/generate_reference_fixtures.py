#!/usr/bin/env python3
"""Define the frozen Code2Wav input shape and its decoder windows.

Verify the contract from the repository root:

    $ conda activate clearx
    $ python - <<'PY'
    from tools.reference.generate_reference_fixtures import (
        CODE2WAV_FIXTURE_CODE_DTYPE,
        CODE2WAV_FIXTURE_CODE_SHAPE,
        code2wav_decode_windows,
        validate_code2wav_fixture_contract,
    )

    validate_code2wav_fixture_contract()
    assert CODE2WAV_FIXTURE_CODE_SHAPE == (1, 16, 601)
    assert CODE2WAV_FIXTURE_CODE_DTYPE == "int64"

    for window in code2wav_decode_windows():
        print(window)
    PY

Expected output:

    Code2WavDecodeWindow(new_frame_start=0, new_frame_stop=300, input_frame_start=0, left_context_frame_count=0)
    Code2WavDecodeWindow(new_frame_start=300, new_frame_stop=600, input_frame_start=275, left_context_frame_count=25)
    Code2WavDecodeWindow(new_frame_start=600, new_frame_stop=601, input_frame_start=575, left_context_frame_count=25)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

CODE2WAV_QUANTIZER_COUNT: Final = 16
CODE2WAV_FIXTURE_FRAME_COUNT: Final = 601
CODE2WAV_FIXTURE_CODE_SHAPE: Final = (
    1,
    CODE2WAV_QUANTIZER_COUNT,
    CODE2WAV_FIXTURE_FRAME_COUNT,
)
CODE2WAV_FIXTURE_CODE_DTYPE: Final = "int64"

CODE2WAV_CHUNK_FRAME_COUNT: Final = 300
CODE2WAV_LEFT_CONTEXT_FRAME_COUNT: Final = 25


@dataclass(frozen=True)
class Code2WavDecodeWindow:
    """Map one new codec-frame interval to its context-inclusive decoder input."""

    new_frame_start: int
    new_frame_stop: int
    input_frame_start: int
    left_context_frame_count: int


def code2wav_decode_windows(
    frame_count: int = CODE2WAV_FIXTURE_FRAME_COUNT,
) -> tuple[Code2WavDecodeWindow, ...]:
    """Return every Code2Wav call required to decode the given codec frames."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")

    windows: list[Code2WavDecodeWindow] = []
    new_frame_start = 0

    while new_frame_start < frame_count:
        new_frame_stop = min(
            new_frame_start + CODE2WAV_CHUNK_FRAME_COUNT,
            frame_count,
        )
        # The pinned Transformers chunked_decode implementation caps the left
        # context at the number of codec frames preceding this decoder call.
        left_context_frame_count = (
            CODE2WAV_LEFT_CONTEXT_FRAME_COUNT
            if new_frame_start - CODE2WAV_LEFT_CONTEXT_FRAME_COUNT > 0
            else new_frame_start
        )
        windows.append(
            Code2WavDecodeWindow(
                new_frame_start=new_frame_start,
                new_frame_stop=new_frame_stop,
                input_frame_start=new_frame_start - left_context_frame_count,
                left_context_frame_count=left_context_frame_count,
            )
        )
        new_frame_start = new_frame_stop

    return tuple(windows)


def validate_code2wav_fixture_contract() -> None:
    """Reject a Code2Wav fixture shape that misses a declared decode state."""
    expected = (
        Code2WavDecodeWindow(0, 300, 0, 0),
        Code2WavDecodeWindow(300, 600, 275, 25),
        Code2WavDecodeWindow(600, 601, 575, 25),
    )
    if code2wav_decode_windows() != expected:
        raise ValueError("601-frame Code2Wav fixture windows changed")

    if len(code2wav_decode_windows(600)) != 2:
        raise ValueError("600 frames must produce exactly two decode windows")
