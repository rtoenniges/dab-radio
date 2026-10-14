"""Tests for the MP2 decoding of classic DAB services through libmpg123."""

from __future__ import annotations

import numpy as np
import pytest

from audio.mp2 import Mp2Decoder, Mp2DecoderError, is_available
from dsp.mp2_frame import CHANNEL_MODE_STEREO, SAMPLES_PER_FRAME, parse_header
from tests.mp2_test_utils import REAL_FRAME, silent_mp2_frame

requires_mpg123 = pytest.mark.skipif(not is_available(), reason="libmpg123 not installed")


def _decode_all(decoder: Mp2Decoder, frames: list[bytes]) -> np.ndarray:
    return np.concatenate([decoder.decode(frame) for frame in frames])


@requires_mpg123
def test_real_frames_decode_to_the_encoded_tone():
    """The frames were encoded from a 1 kHz sine (see `REAL_FRAME`); after decoding,
    the strongest frequency must be exactly that."""
    audio_format = parse_header(REAL_FRAME).audio_format
    with Mp2Decoder(audio_format) as decoder:
        pcm = _decode_all(decoder, [REAL_FRAME] * 6)

        assert decoder.sample_rate == 48_000
        assert decoder.channels == 1

    # mpg123 may hold back the first frame until it has seen the next header, but no
    # more than that.
    assert len(pcm) >= 5 * SAMPLES_PER_FRAME
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float64)))
    peak = np.fft.rfftfreq(len(pcm), 1 / 48_000)[np.argmax(spectrum)]
    assert peak == pytest.approx(1000, abs=10)


@requires_mpg123
def test_stereo_at_24_khz_yields_two_channels_at_the_low_rate():
    frame = silent_mp2_frame(64, lsf=True, channel_mode=CHANNEL_MODE_STEREO)
    with Mp2Decoder(parse_header(frame).audio_format) as decoder:
        pcm = _decode_all(decoder, [frame] * 4)

        assert decoder.sample_rate == 24_000
        assert decoder.channels == 2
    assert len(pcm) % 2 == 0
    assert not pcm.any(), "all bit allocations are zero, so the frames are silence"


@requires_mpg123
def test_closed_decoder_refuses_to_decode():
    decoder = Mp2Decoder(parse_header(REAL_FRAME).audio_format)
    decoder.close()

    with pytest.raises(Mp2DecoderError):
        decoder.decode(REAL_FRAME)
