"""Tests for the AAC decoding of the DAB+ audio units."""

from __future__ import annotations

import numpy as np
import pytest

from audio.aac import AacDecoder, AacDecoderError, audio_specific_config, is_available
from dsp.superframe import AudioFormat

requires_faad = pytest.mark.skipif(not is_available(), reason="FAAD2 (libfaad2) not installed")


def _format(dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False) -> AudioFormat:
    return AudioFormat(
        dac_rate_48k=dac_rate_48k,
        sbr=sbr,
        stereo_core=stereo_core,
        parametric_stereo=parametric_stereo,
        mpeg_surround_config=0,
    )


# All combinations of the format byte that are possible in DAB+ at all (ETSI TS 102
# 563, clause 5.2).
ALL_FORMATS = [
    _format(dac_rate_48k=d, sbr=s, stereo_core=c, parametric_stereo=p)
    for d in (True, False)
    for s in (True, False)
    for c in (True, False)
    for p in ((True, False) if s else (False,))
]


def test_audio_specific_config_without_sbr_is_plain_aac_lc():
    """Derived by hand from ISO/IEC 14496-3, clause 1.6.2.1:

        audioObjectType        5 bits = 2  (AAC LC)   -> 00010
        samplingFrequencyIndex 4 bits = 3  (48 kHz)   -> 0011
        channelConfiguration   4 bits = 2  (stereo)   -> 0010
        frameLengthFlag        1 bit  = 1  (960!)     -> 1
        dependsOnCoreCoder     1 bit  = 0             -> 0
        extensionFlag          1 bit  = 0             -> 0

    gives 00010001 10010100 = 0x11 0x94.
    """
    assert audio_specific_config(_format(sbr=False)) == bytes([0x11, 0x94])


def test_audio_specific_config_with_sbr_uses_hierarchical_signalling():
    """With SBR the SBR object type with the CORE rate comes first, then the output
    rate, and only then the core object type:

        audioObjectType                 5 bits = 5  (SBR)     -> 00101
        samplingFrequencyIndex          4 bits = 6  (24 kHz)  -> 0110
        channelConfiguration            4 bits = 2            -> 0010
        extensionSamplingFrequencyIndex 4 bits = 3  (48 kHz)  -> 0011
        audioObjectType                 5 bits = 2  (AAC LC)  -> 00010
        frameLengthFlag/depends/ext     3 bits                -> 100

    gives 00101011 00010001 10001010 0(padding) = 0x2B 0x11 0x8A 0x00.
    """
    assert audio_specific_config(_format()) == bytes([0x2B, 0x11, 0x8A, 0x00])


def test_audio_specific_config_signals_mono_core_for_parametric_stereo():
    """With PS the AAC core is single-channel; the stereo effect only arises in the
    decoder. PS itself is deliberately not signalled (see `audio_specific_config`),
    so the configuration is identical to that of a real mono core."""
    with_ps = audio_specific_config(_format(stereo_core=False, parametric_stereo=True))
    mono = audio_specific_config(_format(stereo_core=False, parametric_stereo=False))
    assert with_ps == mono
    assert with_ps != audio_specific_config(_format(stereo_core=True))


@pytest.mark.parametrize("audio_format", ALL_FORMATS, ids=lambda f: f.describe())
def test_audio_specific_config_length(audio_format):
    """25 bits with SBR, 16 without, i.e. padded to 4 or 2 bytes."""
    assert len(audio_specific_config(audio_format)) == (4 if audio_format.sbr else 2)


@pytest.mark.parametrize(
    ("audio_format", "expected"),
    [
        (_format(dac_rate_48k=True, sbr=True), 3),
        (_format(dac_rate_48k=True, sbr=False), 6),
        (_format(dac_rate_48k=False, sbr=True), 2),
        (_format(dac_rate_48k=False, sbr=False), 4),
    ],
)
def test_number_of_audio_units_per_superframe(audio_format, expected):
    """ETSI TS 102 563, table 3. A superframe lasts 120 ms, an access unit spans 960
    core samples."""
    assert audio_format.num_audio_units == expected


@requires_faad
@pytest.mark.parametrize("audio_format", ALL_FORMATS, ids=lambda f: f.describe())
def test_faad_accepts_every_dab_plus_format(audio_format):
    """The sharpest test of the AudioSpecificConfig: FAAD2 reads it itself and reports
    back which sample rate it derives from it. A wrongly built configuration would
    yield a different rate here (or be rejected outright)."""
    with AacDecoder(audio_format) as decoder:
        assert decoder.sample_rate == audio_format.sample_rate


@requires_faad
def test_decoder_rejects_garbage_audio_unit():
    rng = np.random.default_rng(0)
    garbage = bytes(rng.integers(0, 256, size=200, dtype=np.uint8))
    with AacDecoder(_format()) as decoder, pytest.raises(AacDecoderError):
        decoder.decode(garbage)


@requires_faad
def test_decoder_returns_nothing_for_empty_audio_unit():
    with AacDecoder(_format()) as decoder:
        assert len(decoder.decode(b"")) == 0


@requires_faad
def test_decoder_refuses_use_after_close():
    decoder = AacDecoder(_format())
    decoder.close()
    with pytest.raises(AacDecoderError):
        decoder.decode(b"\x00" * 10)


@requires_faad
def test_closing_twice_is_harmless():
    decoder = AacDecoder(_format())
    decoder.close()
    decoder.close()
