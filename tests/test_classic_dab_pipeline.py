"""End-to-end test of a classic DAB service through `dsp.pipeline.DspWorker`.

From the CIFs of the MSC to the MP2 frames and the dynamic label: time
deinterleaving, UEP decoding, MP2 frame search and PAD evaluation, i.e. exactly the
path that `DspWorker._decode_msc` takes for a service with ASCTy 0. The OFDM and QPSK
layers in front are left out (see test_msc_eep_integration.py for those).
"""

from __future__ import annotations

import numpy as np

from dab.constants import MODE_I
from dab.crc import crc16_ccitt
from dab.fig import UEP_TABLE, AudioComponent, Subchannel
from dsp.convolutional import soft_bits
from dsp.msc import CIF_BITS, NUM_CIFS_PER_FRAME
from dsp.pipeline import DspWorker
from dsp.time_interleave import _INTERLEAVE_MAP, INTERLEAVE_DEPTH
from tests.mp2_test_utils import encode_uep_logical_frame, silent_mp2_frame, with_pad

SID = 0xD123
SUBCHANNEL_ID = 5
UEP_INDEX = 16  # 64 kbit/s, protection level 3, 48 CU
LABEL = b"MP2 im Radio"


def _dls_segment(text: bytes) -> bytes:
    """A complete dynamic label in one segment: prefix, text, CRC (as in test_dls.py)."""
    body = bytes([0x40 | 0x20 | (len(text) - 1), 0x00]) + text
    crc = crc16_ccitt(body) ^ 0xFFFF
    return body + bytes([crc >> 8, crc & 0xFF])


def _transmitted_cifs(logical_frames: list[np.ndarray], subchannel: Subchannel) -> list[np.ndarray]:
    """UEP-encodes, time-interleaves and places the logical frames into full CIFs."""
    coded = [encode_uep_logical_frame(bits, subchannel) for bits in logical_frames]
    fragment_size = subchannel.size * 64
    delay = np.array([_INTERLEAVE_MAP[i % INTERLEAVE_DEPTH] for i in range(fragment_size)])
    positions = np.arange(fragment_size)
    rng = np.random.default_rng(5)
    cifs = []
    for t in range(len(coded)):
        fragment = np.zeros(fragment_size, dtype=np.uint8)
        valid = t - delay >= 0
        fragment[valid] = np.array(coded)[t - delay[valid], positions[valid]]
        cif = rng.integers(0, 2, size=CIF_BITS).astype(np.uint8)  # other subchannels
        start = subchannel.start_address * 64
        cif[start : start + fragment_size] = fragment
        cifs.append(cif)
    return cifs


def test_classic_dab_service_yields_mp2_frames_and_dynamic_label(monkeypatch):
    profile = UEP_TABLE[UEP_INDEX]
    subchannel = Subchannel(
        subchannel_id=SUBCHANNEL_ID,
        start_address=100,
        size=profile.size,
        is_uep=True,
        protection_index=UEP_INDEX,
    )
    # One MP2 frame (48 kHz, 64 kbit/s: 192 bytes) per logical frame, each carrying the
    # label in its X-PAD: a contents indicator for 16 bytes of DLS, the end marker.
    x_pad = bytes([(4 << 5) | 2, 0x00]) + _dls_segment(LABEL)
    frame = with_pad(silent_mp2_frame(profile.bitrate), x_pad, bytes([0x20, 0x02]), 4)
    assert len(frame) * 8 == 24 * profile.bitrate
    logical_frames = [np.unpackbits(np.frombuffer(frame, dtype=np.uint8))] * 40

    worker = DspWorker(2_048_000.0, MODE_I)
    worker._ensemble.audio_components[SID] = AudioComponent(SUBCHANNEL_ID, 0)
    worker._ensemble.subchannels[SUBCHANNEL_ID] = subchannel
    worker.select_service(SID)
    results = []
    labels = []
    worker.mp2_frame_ready.connect(results.append)
    worker.dynamic_label_ready.connect(labels.append)
    # The CIFs go in directly instead of as QPSK symbols.
    monkeypatch.setattr(
        "dsp.pipeline.assemble_msc_cifs", lambda cifs, *_: [soft_bits(cif) for cif in cifs]
    )

    cifs = _transmitted_cifs(logical_frames, subchannel)
    for start in range(0, len(cifs), NUM_CIFS_PER_FRAME):
        worker._decode_msc(cifs[start : start + NUM_CIFS_PER_FRAME])

    # 40 logical frames minus the deinterleaver start-up and the first frames, which
    # still hold the zeros of its unfilled delay stages.
    assert len(results) >= 20
    assert all(r.sync_ok and r.crc_ok and r.frame == frame for r in results)
    assert all(r.x_pad_length == len(x_pad) for r in results)
    assert labels and labels[0].text == LABEL.decode()
