"""End-to-end test of packet mode data services through `dsp.pipeline.DspWorker`.

The FIC is that of channel 5C: EPG Deutschland in subchannel 30 (24 kbit/s EEP-3A,
packet FEC, address 3), GEPOS in subchannel 32 (8 kbit/s, packet FEC, address 2). The
EPG subchannel carries a small MOT carousel, the GEPOS one an RTCM stream, both built
like a transmitter would: data groups cut into packets, packets into FEC frames with
their nine FEC packets, the logical frames EEP-coded and time-interleaved into CIFs.
From there the worker has to find its way back to the carousel's objects and the
ephemerides. The OFDM and QPSK layers are left out (see test_msc_eep_integration.py for
those).
"""

from __future__ import annotations

import numpy as np

from dab.constants import MODE_I
from dab.crc import crc16_ccitt
from dab.fig import Subchannel, parse_fib
from dsp.convolutional import soft_bits
from dsp.msc import CIF_BITS, NUM_CIFS_PER_FRAME
from dsp.packet import ADT_BYTES
from dsp.pipeline import DataComponentStatus, DspWorker
from dsp.time_interleave import _INTERLEAVE_MAP, INTERLEAVE_DEPTH
from tests.dab_signal import fib
from tests.rtcm_vectors import GPS_G12_FRAME
from tests.test_data_services import FIGS_5C
from tests.test_eep import _encode_eep_logical_frame
from tests.test_mot_directory import _directory, _header
from tests.test_packet import _fec_frame, _packet
from tests.test_tpeg import _application_frame, _service_frame
from tests.test_tpeg import _group as _tpeg_group
from tests.tpeg_vectors import (
    DIRECTORY_GROUP,
    FIGS_7A_TPEG,
    SNI_GROUP,
    TEC_ROADWORKS,
    TPEG_ADDRESS,
    TPEG_SCID,
    TPEG_SUBCHANNEL,
)

EPG_SCID = 0x022
EPG_SUBCHANNEL = 30
EPG_ADDRESS = 3
GEPOS_SCID = 0x020
GEPOS_SUBCHANNEL = 32
GEPOS_ADDRESS = 2
LOGO = bytes(range(256)) * 3


def _mot_data_group(group_type: int, transport_id: int, entity: bytes) -> bytes:
    """One MOT segment in a data group (as in test_mot_directory.py, but as bytes)."""
    raw = bytes([0x40 | 0x20 | 0x10 | group_type, 0x00, 0x80, 0x00, 0x12])
    raw += transport_id.to_bytes(2, "big") + len(entity).to_bytes(2, "big") + entity
    return raw + (crc16_ccitt(raw) ^ 0xFFFF).to_bytes(2, "big")


def _tdc_data_group(continuity: int, payload: bytes) -> bytes:
    """A data group of a transparent data channel, as GEPOS sends them: type 0, CRC,
    the continuity index counting on from group to group."""
    raw = bytes([0x40, continuity << 4]) + payload
    return raw + (crc16_ccitt(raw) ^ 0xFFFF).to_bytes(2, "big")


def _packets(address: int, group: bytes, first_continuity: int) -> list[bytes]:
    chunks = [group[i : i + 19] for i in range(0, len(group), 19)]
    return [
        _packet(
            address,
            chunk,
            first=index == 0,
            last=index == len(chunks) - 1,
            continuity=(first_continuity + index) % 4,
        )
        for index, chunk in enumerate(chunks)
    ]


def _packet_stream(address: int, turns: list[list[bytes]]) -> bytes:
    """The byte stream of a subchannel: the data groups of each turn in FEC frames of
    their own."""
    stream = b""
    continuity = 0
    for groups in turns:
        packets = []
        for group in groups:
            group_packets = _packets(address, group, continuity)
            packets += group_packets
            continuity += len(group_packets)
        table = b"".join(packets)
        table += _packet(0, b"") * (-len(table) % ADT_BYTES // 24)
        for start in range(0, len(table), ADT_BYTES):
            stream += _fec_frame(table[start : start + ADT_BYTES])
    return stream


def _epg_stream(turns: int) -> bytes:
    """The carousel (directory and one logo) `turns` times."""
    directory = _directory([(0x0101, _header(len(LOGO), 2, 3, "logo.png"))])
    groups = [_mot_data_group(6, 0x0F00, directory), _mot_data_group(4, 0x0101, LOGO)]
    return _packet_stream(EPG_ADDRESS, [groups] * turns)


def _gepos_stream(turns: int) -> bytes:
    """The same RTCM frame (a GPS ephemeris) in `turns` data groups."""
    return _packet_stream(
        GEPOS_ADDRESS, [[_tdc_data_group(turn % 16, GPS_G12_FRAME)] for turn in range(turns)]
    )


def _transmitted_cifs(streams: list[tuple[bytes, Subchannel]]) -> list[np.ndarray]:
    """Cuts the byte stream of each subchannel into logical frames, EEP-encodes and
    time-interleaves them and places them into the same CIFs.

    The deinterleaver holds back the last 15 logical frames; zeros at the end push them
    out, and fill the subchannels to the same number of whole transmission frames.
    """
    count = max(-(-len(stream) // (3 * sub.bitrate)) for stream, sub in streams)
    count += INTERLEAVE_DEPTH
    count += -count % NUM_CIFS_PER_FRAME
    rng = np.random.default_rng(11)
    cifs = [rng.integers(0, 2, size=CIF_BITS).astype(np.uint8) for _ in range(count)]
    for stream, subchannel in streams:
        frame_bytes = 3 * subchannel.bitrate  # per 24 ms
        stream += bytes(count * frame_bytes - len(stream))
        frames = [
            np.unpackbits(np.frombuffer(stream[i : i + frame_bytes], dtype=np.uint8))
            for i in range(0, len(stream), frame_bytes)
        ]
        coded = np.array([_encode_eep_logical_frame(bits, subchannel) for bits in frames])
        fragment_size = subchannel.size * 64
        delay = np.array([_INTERLEAVE_MAP[i % INTERLEAVE_DEPTH] for i in range(fragment_size)])
        positions = np.arange(fragment_size)
        start = subchannel.start_address * 64
        for t, cif in enumerate(cifs):
            fragment = np.zeros(fragment_size, dtype=np.uint8)
            valid = t - delay >= 0
            fragment[valid] = coded[t - delay[valid], positions[valid]]
            cif[start : start + fragment_size] = fragment
    return cifs


def _worker_knowing(monkeypatch, figs: tuple[str, ...] = FIGS_5C) -> DspWorker:
    """A worker that knows the data services of 5C (or those described by `figs`)."""
    worker = DspWorker(2_048_000.0, MODE_I)
    for fig in figs:
        parse_fib(fib(bytes.fromhex(fig)), worker._ensemble)
    # The CIFs go in directly instead of as QPSK symbols.
    monkeypatch.setattr(
        "dsp.pipeline.assemble_msc_cifs", lambda cifs, *_, **__: [soft_bits(cif) for cif in cifs]
    )
    return worker


def _receive(worker: DspWorker, streams: dict[int, bytes]) -> None:
    """Sends the byte streams of these subchannels (by SubChId) through the MSC decoding."""
    subchannels = worker._ensemble.subchannels
    cifs = _transmitted_cifs([(stream, subchannels[i]) for i, stream in streams.items()])
    for start in range(0, len(cifs), NUM_CIFS_PER_FRAME):
        worker._decode_msc(cifs[start : start + NUM_CIFS_PER_FRAME])


def test_worker_decodes_the_mot_carousel_of_a_packet_mode_service(monkeypatch):
    worker = _worker_knowing(monkeypatch)
    statuses, fec_frames, statistics, directories, objects = [], [], [], [], []
    worker.data_decoding_ready.connect(statuses.append)
    worker.packet_fec_frame_ready.connect(lambda *args: fec_frames.append(args))
    worker.packet_statistics_ready.connect(lambda *args: statistics.append(args))
    worker.mot_directory_ready.connect(lambda *args: directories.append(args))
    worker.data_object_ready.connect(lambda *args: objects.append(args))

    worker.set_data_component_enabled(EPG_SCID, True)
    _receive(worker, {EPG_SUBCHANNEL: _epg_stream(turns=3)})

    (status,) = statuses[-1]
    assert isinstance(status, DataComponentStatus)
    assert status.component.label == "EPG Deutschland"
    assert len(fec_frames) == 3  # one per turn of the carousel
    assert all(sub == EPG_SUBCHANNEL and f.crc_errors_after == 0 for sub, f in fec_frames)
    assert [(scid, d.transport_id) for scid, d in directories] == [(EPG_SCID, 0x0F00)]
    assert [(scid, o.mot_object.content_name, o.mot_object.body, o.spi) for scid, o in objects] == [
        (EPG_SCID, "logo.png", LOGO, None)
    ]
    subchannel_id, last = statistics[-1]
    assert subchannel_id == EPG_SUBCHANNEL
    assert last.fec and last.addresses[EPG_ADDRESS].data_groups >= 4


def test_worker_decodes_two_services_in_different_subchannels_at_once(monkeypatch):
    worker = _worker_knowing(monkeypatch)
    fec_frames, carousels, objects, rtcm, ephemerides = [], [], [], [], []
    worker.packet_fec_frame_ready.connect(lambda sub, frame: fec_frames.append(sub))
    worker.mot_carousel_ready.connect(lambda *args: carousels.append(args))
    worker.data_object_ready.connect(lambda *args: objects.append(args))
    worker.rtcm_statistics_ready.connect(lambda *args: rtcm.append(args))
    worker.ephemeris_ready.connect(ephemerides.append)

    worker.set_data_component_enabled(EPG_SCID, True)
    worker.set_data_component_enabled(GEPOS_SCID, True)
    _receive(
        worker, {EPG_SUBCHANNEL: _epg_stream(turns=3), GEPOS_SUBCHANNEL: _gepos_stream(turns=3)}
    )

    assert sorted(worker._data_subchannels) == [EPG_SUBCHANNEL, GEPOS_SUBCHANNEL]
    # At 8 kbit/s a FEC frame takes three times as long; the EPG subchannel is filled up
    # with zeros meanwhile, which make FEC frames too.
    assert fec_frames.count(GEPOS_SUBCHANNEL) == 3
    assert fec_frames.count(EPG_SUBCHANNEL) >= 3
    assert [(scid, o.mot_object.content_name) for scid, o in objects] == [(EPG_SCID, "logo.png")]
    assert carousels and all(scid == EPG_SCID for scid, _ in carousels)
    # The same ephemeris three times, reported once.
    assert [e.name for e in ephemerides] == ["G12"]
    assert rtcm and all(scid == GEPOS_SCID for scid, _ in rtcm)
    assert rtcm[-1][1].frames >= 1 and rtcm[-1][1].crc_errors == 0


def test_worker_decodes_the_tpeg_stream_of_ard_tpeg(monkeypatch):
    worker = _worker_knowing(monkeypatch, FIGS_7A_TPEG)
    frames, rtcm = [], []
    worker.tpeg_frame_ready.connect(lambda *args: frames.append(args))
    worker.rtcm_statistics_ready.connect(lambda *args: rtcm.append(args))

    worker.set_data_component_enabled(TPEG_SCID, True)
    # As recorded: a stream directory (continuity index 2) and the SNI (3); then a
    # service frame with a real TEC message.
    tec = _tpeg_group(0, _service_frame(_application_frame(1, [TEC_ROADWORKS])))
    stream = _packet_stream(TPEG_ADDRESS, [[DIRECTORY_GROUP, SNI_GROUP, tec]])
    _receive(worker, {TPEG_SUBCHANNEL: stream})

    assert [scid for scid, _ in frames] == [TPEG_SCID] * 3
    directory, sni, service = (frame for _, frame in frames)
    assert directory.frame.directory == ((22, 98, 211),)
    assert sni.service.name == "ARD TPEG"
    (message,) = service.messages
    assert (message.scid, message.management.message_id, message.effect) == (1, 1745, 7)
    assert service.statistics.lost_data_groups == 0
    # TPEG is not searched for RTCM frames.
    assert rtcm == []


def test_stopping_one_service_keeps_the_others(monkeypatch):
    worker = _worker_knowing(monkeypatch)
    statuses = []
    worker.data_decoding_ready.connect(statuses.append)

    worker.set_data_component_enabled(EPG_SCID, True)
    worker.set_data_component_enabled(GEPOS_SCID, True)
    worker.set_data_component_enabled(GEPOS_SCID, True)  # already decoded: no change
    _receive(worker, {EPG_SUBCHANNEL: b"", GEPOS_SUBCHANNEL: b""})
    worker.set_data_component_enabled(EPG_SCID, False)
    _receive(worker, {GEPOS_SUBCHANNEL: b""})

    assert [[(s.scid, s.component.label) for s in status] for status in statuses] == [
        [(EPG_SCID, "EPG Deutschland")],
        [(EPG_SCID, "EPG Deutschland"), (GEPOS_SCID, "GEPOS")],
        [(GEPOS_SCID, "GEPOS")],
    ]
    # The EPG subchannel is no longer decoded.
    assert list(worker._data_subchannels) == [GEPOS_SUBCHANNEL]

    worker.set_data_component_enabled(GEPOS_SCID, False)

    assert statuses[-1] == ()
    assert not worker._data_subchannels


def test_worker_forgets_the_data_components_on_a_new_ensemble():
    worker = DspWorker(2_048_000.0, MODE_I)
    statuses = []
    worker.data_decoding_ready.connect(statuses.append)
    worker.set_data_component_enabled(EPG_SCID, True)

    worker.restart_stream(1)

    assert statuses == [(DataComponentStatus(EPG_SCID, None),), ()]
    assert not worker._data_components
