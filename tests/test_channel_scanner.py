from unittest.mock import MagicMock, patch

import numpy as np
from rtlsdr.rtlsdr import LibUSBError

from sdr.device import DISCARD_AFTER_OPEN_SAMPLES, RETUNE_SETTLE_SAMPLES
from sdr.exceptions import SdrplayError
from sdr.rspduo import RspDuoSettings
from sdr.scanner import SCAN_BLOCK_SAMPLES, ChannelScanner
from tests.dab_signal import (
    build_stream,
    fib,
    fig0_2_audio_services,
    fig1_ensemble_label,
    fig1_service_label,
    noise,
)

_CHANNELS = {"5B": 176.640, "5C": 178.352, "5D": 180.064}


def _fake_sdr(ensemble_channel: str) -> MagicMock:
    """A stick that receives an ensemble on `ensemble_channel` and only noise on all
    other channels."""
    rng = np.random.default_rng(5)
    fibs = [
        fib(fig1_ensemble_label(0x10BC, "Test Mux")),
        fib(fig1_service_label(0xD001, "Radio Eins")),
        fib(fig0_2_audio_services([(0xD001, 1)])),
    ]
    stream = build_stream(fibs + [fib()] * 9, num_frames=28, rng=rng, start_offset=30_000)
    ensemble_freq = round(_CHANNELS[ensemble_channel] * 1e6)
    sdr = MagicMock()
    position = {"samples": 0}

    def read_samples(num_samples):
        if sdr.center_freq != ensemble_freq:
            return noise(num_samples, rng)
        start = position["samples"]
        position["samples"] += num_samples
        return stream[start : start + num_samples]

    sdr.read_samples.side_effect = read_samples
    return sdr


def _run(scanner: ChannelScanner, sdr: MagicMock) -> dict:
    events = {"started": [], "scanned": [], "errors": [], "finished": []}
    scanner.channel_started.connect(lambda ch, i, n: events["started"].append((ch, i, n)))
    scanner.channel_scanned.connect(lambda ch, info: events["scanned"].append((ch, info)))
    scanner.error.connect(events["errors"].append)
    scanner.scan_finished.connect(events["finished"].append)
    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = ["00000001"]
        mock_rtlsdr_cls.return_value = sdr
        scanner.run()
    return events


def test_scan_reports_every_channel_and_finds_the_ensemble():
    sdr = _fake_sdr("5C")

    events = _run(ChannelScanner(_CHANNELS), sdr)

    assert events["started"] == [("5B", 0, 3), ("5C", 1, 3), ("5D", 2, 3)]
    results = dict(events["scanned"])
    assert results["5B"] is None
    assert results["5D"] is None
    assert results["5C"].label == "Test Mux"
    assert results["5C"].services == ((0xD001, "Radio Eins"),)
    assert events["finished"] == [True]
    assert events["errors"] == []
    sdr.close.assert_called_once()
    # Discarded first are the leftovers of the previously tuned frequency, then the
    # settling after every retune.
    sizes = [call.args[0] for call in sdr.read_samples.call_args_list]
    assert sizes[0] == DISCARD_AFTER_OPEN_SAMPLES
    assert sizes.count(RETUNE_SETTLE_SAMPLES) == 3
    assert set(sizes[1:]) == {RETUNE_SETTLE_SAMPLES, SCAN_BLOCK_SAMPLES}


def test_cancel_discards_the_interrupted_channel():
    sdr = _fake_sdr("5C")
    scanner = ChannelScanner(_CHANNELS)
    # Cancel while the second channel is being read.
    scanner.channel_started.connect(lambda ch, i, n: scanner.cancel() if ch == "5C" else None)

    events = _run(scanner, sdr)

    assert events["scanned"] == [("5B", None)]
    assert events["finished"] == [False]
    sdr.close.assert_called_once()


def test_scan_reports_missing_device():
    scanner = ChannelScanner(_CHANNELS)
    events = {"errors": [], "finished": []}
    scanner.error.connect(events["errors"].append)
    scanner.scan_finished.connect(events["finished"].append)

    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = []
        scanner.run()

    assert len(events["errors"]) == 1
    assert "No RTL-SDR device found" in events["errors"][0]
    assert events["finished"] == [False]


def test_scan_reports_busy_device_and_finishes():
    """Otherwise the GUI would stay in "Suchlauf: verbinde..." for good, since only
    `scan_finished` releases the controls again."""
    scanner = ChannelScanner(_CHANNELS)
    events = {"errors": [], "finished": []}
    scanner.error.connect(events["errors"].append)
    scanner.scan_finished.connect(events["finished"].append)

    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.side_effect = LibUSBError(
            -6, "while reading USB strings (device 0)"
        )
        scanner.run()

    assert len(events["errors"]) == 1
    assert "device busy" in events["errors"][0]
    assert events["finished"] == [False]


def test_scan_reports_read_error_and_closes_device():
    sdr = MagicMock()
    sdr.read_samples.side_effect = OSError("USB read failed")

    events = _run(ChannelScanner(_CHANNELS), sdr)

    assert len(events["errors"]) == 1
    assert "Error during the scan" in events["errors"][0]
    assert events["finished"] == [False]
    sdr.close.assert_called_once()


def test_scan_with_the_rspduo_reads_through_its_reader():
    sdr = _fake_sdr("5C")
    opened = []

    def reader_factory(settings, center_freq):
        opened.append((settings, center_freq))
        return sdr

    scanner = ChannelScanner(
        _CHANNELS, rspduo=RspDuoSettings(tuner=2), rspduo_reader_factory=reader_factory
    )
    events = {"scanned": [], "errors": [], "finished": []}
    scanner.channel_scanned.connect(lambda ch, info: events["scanned"].append((ch, info)))
    scanner.error.connect(events["errors"].append)
    scanner.scan_finished.connect(events["finished"].append)

    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        scanner.run()

    mock_rtlsdr_cls.assert_not_called()
    assert opened == [(RspDuoSettings(tuner=2), 176_640_000)]
    sdr.open.assert_called_once()
    assert dict(events["scanned"])["5C"].label == "Test Mux"
    assert events["finished"] == [True]
    assert events["errors"] == []
    sdr.close.assert_called_once()


def test_scan_reports_an_rspduo_that_cannot_be_opened():
    sdr = MagicMock()
    sdr.open.side_effect = SdrplayError("The RSPduo is busy.")
    scanner = ChannelScanner(
        _CHANNELS, rspduo=RspDuoSettings(), rspduo_reader_factory=lambda s, f: sdr
    )
    events = {"errors": [], "finished": []}
    scanner.error.connect(events["errors"].append)
    scanner.scan_finished.connect(events["finished"].append)

    scanner.run()

    assert events["errors"] == ["The RSPduo is busy."]
    assert events["finished"] == [False]
    sdr.read_samples.assert_not_called()
