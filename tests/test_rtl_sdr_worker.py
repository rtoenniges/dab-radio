import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rtlsdr.rtlsdr import LibUSBError

from sdr.device import DISCARD_AFTER_OPEN_SAMPLES
from sdr.exceptions import RtlSdrError
from sdr.worker import DEFAULT_READ_SIZE, RtlSdrWorker


@pytest.fixture
def worker():
    return RtlSdrWorker(center_freq=178_352_000.0)


def test_open_device_raises_when_no_device_found(worker):
    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = []

        with pytest.raises(RtlSdrError, match="No RTL-SDR device found"):
            worker._open_device()

        mock_rtlsdr_cls.assert_not_called()


def test_open_device_raises_friendly_message_when_busy(worker):
    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = ["00000001"]
        mock_rtlsdr_cls.side_effect = LibUSBError(-6, "Resource busy")

        with pytest.raises(RtlSdrError, match="device busy"):
            worker._open_device()


def test_run_reports_device_busy_while_reading_serials(worker):
    """Reading the serial numbers already opens the stick and fails if another program
    uses it (on Windows with LIBUSB_ERROR_ACCESS). That must end as `error`, otherwise
    the GUI waits for the connection forever."""
    errors = []
    worker.error.connect(errors.append)

    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.side_effect = LibUSBError(
            -3, "while reading USB strings (device 0)"
        )

        worker.run()

    expected = "device busy" if sys.platform == "win32" else "Access"
    assert len(errors) == 1
    assert expected in errors[0]
    mock_rtlsdr_cls.assert_not_called()


def test_open_device_configures_and_returns_sdr(worker):
    mock_sdr = MagicMock()
    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = ["00000001"]
        mock_rtlsdr_cls.return_value = mock_sdr

        result = worker._open_device()

    assert result is mock_sdr
    assert mock_sdr.sample_rate == worker._sample_rate
    assert mock_sdr.center_freq == worker._center_freq
    assert mock_sdr.gain == worker._gain
    # With AGC ("auto", the default) the separate RTL2832 AGC stage must be enabled
    # explicitly as well (see the comment in sdr.device.open_rtlsdr).
    mock_sdr.set_agc_mode.assert_called_once_with(True)


def test_open_device_disables_rtl_agc_for_manual_gain():
    worker = RtlSdrWorker(center_freq=178_352_000.0, gain=20.0)
    mock_sdr = MagicMock()
    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = ["00000001"]
        mock_rtlsdr_cls.return_value = mock_sdr

        worker._open_device()

    # Otherwise the RTL2832's own AGC stage could keep overriding a manually chosen
    # gain (see the comment in sdr.device.open_rtlsdr).
    mock_sdr.set_agc_mode.assert_called_once_with(False)
    assert mock_sdr.gain == 20.0


def test_run_emits_connected_and_samples_then_stops(worker):
    """`run()` reads asynchronously: librtlsdr calls the callback repeatedly until
    `cancel_read_async()` ends the blocking `read_samples_async()`.

    The synchronous path (`read_samples()` in a loop) would not be equivalent here:
    it loses samples between two calls, which makes MSC decoding permanently
    impossible (see `RtlSdrWorker.run`).

    The blocks of the first `DISCARD_AFTER_OPEN_SAMPLES` still come from the
    previously tuned frequency and are not passed on.
    """
    mock_sdr = MagicMock()

    def fake_read_async(callback, num_samples):
        # librtlsdr keeps delivering blocks until the read is cancelled;
        # `cancel_read_async` makes `read_samples_async` return. Every block carries
        # its running number.
        block = 0
        while not mock_sdr.cancel_read_async.called:
            callback(np.full(num_samples, block, dtype=np.complex64), mock_sdr)
            block += 1

    mock_sdr.read_samples_async.side_effect = fake_read_async

    connected_messages = []
    received_batches = []
    stopped_flags = []

    worker.connected.connect(connected_messages.append)
    worker.samples_ready.connect(received_batches.append)
    worker.samples_ready.connect(lambda _samples: setattr(worker, "_running", False))
    worker.stopped.connect(lambda: stopped_flags.append(True))

    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = ["00000001"]
        mock_rtlsdr_cls.return_value = mock_sdr

        worker.run()

    assert len(connected_messages) == 1
    assert len(received_batches) == 1
    first_kept_block = -(-DISCARD_AFTER_OPEN_SAMPLES // DEFAULT_READ_SIZE)
    assert received_batches[0][0] == first_kept_block
    assert stopped_flags == [True]
    mock_sdr.close.assert_called_once()


def _run_with_retune(worker, set_center_freq_result: int = 0) -> tuple[MagicMock, list, list]:
    """Runs the worker over ten blocks with running numbers and retunes it to 5D
    right after the second block kept, the way the GUI thread would between two
    callbacks. One block reaches the callback while the tuner is switching.

    Returns the device mock, the emitted `tuned`/`samples_ready` events in order, and
    what `retune` returned or raised.
    """
    mock_sdr = MagicMock()
    first_kept = -(-DISCARD_AFTER_OPEN_SAMPLES // DEFAULT_READ_SIZE)
    retune_results = []

    def fake_read_async(callback, num_samples):
        block = 0

        def deliver():
            nonlocal block
            callback(np.full(num_samples, block, dtype=np.complex64), mock_sdr)
            block += 1

        def set_center_freq(_dev, _freq):
            deliver()
            return set_center_freq_result

        librtlsdr.rtlsdr_set_center_freq.side_effect = set_center_freq
        while block < 10:
            deliver()
            if block == first_kept + 2:
                try:
                    retune_results.append(worker.retune(180_064_000.0))
                except RtlSdrError as exc:
                    retune_results.append(exc)

    mock_sdr.read_samples_async.side_effect = fake_read_async
    events = []
    worker.tuned.connect(lambda tuning: events.append(("tuned", tuning)))
    worker.samples_ready.connect(lambda samples: events.append(("block", int(samples[0].real))))

    with (
        patch("sdr.device.RtlSdr") as mock_rtlsdr_cls,
        patch("sdr.device.librtlsdr") as librtlsdr,
    ):
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = ["00000001"]
        mock_rtlsdr_cls.return_value = mock_sdr
        worker.run()

    assert mock_rtlsdr_cls.call_count == 1  # the device is never reopened
    mock_sdr.close.assert_called_once()
    librtlsdr.rtlsdr_set_center_freq.assert_called_once_with(mock_sdr.dev_p, 180_064_000)
    return mock_sdr, events, retune_results


def test_retune_discards_the_switch_and_announces_the_new_frequency(worker):
    """Discarded are the block arriving during the switch, the one that was being
    filled at the switch (still the old frequency) and one more for settling. The
    first block of the new frequency is preceded by `tuned`, so that everything
    downstream knows where the new frequency begins (see `RtlSdrWorker.retune`)."""
    opened_tuning = worker.tuning

    _, events, retune_results = _run_with_retune(worker)

    [new_tuning] = retune_results
    assert new_tuning != opened_tuning
    assert worker.tuning == new_tuning
    first_kept = -(-DISCARD_AFTER_OPEN_SAMPLES // DEFAULT_READ_SIZE)
    assert events == [
        ("tuned", opened_tuning),
        ("block", first_kept),
        ("block", first_kept + 1),
        # first_kept + 2: during the switch; + 3 and + 4: filling and settling
        ("tuned", new_tuning),
        *[("block", block) for block in range(first_kept + 5, 10)],
    ]


def test_failed_retune_raises_and_passes_nothing_on(worker):
    """The device stays open (pyrtlsdr's own setter would close it under the reading
    thread), and no samples of an unknown frequency are passed on."""
    _, events, retune_results = _run_with_retune(worker, set_center_freq_result=-1)

    [error] = retune_results
    assert isinstance(error, RtlSdrError)
    assert "Retuning to 180.064 MHz failed" in str(error)
    first_kept = -(-DISCARD_AFTER_OPEN_SAMPLES // DEFAULT_READ_SIZE)
    assert events[-1] == ("block", first_kept + 1)


def test_retune_is_ignored_while_not_reading(worker):
    with patch("sdr.device.librtlsdr") as librtlsdr:
        assert worker.retune(180_064_000.0) is None

    librtlsdr.rtlsdr_set_center_freq.assert_not_called()


def test_run_emits_error_when_read_fails(worker):
    mock_sdr = MagicMock()
    mock_sdr.read_samples_async.side_effect = OSError("USB read failed")

    errors = []
    worker.error.connect(errors.append)

    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = ["00000001"]
        mock_rtlsdr_cls.return_value = mock_sdr

        worker.run()

    assert len(errors) == 1
    assert "Error reading" in errors[0]
    mock_sdr.close.assert_called_once()


def test_run_emits_error_when_no_device_found():
    worker = RtlSdrWorker(center_freq=178_352_000.0)
    errors = []
    worker.error.connect(errors.append)

    with patch("sdr.device.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.get_device_serial_addresses.return_value = []

        worker.run()

    assert len(errors) == 1
    assert "No RTL-SDR device found" in errors[0]
