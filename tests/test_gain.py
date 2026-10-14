from unittest.mock import MagicMock, patch

from sdr.gain import FALLBACK_GAIN_RANGE_DB, FALLBACK_GAINS_DB, query_gain_range, query_gains


def test_query_gain_range_returns_min_and_max():
    mock_sdr = MagicMock()
    mock_sdr.valid_gains_db = [0.0, 14.4, 15.7, 49.6]

    with patch("sdr.gain.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.return_value = mock_sdr

        result = query_gain_range()

    assert result == (0.0, 49.6)
    mock_sdr.close.assert_called_once()


def test_query_gain_range_returns_none_when_no_device():
    with patch("sdr.gain.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.side_effect = OSError("no device")

        result = query_gain_range()

    assert result is None


def test_query_gain_range_returns_none_when_no_gains_reported():
    mock_sdr = MagicMock()
    mock_sdr.valid_gains_db = []

    with patch("sdr.gain.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.return_value = mock_sdr

        result = query_gain_range()

    assert result is None
    mock_sdr.close.assert_called_once()


def test_query_gains_returns_all_steps_in_ascending_order():
    mock_sdr = MagicMock()
    mock_sdr.valid_gains_db = [14.4, 0.0, 49.6, 15.7]

    with patch("sdr.gain.RtlSdr") as mock_rtlsdr_cls:
        mock_rtlsdr_cls.return_value = mock_sdr

        result = query_gains()

    assert result == (0.0, 14.4, 15.7, 49.6)


def test_the_fallback_is_the_r820t_table():
    assert len(FALLBACK_GAINS_DB) == 29
    assert list(FALLBACK_GAINS_DB) == sorted(FALLBACK_GAINS_DB)
    assert FALLBACK_GAIN_RANGE_DB == (0.0, 49.6)
