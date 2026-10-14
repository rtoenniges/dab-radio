"""Receiver access: opening the RTL-SDR stick or a tuner of the SDRplay RSPduo, reading
the IQ stream and scanning channels.

The error messages raised and emitted here are shown to the user in the GUI and are
therefore marked for translation (`i18n`), unlike the log messages.
"""

from sdr.library import install

# Before any module of this package imports `rtlsdr`: pyrtlsdr loads librtlsdr on import.
install()
