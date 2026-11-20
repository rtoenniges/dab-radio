"""The CRC-16 used throughout DAB."""

from __future__ import annotations

import binascii


def crc16_ccitt(data: bytes | bytearray | memoryview) -> int:
    """CRC-16/CCITT over `data`: polynomial x^16 + x^12 + x^5 + 1 (0x1021), NOT
    reflected, register initialised to 0xFFFF.

    ETSI EN 300 401, Annex E describes a generic shift register, but DAB uses the
    non-reflected variant (bits enter the register MSB first and the register shifts
    upwards). The equally common reflected CRC-16/X-25 variant with polynomial 0x8408
    is wrong here.

    The final value is NOT inverted. DAB check fields carry the inverted result, so the
    usual comparison is `crc16_ccitt(...) ^ 0xFFFF == transmitted value`, which is what
    `check_crc` does.

    `binascii.crc_hqx` computes exactly this CRC from the start value given, in C: the
    twelve FIBs of a frame alone took about a millisecond as a loop over the bits in
    Python.
    """
    return binascii.crc_hqx(data, 0xFFFF)


def check_crc(data_with_crc: bytes) -> bool:
    """Checks a block whose last two bytes are the (inverted) CRC."""
    if len(data_with_crc) < 2:
        return False
    received = (data_with_crc[-2] << 8) | data_with_crc[-1]
    return crc16_ccitt(data_with_crc[:-2]) ^ 0xFFFF == received
