"""Tests for the CRC-16 used throughout DAB."""

import random

from dab.crc import check_crc, crc16_ccitt


def _shift_register(data: bytes) -> int:
    """The CRC bit by bit: MSB first into a register starting at 0xFFFF, polynomial
    0x1021, not reflected, not inverted."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def test_crc16_ccitt_matches_standard_test_vector():
    # Well-known reference value of CRC-16/CCITT-FALSE over "123456789".
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_crc16_ccitt_is_the_shift_register_of_the_standard():
    rng = random.Random(5)
    for length in (0, 1, 2, 30, 31, 255, 1000):
        data = bytes(rng.randrange(256) for _ in range(length))
        assert crc16_ccitt(data) == _shift_register(data)
        assert crc16_ccitt(bytearray(data)) == _shift_register(data)
        assert crc16_ccitt(memoryview(data)) == _shift_register(data)


def test_check_crc_accepts_inverted_crc_and_rejects_corruption():
    data = b"DAB check field"
    crc = crc16_ccitt(data) ^ 0xFFFF
    block = data + crc.to_bytes(2, "big")

    assert check_crc(block) is True
    assert check_crc(bytes([block[0] ^ 1]) + block[1:]) is False


def test_check_crc_rejects_blocks_shorter_than_the_crc():
    assert check_crc(b"") is False
    assert check_crc(b"\x00") is False
