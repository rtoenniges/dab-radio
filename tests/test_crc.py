"""Tests for the CRC-16 used throughout DAB."""

from dab.crc import check_crc, crc16_ccitt


def test_crc16_ccitt_matches_standard_test_vector():
    # Well-known reference value of CRC-16/CCITT-FALSE over "123456789".
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_check_crc_accepts_inverted_crc_and_rejects_corruption():
    data = b"DAB check field"
    crc = crc16_ccitt(data) ^ 0xFFFF
    block = data + crc.to_bytes(2, "big")

    assert check_crc(block) is True
    assert check_crc(bytes([block[0] ^ 1]) + block[1:]) is False


def test_check_crc_rejects_blocks_shorter_than_the_crc():
    assert check_crc(b"") is False
    assert check_crc(b"\x00") is False
