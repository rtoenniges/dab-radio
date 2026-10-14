"""Tests for the SPI binary decoding (ETSI TS 102 371) and the programme guide."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from dab.mot import MotObject
from dab.spi import (
    PARAM_PROFILE_SUBSET,
    PARAM_SCOPE_ID,
    Bearer,
    Names,
    ProgrammeGuide,
    Schedule,
    ServiceInformation,
    decode_bearer,
    decode_binary_object,
    decode_spi_object,
    decode_timepoint,
    expand_text,
    merge_programmes,
    profile_names,
)
from tests.spi_vectors import DLF_NOVA_PI_BODY, NDR_SI_GROUP_0, NDR_SI_GROUP_1, NDR_SI_GROUP_2

# Annex C.2 of the standard: a schedule with one programme, "PM" on 18.12.2003.
ANNEX_C2_PI = bytes.fromhex(
    "0235213324168004 33bfc440 8104 33bfc480 2508 8006 40e1ce15c224"
    "1c19 8103 fae451 1104 0102 504d 190c 2c0a 8004 33bfc440 8102 0e10".replace(" ", "")
)

# Annex C.1: the SI of "London 1" with the service "Capital FM", its logos and bearer.
# The bytes as printed in the standard, "imjge/png" typo included.
ANNEX_C1_SI = bytes.fromhex(
    "039e269c8003e1c185100a01084c6f6e646f6e2031110a01084c6f6e646f6e2031"
    "287d100901074361706974616c110c010a4361706974616c20464d"
    "130b2b09820434373953830106"
    "130b2b09820434373952830104"
    "131e2b1c820434373953830102800969 6d6a67652f706e67850200808402 0080".replace(" ", "")
    + "131e2b1c82043437394c830102800969 6d6a67652f706e67850200f0840201 40".replace(" ", "")
    + "2908800640e1c185c479"
)


def _utc(*fields: int) -> float:
    return datetime(*fields, tzinfo=UTC).timestamp()


def _mot(body: bytes, subtype: int, name: str = "x", parameters: dict | None = None) -> MotObject:
    return MotObject(1, 7, subtype, name, body, parameters or {})


def _add(guide: ProgrammeGuide, mot_object: MotObject) -> bool:
    """Hands an object to the guide the way the signal processing does."""
    return guide.add(mot_object, decode_spi_object(mot_object))


def _ndr_si() -> bytes:
    """The body of the NDR SI object: its three segments without data group header,
    segmentation header and CRC."""
    body = b""
    for group in (NDR_SI_GROUP_0, NDR_SI_GROUP_1, NDR_SI_GROUP_2):
        segment = group[7:-2]  # 2 header, 2 segment field, 3 user access
        size = ((segment[0] & 0x1F) << 8) | segment[1]
        body += segment[2 : 2 + size]
    return body


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        # Annex C.2: short form; the LTO flag is set, but no offset byte follows.
        (bytes.fromhex("33bfc440"), _utc(2003, 12, 18, 17, 0)),
        # Deutschlandfunk: short form with LTO +2 h; 22:00 UTC is midnight local time.
        (bytes.fromhex("3be0d58004"), _utc(2026, 10, 2, 22, 0)),
        # Long form with seconds (creation time of a schedule from 5C).
        (bytes.fromhex("3bdfcb801000"), _utc(2026, 9, 28, 14, 0, 4)),
    ],
)
def test_decode_timepoint(data, expected):
    assert decode_timepoint(data) == expected


def test_decode_timepoint_rejects_truncated_data():
    assert decode_timepoint(bytes.fromhex("33bf")) is None
    # The UTC flag asks for the long form, which needs 6 bytes.
    assert decode_timepoint(bytes.fromhex("3bdfcb80")) is None
    assert decode_timepoint(bytes.fromhex("3bdfcb8010")) is None


def test_decode_bearer():
    assert decode_bearer(bytes.fromhex("40e1c185c479")) == Bearer(0xE1, 0xC185, 0xC479, 0)
    # 32-bit SId (data service) and SCIdS 2.
    assert decode_bearer(bytes.fromhex("52e010bce0d110bc")) == Bearer(0xE0, 0x10BC, 0xE0D110BC, 2)
    assert decode_bearer(b"\x40\xe0") is None


def test_expand_text_replaces_tokens():
    tokens = {0x01: b"Nachrichten", 0x02: "für".encode()}
    assert expand_text(b"\x01 \x02 alle", tokens) == "Nachrichten für alle"


def test_annex_c2_programme_information():
    schedule = decode_spi_object(_mot(ANNEX_C2_PI, 1))

    assert isinstance(schedule, Schedule)
    assert schedule.service == Bearer(0xE1, 0xCE15, 0xC224, 0)
    assert (schedule.start, schedule.stop) == (_utc(2003, 12, 18, 17), _utc(2003, 12, 18, 18))
    (programme,) = schedule.programmes
    assert programme.names == Names(medium="PM")
    assert (programme.start, programme.duration) == (_utc(2003, 12, 18, 17), 3600)
    assert programme.short_id == 16442449


def test_annex_c1_service_information():
    information = decode_spi_object(_mot(ANNEX_C1_SI, 0))

    assert isinstance(information, ServiceInformation)
    assert (information.ecc, information.eid) == (0xE1, 0xC185)
    assert information.ensemble_names == Names("London 1", "London 1")
    (service,) = information.services
    assert service.names == Names("Capital", "Capital FM")
    assert service.bearers == (Bearer(0xE1, 0xC185, 0xC479, 0),)
    assert [(logo.content_name, logo.width, logo.height) for logo in service.logos] == [
        ("479S", None, None),
        ("479R", None, None),
        ("479S", 128, 128),
        ("479L", 320, 240),
    ]


def test_real_ndr_service_information():
    information = decode_spi_object(_mot(_ndr_si(), 0))

    assert (information.ecc, information.eid) == (0xE0, 0x1188)
    assert information.ensemble_names == Names("NDR NDS", "NDR NDS HAN")
    assert [s.names.medium for s in information.services] == [
        "NDR Blue",
        "NDR Kultur",
        "N-JOY",
        "NDR Schlager",
        "NDR 1 NDS HAN",
        "NDR 2 NDS",
        "NDR Info Spezial",
        "NDR Info",
    ]
    blue = information.services[0]
    assert blue.bearers == (Bearer(0xE0, 0x1188, 0xD354, 0),)
    assert [logo.content_name for logo in blue.logos] == [
        "e0_d354_32x32_0.png",
        "e0_d354_112x32_1.png",
        "e0_d354_128x128_2.png",
        "e0_d354_320x240_3.png",
    ]


def test_real_deutschlandfunk_nova_schedule_uses_tokens():
    decoded = decode_binary_object(DLF_NOVA_PI_BODY)
    assert decoded is not None
    _root, tokens = decoded
    assert tokens[0x01] == b"Nachrichten" and tokens[0x03] == b"Nova"

    schedule = decode_spi_object(
        _mot(DLF_NOVA_PI_BODY, 1, parameters={PARAM_SCOPE_ID: bytes.fromhex("40e010bcd230")})
    )

    assert schedule.service == Bearer(0xE0, 0x10BC, 0xD230, 0)
    assert (schedule.start, schedule.stop) == (_utc(2026, 10, 2, 22), _utc(2026, 10, 3, 22))
    first = schedule.programmes[0]
    assert first.names == Names(medium="Deutschlandfunk", long="Deutschlandfunk Nova")
    assert (first.start, first.duration) == (_utc(2026, 10, 2, 22), 8 * 3600)
    assert first.description.startswith("07:00-07:03 Nachrichten")
    assert schedule.programmes[-1].names.long == "Club der Republik"
    assert [p.start for p in schedule.programmes] == sorted(p.start for p in schedule.programmes)


def test_advanced_profile_objects_are_skipped():
    assert (
        decode_spi_object(_mot(ANNEX_C2_PI, 1, parameters={PARAM_PROFILE_SUBSET: b"\x02"})) is None
    )


def test_other_objects_are_no_spi():
    assert decode_spi_object(MotObject(1, 2, 3, "logo.png", b"\x89PNG", {})) is None
    assert decode_spi_object(_mot(ANNEX_C2_PI, 0)) is None  # PI data labelled as SI
    assert decode_spi_object(_mot(b"\x05\x00", 1)) is None


def test_profile_names():
    assert profile_names(b"\x01\x02") == ["Basic", "Advanced"]
    assert profile_names(b"\x01\x00\x07") == ["Basic"]  # 0x00 ends the list
    assert profile_names(b"\x03") == ["Profile 3"]
    assert profile_names(b"") == []


def test_programme_guide_names_services_from_si_and_labels():
    guide = ProgrammeGuide()
    _add(guide, _mot(ANNEX_C2_PI, 1, "pi"))  # schedule of SId C224, no SI for it
    _add(guide, _mot(_ndr_si(), 0, "si"))
    _add(guide, MotObject(2, 2, 3, "e0_d354_128x128_2.png", b"png data", {}))

    services = {s.sid: s for s in guide.services({0xC224: "Radio 4"})}

    assert services[0xC224].name == "Radio 4"
    assert services[0xC224].has_schedule
    assert services[0xD354].name == "NDR Blue"
    assert not services[0xD354].has_schedule
    assert services[0xD354].logo == b"png data"
    assert services[0xD384].logo is None
    assert len(guide.programmes(0xC224)) == 1


def test_programme_guide_merges_days_and_drops_outdated_objects():
    guide = ProgrammeGuide()
    _add(guide, _mot(ANNEX_C2_PI, 1, "day 1"))
    _add(guide, _mot(ANNEX_C2_PI, 1, "day 1 again under another name"))

    assert len(guide.programmes(0xC224)) == 1  # the same programme only once

    assert guide.retain({"day 1"}) is True
    assert guide.schedule_count == 1
    assert guide.retain(set()) is True
    assert guide.empty


def test_programme_guide_knows_the_ensembles_of_its_services():
    guide = ProgrammeGuide()
    _add(guide, _mot(ANNEX_C2_PI, 1, "pi"))  # bearer dab:e1.ce15.c224.0
    _add(guide, _mot(_ndr_si(), 0, "si"))

    services = {s.sid: s for s in guide.services({})}

    assert guide.ensemble_ids == {0x1188}
    assert services[0xD354].ensembles == {0x1188}
    assert services[0xC224].ensembles == {0xCE15}


def test_merge_programmes_keeps_each_programme_once():
    pi = decode_spi_object(_mot(ANNEX_C2_PI, 1))
    programme = pi.programmes[0]
    later = replace(programme, start=programme.start + 3600)

    assert merge_programmes([later, programme, programme]) == [programme, later]
