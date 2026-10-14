from datetime import UTC, datetime

import numpy as np
import pytest

from dab.fig import (
    TMID_PACKET_DATA,
    TMID_STREAM_AUDIO,
    Announcement,
    AnnouncementSupport,
    AudioComponent,
    ComponentReference,
    Ensemble,
    OeAnnouncement,
    OeAnnouncementSupport,
    PacketComponent,
    ServiceComponent,
    UserApplication,
    decode_label,
    parse_fib,
)
from tests.dab_signal import fib


def _build_fib_bits(fig_type: int, payload: bytes) -> np.ndarray:
    """Builds a 256-bit FIB (30 bytes of data + 16 CRC bits) from a single FIG, the
    rest padded with 0xFF. The CRC bits are left unset since `parse_fib` does not check
    them (that is the job of `check_fib_crc`)."""
    header = (fig_type << 5) | len(payload)
    data = (bytes([header]) + payload).ljust(30, b"\xff")
    fib_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return np.concatenate([fib_bits, np.zeros(16, dtype=np.uint8)])


def test_decode_label_handles_ascii_utf8_and_ucs2():
    assert decode_label(b"Test Radio      ", charset=0) == "Test Radio"
    assert decode_label("Café".encode().ljust(16, b"\x00"), charset=15) == "Café"
    assert decode_label("Radio".encode("utf-16-be").ljust(16, b"\x00"), charset=6) == "Radio"


def test_parse_fib_extracts_ensemble_label_from_fig1_ext0():
    label = b"Mein Ensemble   "  # 16 characters (EBU Latin/ASCII, charset 0)
    payload = bytes([0x00]) + (0x1234).to_bytes(2, "big") + label + b"\x00\x00"
    fib_bits = _build_fib_bits(fig_type=1, payload=payload)

    ensemble = Ensemble()
    parse_fib(fib_bits, ensemble)

    assert ensemble.eid == 0x1234
    assert ensemble.label == "Mein Ensemble"


def test_parse_fib_extracts_programme_service_label_from_fig1_ext1():
    label = b"Mein Sender     "
    payload = bytes([0x01]) + (0x5678).to_bytes(2, "big") + label + b"\x00\x00"
    fib_bits = _build_fib_bits(fig_type=1, payload=payload)

    ensemble = Ensemble()
    parse_fib(fib_bits, ensemble)

    assert ensemble.services == {0x5678: "Mein Sender"}
    assert ensemble.data_services == {}


def test_parse_fib_extracts_data_service_label_from_fig1_ext5():
    label = b"Mein Datendienst"
    payload = bytes([0x05]) + (0xABCD1234).to_bytes(4, "big") + label + b"\x00\x00"
    fib_bits = _build_fib_bits(fig_type=1, payload=payload)

    ensemble = Ensemble()
    parse_fib(fib_bits, ensemble)

    assert ensemble.data_services == {0xABCD1234: "Mein Datendienst"}
    assert ensemble.services == {}


def test_parse_fib_ignores_non_label_fig_types():
    payload = bytes(range(4))
    fib_bits = _build_fib_bits(fig_type=0, payload=payload)

    ensemble = Ensemble()
    parse_fib(fib_bits, ensemble)

    assert ensemble.label is None
    assert ensemble.services == {}


def test_parse_fib_extracts_eep_subchannel_from_fig0_ext1():
    # SubChId=5, StartAddress=100, EEP profile A, level 2 (raw value 1), size=88 CU.
    payload = bytes([0x01, 0x14, 0x64, 0x84, 0x58])
    fib_bits = _build_fib_bits(fig_type=0, payload=payload)

    ensemble = Ensemble()
    parse_fib(fib_bits, ensemble)

    subchannel = ensemble.subchannels[5]
    assert subchannel.start_address == 100
    assert subchannel.size == 88
    assert subchannel.is_uep is False
    assert subchannel.eep_option == 0
    assert subchannel.protection_index == 1
    assert subchannel.description() == "EEP-A Level 2"


def test_parse_fib_extracts_uep_subchannel_from_fig0_ext1():
    # SubChId=10, StartAddress=200, UEP table index 7.
    payload = bytes([0x01, 0x28, 0xC8, 0x07])
    fib_bits = _build_fib_bits(fig_type=0, payload=payload)

    ensemble = Ensemble()
    parse_fib(fib_bits, ensemble)

    subchannel = ensemble.subchannels[10]
    assert subchannel.start_address == 200
    assert subchannel.is_uep is True
    assert subchannel.protection_index == 7
    # Table index 7 is 48 kbit/s at protection level 3 in 35 CU (ETSI EN 300 401, 11.3.1).
    assert subchannel.size == 35
    assert subchannel.bitrate == 48
    assert subchannel.description() == "UEP Level 3 (table 7)"


def test_parse_fib_extracts_audio_component_from_fig0_ext2():
    # Programme service SId=0x5678, one audio component: SubChId=12, ASCTy=63 (DAB+).
    payload = bytes([0x02, 0x56, 0x78, 0x01, 0x3F, 0x32])
    fib_bits = _build_fib_bits(fig_type=0, payload=payload)

    ensemble = Ensemble()
    parse_fib(fib_bits, ensemble)

    component = ensemble.audio_components[0x5678]
    assert component.subchannel_id == 12
    assert component.is_dab_plus is True
    assert component.is_dab is False


def test_parse_fib_recognises_classic_dab_audio_component():
    # SId=0x5678, one audio component: SubChId=12, ASCTy=0 (DAB, MPEG Layer II).
    payload = bytes([0x02, 0x56, 0x78, 0x01, 0x00, 0x32])
    fib_bits = _build_fib_bits(fig_type=0, payload=payload)

    ensemble = Ensemble()
    parse_fib(fib_bits, ensemble)

    component = ensemble.audio_components[0x5678]
    assert component.subchannel_id == 12
    assert component.is_dab is True
    assert component.is_dab_plus is False


def test_parse_fib_reads_date_and_time_from_fig0_ext10():
    """FIG 0/10, the field as received from the Bundesmux (5C) on 19 September 2026."""
    payload = bytes([0x0A]) + bytes.fromhex("3bdd8b335544")
    ensemble = Ensemble()

    parse_fib(_build_fib_bits(fig_type=0, payload=payload), ensemble)
    first = ensemble.date_time
    parse_fib(_build_fib_bits(fig_type=0, payload=payload), ensemble)

    assert first is not None and first.has_seconds
    assert first.timestamp == datetime(2026, 9, 19, 12, 51, 21, 324000, tzinfo=UTC).timestamp()
    assert ensemble.date_time is not first, "every report must be recognisable as a new one"


@pytest.mark.parametrize(
    ("fig", "expected"),
    [
        # Dlf (5C): Information.
        ("0511d2100003", {0xD210: 3}),
        # Seven services, all Pop Music.
        (
            "1d111299000a121c000a1259000a12e9000a1258000a12c8000a122a000a",
            dict.fromkeys([0x1299, 0x121C, 0x1259, 0x12E9, 0x1258, 0x12C8, 0x122A], 10),
        ),
        # Current Affairs, Pop, Rock, Children's, Easy Listening.
        (
            "1d111d2b00021083000a1a8b000a1b85000b1b2e0012145a000a1a85000c",
            {
                0x1D2B: 2,
                0x1083: 10,
                0x1A8B: 10,
                0x1B85: 11,
                0x1B2E: 18,
                0x145A: 10,
                0x1A85: 12,
            },
        ),
    ],
    ids=["dlf", "seven-pop", "mixed"],
)
def test_parse_fib_reads_programme_types_from_fig0_ext17(fig, expected):
    """FIG 0/17 as received on air in September 2026, from the FIG header on."""
    ensemble = Ensemble()

    parse_fib(fib(bytes.fromhex(fig)), ensemble)

    assert ensemble.programme_types == expected


def test_fig0_ext17_skips_language_and_complementary_code_of_older_transmitters():
    """Before EN 300 401 V2.1.1, the L flag announced a language byte before the code
    and the CC flag a complementary code after it."""
    payload = bytes.fromhex("11" + "d001" + "30" + "08" + "0a" + "05" + "d002" + "00" + "0b")
    ensemble = Ensemble()

    parse_fib(_build_fib_bits(fig_type=0, payload=payload), ensemble)

    assert ensemble.programme_types == {0xD001: 10, 0xD002: 11}


def test_fig0_ext17_for_other_ensembles_is_ignored():
    payload = bytes.fromhex("51d2100003")  # OE flag set
    ensemble = Ensemble()

    parse_fib(_build_fib_bits(fig_type=0, payload=payload), ensemble)

    assert ensemble.programme_types == {}


# --- Data services: FIG 0/2 (packet mode), 0/3, 0/8, 0/13, 0/14 --------------------------
# Real FIGs from the FIB, received on 5C (DR Deutschland) and 7A (NDR NDS HAN).


def _parse(*figs_hex: str) -> Ensemble:
    ensemble = Ensemble()
    for fig in figs_hex:
        parse_fib(fib(bytes.fromhex(fig)), ensemble)
    return ensemble


def test_fig0_ext2_lists_packet_mode_components_of_data_services():
    """5C: EPG Deutschland (SCId 0x022) and GEPOS (SCId 0x020), each the primary and only
    component of a data service with a 32-bit SId."""
    ensemble = _parse("0f22e0d110bc01c08ae0d210bc01c082")

    assert ensemble.service_components == {
        0xE0D110BC: (ServiceComponent(TMID_PACKET_DATA, primary=True, ca=False, scid=0x022),),
        0xE0D210BC: (ServiceComponent(TMID_PACKET_DATA, primary=True, ca=False, scid=0x020),),
    }
    assert ensemble.audio_components == {}


def test_fig0_ext2_keeps_audio_components_as_well():
    ensemble = _parse("1502d210013f2ad220013f2ed230013f32d240013f36")

    assert ensemble.service_components[0xD210] == (
        ServiceComponent(TMID_STREAM_AUDIO, primary=True, ca=False, type_code=63, subchannel_id=10),
    )
    assert ensemble.audio_components[0xD230] == AudioComponent(12, 63)


def test_fig0_ext2_plays_the_primary_audio_component():
    """A secondary audio component listed first must not replace the primary one."""
    secondary = (63 << 8) | (7 << 2) | 0b00
    primary = (63 << 8) | (3 << 2) | 0b10
    payload = bytes([0x02, 0x12, 0x34, 0x02]) + secondary.to_bytes(2, "big")
    payload += primary.to_bytes(2, "big")
    ensemble = _parse((bytes([len(payload)]) + payload).hex())

    assert ensemble.audio_components[0x1234] == AudioComponent(3, 63)
    assert len(ensemble.service_components[0x1234]) == 2


@pytest.mark.parametrize(
    ("fig", "expected"),
    [
        (
            "0b03020005800202203c7803",  # 5C
            {
                0x020: PacketComponent(0x020, 32, 2, dscty=5, data_groups=True),
                0x022: PacketComponent(0x022, 30, 3, dscty=60, data_groups=True),
            },
        ),
        (
            "0b03008005300100903c3401",  # 7A: ARD TPEG and NDR EPG
            {
                0x008: PacketComponent(0x008, 12, 1, dscty=5, data_groups=True),
                0x009: PacketComponent(0x009, 13, 1, dscty=60, data_groups=True),
            },
        ),
    ],
)
def test_fig0_ext3_resolves_packet_mode_components(fig, expected):
    assert _parse(fig).packet_components == expected


def test_fig0_ext3_reads_dg_flag_and_ca_org():
    # SCId 0x123 with CAOrg flag, DG flag 1 (no data groups), DSCTy 24, SubChId 5,
    # address 0x3FF, CAOrg 0xBEEF.
    payload = bytes([0x03, 0x12, 0x31, 0x80 | 24, (5 << 2) | 0x03, 0xFF, 0xBE, 0xEF])
    ensemble = _parse((bytes([len(payload)]) + payload).hex())

    assert ensemble.packet_components[0x123] == PacketComponent(
        0x123, 5, 0x3FF, dscty=24, data_groups=False, ca_org=0xBEEF
    )


def test_fig0_ext8_reads_long_and_short_form():
    long_form = _parse("0f28e0d110bc008022e0d210bc008020")  # data services, SCId
    short_form = _parse("1108d210000ad220000bd230000cd240000d")  # Dlf, SubChId

    assert long_form.component_references == {
        (0xE0D110BC, 0): ComponentReference(scid=0x022),
        (0xE0D210BC, 0): ComponentReference(scid=0x020),
    }
    assert short_form.component_references[(0xD210, 0)] == ComponentReference(subchannel_id=10)
    assert short_form.component_references[(0xD240, 0)] == ComponentReference(subchannel_id=13)


@pytest.mark.parametrize(
    ("fig", "key", "applications"),
    [
        ("0a2de0d110bc0100e20102", (0xE0D110BC, 0), (UserApplication(0x007, b""),)),
        ("082de0d020f0010080", (0xE0D020F0, 0), (UserApplication(0x004, b""),)),
        ("082de0d210bc01bb80", (0xE0D210BC, 0), (UserApplication(0x5DC, b""),)),
        ("092de0d031880100e101", (0xE0D03188, 0), (UserApplication(0x007, b""),)),
        (
            "170dd2100200420c3c8946040500000000d2200100420c3c",
            (0xD210, 0),
            (
                UserApplication(0x002, bytes.fromhex("0c3c")),
                UserApplication(0x44A, bytes.fromhex("040500000000")),
            ),
        ),
    ],
)
def test_fig0_ext13_reads_user_applications(fig, key, applications):
    assert _parse(fig).user_applications[key] == applications


def test_fig0_ext13_with_several_services_in_one_fig():
    ensemble = _parse("170dd2100200420c3c8946040500000000d2200100420c3c")

    assert ensemble.user_applications[(0xD220, 0)] == (
        UserApplication(0x002, bytes.fromhex("0c3c")),
    )


@pytest.mark.parametrize(
    ("fig", "expected"),
    [("030e7981", {30: 1, 32: 1}), ("020e31", {12: 1}), ("020e35", {13: 1})],
)
def test_fig0_ext14_reads_the_fec_scheme_per_subchannel(fig, expected):
    assert _parse(fig).fec_schemes == expected


# --- Frequency information: FIG 0/21 -------------------------------------------------
# On air on 5D (Antenne DE, EId 11F7), September 2026: two FI lists for its own
# ensemble, since a list holds at most two DAB frequencies.
FIG0_21_5D = "1515001211f706002bf60030ac11f7060031f60037ed"


def test_fig0_ext21_reads_the_frequencies_of_an_ensemble():
    ensemble = _parse(FIG0_21_5D)

    assert ensemble.frequencies == {0x11F7: (180_064, 199_360, 204_640, 229_072)}


def test_fig0_ext21_skips_other_lists_and_reads_every_block():
    """A block with an FM list (RDS PI D210, 2 frequencies of 1 byte) before a DAB list
    of another ensemble, then a second block."""
    first = "d210" + "82" + "5a" + "5b" + "1205" + "03" + "003637"  # 222.064 MHz = 11D
    second = "10bc" + "03" + "002b8b"  # 178.352 MHz = 5C
    payload = "15" + f"{len(first) // 2:04x}" + first + f"{len(second) // 2:04x}" + second
    ensemble = Ensemble()

    parse_fib(_build_fib_bits(fig_type=0, payload=bytes.fromhex(payload)), ensemble)

    assert ensemble.frequencies == {0x1205: (222_064,), 0x10BC: (178_352,)}


def test_fig0_ext21_adds_to_what_is_known():
    """A later FIG (made up) repeats 180.064 MHz and adds 222.064 MHz."""
    ensemble = _parse(FIG0_21_5D, "0c15000911f706002bf6003637")

    assert ensemble.frequencies[0x11F7] == (180_064, 199_360, 204_640, 222_064, 229_072)


def test_fig0_ext21_stops_at_a_list_running_past_its_block():
    ensemble = _parse("0915000611f70f002bf6")  # the list claims 7 bytes, 3 are there

    assert ensemble.frequencies == {}


# --- Ensembles carrying a service: FIG 0/24 ------------------------------------------
# On air in September 2026: 5D (EId 11F7) names itself for RTL RADIO (1229), OE flag
# clear; NDR 2 NDS (D882) with the OE flag set, so from an ensemble that does not carry
# it, in seven others, 7A (1188) among them.


@pytest.mark.parametrize(
    ("fig", "expected"),
    [
        ("061812290111f7", {0x1229: (0x11F7,)}),
        (
            "1258d88207115011571177118811a711fd1226",
            {0xD882: (0x1150, 0x1157, 0x1177, 0x1188, 0x11A7, 0x11FD, 0x1226)},
        ),
    ],
)
def test_fig0_ext24_reads_the_ensembles_carrying_a_service(fig, expected):
    assert _parse(fig).service_ensembles == expected


def test_fig0_ext24_reads_32_bit_sids_of_data_services():
    """P/D flag set: a data service in 10BC and 1188; CAId 1 is not evaluated."""
    ensemble = _parse("0a38e01d00201210bc1188")

    assert ensemble.service_ensembles == {0xE01D0020: (0x10BC, 0x1188)}


def test_fig0_ext24_replaces_an_entry_and_drops_it_on_a_change_event():
    """RTL RADIO first in 11F7 alone, then (made up) in 11F7 and 10BC; then the change
    event indication (no EIds)."""
    ensemble = _parse("061812290111f7", "081812290211f710bc")
    assert ensemble.service_ensembles == {0x1229: (0x11F7, 0x10BC)}

    parse_fib(fib(bytes.fromhex("0418122900")), ensemble)
    assert ensemble.service_ensembles == {}


def test_fig0_ext24_stops_at_eids_running_past_the_field():
    ensemble = _parse("0b18" + "12290111f7" + "d220021188")  # two EIds announced, one there

    assert ensemble.service_ensembles == {0x1229: (0x11F7,)}


# --- Announcements: FIG 0/0 (Al flag), FIG 0/18, FIG 0/19 ------------------------------
# FIG 0/0 and 0/18 as received in September 2026. No FIG 0/19 came in 150 s, since no
# announcement was running; those FIGs are built after ETSI EN 300 401, clause 8.1.6.2.


@pytest.mark.parametrize(
    ("fig", "expected"),
    [("050010bc2342", True), ("050011f70870", False)],
    ids=["5C", "5D"],
)
def test_fig0_ext0_reads_the_alarm_flag(fig, expected):
    assert _parse(fig).alarm_announcements is expected


@pytest.mark.parametrize(
    ("fig", "expected"),
    [
        (
            "1312d38500020104d48100020101d88200020102",
            {
                0xD385: AnnouncementSupport(0x0002, (4,)),
                0xD481: AnnouncementSupport(0x0002, (1,)),
                0xD882: AnnouncementSupport(0x0002, (2,)),
            },
        ),
        ("0712d38400020103", {0xD384: AnnouncementSupport(0x0002, (3,))}),
    ],
    ids=["8C", "7A-ndr-info"],
)
def test_fig0_ext18_reads_announcement_support(fig, expected):
    """Road traffic only (b1), every service in a cluster of its own."""
    assert _parse(fig).announcement_support == expected


def test_fig0_ext18_with_several_clusters():
    ensemble = _parse("0812d00100120201" + "07")  # traffic and news, clusters 1 and 7

    assert ensemble.announcement_support == {0xD001: AnnouncementSupport(0x0012, (1, 7))}


def test_fig0_ext18_stops_at_clusters_running_past_the_field():
    ensemble = _parse("0712d0010002020a")  # two clusters announced, one there

    assert ensemble.announcement_support == {}


def test_fig0_ext19_reports_an_announcement_until_it_ends():
    """Cluster 4, road traffic, newly introduced, in subchannel 7; later ongoing, and
    finally ASw flags of zero."""
    ensemble = _parse("051304000287")
    assert ensemble.announcements == {
        4: Announcement(cluster_id=4, flags=0x0002, new=True, subchannel_id=7)
    }

    parse_fib(fib(bytes.fromhex("051304000207")), ensemble)
    assert ensemble.announcements[4].new is False

    parse_fib(fib(bytes.fromhex("051304000007")), ensemble)
    assert ensemble.announcements == {}


def test_fig0_ext19_every_report_is_a_new_object():
    """How `dab.announcement.AnnouncementMonitor` tells that it was confirmed."""
    ensemble = _parse("051304000207")
    first = ensemble.announcements[4]

    parse_fib(fib(bytes.fromhex("051304000207")), ensemble)

    assert ensemble.announcements[4] == first
    assert ensemble.announcements[4] is not first


def test_fig0_ext19_skips_the_region_of_older_transmitters():
    """Before V2.1.1 a set Region flag appended a byte with the region; the next entry
    follows it. Here the alarm cluster (news and alarm) and cluster 2."""
    ensemble = _parse("0a13" + "ff0011c5" + "2a" + "02000403")

    assert ensemble.announcements == {
        0xFF: Announcement(cluster_id=0xFF, flags=0x0011, new=True, subchannel_id=5),
        2: Announcement(cluster_id=2, flags=0x0004, new=False, subchannel_id=3),
    }


# --- Announcements of other ensembles: FIG 0/25, FIG 0/26 ------------------------------
# None of 5C, 5D, 7A and 8C sent either in 10 min on 06.10.2026; the FIGs are built after
# ETSI EN 300 401 V2.1.1, clauses 8.1.6.3 and 8.1.6.4.


def test_fig0_ext25_reads_oe_announcement_support():
    """Dlf Kultur (D220) takes road traffic and news from OE cluster 5, served by
    ensemble 1188, and from OE cluster 6, served by 1215 (number of targets - 1 = 1)."""
    ensemble = _parse("0c19d22000120105118806" + "1215")

    assert ensemble.oe_announcement_support == {
        0xD220: OeAnnouncementSupport(0x0012, ((5, 0x1188), (6, 0x1215)))
    }


def test_fig0_ext25_stops_at_targets_running_past_the_field():
    ensemble = _parse("0a19d22000020105118806")  # two targets announced, one there

    assert ensemble.oe_announcement_support == {}


def test_fig0_ext26_reports_an_announcement_of_another_ensemble_until_it_ends():
    """OE cluster 5, road traffic, newly introduced, carried by NDR 2 (D382) in its
    primary component; later ongoing, and finally ASw flags of zero."""
    ensemble = _parse("071a05000280d382")
    assert ensemble.oe_announcements == {
        (5, 0xD382): OeAnnouncement(
            cluster_id=5, flags=0x0002, new=True, scids=0, target_sid=0xD382
        )
    }

    parse_fib(fib(bytes.fromhex("071a05000200d382")), ensemble)
    assert ensemble.oe_announcements[(5, 0xD382)].new is False

    parse_fib(fib(bytes.fromhex("071a05000000d382")), ensemble)
    assert ensemble.oe_announcements == {}


def test_fig0_ext26_names_the_ensemble_of_an_alarm_announcement():
    """An alarm (OE cluster 0xFF) carries the EId 1188 in place of the SId; beside it
    news in OE cluster 6 from a secondary component (SCIdS 2) of D581."""
    ensemble = _parse("0d1a" + "ff0001801188" + "06001002d581")

    assert ensemble.oe_announcements == {
        (0xFF, 0x1188): OeAnnouncement(0xFF, 0x0001, new=True, scids=0, target_sid=0x1188),
        (6, 0xD581): OeAnnouncement(6, 0x0010, new=False, scids=2, target_sid=0xD581),
    }
