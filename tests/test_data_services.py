"""Tests for putting data services and user applications together from the FIGs."""

from __future__ import annotations

from dab.data_services import (
    DSCTY_MOT,
    DSCTY_TDC,
    USER_APPLICATION_JOURNALINE,
    USER_APPLICATION_SLIDESHOW,
    USER_APPLICATION_SPI,
    XPadApplication,
    component_scids,
    data_components,
    dscty_name,
    user_application_name,
    xpad_application,
    xpad_applications,
)
from dab.fig import (
    TMID_PACKET_DATA,
    TMID_STREAM_DATA,
    ComponentReference,
    Ensemble,
    ServiceComponent,
    UserApplication,
    parse_fib,
)
from tests.dab_signal import fib

# The FIGs of 5C (DR Deutschland) that describe its two data services and the
# Deutschlandfunk, as received (from the FIG header byte on).
FIGS_5C = (
    "0f22e0d110bc01c08ae0d210bc01c082",  # FIG 0/2: EPG Deutschland, GEPOS
    "1502d210013f2ad220013f2ed230013f32d240013f36",  # FIG 0/2: Dlf & Co.
    "0b03020005800202203c7803",  # FIG 0/3
    "0f28e0d110bc008022e0d210bc008020",  # FIG 0/8, long form
    "1108d210000ad220000bd230000cd240000d",  # FIG 0/8, short form
    "0a2de0d110bc0100e20102",  # FIG 0/13: SPI, basic and advanced profile
    "082de0d210bc01bb80",  # FIG 0/13: GEPOS
    "170dd2100200420c3c8946040500000000d2200100420c3c",  # FIG 0/13: SLS, Journaline
    "030e7981",  # FIG 0/14
    "1d0181e8880679d6881259a08836556a88365140942a1d16942a18e08836",  # FIG 0/1
    "3705e0d110bc45504720446575747363686c616e6420f800",  # FIG 1/5 "EPG Deutschland"
    "3705e0d210bc4745504f532020202020202020202020f800",  # FIG 1/5 "GEPOS"
)


def _ensemble(figs=FIGS_5C) -> Ensemble:
    ensemble = Ensemble()
    for fig in figs:
        parse_fib(fib(bytes.fromhex(fig)), ensemble)
    return ensemble


def test_data_components_of_5c():
    epg, gepos = data_components(_ensemble())

    assert (epg.label, epg.sid, epg.scid, epg.scids) == ("EPG Deutschland", 0xE0D110BC, 0x22, 0)
    assert epg.is_packet_mode and epg.decodable
    assert (epg.subchannel_id, epg.packet.packet_address, epg.dscty) == (30, 3, DSCTY_MOT)
    assert (epg.subchannel.bitrate, epg.subchannel.size) == (24, 18)
    assert epg.fec_scheme == 1
    assert epg.app_types == (USER_APPLICATION_SPI,)
    assert epg.applications[0].data == b"\x01\x02"

    assert gepos.label == "GEPOS"
    assert (gepos.subchannel_id, gepos.packet.packet_address, gepos.dscty) == (32, 2, DSCTY_TDC)
    assert gepos.subchannel.bitrate == 8
    assert gepos.app_types == (0x5DC,)


def test_data_components_wait_for_the_figs_they_need():
    """Without FIG 0/3 and FIG 0/1 a component is listed but cannot be decoded yet."""
    epg = data_components(_ensemble(FIGS_5C[:1]))[0]

    assert epg.sid == 0xE0D110BC

    assert epg.packet is None
    assert epg.subchannel is None
    assert epg.applications == ()
    assert not epg.decodable


def test_xpad_applications_of_deutschlandfunk():
    applications = xpad_applications(_ensemble(), 0xD210)

    assert applications == (
        XPadApplication(USER_APPLICATION_SLIDESHOW, 12, True, DSCTY_MOT, False, b""),
        XPadApplication(USER_APPLICATION_JOURNALINE, 4, True, DSCTY_TDC, False, bytes(4)),
    )
    assert applications[1].continuation_type == 5


def test_xpad_applications_are_empty_while_unknown():
    assert xpad_applications(_ensemble(FIGS_5C[:1]), 0xD210) == ()
    assert xpad_applications(_ensemble(), 0xD240) == ()  # no FIG 0/13 for it


def test_xpad_application_skips_the_ca_org():
    # CA flag and CAOrg flag set, X-PAD type 6, DG flag 1, DSCTy 60, CAOrg, then data.
    application = UserApplication(0x44A, bytes([0xC6, 0x80 | 60, 0x12, 0x34, 0x00, 0x00]))

    parsed = xpad_application(application)

    assert (parsed.xpad_app_type, parsed.data_groups, parsed.dscty, parsed.ca) == (
        6,
        False,
        60,
        True,
    )
    assert parsed.data == b"\x00\x00"


def test_secondary_component_takes_its_scids_from_fig0_8():
    ensemble = Ensemble()
    secondary = ServiceComponent(TMID_PACKET_DATA, primary=False, ca=False, scid=0x345)
    stream = ServiceComponent(
        TMID_STREAM_DATA, primary=False, ca=False, type_code=5, subchannel_id=9
    )
    ensemble.component_references[(0x1234, 3)] = ComponentReference(scid=0x345)
    ensemble.component_references[(0x1234, 4)] = ComponentReference(subchannel_id=9)

    assert component_scids(ensemble, 0x1234, secondary) == 3
    assert component_scids(ensemble, 0x1234, stream) == 4
    assert component_scids(ensemble, 0x9999, secondary) is None


def test_names():
    assert user_application_name(USER_APPLICATION_SPI) == "SPI (programme guide)"
    assert user_application_name(0x5DC) == "GEPOS (GNSS, RTCM 3)"
    assert user_application_name(0x123) == "Application 0x123"
    assert dscty_name(DSCTY_MOT) == "MOT"
    assert dscty_name(42) == "DSCTy 42"
