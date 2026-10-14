"""Data services and user applications of an ensemble, put together from several FIGs
(ETSI EN 300 401, clause 6.3.7).

No single FIG says where a data application can be found:

* FIG 0/2 lists the components of each service; a packet mode component only by its
  service component identifier (SCId),
* FIG 0/3 resolves the SCId to subchannel, packet address and data service component
  type (DSCTy),
* FIG 0/14 says whether the subchannel carries the packet FEC,
* FIG 0/13 names the user application (programme guide, TPEG, slideshow, Journaline, ...),
  addressed by SId and the component identifier within the service (SCIdS), which is 0
  for the primary component and otherwise follows from FIG 0/8.

For an application in the X-PAD of an audio component, FIG 0/13 additionally states the
X-PAD application type it uses; for Journaline that is the only way to find it.
"""

from __future__ import annotations

from dataclasses import dataclass

from dab.fig import (
    TMID_PACKET_DATA,
    TMID_STREAM_AUDIO,
    TMID_STREAM_DATA,
    Ensemble,
    PacketComponent,
    ServiceComponent,
    Subchannel,
    UserApplication,
)
from i18n import tr, tr_noop

# User application types (ETSI TS 101 756, table 16).
USER_APPLICATION_SLIDESHOW = 0x002
USER_APPLICATION_TPEG = 0x004
USER_APPLICATION_SPI = 0x007
USER_APPLICATION_JOURNALINE = 0x44A
# Not registered in TS 101 756, but what GEPOS (GNSS data of the BKG, RTCM 3 in a TDC)
# announces on channel 5C; see `dab.rtcm`.
USER_APPLICATION_GEPOS = 0x5DC

_USER_APPLICATION_NAMES = {
    # Translators: user applications of DAB (ETSI TS 101 756, table 16).
    USER_APPLICATION_SLIDESHOW: tr_noop("Slideshow"),
    USER_APPLICATION_TPEG: tr_noop("TPEG (traffic and travel)"),
    USER_APPLICATION_SPI: tr_noop("SPI (programme guide)"),
    0x009: tr_noop("DMB (video)"),
    0x00D: "Filecasting",
    0x00E: "FIS",
    USER_APPLICATION_JOURNALINE: "Journaline",
    USER_APPLICATION_GEPOS: "GEPOS (GNSS, RTCM 3)",
}

# Data service component types (ETSI TS 101 756, table 2b).
DSCTY_TDC = 5
DSCTY_MPEG_TS = 24
DSCTY_MOT = 60
DSCTY_PROPRIETARY = 61

_DSCTY_NAMES = {
    DSCTY_TDC: tr_noop("TDC (transparent data channel)"),
    DSCTY_MPEG_TS: tr_noop("MPEG-2 transport stream"),
    DSCTY_MOT: "MOT",
    # Translators: a data service component type whose format only its maker knows.
    DSCTY_PROPRIETARY: tr_noop("proprietary"),
}


def user_application_name(app_type: int) -> str:
    """Name of a user application type for the GUI, in the language set."""
    name = _USER_APPLICATION_NAMES.get(app_type)
    if name is None:
        return tr("Application 0x{app_type}").format(app_type=f"{app_type:03X}")
    return tr(name)


def dscty_name(dscty: int) -> str:
    """Name of a data service component type for the GUI, in the language set."""
    name = _DSCTY_NAMES.get(dscty)
    return f"DSCTy {dscty}" if name is None else tr(name)


@dataclass(frozen=True)
class XPadApplication:
    """A user application in the X-PAD of an audio component (FIG 0/13 with X-PAD data,
    clause 6.3.6)."""

    app_type: int
    # The lowest X-PAD application type the application uses: its data groups start in
    # this type and continue in the next one.
    xpad_app_type: int
    data_groups: bool  # DG flag 0: carried in MSC data groups
    dscty: int
    ca: bool
    data: bytes  # the application specific part of the user application data

    @property
    def continuation_type(self) -> int:
        return self.xpad_app_type + 1


def xpad_application(application: UserApplication) -> XPadApplication | None:
    """Reads the X-PAD data at the start of the user application data: CA flag, CAOrg
    flag, Rfu and X-PAD application type (5 bits), then DG flag, Rfu and DSCTy (6 bits),
    with the CAOrg flag two more bytes of CAOrg.

    For Deutschlandfunk's Journaline: `04 05` = X-PAD application type 4, data groups,
    DSCTy 5; the four bytes `00 00 00 00` behind them belong to Journaline.
    """
    data = application.data
    if len(data) < 2:
        return None
    offset = 4 if data[0] & 0x40 else 2
    if len(data) < offset:
        return None
    return XPadApplication(
        app_type=application.app_type,
        xpad_app_type=data[0] & 0x1F,
        data_groups=not data[1] & 0x80,
        dscty=data[1] & 0x3F,
        ca=bool(data[0] & 0x80),
        data=data[offset:],
    )


def component_scids(ensemble: Ensemble, sid: int, component: ServiceComponent) -> int | None:
    """The SCIdS of a component: 0 for the primary one (clause 6.3.5), otherwise the
    one FIG 0/8 ties to its subchannel (stream) or SCId (packet); None while unknown."""
    if component.primary:
        return 0
    for (reference_sid, scids), reference in ensemble.component_references.items():
        if reference_sid != sid:
            continue
        if component.tmid == TMID_PACKET_DATA:
            if reference.scid is not None and reference.scid == component.scid:
                return scids
        elif reference.subchannel_id is not None and (
            reference.subchannel_id == component.subchannel_id
        ):
            return scids
    return None


def xpad_applications(ensemble: Ensemble, sid: int) -> tuple[XPadApplication, ...]:
    """The applications FIG 0/13 announces for the X-PAD of the service's audio
    component (the one `Ensemble.audio_components` plays); empty while the FIGs needed
    have not arrived."""
    audio = ensemble.audio_components.get(sid)
    if audio is None:
        return ()
    component = next(
        (
            c
            for c in ensemble.service_components.get(sid, ())
            if c.tmid == TMID_STREAM_AUDIO and c.subchannel_id == audio.subchannel_id
        ),
        None,
    )
    if component is None:
        return ()
    scids = component_scids(ensemble, sid, component)
    if scids is None:
        return ()
    applications = []
    for application in ensemble.user_applications.get((sid, scids), ()):
        parsed = xpad_application(application)
        if parsed is not None:
            applications.append(parsed)
    return tuple(applications)


@dataclass(frozen=True)
class DataComponent:
    """A data component of the ensemble (stream or packet mode) with everything the FIC
    says about it so far. Fields stay None while their FIG has not arrived."""

    sid: int
    label: str
    component: ServiceComponent
    scids: int | None
    packet: PacketComponent | None  # packet mode only, from FIG 0/3
    subchannel: Subchannel | None
    fec_scheme: int | None  # FIG 0/14 of the subchannel
    applications: tuple[UserApplication, ...]

    @property
    def is_packet_mode(self) -> bool:
        return self.component.tmid == TMID_PACKET_DATA

    @property
    def scid(self) -> int | None:
        """Identifies a packet mode component within the ensemble; what the data
        decoding is selected by."""
        return self.component.scid

    @property
    def subchannel_id(self) -> int | None:
        if self.is_packet_mode:
            return None if self.packet is None else self.packet.subchannel_id
        return self.component.subchannel_id

    @property
    def dscty(self) -> int | None:
        if self.is_packet_mode:
            return None if self.packet is None else self.packet.dscty
        return self.component.type_code

    @property
    def app_types(self) -> tuple[int, ...]:
        return tuple(application.app_type for application in self.applications)

    @property
    def decodable(self) -> bool:
        """Whether the packet decoding can start: packet mode, and FIG 0/3 and FIG 0/1
        of its subchannel have arrived. Components under conditional access are not."""
        return (
            self.is_packet_mode
            and self.packet is not None
            and self.subchannel is not None
            and not self.component.ca
        )


def data_components(ensemble: Ensemble) -> list[DataComponent]:
    """All stream and packet mode data components of the ensemble, by label."""
    result = []
    for sid, components in ensemble.service_components.items():
        label = ensemble.data_services.get(sid) or ensemble.services.get(sid) or ""
        for component in components:
            if component.tmid not in (TMID_STREAM_DATA, TMID_PACKET_DATA):
                continue
            packet = None
            if component.tmid == TMID_PACKET_DATA:
                packet = ensemble.packet_components.get(component.scid)
                subchannel_id = None if packet is None else packet.subchannel_id
            else:
                subchannel_id = component.subchannel_id
            scids = component_scids(ensemble, sid, component)
            applications = () if scids is None else ensemble.user_applications.get((sid, scids), ())
            result.append(
                DataComponent(
                    sid=sid,
                    label=label,
                    component=component,
                    scids=scids,
                    packet=packet,
                    subchannel=(
                        None if subchannel_id is None else ensemble.subchannels.get(subchannel_id)
                    ),
                    fec_scheme=(
                        None if subchannel_id is None else ensemble.fec_schemes.get(subchannel_id)
                    ),
                    applications=applications,
                )
            )
    return sorted(result, key=lambda c: (c.label.casefold(), c.sid))
