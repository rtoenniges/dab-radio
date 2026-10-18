"""Fast Information Groups (FIG): the multiplex configuration carried in the FIC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from dab.charset import decode_charset
from dab.timecode import BroadcastTime, parse_date_time
from i18n import tr

# ASCTy (audio service component type) of FIG 0/2, ETSI TS 101 756, table 2a.
ASCTY_DAB = 0  # MPEG-1/2 Audio Layer II ("MP2")
ASCTY_DAB_PLUS = 63  # HE-AAC v2 in superframes

# TMId (transport mechanism identifier) of FIG 0/2, ETSI EN 300 401, clause 6.3.1.
TMID_STREAM_AUDIO = 0
TMID_STREAM_DATA = 1
TMID_PACKET_DATA = 3

# FEC scheme of FIG 0/14 (clause 6.2.2).
FEC_SCHEME_NONE = 0
FEC_SCHEME_RS = 1  # RS(204,188) over the packets, clause 5.3.5

# Range and modulation of an FI list in FIG 0/21 (clause 8.1.8, table 21): the only one
# evaluated is a DAB ensemble; FM, DRM and AMSS lists are skipped.
RM_DAB_ENSEMBLE = 0b0000
# Unit of the frequency of a DAB ensemble in FIG 0/21.
FI_DAB_FREQUENCY_STEP_KHZ = 16


class UepProfile(NamedTuple):
    """One row of the UEP table: what a FIG 0/1 short-form table index stands for."""

    size: int  # subchannel size in CU
    protection_level: int  # 1 (strongest) .. 5 (weakest)
    bitrate: int  # kbit/s


# ETSI EN 300 401, clause 11.3.1 (table 8): the 64 UEP profiles, addressed by the 6-bit
# table index of the FIG 0/1 short form. Cross-checked against welle.io
# (dab-constants.cpp, `ProtLevel`) and against the transmitter side, ODR-DabMod
# (SubchannelSource.cpp, `framesizeCu`).
# fmt: off
UEP_TABLE = tuple(UepProfile(*row) for row in (
    (16, 5, 32),   (21, 4, 32),   (24, 3, 32),   (29, 2, 32),   (35, 1, 32),
    (24, 5, 48),   (29, 4, 48),   (35, 3, 48),   (42, 2, 48),   (52, 1, 48),
    (29, 5, 56),   (35, 4, 56),   (42, 3, 56),   (52, 2, 56),
    (32, 5, 64),   (42, 4, 64),   (48, 3, 64),   (58, 2, 64),   (70, 1, 64),
    (40, 5, 80),   (52, 4, 80),   (58, 3, 80),   (70, 2, 80),   (84, 1, 80),
    (48, 5, 96),   (58, 4, 96),   (70, 3, 96),   (84, 2, 96),   (104, 1, 96),
    (58, 5, 112),  (70, 4, 112),  (84, 3, 112),  (104, 2, 112),
    (64, 5, 128),  (84, 4, 128),  (96, 3, 128),  (116, 2, 128), (140, 1, 128),
    (80, 5, 160),  (104, 4, 160), (116, 3, 160), (140, 2, 160), (168, 1, 160),
    (96, 5, 192),  (116, 4, 192), (140, 3, 192), (168, 2, 192), (208, 1, 192),
    (116, 5, 224), (140, 4, 224), (168, 3, 224), (208, 2, 224), (232, 1, 224),
    (128, 5, 256), (168, 4, 256), (192, 3, 256), (232, 2, 256), (280, 1, 256),
    (160, 5, 320), (208, 4, 320), (280, 2, 320),
    (192, 5, 384), (280, 3, 384), (416, 1, 384),
))
# fmt: on

# ETSI EN 300 401, clause 11.3.2: bit rate of an EEP subchannel from its size
# (CU) per profile and level, cross-checked against welle.io (dab-constants.cpp,
# `Subchannel::bitrate`). Keyed by the 0-based protection_index (0..3 instead of EEP
# levels 1..4).
_EEP_A_BITRATE_DIVISORS = {0: 12, 1: 8, 2: 6, 3: 4}
_EEP_B_BITRATE_DIVISORS = {0: 27, 1: 21, 2: 18, 3: 15}
_EEP_A_BITRATE_MULTIPLIER = 8
_EEP_B_BITRATE_MULTIPLIER = 32


@dataclass
class AudioComponent:
    """Audio service component from FIG 0/2 (ETSI EN 300 401, clause 6.3.2)."""

    subchannel_id: int
    audio_service_component_type: int  # ASCTy: 0 = DAB/MPEG Layer 2, 63 = DAB+/AAC

    @property
    def is_dab_plus(self) -> bool:
        return self.audio_service_component_type == ASCTY_DAB_PLUS

    @property
    def is_dab(self) -> bool:
        """Classic DAB: MPEG Audio Layer II instead of AAC (ETSI EN 300 401, clause 7)."""
        return self.audio_service_component_type == ASCTY_DAB


@dataclass
class Subchannel:
    """Subchannel configuration from FIG 0/1 (ETSI EN 300 401, clause 6.3.1)."""

    subchannel_id: int
    start_address: int  # start within the CIF, in capacity units (CU)
    size: int  # size in CU (for UEP taken from `UEP_TABLE`)
    is_uep: bool
    protection_index: int  # UEP: table index; EEP: level (0-3, internal numbering)
    eep_option: int | None = None  # EEP only: 0 = profile A, 1 = profile B

    @property
    def uep_profile(self) -> UepProfile:
        """The UEP table row of this subchannel (UEP only)."""
        if not self.is_uep:
            raise ValueError("Only UEP subchannels have a UEP profile.")
        return UEP_TABLE[self.protection_index]

    @property
    def bitrate(self) -> int:
        """Bit rate of the carried data in kbit/s: from the table for UEP, from size,
        profile and level for EEP."""
        if self.is_uep:
            return self.uep_profile.bitrate
        if self.eep_option == 0:  # profile A
            divisor = _EEP_A_BITRATE_DIVISORS[self.protection_index]
            return self.size // divisor * _EEP_A_BITRATE_MULTIPLIER
        divisor = _EEP_B_BITRATE_DIVISORS[self.protection_index]  # profile B
        return self.size // divisor * _EEP_B_BITRATE_MULTIPLIER

    def description(self) -> str:
        """Short human-readable form of the protection profile, as shown in the GUI."""
        if self.is_uep:
            level = self.uep_profile.protection_level
            # Translators: a UEP protection level and its row in the UEP table of
            # EN 300 401 (table 8).
            return tr("UEP Level {level} (table {index})").format(
                level=level, index=self.protection_index
            )
        profile = "A" if self.eep_option == 0 else "B"
        return f"EEP-{profile} Level {self.protection_index + 1}"


@dataclass(frozen=True)
class ServiceComponent:
    """One component of a service from FIG 0/2 (ETSI EN 300 401, clause 6.3.1)."""

    tmid: int  # TMID_STREAM_AUDIO, TMID_STREAM_DATA or TMID_PACKET_DATA
    primary: bool
    ca: bool  # access control applies to the whole component
    # ASCTy for stream audio, DSCTy for stream data; packet data carries its DSCTy in
    # FIG 0/3 and has None here.
    type_code: int | None = None
    subchannel_id: int | None = None  # stream modes
    scid: int | None = None  # packet mode: service component identifier, see FIG 0/3


@dataclass(frozen=True)
class PacketComponent:
    """Where a packet mode component travels, from FIG 0/3 (clause 6.3.2)."""

    scid: int
    subchannel_id: int
    packet_address: int
    dscty: int  # data service component type, TS 101 756 table 2b
    data_groups: bool  # DG flag 0: the component is carried in MSC data groups
    ca_org: int | None = None


@dataclass(frozen=True)
class ComponentReference:
    """What FIG 0/8 (clause 6.3.5) ties a service component identifier within the
    service (SCIdS) to: a subchannel (short form) or an SCId (long form)."""

    subchannel_id: int | None = None
    scid: int | None = None


@dataclass(frozen=True)
class UserApplication:
    """One user application of a component from FIG 0/13 (clause 6.3.6).

    `data` is the field exactly as sent. For applications in the X-PAD of an audio
    component it starts with two bytes of X-PAD data (and possibly a CAOrg); whether
    that is the case only follows from the component, see `dab.data_services`.
    """

    app_type: int  # user application type, TS 101 756 table 16
    data: bytes


@dataclass(frozen=True)
class AnnouncementSupport:
    """Which announcements may interrupt a service, from FIG 0/18 (clause 8.1.6.1)."""

    flags: int  # ASu flags: bit i set = announcement type i, see `dab.announcement`
    clusters: tuple[int, ...]  # the announcement clusters the service belongs to


@dataclass(frozen=True)
class Announcement:
    """A running announcement from FIG 0/19 (clause 8.1.6.2)."""

    cluster_id: int  # the cluster it is meant for; 0xFF is the alarm cluster
    flags: int  # ASw flags: the announcement types it covers
    new: bool  # newly introduced (the first seconds) rather than ongoing
    subchannel_id: int  # the subchannel carrying its audio


@dataclass(frozen=True)
class OeAnnouncementSupport:
    """Which announcements carried in other ensembles may interrupt a service, from
    FIG 0/25 (clause 8.1.6.3)."""

    flags: int  # ASu flags, as in FIG 0/18
    # (OE Cluster Id, EId): the OE cluster in the tuned ensemble the service belongs to,
    # and the ensemble carrying its announcements; 1 to 4 of them.
    targets: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class OeAnnouncement:
    """An announcement running in another ensemble, from FIG 0/26 (clause 8.1.6.4)."""

    cluster_id: int  # the OE cluster in the tuned ensemble; 0xFF for alarm announcements
    flags: int  # ASw flags: the announcement types it covers
    new: bool  # newly introduced (the first seconds) rather than ongoing
    scids: int  # the component of the target service carrying it, 0 = primary
    # The service carrying it in the other ensemble; for an alarm announcement the EId
    # of the ensemble making it.
    target_sid: int


@dataclass
class Ensemble:
    """Collects the ensemble and service information that the FIC fields (FIG 0/0,
    FIG 0/1, FIG 0/2, FIG 0/3, FIG 0/8, FIG 0/10, FIG 0/13, FIG 0/14, FIG 0/17, FIG 0/18,
    FIG 0/19, FIG 0/21, FIG 0/24, FIG 0/25, FIG 0/26, FIG 1) deliver spread over many
    frames."""

    label: str | None = None
    eid: int | None = None  # from FIG 1/0, together with the label
    services: dict[int, str] = field(default_factory=dict)
    data_services: dict[int, str] = field(default_factory=dict)
    audio_components: dict[int, AudioComponent] = field(default_factory=dict)  # SId -> component
    subchannels: dict[int, Subchannel] = field(default_factory=dict)  # SubChId -> configuration
    # SId -> all its components in the order of FIG 0/2, audio included.
    service_components: dict[int, tuple[ServiceComponent, ...]] = field(default_factory=dict)
    # SCId -> transport of a packet mode component (FIG 0/3).
    packet_components: dict[int, PacketComponent] = field(default_factory=dict)
    # (SId, SCIdS) -> what the component is (FIG 0/8).
    component_references: dict[tuple[int, int], ComponentReference] = field(default_factory=dict)
    # (SId, SCIdS) -> the user applications of the component (FIG 0/13).
    user_applications: dict[tuple[int, int], tuple[UserApplication, ...]] = field(
        default_factory=dict
    )
    # SubChId -> FEC scheme of a packet mode subchannel (FIG 0/14).
    fec_schemes: dict[int, int] = field(default_factory=dict)
    # SId -> PTy code (FIG 0/17), see `dab.programme_type` for the names.
    programme_types: dict[int, int] = field(default_factory=dict)
    # Al flag of FIG 0/0: alarm announcements interrupt every service of the ensemble.
    alarm_announcements: bool = False
    # SId -> the announcements that may interrupt the service (FIG 0/18).
    announcement_support: dict[int, AnnouncementSupport] = field(default_factory=dict)
    # Cluster Id -> the announcement running for that cluster (FIG 0/19); removed when
    # FIG 0/19 reports its end. Every report is a new object, even when the value
    # repeats, so that a caller can tell how recently it was confirmed (see
    # `dab.announcement.AnnouncementMonitor`).
    announcements: dict[int, Announcement] = field(default_factory=dict)
    # SId -> the announcements of other ensembles that may interrupt the service (FIG 0/25).
    oe_announcement_support: dict[int, OeAnnouncementSupport] = field(default_factory=dict)
    # (OE Cluster Id, target SId) -> the announcement running in another ensemble
    # (FIG 0/26); removed when FIG 0/26 reports its end. A new object on every report,
    # as with `announcements`.
    oe_announcements: dict[tuple[int, int], OeAnnouncement] = field(default_factory=dict)
    # EId -> centre frequencies in kHz of DAB ensembles (FIG 0/21), this one and others,
    # sorted. Frequencies are only ever added: the lists are sent in parts.
    frequencies: dict[int, tuple[int, ...]] = field(default_factory=dict)
    # SId -> EIds of the ensembles carrying the service (FIG 0/24), for services of this
    # ensemble and of others, in the order sent. The tuned ensemble may be among them:
    # 5D names itself for its own services.
    service_ensembles: dict[int, tuple[int, ...]] = field(default_factory=dict)
    # Most recently received time of day (FIG 0/10). Every report is a new object, even
    # when the value repeats, so that a caller can tell that a new one has arrived, e.g.
    # to set the clock at the minute change of the short form.
    date_time: BroadcastTime | None = None

    def add_service(self, sid: int, label: str, *, is_data: bool) -> None:
        if label:
            (self.data_services if is_data else self.services)[sid] = label


def decode_label(raw: bytes, charset: int) -> str:
    """Turns 16 label bytes into text (ETSI EN 300 401, clause 5.2 / TS 101 756).

    See `dab.charset` for the character sets. Labels are padded to 16 bytes; the
    padding (spaces, and null bytes for UTF-8/UTF-16) is stripped.
    """
    return decode_charset(raw, charset).rstrip("\x00 ")


def parse_fib(fib_bits: np.ndarray, ensemble: Ensemble) -> None:
    """Evaluates the FIG fields of a FIB (already CRC-checked) and records what it finds
    in `ensemble`.

    `fib_bits` is a 256-bit array as delivered by `dsp.fic.split_fibs` (240 data bits
    + 16 CRC bits); only the 30 data bytes are evaluated (ETSI EN 300 401, clause 5.2.1).
    """
    if len(fib_bits) != 256:
        raise ValueError("A FIB must be 256 bits long.")
    data = np.packbits(fib_bits[:240])

    ptr = 0
    while ptr < 30:
        header = int(data[ptr])
        if header == 0xFF:  # end marker / padding
            break
        fig_type = header >> 5
        fig_len = header & 0x1F
        if fig_len == 0 or ptr + 1 + fig_len > 30:
            break
        payload = data[ptr + 1 : ptr + 1 + fig_len]
        ptr += 1 + fig_len

        if fig_type == 0:
            _parse_fig0(payload, ensemble)
        elif fig_type == 1:
            _parse_fig1(payload, ensemble)


def _parse_fig0(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0 (MCI/SI): evaluates the extensions needed for MSC decoding (0/1 subchannel
    organisation, 0/2 service organisation, 0/3 packet mode components, 0/14 packet
    FEC), the links to the user applications (0/8 and 0/13), the time of day (0/10),
    against which the slideshow evaluates its time parameters, the programme types
    (0/17), the announcements (0/18, 0/19, with the Al flag of 0/0, and from other
    ensembles 0/25, 0/26), the frequencies of DAB ensembles (0/21) and the ensembles
    carrying a service (0/24)."""
    if payload.size < 1:
        return
    extension = int(payload[0]) & 0x1F
    if extension == 0:
        _parse_fig0_ext0(payload, ensemble)
    elif extension == 1:
        _parse_fig0_ext1(payload, ensemble)
    elif extension == 2:
        _parse_fig0_ext2(payload, ensemble)
    elif extension == 3:
        _parse_fig0_ext3(payload, ensemble)
    elif extension == 8:
        _parse_fig0_ext8(payload, ensemble)
    elif extension == 13:
        _parse_fig0_ext13(payload, ensemble)
    elif extension == 14:
        _parse_fig0_ext14(payload, ensemble)
    elif extension == 10:
        date_time = parse_date_time(payload[1:].tobytes())
        if date_time is not None:
            ensemble.date_time = date_time
    elif extension == 17:
        _parse_fig0_ext17(payload, ensemble)
    elif extension == 18:
        _parse_fig0_ext18(payload, ensemble)
    elif extension == 19:
        _parse_fig0_ext19(payload, ensemble)
    elif extension == 21:
        _parse_fig0_ext21(payload, ensemble)
    elif extension == 24:
        _parse_fig0_ext24(payload, ensemble)
    elif extension == 25:
        _parse_fig0_ext25(payload, ensemble)
    elif extension == 26:
        _parse_fig0_ext26(payload, ensemble)


def _parse_fig0_ext0(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/0 (ensemble information, ETSI EN 300 401 clause 6.4.1): only the Al flag
    is evaluated, which enables alarm announcements for all services.

    Field: EId (16 bits), change flags (2), Al flag (1), CIF count (5 + 8), with change
    flags an occurrence change byte.
    """
    if payload.size >= 4:
        ensemble.alarm_announcements = bool(payload[3] & 0x20)


def _parse_fig0_ext1(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/1 (basic sub-channel organisation, ETSI EN 300 401 clause 6.3.1): start
    address, size (in CU) and protection profile of each subchannel.

    Cross-checked against welle.io (fib-processor.cpp, FIG0Extension1 /
    HandleFIG0Extension1), in particular the meaning of the "short/long form" bit, which
    runs against the naive expectation: 0 = UEP (short form), 1 = EEP (long form).
    """
    entries = payload[1:]  # payload[0] is the FIG 0 header (C/N, OE, P/D, extension)
    p = 0
    while p + 3 <= entries.size:
        header16 = (int(entries[p]) << 8) | int(entries[p + 1])
        subch_id = (header16 >> 10) & 0x3F
        start_address = header16 & 0x3FF
        flag_byte = int(entries[p + 2])

        if (flag_byte >> 7) == 0:  # UEP, "short form"
            table_index = flag_byte & 0x3F
            ensemble.subchannels[subch_id] = Subchannel(
                subchannel_id=subch_id,
                start_address=start_address,
                size=UEP_TABLE[table_index].size,
                is_uep=True,
                protection_index=table_index,
            )
            p += 3
        else:  # EEP, "long form"
            if p + 4 > entries.size:
                break
            option = (flag_byte >> 4) & 0x07
            level = (flag_byte >> 2) & 0x03
            size = ((flag_byte & 0x03) << 8) | int(entries[p + 3])
            ensemble.subchannels[subch_id] = Subchannel(
                subchannel_id=subch_id,
                start_address=start_address,
                size=size,
                is_uep=False,
                protection_index=level,
                eep_option=option,
            )
            p += 4


def _parse_fig0_ext2(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/2 (service organisation, ETSI EN 300 401 clause 6.3.1): the components of
    each service. For audio the SubChId and the ASCTy (audio service component type; 0
    = DAB/MPEG, 63 = DAB+/AAC), for stream data SubChId and DSCTy, for packet data the
    SCId that FIG 0/3 resolves.

    A service's field always lists all its components (clause 6.3.1), so it replaces
    what was known about the service before. The audio component used for playback is
    the primary one, or the first if none is marked primary.

    Cross-checked against welle.io (fib-processor.cpp, FIG0Extension2 /
    HandleFIG0Extension2).
    """
    is_data_service = bool((int(payload[0]) >> 5) & 0x01)
    p = 1
    while p < payload.size:
        sid_length = 4 if is_data_service else 2
        if p + sid_length > payload.size:
            break
        sid = int.from_bytes(bytes(payload[p : p + sid_length]), "big")
        p += sid_length
        if p >= payload.size:
            break

        num_components = int(payload[p]) & 0x0F
        p += 1
        components: list[ServiceComponent] = []
        for _ in range(num_components):
            if p + 2 > payload.size:
                break
            component = (int(payload[p]) << 8) | int(payload[p + 1])
            p += 2
            tmid = (component >> 14) & 0x03
            primary = bool(component & 0x02)
            ca = bool(component & 0x01)
            if tmid in (TMID_STREAM_AUDIO, TMID_STREAM_DATA):
                components.append(
                    ServiceComponent(
                        tmid=tmid,
                        primary=primary,
                        ca=ca,
                        type_code=(component >> 8) & 0x3F,
                        subchannel_id=(component >> 2) & 0x3F,
                    )
                )
            elif tmid == TMID_PACKET_DATA:
                components.append(
                    ServiceComponent(
                        tmid=tmid, primary=primary, ca=ca, scid=(component >> 2) & 0xFFF
                    )
                )
        ensemble.service_components[sid] = tuple(components)

        audio = [c for c in components if c.tmid == TMID_STREAM_AUDIO]
        if audio:
            chosen = next((c for c in audio if c.primary), audio[0])
            ensemble.audio_components[sid] = AudioComponent(
                subchannel_id=chosen.subchannel_id, audio_service_component_type=chosen.type_code
            )


def _parse_fig0_ext3(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/3 (service component in packet mode, clause 6.3.2): subchannel, packet
    address and DSCTy of each packet mode component, identified by its SCId.

    Each entry is 5 bytes: SCId (12 bits), Rfa (3), CAOrg flag, DG flag, Rfu, DSCTy (6),
    SubChId (6), packet address (10); with the CAOrg flag 2 more bytes of CAOrg.
    """
    entries = payload[1:]
    p = 0
    while p + 5 <= entries.size:
        scid = (int(entries[p]) << 4) | (int(entries[p + 1]) >> 4)
        has_ca_org = bool(entries[p + 1] & 0x01)
        flags = int(entries[p + 2])
        address_field = (int(entries[p + 3]) << 8) | int(entries[p + 4])
        p += 5
        ca_org = None
        if has_ca_org:
            if p + 2 > entries.size:
                break
            ca_org = (int(entries[p]) << 8) | int(entries[p + 1])
            p += 2
        ensemble.packet_components[scid] = PacketComponent(
            scid=scid,
            subchannel_id=address_field >> 10,
            packet_address=address_field & 0x3FF,
            dscty=flags & 0x3F,
            data_groups=not flags & 0x80,
            ca_org=ca_org,
        )


def _parse_fig0_ext8(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/8 (service component global definition, clause 6.3.5): links the SCIdS of
    a component, by which FIG 0/13 names it, to its subchannel (short form) or SCId
    (long form).

    Entry: SId (16 or 32 bits by the P/D flag), a byte with the Ext flag and the SCIdS,
    then one byte (L/S = 0: SubChId) or two (L/S = 1: SCId), with the Ext flag one more
    reserved byte.
    """
    sid_length = 4 if (int(payload[0]) >> 5) & 0x01 else 2
    entries = payload[1:]
    p = 0
    while p + sid_length + 2 <= entries.size:
        sid = int.from_bytes(bytes(entries[p : p + sid_length]), "big")
        p += sid_length
        has_extension = bool(entries[p] & 0x80)
        scids = int(entries[p]) & 0x0F
        p += 1
        if entries[p] & 0x80:  # long form
            if p + 2 > entries.size:
                break
            reference = ComponentReference(
                scid=((int(entries[p]) & 0x0F) << 8) | int(entries[p + 1])
            )
            p += 2
        else:
            reference = ComponentReference(subchannel_id=int(entries[p]) & 0x3F)
            p += 1
        if has_extension:
            p += 1
        ensemble.component_references[(sid, scids)] = reference


def _parse_fig0_ext13(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/13 (user application information, clause 6.3.6): which user applications
    (slideshow, SPI, TPEG, Journaline, ...) a component carries.

    Entry: SId (16 or 32 bits by the P/D flag), SCIdS (4 bits), number of applications
    (4 bits), then per application its type (11 bits), the data length (5 bits) and the
    data.
    """
    sid_length = 4 if (int(payload[0]) >> 5) & 0x01 else 2
    entries = payload[1:]
    p = 0
    while p + sid_length + 1 <= entries.size:
        sid = int.from_bytes(bytes(entries[p : p + sid_length]), "big")
        p += sid_length
        scids = int(entries[p]) >> 4
        count = int(entries[p]) & 0x0F
        p += 1
        applications = []
        for _ in range(count):
            if p + 2 > entries.size:
                return
            word = (int(entries[p]) << 8) | int(entries[p + 1])
            length = word & 0x1F
            p += 2
            if p + length > entries.size:
                return
            applications.append(UserApplication(word >> 5, bytes(entries[p : p + length])))
            p += length
        ensemble.user_applications[(sid, scids)] = tuple(applications)


def _parse_fig0_ext14(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/14 (FEC sub-channel organisation, clause 6.2.2): one byte per packet mode
    subchannel, SubChId (6 bits) and FEC scheme (2 bits)."""
    for byte in payload[1:]:
        ensemble.fec_schemes[int(byte) >> 2] = int(byte) & 0x03


def _parse_fig0_ext17(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/17 (programme type, ETSI EN 300 401 clause 8.1.5): one PTy code per service.

    Each entry is SId (16 bits), a flag byte and a byte with the code in its lower 5
    bits. The S/D flag (whether the code follows the individual programme items) is not
    evaluated. Before V2.1.1 the standard defined two further flags in the flag byte:
    L announced a language byte before the code, CC a complementary code after it. Both
    are "reserved" today; like welle.io, their bytes are skipped should an older
    transmitter still send them.

    The OE flag of the header marked entries for services of other ensembles in those
    older versions and is reserved today, so such FIGs are ignored.
    """
    if int(payload[0]) & 0x40:  # OE
        return
    entries = payload[1:]
    p = 0
    while p + 4 <= entries.size:
        sid = (int(entries[p]) << 8) | int(entries[p + 1])
        flags = int(entries[p + 2])
        p += 3
        if flags & 0x20:  # L: language byte
            p += 1
        if p >= entries.size:
            break
        ensemble.programme_types[sid] = int(entries[p]) & 0x1F
        p += 1
        if flags & 0x10:  # CC: complementary code byte
            p += 1


def _parse_fig0_ext18(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/18 (announcement support, ETSI EN 300 401 clause 8.1.6.1): which
    announcement types may interrupt a service, and its announcement clusters.

    Entry: SId (16 bits), ASu flags (16), Rfa (5), number of clusters (3), one byte per
    Cluster Id. Before V2.1.1 the number had 5 bits and the Rfa 3; reading 5 bits covers
    both, since the Rfa bits have to be zero.
    """
    entries = payload[1:]
    p = 0
    while p + 5 <= entries.size:
        sid = (int(entries[p]) << 8) | int(entries[p + 1])
        flags = (int(entries[p + 2]) << 8) | int(entries[p + 3])
        count = int(entries[p + 4]) & 0x1F
        p += 5
        if p + count > entries.size:
            break
        clusters = tuple(int(cluster) for cluster in entries[p : p + count])
        p += count
        ensemble.announcement_support[sid] = AnnouncementSupport(flags, clusters)


def _parse_fig0_ext19(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/19 (announcement switching, ETSI EN 300 401 clause 8.1.6.2): a running
    announcement, for which cluster, of which types and in which subchannel.

    Entry: Cluster Id (8 bits), ASw flags (16), New flag (1), Rfa (1), SubChId (6).
    ASw flags of zero end the announcement. Before V2.1.1 the Rfa bit was the Region
    flag, which appended a byte with the region the announcement was meant for; like
    welle.io, that byte is skipped and the region ignored should an older transmitter
    still send it.
    """
    entries = payload[1:]
    p = 0
    while p + 4 <= entries.size:
        cluster_id = int(entries[p])
        flags = (int(entries[p + 1]) << 8) | int(entries[p + 2])
        byte = int(entries[p + 3])
        p += 5 if byte & 0x40 else 4  # Region flag
        if flags:
            ensemble.announcements[cluster_id] = Announcement(
                cluster_id=cluster_id,
                flags=flags,
                new=bool(byte & 0x80),
                subchannel_id=byte & 0x3F,
            )
        else:
            ensemble.announcements.pop(cluster_id, None)


def _parse_fig0_ext21(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/21 (frequency information, ETSI EN 300 401 clause 8.1.8): on which
    frequencies an ensemble, this one or another, is broadcast.

    The field holds one or more blocks of Rfa (11 bits), the length of the FI lists in
    bytes (5 bits) and the FI lists. An FI list is an Id (the EId for a DAB ensemble,
    else e.g. an RDS PI code), R&M (4 bits), the continuity flag (1 bit), the length of
    the frequency list in bytes (3 bits) and the frequency list. For a DAB ensemble each
    entry is 3 bytes: a control field (5 bits) and the frequency in units of 16 kHz
    (19 bits). With at most 7 bytes a list holds two frequencies, so an ensemble with
    more comes in several lists. Lists for FM, DRM and AMSS are skipped by their
    length. The OE flag in the header does not matter: every list names its ensemble.

    On air (5D): `11f7 06 002bf6 0030ac`, EId 11F7 on 180.064 and 199.360 MHz.
    """
    data = payload[1:]
    block = 0
    while block + 2 <= data.size:
        length = int(data[block + 1]) & 0x1F
        end = min(block + 2 + length, data.size)
        p = block + 2
        while p + 3 <= end:
            identifier = (int(data[p]) << 8) | int(data[p + 1])
            flags = int(data[p + 2])
            rm = flags >> 4
            list_end = p + 3 + (flags & 0x07)
            if list_end > end:
                break
            if rm == RM_DAB_ENSEMBLE:
                found = {
                    ((int(data[q]) & 0x07) << 16 | int(data[q + 1]) << 8 | int(data[q + 2]))
                    * FI_DAB_FREQUENCY_STEP_KHZ
                    for q in range(p + 3, list_end - 2, 3)
                }
                known = ensemble.frequencies.get(identifier, ())
                if not found <= set(known):
                    ensemble.frequencies[identifier] = tuple(sorted(found | set(known)))
            p = list_end
        block += 2 + length


def _parse_fig0_ext24(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/24 (OE services, ETSI EN 300 401 clause 8.1.10): in which ensembles a
    service is carried.

    Entry: SId (16 or 32 bits by the P/D flag), Rfa (1), CAId (3), number of EIds (4),
    and the EIds (16 bits each). The OE flag tells whether the service is carried in the
    tuned ensemble; that also follows from FIG 0/2, so it is not kept. Each entry is
    complete and replaces what was known for the service: with at most 12 EIds (11 for
    a 32-bit SId) it always fits into one FIG, so the C/N flag does not matter. The CAId
    is not evaluated, as a receiver without access control must not. A number of zero
    is the change event indication: the entry is about to change or is withdrawn, so it
    is dropped until it comes again.

    On air (5D): `1229 01 11f7`, RTL RADIO in ensemble 11F7, which is 5D itself.
    """
    sid_length = 4 if (int(payload[0]) >> 5) & 0x01 else 2
    entries = payload[1:]
    p = 0
    while p + sid_length + 1 <= entries.size:
        sid = int.from_bytes(bytes(entries[p : p + sid_length]), "big")
        count = int(entries[p + sid_length]) & 0x0F
        p += sid_length + 1
        if p + 2 * count > entries.size:
            break
        if count:
            ensemble.service_ensembles[sid] = tuple(
                (int(entries[q]) << 8) | int(entries[q + 1]) for q in range(p, p + 2 * count, 2)
            )
        else:
            ensemble.service_ensembles.pop(sid, None)
        p += 2 * count


def _parse_fig0_ext25(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/25 (OE announcement support, ETSI EN 300 401 clause 8.1.6.3): which
    announcement types carried in other ensembles may interrupt a service of this one.

    Entry: SId (16 bits), ASu flags (16), Rfa (6), number of targets - 1 (2), and per
    target the OE Cluster Id (8) and the EId (16) of the ensemble carrying the
    announcements. Read as V2.1.1 defines it; before, the entry held the number of EIds
    (4 bits) and the EIds alone, which cannot be told apart from it. Not seen on air.
    """
    entries = payload[1:]
    p = 0
    while p + 5 <= entries.size:
        sid = (int(entries[p]) << 8) | int(entries[p + 1])
        flags = (int(entries[p + 2]) << 8) | int(entries[p + 3])
        count = (int(entries[p + 4]) & 0x03) + 1
        p += 5
        if p + 3 * count > entries.size:
            break
        targets = tuple(
            (int(entries[q]), (int(entries[q + 1]) << 8) | int(entries[q + 2]))
            for q in range(p, p + 3 * count, 3)
        )
        p += 3 * count
        ensemble.oe_announcement_support[sid] = OeAnnouncementSupport(flags, targets)


def _parse_fig0_ext26(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 0/26 (OE announcement switching, ETSI EN 300 401 clause 8.1.6.4): an
    announcement running in another ensemble, for which OE cluster of this one, of
    which types, and the service there carrying it.

    Entry: OE Cluster Id (8 bits), ASw flags (16), New flag (1), Rfa (3), SCIdS (4),
    target SId (16). ASw flags of zero end the announcement. Read as V2.1.1 defines it;
    before, the entry named the cluster and region in both ensembles and the EId instead
    of the service. Not seen on air either.
    """
    entries = payload[1:]
    p = 0
    while p + 6 <= entries.size:
        cluster_id = int(entries[p])
        flags = (int(entries[p + 1]) << 8) | int(entries[p + 2])
        byte = int(entries[p + 3])
        target_sid = (int(entries[p + 4]) << 8) | int(entries[p + 5])
        p += 6
        key = (cluster_id, target_sid)
        if flags:
            ensemble.oe_announcements[key] = OeAnnouncement(
                cluster_id=cluster_id,
                flags=flags,
                new=bool(byte & 0x80),
                scids=byte & 0x0F,
                target_sid=target_sid,
            )
        else:
            ensemble.oe_announcements.pop(key, None)


def _parse_fig1(payload: np.ndarray, ensemble: Ensemble) -> None:
    """FIG 1 (labels): 16 characters of text plus 2 bytes of character flags each."""
    if payload.size < 1:
        return
    header = int(payload[0])
    charset = header >> 4
    extension = header & 0x07

    if extension == 0 and payload.size >= 21:  # ensemble label
        ensemble.eid = int.from_bytes(bytes(payload[1:3]), "big")
        label = decode_label(bytes(payload[3:19]), charset)
        if label:
            ensemble.label = label
    elif extension == 1 and payload.size >= 21:  # programme service label
        sid = int.from_bytes(bytes(payload[1:3]), "big")
        ensemble.add_service(sid, decode_label(bytes(payload[3:19]), charset), is_data=False)
    elif extension == 5 and payload.size >= 23:  # data service label
        sid = int.from_bytes(bytes(payload[1:5]), "big")
        ensemble.add_service(sid, decode_label(bytes(payload[5:21]), charset), is_data=True)
