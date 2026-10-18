"""Tests for the programme guides of all channels in one list (`dab.combined_guide`)."""

from __future__ import annotations

from dab.channels import channel_for_frequency
from dab.combined_guide import (
    CombinedGuide,
    EnsembleServices,
    GuideEntry,
    GuideGroup,
    channel_order,
    service_logos,
)
from dab.mot import MotObject
from dab.spi import PARAM_SCOPE_ID, ProgrammeGuide, decode_spi_object
from tests.spi_vectors import DLF_NOVA_PI_BODY

# The ensembles of EPG Deutschland (5C), Antenne DE (5D), NDR (7A) and one from North
# Rhine-Westphalia that is receivable only there.
EID_5C, EID_5D, EID_7A, EID_NRW = 0x10BC, 0x11F7, 0x1188, 0x1205


def _tlv(tag: int, value: bytes) -> bytes:
    assert len(value) < 0xFE
    return bytes((tag, len(value))) + value


def _service(name: str, sid: int, *eids: int) -> bytes:
    """A service element with its medium name and one bearer per ensemble."""
    body = _tlv(0x11, _tlv(0x01, name.encode()))
    for eid in eids:
        bearer = bytes((0x40, 0xE0)) + eid.to_bytes(2, "big") + sid.to_bytes(2, "big")
        body += _tlv(0x29, _tlv(0x80, bearer))
    return _tlv(0x28, body)


def _si(eid: int, *services: bytes) -> MotObject:
    """An SI object of the ensemble `eid`, like the SI of EPG Deutschland."""
    ensemble = _tlv(0x80, bytes((0xE0,)) + eid.to_bytes(2, "big")) + b"".join(services)
    return MotObject(1, 7, 0, f"si {eid:04X}", _tlv(0x03, _tlv(0x26, ensemble)), {})


# The schedule of Dlf Nova (D230) from 5C.
DLF_NOVA_PI = MotObject(
    2,
    7,
    1,
    "w20261003dd230c0.EHB",
    DLF_NOVA_PI_BODY,
    {PARAM_SCOPE_ID: bytes.fromhex("40e010bcd230")},
)


def _add(guide: ProgrammeGuide, mot_object: MotObject) -> None:
    guide.add(mot_object, decode_spi_object(mot_object))


def _layout(groups) -> dict:
    return {group.key: [f"{entry.sid:04X}" for entry in group.entries] for group in groups}


def _epg_deutschland() -> CombinedGuide:
    """The guide of 5C as broadcast: every service also has a bearer in 10BC."""
    combined = CombinedGuide()
    guide = combined.guide("5C")
    _add(
        guide,
        _si(
            EID_5C,
            _service("Dlf Nova", 0xD230, EID_5C),
            _service("Schwarzwaldradio", 0x100D, EID_5C),
            _service("80s80s", 0x1057, EID_5C, EID_5D),
            _service("NRW1", 0x149E, EID_5C, EID_NRW),
            _service("N-JOY", 0xD385, EID_5C, EID_7A),
        ),
    )
    _add(guide, DLF_NOVA_PI)
    return combined


def test_services_are_filed_under_the_channel_that_broadcasts_them():
    ensembles = {
        "5C": EnsembleServices("DR Deutschland", {0xD230: "Dlf Nova"}),
        "5D": EnsembleServices("Antenne DE", {0x1057: "80s80s"}),
    }

    groups = _epg_deutschland().groups(ensembles)

    assert _layout(groups) == {
        # Schwarzwaldradio is on no stored list and only has the guide's own bearer:
        # it stays with the channel of the guide.
        ("5C", None): ["D230", "100D"],
        ("5D", None): ["1057"],
        (None, EID_7A): ["D385"],
        (None, EID_NRW): ["149E"],
    }
    assert [group.label for group in groups] == ["DR Deutschland", "Antenne DE", None, None]
    assert all(group.origins == ("5C",) for group in groups)
    nova = groups[0].entries[0]
    assert (nova.name, nova.has_schedule) == ("Dlf Nova", True)
    assert not groups[0].entries[1].has_schedule
    # Only services on the channel's service list can be tuned to.
    listed = {(group.key, entry.sid): entry.listed for group in groups for entry in group.entries}
    assert listed == {
        (("5C", None), 0xD230): True,
        (("5C", None), 0x100D): False,
        (("5D", None), 0x1057): True,
        ((None, EID_7A), 0xD385): False,
        ((None, EID_NRW), 0x149E): False,
    }


def test_an_ensemble_with_a_guide_of_its_own_is_found_by_its_eid():
    combined = _epg_deutschland()
    _add(combined.guide("7A"), _si(EID_7A, _service("NDR Blue", 0xD354, EID_7A)))

    groups = combined.groups({})

    # N-JOY is on no stored list, but its bearer names the ensemble of the 7A guide.
    assert _layout(groups)[("7A", None)] == ["D385", "D354"]
    njoy = groups[1].entries[0]
    assert njoy.origins == ("5C",)


def test_a_service_known_to_several_guides_is_merged():
    combined = _epg_deutschland()
    # A guide on 5D that knows Dlf Nova under another name and carries the same schedule.
    _add(combined.guide("5D"), _si(EID_5D, _service("Deutschlandfunk Nova", 0xD230, EID_5D)))
    _add(combined.guide("5D"), DLF_NOVA_PI)
    ensembles = {
        "5C": EnsembleServices("DR Deutschland", {0xD230: "Dlf Nova"}),
        "5D": EnsembleServices("Antenne DE", {0xD230: "Dlf Nova", 0x1057: "80s80s"}),
    }

    groups = {group.key: group for group in combined.groups(ensembles)}

    on_5c = next(e for e in groups[("5C", None)].entries if e.sid == 0xD230)
    on_5d = next(e for e in groups[("5D", None)].entries if e.sid == 0xD230)
    # Each channel prefers what its own guide says.
    assert on_5c.name == "Dlf Nova"
    assert on_5d.name == "Deutschlandfunk Nova"
    assert on_5c.origins == ("5C", "5D")
    assert on_5d.origins == ("5D", "5C")
    assert combined.programmes(0xD230) == combined.guide("5C").programmes(0xD230)


def test_services_of_the_manual_frequency_have_a_group_of_their_own():
    combined = CombinedGuide()
    _add(combined.guide(None), _si(EID_5C, _service("Dlf Nova", 0xD230, EID_5C)))
    _add(combined.guide("7A"), _si(EID_7A, _service("NDR Blue", 0xD354, EID_7A)))

    groups = combined.groups({None: EnsembleServices("DR Deutschland", {0xD230: "Dlf Nova"})})

    assert [group.key for group in groups] == [("7A", None), (None, None)]
    assert groups[1].label == "DR Deutschland"


def test_a_guide_without_its_si_keeps_its_services():
    combined = CombinedGuide()
    _add(combined.guide("5C"), DLF_NOVA_PI)  # the SI has not come round yet

    groups = combined.groups({}, {0xD230: "Dlf Nova"})

    # The bearer of the schedule names 10BC, but without the SI nothing says that
    # this is the guide's own ensemble; it is not filed as an unknown one.
    assert _layout(groups) == {("5C", None): ["D230"]}
    # Labels name services that no guide names.
    assert groups[0].entries[0].name == "Dlf Nova"
    assert combined.groups({})[0].entries[0].name == "D230"


def test_channel_order():
    assert sorted(["12D", None, "5C", "7A", "X"], key=channel_order) == [
        "5C",
        "7A",
        "12D",
        "X",
        None,
    ]


def test_an_empty_combined_guide():
    combined = CombinedGuide()
    assert combined.empty
    combined.guide("5C")
    assert combined.empty
    assert combined.groups({"5C": EnsembleServices("DR Deutschland", {0xD230: "x"})}) == []
    _add(combined.guide("5C"), DLF_NOVA_PI)
    assert not combined.empty


def test_the_measured_eid_of_a_channel_locates_its_services():
    """N-JOY is on no stored service list, but 7A was received with EId 1188."""
    ensembles = {
        "5C": EnsembleServices("DR Deutschland", {0xD230: "Dlf Nova"}, eid=EID_5C),
        "7A": EnsembleServices("NDR NDS HAN", {0xD354: "NDR Blue"}, eid=EID_7A),
    }

    layout = _layout(_epg_deutschland().groups(ensembles))

    assert layout[("7A", None)] == ["D385"]
    assert (None, EID_7A) not in layout


def test_an_announced_frequency_describes_an_ensemble_but_does_not_locate_it():
    # 8C announces 1205 on 11D; 5D announces its own 11F7 on 8C as well, which here
    # carries another ensemble.
    ensembles = {
        "5D": EnsembleServices("Antenne DE", eid=EID_5D, frequencies={EID_5D: (180_064, 199_360)}),
        "8C": EnsembleServices(
            "NI HANNOVER K8C", eid=0x1215, frequencies={EID_NRW: (222_064, 1_452_960)}
        ),
    }

    groups = {group.key: group for group in _epg_deutschland().groups(ensembles)}

    assert [e.sid for e in groups[("5D", None)].entries] == [0x1057]  # by its EId
    assert ("8C", None) not in groups
    nrw = groups[(None, EID_NRW)]
    assert nrw.frequencies == (222_064, 1_452_960)
    assert nrw.announced_by == ("8C",)
    assert groups[(None, EID_7A)].frequencies == ()


def test_the_eid_of_its_channel_stands_in_for_the_si_of_a_guide():
    """Without the SI, the EId of the channel tells which bearers are the guide's own:
    a schedule of another ensemble then gets a group of its own."""
    combined = CombinedGuide()
    _add(combined.guide("5D"), DLF_NOVA_PI)  # the bearer names 10BC

    unknown = combined.groups({"5D": EnsembleServices("Antenne DE", eid=EID_5D)})
    own = combined.groups({"5D": EnsembleServices("Antenne DE", eid=EID_5C)})

    assert _layout(unknown) == {(None, EID_5C): ["D230"]}
    assert _layout(own) == {("5D", None): ["D230"]}


def test_service_logos_come_from_the_group_of_the_channel_first():
    """A service broadcast on 5C and 5D shows the logo its own channel's guide gives;
    the list of 5D also gets logos that only the guide from 5C sends."""
    group_5c = GuideGroup(
        "5C",
        None,
        "DR Deutschland",
        (
            GuideEntry(0x1057, "80s80s", b"logo from 5C", True, ("5C",)),
            GuideEntry(0xD230, "Dlf Nova", None, True, ("5C",)),
        ),
    )
    group_5d = GuideGroup(
        "5D",
        None,
        "Antenne DE",
        (
            GuideEntry(0x1057, "80s80s", b"logo from 5D", False, ("5D",)),
            GuideEntry(0x10D1, "90s90s", b"90s90s from 5C", False, ("5C",)),
        ),
    )
    groups = [group_5c, group_5d]

    assert service_logos(groups, "5C") == {0x1057: b"logo from 5C", 0x10D1: b"90s90s from 5C"}
    assert service_logos(groups, "5D") == {0x1057: b"logo from 5D", 0x10D1: b"90s90s from 5C"}
    assert service_logos([], "5C") == {}


def test_channel_for_frequency():
    assert channel_for_frequency(178_352) == "5C"
    assert channel_for_frequency(222_064) == "11D"
    assert channel_for_frequency(1_452_960) is None  # L band
