"""Texts for the code tables of TPEG (TEC 3.2, TFP 1.0, OpenLR, SNI) and names for the
components of a message, for the display.

The tables follow the TPEG2 protobuf definitions of TISA (TEC_3_4, TFP_1_1, OLR_1_1),
whose codes TEC 3.2 shares. Most TEC causes come from the ALERT-C event list of RDS-TMC
(ISO 14819-2). The English keeps to the names of the TISA tables where they read well on
their own; the German translation to what German traffic reports say. The entries are
marked with `tr_noop` and translated by the functions below, which the surfaces call.
"""

from __future__ import annotations

from dab.tpeg import AID_EAW, AID_SNI, AID_TEC, AID_TFP
from dab.tpeg_messages import (
    COMPONENT_ADVICE,
    COMPONENT_DIRECT_CAUSE,
    COMPONENT_DIVERSION,
    COMPONENT_EVENT,
    COMPONENT_FLOW_MATRIX,
    COMPONENT_FLOW_VECTOR,
    COMPONENT_LINKED_CAUSE,
    COMPONENT_LOCATION,
    COMPONENT_MANAGEMENT,
    COMPONENT_MESSAGE,
    COMPONENT_SPEED_LIMIT,
    COMPONENT_VEHICLE_RESTRICTION,
    LOCATION_OPENLR,
    LOCATION_TMC,
    OPENLR_LINEAR,
    OPENLR_POINT_ALONG_LINE,
    OPENLR_SHAPE,
)
from i18n import tr, tr_context, tr_noop

_APPLICATIONS = {
    AID_SNI: tr_noop("SNI (service information)"),
    AID_TEC: tr_noop("TEC (traffic events)"),
    AID_TFP: tr_noop("TFP (traffic flow)"),
    AID_EAW: tr_noop("EAW (warnings)"),
}

# tec001: what happens to the traffic flow.
EFFECTS = {
    1: tr_noop("traffic situation unknown"),
    2: tr_noop("free traffic flow"),
    3: tr_noop("heavy traffic"),
    4: tr_noop("slow traffic"),
    5: tr_noop("queuing traffic"),
    6: tr_noop("stationary traffic"),
    # TISA: "no traffic flow".
    7: tr_noop("road closed"),
}

# tec002: the main cause.
CAUSES = {
    1: tr_noop("traffic congestion"),
    2: tr_noop("accident"),
    3: tr_noop("roadworks"),
    4: tr_noop("narrow lanes"),
    5: tr_noop("road impassable"),
    6: tr_noop("slippery road"),
    7: tr_noop("aquaplaning"),
    8: tr_noop("fire"),
    9: tr_noop("hazardous driving conditions"),
    10: tr_noop("objects on the road"),
    11: tr_noop("animals on the road"),
    12: tr_noop("people on the road"),
    13: tr_noop("broken-down vehicle"),
    14: tr_noop("wrong-way driver"),
    15: tr_noop("rescue and recovery work"),
    16: tr_noop("regulatory measure"),
    17: tr_noop("extreme weather"),
    18: tr_noop("reduced visibility"),
    19: tr_noop("precipitation"),
    20: tr_noop("reckless persons"),
    21: tr_noop("over-height warning triggered"),
    22: tr_noop("traffic regulations changed"),
    23: tr_noop("major event"),
    24: tr_noop("service not operating"),
    25: tr_noop("facility not usable"),
    26: tr_noop("slow vehicles"),
    27: tr_noop("dangerous end of queue"),
    28: tr_noop("risk of fire"),
    29: tr_noop("waiting time"),
    30: tr_noop("police checkpoint"),
    31: tr_noop("malfunctioning roadside equipment"),
    100: tr_noop("test message"),
}

# tec101 to tec131: the sub-cause table is the one of the main cause (tec1nn for cause nn).
SUB_CAUSES = {
    1: {1: tr_noop("increased volume of traffic")},
    2: {
        1: tr_noop("multi-vehicle accident"),
        2: tr_noop("serious accident"),
        3: tr_noop("accident involving a lorry"),
        4: tr_noop("accident involving a bus"),
        5: tr_noop("accident involving hazardous materials"),
        6: tr_noop("accident on the opposite carriageway"),
        7: tr_noop("unsecured accident site"),
    },
    3: {
        1: tr_noop("major roadworks"),
        2: tr_noop("road marking work"),
        3: tr_noop("mobile roadworks"),
    },
    4: {
        1: tr_noop("contraflow"),
        2: tr_noop("hard shoulder closed"),
        3: tr_noop("turning lane closed"),
        4: tr_noop("crawler lane closed"),
    },
    5: {
        1: tr_noop("flooding"),
        2: tr_noop("danger of avalanches"),
        3: tr_noop("avalanche blasting"),
        4: tr_noop("landslide"),
        5: tr_noop("chemical spillage"),
        6: tr_noop("winter closure"),
    },
    6: {
        1: tr_noop("hoarfrost"),
        2: tr_noop("fuel on the road"),
        3: tr_noop("mud on the road"),
        4: tr_noop("snow on the road"),
        5: tr_noop("ice on the road"),
        6: tr_noop("black ice"),
        7: tr_noop("oil on the road"),
        8: tr_noop("loose chippings"),
        9: tr_noop("freezing wet roads"),
        10: tr_noop("roads gritted"),
    },
    8: {1: tr_noop("major fire"), 2: tr_noop("forest fire")},
    9: {
        1: tr_noop("rockfall"),
        2: tr_noop("earthquake damage"),
        3: tr_noop("sewer collapse"),
        4: tr_noop("subsidence"),
        5: tr_noop("snowdrifts"),
        6: tr_noop("storm damage"),
        7: tr_noop("burst pipe"),
        8: tr_noop("volcanic eruption"),
        9: tr_noop("falling ice"),
    },
    10: {
        1: tr_noop("shed load"),
        2: tr_noop("vehicle parts"),
        3: tr_noop("tyre debris"),
        4: tr_noop("large objects"),
        5: tr_noop("fallen trees"),
        6: tr_noop("hub caps"),
        7: tr_noop("stationary vehicle"),
    },
    11: {
        1: tr_noop("wild animals"),
        2: tr_noop("herd of animals"),
        3: tr_noop("small animals"),
        4: tr_noop("large animals"),
    },
    12: {1: tr_noop("children"), 2: tr_noop("cyclists"), 3: tr_noop("moped riders")},
    13: {
        1: tr_noop("burning broken-down vehicle"),
        2: tr_noop("unlit broken-down vehicle"),
    },
    15: {
        1: tr_noop("emergency vehicles"),
        2: tr_noop("rescue helicopter landing"),
        3: tr_noop("police operation"),
        4: tr_noop("medical emergency"),
        5: tr_noop("child abduction"),
    },
    16: {
        1: tr_noop("security alert"),
        2: tr_noop("contagious disease"),
        3: tr_noop("environmental protection measure"),
        4: tr_noop("smog alert"),
        5: tr_noop("batch service"),
        6: tr_noop("closed by the authorities"),
    },
    17: {
        1: tr_noop("gale"),
        2: tr_noop("damaging hail"),
        3: tr_noop("hurricane"),
        4: tr_noop("thunderstorm"),
        5: tr_noop("tornado"),
        6: tr_noop("blizzard"),
    },
    18: {
        1: tr_noop("fog"),
        2: tr_noop("smoke"),
        3: tr_noop("heavy snowfall"),
        4: tr_noop("heavy rain"),
        5: tr_noop("heavy hail"),
        6: tr_noop("low sun glare"),
        7: tr_noop("sandstorm"),
        8: tr_noop("swarms of insects"),
    },
    19: {1: tr_noop("heavy rain"), 2: tr_noop("heavy snowfall"), 3: tr_noop("soft hail")},
    20: {
        1: tr_noop("reckless driver"),
        2: tr_noop("gunfire"),
        3: tr_noop("people throwing objects"),
    },
    23: {
        1: tr_noop("sports event"),
        2: tr_noop("demonstration"),
        3: tr_noop("demonstration with vehicles"),
        4: tr_noop("concert"),
        5: tr_noop("trade fair"),
        6: tr_noop("military exercise"),
        7: tr_noop("emergency exercise"),
        8: tr_noop("festival"),
        9: tr_noop("procession"),
    },
    24: {
        1: tr_noop("ferry not operating"),
        2: tr_noop("flights suspended"),
        3: tr_noop("train service suspended"),
        4: tr_noop("bus service suspended"),
    },
    25: {
        1: tr_noop("filling station closed"),
        2: tr_noop("service area closed"),
        3: tr_noop("service area full"),
        4: tr_noop("car park full"),
        5: tr_noop("multi-storey car park closed"),
    },
    26: {
        1: tr_noop("slow maintenance vehicle"),
        2: tr_noop("onlookers"),
        3: tr_noop("abnormal load"),
        4: tr_noop("wide abnormal load"),
        5: tr_noop("convoy"),
        6: tr_noop("snowplough"),
        7: tr_noop("de-icing"),
        8: tr_noop("gritting vehicles"),
    },
    27: {
        1: tr_noop("sudden end of queue"),
        2: tr_noop("end of queue over a hill"),
        3: tr_noop("end of queue around a bend"),
        4: tr_noop("end of queue in a tunnel"),
    },
    28: {1: tr_noop("fuel leak"), 2: tr_noop("gas leak")},
    29: {
        1: tr_noop("waiting time at the border"),
        2: tr_noop("waiting time at the ferry port"),
        3: tr_noop("waiting time at the car train terminal"),
    },
    30: {
        1: tr_noop("permanent police checkpoint"),
        2: tr_noop("temporary police checkpoint"),
    },
    31: {
        1: tr_noop("level crossing failure"),
        2: tr_noop("tunnel ventilation out of order"),
        3: tr_noop("traffic light failure"),
        4: tr_noop("emergency telephones out of order"),
        5: tr_noop("automatic toll lanes out of order"),
    },
}

# tec003.
WARNING_LEVELS = {
    1: tr_noop("for information"),
    2: tr_noop("danger level 1"),
    3: tr_noop("danger level 2"),
    4: tr_noop("danger level 3"),
}

# tec004.
LANE_RESTRICTIONS = {
    1: tr_noop("lanes closed"),
    2: tr_noop("lanes open"),
    3: tr_noop("right lane closed"),
    4: tr_noop("left lane closed"),
    5: tr_noop("hard shoulder open"),
}

# tec005.
ADVICE = {
    1: tr_noop("stop at the next parking area"),
    2: tr_noop("do not overtake"),
    3: tr_noop("do not drive"),
    4: tr_noop("use the hard shoulder"),
    5: tr_noop("wait for a police escort"),
    6: tr_noop("wait for the weather to improve"),
    7: tr_noop("make way for vehicles from behind"),
    8: tr_noop("follow the diversion"),
    9: tr_noop("no diversion recommended"),
    10: tr_noop("do not take another route"),
    11: tr_noop("follow police instructions"),
    12: tr_noop("avoid the area"),
    13: tr_noop("drive carefully"),
    14: tr_noop("do not leave your vehicle"),
    15: tr_noop("listen to the radio"),
    16: tr_noop("use the toll lanes"),
    17: tr_noop("wait for the convoy"),
}

# tec202 to tec216: the sub-advice table is the one of the advice code (tec2nn).
SUB_ADVICE = {
    2: {
        1: tr_noop("do not use the overtaking lanes"),
        2: tr_noop("use the crawler lane"),
        3: tr_noop("use the left lane"),
        4: tr_noop("use the right lane"),
    },
    3: {1: tr_noop("stop safely")},
    7: {
        1: tr_noop("form an emergency corridor"),
        2: tr_noop("make way for emergency vehicles"),
    },
    8: {1: tr_noop("follow the diversion signs")},
    13: {
        1: tr_noop("take care at the entry"),
        2: tr_noop("take care at the exit"),
        3: tr_noop("ice on cable structures"),
    },
    14: {
        1: tr_noop("do not leave your vehicle"),
        2: tr_noop("do not leave your vehicle, close the windows"),
    },
    16: {
        1: tr_noop("use the cash toll lanes"),
        2: tr_noop("use the automatic toll lanes"),
    },
}

# tec006: of the speed.
TENDENCIES = {
    1: tr_noop("rising slightly"),
    2: tr_noop("rising"),
    3: tr_noop("rising strongly"),
    4: tr_noop("falling slightly"),
    5: tr_noop("falling"),
    6: tr_noop("falling strongly"),
    7: tr_noop("steady"),
}

# tec009.
VEHICLE_TYPES = {
    1: tr_noop("cars"),
    2: tr_noop("lorries"),
    3: tr_noop("buses"),
    4: tr_noop("taxis"),
    5: tr_noop("trains"),
    6: tr_noop("motorcycles"),
    7: tr_noop("vehicles with trailers"),
    8: tr_noop("motor vehicles"),
    9: tr_noop("hazardous goods vehicles"),
    10: tr_noop("abnormal loads"),
    11: tr_noop("heavy lorries"),
}

# tec007 without a value; those with one are formatted by `restriction_text`.
_RESTRICTIONS = {
    7: tr_noop("without winter tyres"),
    8: tr_noop("without snow chains"),
    9: tr_noop("with a trailer"),
    10: tr_noop("with a caravan"),
    13: tr_noop("even number plates"),
    14: tr_noop("odd number plates"),
    19: tr_noop("emission standard Euro 3"),
    20: tr_noop("emission standard Euro 3 D4"),
    21: tr_noop("emission standard Euro 4"),
    22: tr_noop("emission standard Euro 5"),
    23: tr_noop("with a petrol engine"),
    24: tr_noop("with a diesel engine"),
    25: tr_noop("running on LPG"),
    26: tr_noop("through traffic"),
    27: tr_noop("access only"),
    28: tr_noop("bound for the area given"),
}
# tec007 with a value: (text with the value as {amount}, divisor for the unit shown).
_MEASURED_RESTRICTIONS = {
    1: (tr_noop("width under {amount} m"), 100),
    2: (tr_noop("width over {amount} m"), 100),
    3: (tr_noop("height under {amount} m"), 100),
    4: (tr_noop("height over {amount} m"), 100),
    5: (tr_noop("weight under {amount} t"), 1000),
    6: (tr_noop("weight over {amount} t"), 1000),
    11: (tr_noop("fewer than {amount} occupants"), 1),
    12: (tr_noop("more than {amount} occupants"), 1),
    15: (tr_noop("length under {amount} m"), 100),
    16: (tr_noop("length over {amount} m"), 100),
    17: (tr_noop("axle load under {amount} t"), 1000),
    18: (tr_noop("axle load over {amount} t"), 1000),
}

# tfp003: the level of service; the codes above 7 add a tendency in steps of 8. Also the
# names of the traffic flow levels on the surfaces (`ui.traffic_model.FLOW_LEVELS`).
_LEVELS_OF_SERVICE = {
    0: tr_noop("unknown"),
    1: tr_noop("free flow"),
    2: tr_noop("heavy"),
    3: tr_noop("slow"),
    4: tr_noop("queuing"),
    5: tr_noop("stationary"),
    6: tr_noop("closed"),
}
_LOS_TENDENCIES = {
    1: tr_noop("steady"),
    2: tr_noop("increasing"),
    3: tr_noop("decreasing"),
    4: tr_noop("increasing strongly"),
    5: tr_noop("decreasing strongly"),
}

# olr002.
FORMS_OF_WAY = {
    1: tr_noop("motorway"),
    2: tr_noop("dual carriageway"),
    3: tr_noop("single carriageway"),
    4: tr_noop("roundabout"),
    5: tr_noop("traffic square"),
    6: tr_noop("slip road"),
    7: tr_noop("other road"),
    8: tr_noop("cycle path"),
    9: tr_noop("footpath"),
    10: tr_noop("pedestrian zone"),
}

_LOCATION_METHODS = {LOCATION_TMC: "TMC", LOCATION_OPENLR: "OpenLR"}

_SNI_TABLES = {
    0: tr_noop("name and description"),
    1: tr_noop("fast tuning (applications per SCID)"),
    3: tr_noop("content description"),
    4: tr_noop("coverage area"),
    7: tr_noop("logo"),
    8: tr_noop("links"),
    11: tr_noop("free text"),
    14: tr_noop("versions"),
    33: tr_noop("message counts"),
}

# Names of the components of a message, by the context they appear in: a component
# identifier is only unique among its siblings. Each entry: (name, context of its
# children).
_COMPONENTS: dict[str, dict[int, tuple[str, str | None]]] = {
    "tec": {COMPONENT_MESSAGE: (tr_noop("TEC message"), "tec_message")},
    "tfp": {COMPONENT_MESSAGE: (tr_noop("TFP message"), "tfp_message")},
    "tec_message": {
        COMPONENT_MANAGEMENT: (tr_noop("Message management (MMC)"), None),
        COMPONENT_LOCATION: (tr_noop("Location referencing (LRC)"), "location"),
        COMPONENT_EVENT: (tr_noop("Event"), "event"),
    },
    "tfp_message": {
        COMPONENT_MANAGEMENT: (tr_noop("Message management (MMC)"), None),
        COMPONENT_LOCATION: (tr_noop("Location referencing (LRC)"), "location"),
        COMPONENT_FLOW_MATRIX: (tr_noop("Flow matrix"), "flow_matrix"),
    },
    "event": {
        COMPONENT_DIRECT_CAUSE: (tr_noop("Direct cause"), None),
        COMPONENT_LINKED_CAUSE: (tr_noop("Linked cause"), None),
        COMPONENT_ADVICE: (tr_noop("Advice"), "event"),
        COMPONENT_VEHICLE_RESTRICTION: (tr_noop("Vehicle restriction"), None),
        COMPONENT_DIVERSION: (tr_noop("Diversion"), None),
        COMPONENT_SPEED_LIMIT: (tr_noop("Speed limit"), None),
    },
    "flow_matrix": {COMPONENT_FLOW_VECTOR: (tr_noop("Flow vector"), None)},
    "location": {
        LOCATION_TMC: (tr_noop("TMC location"), None),
        LOCATION_OPENLR: ("OpenLR", "openlr"),
        254: (tr_noop("Method 254 (not public)"), None),
    },
    "openlr": {
        OPENLR_LINEAR: (tr_noop("Line location"), "openlr_reference"),
        OPENLR_POINT_ALONG_LINE: (tr_noop("Point along a line"), "openlr_reference"),
        11: (tr_noop("Location description"), None),
    },
    "openlr_reference": {OPENLR_SHAPE: (tr_noop("Road geometry (shape)"), None)},
}


def decimal(value: float, spec: str) -> str:
    """`value` formatted by `spec` with the decimal separator of the language."""
    # Translators: the decimal separator of the figures in the traffic messages (km, t, m,
    # coordinates); German traffic reports write a comma.
    return format(value, spec).replace(".", tr_context("decimal separator", "."))


def application_name(aid: int | None) -> str:
    if aid is None:
        return tr("unknown")
    return tr(_APPLICATIONS[aid]) if aid in _APPLICATIONS else f"AID {aid}"


def application_short_name(aid: int) -> str:
    """The abbreviation alone: "TEC", "TFP", ..."""
    return _APPLICATIONS[aid].split()[0] if aid in _APPLICATIONS else f"AID {aid}"


def effect_name(code: int | None) -> str:
    if code is None:
        return "-"
    if code in EFFECTS:
        return tr(EFFECTS[code])
    # Translators: an effect on the traffic flow (tec001) that the radio does not know.
    return tr("effect {code}").format(code=code)


def cause_name(main: int | None, sub: int | None = None) -> str:
    """The sub-cause if it is known (it replaces the main cause), else the main cause."""
    if main is None:
        return tr("linked message")
    if sub is not None and sub in SUB_CAUSES.get(main, {}):
        return tr(SUB_CAUSES[main][sub])
    if main in CAUSES:
        return tr(CAUSES[main])
    return tr("cause {code}").format(code=main)


def advice_name(code: int | None, sub: int | None = None) -> str:
    if code is None:
        # Translators: advice of a traffic message without a code, only a text.
        return tr("note")
    text = tr(ADVICE[code]) if code in ADVICE else tr("advice {code}").format(code=code)
    if sub is not None and sub in SUB_ADVICE.get(code, {}):
        text = f"{text}: {tr(SUB_ADVICE[code][sub])}"
    return text


def restriction_text(kind: int, value: int | None) -> str:
    if kind in _MEASURED_RESTRICTIONS and value is not None:
        text, divisor = _MEASURED_RESTRICTIONS[kind]
        return tr(text).format(amount=decimal(value / divisor, "g"))
    if kind in _RESTRICTIONS:
        return tr(_RESTRICTIONS[kind])
    return tr("restriction {code}").format(code=kind)


def level_of_service_name(code: int | None) -> str:
    if code is None:
        return "-"
    level, tendency = code & 0x07, code >> 3
    if level in _LEVELS_OF_SERVICE:
        text = tr(_LEVELS_OF_SERVICE[level])
    else:
        # Translators: a level of service of the traffic flow (tfp003) the radio does
        # not know.
        text = tr("level {code}").format(code=level)
    if tendency in _LOS_TENDENCIES:
        text = f"{text}, {tr(_LOS_TENDENCIES[tendency])}"
    return text


def form_of_way_name(fow: int) -> str:
    return tr(FORMS_OF_WAY[fow]) if fow in FORMS_OF_WAY else tr("unknown road type")


def location_method_name(cid: int) -> str:
    if cid in _LOCATION_METHODS:
        return _LOCATION_METHODS[cid]
    # Translators: a method of location referencing in TPEG, by its number.
    return tr("method {code}").format(code=cid)


def sni_table_name(table: int) -> str:
    if table in _SNI_TABLES:
        return tr(_SNI_TABLES[table])
    return tr("table {code}").format(code=table)


def component_name(context: str, cid: int) -> tuple[str, str | None]:
    """Name of a component in its context ("tec", "tfp" for the message itself), and
    the context of its children."""
    name, children = _COMPONENTS.get(context, {}).get(cid, (None, None))
    if name is None:
        return tr("Component {code}").format(code=cid), None
    return tr(name), children
