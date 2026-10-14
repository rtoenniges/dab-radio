"""A made-up TMC location table in the exchange format of the BASt download, for tests.

It holds the location codes that the messages of `tests.tpeg_vectors` use (those codes
are broadcast), but invented names, roads, neighbours and coordinates near the OpenLR
lines of the same messages: the table of the BASt itself must not be in the repository.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from dab.tmc_locations import TmcLocationTable

# The 1371 roadworks (location 45524, extent 1 in positive direction) on an "A71".
TEC_A71_PRIMARY = 45524
# The 2554 traffic lights (location 28208, no extent) in a town.
TEC_TOWN = 28208
# The 2242 traffic flow (location 21731, extent 7 in negative direction) on a "B213".
TFP_PRIMARY = 21731
TFP_CHAIN = (21731, 90011, 90012, 90013, 90014, 90015, 90016, 90017)

_NAMES = {
    1: "Testkreuz Nord",
    2: "Testheim-Süd",
    3: "Am Testwall",
    **{10 + i: f"Teststadt {i}" for i in range(8)},
}
# Code: (road or segment number, name id, type (class, tcd, stcd), lon, lat, neg, pos).
_POINTS = {
    45524: ("A71", 1, ("P", "1", "3"), 11.0660, 51.0410, None, 90002),
    90002: ("A71", 2, ("P", "1", "3"), 11.1300, 51.1140, 45524, None),
    28208: ("L123", 3, ("P", "1", "11"), 8.8005, 53.0786, None, None),
    **{
        code: (
            "B213",
            10 + i,
            ("P", "1", "15") if i % 2 == 0 else ("P", "3", "37"),
            8.59 - i * 0.06,
            53.04 - i * 0.02,
            TFP_CHAIN[i + 1] if i + 1 < len(TFP_CHAIN) else None,
            TFP_CHAIN[i - 1] if i > 0 else None,
        )
        for i, code in enumerate(TFP_CHAIN)
    },
}
_POINT_COLUMNS = (
    "CID;TABCD;LCD;CLASS;TCD;STCD;JUNCTIONNUMBER;RNID;N1ID;N2ID;POL_LCD;OTH_LCD;SEG_LCD;"
    "ROA_LCD;INPOS;INNEG;OUTPOS;OUTNEG;PRESENTPOS;PRESENTNEG;DIVERSIONPOS;DIVERSIONNEG;"
    "XCOORD;YCOORD;INTERRUPTSROAD;URBAN;JNID"
)
_ROAD_CODES = {"A71": 9001, "L123": 9002}
_SEGMENT_CODES = {"B213": 9003}


def _coordinate(value: float, digits: int) -> str:
    return f"{round(value * 1e5):+0{digits}d}"


def files() -> dict[str, str]:
    """The .DAT files by name."""
    points = [_POINT_COLUMNS]
    offsets = ["CID;TABCD;LCD;NEG_OFF_LCD;POS_OFF_LCD"]
    for code, (number, name, (cls, tcd, stcd), lon, lat, neg, pos) in _POINTS.items():
        road = _ROAD_CODES.get(number, "")
        segment = _SEGMENT_CODES.get(number, "")
        points.append(
            f"58;1;{code};{cls};{tcd};{stcd};;;{name};;1;;{segment};{road};1;1;1;1;1;1;;;"
            f"{_coordinate(lon, 9)};{_coordinate(lat, 8)};;0;"
        )
        offsets.append(f"58;1;{code};{neg or ''};{pos or ''}")
    return {
        "COUNTRIES.DAT": "CID;ECC;CCD;CNAME\n58;E0;D;Germany\n",
        "LOCATIONDATASETS.DAT": "CID;TABCD;DCOMMENT;VERSION;VERSIONDESCRIPTION\n58;1;;22.0;22.0\n",
        "NAMES.DAT": "CID;LID;NID;NAME;NCOMMENT;OFFICIALNAME\n"
        + "".join(f"58;1;{nid};{name};;\n" for nid, name in _NAMES.items()),
        "SUBTYPES.DAT": "CLASS;TCD;STCD;SDESC;SNATCODE;SNATDESC\n"
        "P;1;3;motorway junction;3;Autobahnanschlussstelle\n"
        "P;1;11;crossroads;11;Kreuzung\n"
        "P;1;15;junction;15;Anschlussstelle\n"
        "P;3;37;named point;37;Benannter Punkt\n",
        "ROADS.DAT": "CID;TABCD;LCD;CLASS;TCD;STCD;ROADNUMBER;RNID;N1ID;N2ID;POL_LCD;PES_LEV;RDID\n"
        + "".join(f"58;1;{code};L;1;1;{number};;;;1;;\n" for number, code in _ROAD_CODES.items()),
        "SEGMENTS.DAT": "CID;TABCD;LCD;CLASS;TCD;STCD;ROADNUMBER;RNID;N1ID;N2ID;ROA_LCD;SEG_LCD;"
        "POL_LCD;RDID\n"
        + "".join(
            f"58;1;{code};L;3;0;{number};;;;;;1;\n" for number, code in _SEGMENT_CODES.items()
        ),
        "POINTS.DAT": "\n".join(points) + "\n",
        "POFFSETS.DAT": "\n".join(offsets) + "\n",
    }


def write_zip(path: Path) -> Path:
    """The files in a ZIP laid out like the download (a folder within a folder), with
    the byte order mark the BASt writes."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files().items():
            archive.writestr(f"LCL22_deutsch/LCL22.0.D_230515/{name}", "﻿" + text)
    return path


def table(directory: Path) -> TmcLocationTable:
    return TmcLocationTable.read_exchange(write_zip(directory / "lcl.zip"))
